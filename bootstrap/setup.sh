#!/usr/bin/env bash
# One-shot environment bootstrap for the surgical phase-recognition stack.
# Recreates: system deps, venv + python deps (incl. mamba-ssm), rclone(R2), gh,
# the code (cholec80_phase repo + Surgical-Mamba + our patches), and the Telegram
# push system. Idempotent-ish; safe to re-run.
#
# Usage on a FRESH VM:
#   1) copy this bootstrap/ dir (or the tarball) to the new machine
#   2) cp secrets.env.template secrets.env  &&  edit secrets.env  (fill creds)
#   3) ./setup.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${HOME}"
VENV="${HOME_DIR}/cholec80_phase/.venv"
say(){ echo; echo "==== $* ===="; }

# ---- 0. load secrets (optional) ----
[ -f "${HERE}/secrets.env" ] && source "${HERE}/secrets.env" || echo "(no secrets.env — telegram/rclone will be skipped)"

# ---- 1. system deps ----
say "1/6 system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg python3.10-venv python3-pip build-essential ninja-build wget curl unzip git

# rclone (R2)
command -v rclone >/dev/null || curl -s https://rclone.org/install.sh | sudo bash
# gh (GitHub CLI)
if ! command -v gh >/dev/null; then
  sudo mkdir -p -m 755 /etc/apt/keyrings
  wget -nv -O- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq gh
fi

# ---- 2. venv + python deps ----
say "2/6 venv + python packages"
mkdir -p "${HOME_DIR}/cholec80_phase"
python3 -m venv "${VENV}"
P="${VENV}/bin/pip"
$P install -q --upgrade pip setuptools wheel ninja packaging
$P install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu129
$P install -q pytorch-lightning torchmetrics timm opencv-python-headless pandas scikit-learn \
  pillow tqdm matplotlib tensorboard scipy einops omegaconf albumentations fvcore wandb huggingface_hub
# mamba-ssm: MUST use --no-build-isolation (else "No module named torch" at build)
say "2b/6 mamba-ssm (no-build-isolation; GPU arch auto)"
ARCH=$( "${VENV}/bin/python" -c "import torch;print('%d.%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo "7.5;8.9" )
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST="${ARCH}" \
  $P install -q causal-conv1d mamba-ssm --no-build-isolation || echo "[warn] mamba build failed — check CUDA toolkit/nvcc"

# ---- 3. code ----
say "3/6 code: repos + our scripts"
cd "${HOME_DIR}"
[ -d cholec80_phase/.git ] || git clone https://github.com/odafeng/cholec80-phase-recognition.git cholec80_phase_repo 2>/dev/null && \
  cp -n cholec80_phase_repo/*.py cholec80_phase_repo/*.sh cholec80_phase/ 2>/dev/null || true
[ -d Surgical-Mamba ] || git clone --depth 1 https://github.com/sukjuoh/Surgical-Mamba.git
# copy our custom Surgical-Mamba scripts (carried in this bootstrap)
cp -n "${HERE}/code/"*.py "${HOME_DIR}/Surgical-Mamba/" 2>/dev/null || true
cp -n "${HERE}/code/"*.sh "${HOME_DIR}/Surgical-Mamba/" 2>/dev/null || true
# apply the train.py --surgenet_weights hook
"${VENV}/bin/python" "${HERE}/apply_surgenet_patch.py" "${HOME_DIR}/Surgical-Mamba/train.py" 2>/dev/null || true
# the run.sh LD_LIBRARY_PATH wrapper (cuBLAS gotcha)
cp -n "${HERE}/code/run.sh" "${HOME_DIR}/cholec80_phase/run.sh" 2>/dev/null || true
chmod +x "${HOME_DIR}"/cholec80_phase/*.sh "${HOME_DIR}"/Surgical-Mamba/*.sh 2>/dev/null || true

# ---- 4. rclone R2 remote ----
say "4/6 rclone R2"
if [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
  rclone config create r2 s3 provider=Cloudflare \
    access_key_id="${R2_ACCESS_KEY_ID}" secret_access_key="${R2_SECRET_ACCESS_KEY}" \
    endpoint="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" region=auto no_check_bucket=true --non-interactive >/dev/null
  echo "r2 remote configured. Fetch data: rclone copy r2:${R2_BUCKET:-cholec80}/cholec80.zip ./ --progress --s3-disable-checksum"
else echo "(no R2 creds in secrets.env — skipped)"; fi

# ---- 5. telegram ----
say "5/6 telegram push"
cp "${HERE}/telegram_notify.sh" "${HOME_DIR}/telegram_notify.sh"; chmod +x "${HOME_DIR}/telegram_notify.sh"
[ -f "${HERE}/telegram_monitor.sh" ] && cp "${HERE}/telegram_monitor.sh" "${HOME_DIR}/" && chmod +x "${HOME_DIR}/telegram_monitor.sh"
if [ -n "${TG_TOKEN:-}" ] && [ -n "${TG_CHAT:-}" ]; then
  printf 'TG_TOKEN="%s"\nTG_CHAT="%s"\n' "${TG_TOKEN}" "${TG_CHAT}" > "${HOME_DIR}/.telegram_creds"
  chmod 600 "${HOME_DIR}/.telegram_creds"
  "${HOME_DIR}/telegram_notify.sh" "✅ New VM bootstrapped. GPU arch: ${ARCH}. (bf16 OK if arch ≥ 8.0)"
else echo "(no TG creds in secrets.env — skipped; fill ~/.telegram_creds manually)"; fi

# ---- 6. sanity ----
say "6/6 sanity check"
"${VENV}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
try:
    import mamba_ssm, timm, scipy; print("mamba-ssm", mamba_ssm.__version__, "OK")
except Exception as e: print("mamba import FAILED:", e)
PY
echo
echo "DONE. Reminders:"
echo " - Run python via:  env -u LD_LIBRARY_PATH \$VENV/bin/python   (cuBLAS hang otherwise)"
echo " - On T4 use --no_amp (no bf16). On L4/A100 (arch≥8.0) use --amp_dtype bfloat16 (faster, stable)."
