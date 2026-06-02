# Surgical phase-recognition env — portable bootstrap

Recreate the entire stack on a fresh GCP VM (T4 / **L4** / A100): system deps, venv +
python (incl. mamba-ssm), rclone(R2), gh, the code (cholec80_phase repo + Surgical-Mamba +
our patches), and the **Telegram push** system.

## Use it
```bash
# on the NEW VM:
tar xzf surgical-env-bootstrap.tar.gz && cd bootstrap
cp secrets.env.template secrets.env      # then edit: fill TG_* and R2_* creds
./setup.sh                               # ~10-20 min (downloads torch + builds mamba)
```

## What it installs
- ffmpeg, python3-venv, build tools, **rclone**, **gh**.
- venv at `~/cholec80_phase/.venv`: torch 2.9.1+cu129, timm, **mamba-ssm + causal-conv1d**
  (via `--no-build-isolation` — the only way it installs), lightning, scipy, opencv, etc.
- Clones `Surgical-Mamba`, copies our scripts (`code/`), applies the `--surgenet_weights`
  hook to train.py (`apply_surgenet_patch.py`).
- Configures rclone `r2` remote from `secrets.env` → `rclone copy r2:cholec80/cholec80.zip ./`.
- Installs the Telegram push (`~/telegram_notify.sh`, `~/telegram_monitor.sh`, `~/.telegram_creds`)
  and sends a "bootstrapped" test push.

## Two gotchas baked in (do not forget)
1. **Always run python via `env -u LD_LIBRARY_PATH`** (or `cholec80_phase/run.sh`) — else
   cuBLAS hangs ("Cannot load symbol cublasLtCreate").
2. **AMP precision by GPU**: T4 (arch 7.5) has NO bf16 → use `--no_amp` (fp32, stable but
   ~2× slower). **L4 / A100 (arch ≥ 8.0) → use `--amp_dtype bfloat16`** (fast AND stable;
   avoids the fp16 Grad-NaN we hit on T4). setup.sh prints the detected arch.

## Not included (by design)
- Data (cholec80.zip — pull from R2), the venv, checkpoints, frames, and any secrets.
- `secrets.env` (you fill it; keep it private).

## Telegram re-arm after reboot
`~/telegram_monitor.sh` hard-codes the training PID — edit it to the live PID, or just call
`~/telegram_notify.sh "msg"` from your own watchers.
