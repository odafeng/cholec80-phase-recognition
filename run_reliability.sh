#!/usr/bin/env bash
# Phase 3 — BUA + calibration matrix (RESEARCH_PLAN.md). Runs unattended in tmux.
#
# For each online head, train 5 Baseline seeds and 5 +BUA seeds (boundary head +
# boundary-weighted loss). Then evaluate the reliability suite under each
# uncertainty source and dump per-video npz for significance:
#   Baseline            : raw softmax
#   +Calib              : global temperature (fit on val)
#   +BUA(full)          : boundary model + temperature + uncertainty smoothing
#   BUA ablations       : -boundary, -smooth, -calib  (switchable components)
#   Ensemble (5 seeds)  : averaged softmax (deep-ensemble uncertainty)
# Finally run significance.py (Baseline vs +BUA) per head.
#
# Usage:  ./run_reliability.sh [FEATDIR]     (default FEATDIR=features = ResNet50)
set -uo pipefail
cd "$(dirname "$0")"
FEAT="${1:-features}"
TAG="${2:-rn50}"                # backbone tag -> separate ckpt/result dirs
SEEDS=(0 1 2 3 4)
HEADS=("tecno:--causal:mstcn" "lovit_causal:--causal:lovit" "asformer:--causal:asformer")
# per-model training hyperparams (LoViT & ASFormer are lr-sensitive)
extra_for() {
  case "$1" in
    lovit)    echo "--lr 3e-4 --layers 5 --stages 2 --d 256 --heads 8" ;;
    asformer) echo "--lr 3e-4 --layers 9 --stages 3 --d 64 --heads 1" ;;
    *)        echo "" ;;
  esac
}
ep_for() { case "$1" in lovit|asformer) echo 30 ;; *) echo 40 ;; esac; }
CK="checkpoints/rel_${TAG}"
RES="results/rel_${TAG}"
mkdir -p "$CK" "$RES" logs
ts() { date "+%F %T"; }
say() { echo; echo "============ [$(ts)] $* ============"; }

R() { ./run.sh "$@"; }   # LD_LIBRARY_PATH-clean venv python

if [ ! -f "$FEAT/video80.pt" ]; then
  echo "ERROR: $FEAT not ready (need Phase 2 features first)"; exit 1
fi

for spec in "${HEADS[@]}"; do
  IFS=":" read -r tag flags model <<< "$spec"
  say "HEAD = $tag  (model=$model $flags) on $FEAT"

  # ---- train 5 Baseline + 5 BUA seeds ----
  for s in "${SEEDS[@]}"; do
    base="$CK/${tag}_base_s${s}.pt"
    bua="$CK/${tag}_bua_s${s}.pt"
    EX="$(extra_for "$model")"; EP="$(ep_for "$model")"
    [ -f "$base" ] || { say "train Baseline $tag seed $s";
      R train_tcn.py --features "$FEAT" --model "$model" $flags $EX --seed "$s" \
        --epochs "$EP" --out "$base" 2>&1 | tee "logs/rel_${tag}_base_s${s}.log"; }
    [ -f "$bua" ] || { say "train +BUA $tag seed $s";
      R train_tcn.py --features "$FEAT" --model "$model" $flags $EX --seed "$s" \
        --boundary --epochs "$EP" --out "$bua" 2>&1 | tee "logs/rel_${tag}_bua_s${s}.log"; }
    # fit global temperature on val for both
    R calibrate.py --ckpt "$base" --features "$FEAT" >/dev/null 2>&1 || true
    R calibrate.py --ckpt "$bua"  --features "$FEAT" >/dev/null 2>&1 || true
  done

  # ---- evaluate variants per seed -> per-video npz ----
  for s in "${SEEDS[@]}"; do
    base="$CK/${tag}_base_s${s}.pt"; bua="$CK/${tag}_bua_s${s}.pt"
    # Baseline (raw)
    R evaluate.py --features "$FEAT" --ckpt "$base" --relaxed \
      --out "$RES/${tag}_baseline" --tag "s${s}" >/dev/null 2>&1
    # +Calib (temperature only)
    R evaluate.py --features "$FEAT" --ckpt "$base" --temp auto --relaxed \
      --out "$RES/${tag}_calib" --tag "s${s}" >/dev/null 2>&1
    # +BUA full (boundary model + temp + smoothing)
    R evaluate.py --features "$FEAT" --ckpt "$bua" --temp auto --smooth --relaxed \
      --out "$RES/${tag}_bua" --tag "s${s}" >/dev/null 2>&1
    # ablations
    R evaluate.py --features "$FEAT" --ckpt "$bua" --temp auto --relaxed \
      --out "$RES/${tag}_bua_minus_smooth" --tag "s${s}" >/dev/null 2>&1     # -smooth
    R evaluate.py --features "$FEAT" --ckpt "$bua" --smooth --relaxed \
      --out "$RES/${tag}_bua_minus_calib" --tag "s${s}" >/dev/null 2>&1      # -calib
    R evaluate.py --features "$FEAT" --ckpt "$base" --temp auto --smooth --relaxed \
      --out "$RES/${tag}_bua_minus_boundary" --tag "s${s}" >/dev/null 2>&1   # -boundary
    echo "  evaluated $tag seed $s (all variants)"
  done

  # ---- significance: Baseline vs +BUA over 5 seeds ----
  say "SIGNIFICANCE $tag : Baseline vs +BUA"
  R significance.py --baseline "$RES/${tag}_baseline" --bua "$RES/${tag}_bua" \
    2>&1 | tee "logs/rel_sig_${tag}.log"
done

# ---- sensitivity sweep (appendix): conclusions stable across tol & stable_k ----
say "SENSITIVITY sweep (tol in 5/10/15, stable_k in 1/3/5) on tecno BUA seed 0"
for tol in 5 10 15; do for k in 1 3 5; do
  R evaluate.py --features "$FEAT" --ckpt "$CK/tecno_bua_s0.pt" --temp auto --smooth \
    --tol "$tol" --stable_k "$k" --out "$RES/sensitivity" --tag "tol${tol}_k${k}" \
    >/dev/null 2>&1
done; done
echo "sensitivity grid written to $RES/sensitivity"

say "PHASE 3 COMPLETE"
echo "PHASE3_EXIT=0"
