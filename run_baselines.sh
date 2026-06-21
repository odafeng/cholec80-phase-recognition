#!/usr/bin/env bash
# QCD #16 item 1 — add Trans-SVNet + Surgformer as additional online posterior sources.
# Trains 5 base seeds of each new head across all 5 backbones (= 50 checkpoints) into the
# same rel_<bb> layout the QCD frontier/stats already read, then fits val temperature for
# each (so temperature-scaled posteriors work). Base seeds only: these are posterior
# sources for the QCD decoder comparison, not the BUA reliability matrix.
#
# Online/causal heads (pass the causality test). Test videos 41-80 are never touched.
# Idempotent: existing checkpoints are skipped. Usage:  ./run_baselines.sh
set -uo pipefail
cd "$(dirname "$0")"

SEEDS=(0 1 2 3 4)
# tag : feature dir
BACKBONES=("rn50:features" "endovit:features_endovit" "endovitft:features_endovit_ft" \
           "e2e:features_e2e" "surgmae:features_surgmae")
# head tag : --model : per-head hyperparams (transformer heads are lr-sensitive)
HEADS=("transsvnet:transsvnet:--lr 3e-4 --d 128 --heads 8 --layers 9 --stages 2" \
       "surgformer:surgformer:--lr 3e-4 --d 192 --heads 6 --layers 6 --stages 2")
EPOCHS=30

R() { ./run.sh "$@"; }   # LD_LIBRARY_PATH-clean venv python
ts() { date "+%F %T"; }
say() { echo; echo "============ [$(ts)] $* ============"; }
mkdir -p logs

for bbspec in "${BACKBONES[@]}"; do
  IFS=":" read -r bb feat <<< "$bbspec"
  if [ ! -f "$feat/video80.pt" ]; then
    echo "[skip] $bb: $feat not ready"; continue
  fi
  CK="checkpoints/rel_${bb}"; mkdir -p "$CK"
  for hspec in "${HEADS[@]}"; do
    IFS=":" read -r tag model hp <<< "$hspec"
    for s in "${SEEDS[@]}"; do
      base="$CK/${tag}_base_s${s}.pt"
      if [ -f "$base" ]; then echo "[have] $base"; continue; fi
      say "train $bb / $tag seed $s"
      R train_tcn.py --features "$feat" --model "$model" --causal $hp --seed "$s" \
        --epochs "$EPOCHS" --out "$base" 2>&1 | tee "logs/base_${bb}_${tag}_s${s}.log"
      R calibrate.py --ckpt "$base" --features "$feat" >/dev/null 2>&1 || true
    done
  done
done

say "BASELINES COMPLETE"
echo "BASELINES_EXIT=0"
