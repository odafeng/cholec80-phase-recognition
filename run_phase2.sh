#!/usr/bin/env bash
# Phase 2 baseline pipeline (RESEARCH_PLAN.md). Runs unattended in tmux.
#   CNN -> ResNet50 features -> {TeCNO causal, MS-TCN offline ref, LoViT-causal}
#   -> reliability evaluation via the new metrics.py suite.
# Each step tees to logs/. Safe to re-run: skips steps whose output exists.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p checkpoints features results logs
ts() { date "+%F %T"; }
say() { echo; echo "==================== [$(ts)] $* ===================="; }

# 1. ResNet50 backbone (Stage 1) -- needs only train(1-32)+val(33-40), already extracted
if [ ! -f checkpoints/resnet50.pt ]; then
  say "STAGE 1: train ResNet50 (32 train / 8 val, 5 epochs)"
  ./run.sh train_cnn.py --frames data/frames --anno data/phase_annotations \
      --epochs 5 --bs 64 --workers 8 --out checkpoints/resnet50.pt 2>&1 | tee logs/train_cnn.log
else
  echo "skip train_cnn (checkpoints/resnet50.pt exists)"
fi

# 2. feature extraction needs ALL 80 videos -> wait for frame extraction to finish
say "WAIT for frame extraction (tmux 'frames') to finish all 80 videos"
while ! grep -q FRAMES_EXIT logs/frames.log 2>/dev/null; do
  echo "  [$(ts)] frames: $(ls data/frames 2>/dev/null | wc -l)/80 ..."; sleep 30
done
echo "  frames done: $(ls data/frames | wc -l) video dirs"

if [ ! -f features/video80.pt ]; then
  say "STAGE 1.5: extract 2048-d ResNet50 features (all 80 videos)"
  ./run.sh extract_features.py --frames data/frames --anno data/phase_annotations \
      --ckpt checkpoints/resnet50.pt --out features 2>&1 | tee logs/extract_features.log
else
  echo "skip extract_features (features/video80.pt exists)"
fi

# 3. temporal baselines (single seed for the first reliability table)
say "STAGE 2a: TeCNO (causal/online)"
./run.sh train_tcn.py --features features --out checkpoints/tecno.pt --causal --epochs 40 \
    2>&1 | tee logs/train_tecno.log
say "STAGE 2b: MS-TCN (offline reference)"
./run.sh train_tcn.py --features features --out checkpoints/mstcn.pt --epochs 40 \
    2>&1 | tee logs/train_mstcn.log
say "STAGE 2c: LoViT-causal (online)   [lr 3e-4, 30 ep -- LoViT is lr-sensitive]"
./run.sh train_tcn.py --features features --out checkpoints/lovit_causal.pt --model lovit \
    --causal --layers 5 --stages 2 --d 256 --heads 8 --lr 3e-4 --epochs 30 \
    2>&1 | tee logs/train_lovit.log

# 4. reliability evaluation (parity numbers + full suite + npz for significance)
say "EVAL: reliability suite (relaxed + segmental + latency + calibration)"
for m in tecno mstcn lovit_causal; do
  echo; echo "######## $m ########"
  ./run.sh evaluate.py --features features --ckpt checkpoints/$m.pt \
      --relaxed --out results/baseline --tag ${m}_baseline_s0 2>&1 | tee logs/eval_$m.log
done

say "PHASE 2 COMPLETE"
echo "PHASE2_EXIT=0"
