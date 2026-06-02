#!/usr/bin/env bash
# Train LoViT (causal + offline) on the existing ResNet50 features, then compare
# all four temporal heads. Reuses features/ from the main run (no re-extraction).
set -euo pipefail
cd "$(dirname "$0")"
say() { echo; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }

say "Train LoViT (causal / online)  -> vs TeCNO"
./run.sh train_tcn.py --features features --out checkpoints/lovit_causal.pt \
    --model lovit --layers 5 --stages 2 --d 256 --heads 8 --epochs 30 --lr 3e-4 --causal

say "Train LoViT (non-causal / offline)  -> vs MS-TCN"
./run.sh train_tcn.py --features features --out checkpoints/lovit_offline.pt \
    --model lovit --layers 5 --stages 2 --d 256 --heads 8 --epochs 30 --lr 3e-4

say "FINAL COMPARISON: all four temporal models on the test set"
for ck in mstcn tecno lovit_offline lovit_causal; do
    ./run.sh evaluate.py --features features --ckpt checkpoints/$ck.pt
done
say "LOVIT ALL DONE"
