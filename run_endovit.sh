#!/usr/bin/env bash
# Train the temporal heads on EndoViT (768-d) features and compare against the
# ResNet50-feature baselines. Same temporal hyperparameters as the main run --
# ONLY the Stage-1 features differ, so any gain is attributable to the features.
set -euo pipefail
cd "$(dirname "$0")"
say() { echo; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }
F=features_endovit

say "MS-TCN on EndoViT features"
./run.sh train_tcn.py --features $F --out checkpoints/endovit_mstcn.pt --model mstcn --epochs 40

say "TeCNO on EndoViT features"
./run.sh train_tcn.py --features $F --out checkpoints/endovit_tecno.pt --model mstcn --epochs 40 --causal

say "LoViT-causal on EndoViT features"
./run.sh train_tcn.py --features $F --out checkpoints/endovit_lovit.pt \
    --model lovit --layers 5 --stages 2 --d 256 --heads 8 --epochs 30 --lr 3e-4 --causal

say "COMPARISON on EndoViT features"
for ck in endovit_mstcn endovit_tecno endovit_lovit; do
    ./run.sh evaluate.py --features $F --ckpt checkpoints/$ck.pt
done
say "ENDOVIT ALL DONE"
