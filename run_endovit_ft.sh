#!/usr/bin/env bash
# Fair experiment: fine-tune EndoViT on phase (like we did ResNet50), extract
# fine-tuned features, retrain temporal heads, compare. Tells us whether surgical
# pretraining actually helps once adapted to the task.
set -euo pipefail
cd "$(dirname "$0")"
say() { echo; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }

say "Stage 1: fine-tune EndoViT on phase"
./run.sh train_cnn_endovit.py --frames data/frames --anno data/phase_annotations \
    --epochs 8 --bs 48 --out checkpoints/endovit_ft.pt

rm -rf features_endovit_ft checkpoints/endovitft_*.pt
say "Stage 1.5: extract fine-tuned EndoViT features"
./run.sh extract_features_endovit.py --frames data/frames --anno data/phase_annotations \
    --out features_endovit_ft --ckpt checkpoints/endovit_ft.pt --bs 64

say "Stage 2: temporal heads on fine-tuned EndoViT features"
F=features_endovit_ft
./run.sh train_tcn.py --features $F --out checkpoints/endovitft_mstcn.pt --model mstcn --epochs 40
./run.sh train_tcn.py --features $F --out checkpoints/endovitft_tecno.pt --model mstcn --epochs 40 --causal
./run.sh train_tcn.py --features $F --out checkpoints/endovitft_lovit.pt \
    --model lovit --layers 5 --stages 2 --d 256 --heads 8 --epochs 30 --lr 3e-4 --causal

say "COMPARISON (fine-tuned EndoViT features)"
for ck in endovitft_mstcn endovitft_tecno endovitft_lovit; do
    ./run.sh evaluate.py --features $F --ckpt checkpoints/$ck.pt
done
say "ENDOVIT_FT ALL DONE"
