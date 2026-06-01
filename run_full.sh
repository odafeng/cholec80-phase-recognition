#!/usr/bin/env bash
# Full Cholec80 pipeline driver. Runs all 5 stages in sequence, logging each.
# Launch via:  nohup ./run_full.sh > logs_full.log 2>&1 &
# Then watch:  tail -f logs_full.log
#
# Always goes through run.sh (clears LD_LIBRARY_PATH -> avoids the cuBLAS hang).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p checkpoints
ts() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo; echo "==================== [$(ts)] $* ===================="; }

say "STAGE 0: extract frames for ALL 80 videos (1 fps)"
./run.sh extract_frames.py --videos data/videos --out data/frames

say "STAGE 1: train ResNet50 (32 train / 8 val)"
./run.sh train_cnn.py --frames data/frames --anno data/phase_annotations \
    --epochs 5 --bs 64 --workers 8 --out checkpoints/resnet50.pt

say "STAGE 1.5: extract 2048-d features for ALL 80 videos"
./run.sh extract_features.py --frames data/frames --anno data/phase_annotations \
    --ckpt checkpoints/resnet50.pt --out features

say "STAGE 2a: train MS-TCN (non-causal)"
./run.sh train_tcn.py --features features --out checkpoints/mstcn.pt --epochs 40

say "STAGE 2b: train TeCNO (causal)"
./run.sh train_tcn.py --features features --out checkpoints/tecno.pt --epochs 40 --causal

say "EVAL: MS-TCN"
./run.sh evaluate.py --features features --ckpt checkpoints/mstcn.pt

say "EVAL: TeCNO"
./run.sh evaluate.py --features features --ckpt checkpoints/tecno.pt

say "ALL DONE"
