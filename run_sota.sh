#!/usr/bin/env bash
# Max-effort attempt to beat the ResNet50+MS-TCN 90.8% baseline by combining four
# levers, each ablatable. Waits for the EndoViT-ft pipeline first (shares the GPU).
set -euo pipefail
cd "$(dirname "$0")"
say() { echo; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }

say "WAIT for EndoViT-ft pipeline to free the GPU"
while ! grep -q "ENDOVIT_FT ALL DONE" logs_endovit_ft8.log 2>/dev/null; do sleep 30; done

# ---- Lever 1: multi-task backbone (phase + tool) -> better features ----
say "Lever1: multi-task ResNet50 (phase+tool)"
./run.sh train_cnn_mtl.py --frames data/frames --phase data/phase_annotations \
    --tool data/tool_annotations --epochs 5 --bs 64 --out checkpoints/resnet50_mtl.pt
./run.sh extract_features.py --frames data/frames --anno data/phase_annotations \
    --ckpt checkpoints/resnet50_mtl.pt --out features_mtl

# ---- Lever 2: feature fusion ----
say "Lever2: fuse features"
./run.sh fuse_features.py --a features_mtl --b features_endovit_ft --out features_mtl_ev
./run.sh fuse_features.py --a features --b features_endovit_ft --out features_rn_ev

# ---- Lever 3: train MS-TCN/TeCNO on each new feature set ----
say "Lever3: temporal heads on new features"
for F in features_mtl features_mtl_ev features_rn_ev; do
    ./run.sh train_tcn.py --features $F --out checkpoints/${F}_mstcn.pt --model mstcn --epochs 40
    ./run.sh train_tcn.py --features $F --out checkpoints/${F}_tecno.pt --model mstcn --epochs 40 --causal
done

say "ABLATION: per-feature-set test accuracy (MS-TCN, offline)"
for F in features features_mtl features_endovit_ft features_mtl_ev features_rn_ev; do
    case $F in
        features) ck=checkpoints/mstcn.pt;;
        features_endovit_ft) ck=checkpoints/endovitft_mstcn.pt;;
        *) ck=checkpoints/${F}_mstcn.pt;;
    esac
    echo "--- feature set: $F ---"
    ./run.sh evaluate.py --features $F --ckpt "$ck"
done

# ---- Lever 4: ensemble the best offline models across feature sets ----
say "Lever4: ENSEMBLE (offline MS-TCN across feature sets)"
./run.sh ensemble_eval.py --models \
    checkpoints/mstcn.pt:features \
    checkpoints/endovitft_mstcn.pt:features_endovit_ft \
    checkpoints/features_mtl_mstcn.pt:features_mtl \
    checkpoints/features_mtl_ev_mstcn.pt:features_mtl_ev \
    checkpoints/features_rn_ev_mstcn.pt:features_rn_ev

say "SOTA ALL DONE"
