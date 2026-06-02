# Cholec80 Phase Recognition — Run Guide

Two-stage pipeline. Stage 1 (CNN) + feature extraction run **once** and feed
**both** temporal models. MS-TCN vs TeCNO differ only by the `--causal` flag.

All commands assume the dedicated venv. Either `source .venv/bin/activate` first,
or prefix every command with `.venv/bin/python`.

```bash
cd /home/KHUser/cholec80_phase
PY=.venv/bin/python
```

## 0. Data layout (after unzip)
Expected (adjust paths to match the actual unzip):
```
data/videos/             video01.mp4 ... video80.mp4
data/phase_annotations/  video01-phase.txt ...
```

## 1. Extract frames at 1 fps  (Stage 0)
```bash
$PY extract_frames.py --videos data/videos --out data/frames
# resumable; ~minutes per video. Produces data/frames/videoXX/00000000.jpg ...
```

## 2. Train ResNet50  (Stage 1)
```bash
$PY train_cnn.py --frames data/frames --anno data/phase_annotations \
   --epochs 5 --bs 64 --out checkpoints/resnet50.pt
# T4: ~10-20 min/epoch. Saves best-val-acc checkpoint.
```

## 3. Extract 2048-d features  (Stage 1.5)
```bash
$PY extract_features.py --frames data/frames --anno data/phase_annotations \
   --ckpt checkpoints/resnet50.pt --out features
# Writes features/videoXX.pt = {feats:(T,2048), labels:(T,)} for all 80 videos.
```

## 4a. Train MS-TCN (non-causal)
```bash
$PY train_tcn.py --features features --out checkpoints/mstcn.pt
```

## 4b. Train TeCNO (causal)
```bash
$PY train_tcn.py --features features --out checkpoints/tecno.pt --causal
```
Stage 2 is fast (features are tiny): tens of seconds per epoch on the T4.

## 5. Evaluate & compare
```bash
$PY evaluate.py --features features --ckpt checkpoints/mstcn.pt
$PY evaluate.py --features features --ckpt checkpoints/tecno.pt
```
Reports frame accuracy, video-averaged accuracy, and per-phase
precision/recall/jaccard.

## Smoke test first (recommended)
Run the whole pipeline on a few videos (e.g. ids 1,2,33,41) before the full run
to confirm everything is wired correctly, using `--only` on extract_frames and
small `--epochs`.

## Results (test = videos 41-80, 32/8/40 split)

| Temporal model | Frame acc | Mean Jaccard |
|---|:---:|:---:|
| **MS-TCN** (offline) | **90.81%** | 76.3 |
| **TeCNO** (online)   | 88.95% | 71.4 |
| LoViT-style (offline) | 83.22% | 61.8 |
| LoViT-style (online)  | 86.19% | 65.0 |

Stage-1 ResNet50 per-frame val accuracy: 81.75%; the temporal model lifts this by
up to ~9%. TeCNO's 88.95% reproduces the original paper (~88.6%). Our from-scratch
LoViT-style Transformer underperforms the TCNs here because the dataset is small
(32 train videos) and the transformer overfits — see **[RESULTS.md](RESULTS.md)**
for the full analysis and the "TCN beats Transformer on small data" lesson.

## Transformer temporal head (LoViT-style)
`lovit.py` is a long-short causal Transformer (attention + conv, multi-stage),
a drop-in replacement for the TCN head:
```bash
./run_lovit.sh   # trains LoViT causal+offline on the same features, compares all four
```
