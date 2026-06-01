# Cholec80 Surgical Phase Recognition — Project Guide

Goal: surgical **phase recognition** on Cholec80, implemented as a **two-stage**
pipeline, comparing two temporal models:
- **MS-TCN**  — multi-stage TCN, *non-causal* (sees past + future).
- **TeCNO**   — *causal* multi-stage TCN (online; only sees the past). Czempiel et al., MICCAI 2020.

Both share Stage 1 (CNN feature extractor) and the same Stage 2 code; the only
difference is a `causal` flag. So Stage 1 / feature extraction runs **once**.

## Environment (IMPORTANT)
- Dedicated venv: `/home/KHUser/cholec80_phase/.venv` (Python 3.10).
- **Always** run with the venv interpreter directly (no need to activate):
  `/home/KHUser/cholec80_phase/.venv/bin/python <script>`
- Stack: torch 2.9.1+cu129, torchvision 0.24.1, pytorch-lightning 2.6.5,
  torchmetrics, opencv-python-headless, pandas, scikit-learn, pillow, tqdm,
  matplotlib, tensorboard. System ffmpeg 4.4.2.
- The OFFICIAL TeCNO repo targets lightning 0.x and is INCOMPATIBLE with this
  stack — we use a clean from-scratch implementation instead.

## Hardware
- GPU: 1× Tesla T4, 15 GB VRAM. CPU: 8 cores. RAM: 29 GB. Disk: ~196 GB free.
- Implications: batch sizes must fit 15 GB; ResNet50 stage-1 ~10–20 min/epoch.

## Data
- Source archive: `/home/KHUser/data/cholec80.zip` (~21 GB+).
- Cholec80 = 80 cholecystectomy videos (25 fps) + per-frame **phase** annotations
  (`*-phase.txt`, labeled at 25fps) + tool annotations (`*-tool.txt`, every 25 frames).
- **7 phases**: Preparation, CalotTriangleDissection, ClippingCutting,
  GallbladderDissection, GallbladderPackaging, CleaningCoagulation, GallbladderRetraction.

## Key decisions
- **Split**: 32 train / 8 val / 40 test  → videos 01–32 / 33–40 / 41–80.
- **Stage 1**: train ResNet50 (ImageNet-init) for per-frame phase classification.
- **Frames**: extract at **1 fps**, resize to 250×250; train crop/resize to 224×224.
- **Stage 2**: extract 2048-d ResNet50 features per frame → one time-series per video;
  train MS-TCN (`causal=False`) and TeCNO (`causal=True`) on these features.
- **Metrics**: video-level Accuracy, Precision, Recall, Jaccard (F1), per-phase + mean.

## Pipeline / directory layout (planned)
```
cholec80_phase/
  .venv/                  # dedicated environment
  data/                   # symlink or extracted dataset
    videos/  VIDEO01.mp4 ...
    phase_annotations/  video01-phase.txt ...
    frames/   video01/00000.jpg ...     # 1fps extracted
  splits.py               # 32/8/40 video id lists
  extract_frames.py       # Stage 0: 1fps frame extraction (ffmpeg)
  dataset.py              # frame + phase-label datasets
  train_cnn.py            # Stage 1: ResNet50 trainer
  extract_features.py     # Stage 1.5: dump 2048-d features per video
  mstcn.py                # MS-TCN / TeCNO model (causal flag)
  train_tcn.py            # Stage 2: train temporal model
  evaluate.py             # metrics
  features/               # saved feature tensors
  checkpoints/
```

## Gotchas (IMPORTANT — see memory cholec80-env-gotchas)
- ALWAYS launch python via `./run.sh <script> ...` (clears LD_LIBRARY_PATH; a bare
  `.venv/bin/python` HANGS with "Cannot load symbol cublasLtCreate").
- Login shell has errexit+pipefail: append `|| true` to greps that may not match.
- Checkpoints load with `weights_only=False` (our trusted files).

## Progress log
- [done] env setup: venv + full stack + ffmpeg.
- [done] download (70 GB) + unzip + inspect: 80 videos + phase/tool annos, format verified.
- [done] SMOKE TEST end-to-end on videos {1,2,33,41}: extract_frames → train_cnn →
  extract_features → train_tcn (MS-TCN + TeCNO) → evaluate. Pipeline validated;
  fixed cuBLAS hang (run.sh) and torch2.6 weights_only load bug. Smoke artifacts use
  `_smoke` suffix (resnet50_smoke.pt, features_smoke/, mstcn_smoke.pt, tecno_smoke.pt).
- [todo] FULL RUN: extract all 80 → train ResNet50 (32 train/8 val) → features (80) →
  MS-TCN + TeCNO → evaluate on test (41-80).
