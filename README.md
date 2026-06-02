# Cholec80 Phase Recognition — Run Guide

Two-stage pipeline. Stage 1 (CNN) + feature extraction run **once** and feed
**both** temporal models. MS-TCN vs TeCNO differ only by the `--causal` flag.

All commands assume the dedicated venv. Either `source .venv/bin/activate` first,
or prefix every command with `.venv/bin/python`.

```bash
cd /home/KHUser/cholec80_phase
PY=.venv/bin/python
```

## 0. Getting the dataset

The Cholec80 dataset is **not** in this repo (it is ~70 GB and license-restricted —
request access from the [CAMMA group](http://camma.u-strasbg.fr/datasets)). We keep
our copy as `cholec80.zip` in a **private** Cloudflare R2 bucket.

### Download from R2 (with rclone)
```bash
# 1. install rclone
curl https://rclone.org/install.sh | sudo bash

# 2. configure an R2 remote (fill in YOUR own credentials — keep them secret)
rclone config create r2 s3 \
  provider=Cloudflare \
  access_key_id=<YOUR_R2_ACCESS_KEY_ID> \
  secret_access_key=<YOUR_R2_SECRET_ACCESS_KEY> \
  endpoint=https://<YOUR_ACCOUNT_ID>.r2.cloudflarestorage.com \
  region=auto no_check_bucket=true

# 3. download (~70 GB)
rclone copy r2:<YOUR_BUCKET>/cholec80.zip ./data/ --progress
```

### Upload notes (lessons learned)
- Use an R2 API token with **Object Read & Write**; it cannot `ListBuckets` or
  `CreateBucket`, so pass `--s3-no-check-bucket` on upload.
- For a 70 GB file add **`--s3-disable-checksum`** — otherwise rclone hashes the
  whole file first and appears to hang at `0 B/s` (it is busy in `s3.prepareUpload`).
- Tune large uploads with `--s3-chunk-size 64M --s3-upload-concurrency 4`.
- Keep the bucket **private** (disable the public `r2.dev` URL) — Cholec80's
  license does not permit public redistribution.

Then unzip into `data/`:
```bash
cd data && python -c "import zipfile; zipfile.ZipFile('cholec80.zip').extractall('.')"
```

## Data layout (after unzip)
```
data/videos/             video01.mp4 ... video80.mp4  (+ videoXX-timestamp.txt)
data/phase_annotations/  video01-phase.txt ...
data/tool_annotations/   video01-tool.txt ...         (used by train_cnn_mtl.py)
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
