#!/usr/bin/env bash
# Self-supervised arm (week-scale). Runs unattended in tmux 'mae'.
#   1) wait for SSL frames (videos 1-40 @5fps) + for the GPU to free (expansion done)
#   2) MAE pretrain a ViT-B/16 surgical backbone FROM SCRATCH (resumable, ckpt/25ep)
#   3) extract surgMAE features (all 80 videos @1fps) + run the full BUA matrix
# ViT-B/16 chosen to match EndoViT's architecture -> isolates the pretraining-data
# variable (ours, test-clean, owned) vs EndoViT (public, possible overlap).
set -uo pipefail
cd "$(dirname "$0")"
say() { echo; echo "######## [$(date '+%F %T')] $* ########"; }

say "wait for SSL frame extraction (videos 1-40 @5fps)"
while ! grep -q SSL_FRAMES_EXIT logs/ssl_frames.log 2>/dev/null; do sleep 30; done
echo "ssl frames: $(find data/frames_ssl -name '*.jpg' 2>/dev/null | wc -l)"

say "wait for expansion to free the GPU"
while tmux has-session -t expansion 2>/dev/null \
      && ! grep -q EXPANSION_EXIT logs/expansion.log 2>/dev/null; do sleep 60; done
echo "GPU free; starting MAE."

say "MAE pretraining (ViT-B/16, from scratch) -- ~1 week on GB10 (~155 img/s ceiling)"
# GB10 does ~155 img/s for ViT-B MAE -> ~43 min/epoch -> 200 ep ~= 6 days.
# Lighter decoder (384/4) is a small free win; the decoder is discarded anyway.
./run.sh mae_pretrain.py --frames data/frames_ssl --epochs 200 --bs 512 \
    --warmup 15 --dec_dim 384 --dec_depth 4 --save_every 10 \
    --out checkpoints/surgmae 2>&1 | tee -a logs/mae_pretrain.log

say "extract surgMAE features (all 80 videos @1fps)"
./run.sh extract_features_mae.py --frames data/frames --anno data/phase_annotations \
    --ckpt checkpoints/surgmae/surgmae_encoder_latest.pt --out features_surgmae \
    2>&1 | tee logs/extract_surgmae.log

say "surgMAE BUA matrix"
bash run_reliability.sh features_surgmae surgmae 2>&1 | tee logs/rel_surgmae.log

say "SSL ARM COMPLETE"
echo "SSL_EXIT=0"
