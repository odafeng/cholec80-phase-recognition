#!/usr/bin/env bash
# Expansion orchestrator (user opted into "do all"). Runs unattended in tmux.
# Chains the GPU-heavy arms; each arm = features -> full BUA matrix (3 heads x
# {baseline,+BUA} x 5 seeds + ablations + significance + sensitivity).
#   A) EndoViT (pretrained) features            -> rel_endovit
#   B) Fine-tuned EndoViT backbone -> features   -> rel_endovitft
#   D) End-to-end backbone -> features           -> rel_e2e
# (C = ASFormer head is added to every matrix via run_reliability.sh HEADS, and the
#  ResNet50 asformer arm runs separately in tmux 'rel_rn50'.)
set -uo pipefail
cd "$(dirname "$0")"
ts() { date "+%F %T"; }
say() { echo; echo "######## [$(ts)] $* ########"; }

# ---- A) EndoViT (pretrained) ----
say "A: waiting for EndoViT feature extraction (tmux endovit_feat) -> 80 files"
while [ "$(ls features_endovit/*.pt 2>/dev/null | wc -l)" -lt 80 ]; do
  grep -q ENDOVIT_FEAT_EXIT logs/endovit_feat.log 2>/dev/null && break; sleep 20
done
# also let the ResNet50 ASFormer matrix finish first to avoid GPU thrash
while tmux has-session -t rel_rn50 2>/dev/null && \
      ! grep -q PHASE3_EXIT logs/rel_rn50_asformer.log 2>/dev/null; do sleep 20; done
say "A: EndoViT (pretrained) BUA matrix"
bash run_reliability.sh features_endovit endovit 2>&1 | tee logs/rel_endovit.log

# ---- B) Fine-tuned EndoViT ----
if [ ! -f checkpoints/endovit_ft.pt ]; then
  say "B: fine-tune EndoViT backbone"
  ./run.sh train_cnn_endovit.py --frames data/frames --anno data/phase_annotations \
     --epochs 5 --out checkpoints/endovit_ft.pt 2>&1 | tee logs/train_endovit_ft.log
fi
if [ ! -f features_endovit_ft/video80.pt ]; then
  say "B: extract fine-tuned EndoViT features"
  ./run.sh extract_features_endovit.py --frames data/frames --anno data/phase_annotations \
     --ckpt checkpoints/endovit_ft.pt --out features_endovit_ft 2>&1 | tee logs/extract_endovit_ft.log
fi
say "B: EndoViT-ft BUA matrix"
bash run_reliability.sh features_endovit_ft endovitft 2>&1 | tee logs/rel_endovitft.log

# ---- D) End-to-end ----
if [ ! -f checkpoints/e2e.pt ]; then
  say "D: end-to-end joint backbone+temporal training"
  ./run.sh train_e2e.py --frames data/frames --anno data/phase_annotations \
     --init checkpoints/resnet50.pt --window 128 --epochs 8 --out checkpoints/e2e.pt \
     2>&1 | tee logs/train_e2e.log
fi
if [ ! -f features_e2e/video80.pt ]; then
  say "D: extract end-to-end backbone features"
  ./run.sh extract_features.py --frames data/frames --anno data/phase_annotations \
     --ckpt checkpoints/e2e.pt --out features_e2e 2>&1 | tee logs/extract_e2e.log
fi
say "D: end-to-end BUA matrix"
bash run_reliability.sh features_e2e e2e 2>&1 | tee logs/rel_e2e.log

say "EXPANSION COMPLETE"
echo "EXPANSION_EXIT=0"
