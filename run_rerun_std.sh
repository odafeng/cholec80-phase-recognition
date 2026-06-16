#!/usr/bin/env bash
# Clean re-run on STANDARDIZED features (fixes the feature-scale confound).
# In-place: each features dir is backed up to *.raw and replaced by its
# standardized version (per-dim z-score from TRAIN stats), so every downstream
# script reads standardized features with ZERO code changes. Then the whole BUA
# matrix is retrained for all 5 backbones. Unattended in tmux.
set -uo pipefail
cd "$(dirname "$0")"
say() { echo; echo "######## [$(date '+%F %T')] $* ########"; }
# backbone tag -> feature dir
BBS=(rn50 endovit endovitft e2e surgmae)
declare -A FEAT=( [rn50]=features [endovit]=features_endovit \
                  [endovitft]=features_endovit_ft [e2e]=features_e2e \
                  [surgmae]=features_surgmae )

# 1) standardize each feature dir in place (backup originals to .raw, once)
for bb in "${BBS[@]}"; do
  d="${FEAT[$bb]}"
  if [ ! -d "${d}.raw" ]; then
    say "standardize $d"
    mv "$d" "${d}.raw"
    ./run.sh standardize_features.py --src "${d}.raw" --dst "$d"
  else
    echo "skip standardize $d (.raw already exists)"
  fi
done

# 2) remove old UNSTANDARDIZED checkpoints/results (regenerable; superseded)
say "clear old unstandardized rel_* checkpoints/results"
rm -rf checkpoints/rel_* results/rel_*

# 3) retrain the full matrix for all backbones on standardized features
for bb in "${BBS[@]}"; do
  say "RE-RUN matrix (standardized) : $bb  on ${FEAT[$bb]}"
  bash run_reliability.sh "${FEAT[$bb]}" "$bb" 2>&1 | tee "logs/rerunstd_${bb}.log"
done

say "STD RERUN COMPLETE"
echo "STDRERUN_EXIT=0"
