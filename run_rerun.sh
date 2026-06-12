#!/usr/bin/env bash
# Re-run after the code-review fixes. Unattended in tmux 'rerun'.
#   - LoViT checkpoints are DELETED so they re-train with the causality fix
#     (GroupNorm-over-time -> per-frame LayerNorm). TeCNO/ASFormer checkpoints are
#     kept (still valid) but ALL heads are re-EVALUATED so the latency / ECE
#     metric fixes propagate into every npz. Significance re-runs per backbone.
set -uo pipefail
cd "$(dirname "$0")"
say() { echo; echo "######## [$(date '+%F %T')] $* ########"; }
declare -A FEAT=( [rn50]=features [endovit]=features_endovit \
                  [endovitft]=features_endovit_ft [e2e]=features_e2e )

for bb in rn50 endovit endovitft e2e; do
  rm -f checkpoints/rel_${bb}/lovit_causal_*.pt checkpoints/rel_${bb}/lovit_causal_*.pt.temp.json
done

for bb in rn50 endovit endovitft e2e; do
  say "RE-RUN $bb : LoViT retrains (causal-fixed); all heads re-eval (fixed metrics)"
  bash run_reliability.sh "${FEAT[$bb]}" "$bb" 2>&1 | tee logs/rerun_${bb}.log
done

say "RERUN COMPLETE -- relaunching MAE pretraining (was paused for the rerun)"
tmux kill-session -t mae 2>/dev/null || true
tmux new-session -d -s mae "bash run_ssl.sh 2>&1 | tee logs/ssl_arm.log"
echo "RERUN_EXIT=0"
