#!/usr/bin/env bash
# One-shot dashboard for all running jobs. Usage: ./status.sh   (or: watch -n10 ./status.sh)
cd "$(dirname "$0")"
hr() { printf '%.0s─' {1..60}; echo; }

hr; echo "TMUX SESSIONS";  hr
tmux ls 2>/dev/null || echo "(no tmux server)"

hr; echo "GPU"; hr
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3

hr; echo "RERUN (LoViT re-train + re-eval)"; hr
if tmux has-session -t rerun 2>/dev/null; then echo "● running"; else
  grep -q RERUN_EXIT logs/rerun.log 2>/dev/null && echo "✓ done" || echo "○ not active"; fi
tr '\r' '\n' < logs/rerun.log 2>/dev/null | grep '########' | tail -2
for bb in rn50 endovit endovitft e2e; do
  printf "  %-10s lovit-retrain %s/10\n" "$bb" "$(ls checkpoints/rel_$bb/lovit_causal_*.pt 2>/dev/null | wc -l)"
done

hr; echo "MAE pretraining (SSL backbone)"; hr
if tmux has-session -t mae 2>/dev/null; then echo "● running"; else echo "○ paused/not started"; fi
grep -E '^epoch' logs/mae_pretrain.log 2>/dev/null | tail -2
echo "  encoder checkpoints: $(ls checkpoints/surgmae/surgmae_ep*.pt 2>/dev/null | wc -l)"

hr; echo "DISK"; hr
df -h . | tail -1
