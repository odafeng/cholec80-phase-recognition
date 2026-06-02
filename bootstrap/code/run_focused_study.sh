#!/usr/bin/env bash
# Focused study: SurgeNet(surgical) vs Control(ImageNet) backbone, SAME architecture
# and protocol -> isolates pretraining data. Waits for the running SurgeNet train
# (pid 70735) to finish, evaluates it, then trains+evaluates the control.
set -euo pipefail
cd /home/KHUser/Surgical-Mamba
VENV=/home/KHUser/cholec80_phase/.venv/bin
EV="env -u LD_LIBRARY_PATH $VENV/python"
say() { echo; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }

say "WAIT for SurgeNet training (pid 70735) to finish"
while kill -0 70735 2>/dev/null; do sleep 120; done

say "SurgeNet training done -> evaluate (treatment)"
$EV eval_trained.py --ckpt checkpoints_surgenet/best_causal.pt \
    --backbone convnextv2_tiny --tag "SurgeNet-surgical" 2>&1 | tail -6

say "Train CONTROL (ImageNet convnextv2, identical settings)"
./run_train_control.sh

say "Control training done -> evaluate (control)"
$EV eval_trained.py --ckpt checkpoints_control/best_causal.pt \
    --backbone convnextv2_tiny --tag "Control-ImageNet" 2>&1 | tail -6

say "FOCUSED STUDY DONE"
