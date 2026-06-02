#!/usr/bin/env bash
# CONTROL run: identical to the SurgeNet run EXCEPT the backbone is ImageNet-init
# (no --surgenet_weights). Same architecture (convnextv2_tiny), same everything →
# isolates ONLY the pretraining data (ImageNet vs surgical).
set -euo pipefail
cd /home/KHUser/Surgical-Mamba
export CUDA_HOME=/usr/local/cuda-12.9
exec env -u LD_LIBRARY_PATH WANDB_MODE=disabled \
  /home/KHUser/cholec80_phase/.venv/bin/python train.py \
    --dataset cholec80 \
    --data_root ./cholec80_preprocessed \
    --phase_annotation_dir phase_ann_pp \
    --tool_annotation_dir _no_tools \
    --backbone convnextv2_tiny \
    --head_chunk_size 32 --chunk_size_block 64 \
    --chunk_size_fast_block 64 --chunk_size_slow_block 64 \
    --no_amp --grad_checkpointing \
    --epochs 50 \
    --save_dir ./checkpoints_control
