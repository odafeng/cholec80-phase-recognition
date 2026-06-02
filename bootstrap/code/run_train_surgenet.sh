#!/usr/bin/env bash
# Phase 2: end-to-end train SurgicalMamba with a SurgeNet (surgical-pretrained
# ConvNeXt-V2) backbone instead of ImageNet ConvNeXt-Tiny. Single-variable test
# of "smuggle in more pretraining data". T4 fixes: clear LD_LIBRARY_PATH (cuBLAS),
# float16 AMP (no bf16 on Turing), grad checkpointing, wandb disabled.
set -euo pipefail
cd /home/KHUser/Surgical-Mamba
export CUDA_HOME=/usr/local/cuda-12.9
export WANDB_MODE=disabled
exec env -u LD_LIBRARY_PATH WANDB_MODE=disabled \
  /home/KHUser/cholec80_phase/.venv/bin/python train.py \
    --dataset cholec80 \
    --data_root ./cholec80_preprocessed \
    --phase_annotation_dir phase_ann_pp \
    --tool_annotation_dir _no_tools \
    --backbone convnextv2_tiny \
    --surgenet_weights surgenet_weights/surgenet_convnextv2.pth \
    --head_chunk_size 32 --chunk_size_block 64 \
    --chunk_size_fast_block 64 --chunk_size_slow_block 64 \
    --no_amp --grad_checkpointing \
    --epochs 50 \
    --save_dir ./checkpoints_surgenet
