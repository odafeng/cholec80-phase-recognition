#!/usr/bin/env bash
# Always launch Python through the project venv WITHOUT the system
# LD_LIBRARY_PATH. The system path (/usr/lib/x86_64-linux-gnu,
# /usr/local/gib/lib64, system CUDA 12.9) otherwise shadows torch's own
# bundled cuBLAS (cu129) and triggers:
#     "Invalid handle. Cannot load symbol cublasLtCreate"
# which silently hangs training. Clearing LD_LIBRARY_PATH lets torch use its
# bundled libs via RPATH.
#
# Usage:  ./run.sh train_cnn.py --frames data/frames ...
cd "$(dirname "$0")"
exec env -u LD_LIBRARY_PATH .venv/bin/python "$@"
