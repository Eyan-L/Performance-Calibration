# Source this before running anything in the kit:  source env.sh
# Fill in the four lines marked EDIT for your machine. Everything else is relative to this folder.
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHON=${PYTHON:-python3}                          # EDIT: Python 3.12 with the packages of requirements_reference.txt
export MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3.8-27B}     # EDIT: local Qwen3.8-27B (BF16) directory
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}   # EDIT: GPUs used by one point (= tp in the manifest)
EXTRA_SITE=${EXTRA_SITE:-}                                # EDIT (optional): extra site-packages dirs, ':'-separated
# The frozen SGLang tree of this kit must come FIRST on PYTHONPATH (do not pip install sglang into the same env).
export PYTHONPATH=$K/src/sglang/python:$K/src/sglang:$K/src${EXTRA_SITE:+:$EXTRA_SITE}
CACHE=${TMPDIR:-/tmp}/${USER}_qwen38_calib
mkdir -p $CACHE
export XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCH_EXTENSIONS_DIR=$CACHE/torch \
       FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TVM_FFI_CACHE_DIR=$CACHE/tvm TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 SGLANG_MOE_EXPERT_READ_TRACKING_MODE=off
unset PYTHONHOME CUDA_LAUNCH_BLOCKING
# flashinfer compiles kernels at first use: `ninja` must be on PATH.
command -v ninja >/dev/null || echo "env.sh: WARNING ninja not found on PATH (flashinfer JIT will fail)"
