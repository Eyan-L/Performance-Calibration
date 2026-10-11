# Sourced inside the PBS job on the H200 node. Same interpreter + site-packages as the r008 release on H200
# (results/code_releases/pd-proportional-rr-r008-nvlink-device-guard/{formal.pbs,config.json}); the SGLang tree is the
# AMD experiment's frozen src/ (byte-identical to r008 sglang/python).
source /etc/profile.d/modules.sh
module purge
module load gnu/gcc-12.3 cuda/13.0 python/3.12.13
unset PYTHONHOME CUDA_LAUNCH_BLOCKING
SRC=/home/bzhou011/ASH/experiments/amd_prefill_decode_perf_r001/src
HK=/home/bzhou011/Highway_kernel
export PYTHON=/usr/local/python/3.12.13/bin/python3.12
export PATH=$PATH:$HK/.venv-sglang/bin   # ninja for flashinfer JIT (78772 failed without it); appended so module python stays first
export PYTHONPATH=$SRC/sglang/python:$SRC/sglang:$SRC:$HK/results/envs/mooncake_cuda13_0.3.13.post1:$HK/.venv-sglang/lib/python3.12/site-packages:/usr/local/pytorch/2.11.0/lib/python3.12/site-packages
CACHE=${TMPDIR:-/tmp}/${USER}_ash_q38_${PBS_JOBID:-manual}
mkdir -p $CACHE
export XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCH_EXTENSIONS_DIR=$CACHE/torch \
       FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TVM_FFI_CACHE_DIR=$CACHE/tvm TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 SGLANG_MOE_EXPERT_READ_TRACKING_MODE=off
