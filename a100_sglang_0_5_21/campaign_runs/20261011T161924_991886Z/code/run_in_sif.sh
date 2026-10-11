#!/usr/bin/env bash
# Run a version check, module, or external script with the official SGLang runtime.
# Examples: bash run_in_sif.sh --check
#           bash run_in_sif.sh --gpu --module sglang.benchmark.one_batch <args>
set -euo pipefail

runtime_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scratch_root=/scratch/users/ntu/n2602695
image_path=$scratch_root/sif/sglang_v0.5.21_cuda13.sif
runtime_options=()
if [[ ${1:-} == --gpu ]]; then
    runtime_options+=(--nv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}")
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    driver_major=${driver_version%%.*}
    compat_policy=${CALIB_CUDA_COMPAT:-auto}
    if [[ $compat_policy != auto && $compat_policy != on && $compat_policy != off ]]; then
        echo 'CALIB_CUDA_COMPAT must be auto, on, or off' >&2
        exit 1
    fi
    if [[ $compat_policy == on || ( $compat_policy == auto && $driver_major -lt 580 ) ]]; then
        # Official image includes cuda-compat-13-0. Prefer its user-space CUDA
        # driver on older data-center drivers; actual compatibility is verified
        # by gpu_preflight.py before any model measurement.
        runtime_options+=(--env 'LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/.singularity.d/libs')
        echo "CUDA 13 forward compatibility enabled; host driver $driver_version" >&2
    else
        echo "Using host CUDA driver $driver_version" >&2
    fi
    shift
fi
if ! command -v singularity >/dev/null 2>&1; then
    module load singularity/4.3.1
fi
if [[ ! -r $image_path ]]; then
    echo "SGLang 0.5.21 image not readable: $image_path" >&2
    exit 1
fi
# --cleanenv removes host environment leakage; -I ignores Python environment
# variables, user site packages, and the working directory as an import source.
exec singularity exec --cleanenv "${runtime_options[@]}" \
    --env OMP_NUM_THREADS=4 --env MAX_JOBS=4 --env HF_HUB_OFFLINE=1 \
    --env "CALIB_JOB_ID=${PBS_JOBID:-${SLURM_JOB_ID:-}}" \
    --bind "$scratch_root:$scratch_root" "$image_path" \
    /opt/sglang/bin/python3 -I -B "$runtime_dir/runtime_guard.py" "$@"
