#!/bin/bash
# Run the whole calibration: backend smoke (fa3 vs flashinfer) -> frozen backend -> full grid. Resumable: run it again
# after an interruption and it continues with the missing points (finished points are never re-run or overwritten).
# Usage:  source env.sh;  bash run_grid.sh <manifest prefix> <run dir> [hours available]
#   e.g.  bash run_grid.sh manifests/qwen38bf16_a100x4 runs/qwen38bf16_a100x4 4
# On a batch system, put the two lines above into your job script (one job = the GPUs of one point).
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd)
PFX=${1:?manifest prefix, e.g. manifests/qwen38bf16_a100x4}; RUN=${2:?run dir}; HOURS=${3:-1000}
: "${PYTHON:?source env.sh first}" "${MODEL_PATH:?source env.sh first}"
STOP=$(( $(date +%s) + HOURS*3600 - 25*60 ))       # stop starting new points 25 min before the end
mkdir -p $RUN
J=$(date +%Y%m%dT%H%M%S)
{ date -u; hostname; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm --format=csv,noheader
  nvidia-smi topo -m; $PYTHON -c "import sglang,torch,flashinfer,sys;print(sglang.__file__,torch.__version__,torch.version.cuda,flashinfer.__version__,sys.version)"
  echo "MODEL_PATH=$MODEL_PATH CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  sha256sum $K/run_points.py $K/select_backend.py $K/ash_bench_shim.py $0 ${PFX}_*.json
} > $RUN/env_$J.txt 2>&1
SMOKE_POINTS="dec_train_bs4_ctx8192 dec_train_bs16_ctx16384 dec_train_bs4_ctx65536 dec_train_bs8_ctx98304 pre_train_g0"
TP=$($PYTHON -c "import json;print(json.load(open('${PFX}_fa3.json')).get('tp', 1))")
# the prefill benchmark supports tp=1 only: with tp > 1 the backend is chosen on the decode smoke points alone
[ "$TP" -gt 1 ] && SMOKE_POINTS="dec_train_bs4_ctx8192 dec_train_bs16_ctx16384 dec_train_bs4_ctx65536 dec_train_bs8_ctx98304"
if [ ! -e $RUN/FROZEN_BACKEND ]; then
  for b in fa3 flashinfer; do
    echo "$(date -u +%FT%TZ) smoke $b"
    $PYTHON $K/run_points.py ${PFX}_$b.json $RUN/smoke/$b --smoke --only $SMOKE_POINTS --stop-at $STOP
  done
  $PYTHON $K/select_backend.py $RUN/smoke $SMOKE_POINTS || { echo "smoke failed: see $RUN/smoke/*/logs"; exit 1; }
  cp ${PFX}_$(cat $RUN/FROZEN_BACKEND).json $RUN/manifest.json
fi
echo "$(date -u +%FT%TZ) grid backend=$(cat $RUN/FROZEN_BACKEND)"
$PYTHON $K/run_points.py $RUN/manifest.json $RUN/grid --stop-at $STOP
echo "$(date -u +%FT%TZ) exit; results $(ls $RUN/grid/*.json 2>/dev/null | wc -l) / $($PYTHON -c "import json;print(len(json.load(open('$RUN/manifest.json'))['points']))")"
