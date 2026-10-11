# qwen38_calib_kit_r001 — decode/prefill step-cost calibration of Qwen3.8-27B (BF16), self-contained

Measures the per-step time of the SGLang decoder and the prefill forward over a pre-registered grid of batch sizes and
context lengths, then fits

- decode step:  `T_ms = t0 + tB*B + tK*K`, with K = attention KV GB read per step = B*ctx*kv_GB_per_token;
- prefill:      `T_ms = p0 + pN*N + pA*A`, with N = batch*len tokens and A = batch*len^2/2/1e6.

Everything the calibration needs is inside this folder. Only three things come from outside: a Python environment, the
GPUs, and the model weights.

## Contents
| Path | What |
|---|---|
| `make_points.py` | pre-registers the grid: B 1..28 x ctx 4K..96K (+ holdout, repeats) and the prefill groups; skips points whose KV exceeds 75% of (GPU memory x tp - checkpoint) |
| `manifests/qwen38bf16_a100x4_{fa3,flashinfer}.json` | grid for 4 x A100 40GB, tp 4 (86 points, 8 skipped); regenerate for other GPUs, see below |
| `run_grid.sh` | backend smoke (fa3 vs flashinfer) -> `select_backend.py` freezes the faster one -> full grid; resumable |
| `run_points.py` | runs the points of a manifest one by one (one process per point, 30 min timeout per point) |
| `select_backend.py` | rule fixed in advance: lowest geometric-mean smoke latency; a backend missing any smoke result is ineligible |
| `ash_bench_shim.py` | small logged shims needed by the benchmarks for this dense model (timing only) |
| `src/` | frozen SGLang source tree used for the measurements (python only) + the two benchmarks: `src/sglang/benchmark/kernels/fused_moe_triton/bench_decode_lane_moe_cta.py` (decode) and `src/test/bench_prefill_sm_sweep.py` (prefill); `src/SOURCE_SHA256SUMS` lists all 3164 files |
| `inputs/corpus_short`, `inputs/corpus_long` | token-id corpora the benchmarks draw requests from (short <= 24K, long 32K..96K), with `train`/`holdout` splits |
| `inputs/model_configs/qwen38bf16.config.json` | model config used by the fit (KV bytes per token) |
| `fit/fit_amd_models.py` | the fit: least squares + 5-fold CV for decode, NNLS for prefill, holdout errors; numpy + scipy |
| `env.sh` | environment template (edit 4 lines) |
| `requirements_reference.txt` | exact package versions of the environment the method was developed with |

## 1. Environment
- Python 3.12 with the packages in `requirements_reference.txt`: in particular torch 2.11.0 (CUDA 13.0), triton 3.6.0,
  flashinfer 0.6.12, sglang_kernel 0.4.3, transformers 5.8.1, numpy, scipy. Install the wheels that match YOUR GPU and
  CUDA version. Do **not** `pip install sglang`: the SGLang code is `src/sglang/python`, which `env.sh` puts first on
  `PYTHONPATH`. `ninja` must be on `PATH`, because flashinfer compiles kernels at first use.
- Weights: Qwen3.8-27B BF16. Check them with `(cd $MODEL_PATH && sha256sum -c <kit>/inputs/Qwen3.8-27B@1d4bf0f2ff60.sha256)` (18 shards).
- Edit the marked lines of `env.sh` (`PYTHON`, `MODEL_PATH`, `CUDA_VISIBLE_DEVICES`, optional `EXTRA_SITE`), then
  `source env.sh` and run the quick check
  `cd src/sglang && $PYTHON -c "import sglang, sglang.bench_one_batch; print(sglang.__file__)"`. It must print a path
  inside this folder.
- Optional: `(cd src && sha256sum -c --quiet SOURCE_SHA256SUMS)` checks that the tree is unchanged.

## 2. Grid for your GPUs
The checkpoint is 56 GB, so it needs `--tp 2` on 80 GB cards and `--tp 4` on 40 GB cards.
```bash
for b in fa3 flashinfer; do
  python3 make_points.py --model qwen38bf16 --gpu a100_40g --tp 4 --attn $b --out manifests/qwen38bf16_a100x4_$b.json
done
```
GPU names: `a100_40g`, `a100_80g`, `h100_80g` (add yours to `HBM_GB` in `make_points.py` if missing).
`CUDA_VISIBLE_DEVICES` must list exactly `tp` GPUs.

## 3. Run
```bash
source env.sh
bash run_grid.sh manifests/qwen38bf16_a100x4 runs/qwen38bf16_a100x4 4     # 4 = hours available in this session/job
```
Run the same command again to resume. Already finished points are skipped and never overwritten.
- `fa3` needs Hopper GPUs. On A100 its smoke fails, so `flashinfer` is selected automatically (see
  `runs/.../backend_choice.json`).
- Progress: `runs/<run>/grid/status.jsonl` (one line per point: rc, seconds, result). Logs:
  `runs/<run>/grid/logs/<point>.log`. Environment record: `runs/<run>/env_<time>.txt`.
- **Limitation with tp > 1**: the prefill benchmark supports `tp_size=1` only. With tp > 1 the prefill points are
  recorded as `skip: prefill_tp1_only` and only the decode model can be fitted. Report this, and do not change the
  benchmark without asking.

## 4. Fit
```bash
$PYTHON fit/fit_amd_models.py runs/qwen38bf16_a100x4/grid --manifest runs/qwen38bf16_a100x4/manifest.json \
  --model-config inputs/model_configs/qwen38bf16.config.json --out results/fit_qwen38bf16_a100x4
```
Output: `results/.../model.json` (coefficients t0, tB, tK, p0, pN, pA, CV and holdout errors), `decoder_points.csv`,
`prefill_points.csv`. Report the coefficients, the CV and holdout errors (overall and per context), and plots of
measured vs predicted step time.

## Rules
- Do not edit `src/`, the benchmarks, `select_backend.py` or a manifest after measuring starts. If something has to
  change, ask first, and write a new manifest or run directory rather than editing an old one.
- Never delete failed points. `status.jsonl` keeps them, and the fit lists missing pre-registered points.
