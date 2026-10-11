# A100 calibration with official SGLang 0.5.21

The SIF and SGLang inference source are unmodified. The external adapter calls
`sglang.benchmark.one_batch` for loading and batch preparation. The reference
kit remains unchanged; its corpus and manifest are inputs, not import paths.
Compilation caches use node-local `/tmp`; results and logs use the specified
scratch run directory.

## Measurement definition

- BF16 weights / KV, seed 0, no radix cache; native all-reduce selection.
- Decode: fixed-context batch, native full CUDA graph, 30 forwards per sample,
  3 warmups and 9 samples. Context preparation uses one request at a time and
  8192-token chunks, outside timing. The repeated batch does not grow its context.
- Prefill: one eager forward of the whole batch, 2 warmups and 5 samples;
  CUDA events bracket `model.forward`, excluding input preparation, attention
  planning, and sampling. The wrapper records events and calls the original
  forward once. It does not change the model output or inference kernels.
- TP workers synchronize before each sample. Raw samples are retained per rank;
  each iteration is aggregated by its maximum rank time before median / p90.
- `--smoke` uses the reference's shorter counts (decode 1/3/10, prefill 1/2).
  Store smoke and full-grid measurements in separate directories.

## Entry points

Run commands on NSCC. GPU measurement commands require a GPU allocation.

```bash
bash run_in_sif.sh --check
bash run_in_sif.sh calibrate_point.py --check-api
```

Use host Python for the driver; it launches one SIF process per manifest point:

```bash
python3 run_points.py /path/to/manifest.json /path/to/new/run \
  --model-path /path/to/BF16/model --smoke \
  --only dec_train_bs4_ctx8192 pre_train_g0 --dry-run
```

Remove `--dry-run` to measure inside a GPU allocation. For a full grid, omit
`--smoke` and `--only`. `--stop-at` takes a Unix deadline; a new point starts
only when its timeout and shutdown margin fit. The default point timeout is
1800 seconds. GPU visibility must contain exactly the manifest's TP count.

## Results and fitting

The complete result JSON schema retains the reference fit's decode `summaries`
and prefill `points` fields. The original `fit/fit_amd_models.py` can read them;
its formulas, cross-validation, holdout and repeat definitions are unchanged.
Partial / failed worker records remain in `.attempts/` and are not declared
complete. `collect_results.py RUN NEW_OUTPUT` recovers a prefill shape only when
all TP workers in the same matching attempt have the full valid sample count.
It includes missing shapes and provenance in `coverage.json`; a failed group
remains incomplete. Selection uses attempt order, never observed latency.
`--require-complete` fails if any requested shape is missing. `fit_results.py`
checks the training design and validation folds before calling the unchanged
reference fit. It rejects smoke data; `--allow-incomplete` is required to fit
a full run with missing registered shapes.

`run_config.json` freezes source hashes, runtime, model and manifest. Changed
code or configuration requires a new run directory. Each attempt has a unique
log and appends to `status.jsonl`; only complete matching results are resumed.

`nscc_qwen38_smoke.pbs` first validates four allocated A100 40GB devices with
BF16 CUDA matmuls, then measures one decode point and four prefill shapes.
Submit an immutable copy with `CALIB_LAUNCH_DIR` and direct PBS output to scratch.
The wrapper automatically uses the image's CUDA 13 forward compatibility
libraries on host drivers older than R580 (`CALIB_CUDA_COMPAT=on/off` overrides
selection). The GPU preflight verifies actual execution and the loaded driver.
Qwen3.6 needs its own model config and memory-aware
manifest; do not reuse the Qwen3.8 manifest for it.
