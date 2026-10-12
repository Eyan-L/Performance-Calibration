# A100 calibration with official SGLang 0.5.21

The SIF and SGLang inference source are unmodified. The external adapter calls
`sglang.benchmark.one_batch` for loading and batch preparation. The reference
kit remains unchanged; its corpus and manifest are inputs, not import paths.
Compilation caches use node-local `/tmp`; results and logs use the specified
scratch run directory.

## Measurement definition

- BF16 weights / KV, seed 0, no radix cache; custom all-reduce is explicitly
  disabled with `--disable-custom-all-reduce` in both measurement and validation.
  Use the same flag in subsequent replay runs to match this communication policy.
- Decode: fixed-context batch, native full CUDA graph, 30 forwards per sample,
  3 warmups and 9 samples. Context preparation uses one request at a time and
  8192-token chunks, outside timing. The repeated batch does not grow its context.
- Prefill: one eager forward of the whole batch, 2 warmups and 5 samples;
  CUDA events bracket `model.forward`, excluding input preparation, attention
  planning, and sampling. The wrapper records events and calls the original
  forward once. It does not change the model output or inference kernels.
- Primary statistics use rank 0, following the reference decode reporting and
  official SGLang multi-TP reporting. All-rank raw samples, per-rank summaries,
  and per-iteration maxima remain available as diagnostics, not fitting inputs.
- There is no added per-sample TP barrier. Native initialization and final
  worker coordination remain; local CUDA synchronization and event completion
  establish the measured intervals. The official `one_batch` benchmark uses
  host wall time, so its timing interval is not substituted for these events.
- Numerical logits, actual batch/context, graph execution, and a TP collective
  are checked in separate validation processes before measurement. Those
  processes produce no timing samples and cannot enter collection or resume.
- TP=4 prefill is an extension of the reference's TP=1-only prefill script.
- Both phases cap the KV pool at the original group's maximum `batch * context`
  plus 32768 tokens. The headroom is an engineering choice, not an official
  SGLang constant. Partial prefill resume preserves the original group capacity.
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
Partial / failed worker records remain in `.attempts/`. Completion of measurement
coverage does not rewrite their original process exit states.
`collect_results.py RUN NEW_OUTPUT` recovers a prefill shape only when
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
That historical smoke retains its frozen code and earlier rank-max / barrier
policy; it is not imported into the campaign below.

## Expanded acceptance and full campaign

```bash
python3 -B campaign.py prepare
# Use the printed campaign_dir for both submissions; send PBS output to scratch.
qsub -v CALIB_CAMPAIGN_DIR=/path/to/campaign \
  -o /path/to/campaign/pilot.pbs.log /path/to/campaign/code/nscc_qwen38_pilot.pbs
```

The 4-hour pilot selects 21 groups / 36 shapes from the unchanged full manifest
and uses **formal sampling counts**, not `--smoke`. It first checks four actual
GPUs, then runs independent numerical checks on the established decode / prefill
shapes, then a formal-count bootstrap. A bootstrap failure stops the expanded
grid. The bootstrap's valid timing samples are resumed by the main pilot driver.

The pilot collects raw evidence, performs the original fit, exports coefficients,
CSV tables and figures, and writes `jobs/JOB/acceptance.json`. Fitting errors are
reported, not used to filter slow points. Incomplete core shapes, unexpected
errors, mismatched evidence, or repeat drift above 10% prevent acceptance.
Non-core shapes with actual OOM evidence are reported as unsupported.

After reviewing a successful `pilot_acceptance.json`, submit the 24-hour phase:

```bash
qsub -v CALIB_CAMPAIGN_DIR=/path/to/campaign \
  -o /path/to/campaign/full.pbs.log /path/to/campaign/code/nscc_qwen38_full.pbs
```

Both jobs use the **same frozen code, full manifest, signature and result store**.
All 36 successful pilot shapes are skipped: ideally 65 groups / 66 shapes remain.
Partial prefill resumes only missing shapes while preserving the original group's
resource configuration. Different physical nodes are allowed only if their
hardware/runtime fingerprints and available communication configuration match.
Each phase's independent numerical checks are separate from reused measurements.
The full phase audits that pilot source attempts and samples remain unchanged.
There is no automatic full-job submission before pilot acceptance and review.

Qwen3.6 needs its own model config and memory-aware manifest; do not reuse the
Qwen3.8 manifest for it.

### Updating the prefill KV pool after an accepted pilot

```bash
python3 -B campaign.py prepare --reuse-decode-from /path/to/original/campaign
# Submit the new campaign's frozen full PBS with CALIB_CAMPAIGN_DIR as above.
```

This creates a separate frozen campaign and does not modify the original code
or measurements. It verifies the original pilot, unchanged decode computation,
timing, arguments, inputs and runtime, then pins all 14 original decode results
and their raw artifacts. Reused records retain their original signatures and
job IDs; collection explicitly records each measurement's source signature.
Every prefill shape is measured again with the capped pool. The full job first
checks the hardware against the source pilot, runs independent forward checks,
and measures the base prefill group and the three previously failing workload
shapes at formal sampling counts. These new samples are reused by the subsequent
full-grid driver. A failed bootstrap stops the full grid and preserves evidence.
