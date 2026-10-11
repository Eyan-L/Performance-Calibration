#!/usr/bin/env python3
"""Run the pre-registered calibration points of a manifest (self-contained kit version of run_points_h200.py;
benchmark commands and arguments unchanged).

Points run sequentially on the host (no container). A deadline (--stop-at, unix s) stops starting new points so a
batch job ends cleanly and the next job resumes. A point with a result file is skipped (resume), never overwritten.
--smoke shortens runs for the backend smoke.
Kit changes: every path is relative to this folder; the model directory comes from $MODEL_PATH; the tensor-parallel
size comes from the manifest ("tp", default 1). The prefill benchmark supports tp_size=1 only, so with tp > 1 prefill
points are recorded as skipped (status.jsonl, rc=null, skip="prefill_tp1_only") instead of being run.
"""
from __future__ import annotations

import argparse, json, os, subprocess, time
from pathlib import Path

E = Path(__file__).resolve().parent
SRC = E / "src"
SHIM = E / "ash_bench_shim.py"
DEC = SRC / "sglang/benchmark/kernels/fused_moe_triton/bench_decode_lane_moe_cta.py"
PRE = SRC / "test/bench_prefill_sm_sweep.py"
CORPUS = {"short": str(E / "inputs/corpus_short"), "long": str(E / "inputs/corpus_long")}
POINT_TIMEOUT_S = 1800


def command(pt, model, attn, out, smoke, tp=1):
    common = ["--model-path", os.environ["MODEL_PATH"], "--random-seed", "0", "--tp-size", str(tp),
              "--kv-cache-dtype", "bfloat16", "--disable-radix-cache", "--attention-backend", attn,
              "--token-corpus-dir", CORPUS[pt["corpus"]], "--token-corpus-split", pt["split"]]
    if pt["kind"] == "decoder":
        bs, ctx = pt["bs"], pt["ctx"]
        warm, iters, steps = (1, 3, 10) if smoke else (pt["warmups"], pt["iters"], pt["steps"])
        return ["$PYTHON", str(SHIM), str(DEC), *common, "--mem-fraction-static", "0.90",
                "--max-running-requests", str(bs), "--max-mamba-cache-size", str(bs),
                "--max-total-tokens", str(bs * ctx + 32768),
                "--short-bs", str(bs), "--short-input-len", str(ctx),
                "--bench-warmups", str(warm), "--bench-iters", str(iters), "--decode-steps", str(steps),
                "--run-modes", "baseline_short_only", "--prefill-chunk-size", "1",
                "--prefill-token-chunk-size", "8192", "--no-png", "--result-prefix", str(out / pt["id"])]
    warm, iters = (1, 2) if smoke else (pt["warmups"], pt["iters"])
    return ["$PYTHON", str(SHIM), str(PRE), *common, "--mem-fraction-static", "0.70",
            "--batch-sizes", *map(str, pt["batches"]), "--request-lengths", *map(str, pt["lengths"]),
            "--bench-warmups", str(warm), "--bench-iters", str(iters), "--prefill-sm-count", "0",
            "--result-file", str(out / f"{pt['id']}.json")]


def done(pt, out):
    return any(out.glob(pt["id"] + "*.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stop-at", type=float, default=float("inf"))
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    for p in man["points"]:
        if (a.only and p["id"] not in a.only) or done(p, a.out):
            continue
        if time.time() + POINT_TIMEOUT_S / 3 > a.stop_at:
            print(json.dumps(dict(stop="deadline", next=p["id"])), flush=True)
            return
        tp = int(man.get("tp", 1))
        if p["kind"] == "prefill" and tp > 1:
            rec = dict(id=p["id"], rc=None, seconds=0, result=False, skip="prefill_tp1_only")
            with open(a.out / "status.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            continue
        cmd = command(p, man["model"], man["attention_backend"], a.out, a.smoke, tp)
        log = a.out / "logs" / f"{p['id']}.log"
        t0 = time.time()
        with open(log, "w") as f:
            f.write("CMD " + " ".join(cmd) + "\n"); f.flush()
            rc = subprocess.call(["bash", "-c", f"cd {SRC}/sglang && exec timeout {POINT_TIMEOUT_S} " + " ".join(cmd)],
                                 stdout=f, stderr=subprocess.STDOUT)
        rec = dict(id=p["id"], rc=rc, seconds=round(time.time() - t0, 1), result=done(p, a.out),
                   job=os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"))
        with open(a.out / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
