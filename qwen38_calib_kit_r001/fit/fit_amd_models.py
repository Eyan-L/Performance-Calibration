#!/usr/bin/env python3
"""Fit AMD decoder/prefill latency models from one grid run directory.

Decoder (same form as the H200 regular decoder, ANALYTICAL_MODEL.md §4.2):
    T_step_ms = t0 + tB*B + tK*K,   K = attention KV GB = B*ctx*kv_gb_per_token
  lstsq on train-split points; five-fold checkerboard CV over
  floor(log2 B)+floor(log2 ctx) (as H200); task-level holdout on holdout split;
  stability from the repeat points.
Prefill (full GPU; H200 §5.1 used T = p0 + F/P_peak, here the FLOP term is split
into its structural parts so no FLOP estimator is needed):
    T_ms = p0 + pN*N + pA*A,   N = batch*len tokens, A = batch*len^2/2 / 1e6
  NNLS weighted by 1/T (relative error); leave-one-point-out CV; holdout split.
Readings never filter points by outcome: every result file in the run is used,
and missing pre-registered points are listed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


def kv_gb_per_token(config: Path) -> float:
    t = json.loads(config.read_text())["text_config"]
    full = t["layer_types"].count("full_attention")
    return full * 2 * t["num_key_value_heads"] * t["head_dim"] * 2 / 1e9  # BF16 KV


def metrics(pred: np.ndarray, meas: np.ndarray) -> dict:
    if len(meas) == 0:
        return {"n": 0}
    ape = np.abs(pred - meas) / meas
    return {"n": int(len(meas)), "mape_percent": float(ape.mean() * 100),
            "p95_ape_percent": float(np.percentile(ape, 95) * 100),
            "max_ape_percent": float(ape.max() * 100)}


def load_decoder(run: Path, kvg: float) -> list[dict]:
    rows = []
    for path in sorted(run.glob("dec_*.json")):
        m = re.fullmatch(r"dec_(train|holdout|repeat)_bs(\d+)_ctx(\d+)", path.stem)
        if not m:
            continue
        split, bs, ctx = m.group(1), int(m.group(2)), int(m.group(3))
        s = json.loads(path.read_text())["summaries"][0]
        rows.append(dict(split=split, bs=bs, ctx=ctx, K=bs * ctx * kvg,
                         step_ms=float(s["total_ms_per_step_median"]),
                         step_ms_p90=float(s["total_ms_per_step_p90"]),
                         step_ms_min=float(s.get("total_ms_per_step_min", float("nan")))))
    return rows


def fit_decoder(rows: list[dict]) -> dict:
    tr = [r for r in rows if r["split"] == "train"]
    X = lambda rs: np.array([[1.0, r["bs"], r["K"]] for r in rs])
    y = lambda rs: np.array([r["step_ms"] for r in rs])
    coef, *_ = np.linalg.lstsq(X(tr), y(tr), rcond=None)
    folds = np.array([(int(math.log2(r["bs"])) + int(math.log2(r["ctx"]))) % 5 for r in tr])
    cv = np.zeros(len(tr))
    for f in sorted(set(folds)):
        c, *_ = np.linalg.lstsq(X(tr)[folds != f], y(tr)[folds != f], rcond=None)
        cv[folds == f] = X(tr)[folds == f] @ c
    ho = [r for r in rows if r["split"] == "holdout"]
    for r in rows:
        r["pred_ms"] = float(np.array([1.0, r["bs"], r["K"]]) @ coef)
    by_ctx = {}
    for ctx in sorted({r["ctx"] for r in tr}):
        sel = [r for r in tr if r["ctx"] == ctx]
        by_ctx[ctx] = metrics(np.array([r["pred_ms"] for r in sel]), y(sel))
    train_index = {(r["bs"], r["ctx"]): r["step_ms"] for r in tr}
    repeats = [dict(bs=r["bs"], ctx=r["ctx"], first=train_index.get((r["bs"], r["ctx"])),
                    repeat=r["step_ms"]) for r in rows if r["split"] == "repeat"]
    for rep in repeats:
        if rep["first"]:
            rep["diff_percent"] = 100 * (rep["repeat"] - rep["first"]) / rep["first"]
    return {
        "formula": "step_ms = t0 + tB*B + tK*K (K = attention KV GB, BF16)",
        "coefficients": {"t0_ms": coef[0], "tB_ms_per_request": coef[1],
                         "tK_ms_per_kv_gb": coef[2]},
        "train": metrics(X(tr) @ coef, y(tr)),
        "cv_checkerboard5": metrics(cv, y(tr)),
        "holdout": metrics(np.array([r["pred_ms"] for r in ho]), y(ho)),
        "train_by_context": by_ctx,
        "repeat_stability": repeats,
        "domain": {"bs": [min(r["bs"] for r in tr), max(r["bs"] for r in tr)],
                   "ctx": [min(r["ctx"] for r in tr), max(r["ctx"] for r in tr)]},
    }


def load_prefill(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("pre_*.json")):
        split = "holdout" if "_holdout_" in path.stem else "train"
        for p in json.loads(path.read_text())["points"]:
            if p.get("status") != "ok":
                rows.append(dict(split=split, batch=p["batch_size"], length=p["request_length"],
                                 status=p.get("status"), ms=float("nan")))
                continue
            rows.append(dict(split=split, batch=int(p["batch_size"]), length=int(p["request_length"]),
                             status="ok", ms=float(p["forward_ms_median"])))
    return rows


def fit_prefill(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    feat = lambda r: np.array([1.0, r["batch"] * r["length"],
                               r["batch"] * r["length"] ** 2 / 2 / 1e6])
    tr = [r for r in ok if r["split"] == "train"]
    ho = [r for r in ok if r["split"] == "holdout"]

    def solve(rs):
        A = np.array([feat(r) / r["ms"] for r in rs])
        return nnls(A, np.ones(len(rs)))[0]

    coef = solve(tr)
    pred = lambda rs, c: np.array([feat(r) @ c for r in rs])
    meas = lambda rs: np.array([r["ms"] for r in rs])
    loo = np.array([feat(r) @ solve(tr[:i] + tr[i + 1:]) for i, r in enumerate(tr)])
    for r in ok:
        r["pred_ms"] = float(feat(r) @ coef)
    return {
        "formula": "T_ms = p0 + pN*N + pA*A, N=batch*len, A=batch*len^2/2/1e6",
        "coefficients": {"p0_ms": coef[0], "pN_ms_per_token": coef[1],
                         "pA_ms_per_mega_attention_pair": coef[2],
                         "linear_tokens_per_s": 1000 / coef[1] if coef[1] > 0 else None},
        "train": metrics(pred(tr, coef), meas(tr)),
        "cv_leave_one_out": metrics(loo, meas(tr)),
        "holdout": metrics(pred(ho, coef), meas(ho)),
        "failed_points": [r for r in rows if r["status"] != "ok"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    man = json.loads(a.manifest.read_text())
    present = {p.stem for p in a.run.glob("*.json")}
    missing = [p["id"] for p in man["points"] if p["id"] not in present]
    kvg = kv_gb_per_token(a.model_config)
    dec, pre = load_decoder(a.run, kvg), load_prefill(a.run)
    # Pre-registered prefill points with no result (the benchmark writes no row when a
    # forward crashes, e.g. HIP illegal access on >=196K-token one-shot prefill).
    got = {(r["split"], r["batch"], r["length"]) for r in pre if r["status"] == "ok"}
    infeasible = [dict(split=p["split"], batch=b, length=l, group=p["id"])
                  for p in man["points"] if p["kind"] == "prefill"
                  for b in p["batches"] for l in p["lengths"] if (p["split"], b, l) not in got]
    model = {"model": man["model"], "gpu": man["gpu"], "attention_backend": man["attention_backend"],
             "run": str(a.run), "kv_gb_per_token": kvg,
             "missing_preregistered_points": missing, "skipped_by_rule": man["skipped"],
             "decoder": fit_decoder(dec), "prefill": fit_prefill(pre),
             "prefill_points_without_result": infeasible}
    a.out.mkdir(parents=True)
    (a.out / "model.json").write_text(json.dumps(model, indent=1, default=float) + "\n")
    with open(a.out / "decoder_points.csv", "w") as f:
        f.write("split,bs,ctx,K_gb,step_ms,step_ms_p90,pred_ms\n")
        for r in dec:
            f.write(f"{r['split']},{r['bs']},{r['ctx']},{r['K']:.4f},{r['step_ms']:.4f},"
                    f"{r['step_ms_p90']:.4f},{r['pred_ms']:.4f}\n")
    with open(a.out / "prefill_points.csv", "w") as f:
        f.write("split,batch,length,status,ms,pred_ms\n")
        for r in pre:
            f.write(f"{r['split']},{r['batch']},{r['length']},{r['status']},{r['ms']:.3f},"
                    f"{r.get('pred_ms', float('nan')):.3f}\n")
    d, p = model["decoder"], model["prefill"]
    print(json.dumps({"missing": len(missing), "prefill_points_without_result": infeasible, "decoder_coef": d["coefficients"],
                      "decoder": {k: d[k] for k in ("train", "cv_checkerboard5", "holdout")},
                      "prefill_coef": p["coefficients"],
                      "prefill": {k: p[k] for k in ("train", "cv_leave_one_out", "holdout")}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
