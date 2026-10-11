#!/usr/bin/env python3
"""Pre-register the measurement points (frozen before submission).

Decoder grid = H200 section-writer grid (B 1..28 x ctx 4K..24K, identical token
selection: same corpus, split, seed) + long contexts 32K/64K/96K drawn from the
derived long corpus.  Holdout = H200 holdout indexes on the holdout split + 3
long-context holdout points.  Prefill = H200 full-GPU batch x length grid + 64K/96K.

Feasibility rule (decided before any measurement): skip a decoder point when
its full-attention KV, B * (ctx + 64) * kv_bytes_per_token, exceeds 75% of
(HBM - checkpoint bytes).  Skipped points are written to the manifest, never
silently dropped.  Prefill forwards are capped at 262144 tokens (H200 rule).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BATCH = [1, 2, 4, 8, 12, 16, 20, 24, 28]
CTX_SHORT = [4096, 8192, 12288, 16384, 24576]
CTX_LONG = [32768, 65536, 98304]
H200_HOLDOUT_INDEXES = [1, 13, 19, 21, 28, 34, 37, 41]
LONG_HOLDOUT = [(2, 32768), (8, 65536), (16, 98304)]
REPEATS = [(4, 8192), (16, 16384), (24, 24576)]  # stability re-measurement

# kv bytes/token = full-attn layers * kv heads * head_dim * 2 (K,V) * 2 (bf16 KV)
MODELS = {
    "qwen36fp8": {"kv_bytes_per_token": 10 * 2 * 256 * 2 * 2, "ckpt_gb": 35.0},
    "qwen38fp8": {"kv_bytes_per_token": 16 * 4 * 256 * 2 * 2, "ckpt_gb": 29.0},
    "qwen36bf16": {"kv_bytes_per_token": 10 * 2 * 256 * 2 * 2, "ckpt_gb": 72.0},
    "qwen38bf16": {"kv_bytes_per_token": 16 * 4 * 256 * 2 * 2, "ckpt_gb": 56.0},
}
HBM_GB = {"mi300x": 192.0, "mi325x": 256.0, "h200": 141.0}
PREFILL_GROUPS = [  # (corpus, batches, lengths): one process each
    ("short", [1], [4096, 8192, 16384, 32768]),
    ("short", [4], [4096, 8192, 16384, 32768]),
    ("short", [8], [4096, 8192, 16384, 32768]),
    ("long", [1, 4], [65536]),
    ("long", [1, 2], [98304]),
]
PREFILL_HOLDOUT = [  # H200 validation grid (batch 1,8 x 8K,32K) + long groups
    ("short", [1, 8], [8192, 32768]),
    ("long", [1, 4], [65536]),
    ("long", [1, 2], [98304]),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--gpu", choices=HBM_GB, required=True)
    p.add_argument("--attn", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    m = MODELS[args.model]
    budget = 0.75 * (HBM_GB[args.gpu] - m["ckpt_gb"]) * 1e9

    points, skipped = [], []

    def decoder(tag, bs, ctx, split, warm=3, iters=9, steps=30):
        corpus = "short" if ctx <= 24576 else "long"
        kv = bs * (ctx + 64) * m["kv_bytes_per_token"]
        row = dict(id=f"dec_{tag}_bs{bs}_ctx{ctx}", kind="decoder", bs=bs, ctx=ctx,
                   split=split, corpus=corpus, warmups=warm, iters=iters, steps=steps,
                   kv_gb=round(kv / 1e9, 2))
        (points if kv <= budget else skipped).append(row)

    for bs in BATCH:
        for ctx in CTX_SHORT + CTX_LONG:
            decoder("train", bs, ctx, "train")
    for idx in H200_HOLDOUT_INDEXES:
        decoder("holdout", BATCH[idx // 5], CTX_SHORT[idx % 5], "holdout")
    for bs, ctx in LONG_HOLDOUT:
        decoder("holdout", bs, ctx, "holdout")
    for split, groups in (("train", PREFILL_GROUPS), ("holdout", PREFILL_HOLDOUT)):
        for g, (corpus, batches, lengths) in enumerate(groups):
            points.append(dict(id=f"pre_{split}_g{g}", kind="prefill", corpus=corpus,
                               split=split, batches=batches, lengths=lengths,
                               warmups=2, iters=5))
    for bs, ctx in REPEATS:  # appended last: measured after the grid, other GPU slot
        decoder("repeat", bs, ctx, "train")

    manifest = dict(model=args.model, gpu=args.gpu, attention_backend=args.attn,
                    kv_budget_gb=round(budget / 1e9, 1), points=points, skipped=skipped)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    with open(args.out.with_suffix(".tsv"), "w") as f:
        for r in points:
            f.write(json.dumps(r) + "\n")
    print(f"{args.model}/{args.gpu}: {len(points)} points, {len(skipped)} skipped "
          f"{[s['id'] for s in skipped]}")


if __name__ == "__main__":
    main()
