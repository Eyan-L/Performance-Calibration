#!/usr/bin/env python3
"""Run a frozen Highway benchmark script on ROCm with minimal, logged shims.

Usage: python ash_bench_shim.py <benchmark.py> [benchmark args...]

Shims (nothing else is changed):
1. Dense Qwen3.5-family checkpoints (model_type qwen3_5_text, e.g. Qwen3.8-27B)
   are rejected by Qwen35MoePerfEstimator.  The benchmarks need *some*
   estimator to emit per-forward DeviceTimer records, so for dense models the
   estimator factory falls back to DenseQwen3PerfEstimator.  Its FLOP/byte
   fields ignore GDN linear layers and FP8 weights and are NOT used for fitting;
   they are recomputed offline.  Measured latency is unaffected.
2. MoE expert-read tracking is forced off (the Highway patched CUDA
   moe_align op does not exist on ROCm).
3. The per-layer in-graph decode timer is disabled (see below).
"""

from __future__ import annotations

import os
import runpy
import sys

os.environ["SGLANG_MOE_EXPERT_READ_TRACKING_MODE"] = "off"

from sglang.srt.observability import model_perf_estimator as mpe  # noqa: E402

_orig = mpe.Qwen35MoePerfEstimator.from_model_config.__func__


def _from_model_config(cls, model_config, tp_size):
    hf = getattr(model_config, "hf_text_config", None) or model_config.hf_config
    if getattr(hf, "model_type", None) == "qwen3_5_text":
        # Same fields as DenseQwen3PerfEstimator.from_model_config, minus its
        # model_type check (it only accepts plain qwen3).  Timing only.
        print("ASH shim: dense qwen3_5_text -> DenseQwen3PerfEstimator (timing only)", flush=True)
        return mpe.DenseQwen3PerfEstimator(
            hidden_size=int(hf.hidden_size),
            intermediate_size=int(hf.intermediate_size),
            num_layers=int(hf.num_hidden_layers),
            tp_size=tp_size,
            num_q_heads_per_gpu=int(model_config.get_num_attention_heads(tp_size)),
            num_kv_heads_per_gpu=int(model_config.get_num_kv_heads(tp_size)),
            head_dim=int(getattr(hf, "head_dim", 0) or model_config.head_dim),
            vocab_size_per_gpu=int(hf.vocab_size) // tp_size,
            weight_bytes=2,
            activation_bytes=2,
            kv_cache_bytes=2,
            tie_word_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
        )
    return _orig(cls, model_config, tp_size)


mpe.Qwen35MoePerfEstimator.from_model_config = classmethod(_from_model_config)

# 3. ROCm forbids external events inside HIP graph capture, which the Highway
#    per-layer in-graph timer uses.  Drop that diagnostic timer; the step
#    latency the fit uses is timed by events outside the graph replay.
from sglang.srt.model_executor import cuda_graph_runner as cgr  # noqa: E402

_orig_cgr_init = cgr.CudaGraphRunner.__init__


def _cgr_init(self, *args, **kwargs):
    if kwargs.get("layer_timer_name"):
        print(f"ASH shim: layer_timer_name={kwargs['layer_timer_name']!r} -> None (ROCm)", flush=True)
        kwargs["layer_timer_name"] = None
    _orig_cgr_init(self, *args, **kwargs)


cgr.CudaGraphRunner.__init__ = _cgr_init

script = sys.argv[1]
sys.argv = sys.argv[1:]
print(f"ASH shim: running {script}", flush=True)
runpy.run_path(script, run_name="__main__")
