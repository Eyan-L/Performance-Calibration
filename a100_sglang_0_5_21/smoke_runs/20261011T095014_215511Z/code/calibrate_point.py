"""Measure reference calibration points using unmodified SGLang 0.5.21."""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import multiprocessing
import os
import tempfile
import time
from array import array
from multiprocessing.connection import wait
from pathlib import Path

from calibration_common import (CONFIG, ROOT, atomic_json, digest_json, format_result,
                                input_coverage, load_tokens, run_specification, sampling, shapes)
from runtime_guard import check_runtime


def native_api():
    module = importlib.import_module('sglang.benchmark.one_batch')
    required = ('load_model', 'extend', 'prepare_synthetic_inputs_for_latency_test',
                '_maybe_prepare_mlp_sync_batch', 'ServerArgs', 'PortArgs', 'publish',
                'SpawnRanks', 'spawn_world_rank', 'initialize_moe_config',
                'initialize_fp8_gemm_config', 'initialize_fp4_gemm_config')
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f'Official benchmark API missing: {name}')
    return module


def native_arguments(module, manifest, point):
    maximum_batch = max(b for b, _ in shapes(point))
    decode = point['kind'] == 'decoder'
    graph = {'prefill': {'backend': 'disabled'},
             'decode': {'backend': 'full' if decode else 'disabled',
                        'max_bs': maximum_batch, 'bs': [maximum_batch]}}
    args = ['--model-path', manifest['_model_path'], '--tp-size', str(manifest['tp']),
            '--random-seed', '0', '--kv-cache-dtype', 'bfloat16', '--disable-radix-cache',
            '--attention-backend', manifest['attention_backend'],
            '--mem-fraction-static', '.90' if decode else '.70',
            '--max-running-requests', str(maximum_batch),
            '--max-mamba-cache-size', str(maximum_batch),
            '--cuda-graph-config', json.dumps(graph)]
    if decode:
        args.extend(['--max-total-tokens', str(point['bs'] * point['ctx'] + 32768)])
    parser = argparse.ArgumentParser()
    module.ServerArgs.add_cli_args(parser)
    return parser.parse_args(args)


def prepare_decode_inputs(module, wrapped, point):
    import torch
    runner = wrapped.torch_runner
    inputs, identity = load_tokens(point, point['bs'], point['ctx'])
    requests = module.prepare_synthetic_inputs_for_latency_test(point['bs'], point['ctx'], inputs)
    wrapped.clear()
    merged, tokens = None, []
    # Preserve the reference's one-request, 8192-token chunked setup. This
    # preparation is not included in decode timing.
    for request in requests:
        full = array('q', request.full_untruncated_fill_ids)
        result = None
        for start in range(0, len(full), 8192):
            end = min(start + 8192, len(full))
            if start:
                request.prefix_indices = runner.req_to_token_pool.req_to_token[
                    request.kv.req_pool_idx, :start].to(request.prefix_indices.dtype)
            request.origin_input_ids = array('q', full[:end])
            request.full_untruncated_fill_ids = array('q', full[:end])
            request.logprob_start_len = -1
            request.set_extend_range(start, end)
            result = module.extend([request], runner)
        next_ids, _, batch = result
        tokens.append(next_ids)
        if merged is None:
            merged = batch
        else:
            merged.merge_batch(batch)
    merged.input_ids = torch.cat(tokens).to(torch.int64)
    merged.prepare_for_decode()
    module._maybe_prepare_mlp_sync_batch(merged, runner)
    forward = module.ForwardBatch.init_new(merged, runner, return_hidden_states_before_norm=False)
    return forward, identity


def measure_decode(module, wrapped, point, smoke):
    import torch
    import torch.distributed as dist
    forward, identity = prepare_decode_inputs(module, wrapped, point)
    runner = wrapped.torch_runner
    graph = runner.decode_cuda_graph_runner
    if graph is None or not graph.can_run_graph(forward):
        raise RuntimeError('Fixed-context decode batch cannot use the official CUDA graph')
    warmups, iterations, steps = sampling(point, smoke)
    samples = []
    with torch.no_grad():
        for iteration in range(warmups + iterations):
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(steps):
                output = runner.forward(forward)
                if not output.can_run_graph:
                    raise RuntimeError('Decode unexpectedly fell back to eager execution')
            end.record()
            end.synchronize()
            if iteration >= warmups:
                samples.append(float(start.elapsed_time(end)))
    return [{'batch_size': point['bs'], 'request_length': point['ctx'],
             'samples_ms': samples, 'input': identity}]


def measure_prefill(module, wrapped, point, smoke, checkpoint):
    import torch
    import torch.distributed as dist
    runner = wrapped.torch_runner
    original = runner.model.forward
    intervals = []

    @functools.wraps(original)
    def timed_forward(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = original(*args, **kwargs)
        end.record()
        intervals.append((start, end))
        return output

    runner.model.forward = timed_forward
    warmups, iterations, _ = sampling(point, smoke)
    results = []
    try:
        with torch.no_grad():
            for batch_size, length in shapes(point):
                inputs, identity = load_tokens(point, batch_size, length)
                samples = []
                for iteration in range(warmups + iterations):
                    wrapped.clear()
                    requests = module.prepare_synthetic_inputs_for_latency_test(batch_size, length, inputs)
                    intervals.clear()
                    if dist.is_initialized():
                        dist.barrier()
                    torch.cuda.synchronize()
                    # One full batch forward, preserving the reference prefill
                    # definition. Allocation, attention planning, and sampling
                    # occur outside the model.forward CUDA event interval.
                    module.extend(requests, runner)
                    torch.cuda.synchronize()
                    if len(intervals) != 1:
                        raise RuntimeError(f'Expected one model forward, observed {len(intervals)}')
                    if iteration >= warmups:
                        samples.append(float(intervals[0][0].elapsed_time(intervals[0][1])))
                results.append({'batch_size': batch_size, 'request_length': length,
                                'samples_ms': samples, 'input': identity})
                checkpoint(results)
    finally:
        runner.model.forward = original
    return results


def worker(server_args, port_args, manifest, point, smoke, gpu_id, rank, directory):
    path = Path(directory) / f'rank{rank}.json'
    record = {'rank': rank, 'metadata': {}, 'completed': False, 'points': [],
              'stage': 'initialize', 'started_at': time.time()}
    atomic_json(path, record)
    dist, module = None, None

    def checkpoint(points):
        record['points'] = points
        atomic_json(path, record)

    try:
        import torch
        import torch.distributed as dist
        runtime = check_runtime()
        module = native_api()
        torch.cuda.set_device(gpu_id)
        properties = torch.cuda.get_device_properties(gpu_id)
        if manifest.get('gpu') == 'a100_40g' and (
                'A100' not in properties.name or not 39 * 2**30 <= properties.total_memory <= 41 * 2**30):
            raise RuntimeError(f'Expected A100 40GB; got {properties.name}, {properties.total_memory} bytes')
        record['metadata'] = {'runtime': runtime, 'gpu_name': properties.name,
                              'gpu_memory_bytes': properties.total_memory, 'gpu_id': gpu_id}
        module.publish(server_args, role='scheduler', ranks=module.SpawnRanks(
            world_rank=module.spawn_world_rank(server_args, tp_rank=rank, pp_rank=0), gpu_id=gpu_id))
        module.initialize_moe_config()
        module.initialize_fp8_gemm_config()
        module.initialize_fp4_gemm_config()
        module.configure_logger(server_args, prefix=f' CALIB-TP{rank}')
        record['stage'] = 'load_model_and_capture_graph'
        atomic_json(path, record)
        wrapped, _ = module.load_model(server_args, port_args, gpu_id, rank)
        weight_dtype = str(wrapped.torch_runner.model_config.dtype)
        kv_dtype = str(wrapped.torch_runner.kv_cache_dtype)
        if weight_dtype != 'torch.bfloat16' or kv_dtype != 'torch.bfloat16':
            raise RuntimeError(f'Expected BF16 weights and KV; got {weight_dtype}, {kv_dtype}')
        record['metadata'].update({'weight_dtype': weight_dtype, 'kv_cache_dtype': kv_dtype,
            'native_custom_all_reduce_disabled': module.resolving_view(server_args).disable_custom_all_reduce})
        record.update(stage='measure', model_ready_at=time.time())
        atomic_json(path, record)
        if point['kind'] == 'decoder':
            record['points'] = measure_decode(module, wrapped, point, smoke)
        else:
            record['points'] = measure_prefill(module, wrapped, point, smoke, checkpoint)
        if dist.is_initialized():
            dist.barrier()
        record.update(completed=True, stage='complete', ended_at=time.time())
        atomic_json(path, record)
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        record['ended_at'] = time.time()
        atomic_json(path, record)
        raise
    finally:
        if dist is not None and module is not None and dist.is_initialized():
            module.destroy_model_parallel()
            module.destroy_distributed_environment()


def launch(server_args, port_args, manifest, point, smoke, directory):
    context = multiprocessing.get_context('spawn')
    processes = []
    try:
        for rank in range(manifest['tp']):
            process = context.Process(target=worker, args=(server_args, port_args, manifest,
                                      point, smoke, rank, rank, str(directory)))
            process.start()
            processes.append(process)
        pending = set(processes)
        while pending:
            ready = wait([p.sentinel for p in pending])
            for process in list(pending):
                if process.sentinel in ready:
                    process.join()
                    pending.remove(process)
                    if process.exitcode != 0:
                        raise RuntimeError(f'TP worker {process.pid} exited {process.exitcode}')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--point-id')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validate-inputs', action='store_true')
    parser.add_argument('--check-api', action='store_true')
    args = parser.parse_args()
    if args.check_api:
        module = native_api()
        # Exercise the official argument parser without resolving GPU settings.
        point = {'kind': 'decoder', 'bs': 1, 'ctx': 4096}
        native_arguments(module, {'tp': 4, 'attention_backend': 'flashinfer', '_model_path': '/unused'}, point)
        print(json.dumps({'native_api': module.__file__, 'tp4_arguments_parse': True}))
        return
    if args.manifest is None:
        parser.error('--manifest is required')
    manifest = json.loads(args.manifest.read_text())
    if args.validate_inputs:
        failures = input_coverage(manifest)
        print(json.dumps({'points': len(manifest['points']), 'input_failures': failures}))
        if failures:
            raise SystemExit(1)
        return
    if not args.point_id or args.model_path is None or args.out is None:
        parser.error('--point-id, --model-path and --out are required for GPU measurement')
    if args.out.exists():
        raise FileExistsError(args.out)
    point = next(p for p in manifest['points'] if p['id'] == args.point_id)
    manifest['_model_path'] = str(args.model_path.resolve())
    runtime = check_runtime()
    specification = run_specification(args.manifest, args.model_path, args.smoke)
    parent = args.out.parent / '.attempts' / point['id']
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
    metadata = {'runtime': runtime, 'tp': manifest['tp'], 'point': point,
                'run_signature': digest_json(specification),
                'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'model_path': manifest['_model_path'], 'attention_backend': manifest['attention_backend'],
                'smoke': args.smoke, 'rank_artifacts': str(directory), 'job': os.environ.get('CALIB_JOB_ID'),
                'gpu_execution_tested': True,
                'timing_definition': 'model.forward CUDA events (prefill); fixed-context graph-backed ModelRunner.forward loop CUDA events (decode)',
                'inference_source': 'unmodified official SGLang 0.5.21'}
    atomic_json(directory / 'request.json', {'point': point, 'manifest': str(args.manifest.resolve()),
                'model_path': manifest['_model_path'], 'runtime': runtime, 'smoke': args.smoke,
                'metadata': {**metadata, 'gpu_execution_tested': False}, 'started_at': time.time()})
    stage = 'resolve_gpu_arguments'
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != manifest['tp']:
            raise RuntimeError(f"Need exactly {manifest['tp']} visible CUDA GPUs")
        module = native_api()
        server_args = module.ServerArgs.from_cli_args(native_arguments(module, manifest, point))
        server_args.resolve_once()
        module._set_envs_and_config(server_args)
        port_args = module.PortArgs.init_new(server_args)
        stage = 'tp_workers'
        launch(server_args, port_args, manifest, point, args.smoke, directory)
        ranks = [json.loads((directory / f'rank{rank}.json').read_text()) for rank in range(manifest['tp'])]
        if not all(r['completed'] for r in ranks):
            raise RuntimeError('Not all TP workers finished')
        stage = 'publish_result'
        result = format_result(point, ranks, metadata)
        atomic_json(args.out, result)
        atomic_json(directory / 'status.json', {'completed': True, 'stage': 'complete', 'ended_at': time.time()})
    except BaseException as exc:
        atomic_json(directory / 'status.json', {'completed': False, 'stage': stage,
                    'error': f'{type(exc).__name__}: {exc}', 'ended_at': time.time()})
        raise
    print(json.dumps({'completed': point['id'], 'result': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
