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

from calibration_common import (CONFIG, ROOT, MEASUREMENT_POLICY, atomic_json, digest_json,
                                input_coverage, load_tokens, run_specification, sampling, shapes)
from runtime_guard import check_runtime
from measurement_store import assemble_group, read_group


def check_logits(logits, batch, length, graph):
    import torch
    valid_shape = logits is not None and logits.ndim == 2 and logits.shape[0] == batch
    finite = valid_shape and bool(torch.isfinite(logits).all().item())
    if not valid_shape or not finite:
        raise RuntimeError('Model forward returned an unexpected shape or nonfinite logits')
    return {'shape_matches': True, 'finite_logits': True, 'logits_shape': list(logits.shape),
            'batch': batch, 'context': length, 'graph': graph,
            'check_scope': 'dedicated validation process; no calibration timing samples'}


def batch_layout(forward):
    """Read cached CPU metadata only; never issue a GPU copy in measurement."""
    lengths = getattr(forward, 'seq_lens_cpu', None)
    values = None if lengths is None else (lengths.tolist() if hasattr(lengths, 'tolist') else list(lengths))
    return {'batch_size': forward.batch_size, 'actual_sequence_lengths': values,
            'actual_input_token_count': forward.input_ids.numel()}


def check_batch(forward, batch, length, decode):
    lengths = getattr(forward, 'seq_lens_cpu', None)
    if lengths is None:
        lengths = forward.seq_lens.detach().cpu()
    values = lengths.tolist() if hasattr(lengths, 'tolist') else list(lengths)
    expected_length = length + 1 if decode else length
    expected_tokens = batch if decode else batch * length
    if forward.batch_size != batch or forward.input_ids.numel() != expected_tokens or values != [expected_length] * batch:
        raise RuntimeError(f'Actual forward layout differs from requested batch/context: {forward.batch_size}, {values}')
    return {'batch_size': forward.batch_size, 'actual_sequence_lengths': values,
            'actual_input_token_count': forward.input_ids.numel(),
            'layout_verified': True}


def communication_state(server_args, numerical=False):
    import torch
    from sglang.srt.distributed.parallel_state import get_tp_group
    group = get_tp_group()
    result = {'requested_custom_all_reduce_disabled': bool(server_args.disable_custom_all_reduce),
              'available_communicators': {}}
    for name in ('ca_comm', 'pynccl_comm', 'torch_symm_mem_comm', 'pymscclpp_comm',
                 'pcie_ipc_comm', 'qr_comm'):
        communicator = getattr(group, name, None)
        result['available_communicators'][name] = {
            'present': communicator is not None,
            'disabled': bool(getattr(communicator, 'disabled', False)) if communicator is not None else None}
    # Only validation processes run this probe. Measurements merely record
    # available communicators; this does not claim to profile kernel dispatch.
    if numerical:
        value = torch.full((32,), float(group.rank + 1), device='cuda')
        reduced = group.all_reduce(value)
        expected = group.world_size * (group.world_size + 1) / 2
        torch.cuda.synchronize()
        result['all_reduce_numeric_check'] = bool(torch.all(reduced == expected).item())
        if not result['all_reduce_numeric_check']:
            raise RuntimeError('TP all-reduce numerical check failed')
    return result


def native_api():
    module = importlib.import_module('sglang.benchmark.one_batch')
    required = ('load_model', 'extend', 'prepare_synthetic_inputs_for_latency_test',
                '_maybe_prepare_mlp_sync_batch', 'ServerArgs', 'PortArgs', 'publish',
                'SpawnRanks', 'spawn_world_rank', 'initialize_moe_config',
                'initialize_fp8_gemm_config', 'initialize_fp4_gemm_config',
                'ForwardBatch', 'resolving_view', 'configure_logger', '_set_envs_and_config',
                'destroy_model_parallel', 'destroy_distributed_environment')
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
    forward, identity = prepare_decode_inputs(module, wrapped, point)
    runner = wrapped.torch_runner
    graph = runner.decode_cuda_graph_runner
    if graph is None or not graph.can_run_graph(forward):
        raise RuntimeError('Fixed-context decode batch cannot use the official CUDA graph')
    warmups, iterations, steps = sampling(point, smoke)
    layout = batch_layout(forward)
    samples = []
    with torch.no_grad():
        for iteration in range(warmups + iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(steps):
                output = runner.forward(forward)
                if not output.can_run_graph:
                    raise RuntimeError('Decode unexpectedly fell back to eager execution')
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            torch.cuda.synchronize()
            if iteration >= warmups:
                samples.append(elapsed)
    return [{'batch_size': point['bs'], 'request_length': point['ctx'],
             'samples_ms': samples, 'input': identity, 'layout': layout}]


def measure_prefill(module, wrapped, point, smoke, checkpoint, requested_shapes, activate):
    import torch
    runner = wrapped.torch_runner
    original = runner.model.forward
    intervals = []
    layout = None

    @functools.wraps(original)
    def timed_forward(*args, **kwargs):
        nonlocal layout
        if layout is None:
            forward = args[2] if len(args) >= 3 else kwargs['forward_batch']
            layout = batch_layout(forward)
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
            for batch_size, length in requested_shapes:
                layout = None
                activate(batch_size, length)
                inputs, identity = load_tokens(point, batch_size, length)
                samples = []
                for iteration in range(warmups + iterations):
                    wrapped.clear()
                    requests = module.prepare_synthetic_inputs_for_latency_test(batch_size, length, inputs)
                    intervals.clear()
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
                                'samples_ms': samples, 'input': identity, 'layout': layout})
                checkpoint(results)
    finally:
        runner.model.forward = original
    return results


def validate_forward(module, wrapped, point, requested_shapes, checkpoint, activate):
    """Check real logits/layout in a separate process, never fit these forwards."""
    import torch
    runner = wrapped.torch_runner
    if point['kind'] == 'decoder':
        activate(point['bs'], point['ctx'])
        forward, identity = prepare_decode_inputs(module, wrapped, point)
        graph = runner.decode_cuda_graph_runner
        if graph is None or not graph.can_run_graph(forward):
            raise RuntimeError('Validation batch cannot use the native decode graph')
        layout = check_batch(forward, point['bs'], point['ctx'], True)
        with torch.no_grad():
            output = runner.forward(forward)
            torch.cuda.synchronize()
        if not output.can_run_graph:
            raise RuntimeError('Validation decode fell back to eager')
        checks = check_logits(output.logits_output.next_token_logits, point['bs'], point['ctx'], True)
        return [{'batch_size': point['bs'], 'request_length': point['ctx'], 'input': identity,
                 'validation': {**checks, **layout}}]
    original, layout = runner.model.forward, None

    @functools.wraps(original)
    def observed_forward(*args, **kwargs):
        nonlocal layout
        forward = args[2] if len(args) >= 3 else kwargs['forward_batch']
        layout = check_batch(forward, batch_size, length, False)
        return original(*args, **kwargs)

    runner.model.forward = observed_forward
    results = []
    try:
        with torch.no_grad():
            for batch_size, length in requested_shapes:
                activate(batch_size, length)
                inputs, identity = load_tokens(point, batch_size, length)
                wrapped.clear()
                requests = module.prepare_synthetic_inputs_for_latency_test(batch_size, length, inputs)
                _, logits, _ = module.extend(requests, runner)
                torch.cuda.synchronize()
                checks = check_logits(logits, batch_size, length, False)
                if layout is None:
                    raise RuntimeError('Validation did not observe a prefill forward')
                results.append({'batch_size': batch_size, 'request_length': length, 'input': identity,
                                'validation': {**checks, **layout}})
                checkpoint(results)
    finally:
        runner.model.forward = original
    return results


def worker(server_args, port_args, manifest, point, smoke, gpu_id, rank, directory, requested_shapes, validation_only):
    path = Path(directory) / f'rank{rank}.json'
    record = {'rank': rank, 'metadata': {}, 'completed': False, 'points': [],
              'stage': 'initialize', 'started_at': time.time()}
    atomic_json(path, record)
    dist, module = None, None

    def checkpoint(points):
        record['points'] = points
        atomic_json(path, record)

    def activate(batch, length):
        record['active_shape'] = [batch, length]
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
                              'purpose': 'forward_validation' if validation_only else 'measurement',
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
            'native_custom_all_reduce_disabled': module.resolving_view(server_args).disable_custom_all_reduce,
            'communication': communication_state(module.resolving_view(server_args), numerical=validation_only)})
        record.update(stage='measure', model_ready_at=time.time())
        atomic_json(path, record)
        if validation_only:
            record['points'] = validate_forward(module, wrapped, point, requested_shapes, checkpoint, activate)
        elif point['kind'] == 'decoder':
            activate(point['bs'], point['ctx'])
            record['points'] = measure_decode(module, wrapped, point, smoke)
        else:
            record['points'] = measure_prefill(module, wrapped, point, smoke, checkpoint,
                                             requested_shapes, activate)
        if dist.is_initialized():
            dist.barrier()
        record.update(completed=True, stage='complete', ended_at=time.time())
        atomic_json(path, record)
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        record['error_kind'] = 'out_of_memory' if type(exc).__name__ == 'OutOfMemoryError' else 'execution_error'
        record['ended_at'] = time.time()
        atomic_json(path, record)
        raise
    finally:
        if dist is not None and module is not None and dist.is_initialized():
            module.destroy_model_parallel()
            module.destroy_distributed_environment()


def launch(server_args, port_args, manifest, point, smoke, directory, requested_shapes, validation_only):
    context = multiprocessing.get_context('spawn')
    processes = []
    try:
        for rank in range(manifest['tp']):
            process = context.Process(target=worker, args=(server_args, port_args, manifest,
                                      point, smoke, rank, rank, str(directory), requested_shapes, validation_only))
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
    parser.add_argument('--validation-only', action='store_true',
                        help='Separate numerical/layout validation; produces no timing samples')
    parser.add_argument('--only-shapes', nargs='+', help='Prefill shapes to measure, as BATCH:LENGTH')
    parser.add_argument('--validate-inputs', action='store_true')
    parser.add_argument('--check-api', action='store_true')
    args = parser.parse_args()
    if args.check_api:
        module = native_api()
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        if not callable(get_tp_group) or not callable(getattr(GroupCoordinator, 'all_reduce', None)):
            raise RuntimeError('Native TP communication inspection API is missing')
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
    requested_shapes = shapes(point)
    if args.only_shapes:
        if point['kind'] != 'prefill':
            parser.error('--only-shapes is for partial prefill resume only')
        requested = [tuple(map(int, value.split(':'))) for value in args.only_shapes]
        if len(set(requested)) != len(requested) or not set(requested) <= set(requested_shapes):
            parser.error('Unknown or duplicate prefill shape')
        requested_shapes = [s for s in requested_shapes if s in requested]
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
                'purpose': 'forward_validation' if args.validation_only else 'measurement',
                'measurement_policy': MEASUREMENT_POLICY,
                'gpu_execution_tested': True,
                'timing_definition': 'model.forward CUDA events (prefill); fixed-context graph-backed ModelRunner.forward loop CUDA events (decode)',
                'inference_source': 'unmodified official SGLang 0.5.21'}
    atomic_json(directory / 'request.json', {'point': point, 'manifest': str(args.manifest.resolve()),
                'requested_shapes': requested_shapes,
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
        launch(server_args, port_args, manifest, point, args.smoke, directory, requested_shapes, args.validation_only)
        ranks = [json.loads((directory / f'rank{rank}.json').read_text()) for rank in range(manifest['tp'])]
        if not all(r['completed'] for r in ranks):
            raise RuntimeError('Not all TP workers finished')
        stage = 'publish_result'
        atomic_json(directory / 'status.json', {'completed': True, 'stage': 'complete', 'ended_at': time.time()})
        if args.validation_only:
            atomic_json(args.out, {'schema': 'native-forward-validation.v1', 'completed': True,
                        'metadata': metadata, 'ranks': ranks})
            print(json.dumps({'validated': point['id'], 'result': str(args.out)}), flush=True)
            return
        recovered, _, invalid = read_group(args.out.parent, point, metadata['run_signature'], manifest['tp'], args.smoke)
        result = assemble_group(point, recovered)
        if result is None:
            raise RuntimeError(f'Missing shapes after successful workers; invalid records: {invalid}')
        atomic_json(args.out, result)
    except BaseException as exc:
        atomic_json(directory / 'status.json', {'completed': False, 'stage': stage,
                    'error': f'{type(exc).__name__}: {exc}', 'ended_at': time.time()})
        raise
    print(json.dumps({'completed': point['id'], 'result': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
