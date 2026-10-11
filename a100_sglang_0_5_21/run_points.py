"""Run the unchanged reference manifest through the official SGLang adapter.

Execute this driver with NSCC host Python. Each point runs in a separate SIF
process; run a smoke separately from the full grid. Failed attempts and logs
are retained. Resume uses complete matching samples from all ranks of the same
attempt. Prefill retries measure only missing shapes, with original resources.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path

from calibration_common import (ROOT, atomic_json, digest_json, input_coverage,
                                run_specification, validate_result)
from measurement_store import assemble_group, missing_shapes, read_group


def complete_result(path, point, signature, tp, smoke):
    if not path.is_file():
        return False
    recovered, canonical, invalid = read_group(path.parent, point, signature, tp, smoke)
    return canonical is not None and not invalid and not missing_shapes(point, recovered)


def command(args, point, result, pending_shapes=None):
    cmd = ['bash', str(ROOT / 'run_in_sif.sh'), '--gpu', str(ROOT / 'calibrate_point.py'),
           '--manifest', str(args.manifest.resolve()), '--point-id', point['id'],
           '--model-path', str(args.model_path.resolve()), '--out', str(result)]
    if args.smoke:
        cmd.append('--smoke')
    if point['kind'] == 'prefill' and pending_shapes is not None:
        cmd.extend(['--only-shapes', *[f'{b}:{length}' for b, length in pending_shapes]])
    return cmd


def execute(cmd, log, timeout):
    with log.open('x') as stream:
        stream.write('CMD ' + shlex.join(cmd) + '\n')
        stream.flush()
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def failure_kind(run, point, started, rc):
    if rc == 124:
        return 'timeout'
    for path in (run / '.attempts' / point['id']).glob('attempt-*/request.json'):
        request = json.loads(path.read_text())
        if request.get('started_at', 0) < started - 1:
            continue
        if any(json.loads(p.read_text()).get('error_kind') == 'out_of_memory'
               for p in path.parent.glob('rank*.json')):
            return 'out_of_memory'
    return 'execution_or_result_error'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--model-path', type=Path, default=os.environ.get('MODEL_PATH'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--stop-at', type=float, default=float('inf'))
    parser.add_argument('--point-timeout', type=int, default=1800)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--completion-out', type=Path)
    args = parser.parse_args()
    if args.model_path is None:
        parser.error('Provide --model-path or MODEL_PATH')
    args.out = args.out.resolve()
    manifest = json.loads(args.manifest.read_text())
    if input_coverage(manifest):
        raise ValueError('Manifest requests more eligible corpus prompts than available')
    if args.only and not set(args.only) <= {p['id'] for p in manifest['points']}:
        parser.error('--only includes a point not in the manifest')
    selected = [p for p in manifest['points'] if not args.only or p['id'] in args.only]
    specification = run_specification(args.manifest, args.model_path, args.smoke)
    signature = digest_json(specification)
    if args.dry_run:
        print(json.dumps({'run_signature': signature, 'points': len(selected), 'tp': manifest['tp'],
                          'commands': [command(args, p, args.out / (p['id'] + '.json')) for p in selected]}, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    run_lock = (args.out / '.driver.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another driver is already using this run directory')
    frozen = args.out / 'run_config.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != specification:
            raise ValueError('Runtime/model/manifest/code changed; use a new run directory')
    else:
        if any(args.out.glob('dec_*.json')) or any(args.out.glob('pre_*.json')):
            raise ValueError('Existing results have no native run provenance; use a new directory')
        atomic_json(frozen, specification)
    atomic_json(args.out / 'manifest.json', manifest)
    (args.out / 'logs').mkdir(exist_ok=True)
    failed = 0
    for point in selected:
        result = args.out / (point['id'] + '.json')
        recovered, canonical, invalid = read_group(args.out, point, signature, manifest['tp'], args.smoke)
        pending_shapes = missing_shapes(point, recovered)
        if not pending_shapes:
            if canonical is None:
                atomic_json(result, assemble_group(point, recovered))
            print(json.dumps({'resume': point['id']}), flush=True)
            continue
        if result.exists():
            raise ValueError(f'Incomplete canonical result: {result}; preserve it and choose a new directory')
        if time.time() + args.point_timeout + 30 > args.stop_at:
            print(json.dumps({'stop': 'deadline', 'next': point['id']}), flush=True)
            break
        log = args.out / 'logs' / (point['id'] + '-' + uuid.uuid4().hex[:12] + '.log')
        started = time.time()
        error = None
        try:
            rc = execute(command(args, point, result, pending_shapes), log, args.point_timeout)
        except Exception as exc:
            rc, error = 1, f'{type(exc).__name__}: {exc}'
        try:
            good = rc == 0 and complete_result(result, point, signature, manifest['tp'], args.smoke)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            good, error = False, f'Invalid result: {exc}'
        record = {'id': point['id'], 'rc': rc, 'result': good, 'seconds': round(time.time() - started, 1),
                  'log': str(log), 'job': os.environ.get('PBS_JOBID') or os.environ.get('SLURM_JOB_ID'),
                  'run_signature': signature, 'requested_shapes': pending_shapes,
                  'reused_shapes': [{'batch': b, 'length': length, 'source': r['source']}
                                    for (b, length), r in recovered.items()]}
        if error:
            record['error'] = error
        if not good:
            record['failure_kind'] = failure_kind(args.out, point, started, rc)
        with (args.out / 'status.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        failed += not good
        if not good and record['failure_kind'] != 'out_of_memory':
            print(json.dumps({'stop': 'unexpected_failure', 'id': point['id']}), flush=True)
            break
    remaining = []
    for point in selected:
        try:
            complete = complete_result(args.out / (point['id'] + '.json'), point,
                                       signature, manifest['tp'], args.smoke)
        except (ValueError, KeyError, IndexError, TypeError):
            complete = False
        if not complete:
            remaining.append(point['id'])
    summary = {'completed': not remaining, 'selected': [p['id'] for p in selected],
               'remaining': remaining, 'failed_attempts': failed, 'run_signature': signature,
               'job': os.environ.get('PBS_JOBID') or os.environ.get('SLURM_JOB_ID')}
    atomic_json(args.out / 'completion.json', summary)
    if args.completion_out:
        atomic_json(args.completion_out, summary)
    print(json.dumps(summary), flush=True)
    if remaining:
        raise SystemExit(1 if failed else 2)


if __name__ == '__main__':
    main()
