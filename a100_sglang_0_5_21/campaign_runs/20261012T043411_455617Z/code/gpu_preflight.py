"""Verify the actual allocated A100 devices and CUDA runtime before calibration."""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import time
from pathlib import Path

from calibration_common import atomic_json
from runtime_guard import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    report = {'completed': False, 'started_at': time.time(), 'devices': [],
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'ld_library_path': os.environ.get('LD_LIBRARY_PATH'),
              'hostname': os.uname().nodename}
    try:
        report['runtime'] = check_runtime()
        import torch
        report['nvidia_smi'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip()
        hardware = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit',
            '--format=csv,noheader,nounits'], text=True)
        report['hardware'] = [dict(zip(('index', 'name', 'uuid', 'driver', 'memory_mb',
                                       'max_sm_mhz', 'max_memory_mhz', 'power_limit_w'),
                                      [v.strip() for v in row]))
                              for row in csv.reader(io.StringIO(hardware))]
        report['topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != args.tp:
            raise RuntimeError(f'Need exactly {args.tp} usable CUDA GPUs')
        for device in range(args.tp):
            torch.cuda.set_device(device)
            props = torch.cuda.get_device_properties(device)
            if 'A100' not in props.name or not 39 * 2**30 <= props.total_memory <= 41 * 2**30:
                raise RuntimeError(f'Expected A100 40GB; got {props.name}, {props.total_memory} bytes')
            value = torch.ones((512, 512), dtype=torch.bfloat16, device=f'cuda:{device}')
            output = value @ value
            torch.cuda.synchronize(device)
            if not torch.all(output == 512).item():
                raise RuntimeError(f'BF16 CUDA matmul failed on device {device}')
            report['devices'].append({'rank': device, 'name': props.name,
                'memory_bytes': props.total_memory, 'compute_capability': [props.major, props.minor],
                'bf16_matmul': True})
        report['cuda_driver_libraries'] = sorted({line.split()[-1] for line in
            Path('/proc/self/maps').read_text().splitlines() if 'libcuda.so' in line})
        runtime = report['runtime']
        links = [line.split()[1:args.tp + 1] for line in report['topology'].splitlines()
                 if line.split() and line.split()[0] in {f'GPU{i}' for i in range(args.tp)}
                 and 'X' in line.split()[1:args.tp + 1]]
        report['environment_fingerprint'] = {
            'runtime': {k: runtime[k] for k in ('sglang_distribution_version', 'sglang_source_version',
                                               'torch', 'torch_cuda', 'transformers', 'python', 'image')},
            'devices': sorted([(d['name'], d['memory_bytes'], d['compute_capability']) for d in report['devices']]),
            'hardware': sorted([tuple(d[k] for k in ('name', 'driver', 'memory_mb', 'max_sm_mhz',
                                                     'max_memory_mhz', 'power_limit_w')) for d in report['hardware']]),
            'cuda_driver_libraries': [Path(p).name for p in report['cuda_driver_libraries']],
            'gpu_links': links}
        # JSON normalization makes in-memory tuples equal to their persisted lists.
        import json
        report['environment_fingerprint'] = json.loads(json.dumps(report['environment_fingerprint']))
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if not reference.get('completed') or reference.get('environment_fingerprint') != report['environment_fingerprint']:
                raise RuntimeError('Allocated hardware/runtime differs from the accepted pilot; do not mix timings')
            report['reference_environment_matched'] = str(args.reference)
        report['completed'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['ended_at'] = time.time()
        atomic_json(args.out, report)
    print(f'GPU preflight passed: {args.out}', flush=True)


if __name__ == '__main__':
    main()
