"""Verify the actual allocated A100 devices and CUDA runtime before calibration."""
from __future__ import annotations

import argparse
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
