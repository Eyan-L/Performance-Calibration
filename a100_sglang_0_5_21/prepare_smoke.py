"""Freeze the calibration code before submitting a PBS smoke."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

from calibration_common import ROOT, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    launch = (args.out or ROOT / 'smoke_runs' / stamp).resolve()
    if launch.exists():
        raise FileExistsError(launch)
    (launch / 'code').mkdir(parents=True)
    files = ('runtime_config.json', 'runtime_guard.py', 'run_in_sif.sh',
             'calibration_common.py', 'calibrate_point.py', 'run_points.py',
             'collect_results.py', 'fit_results.py', 'gpu_preflight.py',
             'nscc_qwen38_smoke.pbs', 'README.md')
    hashes = {}
    for name in files:
        shutil.copyfile(ROOT / name, launch / 'code' / name)
        hashes[name] = hashlib.sha256((launch / 'code' / name).read_bytes()).hexdigest()
    report = {'launch_dir': str(launch), 'files_sha256': hashes,
              'pbs_script': str(launch / 'code' / 'nscc_qwen38_smoke.pbs'),
              'points': ['dec_train_bs4_ctx8192', 'pre_train_g0']}
    atomic_json(launch / 'launch.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}), flush=True)


if __name__ == '__main__':
    main()
