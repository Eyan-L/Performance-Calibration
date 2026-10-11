"""Freeze checked code and submission commands; never submit a PBS job."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration_common import atomic_json
from campaign import prepare

cpu_path, sif_path = map(Path, sys.argv[1:])
cpu, sif = json.loads(cpu_path.read_text()), json.loads(sif_path.read_text())
if not cpu.get('passed') or not sif.get('passed'):
    raise RuntimeError('Pre-submission CPU / SIF checks have not passed')
info = prepare()
if cpu.get('run_signature') != info['run_signature']:
    raise ValueError('CPU checks were run against different measurement code/configuration')
directory = Path(info['campaign_dir'])
for phase in ('pilot', 'full'):
    cmd = ['python3', '-B', str(directory / 'code/run_points.py'), str(directory / 'manifest.json'),
           str(directory / 'results'), '--model-path', info['model_path'], '--dry-run']
    if phase == 'pilot':
        cmd += ['--only', *info['pilot_ids']]
    plan = json.loads(subprocess.check_output(cmd, text=True, cwd=directory / 'code'))
    if plan['run_signature'] != info['run_signature'] or plan['points'] != info[phase + '_groups']:
        raise ValueError('Frozen phase plan / signature mismatch')
    if any('--smoke' in command for command in plan['commands']):
        raise ValueError('Campaign must use formal sampling counts')
    atomic_json(directory / (phase + '_plan.json'), plan)
subprocess.run(['bash', '-n', str(directory / 'code/run_in_sif.sh'),
                str(directory / 'code/nscc_qwen38_pilot.pbs'),
                str(directory / 'code/nscc_qwen38_full.pbs')], check=True)
commands = {phase: shlex.join(['qsub', '-v', 'CALIB_CAMPAIGN_DIR=' + str(directory),
    '-o', str(directory / (phase + '.pbs.log')), str(directory / ('code/nscc_qwen38_' + phase + '.pbs'))])
    for phase in ('pilot', 'full')}
report = {'status': 'ready_not_submitted', 'pbs_submitted': False, 'gpu_acceptance_passed': False,
          'campaign_dir': str(directory), 'run_signature': info['run_signature'],
          'cpu_checks': str(cpu_path), 'sif_cpu_checks': str(sif_path),
          'pilot': {'walltime': '04:00:00', 'groups': 21, 'shapes': 36,
                    'bootstrap_groups': ['dec_train_bs4_ctx8192', 'pre_train_g0'],
                    'bootstrap_shapes': 5, 'remaining_after_bootstrap_shapes': 31,
                    'sampling': 'formal counts; no --smoke'},
          'full': {'walltime': '24:00:00', 'groups': 86, 'shapes': 102,
                   'remaining_if_all_pilot_shapes_pass': 66,
                   'remaining_groups_if_all_pilot_groups_pass': 65,
                   'requires_pilot_acceptance': True},
          'submission_commands_not_executed': commands}
atomic_json(directory / 'pre_submission.json', report)
print(json.dumps(report, indent=2))
