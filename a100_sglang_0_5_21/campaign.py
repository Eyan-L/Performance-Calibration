"""Freeze a full calibration, run its pilot, then resume the same measurements."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from calibration_common import CONFIG, KIT, ROOT, atomic_json, digest_json, run_specification, shapes
from run_points import execute as execute_process

FILES = ('runtime_config.json', 'runtime_guard.py', 'run_in_sif.sh', 'calibration_common.py',
         'calibrate_point.py', 'run_points.py', 'collect_results.py', 'fit_results.py', 'gpu_preflight.py',
         'measurement_store.py', 'campaign.py', 'verify_campaign.py', 'report_results.py',
         'nscc_qwen38_pilot.pbs', 'nscc_qwen38_full.pbs', 'README.md', 'decode_reuse.py')
MODEL = Path('/scratch/users/ntu/n2602695/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')


def pilot_ids():
    train = [f'dec_train_bs{b}_ctx{length}' for b in (1, 4, 16) for length in (4096, 16384)]
    train += ['dec_train_bs4_ctx8192', 'dec_train_bs4_ctx65536', 'dec_train_bs8_ctx98304',
              'dec_train_bs12_ctx98304', 'dec_train_bs28_ctx32768']
    return train + ['dec_holdout_bs4_ctx16384', 'dec_holdout_bs8_ctx65536',
                    'dec_repeat_bs16_ctx16384'] + [f'pre_train_g{i}' for i in range(5)] + [
                        'pre_holdout_g0', 'pre_holdout_g2']


def prepare(out=None, reuse_decode_from=None):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = (out or ROOT / 'campaign_runs' / stamp).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    (destination / 'code').mkdir(parents=True)
    hashes = {}
    for name in FILES:
        shutil.copyfile(ROOT / name, destination / 'code' / name)
        hashes[name] = hashlib.sha256((destination / 'code' / name).read_bytes()).hexdigest()
    shutil.copyfile(KIT / 'manifests/qwen38bf16_a100x4_flashinfer.json', destination / 'manifest.json')
    manifest = json.loads((destination / 'manifest.json').read_text())
    selected = [p for p in manifest['points'] if p['id'] in pilot_ids()]
    if len(selected) != 21 or sum(len(shapes(p)) for p in selected) != 36:
        raise ValueError('Pilot selection differs from the agreed 21 groups / 36 shapes')
    specification = run_specification(destination / 'manifest.json', MODEL, False)
    specification['implementation_sha256'] = {n: hashes[n] for n in specification['implementation_sha256']}
    image_stat = Path(CONFIG['image']).stat()
    report = {'schema': 'qwen38-calibration-campaign.v1', 'created_at': time.time(),
              'campaign_dir': str(destination), 'model_path': str(MODEL), 'files_sha256': hashes,
              'run_signature': digest_json(specification), 'pilot_ids': [p['id'] for p in selected],
              'measurement_policy': specification['measurement_policy'],
              'pilot_groups': 21, 'pilot_shapes': 36, 'full_groups': len(manifest['points']),
              'full_shapes': sum(len(shapes(p)) for p in manifest['points']),
              'expected_remaining_groups': len(manifest['points']) - 21,
              'expected_remaining_shapes': sum(len(shapes(p)) for p in manifest['points']) - 36,
              'image_stat': {'size': image_stat.st_size, 'mtime_ns': image_stat.st_mtime_ns}}
    if reuse_decode_from:
        from decode_reuse import prepare_reuse
        report['decode_reuse'] = prepare_reuse(reuse_decode_from, destination, specification, report)
        report['expected_remaining_groups'] = report['full_groups'] - report['decode_reuse']['decode_shapes_reused']
        report['expected_remaining_shapes'] = report['full_shapes'] - report['decode_reuse']['decode_shapes_reused']
    atomic_json(destination / 'campaign.json', report)
    (destination / 'jobs').mkdir()
    return report


def load_campaign(directory):
    directory = Path(directory).resolve()
    info = json.loads((directory / 'campaign.json').read_text())
    for name, expected in info['files_sha256'].items():
        if hashlib.sha256((directory / 'code' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen campaign code changed: {name}')
    specification = run_specification(directory / 'manifest.json', Path(info['model_path']), False)
    if digest_json(specification) != info['run_signature']:
        raise ValueError('Frozen manifest/model/corpus/runtime changed; do not reuse measurements')
    stat = Path(CONFIG['image']).stat()
    if info['image_stat'] != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
        raise ValueError('SIF changed after preparation')
    if info.get('decode_reuse'):
        from decode_reuse import validate_catalog
        validate_catalog(directory, info)
    return directory, info


def execute(directory, phase):
    directory, info = load_campaign(directory)
    job = os.environ.get('PBS_JOBID')
    if not job:
        raise RuntimeError('Campaign execution requires a PBS GPU allocation')
    lock = (directory / '.campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job_dir = directory / 'jobs' / job
    job_dir.mkdir(exist_ok=False)
    status = {'job': job, 'phase': phase, 'stage': 'environment', 'completed': False,
              'started_at': time.time(), 'run_signature': info['run_signature']}
    atomic_json(job_dir / 'status.json', status)
    approval = None
    try:
        if phase == 'full':
            source_reuse = info.get('decode_reuse')
            acceptance_file = 'source_pilot_acceptance.json' if source_reuse else 'pilot_acceptance.json'
            approval = json.loads((directory / acceptance_file).read_text())
            expected_signature = source_reuse['source_run_signature'] if source_reuse else info['run_signature']
            if not approval.get('passed') or approval.get('run_signature') != expected_signature:
                raise ValueError('Pilot has not passed acceptance; full grid is not allowed')
        def run(cmd, name, timeout):
            rc = execute_process([str(x) for x in cmd], job_dir / (name + '.log'), timeout)
            print(json.dumps({'stage': name, 'rc': rc, 'log': str(job_dir / (name + '.log'))}), flush=True)
            return rc
        runtime = directory / 'code'
        preflight = ['bash', runtime / 'run_in_sif.sh', '--gpu', runtime / 'gpu_preflight.py',
                     '--out', job_dir / 'gpu_preflight.json', '--tp', '4']
        if approval:
            preflight += ['--reference', approval['gpu_preflight']]
        status['stage'] = 'gpu_preflight'
        atomic_json(job_dir / 'status.json', status)
        if run(preflight, 'gpu_preflight', 600):
            raise RuntimeError('GPU preflight failed')
        status['stage'] = 'independent_forward_validation'
        atomic_json(job_dir / 'status.json', status)
        validation_dir = job_dir / 'forward_validation'
        validation_dir.mkdir()
        for point_id in ('dec_train_bs4_ctx8192', 'pre_train_g0'):
            validation = ['bash', runtime / 'run_in_sif.sh', '--gpu', runtime / 'calibrate_point.py',
                          '--manifest', directory / 'manifest.json', '--point-id', point_id,
                          '--model-path', info['model_path'], '--out', validation_dir / (point_id + '.json'),
                          '--validation-only']
            if run(validation, 'validate_' + point_id, 1800):
                raise RuntimeError('Independent numerical/layout validation failed; no timings started')
        selected = info['pilot_ids'] if phase == 'pilot' else []
        # Same manifest, code and raw directory in both jobs. Only the selected
        # IDs differ; they are deliberately not part of the measurement signature.
        wall = 4 * 3600 if phase == 'pilot' else 24 * 3600
        stop_at = status['started_at'] + wall - 20 * 60
        measure = ['python3', runtime / 'run_points.py', directory / 'manifest.json', directory / 'results',
                   '--model-path', info['model_path'], '--point-timeout', '1800', '--stop-at', str(stop_at),
                   '--completion-out', job_dir / 'completion.json']
        if selected:
            measure += ['--only', *selected]
        status['stage'] = 'measurement'
        atomic_json(job_dir / 'status.json', status)
        if phase == 'pilot' or info.get('decode_reuse'):
            # Detect protocol/instrumentation errors on already proven shapes
            # before spending hours on the expanded grid. These points remain
            # in the shared results and will be skipped in the next call.
            bootstrap = measure[:measure.index('--completion-out')]
            bootstrap_ids = ['dec_train_bs4_ctx8192', 'pre_train_g0']
            if info.get('decode_reuse'):
                # Verify the three formerly failing workload shapes before the
                # remaining grid; their formal samples are reused below.
                bootstrap_ids += ['pre_train_g2', 'pre_train_g3', 'pre_train_g4']
            bootstrap += ['--completion-out', job_dir / 'bootstrap_completion.json', '--only', *bootstrap_ids]
            if run(bootstrap, 'bootstrap_measurement', 7800 if info.get('decode_reuse') else 3900):
                diagnostic = ['python3', runtime / 'collect_results.py', directory / 'results',
                              job_dir / 'bootstrap_collected', '--only', *bootstrap_ids]
                run(diagnostic, 'bootstrap_collection', 300)
                raise RuntimeError('Formal-count bootstrap failed; expanded measurement stopped immediately')
        status['measurement_rc'] = run(measure, 'measurement', wall)
        status['stage'] = 'collection'
        atomic_json(job_dir / 'status.json', status)
        collect = ['python3', runtime / 'collect_results.py', directory / 'results', job_dir / 'collected']
        if selected:
            collect += ['--only', *selected]
        if run(collect, 'collection', 300):
            raise RuntimeError('Collection validation failed')
        status['stage'] = 'fitting'
        atomic_json(job_dir / 'status.json', status)
        audit = json.loads((job_dir / 'collected/coverage.json').read_text())
        fit = ['bash', runtime / 'run_in_sif.sh', runtime / 'fit_results.py', job_dir / 'collected',
               '--out', job_dir / 'fit']
        if not audit['measurement_coverage_complete']:
            fit += ['--allow-incomplete']
        if run(fit, 'fitting', 600):
            raise RuntimeError('Real-data fitting / report generation failed')
        status['stage'] = 'acceptance'
        atomic_json(job_dir / 'status.json', status)
        from verify_campaign import verify
        acceptance = verify(directory, job_dir, phase)
        atomic_json(job_dir / 'acceptance.json', acceptance)
        if not acceptance['passed']:
            raise RuntimeError('Acceptance failed; inspect acceptance.json before any full submission')
        if phase == 'pilot':
            atomic_json(directory / 'pilot_acceptance.json', acceptance)
        else:
            atomic_json(directory / 'final_acceptance.json', acceptance)
        status.update(completed=True, stage='complete', exit_code=0)
    except BaseException as exc:
        status.update(error=f'{type(exc).__name__}: {exc}', exit_code=1)
        raise
    finally:
        status['ended_at'] = time.time()
        atomic_json(job_dir / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--out', type=Path)
    prep.add_argument('--reuse-decode-from', type=Path,
                      help='Reuse accepted unchanged decode only; remeasure every prefill under the new KV cap')
    run = commands.add_parser('run')
    run.add_argument('campaign', type=Path)
    run.add_argument('--phase', choices=('pilot', 'full'), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        info = prepare(args.out, args.reuse_decode_from)
        print(json.dumps(info, indent=2))
    else:
        execute(args.campaign, args.phase)


if __name__ == '__main__':
    main()
