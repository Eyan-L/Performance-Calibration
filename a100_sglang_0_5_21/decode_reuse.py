"""Reuse unchanged decode measurements with their original signatures and evidence.

This migration only permits the prefill KV-cap change. It never imports prefill,
rewrites source records, or treats old measurements as executions of new code.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from calibration_common import ROOT, atomic_json, digest_json, shapes


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pins(pins):
    for name, expected in pins.items():
        if file_sha(name) != expected:
            raise ValueError(f'Reused decode source changed: {name}')


def check_decode_compatibility(source, specification):
    original = json.loads((source / 'results/run_config.json').read_text())
    excluded = {'implementation_sha256', 'kv_pool_policy'}
    if ({k: v for k, v in original.items() if k not in excluded} !=
            {k: v for k, v in specification.items() if k not in excluded}):
        raise ValueError('Model/input/runtime/timing policy differs; decode cannot be reused')
    for name in ('runtime_guard.py', 'run_in_sif.sh', 'gpu_preflight.py'):
        if original['implementation_sha256'][name] != specification['implementation_sha256'][name]:
            raise ValueError(f'Decode runtime changed: {name}')
    # All model execution, input preparation, timing and formatting functions
    # must be identical. Only argument construction and signature bookkeeping
    # may differ, and decode arguments are checked against the official parser.
    for name, allowed in (('calibrate_point.py', {'native_arguments'}),
                          ('calibration_common.py', {'run_specification'})):
        def nodes(path):
            return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name not in allowed]
        if nodes(source / 'code' / name) != nodes(ROOT / name):
            raise ValueError(f'Decode computation/timing changed: {name}')
    old_args = next(n for n in ast.parse((source / 'code/calibrate_point.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'native_arguments')
    new_args = next(n for n in ast.parse((ROOT / 'calibrate_point.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'native_arguments')
    # Check exact argument-list equality for every decode point without GPU
    # initialization. This also validates the per-group resource settings.
    class ArgumentModule:
        class ServerArgs:
            @staticmethod
            def add_cli_args(parser):
                for flag in ('model-path', 'tp-size', 'random-seed', 'kv-cache-dtype',
                             'attention-backend', 'mem-fraction-static', 'max-running-requests',
                             'max-mamba-cache-size', 'cuda-graph-config', 'max-total-tokens'):
                    parser.add_argument('--' + flag)
                parser.add_argument('--disable-radix-cache', action='store_true')
                parser.add_argument('--disable-custom-all-reduce', action='store_true')
    import argparse
    namespace = {'argparse': argparse, 'json': json, 'shapes': shapes}
    functions = []
    for node in (old_args, new_args):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<audited native_arguments>', 'exec'), namespace)
        functions.append(namespace['native_arguments'])
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest['_model_path'] = specification['model_path']
    for point in manifest['points']:
        if point['kind'] == 'decoder' and (
                vars(functions[0](ArgumentModule, manifest, point)) !=
                vars(functions[1](ArgumentModule, manifest, point))):
            raise ValueError(f'Decode arguments changed: {point["id"]}')
    return original


def prepare_reuse(source, directory, specification, info):
    from measurement_store import read_group
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if source == directory or (source / 'results/decode_reuse.json').exists():
        raise ValueError('Only a direct, original accepted pilot may be reused')
    original_info = json.loads((source / 'campaign.json').read_text())
    for name, expected in original_info['files_sha256'].items():
        if file_sha(source / 'code' / name) != expected:
            raise ValueError(f'Original frozen campaign changed: {name}')
    original = check_decode_compatibility(source, specification)
    approval = json.loads((source / 'pilot_acceptance.json').read_text())
    old_signature = digest_json(original)
    if (not approval.get('passed') or original_info['run_signature'] != old_signature
            or approval['run_signature'] != old_signature or approval.get('errors')):
        raise ValueError('Source pilot is not accepted')
    # Re-run the source revision's complete offline audit before pinning it.
    audit_program = (
        'import json,sys; from pathlib import Path; '
        'sys.path.insert(0, str(Path(sys.argv[1])/"code")); '
        'from verify_campaign import verify; '
        'a=verify(Path(sys.argv[1]),Path(sys.argv[2]),"pilot"); '
        'expected=json.loads((Path(sys.argv[1])/"pilot_acceptance.json").read_text()); '
        'assert a==expected and a["passed"], "Source pilot audit changed"'
    )
    subprocess.run([sys.executable, '-B', '-c', audit_program, str(source), approval['job_dir']],
                   check=True, cwd=source / 'code')
    source_pins = {str(source / name): file_sha(source / name) for name in
                   ('campaign.json', 'pilot_acceptance.json', 'results/run_config.json', 'manifest.json')}
    source_pins.update({str(source / 'code' / n): h for n, h in original_info['files_sha256'].items()})
    source_pins[approval['gpu_preflight']] = file_sha(approval['gpu_preflight'])
    points = {}
    manifest = json.loads((source / 'manifest.json').read_text())
    for point in manifest['points']:
        if point['kind'] != 'decoder' or point['id'] not in info['pilot_ids']:
            continue
        recovered, canonical, invalid = read_group(source / 'results', point, old_signature,
                                                   specification['tp'], False)
        if invalid or canonical is None or set(recovered) != set(shapes(point)):
            raise ValueError(f'Source decode is incomplete: {point["id"]}')
        accepted = next(r for r in approval['validated_shapes'] if r['id'] == point['id'])
        item = next(iter(recovered.values()))
        if item['source'] != accepted['source'] or digest_json(item['result']) != accepted['source_result_sha256']:
            raise ValueError('Source decode differs from accepted evidence')
        canonical_path = source / 'results' / (point['id'] + '.json')
        pins = {str(canonical_path): file_sha(canonical_path)}
        pins.update({str(p): file_sha(p) for p in Path(item['source']).glob('*.json')})
        points[point['id']] = {'point': point, 'source': item['source'], 'job': item['job'],
                              'source_result_sha256': accepted['source_result_sha256'], 'files_sha256': pins}
    if len(points) != 14:
        raise ValueError('Expected all 14 accepted pilot decode groups')
    catalog = {'schema': 'prefill-kv-cap-decode-reuse.v1', 'target_run_signature': info['run_signature'],
               'source_run_signature': old_signature, 'source_campaign': str(source),
               'source_files_sha256': source_pins, 'points': points,
               'prefill_shapes_reused': 0, 'decode_arguments_identical': True,
               'decode_computation_and_timing_identical': True}
    run = directory / 'results'
    run.mkdir()
    atomic_json(run / 'run_config.json', specification)
    atomic_json(run / 'manifest.json', manifest)
    atomic_json(run / 'decode_reuse.json', catalog)
    atomic_json(directory / 'source_pilot_acceptance.json', approval)
    return {'source_campaign': str(source), 'source_run_signature': old_signature,
            'catalog_sha256': file_sha(run / 'decode_reuse.json'),
            'source_acceptance_sha256': file_sha(directory / 'source_pilot_acceptance.json'),
            'decode_shapes_reused': len(points), 'prefill_shapes_reused': 0}


def validate_catalog(directory, info):
    path = directory / 'results/decode_reuse.json'
    reference = info['decode_reuse']
    if (file_sha(path) != reference['catalog_sha256'] or
            file_sha(directory / 'source_pilot_acceptance.json') != reference['source_acceptance_sha256']):
        raise ValueError('Frozen decode reuse authorization changed')
    catalog = json.loads(path.read_text())
    if (catalog['target_run_signature'] != info['run_signature'] or
            catalog['source_run_signature'] != reference['source_run_signature'] or
            catalog['source_campaign'] != reference['source_campaign']):
        raise ValueError('Decode reuse campaign binding mismatch')
    check_pins(catalog['source_files_sha256'])
    for entry in catalog['points'].values():
        check_pins(entry['files_sha256'])


def read_reused_decode(run, point, signature, tp, smoke):
    catalog_path = run / 'decode_reuse.json'
    if not catalog_path.exists():
        return None
    catalog = json.loads(catalog_path.read_text())
    if catalog['target_run_signature'] != signature:
        raise ValueError('Decode reuse target signature mismatch')
    entry = catalog['points'].get(point['id'])
    if entry is None:
        return None
    if smoke or point['kind'] != 'decoder' or entry['point'] != point:
        raise ValueError('Decode reuse cannot import prefill, smoke, or changed shapes')
    check_pins(catalog['source_files_sha256'])
    check_pins(entry['files_sha256'])
    source_run = Path(catalog['source_campaign']) / 'results'
    if source_run.resolve() == run.resolve() or (source_run / 'decode_reuse.json').exists():
        raise ValueError('Recursive reuse is not supported')
    from measurement_store import assemble_group, read_group
    recovered, _, invalid = read_group(source_run, point, catalog['source_run_signature'], tp, smoke)
    item = recovered.get(shapes(point)[0])
    if invalid or item is None or (item['source'] != entry['source'] or item['job'] != entry['job']
                                  or digest_json(item['result']) != entry['source_result_sha256']):
        raise ValueError('Reused decode differs from its original raw evidence')
    if any((run / '.attempts' / point['id']).glob('attempt-*')):
        raise ValueError('Reused decode was measured again')
    path = run / (point['id'] + '.json')
    canonical = json.loads(path.read_text()) if path.exists() else None
    if canonical is not None and canonical != assemble_group(point, recovered):
        raise ValueError('Published reused decode changed the original result/signature')
    return recovered, canonical, []
