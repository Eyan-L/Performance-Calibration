"""Collect validated shapes while retaining their original attempts and jobs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from calibration_common import atomic_json, digest_json, shapes
from measurement_store import assemble_group, missing_shapes, read_group


def collect(run, out, only=None):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    specification = json.loads((run / 'run_config.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    signature = digest_json(specification)
    tp, smoke = specification['tp'], specification['smoke']
    selected = [p for p in manifest['points'] if not only or p['id'] in only]
    if only and set(only) != {p['id'] for p in selected}:
        raise ValueError('Unknown requested point')
    out.mkdir(parents=True)
    audit = {'run_signature': signature, 'selected': [p['id'] for p in selected],
             'completed_groups': [], 'partial_groups': [], 'missing_shapes': [],
             'invalid_artifacts': [], 'recovered_shapes': [], 'shape_sources': [], 'smoke': smoke,
             'full_grid_coverage_complete': False,
             'selected_shape_count': sum(len(shapes(p)) for p in selected)}
    for point in selected:
        canonical = run / (point['id'] + '.json')
        recovered, result, invalid = read_group(run, point, signature, tp, smoke)
        audit['invalid_artifacts'].extend(invalid)
        missing = missing_shapes(point, recovered)
        audit['missing_shapes'].extend({'id': point['id'], 'batch': b, 'length': l} for b, l in missing)
        for (b, length), item in recovered.items():
            source = {'id': point['id'], 'batch': b, 'length': length,
                      'source': item['source'], 'job': item['job'],
                      'measurement_run_signature': item['result']['metadata']['run_signature']}
            audit['shape_sources'].append(source)
            if result is None:
                audit['recovered_shapes'].append(source)
        if result is not None:
            shutil.copyfile(canonical, out / canonical.name)
        elif not missing:
            atomic_json(out / canonical.name, assemble_group(point, recovered))
        elif point['kind'] == 'prefill':
            rows = [recovered[s]['result']['points'][0] if s in recovered else
                    {'batch_size': s[0], 'request_length': s[1], 'status': 'missing',
                     'error': 'No complete matching samples across all TP ranks'} for s in shapes(point)]
            atomic_json(out / canonical.name, {'completed': False, 'points': rows,
                        'metadata': {'run_signature': signature, 'partial_recovery': True}})
        audit['partial_groups' if missing else 'completed_groups'].append(point['id'])
    audit['measurement_coverage_complete'] = not audit['missing_shapes']
    audit['full_grid_coverage_complete'] = (
        audit['measurement_coverage_complete'] and len(selected) == len(manifest['points']))
    audit['fit_design_verified'] = False
    atomic_json(out / 'coverage.json', audit)
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'run_config.json', specification)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    audit = collect(args.run, args.out, args.only)
    print(json.dumps(audit), flush=True)
    if args.require_complete and not audit['measurement_coverage_complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
