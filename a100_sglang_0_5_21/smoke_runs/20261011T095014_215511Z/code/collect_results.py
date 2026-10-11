"""Collect all validated measurements, including partial TP prefill groups.

Collection creates a separate directory accepted by the unchanged reference
fit readers. It never marks a failed run complete or replaces raw artifacts.
Shape recovery uses the first valid attempt in creation order, never latency.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from calibration_common import (atomic_json, digest_json, format_result,
                                shapes, validate_result)


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
             'invalid_artifacts': [], 'recovered_shapes': [], 'smoke': smoke}
    for point in selected:
        canonical = run / (point['id'] + '.json')
        if canonical.exists():
            result = json.loads(canonical.read_text())
            validate_result(result, point, signature, tp, smoke)
            shutil.copyfile(canonical, out / canonical.name)
            audit['completed_groups'].append(point['id'])
            continue
        recovered = {}
        parent = run / '.attempts' / point['id']
        requests = sorted(parent.glob('attempt-*/request.json'),
                          key=lambda p: (json.loads(p.read_text()).get('started_at', 0), str(p)))
        for request_path in requests:
            request = json.loads(request_path.read_text())
            metadata = request.get('metadata', {})
            if (metadata.get('run_signature') != signature or request.get('point') != point
                    or request.get('smoke') != smoke or metadata.get('tp') != tp):
                audit['invalid_artifacts'].append({'path': str(request_path), 'error': 'provenance mismatch'})
                continue
            ranks = []
            for rank in range(tp):
                path = request_path.parent / f'rank{rank}.json'
                if path.exists():
                    ranks.append(json.loads(path.read_text()))
            if len(ranks) != tp or point['kind'] == 'decoder' and not all(r['completed'] for r in ranks):
                continue
            for batch, length in shapes(point):
                if (batch, length) in recovered:
                    continue
                subranks = []
                for rank in ranks:
                    rows = [r for r in rank['points'] if
                            (r['batch_size'], r['request_length']) == (batch, length)]
                    if len(rows) != 1:
                        break
                    subranks.append({**rank, 'points': rows})
                if len(subranks) != tp:
                    continue
                subpoint = point if point['kind'] == 'decoder' else {**point, 'batches': [batch], 'lengths': [length]}
                try:
                    if any(r['metadata'].get('weight_dtype') != 'torch.bfloat16'
                           or r['metadata'].get('kv_cache_dtype') != 'torch.bfloat16' for r in subranks):
                        raise ValueError('Missing verified BF16 worker metadata')
                    result = format_result(subpoint, subranks, {**metadata,
                        'point': subpoint, 'gpu_execution_tested': True})
                    validate_result(result, subpoint, signature, tp, smoke)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    audit['invalid_artifacts'].append({'path': str(request_path.parent), 'error': str(exc)})
                    continue
                recovered[(batch, length)] = result
                audit['recovered_shapes'].append({'id': point['id'], 'batch': batch,
                    'length': length, 'source': str(request_path.parent)})
        missing = [(b, l) for b, l in shapes(point) if (b, l) not in recovered]
        audit['missing_shapes'].extend({'id': point['id'], 'batch': b, 'length': l} for b, l in missing)
        if point['kind'] == 'decoder':
            if recovered:
                atomic_json(out / canonical.name, next(iter(recovered.values())))
        else:
            rows = []
            for batch, length in shapes(point):
                if (batch, length) in recovered:
                    rows.append(recovered[(batch, length)]['points'][0])
                else:
                    rows.append({'batch_size': batch, 'request_length': length,
                                 'status': 'missing', 'error': 'No complete matching samples across all TP ranks'})
            atomic_json(out / canonical.name, {'completed': False, 'points': rows,
                        'metadata': {'run_signature': signature, 'partial_recovery': True}})
        audit['partial_groups' if missing else 'completed_groups'].append(point['id'])
    audit['measurement_coverage_complete'] = not audit['missing_shapes']
    # Coverage alone does not establish that the regression is identifiable.
    # fit_results.py separately checks matrix rank and validation folds.
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
