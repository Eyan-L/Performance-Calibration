"""Validate and reuse complete shapes without combining ranks across attempts."""
from __future__ import annotations

import json
from pathlib import Path

from calibration_common import MEASUREMENT_POLICY, format_result, shapes, validate_result


def read_group(run, point, signature, tp, smoke):
    """Return first valid attempt per shape, canonical result, and audit errors."""
    run = Path(run)
    from decode_reuse import read_reused_decode
    reused = read_reused_decode(run, point, signature, tp, smoke)
    if reused is not None:
        return reused
    canonical_path = run / (point['id'] + '.json')
    canonical = None
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
        validate_result(canonical, point, signature, tp, smoke)
    recovered, invalid = {}, []
    requests = []
    for path in (run / '.attempts' / point['id']).glob('attempt-*/request.json'):
        try:
            request = json.loads(path.read_text())
            requests.append((request.get('started_at', 0), str(path), path, request))
        except (ValueError, OSError) as exc:
            invalid.append({'path': str(path), 'error': str(exc)})
    for _, _, request_path, request in sorted(requests):
        metadata = request.get('metadata', {})
        if (metadata.get('run_signature') != signature or request.get('point') != point
                or request.get('smoke') != smoke or metadata.get('tp') != tp
                or metadata.get('purpose') != 'measurement'
                or metadata.get('measurement_policy') != MEASUREMENT_POLICY):
            invalid.append({'path': str(request_path), 'error': 'provenance mismatch'})
            continue
        ranks = []
        try:
            for rank in range(tp):
                path = request_path.parent / f'rank{rank}.json'
                if path.exists():
                    item = json.loads(path.read_text())
                    if item.get('rank') != rank:
                        raise ValueError('Rank identity mismatch')
                    ranks.append(item)
        except (ValueError, OSError) as exc:
            invalid.append({'path': str(request_path.parent), 'error': str(exc)})
            continue
        if len(ranks) != tp or point['kind'] == 'decoder' and not all(r.get('completed') for r in ranks):
            continue
        for batch, length in shapes(point):
            if (batch, length) in recovered:
                continue
            subranks = []
            for rank in ranks:
                rows = [r for r in rank.get('points', []) if
                        (r.get('batch_size'), r.get('request_length')) == (batch, length)]
                if len(rows) != 1:
                    break
                subranks.append({**rank, 'points': rows})
            if len(subranks) != tp:
                continue
            subpoint = point if point['kind'] == 'decoder' else {**point, 'batches': [batch], 'lengths': [length]}
            try:
                if any(r['metadata'].get('weight_dtype') != 'torch.bfloat16'
                       or r['metadata'].get('kv_cache_dtype') != 'torch.bfloat16'
                       or r['metadata'].get('purpose') != 'measurement' for r in subranks):
                    raise ValueError('Missing verified BF16 worker metadata')
                result = format_result(subpoint, subranks, {**metadata,
                    'point': subpoint, 'gpu_execution_tested': True})
                validate_result(result, subpoint, signature, tp, smoke)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                invalid.append({'path': str(request_path.parent), 'error': str(exc)})
                continue
            recovered[(batch, length)] = {'result': result, 'source': str(request_path.parent),
                'job': metadata.get('job'), 'rank_records': subranks}
    if canonical is not None:
        if set(recovered) != set(shapes(point)):
            raise ValueError(f'Canonical result has missing raw evidence: {canonical_path}')
        # Validate canonical timings against the original rank samples, not just
        # against their own embedded summaries. Sources are selected by time.
        if point['kind'] == 'decoder':
            expected = next(iter(recovered.values()))['result']
            if any(canonical[k] != expected[k] for k in ('rows', 'summaries', 'input', 'rank_layouts', 'rank_diagnostics')):
                raise ValueError(f'Canonical decode differs from raw samples: {canonical_path}')
        else:
            expected = [recovered[s]['result']['points'][0] for s in shapes(point)]
            if canonical['points'] != expected:
                raise ValueError(f'Canonical prefill differs from raw samples: {canonical_path}')
    return recovered, canonical, invalid


def assemble_group(point, recovered):
    """Publish measurement coverage; retain each shape's actual execution source."""
    if set(recovered) != set(shapes(point)):
        return None
    if point['kind'] == 'decoder':
        return next(iter(recovered.values()))['result']
    first = recovered[shapes(point)[0]]['result']
    sources = [{'batch': b, 'length': l, 'attempt': recovered[(b, l)]['source'],
                'job': recovered[(b, l)]['job'],
                'rank_metadata': recovered[(b, l)]['result']['metadata']['rank_metadata']}
               for b, l in shapes(point)]
    metadata = {**first['metadata'], 'point': point, 'shape_sources': sources,
                'completion_scope': 'measurement samples; original attempt statuses are retained'}
    if len({s['attempt'] for s in sources}) > 1:
        metadata['rank_metadata'] = None
        metadata['rank_artifacts'] = None
    return {'completed': True, 'metadata': metadata,
            'points': [recovered[s]['result']['points'][0] for s in shapes(point)]}


def missing_shapes(point, recovered):
    return [s for s in shapes(point) if s not in recovered]
