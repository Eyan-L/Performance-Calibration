"""Audit real GPU evidence and release a campaign only after pilot acceptance."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

from calibration_common import KIT, MEASUREMENT_POLICY, digest_json, shapes
from measurement_store import read_group


def check_input(point, identity):
    root = KIT / ('inputs/corpus_' + point['corpus'])
    index = json.loads((root / 'index.json').read_text())
    batch, length = identity['batch_size'], identity['request_length']
    if identity['split'] != point['split'] or len(identity['selected_entries']) != batch:
        raise ValueError('Input split or batch mismatch')
    digest = hashlib.sha256()
    with (root / 'token_ids.npy').open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid corpus file')
        version = stream.read(2)[0]
        header_size = struct.unpack('<H' if version == 1 else '<I', stream.read(2 if version == 1 else 4))[0]
        header = ast.literal_eval(stream.read(header_size).decode('latin1').strip())
        if header['descr'] not in ('<i4', '<u4') or header['fortran_order']:
            raise ValueError('Unexpected token corpus storage')
        data_start = stream.tell()
        for entry in identity['selected_entries']:
            if entry not in index['entries'] or entry['split'] != point['split'] or entry['length'] < length:
                raise ValueError('Recorded prompt does not belong to the declared corpus split')
            offset = entry['offset'] + entry['length'] - length
            if offset < 0 or offset + length > header['shape'][0]:
                raise ValueError('Invalid recorded token slice')
            stream.seek(data_start + 4 * offset)
            data = stream.read(4 * length)
            if len(data) != 4 * length:
                raise ValueError('Truncated corpus')
            digest.update(data)
    if digest.hexdigest() != identity['token_sha256']:
        raise ValueError('Recorded input differs from actual corpus token IDs')


def oom_evidence(run, point, shape, signature):
    evidence = []
    for request_path in (run / '.attempts' / point['id']).glob('attempt-*/request.json'):
        request = json.loads(request_path.read_text())
        if request.get('metadata', {}).get('run_signature') != signature:
            continue
        for rank_path in request_path.parent.glob('rank*.json'):
            rank = json.loads(rank_path.read_text())
            if rank.get('error_kind') == 'out_of_memory' and (
                    rank.get('active_shape') == list(shape) or rank.get('stage') == 'load_model_and_capture_graph'):
                evidence.append({'path': str(rank_path), 'error': rank['error']})
    return evidence


def core_shape(point, shape):
    batch, length = shape
    if point['kind'] == 'decoder':
        return point['id'].startswith('dec_train_') and (
            batch in (1, 4, 16) and length in (4096, 16384) or batch == 4 and length == 8192)
    return point['split'] == 'train' and batch in (1, 4) and length in (4096, 8192, 16384)


def communication_profile(comm):
    return {'requested_custom_all_reduce_disabled': comm['requested_custom_all_reduce_disabled'],
            'available_communicators': comm['available_communicators']}


def verify_forward_validation(job_dir, manifest, signature):
    """Validate separate GPU execution evidence without treating it as timings."""
    profiles, evidence = {}, []
    for point_id in ('dec_train_bs4_ctx8192', 'pre_train_g0'):
        point = next(p for p in manifest['points'] if p['id'] == point_id)
        path = job_dir / 'forward_validation' / (point_id + '.json')
        result = json.loads(path.read_text())
        metadata = result['metadata']
        if (not result.get('completed') or metadata.get('purpose') != 'forward_validation'
                or not metadata.get('gpu_execution_tested') or metadata.get('run_signature') != signature
                or metadata.get('point') != point or metadata.get('tp') != manifest['tp']
                or metadata.get('measurement_policy') != MEASUREMENT_POLICY):
            raise ValueError('Independent validation provenance/completion mismatch')
        attempt = Path(metadata['rank_artifacts'])
        if not json.loads((attempt / 'status.json').read_text()).get('completed'):
            raise ValueError('Independent validation process did not finish')
        ranks = result['ranks']
        if [r['rank'] for r in ranks] != list(range(manifest['tp'])):
            raise ValueError('Independent validation ranks are incomplete')
        token_hashes = []
        for rank in ranks:
            if rank != json.loads((attempt / f'rank{rank["rank"]}.json').read_text()):
                raise ValueError('Independent validation differs from original rank artifacts')
            rm = rank['metadata']
            if (not rank.get('completed') or rm.get('purpose') != 'forward_validation'
                    or rm.get('weight_dtype') != 'torch.bfloat16' or rm.get('kv_cache_dtype') != 'torch.bfloat16'
                    or [(p['batch_size'], p['request_length']) for p in rank['points']] != shapes(point)):
                raise ValueError('Independent validation worker/layout coverage mismatch')
            comm = rm['communication']
            if not comm.get('all_reduce_numeric_check'):
                raise ValueError('Independent TP numerical communication check failed')
            profile = communication_profile(comm)
            profiles[digest_json(profile)] = profile
            hashes = []
            for row in rank['points']:
                batch, length = row['batch_size'], row['request_length']
                checks = row['validation']
                decode = point['kind'] == 'decoder'
                if ('samples_ms' in row or not all(checks.get(k) for k in
                        ('shape_matches', 'finite_logits', 'layout_verified'))
                        or checks.get('graph') != decode or checks.get('batch') != batch
                        or checks.get('context') != length or checks.get('batch_size') != batch
                        or checks.get('actual_sequence_lengths') != [length + decode] * batch
                        or checks.get('actual_input_token_count') != (batch if decode else batch * length)):
                    raise ValueError('Independent forward numerical/layout/graph check failed')
                check_input(point, row['input'])
                hashes.append(row['input']['token_sha256'])
            token_hashes.append(hashes)
        if any(h != token_hashes[0] for h in token_hashes):
            raise ValueError('Independent validation TP inputs differ')
        evidence.append({'id': point_id, 'path': str(path), 'sha256': digest_json(result),
                         'validated_shapes': len(shapes(point))})
    return profiles, evidence


def verify(directory, job_dir, phase):
    from campaign import load_campaign
    directory, info = load_campaign(directory)
    job_dir = Path(job_dir)
    signature = info['run_signature']
    manifest = json.loads((directory / 'manifest.json').read_text())
    selection = info['pilot_ids'] if phase == 'pilot' else [p['id'] for p in manifest['points']]
    points = [p for p in manifest['points'] if p['id'] in selection]
    run = directory / 'results'
    audit = {'passed': False, 'phase': phase, 'job_dir': str(job_dir), 'run_signature': signature,
             'gpu_preflight': str(job_dir / 'gpu_preflight.json'), 'errors': [],
             'validated_shapes': [], 'unsupported_shapes': [], 'communication_profiles': [],
             'fit_accuracy_is_not_an_acceptance_filter': True}
    preflight = json.loads((job_dir / 'gpu_preflight.json').read_text())
    if not preflight.get('completed') or len(preflight['devices']) != 4:
        audit['errors'].append('Actual four-GPU preflight did not pass')
    profiles, measurements = {}, {}
    try:
        profiles, evidence = verify_forward_validation(job_dir, manifest, signature)
        audit['independent_forward_validation'] = evidence
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        audit['errors'].append('Independent GPU validation: ' + str(exc))
    for point in points:
        try:
            recovered, _, invalid = read_group(run, point, signature, manifest['tp'], False)
            if invalid:
                audit['errors'].append({'id': point['id'], 'invalid_artifacts': invalid})
            for shape in shapes(point):
                if shape not in recovered:
                    evidence = oom_evidence(run, point, shape, signature)
                    if evidence and not core_shape(point, shape):
                        audit['unsupported_shapes'].append({'id': point['id'], 'batch': shape[0],
                                                            'length': shape[1], 'oom_evidence': evidence})
                    else:
                        audit['errors'].append({'id': point['id'], 'missing_shape': shape,
                                                'core': core_shape(point, shape)})
                    continue
                item = recovered[shape]
                result = item['result']
                identity = result['input'] if point['kind'] == 'decoder' else result['points'][0]['input']
                check_input(point, identity)
                for rank in item['rank_records']:
                    comm = rank['metadata'].get('communication', {})
                    if 'all_reduce_numeric_check' in comm:
                        raise ValueError('Numerical communication probe was mixed into a measurement process')
                    profile = communication_profile(comm)
                    key = digest_json(profile)
                    profiles[key] = profile
                prefix = point['id'].split('_')[1]
                value = (result['summaries'][0]['total_ms_per_step_median'] if point['kind'] == 'decoder'
                         else result['points'][0]['forward_ms_median'])
                measurements[(point['kind'], prefix, *shape)] = value
                audit['validated_shapes'].append({'id': point['id'], 'batch': shape[0], 'length': shape[1],
                    'source': item['source'], 'job': item['job'], 'input_token_sha256': identity['token_sha256'],
                    'source_result_sha256': digest_json(result)})
        except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            audit['errors'].append({'id': point['id'], 'error': str(exc)})
    audit['communication_profiles'] = list(profiles.values())
    if len(profiles) != 1:
        audit['errors'].append('Communication setup differs between ranks or points')
    completion = json.loads((job_dir / 'completion.json').read_text())
    if set(completion['selected']) != set(selection) or completion['run_signature'] != signature:
        audit['errors'].append('Driver selection/signature differs from the expected phase')
    status_path = run / 'status.jsonl'
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] not in selection:
                continue
            if row.get('run_signature') != signature:
                audit['errors'].append('Measurement status provenance mismatch')
            if not row.get('result') and row.get('failure_kind') != 'out_of_memory':
                audit['errors'].append({'id': row['id'], 'unexpected_execution_failure': row})
    fit = json.loads((job_dir / 'fit/model.json').read_text())
    fit_validation = json.loads((job_dir / 'fit/fit_validation.json').read_text())
    if not fit_validation.get('training_design_verified') or fit_validation['run_signature'] != signature:
        audit['errors'].append('Actual-data fitting design did not pass')
    try:
        for kind, filename in (('decoder', 'decoder_points.csv'), ('prefill', 'prefill_points.csv')):
            with (job_dir / 'fit' / filename).open() as stream:
                for row in csv.DictReader(stream):
                    if kind == 'prefill' and row['status'] != 'ok':
                        continue
                    batch, length = ((int(row['bs']), int(row['ctx'])) if kind == 'decoder'
                                     else (int(row['batch']), int(row['length'])))
                    measured = float(row['step_ms'] if kind == 'decoder' else row['ms'])
                    expected = measurements[(kind, row['split'], batch, length)]
                    # The unchanged fitter writes decode with four decimal
                    # places and prefill with three; account only for rounding.
                    tolerance = 0.000051 if kind == 'decoder' else 0.000501
                    if not math.isclose(measured, expected, abs_tol=tolerance, rel_tol=0):
                        raise ValueError('Fit reader/export changed the observed timings')
        for kind, cv in (('decoder', 'cv_checkerboard5'), ('prefill', 'cv_leave_one_out')):
            if not all(v is None or math.isfinite(v) for v in fit[kind]['coefficients'].values()):
                raise ValueError('Nonfinite fitted coefficients')
            for label in ('train', cv, 'holdout'):
                metrics = fit[kind][label]
                if metrics['n'] <= 0 or not all(math.isfinite(v) for v in metrics.values()):
                    raise ValueError('Missing or nonfinite actual-data validation metrics')
        repeats = fit['decoder']['repeat_stability']
        if not repeats:
            raise ValueError('No actual repeat-stability measurement')
        for repeat in repeats:
            if repeat.get('first') is None or not math.isfinite(repeat['diff_percent']) or abs(repeat['diff_percent']) > 10:
                raise ValueError('Repeat drift exceeds 10%; investigate before continuing')
        audit['repeat_stability'] = repeats
        for name in ('report.md', 'decoder_errors.png', 'prefill_errors.png'):
            if not (job_dir / 'fit' / name).is_file() or (job_dir / 'fit' / name).stat().st_size == 0:
                raise ValueError('Missing generated report/figure')
    except (ValueError, KeyError, TypeError) as exc:
        audit['errors'].append(str(exc))
    if phase == 'full':
        pilot = json.loads((directory / 'pilot_acceptance.json').read_text())
        current = {(r['id'], r['batch'], r['length']): r for r in audit['validated_shapes']}
        for previous in pilot['validated_shapes']:
            item = current.get((previous['id'], previous['batch'], previous['length']))
            if item is None or any(item[k] != previous[k] for k in ('source', 'source_result_sha256')):
                audit['errors'].append('Accepted pilot data changed or were measured again')
        if pilot['communication_profiles'] != audit['communication_profiles']:
            audit['errors'].append('Effective communication setup differs from the accepted pilot')
        audit['pilot_shapes_reused'] = len(pilot['validated_shapes'])
    audit['measurement_coverage_complete'] = len(audit['validated_shapes']) == sum(len(shapes(p)) for p in points)
    audit['selected_shapes'] = sum(len(shapes(p)) for p in points)
    audit['remaining_registered_groups'] = fit['missing_preregistered_points']
    audit['passed'] = not audit['errors']
    return audit
