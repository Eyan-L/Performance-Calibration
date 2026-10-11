"""Corpus selection and result formatting shared by the native calibration tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'runtime_config.json').read_text())
KIT = Path(CONFIG['reference_kit'])
MEASUREMENT_POLICY = {
    'primary_rank': 0,
    'rank_aggregation': 'rank 0 per iteration, then summarize',
    'auxiliary_aggregation': 'max across ranks for each iteration, then summarize',
    'per_sample_tp_barrier': False,
    'numerical_checks': 'separate validation process; not in measurement loops',
    'prefill_tp4': 'extension of the reference TP=1 prefill benchmark',
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def shapes(point):
    if point['kind'] == 'decoder':
        return [(point['bs'], point['ctx'])]
    if point['kind'] != 'prefill':
        raise ValueError(f"Unknown point kind: {point['kind']}")
    return [(b, length) for b in point['batches'] for length in point['lengths']]


def sampling(point, smoke):
    if smoke:
        return (1, 3, 10) if point['kind'] == 'decoder' else (1, 2, None)
    return point['warmups'], point['iters'], point.get('steps')


def load_tokens(point, batch_size, length, seed=0):
    # Same corpus, split, suffix selection, and point seed as the reference kit.
    import numpy as np
    root = KIT / 'inputs' / ('corpus_' + point['corpus'])
    index = json.loads((root / 'index.json').read_text())
    candidates = [e for e in index['entries'] if e['split'] == point['split'] and e['length'] >= length]
    if len(candidates) < batch_size:
        raise ValueError(f'{point["id"]}: only {len(candidates)} suitable prompts, need {batch_size}')
    rng = np.random.default_rng(seed + batch_size * 1_000_003 + length * 101)
    selected = [candidates[int(i)] for i in rng.permutation(len(candidates))[:batch_size]]
    tokens = np.load(root / 'token_ids.npy', mmap_mode='r', allow_pickle=False)
    inputs = []
    for entry in selected:
        end = entry['offset'] + entry['length']
        start = end - length
        if not 0 <= start < end <= tokens.size:
            raise ValueError('Invalid corpus offsets')
        inputs.append(tokens[start:end].astype(np.int64).tolist())
    metadata = {
        'batch_size': batch_size, 'request_length': length, 'split': point['split'],
        'selected_entries': selected,
        'token_sha256': hashlib.sha256(np.asarray(inputs, dtype=np.int32).tobytes()).hexdigest(),
        'corpus_token_sha256': index['token_sha256'],
    }
    return inputs, metadata


def input_coverage(manifest):
    indexes = {c: json.loads((KIT / 'inputs' / ('corpus_' + c) / 'index.json').read_text()) for c in ('short', 'long')}
    failures = []
    for point in manifest['points']:
        for batch, length in shapes(point):
            available = sum(e['split'] == point['split'] and e['length'] >= length for e in indexes[point['corpus']]['entries'])
            if available < batch:
                failures.append({'id': point['id'], 'batch': batch, 'length': length, 'available': available})
    return failures


def run_specification(manifest_path, model_path, smoke):
    manifest_path, model_path = Path(manifest_path).resolve(), Path(model_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    config_path = model_path / 'config.json'
    text = json.loads(config_path.read_text()).get('text_config', {})
    expected = {'qwen38bf16': ('qwen3_5_text', 16, 4),
                'qwen36bf16': ('qwen3_5_moe_text', 10, 2)}
    if manifest['model'] not in expected:
        raise ValueError('This adapter currently measures BF16 Qwen3.6 / Qwen3.8 only')
    actual = (text.get('model_type'), text.get('layer_types', []).count('full_attention'), text.get('num_key_value_heads'))
    if actual != expected[manifest['model']]:
        raise ValueError(f"Model config {actual} does not match manifest {manifest['model']}")
    sources = ('runtime_guard.py', 'run_in_sif.sh', 'calibration_common.py',
               'calibrate_point.py', 'run_points.py', 'collect_results.py', 'gpu_preflight.py', 'fit_results.py',
               'measurement_store.py', 'campaign.py', 'verify_campaign.py', 'report_results.py')
    return {'schema': 'native-sglang-calibration.v2', 'runtime': CONFIG,
            'measurement_policy': MEASUREMENT_POLICY,
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'model_path': str(model_path), 'model_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'corpus_index_sha256': {c: hashlib.sha256((KIT / 'inputs' / ('corpus_' + c) / 'index.json').read_bytes()).hexdigest()
                                    for c in ('short', 'long')},
            'reference_fit_sha256': hashlib.sha256((KIT / 'fit/fit_amd_models.py').read_bytes()).hexdigest(),
            'reference_model_config_sha256': hashlib.sha256(
                (KIT / 'inputs/model_configs' / (manifest['model'] + '.config.json')).read_bytes()).hexdigest(),
            'smoke': smoke, 'tp': manifest['tp'], 'attention_backend': manifest['attention_backend'],
            'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}


def rank_max_samples(rank_samples):
    if not rank_samples or not rank_samples[0] or any(len(r) != len(rank_samples[0]) for r in rank_samples):
        raise ValueError('Missing or unequal TP rank samples')
    if any(not math.isfinite(v) or v <= 0 for r in rank_samples for v in r):
        raise ValueError('Invalid GPU timing sample')
    return [max(sample) for sample in zip(*rank_samples)]


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def decode_summary(totals, steps):
    per_step = [v / steps for v in totals]
    return {'mode': 'baseline_short_only', 'decode_steps': steps,
            'total_ms_per_step_median': statistics.median(per_step),
            'total_ms_per_step_mean': statistics.fmean(per_step),
            'total_ms_per_step_min': min(per_step),
            'total_ms_per_step_p90': sorted(per_step)[int(0.9 * (len(per_step) - 1))]}


def format_result(point, ranks, metadata):
    warmups, iterations, steps = sampling(point, metadata['smoke'])
    if len(ranks) != metadata['tp'] or sorted(r['rank'] for r in ranks) != list(range(metadata['tp'])):
        raise ValueError('Incomplete TP workers')
    ranks = sorted(ranks, key=lambda r: r['rank'])
    if metadata.get('purpose') != 'measurement':
        raise ValueError('Only measurement attempts may produce calibration results')
    metadata = {**metadata, 'warmups': warmups, 'iterations': iterations, 'decode_steps': steps,
                'rank_aggregation': MEASUREMENT_POLICY['rank_aggregation'],
                'measurement_policy': MEASUREMENT_POLICY,
                'rank_metadata': [r['metadata'] for r in ranks]}
    shape_results = []
    for position, (batch, length) in enumerate(shapes(point)):
        records = [r['points'][position] for r in ranks]
        if any((r['batch_size'], r['request_length']) != (batch, length) or len(r['samples_ms']) != iterations for r in records):
            raise ValueError('TP sample shape mismatch')
        if len({r['input']['token_sha256'] for r in records}) != 1:
            raise ValueError('TP ranks received different input tokens')
        maximum = rank_max_samples([r['samples_ms'] for r in records])
        values = list(records[0]['samples_ms'])
        shape_results.append((batch, length, values, maximum, records))
    if point['kind'] == 'decoder':
        _, _, totals, maximum, records = shape_results[0]
        summary = decode_summary(totals, steps)
        rows = [{'iter': i, 'total_ms': total, 'total_ms_per_step': total / steps,
                 'rank_total_ms': [r['samples_ms'][i] for r in records]} for i, total in enumerate(totals)]
        return {'metadata': metadata, 'completed': True, 'summaries': [summary], 'rows': rows,
                'input': records[0]['input'],
                'rank_layouts': [r['layout'] for r in records],
                'rank_diagnostics': {'max_total_ms': maximum,
                    'max_summary': decode_summary(maximum, steps),
                    'per_rank_summaries': [decode_summary(r['samples_ms'], steps) for r in records]}}
    rows = []
    for batch, length, values, maximum, records in shape_results:
        rows.append({'batch_size': batch, 'request_length': length, 'status': 'ok',
                     'prompt_tokens': batch * length,
                     'forward_ms_median': statistics.median(values),
                     'forward_ms_mean': statistics.fmean(values),
                     'forward_ms_p90': percentile(values, .9),
                     'iterations': iterations, 'samples_ms': values,
                     'rank_samples_ms': [r['samples_ms'] for r in records],
                     'input': records[0]['input'],
                     'rank_layouts': [r['layout'] for r in records],
                     'rank_diagnostics': {'max_samples_ms': maximum,
                         'max_median_ms': statistics.median(maximum),
                         'per_rank_medians_ms': [statistics.median(r['samples_ms']) for r in records]}})
    return {'metadata': metadata, 'completed': True, 'points': rows}


def validate_result(result, point, signature, tp, smoke):
    """Validate canonical timings rather than trusting an exit code or row count."""
    metadata = result.get('metadata', {})
    if (not result.get('completed') or metadata.get('run_signature') != signature
            or metadata.get('tp') != tp or metadata.get('smoke') != smoke
            or metadata.get('point') != point or not metadata.get('gpu_execution_tested')
            or metadata.get('purpose') != 'measurement'
            or metadata.get('measurement_policy') != MEASUREMENT_POLICY
            or metadata.get('rank_aggregation') != MEASUREMENT_POLICY['rank_aggregation']):
        raise ValueError('Result completion or provenance does not match the run')
    iterations, steps = sampling(point, smoke)[1:]
    if point['kind'] == 'decoder':
        rows = result.get('rows', [])
        if len(rows) != iterations or len(result.get('summaries', [])) != 1:
            raise ValueError('Incomplete decode measurement')
        if any(len(r['rank_total_ms']) != tp for r in rows):
            raise ValueError('Decode rank count mismatch')
        rank_samples = [[r['rank_total_ms'][rank] for r in rows] for rank in range(tp)]
        maximum = rank_max_samples(rank_samples)
        values = rank_samples[0]
        if any(not math.isclose(row['total_ms'], value, rel_tol=1e-9)
               or not math.isclose(row['total_ms_per_step'], value / steps, rel_tol=1e-9)
               for row, value in zip(rows, values)):
            raise ValueError('Decode aggregation mismatch')
        if result['summaries'][0] != decode_summary(values, steps):
            raise ValueError('Decode summary mismatch')
        if result.get('rank_diagnostics') != {'max_total_ms': maximum,
                'max_summary': decode_summary(maximum, steps),
                'per_rank_summaries': [decode_summary(r, steps) for r in rank_samples]}:
            raise ValueError('Decode diagnostic aggregation mismatch')
        layout_rows = [(point['bs'], point['ctx'], result.get('rank_layouts', []))]
    else:
        rows = result.get('points', [])
        if [(r['batch_size'], r['request_length']) for r in rows] != shapes(point):
            raise ValueError('Prefill shape coverage mismatch')
        for row in rows:
            if row.get('status') != 'ok' or len(row.get('samples_ms', [])) != iterations:
                raise ValueError('Incomplete prefill measurement')
            if len(row.get('rank_samples_ms', [])) != tp:
                raise ValueError('Prefill rank count mismatch')
            maximum = rank_max_samples(row['rank_samples_ms'])
            values = row['rank_samples_ms'][0]
            if len(values) != iterations or any(not math.isclose(a, b, rel_tol=1e-9)
                    for a, b in zip(row['samples_ms'], values)) or not math.isclose(
                    row['forward_ms_median'], statistics.median(values), rel_tol=1e-9):
                raise ValueError('Prefill aggregation mismatch')
            if (not math.isclose(row['forward_ms_mean'], statistics.fmean(values), rel_tol=1e-9)
                    or not math.isclose(row['forward_ms_p90'], percentile(values, .9), rel_tol=1e-9)
                    or row['iterations'] != iterations
                    or row['rank_diagnostics'] != {'max_samples_ms': maximum,
                        'max_median_ms': statistics.median(maximum),
                        'per_rank_medians_ms': [statistics.median(r) for r in row['rank_samples_ms']]}):
                raise ValueError('Prefill summary or diagnostic mismatch')
        layout_rows = [(r['batch_size'], r['request_length'], r.get('rank_layouts', [])) for r in rows]
    for batch, length, layouts in layout_rows:
        expected_length = length + (point['kind'] == 'decoder')
        expected_tokens = batch if point['kind'] == 'decoder' else batch * length
        if len(layouts) != tp or any(l.get('batch_size') != batch
                or l.get('actual_input_token_count') != expected_tokens
                or l.get('actual_sequence_lengths') not in (None, [expected_length] * batch)
                for l in layouts):
            raise ValueError('Recorded forward layout differs from requested shape')
