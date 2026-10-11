"""CPU-only checks; synthetic fixtures here are never calibration measurements."""
from __future__ import annotations

import ast
import copy
import io
import json
import math
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration_common import (KIT, MEASUREMENT_POLICY, atomic_json, digest_json,
                                format_result, run_specification, sampling, shapes, validate_result)
from campaign import MODEL, pilot_ids
import campaign
from collect_results import collect
from measurement_store import assemble_group, missing_shapes, read_group
import run_points

MANIFEST = KIT / 'manifests/qwen38bf16_a100x4_flashinfer.json'
manifest = json.loads(MANIFEST.read_text())
specification = run_specification(MANIFEST, MODEL, False)
signature = digest_json(specification)
destination = Path(tempfile.mkdtemp(prefix='cpu-', dir=Path(__file__).parent))
checks = []


def passed(name):
    checks.append(name)


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError, IndexError, TypeError):
        return
    raise AssertionError('Invalid evidence was accepted')


def metadata(point):
    return {'run_signature': signature, 'tp': 4, 'smoke': False, 'point': point,
            'purpose': 'measurement', 'measurement_policy': MEASUREMENT_POLICY,
            'gpu_execution_tested': True, 'job': 'SYNTHETIC-CPU-FIXTURE',
            'validation_fixture_only': True}


def fixture_ranks(point, selected=None, multiplier=1, completed=True):
    rows = []
    for rank in range(4):
        points = []
        for batch, length in (selected or shapes(point)):
            if point['kind'] == 'decoder':
                base = (2.0 + .04 * batch + .1 * batch * length * .000065536) * point['steps']
            else:
                base = 1.0 + .0123456789 * batch * length + .005 * batch * length ** 2 / 2 / 1e6
            count = sampling(point, False)[1]
            samples = [base * multiplier * (1 + .03 * rank + .001 * (i - (count - 1) / 2))
                       for i in range(count)]
            points.append({'batch_size': batch, 'request_length': length, 'samples_ms': samples,
                'input': {'token_sha256': f'SYNTHETIC-{batch}-{length}'},
                'layout': {'batch_size': batch,
                           'actual_sequence_lengths': [length + (point['kind'] == 'decoder')] * batch,
                           'actual_input_token_count': batch if point['kind'] == 'decoder' else batch * length}})
        rows.append({'rank': rank, 'completed': completed, 'points': points,
                     'metadata': {'weight_dtype': 'torch.bfloat16', 'kv_cache_dtype': 'torch.bfloat16',
                                  'purpose': 'measurement', 'validation_fixture_only': True}})
    return rows


def attempt(run, point, name, started, ranks, purpose='measurement'):
    directory = run / '.attempts' / point['id'] / name
    md = {**metadata(point), 'purpose': purpose}
    atomic_json(directory / 'request.json', {'point': point, 'smoke': False,
                                            'started_at': started, 'metadata': md})
    for rank in ranks:
        atomic_json(directory / f'rank{rank["rank"]}.json', rank)
    return directory


selected = [p for p in manifest['points'] if p['id'] in pilot_ids()]
assert len(manifest['points']) == 86 and sum(len(shapes(p)) for p in manifest['points']) == 102
assert len(selected) == 21 and sum(len(shapes(p)) for p in selected) == 36
assert {p['id'] for p in selected} >= {'dec_train_bs4_ctx8192', 'pre_train_g0'}
assert shapes(next(p for p in selected if p['id'] == 'pre_train_g0')) == [
    (1, 4096), (1, 8192), (1, 16384), (1, 32768)]
passed('full=86/102; pilot=21/36; all five historical smoke shapes included')

for point in selected:
    ranks = fixture_ranks(point)
    result = format_result(point, ranks, metadata(point))
    validate_result(result, point, signature, 4, False)
    if point['kind'] == 'decoder':
        assert all(row['total_ms'] == ranks[0]['points'][0]['samples_ms'][i]
                   for i, row in enumerate(result['rows']))
        assert result['rank_diagnostics']['max_summary']['total_ms_per_step_median'] > result['summaries'][0]['total_ms_per_step_median']
        broken = copy.deepcopy(result)
        broken['rows'][0]['total_ms'] = broken['rows'][0]['rank_total_ms'][-1]
    else:
        assert all(row['samples_ms'] == ranks[0]['points'][i]['samples_ms']
                   for i, row in enumerate(result['points']))
        broken = copy.deepcopy(result)
        broken['points'][0]['samples_ms'][0] += .1
    rejected(lambda: validate_result(broken, point, signature, 4, False))
    rejected(lambda: format_result(point, ranks, {**metadata(point), 'purpose': 'forward_validation'}))
passed('rank0 primary and all-rank maxima diagnostics; altered aggregation and validation attempts rejected')

point = next(p for p in selected if p['id'] == 'pre_train_g0')
partial = destination / 'partial'
first = attempt(partial, point, 'attempt-001', 1, fixture_ranks(point, shapes(point)[:2], completed=False))
recovered, canonical, invalid = read_group(partial, point, signature, 4, False)
assert canonical is None and not invalid and list(recovered) == shapes(point)[:2]
assert missing_shapes(point, recovered) == shapes(point)[2:]
second = attempt(partial, point, 'attempt-002', 2, fixture_ranks(point, shapes(point)[2:], completed=True))
# A faster later duplicate must not replace the first successful observation.
attempt(partial, point, 'attempt-003', 3, fixture_ranks(point, multiplier=.5))
recovered, _, invalid = read_group(partial, point, signature, 4, False)
assert not invalid and [recovered[s]['source'] for s in shapes(point)] == [str(first)] * 2 + [str(second)] * 2
group = assemble_group(point, recovered)
validate_result(group, point, signature, 4, False)
atomic_json(partial / (point['id'] + '.json'), group)
read_group(partial, point, signature, 4, False)
passed('partial prefill resumed by shape; same-attempt ranks; original failures retained; no speed-based selection')

for label, modify in (
        ('incomplete-rank', lambda ranks: ranks.pop()),
        ('unequal-sample-count', lambda ranks: ranks[2]['points'][0]['samples_ms'].pop()),
        ('input-mismatch', lambda ranks: ranks[2]['points'][0]['input'].update(token_sha256='DIFFERENT')),
        ('shape-mismatch', lambda ranks: ranks[2]['points'][0]['layout'].update(actual_sequence_lengths=[1])),
        ('wrong-dtype', lambda ranks: ranks[2]['metadata'].update(weight_dtype='torch.float16'))):
    run = destination / label
    ranks = fixture_ranks(point)
    modify(ranks)
    attempt(run, point, 'attempt-001', 1, ranks)
    recovered, _, _ = read_group(run, point, signature, 4, False)
    assert shapes(point)[0] not in recovered
passed('incomplete ranks/samples, differing inputs, actual layout mismatch and wrong dtype rejected')

validation_run = destination / 'validation-exclusion'
attempt(validation_run, point, 'attempt-001', 1, fixture_ranks(point), purpose='forward_validation')
assert not read_group(validation_run, point, signature, 4, False)[0]
changed = {**specification, 'measurement_policy': {**MEASUREMENT_POLICY, 'primary_rank': 3}}
rejected(lambda: read_group(partial, point, digest_json(changed), 4, False))
passed('validation output and changed measurement signatures cannot be reused')

tree = ast.parse((ROOT / 'calibrate_point.py').read_text())
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('measure_decode', 'measure_prefill'):
        forbidden = ('dist.barrier', 'check_logits', 'check_batch', 'communication_state')
        assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) in forbidden for n in ast.walk(node))
passed('measurement functions contain no TP barrier or numerical checks')

# Populate all pilot fixtures for CPU fitting/export checks. Leave one non-core
# prefill shape missing to exercise the reference fitter's NaN failure rows.
synthetic = destination / 'synthetic_run'
atomic_json(synthetic / 'run_config.json', specification)
atomic_json(synthetic / 'manifest.json', manifest)
last_pref = next(p for p in selected if p['id'] == 'pre_train_g4')
for point in selected:
    requested = shapes(point)[:-1] if point == last_pref else shapes(point)
    ranks = fixture_ranks(point, requested)
    attempt(synthetic, point, 'attempt-001', 1, ranks)
    if point != last_pref:
        atomic_json(synthetic / (point['id'] + '.json'), format_result(point, ranks, metadata(point)))
audit = collect(synthetic, destination / 'collected', [p['id'] for p in selected])
assert not audit['measurement_coverage_complete'] and len(audit['missing_shapes']) == 1
assert len(audit['shape_sources']) == 35
passed('partial collection keeps the full original manifest and explicit missing shape')

# Existing complete results must cause no GPU launch on resume.
saved = sys.argv
try:
    sys.argv = ['run_points.py', str(MANIFEST), str(synthetic), '--model-path', str(MODEL),
                '--only', 'dec_train_bs4_ctx8192', 'pre_train_g0']
    with patch.object(run_points, 'execute', side_effect=AssertionError('Resume launched GPU')), redirect_stdout(io.StringIO()):
        run_points.main()
finally:
    sys.argv = saved
passed('driver skips completed historical-smoke shapes without GPU relaunch')

# If the repeated small smoke fails, the expanded grid must never start.
gate = destination / 'failure_gate'
(gate / 'jobs').mkdir(parents=True)
commands = []


def simulated_process(cmd, log, timeout):
    commands.append(cmd)
    return 1 if log.name == 'bootstrap_measurement.log' else 0


with patch.object(campaign, 'load_campaign', return_value=(gate, {
        'run_signature': signature, 'model_path': str(MODEL), 'pilot_ids': pilot_ids()})), \
        patch.object(campaign, 'execute_process', side_effect=simulated_process), \
        patch.dict('os.environ', {'PBS_JOBID': 'SYNTHETIC-CPU-GATE'}), redirect_stdout(io.StringIO()):
    try:
        campaign.execute(gate, 'pilot')
    except RuntimeError as exc:
        assert 'bootstrap failed' in str(exc)
    else:
        raise AssertionError('Bootstrap failure did not stop the campaign')
assert len([c for c in commands if '--validation-only' in c]) == 2
measure_commands = [c for c in commands if any(str(v).endswith('/run_points.py') for v in c)]
assert len(measure_commands) == 1
assert measure_commands[0][measure_commands[0].index('--only') + 1:] == ['dec_train_bs4_ctx8192', 'pre_train_g0']
assert not json.loads((gate / 'jobs/SYNTHETIC-CPU-GATE/status.json').read_text())['completed']
passed('independent checks precede timing; small-smoke failure prevents expanded measurement')

rc = run_points.execute([sys.executable, '-c', 'import time; time.sleep(5)'], destination / 'timeout.log', 1)
assert rc == 124
passed('timeout terminates its own measurement process group')

atomic_json(destination / 'cpu_checks.json', {'passed': True, 'gpu_execution_tested': False,
    'synthetic_fixtures_only': True, 'checks': checks, 'run_signature': signature,
    'synthetic_collected': str(destination / 'collected')})
print(json.dumps({'passed': True, 'checks': checks, 'directory': str(destination)}, indent=2))
