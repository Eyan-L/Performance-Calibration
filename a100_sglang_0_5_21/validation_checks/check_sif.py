"""Inspect native APIs and CPU-only fitting outputs inside the actual SIF."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration_common import MEASUREMENT_POLICY, atomic_json
from calibrate_point import check_logits, main as check_api
from runtime_guard import check_runtime

fitted = Path(sys.argv[1])
sys.argv = ['calibrate_point.py', '--check-api']
check_api()


def reject_constant(value):
    raise ValueError('Non-standard JSON number: ' + value)


model = json.loads((fitted / 'model.json').read_text(), parse_constant=reject_constant)
assert model['measurement_policy'] == MEASUREMENT_POLICY
assert model['calibration_scope'] == 'pilot / incomplete registered grid'
assert len(model['prefill']['failed_points']) == 1
assert model['prefill']['failed_points'][0]['ms'] is None
assert math.isclose(model['decoder']['coefficients']['t0_ms'], 2, abs_tol=1e-10)
assert math.isclose(model['prefill']['coefficients']['pN_ms_per_token'], .0123456789, abs_tol=1e-10)
for name in ('report.md', 'decoder_errors.png', 'prefill_errors.png', 'decoder_errors.svg', 'prefill_errors.svg'):
    assert (fitted / name).stat().st_size > 0
assert 'rank 0' in (fitted / 'report.md').read_text()
coverage = json.loads((fitted.parent / 'collected/coverage.json').read_text())
assert len(coverage['shape_sources']) == 35

# Exercise numerical guards on CPU tensors; this is not GPU acceptance.
import torch
assert check_logits(torch.zeros((4, 16)), 4, 8192, True)['finite_logits']
for value in (torch.full((4, 16), float('nan')), torch.zeros((3, 16))):
    try:
        check_logits(value, 4, 8192, True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Invalid logits were accepted')
report = {'passed': True, 'gpu_execution_tested': False, 'synthetic_fixtures_only': True,
          'runtime': check_runtime(), 'native_api_and_tp_parser': True,
          'independent_validation_api': True, 'fitting_known_coefficients': True,
          'missing_prefill_json_null': True, 'csv_and_figures': True,
          'numerical_guard_cpu_check': True}
atomic_json(fitted.parent / 'sif_cpu_checks.json', report)
print(json.dumps(report, indent=2))
