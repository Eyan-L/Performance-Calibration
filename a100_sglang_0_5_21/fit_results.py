"""Validate fitting inputs, then run the unchanged reference cost-model fit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

from calibration_common import KIT, atomic_json
from report_results import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collected', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    audit = json.loads((args.collected / 'coverage.json').read_text())
    if audit['smoke']:
        raise ValueError('Smoke timings verify execution; they cannot calibrate the full cost model')
    if not audit['measurement_coverage_complete'] and not args.allow_incomplete:
        raise ValueError('Some registered shapes are missing; inspect coverage.json before --allow-incomplete')
    import numpy as np
    spec = importlib.util.spec_from_file_location('reference_fit', KIT / 'fit' / 'fit_amd_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = KIT / 'inputs' / 'model_configs' / 'qwen38bf16.config.json'
    manifest = json.loads((args.collected / 'manifest.json').read_text())
    if manifest['model'] != 'qwen38bf16':
        raise ValueError('Provide a separate model config before fitting another model')
    decoder = [r for r in module.load_decoder(args.collected, module.kv_gb_per_token(config)) if r['split'] == 'train']
    prefill = [r for r in module.load_prefill(args.collected) if r['split'] == 'train' and r['status'] == 'ok']
    d = np.array([[1, r['bs'], r['K']] for r in decoder], dtype=float).reshape(-1, 3)
    p = np.array([[1, r['batch'] * r['length'], r['batch'] * r['length'] ** 2 / 2 / 1e6]
                  for r in prefill], dtype=float).reshape(-1, 3)
    if len(d) < 3 or len(p) < 4 or np.linalg.matrix_rank(d) != 3 or np.linalg.matrix_rank(p) != 3:
        raise ValueError('Insufficient independent training shapes for the three model coefficients')
    folds = np.array([(int(math.log2(r['bs'])) + int(math.log2(r['ctx']))) % 5 for r in decoder])
    if any(len(d[folds != fold]) < 3 or np.linalg.matrix_rank(d[folds != fold]) != 3 for fold in set(folds)):
        raise ValueError('Decoder cross-validation training fold has insufficient independent shapes')
    if any(np.linalg.matrix_rank(np.delete(p, i, axis=0)) != 3 for i in range(len(p))):
        raise ValueError('Prefill leave-one-out training fold has insufficient independent shapes')
    sys.argv = [str(spec.origin), str(args.collected), '--manifest', str(args.collected / 'manifest.json'),
                '--model-config', str(config), '--out', str(args.out)]
    module.main()
    atomic_json(args.out / 'fit_validation.json', {
        'training_design_verified': True, 'decoder_training_rows': len(d), 'prefill_training_rows': len(p),
        'decoder_cv_training_folds_rank3': True, 'prefill_loo_training_folds_rank3': True,
        'scope': 'complete' if audit.get('full_grid_coverage_complete') else 'pilot_or_incomplete',
        'run_signature': audit['run_signature']})
    render_report(args.collected, args.out)


if __name__ == '__main__':
    main()
