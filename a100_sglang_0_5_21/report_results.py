"""Render measurements and cost-model errors without altering the reference fit."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from calibration_common import atomic_json


def render_report(collected, fitted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    collected, fitted = Path(collected), Path(fitted)
    model = json.loads((fitted / 'model.json').read_text())
    audit = json.loads((collected / 'coverage.json').read_text())
    groups = []
    for phase in ('decoder', 'prefill'):
        with (fitted / (phase + '_points.csv')).open() as f:
            rows = list(csv.DictReader(f))
        used = []
        for row in rows:
            if phase == 'prefill' and row['status'] != 'ok':
                continue
            measured = float(row['step_ms'] if phase == 'decoder' else row['ms'])
            predicted = float(row['pred_ms'])
            if not math.isfinite(measured) or measured <= 0 or not math.isfinite(predicted):
                raise ValueError('Fit exported an invalid measurement or prediction')
            used.append({**row, 'measured': measured, 'predicted': predicted,
                         'ape_percent': abs(predicted - measured) / measured * 100})
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for split, marker in (('train', 'o'), ('holdout', '^'), ('repeat', 's')):
            selected = [r for r in used if r['split'] == split]
            if selected:
                axes[0].scatter([r['measured'] for r in selected], [r['predicted'] for r in selected],
                                label=split, marker=marker, s=28)
        limits = [min(r['measured'] for r in used), max(r['measured'] for r in used)]
        axes[0].plot(limits, limits, color='grey', linewidth=1)
        axes[0].set(xlabel='Measured GPU time (ms)', ylabel='Predicted GPU time (ms)', title=phase.title())
        axes[0].legend()
        train = [r for r in used if r['split'] == 'train']
        bkey, lkey = ('bs', 'ctx') if phase == 'decoder' else ('batch', 'length')
        batches = sorted({int(r[bkey]) for r in train})
        lengths = sorted({int(r[lkey]) for r in train})
        errors = np.full((len(batches), len(lengths)), np.nan)
        for row in train:
            errors[batches.index(int(row[bkey])), lengths.index(int(row[lkey]))] = row['ape_percent']
        shown = axes[1].imshow(errors, aspect='auto', cmap='YlOrRd', vmin=0)
        axes[1].set_xticks(range(len(lengths)), [f'{l / 1024:g}K' for l in lengths], rotation=45)
        axes[1].set_yticks(range(len(batches)), batches)
        axes[1].set(xlabel='Context / input length', ylabel='Batch', title='Training-point absolute error (%)')
        figure.colorbar(shown, ax=axes[1], label='Absolute percentage error')
        figure.savefig(fitted / (phase + '_errors.png'), dpi=160)
        figure.savefig(fitted / (phase + '_errors.svg'))
        plt.close(figure)
        groups.append((phase, used))
    partial = not audit.get('full_grid_coverage_complete', False)
    model['calibration_scope'] = 'pilot / incomplete registered grid' if partial else 'complete registered grid'
    model['selected_groups'] = audit['selected']
    model['validated_shape_sources'] = audit.get('shape_sources', [])
    # The reference fitter uses NaN for unavailable failed prefill timings.
    # Preserve the failure row but represent missing data as JSON null. Valid
    # measurements/predictions were checked above and are never transformed.
    for row in model['prefill'].get('failed_points', []):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
    specification = json.loads((collected / 'run_config.json').read_text())
    model['measurement_policy'] = specification['measurement_policy']
    communication = 'native communication selection'
    if 'communication_policy' in specification:
        model['communication_policy'] = specification['communication_policy']
        if specification['communication_policy'].get('disable_custom_all_reduce'):
            communication = 'custom all-reduce explicitly disabled'
    if 'kv_pool_policy' in specification:
        model['kv_pool_policy'] = specification['kv_pool_policy']
    atomic_json(fitted / 'model.json', model)
    label = 'Qwen3.6-35B-A3B' if model['model'] == 'qwen36bf16' else 'Qwen3.8-27B'
    lines = [f'# {label} A100 TP={specification["tp"]} calibration', '',
             '**Scope:** ' + model['calibration_scope'], '',
             f'SGLang 0.5.21; BF16 weights/KV; FlashInfer; {communication}.', '',
             'Times cover GPU forward only. Loading, compilation, input preparation and sampling are excluded.', '',
             'Primary timings use rank 0. All-rank samples and per-iteration maxima are diagnostics only.', '',
             'No added per-sample TP barrier. Numerical forward checks run in separate validation processes.', '',
             'TP=4 prefill is an extension of the reference TP=1 measurement.', '',
             'Training data determine coefficients; holdout and repeat data do not enter fitting.', '',
             '| Phase | Evaluation | N | MAPE (%) | P95 error (%) | Max error (%) |',
             '|---|---|---:|---:|---:|---:|']
    if 'kv_pool_policy' in specification:
        position = lines.index('| Phase | Evaluation | N | MAPE (%) | P95 error (%) | Max error (%) |')
        lines[position:position] = [
            'KV pool cap: maximum batch × context in the original group + 32768 tokens.', '',
            'Per-shape source signatures and job IDs identify any reused decode measurements; '
            'prefill measurements use this campaign\'s capped pool.', '']
    for phase, validation in (('decoder', 'cv_checkerboard5'), ('prefill', 'cv_leave_one_out')):
        for name in ('train', validation, 'holdout'):
            item = model[phase][name]
            lines.append(f'| {phase} | {name} | {item["n"]} | {item.get("mape_percent", float("nan")):.2f} | '
                         f'{item.get("p95_ape_percent", float("nan")):.2f} | {item.get("max_ape_percent", float("nan")):.2f} |')
    lines += ['', '## Coefficients', '', '```json',
              json.dumps({p: model[p]['coefficients'] for p in ('decoder', 'prefill')}, indent=2), '```', '',
              '## Accuracy by measured region', '']
    for phase, rows in groups:
        worst = sorted(rows, key=lambda r: r['ape_percent'], reverse=True)[:5]
        lines.append(f'### {phase.title()}: largest observed point errors')
        lines.append('')
        for row in worst:
            batch, length = (row['bs'], row['ctx']) if phase == 'decoder' else (row['batch'], row['length'])
            lines.append(f'- {row["split"]}: batch={batch}, length={length}, '
                         f'measured={row["measured"]:.3f} ms, predicted={row["predicted"]:.3f} ms, '
                         f'error={row["ape_percent"]:.2f}%.')
        lines += ['', f'![{phase} errors]({phase}_errors.png)', '']
    lines += ['## Repeat stability', '', '```json',
              json.dumps(model['decoder']['repeat_stability'], indent=2), '```', '',
              '## Coverage', '',
              f'Validated shapes: {len(audit.get("shape_sources", []))}/{audit["selected_shape_count"]} selected.',
              'Missing or unselected points are not automatically classified as OOM.', '', '```json',
              json.dumps({'missing_selected_shapes': audit['missing_shapes'],
                          'registered_groups_without_file': model['missing_preregistered_points'],
                          'skipped_by_original_memory_rule': model['skipped_by_rule']}, indent=2), '```', '']
    (fitted / 'report.md').write_text('\n'.join(lines))
