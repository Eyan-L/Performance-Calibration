"""Prepare Qwen3.6's model config and A100 TP=4 grid from the reference design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration_common import KIT, ROOT, atomic_json, input_coverage

MODEL = Path('/scratch/users/ntu/n2602695/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=MODEL)
    args = parser.parse_args()
    model_path = args.model_path.resolve()
    config = json.loads((model_path / 'config.json').read_text())
    text = config['text_config']
    if (text['model_type'], text['layer_types'].count('full_attention'), text['num_key_value_heads']) != (
            'qwen3_5_moe_text', 10, 2):
        raise ValueError('Expected BF16 Qwen3.6-35B-A3B')
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    missing = [name for name in set(index['weight_map'].values()) if not (model_path / name).is_file()]
    if missing:
        raise ValueError(f'Missing checkpoint shards: {missing}')
    weights = float(index['metadata']['total_size'])
    tp = 4
    kv_bytes = text['layer_types'].count('full_attention') * 2 * text['num_key_value_heads'] * text['head_dim'] * 2
    replication = max(1, tp // text['num_key_value_heads'])
    budget = .75 * (40e9 * tp - weights)
    reference = json.loads((KIT / 'manifests/qwen38bf16_a100x4_flashinfer.json').read_text())
    points, skipped = [], []
    # Include the original skipped decode candidates and apply Qwen3.6's own
    # physical-memory rule before measuring; fit K remains logical attention KV.
    for point in reference['points'] + reference['skipped']:
        point = dict(point)
        if point['kind'] == 'decoder':
            logical = point['bs'] * (point['ctx'] + 64) * kv_bytes
            physical = (point['bs'] * (point['ctx'] + 64) + 32768) * kv_bytes * replication
            point['kv_gb'] = round(logical / 1e9, 2)
            point['physical_kv_with_headroom_gb'] = round(physical / 1e9, 3)
            if physical > budget:
                point['skip_reason'] = 'physical KV including replication and headroom exceeds preregistered budget'
                skipped.append(point)
                continue
        points.append(point)
    # Preserve the reference execution order: train, holdout, prefill, repeat.
    order = {'dec_train': 0, 'dec_holdout': 1, 'pre_train': 2, 'pre_holdout': 3, 'dec_repeat': 4}
    holdout_order = [p['id'] for p in reference['points'] + reference['skipped']
                     if p['id'].startswith('dec_holdout_')]
    points.sort(key=lambda p: (order['_'.join(p['id'].split('_')[:2])],
                              holdout_order.index(p['id']) if p['id'].startswith('dec_holdout_') else p.get('bs', 0),
                              p.get('ctx', 0), p['id']))
    manifest = {'model': 'qwen36bf16', 'gpu': 'a100_40g', 'tp': tp,
                'attention_backend': 'flashinfer', 'kv_budget_gb': budget / 1e9,
                'kv_budget_basis': 'physical KV across all TP ranks, including 32768-token headroom',
                'kv_replication_factor': replication, 'checkpoint_bytes': int(weights),
                'points': points, 'skipped': skipped}
    failures = input_coverage(manifest)
    if failures:
        raise ValueError(f'Reference corpus cannot cover this grid: {failures}')
    config_path = ROOT / 'model_configs/qwen36bf16.config.json'
    manifest_path = ROOT / 'manifests/qwen36bf16_a100x4_flashinfer.json'
    atomic_json(config_path, config)
    atomic_json(manifest_path, manifest)
    print(json.dumps({'model_path': str(model_path), 'model_config': str(config_path),
                      'manifest': str(manifest_path), 'groups': len(points),
                      'shapes': sum(1 if p['kind'] == 'decoder' else len(p['batches'])*len(p['lengths']) for p in points),
                      'skipped_decode': len(skipped), 'kv_replication_factor': replication,
                      'physical_kv_budget_gb': budget / 1e9}, indent=2))


if __name__ == '__main__':
    main()
