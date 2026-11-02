"""E020: launcher pomiaru wrażliwości pojedynczych grup na Tiny ImageNet.

Eksperyment: 2026-10-30_E020_mechanizm_alokacji; pełny opis: OPIS.md.
Wejście: zamrożony ``configs/sensitivity.json`` i inwentarz. Wyjście:
unikalny ``runs/E020/<UTC>_<git>_sens_<token>/`` z manifestem, snapshotami,
przypisaniem itemów do czterech GPU, logami i wynikami 49 przebiegów.
Launcher nie trenuje wag i wymaga czterech wolnych GPU.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e020_common as common  # noqa: E402


def build_items(cfg):
    """Buduje 49 itemów: kontrola + 12 grup x 4 poziomy."""
    items = [{
        'name': f"{cfg['dataset']}__control__r000",
        'dataset': cfg['dataset'],
        'group_index': None,
        'group_name': None,
        'level': 0.0,
        'control': True,
    }]
    for group_index, group_name in enumerate(common.GROUPS):
        for level in cfg['levels']:
            items.append({
                'name': (f"{cfg['dataset']}__g{group_index:02d}_"
                         f"{group_name.replace('.', '_')}__"
                         f"r{int(round(float(level) * 100)):03d}"),
                'dataset': cfg['dataset'],
                'group_index': int(group_index),
                'group_name': group_name,
                'level': float(level),
                'control': False,
            })
    expected = 1 + len(common.GROUPS) * len(cfg['levels'])
    common.require(len(items) == expected,
                   f'Oczekiwano {expected} itemów, jest {len(items)}')
    common.require(len({item['name'] for item in items}) == len(items),
                   'Nazwy itemów nie są unikalne')
    return items


def main():
    cfg = common.load_sensitivity_config()
    inventory = common.load_inventory()
    meta = inventory['datasets'][cfg['dataset']]
    common.require(cfg['dataset'] == 'tiny', 'Ten launcher obsługuje Tiny')
    common.require(meta['seeds'][str(cfg['seed'])]['checkpoint_sha256']
                   == cfg['checkpoint_sha256'],
                   'Checkpoint w configu nie zgadza się z inwentarzem')
    common.require(meta['split_sha256'] == cfg['split_sha256'],
                   'Split w configu nie zgadza się z inwentarzem')

    info = common.preflight(gpu_count=cfg['gpu_count'], min_disk_gib=1.5)
    run_dir = common.create_run_dir(
        'sens', secrets.token_hex(4),
        sources={
            'e020_common.py': HERE / 'e020_common.py',
            'sensitivity_worker.py': HERE / 'sensitivity_worker.py',
            'config_sensitivity.json': HERE / 'configs' / 'sensitivity.json',
            'inventory.json': HERE / 'configs' / 'inventory.json',
        })
    config_copy = run_dir / 'config.json'
    config_copy.write_text(
        (HERE / 'configs' / 'sensitivity.json').read_text(encoding='utf-8'),
        encoding='utf-8')
    items = build_items(cfg)
    assignments = common.shard_items(items, cfg['gpu_count'])
    common.write_manifest(run_dir, cfg, items, {
        str(gpu): [item['name'] for item in shard]
        for gpu, shard in assignments.items()})
    common.write_run_environment(run_dir, info)
    print(f'Run: {run_dir}')
    print(f'Items: {len(items)}; GPU: {cfg["gpu_count"]}; '
          f'wolne: {info["free_gib"]:.2f} GiB')

    codes = common.launch_shards(
        run_dir, HERE / 'sensitivity_worker.py', config_copy, items,
        gpu_count=cfg['gpu_count'], timeout_seconds=5400)

    results = {}
    for item in items:
        path = run_dir / 'items' / item['name'] / 'summary.json'
        if path.is_file():
            summary = common.read_json(path)
            results[item['name']] = summary.get('status')
        else:
            results[item['name']] = 'missing'
    counts = {}
    for status in results.values():
        counts[status] = counts.get(status, 0) + 1
    completed = counts.get('completed', 0)
    status = 'completed' if completed == len(items) else 'failed'
    common.write_json_atomic(run_dir / 'summary.json', {
        'experiment': common.EXPERIMENT,
        'kind': 'sensitivity_tiny',
        'status': status,
        'items_total': len(items),
        'items_completed': completed,
        'status_counts': counts,
        'exit_codes': {str(gpu): int(code) for gpu, code in codes.items()},
        'finished_utc': common.utc_now(),
        'official_test_used': False,
        'reserve_used': False,
    })
    common.finish_manifest(run_dir, status, {
        'items_total': len(items),
        'items_completed': completed,
        'status_counts': counts,
        'exit_codes': {str(gpu): int(code) for gpu, code in codes.items()},
    })
    print(f'Status: {status}; completed: {completed}/{len(items)}')
    if status != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
