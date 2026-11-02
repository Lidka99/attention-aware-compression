"""E020: launcher ablacji ANTI-TILT na trzech zbiorach.

Eksperyment: 2026-10-30_E020_mechanizm_alokacji; pełny opis: OPIS.md.
Wejście: zamrożony ``configs/ablation.json`` i inwentarz. Wyjście:
unikalny ``runs/E020/<UTC>_<git>_abl_<token>/`` z 36 przebiegami
(3 zbiory x 4 seedy x 3 budżety), cztery GPU, logi i manifest.
Bez treningu wag; UNIFORM i TILT+0.30 pochodzą z istniejących runów.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e020_common as common  # noqa: E402


def build_items(cfg):
    """Buduje 36 itemów ablacji ANTI-TILT."""
    items = []
    for dataset in cfg['datasets']:
        for seed in cfg['seeds']:
            for budget_value in cfg['budgets']:
                tag = f"b{int(round(float(budget_value) * 100))}"
                items.append({
                    'name': (f"{dataset}_seed{int(seed)}_{tag}_"
                             f"{cfg['variant']['name']}"),
                    'dataset': dataset,
                    'seed': int(seed),
                    'budget': float(budget_value),
                })
    expected = (len(cfg['datasets']) * len(cfg['seeds'])
                * len(cfg['budgets']))
    common.require(len(items) == expected,
                   f'Oczekiwano {expected} itemów, jest {len(items)}')
    common.require(len({(i['dataset'], i['seed'], i['budget'])
                        for i in items}) == len(items),
                   'Itemy nie są unikalne')
    return items


def reference_path(dataset, seed, budget_value):
    """Ścieżka istniejącego summary UNIFORM."""
    cfg = common.load_ablation_config()
    run = common.ROOT / cfg['reference_fold_runs'][dataset]
    tag = f"b{int(round(float(budget_value) * 100))}_uniform"
    if dataset == 'cifar10':
        return run / 'seeds' / f'seed_{seed}' / 'allocations' / tag / 'summary.json'
    return run / 'allocations' / f'seed_{seed}' / tag / 'summary.json'


def main():
    cfg = common.load_ablation_config()
    inventory = common.load_inventory()
    for dataset in cfg['datasets']:
        common.require(dataset in inventory['datasets'],
                       f'Nieznany zbiór: {dataset}')
    info = common.preflight(gpu_count=cfg['gpu_count'], min_disk_gib=1.5)

    items = build_items(cfg)
    for item in items:
        path = reference_path(item['dataset'], item['seed'], item['budget'])
        common.require(path.is_file(),
                       f'Brak referencji UNIFORM: {path}')
        reference = common.read_json(path)
        common.require(reference.get('status') == 'completed',
                       f'Referencja nie jest completed: {path}')

    run_dir = common.create_run_dir(
        'abl', secrets.token_hex(4),
        sources={
            'e020_common.py': HERE / 'e020_common.py',
            'ablation_worker.py': HERE / 'ablation_worker.py',
            'config_ablation.json': HERE / 'configs' / 'ablation.json',
            'inventory.json': HERE / 'configs' / 'inventory.json',
        })
    config_copy = run_dir / 'config.json'
    config_copy.write_text(
        (HERE / 'configs' / 'ablation.json').read_text(encoding='utf-8'),
        encoding='utf-8')
    assignments = common.shard_items(items, cfg['gpu_count'])
    common.write_manifest(run_dir, cfg, items, {
        str(gpu): [item['name'] for item in shard]
        for gpu, shard in assignments.items()})
    common.write_run_environment(run_dir, info)
    print(f'Run: {run_dir}')
    print(f'Items: {len(items)}; GPU: {cfg["gpu_count"]}; '
          f'wolne: {info["free_gib"]:.2f} GiB')

    codes = common.launch_shards(
        run_dir, HERE / 'ablation_worker.py', config_copy, items,
        gpu_count=cfg['gpu_count'], timeout_seconds=5400)

    results = {}
    for item in items:
        path = (run_dir / 'allocations' / item['dataset']
                / f"seed_{item['seed']}" / item['name'] / 'summary.json')
        if path.is_file():
            results[item['name']] = common.read_json(path).get('status')
        else:
            results[item['name']] = 'missing'
    counts = {}
    for status in results.values():
        counts[status] = counts.get(status, 0) + 1
    completed = counts.get('completed', 0)
    status = 'completed' if completed == len(items) else 'failed'
    common.write_json_atomic(run_dir / 'summary.json', {
        'experiment': common.EXPERIMENT,
        'kind': 'ablation_antitilt',
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
