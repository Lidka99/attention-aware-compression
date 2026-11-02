"""E020: preflight CPU bez GPU dla protokołu mechanizmu alokacji.

Eksperyment: 2026-10-30_E020_mechanizm_alokacji; pełny opis: OPIS.md.
Sprawdza: statusy configów, hashe checkpointów/splitów, kompletność list
itemów, istnienie referencji UNIFORM, import operatorów E012/E014, liczbę
parametrów modeli oraz solver ANTI-TILT (bez BN i bez oceny dev).
Nie uruchamia obliczeń GPU i nie modyfikuje wyników.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e020_common as common  # noqa: E402


def check_inventory():
    inventory = common.load_inventory()
    common.require(inventory.get('status') == 'frozen', 'Inwentarz frozen')
    for dataset, meta in inventory['datasets'].items():
        split = Path(meta['split'])
        common.require(split.is_file(), f'Brak splitu {split}')
        common.require(common.digest(split) == meta['split_sha256'],
                       f'Zły hash splitu {dataset}')
        for seed, entry in meta['seeds'].items():
            checkpoint = Path(entry['checkpoint'])
            common.require(checkpoint.is_file(),
                           f'Brak checkpointu {dataset}/{seed}')
            common.require(
                common.digest(checkpoint) == entry['checkpoint_sha256'],
                f'Zły hash checkpointu {dataset}/{seed}')
            common.require(
                0.0 < float(entry['baseline_top1']) <= 1.0,
                f'Zły baseline top1 {dataset}/{seed}')
    print(f'Inwentarz OK: {len(inventory["datasets"])} zbiory, 4 seedy każdy')
    return inventory


def check_sensitivity(inventory):
    cfg = common.load_sensitivity_config()
    meta = inventory['datasets'][cfg['dataset']]
    common.require(cfg['dataset'] == 'tiny', 'Sensitivity dotyczy Tiny')
    common.require(
        cfg['checkpoint_sha256']
        == meta['seeds'][str(cfg['seed'])]['checkpoint_sha256'],
        'Checkpoint sensitivity nie zgadza się z inwentarzem')
    common.require(cfg['split_sha256'] == meta['split_sha256'],
                   'Split sensitivity nie zgadza się z inwentarzem')
    expected = 1 + len(common.GROUPS) * len(cfg['levels'])
    common.require(expected == 49, 'Oczekiwano 49 przebiegów S1 Tiny')
    print(f'Sensitivity OK: {expected} itemów, seed {cfg["seed"]}')


def check_ablation(inventory):
    cfg = common.load_ablation_config()
    expected = len(cfg['datasets']) * len(cfg['seeds']) * len(cfg['budgets'])
    common.require(expected == 36, 'Oczekiwano 36 przebiegów ablacji')
    for dataset in cfg['datasets']:
        common.require(dataset in inventory['datasets'],
                       f'Nieznany zbiór {dataset}')
        run = common.ROOT / cfg['reference_fold_runs'][dataset]
        common.require(run.is_dir(), f'Brak referencyjnego runu {run}')
        for seed in cfg['seeds']:
            for budget_value in cfg['budgets']:
                tag = f"b{int(round(float(budget_value) * 100))}_uniform"
                if dataset == 'cifar10':
                    path = (run / 'seeds' / f'seed_{seed}' / 'allocations'
                            / tag / 'summary.json')
                else:
                    path = (run / 'allocations' / f'seed_{seed}' / tag
                            / 'summary.json')
                common.require(path.is_file(), f'Brak referencji {path}')
                reference = common.read_json(path)
                common.require(reference.get('status') == 'completed',
                               f'Referencja niecompleted {path}')
    print(f'Ablation OK: {expected} itemów, referencje UNIFORM istnieją')


def check_models_and_solver(inventory):
    budget = common.folding_budget()
    for dataset in ('cifar10', 'cifar100', 'tiny'):
        model, entry = common.load_checkpoint(dataset, 42, inventory)
        params = common.count_parameters(model)
        common.require(params == int(inventory['datasets'][dataset]['params']),
                       f'Zła liczba parametrów {dataset}')
        nominal = int(round((1.0 - 0.5) * params))
        uniform = budget.solve_base_for_budget(model, 'uniform', nominal)
        solved = budget.solve_base_for_budget(
            model, 'tilt', int(uniform['params']), tilt=-0.30)
        repaired = budget.repair_budget(
            model, solved['ratios'], int(uniform['params']))
        common.require(int(repaired['params']) <= int(uniform['params']),
                       f'ANTI-TILT przekracza target dla {dataset}')
        common.require(
            int(repaired['params']) >= 0.99 * int(uniform['params']),
            f'ANTI-TILT zbyt daleko od targetu dla {dataset}')
        print(f'Solver {dataset}: baseline={params}, uniform={uniform["params"]}, '
              f'antitilt={repaired["params"]}, '
              f'ratios={[round(v, 3) for v in repaired["ratios"]]}')
        del model


def main():
    inventory = check_inventory()
    check_sensitivity(inventory)
    check_ablation(inventory)
    check_models_and_solver(inventory)
    print('PREFLIGHT CPU: OK')


if __name__ == '__main__':
    main()
