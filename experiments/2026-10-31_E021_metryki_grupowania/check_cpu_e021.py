"""E021: kontrole CPU benchmarku metryk (bez GPU i bez dev).

Eksperyment: 2026-10-31_E021_metryki_grupowania; pełny protokół: OPIS.md.
Sprawdza zgodność reference z operatorem E012/E014, determinizm i równy
budżet wszystkich metryk na jednym checkpoincie. Nie ładuje dev.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e021_common as common  # noqa: E402
import grouping_e021  # noqa: E402

WORKER = common.base.load_module(HERE / 'worker.py', '_e021_check_worker')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def state_equal(a, b):
    left, right = a.state_dict(), b.state_dict()
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left)


def main():
    cfg = common.load_config()
    require(len(common.build_tasks(cfg)) == 72, 'Zła liczba zadań')
    entry = common.dataset_entry(cfg, 'cifar10_s42')
    bundle = common.load_bundle(entry)
    folding = common.folding_module()
    nominal = int(round(0.5 * bundle['params']))
    solved = common.folding_budget.solve_base_for_budget(
        bundle['model'], 'uniform', nominal)
    ratios = [float(value) for value in solved['ratios']]

    task = {'dataset': 'cifar10_s42', 'kind': 'official_weights',
            'budget': 0.5}
    model, groups = WORKER.fold_with_metric(bundle['model'], ratios, task)
    official, diag = common.folding_budget.fold_with_ratios(
        bundle['model'], ratios, seed=42, deterministic=False,
        preserve_fc_bias=True)
    require(state_equal(model, official),
            'Reference nie odtwarza operatora E012/E014')
    require([group['label_digest'] for group in groups]
            == [common.base.labels_digest(item['cluster_labels'])
                for item in diag['groups']],
            'Reference nie odtwarza etykiet E012')
    expected = common.folding_budget.parameter_count_for_ratios(
        bundle['model'], ratios)
    require(folding.count_parameters(model) == expected,
            'Zła liczba parametrów reference')

    kinds = ['official_weights', 'agg_euclidean', 'agg_manhattan',
             'agg_cosine', 'agg_correlation', 'ward_euclidean',
             'standardized', 'random_balanced']
    params = {}
    for kind in kinds:
        current_task = dict(task, kind=kind)
        folded, _ = WORKER.fold_with_metric(
            copy.deepcopy(bundle['model']), ratios, current_task)
        value = folding.count_parameters(folded)
        params[kind] = value
        require(value == expected,
                f'{kind}: {value} != {expected}')
        require(folding.state_dict_all_finite(folded),
                f'{kind}: niefinitywne parametry')
    again, again_groups = WORKER.fold_with_metric(
        copy.deepcopy(bundle['model']), ratios, dict(task, kind='agg_manhattan'))
    first, first_groups = WORKER.fold_with_metric(
        copy.deepcopy(bundle['model']), ratios, dict(task, kind='agg_manhattan'))
    require(state_equal(first, again), 'Brak determinizmu agg_l1')
    require([g['label_digest'] for g in first_groups]
            == [g['label_digest'] for g in again_groups],
            'Brak determinizmu etykiet agg_l1')

    summary = {
        'status': 'passed', 'params': params,
        'reference_matches_e012_e014': True,
        'deterministic_agg_l1': True,
        'dev_loaded': False, 'gpu_used': False, 'tasks': 72,
    }
    common.write_json_atomic(HERE / 'TESTY_WYNIK.json', summary)
    print('E021_CPU_CHECKS=passed', flush=True)
    print(f'E021_PARAMS={params}', flush=True)


if __name__ == '__main__':
    main()
