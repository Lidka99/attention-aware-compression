"""E018: kontrole CPU przed replikacją cosine.

Eksperyment: 2026-10-29_E018_replikacja_cosine; pełny protokół: OPIS.md.
Sprawdza konfigurację, determinizm i równy budżet reference vs cosine
na seedzie 43. Nie ładuje dev i nie używa GPU.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e018_common as common  # noqa: E402

E017 = common.E017
if str(E017) not in sys.path:
    sys.path.insert(0, str(E017))

import e017_common as base  # noqa: E402

E017_WORKER = base.load_module(E017 / 'worker.py', '_e018_cpu_worker')


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def state_equal(first, second):
    """Porównuje dokładnie wszystkie tensory state_dict."""
    left = first.state_dict()
    right = second.state_dict()
    if set(left) != set(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def main():
    cfg = common.load_config()
    tasks = common.build_tasks(cfg)
    require(len(tasks) == 24, 'Zła liczba zadań')
    require([task['seed'] for task in tasks if task['dataset'] == 'cifar10_s43'],
            'Brak zadań dla cifar10_s43')
    entry = common.dataset_entry(cfg, 'cifar10_s43')
    bundle = common.load_bundle(entry)
    fn = common.folding_module()
    budget = 0.5

    def task_for(kind):
        return {
            'dataset': 'cifar10_s43', 'kind': kind, 'replicate': 0,
            'budget': budget, 'preserve_fc_bias': True, 'fold_seed': 42,
            'measure': kind,
        }

    started = time.monotonic()
    reference, reference_groups = E017_WORKER.fold_with_measure(
        copy.deepcopy(bundle['model']),
        task_for('official_weights'), fn)
    again, again_groups = E017_WORKER.fold_with_measure(
        copy.deepcopy(bundle['model']),
        task_for('official_weights'), fn)
    require(state_equal(reference, again), 'Brak determinizmu reference')
    require([group['label_digest'] for group in reference_groups]
            == [group['label_digest'] for group in again_groups],
            'Brak determinizmu etykiet')
    cosine, cosine_groups = E017_WORKER.fold_with_measure(
        copy.deepcopy(bundle['model']),
        task_for('cosine_weights'), fn)
    reference_params = fn.count_parameters(reference)
    cosine_params = fn.count_parameters(cosine)
    require(reference_params == cosine_params,
            f'Różne budżety: {reference_params} != {cosine_params}')
    require(fn.state_dict_all_finite(cosine),
            'Niefinitywne parametry cosine')
    require(len(reference_groups) == len(cosine_groups) == 12,
            'Oczekiwano 12 grup')
    for group in reference_groups + cosine_groups:
        require(sum(group['cluster_counts']) == group['n_channels'],
                f"{group['group']}: złe rozmiary klastrów")
        require(len(group['cluster_counts']) == group['n_clusters'],
                f"{group['group']}: zła liczba klastrów")

    summary = {
        'status': 'passed',
        'deterministic': True,
        'params': {'reference': reference_params, 'cosine': cosine_params},
        'cpu_seconds': time.monotonic() - started,
        'dev_loaded': False,
        'gpu_used': False,
        'tasks': len(tasks),
    }
    common.write_json_atomic(HERE / 'TESTY_WYNIK.json', summary)
    print('E018_CPU_CHECKS=passed', flush=True)
    print(f'E018_PARAMS={summary["params"]}', flush=True)


if __name__ == '__main__':
    main()
