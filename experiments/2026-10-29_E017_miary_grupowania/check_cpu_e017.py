"""E017: kontrole poprawności na CPU przed uruchomieniem na GPU.

Eksperyment: 2026-10-29_E017_miary_grupowania; pełny protokół: OPIS.md.
Sprawdza równoważność ``reference`` z E012, determinizm, równy budżet
wszystkich miar i kontrolę losową. Nie ładuje dev i nie używa GPU.
"""
from __future__ import annotations

import copy
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e017_common as common  # noqa: E402
import grouping  # noqa: E402
import worker  # noqa: E402


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
    require(len(tasks) == 42, 'Zła liczba zadań')
    entry = next(item for item in cfg['datasets']
                 if item['name'] == 'cifar10')
    bundle = common.load_dataset(entry)
    fn = common.folding_module()
    budget = 0.5

    def task_for(kind, replicate=0):
        return {
            'dataset': 'cifar10', 'kind': kind, 'replicate': replicate,
            'budget': budget, 'preserve_fc_bias': True, 'fold_seed': 42,
            'measure': kind,
        }

    started = time.monotonic()
    reference, reference_groups = worker.fold_with_measure(
        bundle['model'], task_for('official_weights'), fn)
    official, official_diag = fn.fold(
        bundle['model'], budget, seed=42, preserve_fc_bias=True)
    require(state_equal(reference, official),
            'Reference nie odtwarza modelu E012')
    require([group['label_digest'] for group in reference_groups]
            == [common.labels_digest(item['cluster_labels'])
                for item in official_diag['groups']],
            'Reference nie odtwarza etykiet E012')
    reference_params = fn.count_parameters(reference)

    again, again_groups = worker.fold_with_measure(
        bundle['model'], task_for('official_weights'), fn)
    require(state_equal(reference, again), 'Brak determinizmu reference')
    require([group['label_digest'] for group in again_groups]
            == [group['label_digest'] for group in reference_groups],
            'Brak determinizmu etykiet')

    results = {'reference': reference_params}
    for measure in cfg['measures'][1:]:
        model, groups = worker.fold_with_measure(
            copy.deepcopy(bundle['model']),
            task_for(measure['kind'], int(measure.get('replicate', 0))), fn)
        params = fn.count_parameters(model)
        results[measure['name']] = params
        require(params == reference_params,
                f"{measure['name']}: inna liczba parametrów {params} "
                f"!= {reference_params}")
        require(fn.state_dict_all_finite(model),
                f"{measure['name']}: niefinitywne parametry")
        for group in groups:
            require(sum(group['cluster_counts']) == group['n_channels'],
                    f"{measure['name']}/{group['group']}: złe rozmiary")
            require(len(group['cluster_counts']) == group['n_clusters'],
                    f"{measure['name']}/{group['group']}: zła liczba klastrów")
        if measure['kind'] == 'bn_effective_weights':
            require(any(group['used_bn_scale'] for group in groups),
                    'bn_effective nie użył skali BN')

    features = torch.randn(64, 8, generator=torch.Generator().manual_seed(7))
    reference_labels = grouping._label_from_clustering(features, 21, 64)
    random_labels = torch.empty(64, dtype=torch.int64)
    permutation = torch.randperm(
        64, generator=torch.Generator().manual_seed(9))
    random_labels[permutation] = reference_labels
    require(sorted(torch.bincount(random_labels).tolist())
            == sorted(torch.bincount(reference_labels).tolist()),
            'Kontrola losowa nie zachowuje rozmiarów klastrów')

    summary = {
        'status': 'passed',
        'reference_matches_e012': True,
        'deterministic': True,
        'params': results,
        'cpu_seconds': time.monotonic() - started,
        'dev_loaded': False,
        'gpu_used': False,
        'tasks': len(tasks),
    }
    common.write_json_atomic(HERE / 'TESTY_WYNIK.json', summary)
    print('E017_CPU_CHECKS=passed', flush=True)
    print(f"E017_PARAMS={results}", flush=True)


if __name__ == '__main__':
    main()
