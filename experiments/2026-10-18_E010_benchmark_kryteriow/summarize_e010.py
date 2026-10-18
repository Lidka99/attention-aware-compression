"""E010: podsumowanie, sparowany bootstrap i decyzja pre-registered.

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: merge/dev/*.npz (zero-shot) i repair/*/dev_final.npz (T epok).
Wyjscia: summary.json z metrykami, porownaniami vs RANDOM, pierwszorzednym
testem (Tiny 30%, T=5, Holm) i replikacja (CIFAR-100 10%, Tiny 10%).
Bootstrap: dokladny (zgodny z E003) dla primary/replikacji; szybki wektorowy
dla porownan eksploracyjnych. Bez GPU.
"""
import argparse
from pathlib import Path

import numpy as np

from support_e010 import (dataset_cfg, load_config, read_json, write_json, require,
                          DATASET_KEYS)


def logsumexp_rows(logits):
    maximum = logits.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))


def cell_metrics(path):
    with np.load(path, allow_pickle=False) as f:
        ids, targets, logits = f['ids'], f['targets'], f['logits']
    require(logits.shape[0] == len(ids) == len(targets), f'Shape mismatch: {path}')
    maximum = logits.max(axis=1, keepdims=True)
    lse = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    nll = float(np.mean(lse - logits[np.arange(len(targets)), targets]))
    correct = logits.argmax(axis=1) == targets
    return {'ids': ids, 'targets': targets, 'correct': correct,
            'top1': float(correct.mean()), 'nll': nll, 'count': int(len(targets))}


def _groups(targets):
    labels = np.asarray(targets).ravel()
    groups = [np.flatnonzero(labels == cls) for cls in np.unique(labels)]
    require(len({len(g) for g in groups}) == 1, 'Unequal class counts')
    return groups


def bootstrap_exact(pre, post, targets, reps, seed):
    """Sparowany bootstrap jak w E003 paired_bootstrap (ten sam uklad losowan)."""
    pre = np.asarray(pre).astype(bool).ravel()
    post = np.asarray(post).astype(bool).ravel()
    groups = _groups(targets)
    total = len(pre)
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(int(reps), dtype=np.float64)
    for repetition in range(int(reps)):
        difference = 0.0
        for group in groups:
            draw = group[rng.integers(0, group.size, group.size)]
            difference += float(post[draw].sum()) - float(pre[draw].sum())
        distribution[repetition] = difference / total
    delta = float(post.mean()) - float(pre.mean())
    low, high = np.percentile(distribution, [2.5, 97.5])
    p_one_sided = (np.count_nonzero(distribution <= 0.0) + 1) / (int(reps) + 1)
    return {'delta_top1': delta, 'ci95_top1': [float(low), float(high)],
            'p_one_sided': float(p_one_sided), 'repetitions': int(reps),
            'seed': int(seed), 'stratified_by': 'class'}


def bootstrap_fast(pre, post, targets, reps, seed):
    """Wektorowy wariant dla porownan eksploracyjnych (ta sama procedura)."""
    pre = np.asarray(pre).astype(bool).ravel()
    post = np.asarray(post).astype(bool).ravel()
    groups = _groups(targets)
    total = len(pre)
    rng = np.random.default_rng(int(seed))
    distribution = np.zeros(int(reps), dtype=np.float64)
    for group in groups:
        draws = rng.integers(0, group.size, size=(int(reps), group.size))
        distribution += post[group][draws].sum(axis=1) - pre[group][draws].sum(axis=1)
    distribution /= total
    delta = float(post.mean()) - float(pre.mean())
    low, high = np.percentile(distribution, [2.5, 97.5])
    p_one_sided = (np.count_nonzero(distribution <= 0.0) + 1) / (int(reps) + 1)
    return {'delta_top1': delta, 'ci95_top1': [float(low), float(high)],
            'p_one_sided': float(p_one_sided), 'repetitions': int(reps),
            'seed': int(seed), 'stratified_by': 'class'}


def holm(p_values, alpha):
    order = np.argsort(np.asarray(p_values))
    rejects = np.zeros(len(p_values), dtype=bool)
    for position, index in enumerate(order):
        if p_values[index] <= alpha / (len(p_values) - position):
            rejects[index] = True
        else:
            break
    return rejects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = args.run_dir.resolve()
    reps = int(cfg['bootstrap']['reps'])
    seed = int(cfg['bootstrap']['seed'])
    margin = float(cfg['decision']['primary']['margin_pp']) / 100.0
    alpha = float(cfg['decision']['primary']['alpha'])
    metrics, cells = {}, {}
    for key in DATASET_KEYS:
        ds = dataset_cfg(cfg, key)
        metrics[key] = {}
        cells[key] = {}
        base = cell_metrics(run / 'merge' / 'dev' / f'{key}_BASE.npz')
        cells[key]['BASE'] = base
        for ratio, spec in ds['criteria']['ratios'].items():
            metrics[key][ratio] = {}
            for criterion in spec['criteria']:
                zero = cell_metrics(run / 'merge' / 'dev' / f'{key}_{ratio}_{criterion}.npz')
                cells[key][f'{ratio}_{criterion}_T0'] = zero
                metrics[key][ratio][criterion] = {'T0': {'top1': zero['top1'], 'nll': zero['nll'],
                                                         'correct': int(zero['correct'].sum()),
                                                         'count': zero['count']}}
                for epochs in spec['repair_epochs']:
                    path = run / 'repair' / f'{key}_{ratio}_{criterion}_T{epochs}' / 'dev_final.npz'
                    cell = cell_metrics(path)
                    cells[key][f'{ratio}_{criterion}_T{epochs}'] = cell
                    metrics[key][ratio][criterion][f'T{epochs}'] = {
                        'top1': cell['top1'], 'nll': cell['nll'],
                        'correct': int(cell['correct'].sum()), 'count': cell['count']}
        targets = base['targets']
        for name, cell in cells[key].items():
            require(np.array_equal(cell['targets'], targets), f'Targets mismatch: {name}')
    comparisons = {}
    for key in DATASET_KEYS:
        ds = dataset_cfg(cfg, key)
        for ratio, spec in ds['criteria']['ratios'].items():
            random_key = f'{ratio}_RANDOM_T0'
            for criterion in spec['criteria']:
                cell_key = f'{ratio}_{criterion}_T0'
                result = bootstrap_fast(cells[key][random_key]['correct'],
                                        cells[key][cell_key]['correct'],
                                        cells[key][random_key]['targets'], reps, seed)
                comparisons[f'{key}_{ratio}_T0_{criterion}_minus_RANDOM'] = result
                result = bootstrap_fast(cells[key]['BASE']['correct'],
                                        cells[key][cell_key]['correct'],
                                        cells[key]['BASE']['targets'], reps, seed)
                comparisons[f'{key}_{ratio}_T0_{criterion}_minus_BASE'] = result
            for epochs in spec['repair_epochs']:
                random_cell = cells[key][f'{ratio}_RANDOM_T{epochs}']
                for criterion in spec['criteria']:
                    cell = cells[key][f'{ratio}_{criterion}_T{epochs}']
                    exact = (key == cfg['decision']['primary']['dataset']
                             and ratio == cfg['decision']['primary']['ratio']
                             and epochs == int(cfg['decision']['primary']['T']))
                    function = bootstrap_exact if exact else bootstrap_fast
                    comparisons[f'{key}_{ratio}_T{epochs}_{criterion}_minus_RANDOM'] = function(
                        random_cell['correct'], cell['correct'], random_cell['targets'], reps, seed)
                for reference in ('PRE', 'POST'):
                    reference_cell = cells[key][f'{ratio}_{reference}_T{epochs}']
                    comparisons[f'{key}_{ratio}_T{epochs}_{reference}_minus_RANDOM'] = bootstrap_fast(
                        random_cell['correct'], reference_cell['correct'],
                        random_cell['targets'], reps, seed)
    primary = cfg['decision']['primary']
    p_values = []
    for criterion in primary['criteria']:
        result = comparisons[f'{primary["dataset"]}_{primary["ratio"]}_'
                             f'T{primary["T"]}_{criterion}_minus_RANDOM']
        p_values.append(result['p_one_sided'])
    rejects = holm(p_values, alpha)
    primary_rows = {}
    for criterion, p_value, reject in zip(primary['criteria'], p_values, rejects):
        result = comparisons[f'{primary["dataset"]}_{primary["ratio"]}_'
                             f'T{primary["T"]}_{criterion}_minus_RANDOM']
        survivor = bool(result['delta_top1'] >= margin and reject)
        primary_rows[criterion] = {'delta_top1': result['delta_top1'],
                                   'ci95_top1': result['ci95_top1'],
                                   'p_one_sided': p_value, 'holm_reject': bool(reject),
                                   'margin_pp': margin * 100, 'survivor': survivor}
    survivors = [criterion for criterion, row in primary_rows.items() if row['survivor']]
    replication = {}
    for spec in cfg['decision']['replication']:
        ds = dataset_cfg(cfg, spec['dataset'])
        ratio_spec = ds['criteria']['ratios'][spec['ratio']]
        for criterion in survivors:
            if criterion not in ratio_spec['criteria']:
                replication[f'{spec["dataset"]}_{spec["ratio"]}_{criterion}'] = {
                    'status': 'not_in_grid'}
                continue
            name = (f'{spec["dataset"]}_{spec["ratio"]}_T{spec["T"]}_'
                    f'{criterion}_minus_RANDOM')
            result = comparisons.get(name)
            require(result is not None, f'Missing replication comparison: {name}')
            success = bool(result['delta_top1'] >= margin and result['ci95_top1'][0] > 0)
            replication[f'{spec["dataset"]}_{spec["ratio"]}_{criterion}'] = {
                'delta_top1': result['delta_top1'], 'ci95_top1': result['ci95_top1'],
                'margin_pp': margin * 100, 'success': success}
    decision = {
        'primary': {'dataset': primary['dataset'], 'ratio': primary['ratio'],
                    'T': primary['T'], 'criteria': primary_rows,
                    'survivors': survivors},
        'replication': replication,
        'survivors_replicated': [key for key, row in replication.items()
                                 if row.get('success')],
        'conclusion': ('positive_criterion_found' if any(
            row.get('success') for row in replication.values())
            else 'no_survivor' if not survivors else 'survivor_not_replicated')}
    summary = {'experiment': cfg['experiment'], 'run_id': run.name,
               'bootstrap': {'reps': reps, 'seed': seed, 'stratified_by': 'class'},
               'metrics': metrics, 'comparisons': comparisons,
               'primary': primary_rows, 'decision': decision}
    write_json(run / 'summary.json', summary)
    print('E010_SUMMARY=' + __import__('json').dumps(
        {'primary': primary_rows, 'decision': decision['conclusion']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
