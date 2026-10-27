"""E013: analiza przesiewu alokacji — tabele i sparowany bootstrap.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Wejście: katalog biegu E013 z summary.json i allocations/*/logits.npz.
Wyjście: analysis.json oraz tabela Markdown na stdout; bez GPU i bez treningu.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENT = '2026-10-25_E013_alokacja_budzetu_foldingu'
OPIS_PATH = 'experiments/2026-10-25_E013_alokacja_budzetu_foldingu/OPIS.md'
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261027


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load_logits(run_dir, name):
    path = Path(run_dir) / 'allocations' / name / 'logits.npz'
    require(path.is_file(), f'Brak {path}')
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def metrics(logits, labels):
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    nll = float(F.cross_entropy(logits_t, labels_t))
    correct = logits_t.argmax(1).cpu().numpy() == labels
    return {
        'top1': float(correct.mean()),
        'correct': int(correct.sum()),
        'count': int(len(labels)),
        'nll': nll,
        'correctness': correct,
    }


def logit_metrics(candidate, baseline):
    candidate = torch.as_tensor(candidate, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    diff = candidate - baseline
    logp_candidate = F.log_softmax(candidate, dim=1)
    logp_baseline = F.log_softmax(baseline, dim=1)
    return {
        'prediction_agreement': float(
            (candidate.argmax(1) == baseline.argmax(1)).float().mean()),
        'logit_mae': float(diff.abs().mean()),
        'logit_rmse': float(diff.pow(2).mean().sqrt()),
        'logit_max_abs': float(diff.abs().max()),
        'kl_baseline_candidate': float(F.kl_div(
            logp_candidate, logp_baseline.exp(), reduction='batchmean')),
        'kl_candidate_baseline': float(F.kl_div(
            logp_baseline, logp_candidate.exp(), reduction='batchmean')),
    }


def paired_bootstrap(correct_a, correct_b, labels):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    classes = np.unique(labels)
    pools = [np.flatnonzero(labels == value) for value in classes]
    values = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPS):
        sample = np.concatenate([
            rng.choice(pool, size=len(pool), replace=True) for pool in pools
        ])
        values[index] = delta[sample].mean()
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        'delta_top1': float(delta.mean()),
        'ci95': [float(lower), float(upper)],
        'reps': BOOTSTRAP_REPS,
        'seed': BOOTSTRAP_SEED,
        'stratified_by': 'class',
    }


def mcnemar(correct_a, correct_b):
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        # dwustronny dokładny test znaków jako McNemar
        tail = min(a_only, b_only)
        p_value = min(1.0, 2.0 * sum(
            math.comb(discordant, k) for k in range(0, tail + 1)
        ) / (2 ** discordant))
    return {'a_only': a_only, 'b_only': b_only, 'p_value': float(p_value)}


def summarize(run_dir, name):
    summary = read_json(Path(run_dir) / 'allocations' / name / 'summary.json')
    data = load_logits(run_dir, name)
    labels = data['labels']
    baseline = data['baseline']
    post = data['post_bn']
    pre = data['pre_bn']
    post_metrics = metrics(post, labels)
    pre_metrics = metrics(pre, labels)
    baseline_metrics = metrics(baseline, labels)
    return {
        'name': name,
        'budget': float(summary['spec']['budget']),
        'family': summary['spec']['family'],
        'params': int(summary['compressed']['params']),
        'param_reduction_fraction': float(
            summary['compressed']['param_reduction_fraction']),
        'pre_bn': {key: value for key, value in pre_metrics.items()
                   if key != 'correctness'},
        'post_bn': {key: value for key, value in post_metrics.items()
                    if key != 'correctness'},
        'baseline': {key: value for key, value in baseline_metrics.items()
                     if key != 'correctness'},
        'pre_bn_vs_baseline': logit_metrics(pre, baseline),
        'post_bn_vs_baseline': logit_metrics(post, baseline),
        'correctness_post': post_metrics['correctness'],
        'allocation': summary['allocation'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    aggregate = read_json(run_dir / 'summary.json')
    names = sorted(aggregate)
    results = {name: summarize(run_dir, name) for name in names}
    labels = load_logits(run_dir, names[0])['labels']

    comparisons = {}
    for budget in sorted({item['budget'] for item in results.values()}):
        for family in ('uniform', 'tilt_p030', 'tilt_p015', 'random_102'):
            target = next((name for name in names
                           if results[name]['budget'] == budget
                           and results[name]['name'].endswith(family)), None)
            if target is None:
                continue
            base_name = next(name for name in names
                             if results[name]['budget'] == budget
                             and name.endswith('uniform'))
            key = f'{target}_vs_uniform'
            comparisons[key] = {
                'candidate': target,
                'reference': base_name,
                'budget': budget,
                'paired_bootstrap': paired_bootstrap(
                    results[target]['correctness_post'],
                    results[base_name]['correctness_post'], labels),
                'mcnemar': mcnemar(
                    results[target]['correctness_post'],
                    results[base_name]['correctness_post']),
                'delta_nll': (
                    results[target]['post_bn']['nll']
                    - results[base_name]['post_bn']['nll']),
                'delta_agreement': (
                    results[target]['post_bn_vs_baseline']['prediction_agreement']
                    - results[base_name]['post_bn_vs_baseline']['prediction_agreement']),
                'delta_logit_mae': (
                    results[target]['post_bn_vs_baseline']['logit_mae']
                    - results[base_name]['post_bn_vs_baseline']['logit_mae']),
            }

    output = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'run_dir': str(run_dir),
        'metrics': {
            name: {key: value for key, value in item.items()
                   if key != 'correctness_post'}
            for name, item in results.items()
        },
        'comparisons': comparisons,
    }
    write_json(run_dir / 'analysis.json', output)

    print('| Budżet | Wariant | Params | Pre-BN | Post-BN | Δ vs uniform | 95% CI | McNemar p | Agreement | MAE |')
    print('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for budget in sorted({item['budget'] for item in results.values()}):
        base = next(item for item in results.values()
                    if item['budget'] == budget
                    and item['name'].endswith('uniform'))
        for name in sorted(
                (name for name in names if results[name]['budget'] == budget),
                key=lambda value: -results[value]['post_bn']['top1']):
            item = results[name]
            key = f'{name}_vs_uniform'
            comp = comparisons.get(key)
            if comp:
                delta = comp['paired_bootstrap']['delta_top1'] * 100
                ci = [value * 100 for value in comp['paired_bootstrap']['ci95']]
                p_value = comp['mcnemar']['p_value']
            else:
                delta = ci = p_value = None
            print(
                f"| {budget:.2f} | {item['name']} | {item['params']} | "
                f"{item['pre_bn']['top1']*100:.2f}% | "
                f"{item['post_bn']['top1']*100:.2f}% | "
                f"{'' if delta is None else f'{delta:+.2f}'} | "
                f"{'' if ci is None else f'[{ci[0]:+.2f}; {ci[1]:+.2f}]'} | "
                f"{'' if p_value is None else f'{p_value:.3g}'} | "
                f"{item['post_bn_vs_baseline']['prediction_agreement']*100:.2f}% | "
                f"{item['post_bn_vs_baseline']['logit_mae']:.3f} |")
    print(f'ANALYSIS_JSON={run_dir / "analysis.json"}')


if __name__ == '__main__':
    main()
