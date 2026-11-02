"""E019: analiza CPU zamrożonego porównania alokacji na Tiny ImageNet.

Eksperyment: 2026-10-29_E019_tinyimagenet_alokacja; pełny opis: OPIS.md.
Wejście: ukończony katalog ``runs/E019/fold_*`` z 48 plikami summary.json.
Wyjście: nowy katalog z ``analysis.json``, ``analysis.md``,
``seed_level.csv`` i ``summary_stats.csv``. Skrypt nie trenuje, nie używa GPU
i nie zmienia wyników źródłowych. Jednostką wnioskowania jest seed (n=4);
trzy budżety dzielą te same cztery checkpointy, więc 12 par nie jest 12
niezależnymi replikacjami.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = '2026-10-29_E019_tinyimagenet_alokacja'
OPIS_PATH = 'experiments/2026-10-29_E019_tinyimagenet_alokacja/OPIS.md'
SEEDS = (42, 43, 44, 45)
BUDGETS = (30, 50, 70)
VARIANTS = ('uniform', 'tilt_p030', 'tilt_p015', 'mag2')
CANDIDATES = ('tilt_p030', 'tilt_p015', 'mag2')
LABELS = {
    'uniform': 'UNIFORM',
    'tilt_p030': 'TILT+0.30',
    'tilt_p015': 'TILT+0.15',
    'mag2': 'MAG2',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
        encoding='utf-8')


def pl(value, digits=2, signed=False):
    text = f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'
    return text.replace('.', ',')


def describe(values):
    array = np.asarray(list(values), dtype=float)
    require(array.size >= 2, 'Potrzeba co najmniej dwóch wartości')
    n = int(array.size)
    return {
        'n': n,
        'mean': float(array.mean()),
        'sd': float(array.std(ddof=1)),
        'min': float(array.min()),
        'max': float(array.max()),
        'positive_count': int(np.sum(array > 0)),
    }


def load_run(fold_run):
    fold_run = Path(fold_run).resolve()
    manifest_path = fold_run / 'manifest.json'
    require(manifest_path.is_file(), f'Brak manifestu: {manifest_path}')
    manifest = read_json(manifest_path)
    require(manifest.get('experiment') == EXPERIMENT,
            'Manifest dotyczy innego eksperymentu')
    require(manifest.get('status') == 'completed',
            'Bieg foldingu nie jest ukończony')
    require(manifest.get('official_test_used') is False,
            'Bieg użył oficjalnego testu')
    require(manifest.get('reserve_used') is False,
            'Bieg użył rezerwy')
    data = {}
    inputs = [manifest_path]
    for seed in SEEDS:
        data[seed] = {}
        for budget in BUDGETS:
            for variant in VARIANTS:
                name = f'b{budget}_{variant}'
                summary_path = (
                    fold_run / 'allocations' / f'seed_{seed}' / name
                    / 'summary.json')
                require(summary_path.is_file(), f'Brak {summary_path}')
                summary = read_json(summary_path)
                require(summary.get('status') == 'completed',
                        f'{seed}/{name}: {summary.get("status")}')
                require(summary.get('official_test_used') is False,
                        f'{seed}/{name}: użyto testu')
                post = summary['compressed']['post_bn']
                pre = summary['compressed']['pre_bn']
                require(post['count'] == 10000, f'{seed}/{name}: dev != 10000')
                data[seed][name] = {
                    'seed': seed,
                    'budget': budget,
                    'variant': variant,
                    'name': name,
                    'params': int(summary['compressed']['params']),
                    'params_before': int(summary['compressed']['params_before']),
                    'nominal_target_params': int(
                        summary['nominal_target_params']),
                    'uniform_reference_params': int(
                        summary['uniform_reference_params']),
                    'target_params': int(summary['target_params']),
                    'nominal_reduction_pct': float(
                        summary['nominal_target_reduction_fraction'] * 100.0),
                    'target_reduction_pct': float(
                        summary['target_reduction_fraction'] * 100.0),
                    'realized_reduction_pct': float(
                        summary['compressed']['param_reduction_fraction'] * 100.0),
                    'top1': float(post['top1']),
                    'nll': float(post['nll']),
                    'pre_top1': float(pre['top1']),
                    'prediction_agreement': float(
                        summary['compressed']['post_bn_vs_baseline'][
                            'prediction_agreement']),
                    'logit_mae': float(
                        summary['compressed']['post_bn_vs_baseline']['logit_mae']),
                    'logit_rmse': float(
                        summary['compressed']['post_bn_vs_baseline']['logit_rmse']),
                    'kl_baseline_candidate': float(
                        summary['compressed']['post_bn_vs_baseline'][
                            'kl_baseline_candidate']),
                    'baseline_top1': float(summary['baseline']['metrics']['top1']),
                    'baseline_nll': float(summary['baseline']['metrics']['nll']),
                }
                inputs.append(summary_path)
    for seed in SEEDS:
        signatures = {
            (
                data[seed][f'b{budget}_{variant}']['baseline_top1'],
                data[seed][f'b{budget}_{variant}']['baseline_nll'],
            )
            for budget in BUDGETS for variant in VARIANTS
        }
        require(len(signatures) == 1, f'seed {seed}: baseline się różni')
    for seed in SEEDS:
        for budget in BUDGETS:
            uniform = data[seed][f'b{budget}_uniform']
            nominal = {
                data[seed][f'b{budget}_{variant}']['nominal_target_params']
                for variant in VARIANTS}
            targets = {
                data[seed][f'b{budget}_{variant}']['target_params']
                for variant in VARIANTS}
            require(len(nominal) == 1,
                    f'seed {seed}/{budget}%: różne nominalne cele')
            require(len(targets) == 1,
                    f'seed {seed}/{budget}%: różne wspólne targety')
            require(targets.pop() == uniform['params'],
                    f'seed {seed}/{budget}%: target != faktyczny UNIFORM')
    for budget in BUDGETS:
        for variant in VARIANTS:
            name = f'b{budget}_{variant}'
            params = {data[seed][name]['params'] for seed in SEEDS}
            require(len(params) == 1,
                    f'{name}: parametry różnią się między seedami: {params}')
    return manifest, data, inputs


def budget_rows(data):
    rows = []
    for budget in BUDGETS:
        uniform = data[SEEDS[0]][f'b{budget}_uniform']
        for variant in VARIANTS:
            item = data[SEEDS[0]][f'b{budget}_{variant}']
            rows.append({
                'nominal_budget': budget,
                'variant': variant,
                'params_before': item['params_before'],
                'nominal_target_params': item['nominal_target_params'],
                'effective_target_params': item['target_params'],
                'params': item['params'],
                'nominal_reduction_pct': item['nominal_reduction_pct'],
                'target_reduction_pct': item['target_reduction_pct'],
                'realized_reduction_pct': item['realized_reduction_pct'],
                'deviation_from_nominal_pp': (
                    item['realized_reduction_pct']
                    - item['nominal_reduction_pct']),
                'deviation_from_target_pp': (
                    item['realized_reduction_pct']
                    - item['target_reduction_pct']),
                'extra_params_vs_uniform': item['params'] - uniform['params'],
            })
    return rows


def secondary(data):
    metrics = (
        ('nll', 'ΔNLL', '', 1.0, 4),
        ('prediction_agreement', 'Δagreement', 'pp', 100.0, 2),
        ('logit_mae', 'ΔMAE', '', 1.0, 4),
        ('logit_rmse', 'ΔRMSE', '', 1.0, 4),
        ('kl_baseline_candidate', 'ΔKL', '', 1.0, 4),
    )
    output = {}
    for budget in BUDGETS:
        output[str(budget)] = {}
        for metric, label, unit, scale, digits in metrics:
            differences = [
                float(
                    data[seed][f'b{budget}_tilt_p030'][metric]
                    - data[seed][f'b{budget}_uniform'][metric])
                for seed in SEEDS
            ]
            output[str(budget)][metric] = {
                'label': label,
                'unit': unit,
                'scale': scale,
                'digits': digits,
                'differences': {
                    str(seed): value for seed, value in zip(SEEDS, differences)
                },
                'summary': describe(differences),
            }
    return output


def paired(data, candidate, reference='uniform'):
    output = {}
    for budget in BUDGETS:
        differences = {
            str(seed): float(
                (data[seed][f'b{budget}_{candidate}']['top1']
                 - data[seed][f'b{budget}_{reference}']['top1']) * 100.0)
            for seed in SEEDS
        }
        output[str(budget)] = {
            'differences_pp': differences,
            'summary': describe(differences.values()),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    fold_run = args.fold_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(not output_dir.exists(), f'Katalog wyjściowy istnieje: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest, data, inputs = load_run(fold_run)
    paired_results = {
        candidate: paired(data, candidate) for candidate in CANDIDATES
    }
    variant_means = {}
    for budget in BUDGETS:
        variant_means[str(budget)] = {}
        for variant in VARIANTS:
            items = [data[seed][f'b{budget}_{variant}'] for seed in SEEDS]
            variant_means[str(budget)][variant] = {
                'mean_top1_pct': float(np.mean([item['top1'] for item in items]) * 100.0),
                'sd_top1_pct': float(np.std([item['top1'] for item in items], ddof=1) * 100.0),
                'mean_nll': float(np.mean([item['nll'] for item in items])),
                'mean_params': int(np.mean([item['params'] for item in items])),
                'mean_nominal_reduction_pct': float(
                    np.mean([item['nominal_reduction_pct'] for item in items])),
                'mean_target_reduction_pct': float(
                    np.mean([item['target_reduction_pct'] for item in items])),
                'mean_reduction_pct': float(
                    np.mean([item['realized_reduction_pct'] for item in items])),
                'seed_top1_pct': {
                    str(seed): float(data[seed][f'b{budget}_{variant}']['top1'] * 100.0)
                    for seed in SEEDS
                },
            }
    positive = sum(
        1 for budget in BUDGETS
        for value in paired_results['tilt_p030'][str(budget)]['differences_pp'].values()
        if value > 0)
    analysis = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'fold_run': str(fold_run),
        'run_id': manifest.get('run_id'),
        'git_sha': manifest.get('git_sha'),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': list(SEEDS),
        'budgets': list(BUDGETS),
        'variants': list(VARIANTS),
        'analysis_unit': (
            'seed (n=4); 12 par seed×budżet nie jest 12 niezależnymi replikacjami'),
        'target_policy': (
            'per budget: target = faktyczna liczba parametrów UNIFORM dla '
            'nominalnego ratio; TILT repair i MAG2 dopasowane do tego samego '
            'targetu; raportowana redukcja nominalna i faktyczna'),
        'budget_rows': budget_rows(data),
        'paired': paired_results,
        'secondary_tilt030_vs_uniform': secondary(data),
        'variant_means': variant_means,
        'tilt030_positive_pairs': {'positive': positive, 'total': 12},
        'inputs': [
            {'path': str(path), 'sha256': sha256_file(path)} for path in inputs
        ],
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
        },
        'official_test_used': False,
        'reserve_used': False,
    }
    write_json(output_dir / 'analysis.json', analysis)

    seed_rows = []
    for seed in SEEDS:
        for budget in BUDGETS:
            uniform = data[seed][f'b{budget}_uniform']
            for variant in VARIANTS:
                item = data[seed][f'b{budget}_{variant}']
                seed_rows.append({
                    'seed': seed,
                    'nominal_budget': budget,
                    'variant': variant,
                    'params': item['params'],
                    'params_before': item['params_before'],
                    'nominal_target_params': item['nominal_target_params'],
                    'effective_target_params': item['target_params'],
                    'nominal_reduction_pct': item['nominal_reduction_pct'],
                    'target_reduction_pct': item['target_reduction_pct'],
                    'realized_reduction_pct': item['realized_reduction_pct'],
                    'top1_post_bn': item['top1'],
                    'nll_post_bn': item['nll'],
                    'delta_top1_vs_uniform_pp': (
                        (item['top1'] - uniform['top1']) * 100.0),
                    'prediction_agreement': item['prediction_agreement'],
                    'logit_mae': item['logit_mae'],
                    'logit_rmse': item['logit_rmse'],
                    'kl_baseline_candidate': item['kl_baseline_candidate'],
                })
    with (output_dir / 'seed_level.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_rows = []
    for candidate in CANDIDATES:
        for budget in BUDGETS:
            summary = paired_results[candidate][str(budget)]['summary']
            summary_rows.append({
                'candidate': candidate,
                'nominal_budget': budget,
                'n': summary['n'],
                'mean_diff_pp': summary['mean'],
                'sd_diff_pp': summary['sd'],
                'min_diff_pp': summary['min'],
                'max_diff_pp': summary['max'],
                'positive_count': summary['positive_count'],
            })
    with (output_dir / 'summary_stats.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        '# E019 — analiza alokacji Tiny ImageNet',
        '',
        f'Run foldingu: `{manifest.get("run_id")}` (git `{manifest.get("git_sha")}`)',
        '',
        'Jednostką jest seed (n=4). Trzy budżety dzielą te same cztery '
        'checkpointy; 12 różnic seed×budżet nie jest 12 niezależnymi replikacjami.',
        '',
        '## Rzeczywiste budżety',
        '',
        'Cel wspólny to faktyczna liczba parametrów UNIFORM dla nominalnego '
        'budżetu; TILT i MAG2 są do niego dopasowane.',
        '',
        '| Budżet | Wariant | Cel nominalny | Cel = UNIFORM | Parametry | Redukcja nominalna | Redukcja faktyczna | Faktyczna − cel | Extra vs UNIFORM |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in analysis['budget_rows']:
        extra = '—' if row['variant'] == 'uniform' else str(
            row['extra_params_vs_uniform'])
        lines.append(
            f"| {row['nominal_budget']}% | {LABELS[row['variant']]} | "
            f"{row['nominal_target_params']} | {row['effective_target_params']} | "
            f"{row['params']} | {pl(row['nominal_reduction_pct'], 3)}% | "
            f"{pl(row['realized_reduction_pct'], 3)}% | "
            f"{pl(row['deviation_from_target_pp'], 3, True)} pp | {extra} |")
    lines.extend([
        '',
        'Tolerancja: realized params nie mogą przekroczyć wspólnego targetu '
        'i mogą być od niego niższe o najwyżej 1,0%; w praktyce odchylenie '
        'TILT/MAG2 od UNIFORM jest rzędu 0,001–0,003%. Dodatnia wartość '
        '„Faktyczna − cel” oznacza mniej parametrów niż UNIFORM. Kolumna '
        '„Redukcja nominalna” odnosi się do `(1-budget)`, a „faktyczna” do '
        'rzeczywistej liczby parametrów po kompresji.',
        '',
        '## TILT+0.30 vs UNIFORM',
        '',
        '| Budżet | seed42 | seed43 | seed44 | seed45 | średnia | SD | dodatnie |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for budget in BUDGETS:
        block = paired_results['tilt_p030'][str(budget)]
        diffs = block['differences_pp']
        summary = block['summary']
        lines.append(
            f"| {budget}% | {pl(diffs['42'], 2, True)} | "
            f"{pl(diffs['43'], 2, True)} | {pl(diffs['44'], 2, True)} | "
            f"{pl(diffs['45'], 2, True)} | {pl(summary['mean'], 3, True)} | "
            f"{pl(summary['sd'], 3)} | {summary['positive_count']}/4 |")
    lines.extend([
        '',
        f"Dodatnie pary TILT+0.30 vs UNIFORM: "
        f"**{positive}/12**.",
        '',
        '## Metryki wtórne TILT+0.30 vs UNIFORM (średnia ± SD różnicy)',
        '',
        'Dla NLL, MAE, RMSE i KL ujemna różnica oznacza lepszy TILT; dla '
        'agreement dodatnia oznacza lepszy TILT.',
        '',
        '| Budżet | ΔNLL | Δagreement | ΔMAE | ΔRMSE | ΔKL |',
        '|---:|---:|---:|---:|---:|---:|',
    ])
    for budget in BUDGETS:
        cells = []
        for metric in ('nll', 'prediction_agreement', 'logit_mae',
                       'logit_rmse', 'kl_baseline_candidate'):
            block = analysis['secondary_tilt030_vs_uniform'][str(budget)][metric]
            summary = block['summary']
            scale = block['scale']
            digits = block['digits']
            cells.append(
                f"{pl(summary['mean'] * scale, digits, True)} ± "
                f"{pl(summary['sd'] * scale, digits)}")
        lines.append(f"| {budget}% | " + " | ".join(cells) + " |")
    lines.extend([
        '',
        '## Pozostałe warianty vs UNIFORM (średnia ± SD, pp)',
        '',
        '| Wariant | Budżet | różnica |',
        '|---|---:|---:|',
    ])
    for candidate in ('tilt_p015', 'mag2'):
        for budget in BUDGETS:
            summary = paired_results[candidate][str(budget)]['summary']
            lines.append(
                f"| {LABELS[candidate]} | {budget}% | "
                f"{pl(summary['mean'], 3, True)} ± {pl(summary['sd'], 3)} |")
    lines.extend([
        '',
        '## Ograniczenia',
        '',
        'To jeden dataset (Tiny ImageNet), dev i cztery checkpointy z jednego '
        'przepisu treningowego. Brak niezależnego testu; oficjalny test i '
        'rezerwa 10 000 obrazów pozostały nietknięte. Wynik jest adaptacją '
        'operatora E012, nie reprodukcją liczb artykułu 1:1.',
    ])
    (output_dir / 'analysis.md').write_text('\n'.join(lines) + '\n',
                                            encoding='utf-8')
    print(f'E019_ANALYSIS={output_dir}')
    print(f'TILT_P030_POSITIVE={positive}/12')
    for budget in BUDGETS:
        summary = paired_results['tilt_p030'][str(budget)]['summary']
        print(f'  {budget}%: {summary["mean"]:+.3f} ± {summary["sd"]:.3f} pp')
    return 0


if __name__ == '__main__':
    sys.exit(main())
