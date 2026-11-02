"""E018: analiza replikacji cosine (CPU).

Eksperyment: 2026-10-29_E018_replikacja_cosine; pełny protokół: OPIS.md.
Wejście: ukończony katalog ``runs/E018/<RUN>`` z 24 summary.
Wyjście: ``analysis.json``, ``analysis.md`` i ``summary.csv`` w nowym katalogu;
decyzja według kryterium głównego z OPIS.md. Bez treningu i bez GPU.
"""
from __future__ import annotations

import argparse
import csv
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e018_common as common  # noqa: E402

RUNS = common.ROOT / 'runs' / 'E018'


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def output_dir():
    """Tworzy unikalny katalog analizy."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = RUNS / 'analizy'
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        path = base / f'{stamp}_{secrets.token_hex(4)}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć katalogu analizy')


def summary_path(run, task, budget):
    return (run / 'tasks' / task['dataset'] / task['measure']
            / f'b{int(round(budget * 100)):02d}' / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    manifest = common.read_json(run / 'manifest.json')
    require(manifest.get('status') == 'completed', 'Bieg nie jest ukończony')
    require(manifest.get('official_test_used') is False,
            'Bieg użył oficjalnego testu')
    require(manifest.get('reserve_used') is False, 'Bieg użył rezerwy')
    cfg = common.load_config()
    require(manifest.get('config_sha256')
            == common.digest(common.CONFIG_PATH),
            'Config biegu różni się od zamrożonego')

    rows = []
    for task in common.build_tasks(cfg):
        path = summary_path(run, task, task['budget'])
        require(path.is_file(), f'Brak summary: {path}')
        summary = common.read_json(path)
        require(summary.get('status') == 'completed',
                f"{task['name']}: status {summary.get('status')}")
        require(summary.get('official_test_used') is False,
                f"{task['name']}: użyto testu")
        require(summary.get('reserve_used') is False,
                f"{task['name']}: użyto rezerwy")
        rows.append({
            'dataset': task['dataset'],
            'family': task['family'],
            'seed': task['seed'],
            'budget': task['budget'],
            'measure': task['measure'],
            'params_after': summary['params_after'],
            'post_top1': float(summary['compressed']['post_bn']['top1']),
            'post_nll': float(summary['compressed']['post_bn']['nll']),
        })
    require(len(rows) == 24, 'Oczekiwano 24 wyników')
    for tag in {row['dataset'] for row in rows}:
        for budget in cfg['budgets']:
            values = {row['params_after'] for row in rows
                      if row['dataset'] == tag and row['budget'] == budget}
            require(len(values) == 1, f'Nierówne budżety: {tag} {budget}')

    table = {}
    for row in rows:
        key = (row['dataset'], row['budget'])
        table.setdefault(key, {})[row['measure']] = row
    comparison = {}
    for (tag, budget), values in table.items():
        reference = values['reference']['post_top1']
        cosine = values['cosine']['post_top1']
        comparison[f'{tag}|{budget:.2f}'] = {
            'dataset': tag,
            'family': values['reference']['family'],
            'seed': values['reference']['seed'],
            'budget': budget,
            'reference_top1': reference,
            'cosine_top1': cosine,
            'delta_pp': 100.0 * (cosine - reference),
            'reference_nll': values['reference']['post_nll'],
            'cosine_nll': values['cosine']['post_nll'],
            'params_after': values['reference']['params_after'],
        }

    primary = cfg['primary']
    primary_rows = [item for item in comparison.values()
                    if item['family'] == primary['dataset_family']
                    and abs(item['budget'] - float(primary['budget'])) < 1e-9
                    and item['seed'] in primary['seeds']]
    require(len(primary_rows) == len(primary['seeds']),
            'Brak kompletu seedów testu głównego')
    deltas = [item['delta_pp'] for item in primary_rows]
    mean_delta = sum(deltas) / len(deltas)
    all_positive = all(value > 0.0 for value in deltas)
    passed = (all_positive if primary['all_positive'] else True) and (
        mean_delta >= float(primary['mean_min_pp']))

    specificity = cfg['specificity']
    specificity_rows = [item for item in comparison.values()
                        if item['family'] == specificity['dataset_family']
                        and abs(item['budget'] - float(specificity['budget']))
                        < 1e-9]
    specificity_mean = (
        sum(item['delta_pp'] for item in specificity_rows)
        / len(specificity_rows)) if specificity_rows else None
    secondary_budget = float(cfg['secondary_budget'])
    secondary_rows = [item for item in comparison.values()
                      if item['family'] == 'cifar10'
                      and abs(item['budget'] - secondary_budget) < 1e-9]
    secondary_mean = (
        sum(item['delta_pp'] for item in secondary_rows)
        / len(secondary_rows)) if secondary_rows else None

    decision = {
        'primary_passed': bool(passed),
        'primary_mean_delta_pp': mean_delta,
        'primary_deltas_pp': deltas,
        'primary_all_positive': all_positive,
        'primary_mean_min_pp': float(primary['mean_min_pp']),
        'specificity_cifar100_mean_pp': specificity_mean,
        'secondary_cifar10_30_mean_pp': secondary_mean,
    }
    out = output_dir()
    analysis = {
        'run': run.name,
        'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH,
        'status': 'completed',
        'comparison': comparison,
        'decision': decision,
    }
    common.write_json_atomic(out / 'analysis.json', analysis)

    lines = [
        '# E018 — replikacja cosine: analiza',
        '',
        'Różnica `cosine − reference` w pp (dev top-1 po BN). '
        'Seedy 43–45 nie brały udziału w odkryciu efektu.',
        '',
        '| dataset | seed | budżet | reference | cosine | delta | params |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for key in sorted(comparison):
        item = comparison[key]
        lines.append(
            f"| {item['dataset']} | {item['seed']} | "
            f"{int(round(item['budget']*100))}% | "
            f"{100*item['reference_top1']:.2f} | "
            f"{100*item['cosine_top1']:.2f} | "
            f"{item['delta_pp']:+.2f} | {item['params_after']} |")
    lines += [
        '',
        '## Decyzja',
        '',
        f"- Test główny (CIFAR-10, 50%): średnia **{mean_delta:+.2f} pp**, "
        f"dodatnie na {sum(1 for value in deltas if value > 0)}/3 seedach.",
        f"- Próg: średnia ≥ {float(primary['mean_min_pp']):+.2f} pp i wszystkie "
        'seedy dodatnie → **'
        + ('SPEŁNIONY' if passed else 'NIESPEŁNIONY') + '**.',
        ('- Kontrola specyficzności (CIFAR-100, 50%): '
         f'{specificity_mean:+.2f} pp.' if specificity_mean is not None
         else '- Brak kontroli specyficzności.'),
        ('- Budżet wtórny (CIFAR-10, 30%): '
         f'{secondary_mean:+.2f} pp.' if secondary_mean is not None
         else '- Brak budżetu wtórnego.'),
        '',
        'Jedna architektura i dwa zbiory. Spełnienie kryterium nie jest '
        'podstawą publikacji bez analizy mechanizmu i dalszych kontroli.',
    ]
    (out / 'analysis.md').write_text('\n'.join(lines) + '\n',
                                     encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'seed', 'budget', 'reference_top1',
                         'cosine_top1', 'delta_pp', 'params_after'])
        for key in sorted(comparison):
            item = comparison[key]
            writer.writerow([
                item['dataset'], item['seed'], item['budget'],
                f"{item['reference_top1']:.6f}",
                f"{item['cosine_top1']:.6f}",
                f"{item['delta_pp']:.4f}",
                item['params_after'],
            ])
    print(f'E018_ANALYSIS={out}', flush=True)
    print(f'E018_PRIMARY={"YES" if passed else "NO"}', flush=True)
    print('E018_ANALYSIS_OK', flush=True)


if __name__ == '__main__':
    main()
