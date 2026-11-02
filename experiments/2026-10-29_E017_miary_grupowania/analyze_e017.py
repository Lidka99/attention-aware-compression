"""E017: analiza benchmarku miar grupowania (CPU).

Eksperyment: 2026-10-29_E017_miary_grupowania; pełny protokół: OPIS.md.
Wejście: ukończony katalog ``runs/E017/<RUN>`` z 42 summary.
Wyjście: ``analysis.json``, ``analysis.md`` i ``summary.csv`` w nowym katalogu
``runs/E017/analizy/<UTC>_<token>/``; decyzja według kryterium z OPIS.md.
Skrypt nie trenuje i nie używa GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e017_common as common  # noqa: E402

RUNS = common.ROOT / 'runs' / 'E017'


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


def summary_path(run, task):
    return (run / 'tasks' / task['dataset'] / task['measure']
            / f"b{int(round(task['budget'] * 100)):02d}" / 'summary.json')


def load_rows(run, cfg):
    """Wczytuje i weryfikuje 42 summary."""
    rows = []
    for task in common.build_tasks(cfg):
        path = summary_path(run, task)
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
            'measure': task['measure'],
            'role': task['role'],
            'budget': float(task['budget']),
            'params_before': int(summary['params_before']),
            'params_after': int(summary['params_after']),
            'reduction': float(summary['reduction_fraction']),
            'baseline_top1': float(summary['baseline']['metrics']['top1']),
            'pre_top1': float(summary['compressed']['pre_bn']['top1']),
            'post_top1': float(summary['compressed']['post_bn']['top1']),
            'post_nll': float(summary['compressed']['post_bn']['nll']),
            'agreement': float(
                summary['compressed']['post_bn_vs_baseline'][
                    'prediction_agreement']),
            'groups': summary['groups'],
        })
    require(len(rows) == 42, 'Zła liczba wyników')
    for dataset in cfg['datasets']:
        for budget in cfg['budgets']:
            values = {row['params_after'] for row in rows
                      if row['dataset'] == dataset['name']
                      and row['budget'] == budget}
            require(len(values) == 1, 'Nierówne budżety między miarami')
    return rows


def build_table(rows):
    """Buduje tabelę dataset/budget/miara oraz średnią kontroli losowej."""
    table = {}
    for row in rows:
        key = (row['dataset'], row['budget'])
        table.setdefault(key, {})[row['measure']] = row
    output = {}
    for (dataset, budget), values in table.items():
        random_rows = [values[f'random{i}'] for i in (1, 2, 3)]
        random_mean = sum(item['post_top1'] for item in random_rows) / 3.0
        reference = values['reference']['post_top1']
        output[f'{dataset}|{budget:.2f}'] = {
            'dataset': dataset,
            'budget': budget,
            'reference': reference,
            'random_mean': random_mean,
            'random_top1': [item['post_top1'] for item in random_rows],
            'measures': {
                name: {
                    'post_top1': item['post_top1'],
                    'pre_top1': item['pre_top1'],
                    'post_nll': item['post_nll'],
                    'agreement': item['agreement'],
                    'params_after': item['params_after'],
                    'delta_vs_reference_pp': 100.0 * (
                        item['post_top1'] - reference),
                    'delta_vs_random_pp': 100.0 * (
                        item['post_top1'] - random_mean),
                }
                for name, item in values.items()
            },
        }
    return output


def decide(table, cfg):
    """Stosuje zamrożone kryterium sukcesu z OPIS.md."""
    success = cfg['success']
    required = [float(value) for value in success['required_budgets']]
    datasets = success['required_datasets']
    min_reference = float(success['min_gain_vs_reference_pp'])
    min_random = float(success['min_gain_vs_random_pp'])
    candidates = [item['name'] for item in cfg['measures']
                  if item['role'] == 'candidate']
    decisions = {}
    for name in candidates:
        checks = {}
        passed = True
        for dataset in datasets:
            for budget in required:
                cell = table[f'{dataset}|{budget:.2f}']['measures'][name]
                ok = (cell['delta_vs_reference_pp'] >= min_reference
                      and cell['delta_vs_random_pp'] >= min_random)
                checks[f'{dataset}|{budget:.2f}'] = {
                    'delta_vs_reference_pp': cell['delta_vs_reference_pp'],
                    'delta_vs_random_pp': cell['delta_vs_random_pp'],
                    'passed': ok,
                }
                passed = passed and ok
        decisions[name] = {'passed': passed, 'checks': checks}
    return {
        'criteria': success,
        'candidates': decisions,
        'confirmed': [name for name, value in decisions.items()
                      if value['passed']],
    }


def write_markdown(path, table, decision):
    """Zapisuje czytelny raport Markdown."""
    lines = [
        '# E017 — benchmark miar grupowania: analiza',
        '',
        'Dev top-1 po korekcie BN [%]. Jedna ocena na zbiór (seed42); '
        'to etap rozwojowy, nie replikacja.',
        '',
        '## Wyniki',
        '',
        '| dataset | budżet | reference | random (średnia) | cosine | '
        'standardized | bn_effective |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for key in sorted(table):
        cell = table[key]
        measures = cell['measures']
        lines.append(
            f"| {cell['dataset']} | {int(round(cell['budget']*100))}% | "
            f"{100*cell['reference']:.2f} | {100*cell['random_mean']:.2f} | "
            f"{100*measures['cosine']['post_top1']:.2f} | "
            f"{100*measures['standardized']['post_top1']:.2f} | "
            f"{100*measures['bn_effective']['post_top1']:.2f} |")
    lines += [
        '',
        '## Różnice względem reference [pp]',
        '',
        '| dataset | budżet | cosine | standardized | bn_effective | '
        'random1..3 (zakres) |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for key in sorted(table):
        cell = table[key]
        measures = cell['measures']
        random_values = [100 * (value - cell['reference'])
                          for value in cell['random_top1']]
        lines.append(
            f"| {cell['dataset']} | {int(round(cell['budget']*100))}% | "
            f"{measures['cosine']['delta_vs_reference_pp']:+.2f} | "
            f"{measures['standardized']['delta_vs_reference_pp']:+.2f} | "
            f"{measures['bn_effective']['delta_vs_reference_pp']:+.2f} | "
            f"{min(random_values):+.2f}…{max(random_values):+.2f} |")
    lines += [
        '',
        '## Decyzja',
        '',
    ]
    for name, value in decision['candidates'].items():
        lines.append(
            f"- **{name}**: {'SPEŁNIA' if value['passed'] else 'NIE SPEŁNIA'} "
            'kryterium (+0,50 pp vs reference i vs random przy 50% i 70% '
            'na obu zbiorach).')
    lines += [
        '',
        'Jedna ocena na zbiór nie pozwala wnioskować o istotności. '
        'Spełnienie kryterium oznacza wyłącznie rekomendację replikacji '
        'na pozostałych seedach.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    manifest = common.read_json(run / 'manifest.json')
    require(manifest.get('status') == 'completed',
            'Bieg nie jest ukończony')
    require(manifest.get('official_test_used') is False,
            'Bieg użył oficjalnego testu')
    require(manifest.get('reserve_used') is False, 'Bieg użył rezerwy')
    cfg = common.load_config()
    require(manifest.get('config_sha256')
            == common.digest(common.CONFIG_PATH),
            'Config biegu różni się od zamrożonego')
    rows = load_rows(run, cfg)
    table = build_table(rows)
    decision = decide(table, cfg)
    out = output_dir()
    analysis = {
        'run': run.name,
        'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH,
        'status': 'completed',
        'table': table,
        'decision': decision,
        'rows': [{key: value for key, value in row.items()
                  if key != 'groups'} for row in rows],
    }
    common.write_json_atomic(out / 'analysis.json', analysis)
    write_markdown(out / 'analysis.md', table, decision)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'budget', 'measure', 'pre_top1',
                         'post_top1', 'post_nll', 'agreement',
                         'delta_vs_reference_pp', 'delta_vs_random_pp'])
        for key in sorted(table):
            cell = table[key]
            for name, value in sorted(cell['measures'].items()):
                writer.writerow([
                    cell['dataset'], cell['budget'], name,
                    f"{value['pre_top1']:.6f}",
                    f"{value['post_top1']:.6f}",
                    f"{value['post_nll']:.6f}",
                    f"{value['agreement']:.6f}",
                    f"{value['delta_vs_reference_pp']:.4f}",
                    f"{value['delta_vs_random_pp']:.4f}",
                ])
    print(f'E017_ANALYSIS={out}', flush=True)
    print(f"E017_CONFIRMED={decision['confirmed']}", flush=True)
    print('E017_ANALYSIS_OK', flush=True)


if __name__ == '__main__':
    main()
