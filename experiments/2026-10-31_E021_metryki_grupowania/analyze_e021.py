"""E021: analiza benchmarku metryk grupowania (CPU).

Eksperyment: 2026-10-31_E021_metryki_grupowania; pełny protokół: OPIS.md.
Wejście: ukończony katalog z 72 summary. Wyjście: analysis.json/md i CSV;
decyzja według kryterium zapisanego w OPIS.md. Bez treningu i GPU.
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

import e021_common as common  # noqa: E402

RUNS = common.ROOT / 'runs' / 'E021'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def output_dir():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    manifest = common.read_json(run / 'manifest.json')
    require(manifest.get('status') == 'completed', 'Bieg nieukończony')
    require(manifest.get('official_test_used') is False, 'Użyto testu')
    require(manifest.get('reserve_used') is False, 'Użyto rezerwy')
    cfg = common.load_config()
    require(manifest.get('config_sha256') == common.digest(common.CONFIG_PATH),
            'Config biegu różni się od zamrożonego')

    rows = []
    for task in common.build_tasks(cfg):
        path = (run / 'tasks' / task['dataset'] / task['variant']
                / f"b{int(round(task['budget'] * 100)):02d}" / 'summary.json')
        require(path.is_file(), f'Brak summary: {path}')
        summary = common.read_json(path)
        require(summary.get('status') == 'completed',
                f"{task['name']}: {summary.get('status')}")
        rows.append({
            'dataset': task['dataset'], 'variant': task['variant'],
            'role': task['role'], 'budget': task['budget'],
            'params_after': summary['params_after'],
            'pre_top1': float(summary['compressed']['pre_bn']['top1']),
            'post_top1': float(summary['compressed']['post_bn']['top1']),
            'post_nll': float(summary['compressed']['post_bn']['nll']),
            'agreement': float(summary['compressed'][
                'post_bn_vs_baseline']['prediction_agreement']),
        })
    require(len(rows) == 72, 'Oczekiwano 72 wyników')

    table = {}
    for row in rows:
        table.setdefault((row['dataset'], row['budget']), {})[
            row['variant']] = row
    comparison = {}
    for (dataset, budget), values in table.items():
        reference = values['reference_hkmeans']['post_top1']
        random = values['random_balanced']['post_top1']
        comparison[f'{dataset}|{budget:.2f}'] = {
            'dataset': dataset, 'budget': budget,
            'reference': reference, 'random': random,
            'variants': {
                name: {
                    'post_top1': item['post_top1'],
                    'pre_top1': item['pre_top1'],
                    'post_nll': item['post_nll'],
                    'agreement': item['agreement'],
                    'params_after': item['params_after'],
                    'delta_vs_reference_pp': 100.0 * (
                        item['post_top1'] - reference),
                    'delta_vs_random_pp': 100.0 * (
                        item['post_top1'] - random),
                }
                for name, item in values.items()
            },
        }

    success = cfg['success']
    budgets = [float(value) for value in success['required_budgets']]
    candidates = [item['name'] for item in cfg['variants']
                  if item['role'] == 'candidate']
    min_ref = float(success['min_gain_vs_reference_pp'])
    min_rand = float(success['min_gain_vs_random_pp'])
    decisions = {}
    for name in candidates:
        per_dataset = {}
        for entry in cfg['datasets']:
            ok = True
            deltas = {}
            for budget in budgets:
                cell = comparison[f"{entry['tag']}|{budget:.2f}"][
                    'variants'][name]
                passed = (cell['delta_vs_reference_pp'] >= min_ref
                          and cell['delta_vs_random_pp'] >= min_rand)
                deltas[f'{budget:.2f}'] = {
                    'vs_reference_pp': cell['delta_vs_reference_pp'],
                    'vs_random_pp': cell['delta_vs_random_pp'],
                    'passed': passed,
                }
                ok = ok and passed
            per_dataset[entry['tag']] = {'passed': ok, 'budgets': deltas}
        count = sum(1 for value in per_dataset.values() if value['passed'])
        decisions[name] = {
            'per_dataset': per_dataset,
            'datasets_passed': count,
            'strong': count == len(cfg['datasets']),
            'promising': count >= int(success['promising_min_datasets']),
        }
    confirmed = [name for name, value in decisions.items()
                 if value['strong']]
    promising = [name for name, value in decisions.items()
                 if value['promising'] and not value['strong']]

    out = output_dir()
    analysis = {
        'run': run.name, 'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH, 'comparison': comparison,
        'decisions': decisions, 'confirmed_strong': confirmed,
        'promising': promising,
    }
    common.write_json_atomic(out / 'analysis.json', analysis)

    lines = [
        '# E021 — benchmark metryk grupowania: analiza', '',
        'Dev top-1 po BN [%], seed42. Jedna ocena na zbiór; to etap '
        'rozwojowy, nie replikacja.', '',
        '| dataset | budżet | reference | random | L1 | L2 | cosine | '
        'correlation | Ward | standardized |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    order = ['reference_hkmeans', 'random_balanced', 'agg_l1', 'agg_l2',
             'agg_cosine', 'agg_correlation', 'ward_l2',
             'standardized_hkmeans']
    for key in sorted(comparison):
        cell = comparison[key]
        values = cell['variants']
        lines.append(
            f"| {cell['dataset']} | {int(round(cell['budget']*100))}% | "
            + ' | '.join(f"{100*values[name]['post_top1']:.2f}"
                         for name in order) + ' |')
    lines += ['', '## Różnice vs reference [pp]', '',
              '| dataset | budżet | L1 | L2 | cosine | correlation | Ward | '
              'standardized |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in sorted(comparison):
        cell = comparison[key]
        values = cell['variants']
        names = ['agg_l1', 'agg_l2', 'agg_cosine', 'agg_correlation',
                 'ward_l2', 'standardized_hkmeans']
        lines.append(
            f"| {cell['dataset']} | {int(round(cell['budget']*100))}% | "
            + ' | '.join(
                f"{values[name]['delta_vs_reference_pp']:+.2f}"
                for name in names) + ' |')
    lines += ['', '## Decyzja', '']
    for name, value in decisions.items():
        lines.append(
            f"- **{name}**: {value['datasets_passed']}/3 datasetów spełnia "
            f"próg +{min_ref:.2f} pp vs reference i vs random przy 50% i 70%."
            + (' **STRONG**' if value['strong'] else
               (' **PROMISING**' if value['promising'] else '')))
    lines += [
        '', 'Brak przedziałów ufności: jeden seed. Wynik „strong” oznacza '
        'wyłącznie rekomendację replikacji na seedach 43–45, nie publikację.']
    (out / 'analysis.md').write_text('\n'.join(lines) + '\n',
                                     encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'budget', 'variant', 'post_top1',
                         'post_nll', 'delta_vs_reference_pp',
                         'delta_vs_random_pp', 'params_after'])
        for key in sorted(comparison):
            cell = comparison[key]
            for name, value in sorted(cell['variants'].items()):
                writer.writerow([
                    cell['dataset'], cell['budget'], name,
                    f"{value['post_top1']:.6f}", f"{value['post_nll']:.6f}",
                    f"{value['delta_vs_reference_pp']:.4f}",
                    f"{value['delta_vs_random_pp']:.4f}",
                    value['params_after']])
    print(f'E021_ANALYSIS={out}', flush=True)
    print(f'E021_STRONG={confirmed}', flush=True)
    print(f'E021_PROMISING={promising}', flush=True)
    print('E021_ANALYSIS_OK', flush=True)


if __name__ == '__main__':
    main()
