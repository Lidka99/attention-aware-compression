"""E004: analiza wyników wspólnego dostrajania i wykresy (tylko CPU).
Eksperyment: 2026-10-10_E004_wspolne_dostrajanie; protokół: PROTOKOL_v1.md i OPIS.md.
Wejścia: nowy runs/E004 z metrics_<wariant>.jsonl i dev_final_<wariant>.npz.
Wyjścia: summary.json w biegu i folderze wykresów, metrics.csv, 3 wykresy PNG/PDF.
Sparowany bootstrap POST−PRE (jak E003), porównanie z wynikami E003 i BASE.
Bez GPU, bez modelu, bez nowej inferencji; nie nadpisuje istniejących plików.
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from support_e004 import HERE, ROOT, E003, digest, read_json, write_json, require

sys.path.insert(0, str(E003))
from summarize import paired_bootstrap

MODES = ('BASE', 'PRE', 'POST', 'RANDOM')
COLORS = {'BASE': '#6b7280', 'PRE': '#2563eb', 'POST': '#dc2626', 'RANDOM': '#16a34a'}
LIMITATIONS = (
    'Wyniki dotyczą jednego modelu bazowego E001 (seed 42) i jednego wspólnego dostrojenia; '
    'nie są niezależnym testem generalizacji ani dowodem przewagi metody.',
    'Dev wybierał checkpoint E001 i był używany w E002/E003; nie jest niezależnym testem. '
    'Bootstrap opisuje zmienność obrazów przy jednym modelu, nie zmienność między treningami.',
    'Budżet 120 epok zaadaptowano z protokołu GReg (ICLR 2021) dla VGG/ResNet na CIFAR; '
    'to adaptacja, nie reprodukcja ani twierdzenie o optymalności dla EfficientNet-B0.',
    'Wynik główny to epoka 120 (ostatnia), wspólna dla wszystkich wariantów; epoka najlepsza '
    'na dev jest wyłącznie diagnostyką. Brak wczesnego zatrzymania i wyboru po fakcie.',
    'Cztery warianty to jeden przebieg sekwencyjny na 4 GPU, nie cztery niezależne modele.',
)


def rows_of(run, variant, cfg):
    path = run / f'metrics_{variant}.jsonl'
    rows = [__import__('json').loads(line) for line in path.read_text().splitlines()]
    require([r['epoch'] for r in rows] == list(range(1, cfg['fine_tune']['epochs'] + 1)),
            f'{variant}: epoch log mismatch')
    for row in rows:
        require(row['dev_count'] == cfg['data']['dev_count'] and row['dev_correct'] == int(
            round(row['dev_top1'] * row['dev_count'])), f'{variant}: metric mismatch')
        require(row['train_count'] == 39936, f'{variant}: unexpected train_count')
    return rows


def load_final(run, variant, cfg):
    with np.load(run / f'dev_final_{variant}.npz', allow_pickle=False) as data:
        entry = {k: data[k].copy() for k in data.files}
    require(entry['ids'].size == cfg['data']['dev_count']
            and len(np.unique(entry['ids'])) == cfg['data']['dev_count'], f'{variant}: ID mismatch')
    require(np.bincount(entry['targets'], minlength=100).tolist() == [50] * 100,
            f'{variant}: dev balance mismatch')
    require(entry['logits'].shape == (cfg['data']['dev_count'], 100)
            and entry['logits'].dtype == np.float32 and np.isfinite(entry['logits']).all(),
            f'{variant}: logits invalid')
    require(np.array_equal(entry['preds'], entry['logits'].argmax(1)),
            f'{variant}: preds differ from argmax(logits)')
    correct = entry['preds'] == entry['targets']
    return entry, correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plots-dir', type=Path, required=True)
    args = parser.parse_args()
    run, plots = args.run_dir.resolve(), args.plots_dir.resolve()
    require(not (run / 'summary.json').exists() and not plots.exists(), 'Refusing to overwrite')
    cfg = read_json(run / 'config.resolved.json')
    metrics = {v: rows_of(run, v, cfg) for v in MODES}
    entries, final, correct = {}, {}, {}
    ids_ref = None
    for variant in MODES:
        entry, flags = load_final(run, variant, cfg)
        if ids_ref is None:
            ids_ref = entry['ids']
        require(np.array_equal(entry['ids'], ids_ref), 'Dev order differs between variants')
        final[variant] = metrics[variant][-1]
        require(int(flags.sum()) == final[variant]['dev_correct'],
                f'{variant}: npz and epoch-120 metrics disagree')
        entries[variant], correct[variant] = entry, flags
    targets = entries['BASE']['targets']
    identity = paired_bootstrap(correct['BASE'], correct['BASE'], targets, 200, cfg['bootstrap_seed'])
    require(identity['delta_correct'] == 0 and identity['ci95_correct'] == [0, 0],
            'Bootstrap sanity check failed')
    comparisons = {}
    for baseline, challenger in (('PRE', 'POST'), ('RANDOM', 'POST'), ('RANDOM', 'PRE')):
        comparisons[f'{challenger}_minus_{baseline}'] = paired_bootstrap(
            correct[baseline], correct[challenger], targets,
            cfg['bootstrap_repetitions'], cfg['bootstrap_seed'])
    start = cfg['reference_start_accuracy']
    summary = {'experiment': cfg['experiment'], 'run_id': run.name, 'status': 'completed',
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'epochs': cfg['fine_tune']['epochs'], 'variants': {},
               'start_after_E003_top1': start, 'comparisons': comparisons,
               'limitations': LIMITATIONS}
    for variant in MODES:
        best = max(metrics[variant], key=lambda r: r['dev_top1'])
        summary['variants'][variant] = {
            'start_top1_after_E003': start[variant], 'final_top1': final[variant]['dev_top1'],
            'final_correct': final[variant]['dev_correct'], 'final_nll': final[variant]['dev_nll'],
            'final_epoch': final[variant]['epoch'],
            'best_dev_top1_diagnostic': best['dev_top1'], 'best_dev_epoch_diagnostic': best['epoch'],
            'gain_vs_E003_pp': 100 * (final[variant]['dev_top1'] - start[variant])}
    plots.mkdir(parents=True, exist_ok=False)
    epochs = list(range(1, cfg['fine_tune']['epochs'] + 1))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant in MODES:
        ax.plot(epochs, [100 * r['dev_top1'] for r in metrics[variant]], color=COLORS[variant],
                linewidth=1.6, label=f"{variant} (start E003: {100 * start[variant]:.2f}%)")
    ax.axvline(cfg['fine_tune']['epochs'], color='#555555', linestyle='--', linewidth=1)
    ax.set(xlabel='Epoka dostrajania', ylabel='Dev top-1 [%]',
           title='E004: dev top-1 w trakcie wspólnego 120-epokowego dostrajania')
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / 'dev_top1_curves.png', dpi=150)
    fig.savefig(plots / 'dev_top1_curves.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant in MODES:
        ax.plot(epochs, [r['dev_nll'] for r in metrics[variant]], color=COLORS[variant],
                linewidth=1.6, label=variant)
    ax.set(xlabel='Epoka dostrajania', ylabel='Dev NLL (strata)',
           title='E004: dev NLL w trakcie wspólnego dostrajania')
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / 'dev_nll_curves.png', dpi=150)
    fig.savefig(plots / 'dev_nll_curves.pdf')
    plt.close(fig)
    delta = [m['dev_top1'] - p['dev_top1'] for m, p in zip(metrics['POST'], metrics['PRE'])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, [100 * d for d in delta], color='#111827', linewidth=1.6)
    ax.axhline(0, color='#555555', linestyle='--', linewidth=1)
    ax.set(xlabel='Epoka dostrajania', ylabel='POST − PRE [pp]',
           title='E004: różnica trafności POST−PRE w trakcie dostrajania (dev, 5000 obrazów)')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(plots / 'post_minus_pre_curve.png', dpi=150)
    fig.savefig(plots / 'post_minus_pre_curve.pdf')
    plt.close(fig)
    with (run / 'metrics_final.csv').open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['variant', 'start_top1_E003', 'final_epoch', 'final_correct',
                         'final_top1', 'final_nll', 'best_dev_top1', 'best_dev_epoch'])
        for variant in MODES:
            entry = summary['variants'][variant]
            writer.writerow([variant, entry['start_top1_after_E003'], entry['final_epoch'],
                             entry['final_correct'], entry['final_top1'], entry['final_nll'],
                             entry['best_dev_top1_diagnostic'], entry['best_dev_epoch_diagnostic']])
    import shutil
    shutil.copy2(run / 'metrics_final.csv', plots / 'metrics_final.csv')
    plots_hash = {p.name: digest(p) for p in sorted(plots.iterdir())}
    summary['plots_sha256'] = plots_hash
    write_json(run / 'summary.json', summary)
    write_json(plots / 'summary.json', summary)
    print(f"E004_SUMMARY=completed final=" + ', '.join(
        f"{v}:{100 * summary['variants'][v]['final_top1']:.2f}%" for v in MODES), flush=True)
    for name, comp in comparisons.items():
        print(f"{name}: delta={comp['delta_correct']} CI={comp['ci95_correct']}", flush=True)


if __name__ == '__main__':
    main()
