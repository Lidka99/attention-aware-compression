"""2026-10-01_E002_obserwacja_se: agreguje odczyt SE i tworzy tabele/wykresy.
Wejścia: nowy bieg z czterema rank*.npz, konfiguracja i utrwalone ID A/B.
Wyjścia: tabele wszystkich par, podsumowanie i wykresy w folderze E002.
Protokół: PROTOKOL_v1.md; szkic: OPIS.md. Analiza wyłącznie CPU, bez inferencji.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import HERE, digest, read_json, require, write_json
from pair_metrics import compute_pairs, select_pairs

METRICS = ('s_h', 's_z', 'delta_s', 'd_h', 'd_z', 'b_mean', 'b_rms', 'b_std', 'l_mean', 'l_std')


def stats(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    result = {'count': len(values), 'finite': len(finite), 'undefined': len(values) - len(finite)}
    if len(finite):
        q = np.quantile(finite, [0, .25, .5, .75, 1])
        result.update(dict(zip(('min', 'q25', 'median', 'q75', 'max'), map(float, q))))
    return result


def table(path, arrays):
    names = list(arrays)
    with Path(path).open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for values in zip(*(arrays[name] for name in names)):
            writer.writerow([v.item() if hasattr(v, 'item') else v for v in values])


def save_figure(fig, plots, stem):
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.text(.02, .01, 'E002: jeden blok, jeden model; obrazy train. Pary wybrane na A, pokazane na B.\n'
             'Obserwacja nie jest pomiarem błędu scalenia ani niezależnym testem generalizacji.', fontsize=8)
    fig.savefig(plots / f'{stem}.png', dpi=160)
    fig.savefig(plots / f'{stem}.pdf')
    plt.close(fig)


def scatter(pairs, chosen, xkey, ykey, xlabel, ylabel, plots, stem, diagonal=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = np.isfinite(pairs[xkey]) & np.isfinite(pairs[ykey])
    ax.scatter(pairs[xkey][valid], pairs[ykey][valid], s=2, alpha=.12, color='#6b7280',
               rasterized=True, label='Wszystkie określone pary na B')
    selected_valid = chosen[valid[chosen]]
    ax.scatter(pairs[xkey][selected_valid], pairs[ykey][selected_valid], s=18, alpha=.85,
               color='#dc2626', rasterized=True, label='100 par wybranych według s_h na A')
    if diagonal:
        lo = min(float(pairs[xkey][valid].min()), float(pairs[ykey][valid].min()))
        hi = max(float(pairs[xkey][valid].max()), float(pairs[ykey][valid].max()))
        ax.plot([lo, hi], [lo, hi], '--', color='#2563eb', linewidth=1)
    ax.set(xlabel=xlabel, ylabel=ylabel, title='SE features.4.1 — podzbiór B')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    save_figure(fig, plots, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plots-dir', type=Path, required=True)
    args = parser.parse_args()
    run, plots = args.run_dir.resolve(), args.plots_dir.resolve()
    cfg = read_json(run / 'config.resolved.json')
    samples = read_json(run / 'SAMPLES.json')
    require(not (run / 'summary.json').exists() and not plots.exists(), 'Refusing analysis overwrite')
    arrays = []
    for rank in range(4):
        with np.load(run / f'rank{rank}.npz', allow_pickle=False) as f:
            entry = {key: f[key].copy() for key in ('ids', 'targets', 'h', 'a', 'z')}
        records = samples['records'][rank::4]
        require(entry['ids'].tolist() == [r['id'] for r in records]
                and entry['targets'].tolist() == [r['target'] for r in records], 'Rank ID/label mismatch')
        require(entry['h'].shape == entry['z'].shape == (250, 480, 6, 6)
                and entry['a'].shape == (250, 480)
                and all(entry[k].dtype == np.float32 for k in ('h', 'a', 'z')), 'Invalid raw shape/precision')
        require(np.allclose(entry['z'], entry['a'][:, :, None, None] * entry['h'],
                            atol=cfg['hook_identity_atol'], rtol=cfg['hook_identity_rtol']), 'Raw SE identity mismatch')
        arrays.append(entry)
    merged = {key: np.concatenate([entry[key] for entry in arrays]) for key in arrays[0]}
    order = np.argsort(merged['ids'])
    merged = {key: values[order] for key, values in merged.items()}
    require(merged['ids'].tolist() == [r['id'] for r in samples['records']]
            and len(np.unique(merged['ids'])) == 1000, 'Global ID coverage mismatch')
    require(merged['targets'].tolist() == [r['target'] for r in samples['records']], 'Global label mismatch')
    results, channels, diagnostics = {}, {}, {}
    for half in ('A', 'B'):
        mask = np.isin(merged['ids'], samples['halves'][half])
        require(mask.sum() == 500, 'Half count mismatch')
        results[half], channels[half], diagnostics[half] = compute_pairs(
            merged['h'][mask], merged['z'][mask], merged['a'][mask], cfg)
        require(len(results[half]['c']) == cfg['total_channel_pairs'], 'Pair count mismatch')
        table(run / f'pairs_{half}.csv', results[half])
        table(run / f'channels_{half}.csv', channels[half])
        np.savez(run / f'pairs_{half}.npz', **results[half])
    chosen = select_pairs(results['A'], cfg['pairs_selected_on_A'])
    selected = {'c': results['A']['c'][chosen], 'd': results['A']['d'][chosen]}
    for half in ('A', 'B'):
        selected.update({f'{half}_{key}': results[half][key][chosen] for key in METRICS})
    table(run / 'selected_pairs.csv', selected)
    summary = {'status': 'completed', 'experiment': cfg['experiment'], 'count': 1000,
               'unique_ids': 1000, 'half_counts': {'A': 500, 'B': 500},
               'sample_manifest_sha256': digest(run / 'SAMPLES.json'),
               'selected_pair_count': len(chosen),
               'selected_channel_count': len(np.unique(np.concatenate([selected['c'], selected['d']]))),
               'diagnostics': diagnostics,
               'all_pairs': {half: {key: stats(results[half][key]) for key in METRICS} for half in ('A', 'B')},
               'selected_pairs': {half: {key: stats(results[half][key][chosen]) for key in METRICS}
                                  for half in ('A', 'B')},
               'channels': {half: {key: stats(values) for key, values in channels[half].items() if key != 'c'}
                            for half in ('A', 'B')},
               'raw_sha256': {f'rank{r}.npz': digest(run / f'rank{r}.npz') for r in range(4)},
               'weights_changed': False, 'compression_measured': False,
               'official_test_used': False, 'reserve_used': False, 'dev_used': False}
    plots.mkdir(parents=True, exist_ok=False)
    scatter(results['B'], chosen, 's_h', 's_z', 'Cosinus przed SE (s_h)', 'Cosinus po SE (s_z)',
            plots, 'cosine_before_after', True)
    scatter(results['B'], chosen, 's_h', 'l_std', 'Cosinus przed SE (s_h)',
            'Zmienność log-stosunku wag SE (std l)', plots, 'cosine_gate_variability')
    scatter(results['B'], chosen, 'd_h', 'd_z', 'Różnica aktywacji przed SE (d_h)',
            'Różnica aktywacji po SE (d_z)', plots, 'difference_before_after', True)
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = np.linspace(0, max(float(results['B']['l_std'].max()), 1e-12), 51)
    ax.hist(results['B']['l_std'], bins=edges, density=True, histtype='step', label='Wszystkie pary na B')
    ax.hist(results['B']['l_std'][chosen], bins=edges, density=True, histtype='step',
            label='Pary wybrane na A')
    ax.set(xlabel='Std log-stosunku wag', ylabel='Gęstość', title='Zmienność względnego ważenia SE na B')
    ax.legend(); ax.grid(alpha=.25)
    save_figure(fig, plots, 'gate_ratio_distribution')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].hist(channels['B']['a_mean'], bins=30)
    axes[1].hist(channels['B']['a_std'], bins=30)
    axes[0].set(xlabel='Średnia waga SE kanału', ylabel='Liczba kanałów')
    axes[1].set(xlabel='Std wagi SE kanału', ylabel='Liczba kanałów')
    fig.suptitle('Wagi 480 kanałów SE na B')
    save_figure(fig, plots, 'channel_gates')
    write_json(plots / 'summary.json', summary)
    table(plots / 'selected_pairs.csv', selected)
    write_json(run / 'summary.json', summary)
    print(f'Analysis completed: pairs per half={len(results["A"]["c"])}; selected={len(chosen)}; '
          f'unique selected channels={summary["selected_channel_count"]}', flush=True)


if __name__ == '__main__':
    main()
