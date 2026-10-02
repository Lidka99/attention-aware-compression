"""2026-10-01_E002_obserwacja_se: analiza stałego tła na zapisanych mapach.
Wejścia: niezmienione rank*.npz i wcześniejszy wybór par; config.json.
Wyjścia: nowe tabele rozkładu energii/cosinusów, podsumowanie i wykresy.
Protokół: OPIS.md obok tego pliku; kontekst eksperymentu: ../../OPIS.md.
Wyłącznie CPU, bez odczytu checkpointu/danych i bez inferencji.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from metrics import decompose_maps

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ROOT = HERE.parents[3]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False) + '\n')


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = {'count': int(values.size), 'finite': int(finite.size),
              'undefined': int(values.size - finite.size)}
    if finite.size:
        result.update(dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                               map(float, np.quantile(finite, [0, .25, .5, .75, 1])))))
    return result


def write_table(path, columns):
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        for row in zip(*columns.values()):
            writer.writerow([v.item() if hasattr(v, 'item') else v for v in row])


def verify_source(source, cfg):
    require(digest(source / 'artifacts.sha256') == cfg['source_artifacts_sha256'],
            'Source checksum manifest mismatch')
    hashes = {}
    for line in (source / 'artifacts.sha256').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (source / relative).resolve()
        require(path.is_relative_to(source.resolve()), 'Unsafe artifact path')
        require(digest(path) == expected, f'Source artifact changed: {relative}')
        hashes[relative] = expected
    require(digest(source / 'SAMPLES.json') == cfg['sample_manifest_sha256']
            and digest(source / 'selected_pairs.csv') == cfg['selected_pairs_sha256'],
            'Sample/pair selection manifest mismatch')
    for rank, expected in enumerate(cfg['source_rank_sha256']):
        require(hashes[f'rank{rank}.npz'] == expected, 'Raw rank mismatch')
    source_manifest = read_json(source / 'manifest.json')
    collection = read_json(source / 'collection.json')
    require(source_manifest['status'] == 'completed' and source_manifest['exit_code'] == 0
            and len(collection['ranks']) == 4
            and [r['rank'] for r in collection['ranks']] == list(range(4))
            and all(r['count'] == 250 for r in collection['ranks'])
            and not collection['weights_changed'], 'Invalid source GPU provenance')
    return hashes


def load_inputs(source, cfg):
    samples = read_json(source / 'SAMPLES.json')
    shards = []
    for rank in range(4):
        with np.load(source / f'rank{rank}.npz', allow_pickle=False) as data:
            shard = {k: data[k].copy() for k in ('ids', 'targets', 'h', 'z')}
        expected = samples['records'][rank::4]
        require(shard['ids'].tolist() == [r['id'] for r in expected]
                and shard['targets'].tolist() == [r['target'] for r in expected], 'Rank ID/label mismatch')
        require(shard['h'].shape == shard['z'].shape == (250, 480, 6, 6)
                and shard['h'].dtype == shard['z'].dtype == np.float32, 'Raw shape/dtype mismatch')
        shards.append(shard)
    merged = {key: np.concatenate([shard[key] for shard in shards]) for key in shards[0]}
    order = np.argsort(merged['ids'])
    merged = {key: value[order] for key, value in merged.items()}
    require(len(np.unique(merged['ids'])) == cfg['sample_count']
            and merged['ids'].tolist() == [r['id'] for r in samples['records']]
            and merged['targets'].tolist() == [r['target'] for r in samples['records']], 'Global coverage mismatch')
    require(not set(samples['halves']['A']) & set(samples['halves']['B']), 'A/B overlap')
    for half in ('A', 'B'):
        mask = np.isin(merged['ids'], samples['halves'][half])
        require(mask.sum() == cfg['half_count']
                and np.bincount(merged['targets'][mask], minlength=100).tolist() == [5] * 100,
                'Half/class coverage mismatch')
    with (source / 'selected_pairs.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    pairs = np.asarray([[int(row['c']), int(row['d'])] for row in rows], dtype=np.int64)
    selected_channels = np.unique(pairs)
    require(len(rows) == cfg['selected_pair_count'] and len(selected_channels) == cfg['selected_channel_count'],
            'Unexpected pair/channel selection size')
    return merged, samples, rows, pairs, selected_channels


def save_figure(fig, plots, name):
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.text(.015, .015, 'E002: analiza post hoc zapisanych aktywacji. Te same pary wybrane wcześniej na A; pokazano B.\n'
             'Centrowanie jest diagnostyką, nie zmianą modelu i nie dowodem bezpiecznego scalenia.', fontsize=8)
    fig.savefig(plots / f'{name}.png', dpi=160)
    fig.savefig(plots / f'{name}.pdf')
    plt.close(fig)


def make_plots(pairs, channels, chosen, plots):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, stage in enumerate(('h', 'z')):
        for col, (key, label) in enumerate((('s_global', 'Odjęcie stałej średniej'),
                                           ('s_spatial', 'Odjęcie średniej każdej mapy'))):
            table = pairs['B'][stage]
            ax = axes[row, col]
            valid = np.isfinite(table['s_raw']) & np.isfinite(table[key])
            ax.scatter(table['s_raw'][valid], table[key][valid], s=18, alpha=.8)
            lo, hi = -1, 1
            ax.plot([lo, hi], [lo, hi], '--', color='#555555', linewidth=1)
            ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Cosinus surowy', ylabel='Cosinus centrowany',
                   title=f'{stage}: {label}; określone {int(valid.sum())}/100')
            ax.grid(alpha=.25)
    fig.suptitle('Kontrola podobieństwa po centrowaniu — B')
    save_figure(fig, plots, 'centered_cosines')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for stage, ax in zip(('h', 'z'), axes):
        values = channels['B'][stage]['frac_const']
        all_finite = values[np.isfinite(values)]
        selected = values[chosen]; selected = selected[np.isfinite(selected)]
        ax.hist(all_finite, bins=np.linspace(0, 1, 21), density=True, histtype='step', label='Wszystkie 480 kanałów')
        ax.hist(selected, bins=np.linspace(0, 1, 21), density=True, histtype='step', label='Wybrane 19 kanałów')
        ax.set(xlabel='Udział energii stałego tła', ylabel='Gęstość', title=stage)
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.suptitle('Energia stałej średniej — B')
    save_figure(fig, plots, 'constant_energy')
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for stage, ax in zip(('h', 'z'), axes):
        bottom = np.zeros(len(chosen))
        for key, color, label in (('frac_const', '#9ca3af', 'Stała średnia'),
                                   ('frac_image', '#f59e0b', 'Średnia zmienna między obrazami'),
                                   ('frac_spatial', '#2563eb', 'Szczegóły przestrzenne')):
            values = channels['B'][stage][key][chosen]
            ax.bar(np.arange(len(chosen)), values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set(xticks=np.arange(len(chosen)), xticklabels=[str(c) for c in chosen],
               ylim=(0, 1.02), ylabel='Udział energii', xlabel='ID kanału', title=stage)
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Trzy składowe energii wybranych kanałów — B')
    save_figure(fig, plots, 'energy_components')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plots-dir', type=Path, required=True)
    args = parser.parse_args()
    run, plots = args.run_dir.resolve(), args.plots_dir.resolve()
    require(not (run / 'summary.json').exists() and not plots.exists(), 'Refusing to overwrite analysis')
    cfg = read_json(run / 'config.resolved.json')
    require(cfg['expected_versions'] == {'python': platform.python_version(), 'numpy': np.__version__},
            'CPU version mismatch')
    source = ROOT / cfg['source_run']
    input_hashes = verify_source(source, cfg)
    raw, samples, original_rows, pair_ids, chosen = load_inputs(source, cfg)
    channel_results, pair_results, diagnostics = {}, {}, {}
    summary = {'experiment': cfg['experiment'], 'analysis': cfg['analysis'], 'post_hoc': True,
               'source_run': cfg['source_run'], 'count': len(raw['ids']), 'unique_ids': len(np.unique(raw['ids'])),
               'half_counts': {'A': 500, 'B': 500}, 'source_gpu_count': 4, 'analysis_device': 'CPU',
               'inference_performed': False, 'selected_pair_count': len(pair_ids),
               'selected_channels': chosen.tolist(), 'channels': {}, 'pairs': {}, 'diagnostics': {},
               'max_original_cosine_abs_difference': 0.0, 'weights_changed': False}
    for half in ('A', 'B'):
        mask = np.isin(raw['ids'], samples['halves'][half])
        channel_results[half], pair_results[half], diagnostics[half] = {}, {}, {}
        summary['channels'][half], summary['pairs'][half] = {}, {}
        for stage in ('h', 'z'):
            channel, pair, diag = decompose_maps(raw[stage][mask], pair_ids, cfg)
            original = np.asarray([float(row[f'{half}_s_{stage}']) for row in original_rows])
            same_valid = np.isfinite(original) == np.isfinite(pair['s_raw'])
            require(same_valid.all(), 'Raw cosine definition mismatch')
            finite = np.isfinite(original)
            difference = float(np.max(np.abs(original[finite] - pair['s_raw'][finite]))) if finite.any() else 0.0
            require(difference <= cfg['identity_atol'], 'Raw cosines differ from source analysis')
            summary['max_original_cosine_abs_difference'] = max(summary['max_original_cosine_abs_difference'], difference)
            channel_results[half][stage], pair_results[half][stage], diagnostics[half][stage] = channel, pair, diag
            summary['channels'][half][stage] = {
                group: {key: stats(values[index]) for key, values in channel.items() if key != 'c'}
                for group, index in (('all', np.arange(480)), ('selected', chosen))}
            summary['channels'][half][stage]['majority_constant_selected'] = int(np.count_nonzero(channel['frac_const'][chosen] > .5))
            summary['pairs'][half][stage] = {key: stats(values) for key, values in pair.items() if key not in ('c', 'd')}
        channel_table = {'c': np.arange(480), 'selected': np.isin(np.arange(480), chosen)}
        pair_table = {'c': pair_ids[:, 0], 'd': pair_ids[:, 1]}
        for stage in ('h', 'z'):
            channel_table.update({f'{stage}_{key}': value for key, value in channel_results[half][stage].items() if key != 'c'})
            pair_table.update({f'{stage}_{key}': value for key, value in pair_results[half][stage].items() if key not in ('c', 'd')})
        write_table(run / f'channels_{half}.csv', channel_table)
        write_table(run / f'pairs_{half}.csv', pair_table)
    summary['diagnostics'] = diagnostics
    write_json(run / 'inputs_verified.json', {'source_files_sha256': input_hashes, 'source_run': cfg['source_run']})
    plots.mkdir(parents=True, exist_ok=False)
    make_plots(pair_results, channel_results, chosen, plots)
    import shutil
    for half in ('A', 'B'):
        for kind in ('channels', 'pairs'):
            shutil.copy2(run / f'{kind}_{half}.csv', plots / f'{kind}_{half}.csv')
    summary['status'] = 'completed'
    write_json(run / 'summary.json', summary)
    write_json(plots / 'summary.json', summary)
    print(f'CONSTANT_ANALYSIS=completed; original_cosine_max_difference={summary["max_original_cosine_abs_difference"]}', flush=True)
    print(json.dumps({'B_selected_h': summary['channels']['B']['h']['selected']['frac_const'],
                      'B_h_global_cosine': summary['pairs']['B']['h']['s_global'],
                      'B_h_spatial_cosine': summary['pairs']['B']['h']['s_spatial']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
