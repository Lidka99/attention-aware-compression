"""2026-10-01_E002_obserwacja_se: kontrola szablonu i zmian odpowiedzi.
Wejścia: zapisane h/z, utrwalone pary/ID i wyniki kontroli średniej E002.
Wyjścia: nowe tabele, podsumowanie i wykresy, bez modyfikowania źródła.
Protokół: OPIS.md obok; kontekst: ../../OPIS.md. Wyłącznie analiza CPU.
"""
import argparse
import csv
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ROOT = HERE.parents[3]
SHARED = HERE.parent / '2026-10-02_skladowa_stala'
sys.path.insert(0, str(SHARED))
from analyze import digest, read_json, write_json, require, stats, write_table, verify_source, load_inputs
from template_metrics import decompose_templates


def code_hashes():
    return {'local': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
            'dependencies': {str(p.relative_to(ROOT)): digest(p)
                             for p in (SHARED / 'analyze.py', SHARED / 'metrics.py')}}


def verify_reference(path, cfg):
    require(digest(path / 'artifacts.sha256') == cfg['constant_reference_artifacts_sha256'],
            'Constant reference checksum manifest mismatch')
    hashes = {}
    for line in (path / 'artifacts.sha256').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        full = (path / relative).resolve()
        require(full.is_relative_to(path.resolve()) and digest(full) == expected,
                f'Constant reference artifact changed: {relative}')
        hashes[relative] = expected
    require(read_json(path / 'manifest.json')['status'] == 'completed', 'Unfinished reference analysis')
    return hashes


def plot_save(fig, folder, name):
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.text(.015, .015, 'E002: analiza post hoc, te same 100 par wybrane na A; prezentacja B.\n'
             'Szablon szacowany na danej połowie. Podobieństwo nie dowodzi bezpiecznego scalenia.', fontsize=8)
    fig.savefig(folder / f'{name}.png', dpi=160)
    fig.savefig(folder / f'{name}.pdf')
    plt.close(fig)


def make_plots(pairs, channels, selected, folder):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, stage in enumerate(('h', 'z')):
        for col, (key, title) in enumerate((('s_dynamic', 'Zmiany względem średniej mapy'),
                                            ('s_interaction', 'Zmiany wzoru przestrzennego'))):
            p = pairs['B'][stage]; ax = axes[row, col]
            valid = np.isfinite(p['s_raw']) & np.isfinite(p[key])
            ax.scatter(p['s_raw'][valid], p[key][valid], s=18, alpha=.8)
            ax.plot([-1, 1], [-1, 1], '--', color='#555555', linewidth=1)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='Cosinus surowy', ylabel='Cosinus zmian',
                   title=f'{stage}: {title}\nokreślone {int(valid.sum())}/100')
            ax.grid(alpha=.25)
    fig.suptitle('Podobieństwo odpowiedzi po odjęciu szablonu — B')
    plot_save(fig, folder, 'response_cosines')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for stage, ax in zip(('h', 'z'), axes):
        values = channels['B'][stage]['frac_template']
        ax.hist(values[np.isfinite(values)], bins=np.linspace(0, 1, 21), density=True,
                histtype='step', label='Wszystkie 480 kanałów')
        chosen = values[selected]; chosen = chosen[np.isfinite(chosen)]
        ax.hist(chosen, bins=np.linspace(0, 1, 21), density=True, histtype='step', label='Wybrane 19 kanałów')
        ax.set(xlabel='Udział energii całej średniej mapy', ylabel='Gęstość', title=stage)
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.suptitle('Energia szablonu przestrzennego — B')
    plot_save(fig, folder, 'template_energy')
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for stage, ax in zip(('h', 'z'), axes):
        bottom = np.zeros(len(selected))
        for key, color, label in (('frac_const', '#9ca3af', 'Stała średnia'),
                                  ('frac_image', '#f59e0b', 'Zmienna średnia obrazu'),
                                  ('frac_spatial_template', '#10b981', 'Stały wzór przestrzenny'),
                                  ('frac_interaction', '#2563eb', 'Wzór zmienny między obrazami')):
            values = channels['B'][stage][key][selected]
            ax.bar(np.arange(len(selected)), values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set(xticks=np.arange(len(selected)), xticklabels=[str(c) for c in selected],
               ylim=(0, 1.02), xlabel='ID kanału', ylabel='Udział energii', title=stage)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Cztery składowe energii wybranych kanałów — B')
    plot_save(fig, folder, 'four_components')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plots-dir', type=Path, required=True)
    args = parser.parse_args()
    run, plots = args.run_dir.resolve(), args.plots_dir.resolve()
    require(not plots.exists() and not (run / 'summary.json').exists(), 'Refusing analysis overwrite')
    cfg = read_json(run / 'config.resolved.json')
    require(cfg['expected_versions'] == {'python': platform.python_version(), 'numpy': np.__version__},
            'CPU version mismatch')
    source, reference = ROOT / cfg['source_run'], ROOT / cfg['constant_reference_run']
    input_hashes = verify_source(source, cfg)
    reference_hashes = verify_reference(reference, cfg)
    raw, samples, original_rows, pair_ids, selected = load_inputs(source, cfg)
    channels, pairs, diagnostics = {}, {}, {}
    summary = {'experiment': cfg['experiment'], 'analysis': cfg['analysis'], 'post_hoc': True,
               'source_run': cfg['source_run'], 'count': len(raw['ids']), 'unique_ids': len(np.unique(raw['ids'])),
               'half_counts': {'A': 500, 'B': 500}, 'source_gpu_count': 4, 'analysis_device': 'CPU',
               'inference_performed': False, 'weights_changed': False, 'selected_channels': selected.tolist(),
               'selected_pair_count': len(pair_ids), 'channels': {}, 'pairs': {}, 'diagnostics': {},
               'max_original_cosine_abs_difference': 0.0, 'max_previous_fraction_abs_difference': 0.0}
    for half in ('A', 'B'):
        mask = np.isin(raw['ids'], samples['halves'][half])
        with (reference / f'channels_{half}.csv').open(newline='') as stream:
            previous = list(csv.DictReader(stream))
        require([int(row['c']) for row in previous] == list(range(480)), 'Reference channel order mismatch')
        channels[half], pairs[half], diagnostics[half] = {}, {}, {}
        summary['channels'][half], summary['pairs'][half] = {}, {}
        for stage in ('h', 'z'):
            channel, pair, diag = decompose_templates(raw[stage][mask], pair_ids, cfg)
            original = np.asarray([float(row[f'{half}_s_{stage}']) for row in original_rows])
            require(np.array_equal(np.isfinite(original), np.isfinite(pair['s_raw'])), 'Cosine definition mismatch')
            valid = np.isfinite(original)
            error = float(np.max(np.abs(original[valid] - pair['s_raw'][valid]))) if valid.any() else 0.0
            require(error <= cfg['identity_atol'], 'Original cosine mismatch')
            summary['max_original_cosine_abs_difference'] = max(summary['max_original_cosine_abs_difference'], error)
            for key, current in (('frac_const', channel['frac_const']), ('frac_image', channel['frac_image']),
                                 ('frac_spatial', channel['frac_spatial_template'] + channel['frac_interaction'])):
                old = np.asarray([float(row[f'{stage}_{key}']) for row in previous])
                require(np.array_equal(np.isfinite(old), np.isfinite(current)), 'Reference fraction definition mismatch')
                valid = np.isfinite(old)
                error = float(np.max(np.abs(old[valid] - current[valid]))) if valid.any() else 0.0
                require(error <= cfg['identity_atol'], 'Previous fraction mismatch')
                summary['max_previous_fraction_abs_difference'] = max(summary['max_previous_fraction_abs_difference'], error)
            channels[half][stage], pairs[half][stage], diagnostics[half][stage] = channel, pair, diag
            summary['channels'][half][stage] = {
                group: {key: stats(values[index]) for key, values in channel.items() if key != 'c'}
                for group, index in (('all', np.arange(480)), ('selected', selected))}
            summary['channels'][half][stage]['majority_template_selected'] = int(np.count_nonzero(channel['frac_template'][selected] > .5))
            summary['pairs'][half][stage] = {key: stats(value) for key, value in pair.items() if key not in ('c', 'd')}
        ct = {'c': np.arange(480), 'selected': np.isin(np.arange(480), selected)}
        pt = {'c': pair_ids[:, 0], 'd': pair_ids[:, 1]}
        for stage in ('h', 'z'):
            ct.update({f'{stage}_{key}': value for key, value in channels[half][stage].items() if key != 'c'})
            pt.update({f'{stage}_{key}': value for key, value in pairs[half][stage].items() if key not in ('c', 'd')})
        write_table(run / f'channels_{half}.csv', ct)
        write_table(run / f'pairs_{half}.csv', pt)
    summary['diagnostics'] = diagnostics
    write_json(run / 'inputs_verified.json', {'source_files_sha256': input_hashes,
               'reference_files_sha256': reference_hashes, 'code_dependencies': code_hashes()['dependencies']})
    plots.mkdir(parents=True, exist_ok=False)
    make_plots(pairs, channels, selected, plots)
    for half in ('A', 'B'):
        for kind in ('channels', 'pairs'):
            shutil.copy2(run / f'{kind}_{half}.csv', plots / f'{kind}_{half}.csv')
    summary['status'] = 'completed'
    write_json(run / 'summary.json', summary)
    write_json(plots / 'summary.json', summary)
    print('TEMPLATE_ANALYSIS=completed', flush=True)
    for stage in ('h', 'z'):
        print(f'B_{stage}: template_fraction={summary["channels"]["B"][stage]["selected"]["frac_template"]} '
              f'dynamic_cosine={summary["pairs"]["B"][stage]["s_dynamic"]} '
              f'interaction_cosine={summary["pairs"]["B"][stage]["s_interaction"]}', flush=True)


if __name__ == '__main__':
    main()
