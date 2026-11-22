"""2026-11-15_E041_kontrola_p_mac: tabele i wykres z zapisanych metryk.
Wejście: per_seed.csv/analysis.json zakończonego runu; wyjście: TABLES.md,
cells.csv, audyt30kontrastów i figury. Bez inferencji; pełny opis: ../OPIS.md.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
PAIRS = ('resnet_cifar10', 'resnet_cifar100')
MODES = ('CMA', 'UP_REPAIR', 'PRE')
IDS = ('mean4', '101', '102', '103', '104')
SEEDS = (42, 43, 44, 45)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits=4, signed=False):
    return format(float(value), ('+' if signed else '') + f'.{digits}f').replace('.', ',')


def main(run):
    with (run / 'analysis/per_seed.csv').open() as f:
        raw = list(csv.DictReader(f))
    assert len(raw) == 96
    analysis = json.loads((run / 'analysis/analysis.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    original = {(r['pair'], r['mode'], r['comparator']): r for r in analysis['contrasts']}
    cells = []
    for mode in MODES:
        for pair in PAIRS:
            for comparator in IDS:
                t, c, nll, values = [], [], [], []
                for seed in SEEDS:
                    selected = [r for r in raw if r['pair'] == pair and r['mode'] == mode
                                and int(r['seed']) == seed
                                and (comparator == 'mean4' or r['control'] == comparator)]
                    assert len(selected) == (4 if comparator == 'mean4' else 1)
                    teacher = float(selected[0]['T'])
                    control = float(np.mean([float(r['C']) for r in selected]))
                    t.append(teacher); c.append(control); values.append(teacher - control)
                    nll.append(float(selected[0]['T_nll']) - np.mean([float(r['C_nll']) for r in selected]))
                mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
                ref = original[(pair, mode, comparator)]
                assert np.allclose(values, ref['per_seed'], rtol=0, atol=1e-10)
                assert abs(mean - ref['mean']) < 1e-10 and abs(sd - ref['sd']) < 1e-10
                assert sum(v > 0 for v in values) == ref['positive']
                assert abs(float(np.mean(nll)) - ref['nll']['mean']) < 1e-10
                cells.append(dict(pair=pair, mode=mode, comparator=comparator,
                    T=float(np.mean(t)), C=float(np.mean(c)), mean=mean, sd=sd,
                    positive=ref['positive'], direction=ref['direction'],
                    magnitude_1pp=ref['magnitude_1pp'], nll=float(np.mean(nll)),
                    **{f'seed_{s}': v for s, v in zip(SEEDS, values)}))
    assert len(cells) == 30
    sources = {str((run / p).relative_to(ROOT)): digest(run / p)
               for p in ('analysis/per_seed.csv', 'analysis/analysis.json', 'manifest.json', 'frozen.json')}
    text = ['# E041 — tabele z zapisanych ocen', '',
            f'Run: `{run.relative_to(ROOT)}`. Top1 w%, różnice i SD wpp.',
            'Średnia4kontroli najpierw w obrębie seeda; SD po4seedach.',
            '**Wszystkie wyniki dotyczą dev. Tiny ImageNet nie był objęty E041.**', '']
    for mode in MODES:
        text += [f'## {mode}', '',
                 '| Zbiór | Kontrola | TILT | Kontrola/średnia4 | T−C ±SD [pp] | Dodatnie |',
                 '|---|---|---:|---:|---:|---:|']
        for r in [r for r in cells if r['mode'] == mode]:
            ds = r['pair'].replace('resnet_', '')
            text.append(f'| {ds} | {r["comparator"]} | {number(r["T"])} | {number(r["C"], 5)} | '
                        f'{number(r["mean"], 5, True)} ±{number(r["sd"])} | {r["positive"]}/4 |')
        text.append('')
    text += ['## Primary per seed, CMA', '', '| Zbiór | Seed42 | Seed43 | Seed44 | Seed45 |',
             '|---|---:|---:|---:|---:|']
    for r in [r for r in cells if r['mode'] == 'CMA' and r['comparator'] == 'mean4']:
        text.append('| ' + r['pair'] + ' | ' + ' | '.join(number(r[f'seed_{s}'], 5, True) for s in SEEDS) + ' |')
    text += ['', 'Źródła i przeliczenie30kontrastów: `analysis/table_audit.json` w runie.',
             'Kod: `analizy/tabele.py`. Brak ponownej oceny modeli.', '']
    (HERE / 'TABLES.md').write_text('\n'.join(text))
    with (HERE / 'cells.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader(); writer.writerows(cells)
    out = HERE / 'figures' / run.name
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = plt.get_cmap('tab10').colors
    for ax, pair, label in zip(axes, PAIRS, ('CIFAR-10', 'CIFAR-100')):
        selected = [r for r in cells if r['pair'] == pair and r['mode'] == 'CMA']
        for i, row in enumerate(selected):
            for j, seed in enumerate(SEEDS):
                ax.scatter(i + (j - 1.5) * .06, row[f'seed_{seed}'], color=colors[j], s=23,
                           label=f'seed {seed}' if i == 0 else None, zorder=3)
            ax.errorbar(i, row['mean'], yerr=row['sd'], fmt='kD', markersize=4, capsize=3)
        ax.axhline(0, color='0.45', linestyle='--', linewidth=1)
        ax.set(title=label, xticks=range(5), xticklabels=('Średnia4', '101', '102', '103', '104'),
               xlabel='Dopasowane kontrole', ylabel='TILT − kontrola [pp], CMA')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Alokacja przy dopasowanych parametrach i MAC')
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'e041_kontrasty.{suffix}', dpi=180)
    plt.close(fig)
    record = dict(passed=True, contrasts_recomputed=30, source_hashes=sources,
                  code_sha256=digest(Path(__file__)),
                  outputs={str(p.relative_to(ROOT)): digest(p) for p in
                           (HERE / 'TABLES.md', HERE / 'cells.csv', out / 'e041_kontrasty.png', out / 'e041_kontrasty.pdf')})
    (run / 'analysis/table_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'README.md').write_text('# E041 — wykres kontroli\n\n'
        f'Run `{run.relative_to(ROOT)}`; tylko zapisane metryki dev. Punkty:4seedy, '
        'diament:średnia, belki:SD(ddof=1), nie przedziały istotności. '
        'Dodatnia różnica oznacza przewagę TILT.\n\n'
        'Kod `../../analizy/tabele.py`; źródła i SHA256: '
        f'`{run.relative_to(ROOT)}/analysis/table_audit.json`.\n')
    for row in cells:
        if row['mode'] in ('CMA', 'UP_REPAIR') and row['comparator'] == 'mean4':
            print(row)
    print('TABLE AUDIT PASS30', out.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run.resolve())
