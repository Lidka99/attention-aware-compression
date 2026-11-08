"""2026-11-08_E039_przestawienia_tilt: tabele i wykresy kontroli; OPIS.md.
Wejście: ukończony run; wyjście: TABLES.md/CSV i lokalne PNG/PDF z hashami.
Analiza opisowa całej puli, bez ponownej oceny modeli i bez GPU.
"""
import json
import sys
from pathlib import Path

from experiment import HERE, PAIRS, PERM_SEEDS, rt


def main(run):
    _, frozen = rt.verify_run(run)
    analysis = rt.read(run / 'analysis/analysis.json')
    lines = ['# E039 — wszystkie wyniki i koszty', '', 'Średnie4 seedów; top1 w%, kontrasty i SD wpp.', '',
        '| Para | Redukcja | BN | U | T | SHUFFLED mean4 | T−mean4 ±SD | D dodatnie |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    cells, per_permutation = [], []
    for pair in PAIRS:
        for tag in rt.TAGS:
            for mode in rt.MODES:
                rows = [r for r in analysis['rows'] if (r['pair'], r['tag'], r['mode']) == (pair, tag, mode)]
                c = next(r for r in analysis['contrasts'] if (r['pair'], r['tag'], r['mode'], r['comparator']) ==
                         (pair, tag, mode, 'mean4'))
                means = {k: float(rt.np.mean([r[k] for r in rows])) for k in ('U', 'T', 'S')}
                vals = frozen['shuffled'][pair][tag]
                tmac = frozen['costs'][pair][tag]['T']
                costs = [v['mac'] / tmac for v in vals.values()]
                params = [v['params'] / frozen['plans'][pair][tag]['params']['U'] for v in vals.values()]
                individual = []
                for permutation in PERM_SEEDS:
                    items = [r for r in rows if r['permutation'] == permutation]
                    d = rt.stats(r['T'] - r['S'] for r in items)
                    v = vals[str(permutation)]
                    per_permutation.append(dict(pair=pair, tag=tag, mode=mode, permutation=permutation,
                        S=float(rt.np.mean([r['S'] for r in items])), T_minus_S=d['mean'], S_minus_U=float(rt.np.mean([r['S'] - r['U'] for r in items])),
                        sd=d['sd'], positive=d['positive'], params=v['params'], mac=v['mac'], mac_S_over_T=v['mac'] / tmac))
                    individual.append(d['mean'])
                cells.append(dict(pair=pair, tag=tag, mode=mode, **means, D=c['mean'], D_sd=c['sd'], positive=c['positive'],
                    each_permutation_positive=sum(d > 0 for d in individual), mac_S_over_T_min=min(costs), mac_S_over_T_max=max(costs),
                    params_S_over_U_min=min(params), params_S_over_U_max=max(params)))
                lines.append(f"| {pair} | {tag[1:]}% | {mode} | {means['U']:.4f} | {means['T']:.4f} | {means['S']:.4f} | "
                             f"{c['mean']:+.4f} ±{c['sd']:.4f} | {c['positive']}/4 |")
    lines += ['', '## Każda permutacja osobno, łącznie z kosztami', '',
        '| Para | Redukcja | BN | Perm | SHUFFLED | T−S | S−U | Dodatnie T−S | P(S) | MAC(S) | MAC(S)/MAC(T) |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in per_permutation:
        lines.append(f"| {r['pair']} | {r['tag'][1:]}% | {r['mode']} | {r['permutation']} | {r['S']:.4f} | "
                     f"{r['T_minus_S']:+.4f} | {r['S_minus_U']:+.4f} | {r['positive']}/4 | {r['params']} | {r['mac']} | {r['mac_S_over_T']:.4f} |")
    (HERE / 'TABLES.md').write_text('\n'.join(lines) + '\n')
    rt.csv_write(HERE / 'cells.csv', cells)
    rt.csv_write(HERE / 'permutations.csv', per_permutation)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, pair in zip(axes, PAIRS):
        for i, tag in enumerate(rt.TAGS):
            c = next(r for r in analysis['primary'] if r['pair'] == pair and r['tag'] == tag)
            ax.scatter([i] * 4, c['per_seed'], alpha=.7)
            ax.errorbar(i, c['mean'], yerr=c['sd'], fmt='ks', capsize=4)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xticks(range(3), ['30%', '50%', '70%'])
        ax.set_xlabel('Redukcja parametrów')
        ax.set_title(pair.replace('resnet_', 'ResNet18 / '))
        ax.set_ylabel('TILT − średnia4 przestawień [pp]')
    fig.suptitle('E039: CMA, dev; punkty seedów i średnia±SD')
    folder = HERE / 'figures'
    folder.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'e039_kolejnosc.{ext}', dpi=160)
    plt.close(fig)
    rt.write(folder / 'provenance.json', dict(run=rt.rel(run), analysis_sha256=rt.digest(run / 'analysis/analysis.json'),
        script_sha256=rt.digest(Path(__file__)), files={p.name: rt.digest(p) for p in folder.glob('e039_kolejnosc.*')}))
    for row in cells:
        if row['mode'] == 'CMA':
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
