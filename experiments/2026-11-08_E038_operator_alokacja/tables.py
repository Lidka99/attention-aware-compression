"""2026-11-08_E038_operator_alokacja: tabele/wykres z zamrożonych ocen; OPIS.md.
Wejście: analysis.json; wyjście: lokalne TABLES.md, CSV, PNG/PDF i ich hashe.
Analiza opisowa po wykonaniu, bez GPU i ponownej oceny modeli.
"""
import json
import sys
from pathlib import Path

import runtime as rt


def main(run):
    result = rt.read(run / 'analysis/analysis.json')
    rt.verify_run(run)
    lines = ['# E038 — wszystkie kontrasty', '', 'Top1 i kontrasty w pp; średnie czterech seedów. SD dotyczy I.', '',
        '| Para | Redukcja | BN | F/U | F/T | M/U | M/T | ΔF | ΔM | I ±SD | I dodatnie |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    cells = []
    for pair in rt.PAIRS:
        for tag in rt.TAGS:
            for mode in rt.MODES:
                rows = [r for r in result['rows'] if (r['pair'], r['tag'], r['mode']) == (pair, tag, mode)]
                means = {k: float(rt.np.mean([r[k] for r in rows])) for k in ('FU', 'FT', 'MU', 'MT', 'delta_F', 'delta_M', 'I')}
                st = rt.stats(r['I'] for r in rows)
                cells.append(dict(pair=pair, tag=tag, mode=mode, **means, I_sd=st['sd'], I_positive=st['positive']))
                vals = [f'{means[k]:.4f}' for k in ('FU', 'FT', 'MU', 'MT', 'delta_F', 'delta_M')]
                lines.append(f"| {pair} | {tag[1:]}% | {mode} | " + ' | '.join(vals) +
                    f" | {means['I']:+.4f} ±{st['sd']:.4f} | {st['positive']}/4 |")
    here = Path(__file__).resolve().parent
    (here / 'TABLES.md').write_text('\n'.join(lines) + '\n')
    rt.csv_write(here / 'cells.csv', cells)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [p.replace('resnet_', 'R/').replace('preact_', 'P/') for p in rt.PAIRS]
    x = rt.np.arange(len(labels))
    for i, (key, label) in enumerate((('delta_F', 'Folding'), ('delta_M', 'MAG2'))):
        series = [[r[key] for r in result['rows'] if r['pair'] == p and r['tag'] == 'b50' and r['mode'] == 'CMA'] for p in rt.PAIRS]
        axes[0].bar(x + (i - .5) * .35, [rt.np.mean(v) for v in series], .35,
                    yerr=[rt.np.std(v, ddof=1) for v in series], capsize=3, label=label)
    for i, p in enumerate(rt.PAIRS):
        values = [r['I'] for r in result['rows'] if r['pair'] == p and r['tag'] == 'b50' and r['mode'] == 'CMA']
        axes[1].scatter([i] * 4, values, alpha=.7)
        axes[1].errorbar(i, rt.np.mean(values), yerr=rt.np.std(values, ddof=1), fmt='ks', capsize=4)
    for ax in axes:
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xticks(x, labels, rotation=20)
        ax.set_ylabel('Różnica top1 [pp]')
    axes[0].set_title('Zysk TILT−UNIFORM: 50%, CMA, dev')
    axes[0].legend()
    axes[1].set_title('Interakcja I=ΔF−ΔM; punkty seedów, średnia±SD')
    folder = here / 'figures'
    folder.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'e038_interakcja.{ext}', dpi=160)
    plt.close(fig)
    rt.write(folder / 'provenance.json', dict(run=rt.rel(run), analysis_sha256=rt.digest(run / 'analysis/analysis.json'),
        script_sha256=rt.digest(Path(__file__)), files={p.name: rt.digest(p) for p in folder.glob('e038_interakcja.*')}))
    for row in cells:
        if row['mode'] == 'CMA':
            print(json.dumps(row))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
