"""E049, figura dopasowanych parametrów: TILT vs UNIFORM przy b30/b50/b70.

Co robi: rysuje dwa panele (CIFAR-100, Tiny ImageNet) dla nominalnej redukcji
parametrów 30/50/70%. W każdym wierszu 4 sparowane różnice top-1 TILT-UNIFORM
(po seedach 42-45) i ich średnia; wspólna oś x, linia zera. W prawej kolumnie
podano iloraz MAC TILT/UNIFORM policzony z danych (bez sugerowania równego
MAC). Wejście: wyłącznie `runs/E049/eval_20261124T234559Z_b4ca081b/per_seed.csv`
i `audit.json` (`split=benchmark`); skrypt sprawdza dopasowanie P do 0,1%
oraz zgodność różnic i dodatnich par z audytem. Wyjście:
`e049_matched_parameters_en.pdf` i `.png` obok skryptu.
Protokół: ../../OPIS.md; odtworzenie i opis: PARAMETERS.md.
Bez GPU, bez treningu i bez nowych punktacji.
"""

import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'runs' / 'E049' / 'eval_20261124T234559Z_b4ca081b'
SEEDS = (42, 43, 44, 45)
N = 10000
BUDGETS = (30, 50, 70)
DATASETS = ('cifar100', 'tiny')
NAMES = {'cifar100': 'CIFAR-100', 'tiny': 'Tiny ImageNet'}
COLOR = '#6a4c93'
INK = '#202a35'
MUTED = '#4b5966'
GREY = '#6d7781'
JITTER = 0.16  # dwa tory (+/-); sąsiednie x trafiają na przeciwne tory


def load():
    raw = (RUN / 'per_seed.csv').read_bytes()
    audit = json.loads((RUN / 'audit.json').read_text())
    rows = list(csv.DictReader(raw.decode().splitlines()))
    bench = [r for r in rows if r['split'] == 'benchmark']
    variants = {f'b{b}_{p}' for b in BUDGETS for p in ('U', 'T')}
    assert {r['dataset'] for r in bench} == set(DATASETS)
    assert variants <= {r['variant'] for r in bench}
    assert all(int(r['n']) == N for r in bench if r['variant'] in variants)
    cell = {(r['dataset'], r['variant'], int(r['seed'])): r for r in bench}
    assert len(cell) == len(bench)
    assert set(cell) == {(d, v, s) for d in DATASETS for v in
                         {r['variant'] for r in bench} for s in SEEDS}
    return raw, audit, cell


def measure(cell):
    out = {}
    for d in DATASETS:
        out[d] = {}
        for b in BUDGETS:
            sample = {}
            for part in ('U', 'T'):
                v = f'b{b}_{part}'
                rows = [cell[d, v, s] for s in SEEDS]
                (params, mac), = {(int(r['parameters']), int(r['mac'])) for r in rows}
                top1 = [100 * int(r['correct']) / N for r in rows]
                assert top1 == [float(r['top1_pct']) for r in rows]
                sample[part] = dict(params=params, mac=mac, top1=top1)
            diff = [t - u for t, u in zip(sample['T']['top1'], sample['U']['top1'])]
            out[d][b] = dict(diff=diff, **sample)
    return out


def verify(m, audit):
    assert audit['status'] == 'passed' and audit['joint_gate_met'] is False
    assert audit['benchmark_previously_used'] is True
    for d in DATASETS:
        a = audit['datasets'][d]['benchmark']
        assert not audit['datasets'][d]['passed']
        for b in BUDGETS:
            row = m[d][b]
            for part in ('U', 'T'):
                v = f'b{b}_{part}'
                assert math.isclose(st.mean(row[part]['top1']),
                                    a['top1_mean_pct'][v], abs_tol=1e-9), (d, v)
            effect = a['contrasts'][f'b{b}_T-b{b}_U']
            assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in
                       zip(row['diff'], effect['per_seed_pp'])), (d, b)
            assert math.isclose(st.mean(row['diff']), effect['mean_pp'], abs_tol=1e-9)
            assert math.isclose(st.stdev(row['diff']), effect['sd_pp'], abs_tol=1e-9)
            assert effect['positive_pairs'] == sum(x > 0 for x in row['diff']) == len(SEEDS)
            # dopasowanie liczby P do 0,1% i jawnie wyższy MAC TILT
            pu, pt = row['U']['params'], row['T']['params']
            row['p_ratio'] = pt / pu
            row['p_dev_pct'] = 100 * abs(pt - pu) / pu
            assert pt <= pu and row['p_dev_pct'] <= 0.1, (d, b)
            row['mac_ratio'] = row['T']['mac'] / row['U']['mac']
            assert row['T']['mac'] > row['U']['mac']


def style(ax, axis):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#e1e7ed', linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#a1acb8')
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=3, width=.6, color='#a1acb8', labelsize=7.8, pad=2)


def seed_dots(ax, values, y):
    rank = {i: r for r, i in enumerate(sorted(range(len(values)), key=values.__getitem__))}
    for i, value in enumerate(values):
        ax.plot(value, y + (JITTER if rank[i] % 2 == 0 else -JITTER),
                linestyle='none', marker='o', markersize=4.1,
                markerfacecolor='white', markeredgecolor=COLOR,
                markeredgewidth=1.0, zorder=3)


def mean_mark(ax, mean, y, text):
    ax.plot(mean, y, linestyle='none', marker='D', markersize=5.6,
            color=COLOR, markeredgecolor='white', markeredgewidth=0.7, zorder=4)
    ax.annotate(text, (mean, y), xytext=(0, 11), textcoords='offset points',
                va='bottom', ha='center', fontsize=7.6, color=COLOR,
                fontweight='bold', zorder=5)


def draw(m):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': '#344352',
                         'ytick.color': '#344352'})
    fig = plt.figure(figsize=(6.6929, 3.35), facecolor='white')  # 170 mm szerokości druku
    outer = fig.add_gridspec(1, 2, left=.098, right=.980, bottom=.175, top=.700,
                             wspace=.60)
    ylim = (-.62, 2.62)
    xlim = (-.7, 13.6)

    for i, d in enumerate(DATASETS):
        cell = outer[0, i].subgridspec(1, 2, width_ratios=(1, .30), wspace=.06)
        ax = fig.add_subplot(cell[0, 0])
        ratio_ax = fig.add_subplot(cell[0, 1])

        x0 = ax.get_position().x0
        fig.text(x0, .945, f'{"AB"[i]}  {NAMES[d]}', ha='left',
                 fontsize=9.6, fontweight='bold')
        fig.text(x0, .872, 'After CMA; P matched within 0.1%', ha='left',
                 fontsize=7.2, color=MUTED)

        ax.axvline(0, color=GREY, linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
        for (b, y) in zip(BUDGETS, (2, 1, 0)):
            row = m[d][b]
            seed_dots(ax, row['diff'], y)
            mean_mark(ax, st.mean(row['diff']), y, f'{st.mean(row["diff"]):+.3f}')
        ax.set(xlim=xlim, ylim=ylim, yticks=(2, 1, 0),
               yticklabels=[f'{b}%' for b in BUDGETS],
               xlabel='TILT − UNIFORM top-1 (pp)')
        if i == 0:
            ax.set_ylabel('Nominal P reduction')
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.xaxis.set_major_formatter(lambda v, _: f'{v:g}')
        style(ax, 'x')

        ratio_ax.set(xlim=(0, 1), ylim=ylim)
        ratio_ax.set_axis_off()
        ratio_ax.text(.5, 1.035, 'MAC ratio\nTILT / UNIFORM', transform=ratio_ax.transAxes,
                      ha='center', va='bottom', fontsize=6.5, color=MUTED, linespacing=1.15)
        for b, y in zip(BUDGETS, (2, 1, 0)):
            ratio_ax.text(.5, y, f'×{m[d][b]["mac_ratio"]:.3f}',
                          transform=ratio_ax.transData, ha='center', va='center',
                          fontsize=7.5, color=INK)

    handles = [Line2D([], [], linestyle='none', marker='o', markersize=4.1,
                      markerfacecolor='white', markeredgecolor='#66727e',
                      markeredgewidth=1.0, label='Seed 42–45'),
               Line2D([], [], linestyle='none', marker='D', markersize=5.2,
                      color='#66727e', label='Mean')]
    fig.axes[0].legend(handles=handles, loc='lower left', frameon=False, fontsize=7,
                       ncol=2, handletextpad=.35, columnspacing=1.0,
                       borderaxespad=.15)
    return fig


def main():
    if not __debug__:
        raise RuntimeError('Uruchom bez -O: kontrole wejść używają assert.')
    raw, audit, cell = load()
    metrics = measure(cell)
    verify(metrics, audit)
    fig = draw(metrics)
    png = HERE / 'e049_matched_parameters_en.png'
    pdf = HERE / 'e049_matched_parameters_en.pdf'
    fig.savefig(png, dpi=400,
                metadata={'Software': f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(pdf, metadata={'Title': 'E049 matched-P TILT vs UNIFORM (b30/b50/b70)',
                               'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(f'PASS: P dopasowane do 0,1%, różnice i pary dodatnie zgodne z audytem; '
          f'Matplotlib {matplotlib.__version__}.')
    for d in DATASETS:
        for b in BUDGETS:
            row = metrics[d][b]
            print(f'{d} b{b}: TILT-U {st.mean(row["diff"]):+.4f} pp '
                  f'(+ {sum(x > 0 for x in row["diff"])}/4), '
                  f'P_T/P_U {row["p_ratio"]:.5f} ({row["p_dev_pct"]:.3f}%), '
                  f'MAC_T/MAC_U {row["mac_ratio"]:.4f}')
    print('input per_seed.csv sha256:', hashlib.sha256(raw).hexdigest())
    print('input audit.json sha256:  ',
          hashlib.sha256((RUN / 'audit.json').read_bytes()).hexdigest())
    for path in (png, pdf):
        print(path.relative_to(ROOT), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
