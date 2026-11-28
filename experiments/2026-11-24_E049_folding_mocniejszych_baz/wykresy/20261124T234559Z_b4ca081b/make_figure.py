"""E049, figura manuskryptu: łączne bramki b10 na CIFAR-100 i Tiny ImageNet.

Co robi: rysuje dwa panele A/B wyłącznie z zapisanego przebiegu
`runs/E049/eval_20261124T234559Z_b4ca081b/` (`per_seed.csv`, `audit.json`,
split=benchmark). Panel A: sparowane różnice top-1 L06-UNIFORM i L06-T-proj
po seedach 42-45 ze średnimi oraz progiem +1 pp zaznaczonym wyłącznie dla
wierszy L06-UNIFORM. Panel B: strata L06 do pełnego modelu po CMA po seedach
ze średnimi oraz progiem <=3 pp.

Wejście: `per_seed.csv` (top-1 per seed, split=benchmark) i `audit.json`
(średnie, kontrasty, straty) z katalogu runu; skrypt przelicza wartości
z trafień i porównuje je z audytem. Wyjście: `e049_b10_joint_caps_en.pdf`
i `.png` obok skryptu. Szczegóły protokołu: ../../OPIS.md; odtworzenie: README.md.
Bez GPU, bez nowych punktacji i bez treningu.
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
DATASETS = ('cifar100', 'tiny')
NAMES = {'cifar100': 'CIFAR-100', 'tiny': 'Tiny ImageNet'}
COLORS = {'U': '#215d8a', 'Tproj': '#ae4619', 'loss': '#2f6f5e'}
INK = '#202a35'
MUTED = '#4b5966'
GREY = '#6d7781'
JITTER_A = (-0.09, -0.03, 0.03, 0.09)
JITTER_B = 0.34  # dwa tory (+/-); sąsiednie x trafiają na przeciwne tory


def load():
    raw = (RUN / 'per_seed.csv').read_bytes()
    audit = json.loads((RUN / 'audit.json').read_text())
    rows = list(csv.DictReader(raw.decode().splitlines()))
    bench = [r for r in rows if r['split'] == 'benchmark']
    assert len(bench) == len(rows) - len([r for r in rows if r['split'] == 'dev'])
    needed = {'full', 'b10_U', 'b10_L06', 'b10_Tproj'}
    assert {r['variant'] for r in bench} == needed | {
        'full_pre', 'b30_U', 'b30_T', 'b50_U', 'b50_T', 'b70_U', 'b70_T'}
    assert {r['dataset'] for r in bench} == set(DATASETS)
    assert all(int(r['n']) == N for r in bench)
    cell = {(r['dataset'], r['variant'], int(r['seed'])): r for r in bench}
    assert len(cell) == len(bench)
    assert set(cell) == {(d, v, s) for d in DATASETS for v in
                         {r['variant'] for r in bench} for s in SEEDS}
    return raw, audit, cell


def measure(cell):
    out = {}
    for d in DATASETS:
        top1, seed_top1, cost = {}, {}, {}
        for v in ('full_pre', 'full', 'b10_U', 'b10_L06', 'b10_Tproj'):
            sample = [cell[d, v, s] for s in SEEDS]
            (params, mac), = {(int(r['parameters']), int(r['mac'])) for r in sample}
            hits = [int(r['correct']) for r in sample]
            assert all(0 <= k <= N for k in hits)
            seed_top1[v] = [100 * k / N for k in hits]
            assert seed_top1[v] == [float(r['top1_pct']) for r in sample]
            top1[v] = st.mean(seed_top1[v])
            cost[v] = (params, mac)
        contrasts = {
            name: [l - o for l, o in zip(seed_top1['b10_L06'], seed_top1[ref])]
            for name, ref in (('U', 'b10_U'), ('Tproj', 'b10_Tproj'))
        }
        loss = [f - l for f, l in zip(seed_top1['full'], seed_top1['b10_L06'])]
        out[d] = dict(top1=top1, seed_top1=seed_top1, contrasts=contrasts,
                      loss=loss, cost=cost)
    return out


def verify(m, audit):
    assert audit['status'] == 'passed' and audit['baseline_audit']['status'] == 'passed'
    assert audit['baseline_audit']['results']['cifar100']['quality_gate_met']
    assert audit['baseline_audit']['results']['tiny']['quality_gate_met']
    assert audit['benchmark_previously_used'] is True
    assert audit['joint_gate_met'] is False
    for d in DATASETS:
        a = audit['datasets'][d]['benchmark']
        assert audit['datasets'][d]['E048_quality_gate_met'] is True
        assert not audit['datasets'][d]['passed']
        for v, value in m[d]['top1'].items():
            assert math.isclose(value, a['top1_mean_pct'][v], abs_tol=1e-9), (d, v)
        for name in ('U', 'Tproj'):
            effect = a['contrasts']['b10_L06-b10_' + name]
            pair = m[d]['contrasts'][name]
            assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in
                       zip(pair, effect['per_seed_pp'])), (d, name)
            assert math.isclose(st.mean(pair), effect['mean_pp'], abs_tol=1e-9)
            assert effect['positive_pairs'] == sum(x > 0 for x in pair) == len(SEEDS)
            assert math.isclose(st.stdev(pair), effect['sd_pp'], abs_tol=1e-9)
        assert math.isclose(st.mean(m[d]['loss']), a['loss_to_full_pp']['b10_L06'],
                            abs_tol=1e-9)
        assert math.isclose(m[d]['top1']['full'] - m[d]['top1']['b10_L06'],
                            a['loss_to_full_pp']['b10_L06'], abs_tol=1e-9)
        assert a['loss_to_full_pp']['full'] == 0.0
        # b10 U, L06 i T-proj mieszczą się w 99,9-100% kosztu U w P i MAC
        for v in ('b10_L06', 'b10_Tproj'):
            for k, i in (('params', 0), ('mac', 1)):
                assert 999 * m[d]['cost']['b10_U'][i] <= 1000 * m[d]['cost'][v][i] \
                    <= 1000 * m[d]['cost']['b10_U'][i], (d, v, k)
        # pary L06 > U i L06 > T-proj oraz strata L06 < straty U
        assert all(x > 0 for x in m[d]['contrasts']['U'])
        assert all(x > 0 for x in m[d]['contrasts']['Tproj'])
        assert m[d]['loss'] == [f - l for f, l in
                                zip(m[d]['seed_top1']['full'], m[d]['seed_top1']['b10_L06'])]
        assert st.mean(m[d]['loss']) < a['loss_to_full_pp']['b10_U']


def style(ax, axis):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#e1e7ed', linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#a1acb8')
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=3, width=.6, color='#a1acb8', labelsize=7.8, pad=2)


def seed_dots(ax, values, y, color, jitter):
    if isinstance(jitter, float):
        rank = {i: r for r, i in enumerate(sorted(range(len(values)), key=values.__getitem__))}
        jitter = [jitter if rank[i] % 2 == 0 else -jitter for i in range(len(values))]
    for value, dy in zip(values, jitter):
        ax.plot(value, y + dy, linestyle='none', marker='o', markersize=4.1,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.0, zorder=3)


def mean_mark(ax, value, y, color, text, dx=0, dy=0, ha='center', va='bottom', size=7.8):
    ax.plot(value, y, linestyle='none', marker='D', markersize=5.6,
            color=color, markeredgecolor='white', markeredgewidth=0.7, zorder=4)
    ax.annotate(text, (value, y), xytext=(dx, dy), textcoords='offset points',
                va=va, ha=ha, fontsize=size, color=color,
                fontweight='bold', zorder=5)


def gate_cap(ax, x, y, half=0.22, color=GREY):
    ax.plot([x, x], [y - half, y + half], color=color, linewidth=1.1,
            linestyle=(0, (2.2, 1.5)), zorder=2)
    for yy in (y - half, y + half):
        ax.plot([x - .045, x + .045], [yy, yy], color=color, linewidth=1.1, zorder=2)


def draw(m):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': '#344352',
                         'ytick.color': '#344352'})
    fig = plt.figure(figsize=(6.6929, 3.50), facecolor='white')  # 170 mm szerokości druku
    grid = fig.add_gridspec(1, 2, left=.176, right=.986, bottom=.165, top=.715,
                            wspace=.46)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    def header(panel, title, subtitle):
        x = panel.get_position().x0
        fig.text(x, .945, title, ha='left', fontsize=9.6, fontweight='bold')
        fig.text(x, .868, subtitle, ha='left', fontsize=7.3, color=MUTED)

    header(ax_a, 'A  Paired top-1 gains at b10',
           'Frozen offsets vs. baseline; 4/4 positive pairs')
    header(ax_b, 'B  Loss to the full model',
           'Frozen offsets after CMA; lower is better')

    # --- Panel A: 4 wiersze, UNIFORM u góry, T-proj u dołu ---
    rows = (('U', 'cifar100', 3), ('U', 'tiny', 2),
            ('Tproj', 'cifar100', 1), ('Tproj', 'tiny', 0))
    ax_a.axvline(0, color=GREY, linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
    for comp, d, y in rows:
        color = COLORS[comp]
        seed_dots(ax_a, m[d]['contrasts'][comp], y, color, JITTER_A)
        mean = st.mean(m[d]['contrasts'][comp])
        mean_mark(ax_a, mean, y, color, f'{mean:+.3f}', dy=9, size=7.6)
    for y in (3, 2):  # próg +1 pp dotyczy wyłącznie L06-UNIFORM
        gate_cap(ax_a, 1.0, y)
    ax_a.annotate('+1 pp gate', (1.42, 2.50), va='center', ha='left',
                  fontsize=7.1, color=GREY, zorder=5)
    # etykiety grup poza osią, obrócone, wspólne dla dwóch wierszy każda
    for text, y_center, color in (('UNIFORM', .504, COLORS['U']),
                                  ('T-proj', .307, COLORS['Tproj'])):
        fig.text(.028, y_center, text, rotation=90, ha='center', va='center',
                 fontsize=7.3, fontweight='bold', color=color)
    ax_a.axhline(1.5, color='#dde3ea', linewidth=.7, zorder=0)
    ax_a.set(xlim=(-.45, 3.75), ylim=(-.90, 3.90), yticks=(3, 2, 1, 0),
             yticklabels=[NAMES['cifar100'], NAMES['tiny'],
                          NAMES['cifar100'], NAMES['tiny']],
             xlabel='Paired top-1 difference (pp)')
    ax_a.xaxis.set_major_locator(MultipleLocator(1))
    ax_a.xaxis.set_major_formatter(lambda v, _: f'{v:g}')
    style(ax_a, 'x')

    # --- Panel B: strata L06 do pełnego po CMA ---
    ax_b.axvline(0, color=GREY, linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
    ax_b.axvline(3.0, color=GREY, linewidth=1.1, linestyle=(0, (2.2, 1.5)), zorder=1)
    ax_b.annotate('3 pp gate', (3.07, 1.62), va='center', ha='left',
                  fontsize=7.1, color=GREY,
                  bbox=dict(facecolor='white', edgecolor='none', pad=0.8), zorder=5)
    for d, y in (('cifar100', 1), ('tiny', 0)):
        color = COLORS['loss']
        seed_dots(ax_b, m[d]['loss'], y, color, JITTER_B)
        mean = st.mean(m[d]['loss'])
        mean_mark(ax_b, mean, y, color, f'{mean:.2f}', dx=-7, ha='right', va='center')
    ax_b.set(xlim=(-.30, 6.55), ylim=(-.86, 1.95), yticks=(1, 0),
             yticklabels=[NAMES['cifar100'], NAMES['tiny']],
             xlabel='Paired top-1 loss (pp)')
    ax_b.xaxis.set_major_locator(MultipleLocator(1))
    ax_b.xaxis.set_major_formatter(lambda v, _: f'{v:g}')
    style(ax_b, 'x')

    handles = [Line2D([], [], linestyle='none', marker='o', markersize=4.1,
                      markerfacecolor='white', markeredgecolor='#66727e',
                      markeredgewidth=1.0, label='Seed 42–45'),
               Line2D([], [], linestyle='none', marker='D', markersize=5.2,
                      color='#66727e', label='Mean')]
    for panel in (ax_a, ax_b):
        panel.legend(handles=handles, loc='lower right', frameon=False, fontsize=7,
                     ncol=2, handletextpad=.35, columnspacing=1.0,
                     borderaxespad=.2)
    return fig


def main():
    if not __debug__:
        raise RuntimeError('Uruchom bez -O: kontrole wejść używają assert.')
    raw, audit, cell = load()
    metrics = measure(cell)
    verify(metrics, audit)
    fig = draw(metrics)
    visible = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert not any('L06' in t for t in visible), 'Widoczny identyfikator L06 na figurze'
    png = HERE / 'e049_b10_joint_caps_en.png'
    pdf = HERE / 'e049_b10_joint_caps_en.pdf'
    fig.savefig(png, dpi=400,
                metadata={'Software': f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(pdf, metadata={'Title': 'E049 b10 joint gates: CIFAR-100 and Tiny ImageNet',
                               'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(f'PASS: per_seed.csv i audit.json zgodne (split=benchmark); '
          f'Matplotlib {matplotlib.__version__}.')
    for d in DATASETS:
        print(f'{d}: L06-U {st.mean(metrics[d]["contrasts"]["U"]):+.4f} pp, '
              f'L06-Tproj {st.mean(metrics[d]["contrasts"]["Tproj"]):+.4f} pp, '
              f'loss L06 {st.mean(metrics[d]["loss"]):.4f} pp')
    print('input per_seed.csv sha256:', hashlib.sha256(raw).hexdigest())
    print('input audit.json sha256:  ',
          hashlib.sha256((RUN / 'audit.json').read_bytes()).hexdigest())
    for path in (png, pdf):
        print(path.relative_to(ROOT), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
