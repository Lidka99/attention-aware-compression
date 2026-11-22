"""E045: dwupanelowa krzywa jakość–koszt łagodnego UNIFORM na Tiny ImageNet dev.

Co robi: rysuje bezwzględny top-1 po CMA względem rzeczywistej redukcji P
(lewy panel) oraz sparowaną stratę pełny−U względem rzeczywistej redukcji MAC
(prawy panel). Średnia i punkty czterech seedów (42–45). Punkty 0/10/20% to
nowe pomiary E045, punkty b30/b50 to reuse U z dev E035.
Wejście: tylko runs/E045/20261122T031350Z_a58b56a3/analysis/{summary.json,
per_seed.csv,audit.json} (hashe SHA256 sprawdzane poniżej).
Wyjście: e045_krzywa.png i e045_krzywa.pdf w tym folderze.
Bez treningu, obrazów, wag, inferencji i GPU; protokół: ../../OPIS.md.
"""

import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault('SOURCE_DATE_EPOCH', '1795305600')
os.environ['TZ'] = 'UTC'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'runs/E045/20261122T031350Z_a58b56a3'
ANALYSIS = RUN / 'analysis'
INPUTS = {
    'summary.json': 'ca6a0d74db6974d52f3795e82f3c7803503c225adfe78fa203aa03024b4e8811',
    'per_seed.csv': '486309557455840de25852f166e55e6c909cbb1aa23e477418f8ed3057a2a992',
    'audit.json': 'fcca52a0fbd3020931b34d275f6fb84f278ac1d47f6575a850c5c0d3df20ad47',
}
ORDER = ['teacher', 'b10', 'b20', 'b30', 'b50']
LABELS = {
    'teacher': '0%',
    'b10': 'b10',
    'b20': 'b20',
    'b30': 'b30*',
    'b50': 'b50*',
}
SOURCES = {
    'E045': ('#0072B2', 'o', 'E045: 0/10/20% (nowe)'),
    'E035': ('#D55E00', 's', 'E035 dev: b30/b50* (reuse U)'),
}
SEEDS = [42, 43, 44, 45]
JITTER = [-0.32, -0.11, 0.11, 0.32]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def pl(value, digits=2):
    return f'{value:.{digits}f}'.translate(str.maketrans({'.': ','})).replace('-', '−')


def pct_tick(value, _position):
    return pl(value, 0) if float(value).is_integer() else pl(value, 1)


def load():
    for name, expected in INPUTS.items():
        actual = sha256(ANALYSIS / name)
        assert actual == expected, f'{name}: {actual} != {expected}'
    summary = json.loads((ANALYSIS / 'summary.json').read_text(encoding='utf-8'))
    per_seed = read_csv(ANALYSIS / 'per_seed.csv')
    audit = json.loads((ANALYSIS / 'audit.json').read_text(encoding='utf-8'))

    assert audit['status'] == 'PASS'
    assert audit['phase'] == 'dev' and audit['val_used'] is False
    assert audit['four_gpu_exit0'] is True
    assert audit['rows'] == len(per_seed) == 20
    assert audit['teacher_exact_E035'] == 4
    assert summary['dataset'] == 'Tiny ImageNet dev'
    assert summary['seeds'] == SEEDS

    keys = [(int(row['seed']), row['budget']) for row in per_seed]
    assert len(keys) == len(set(keys)), 'duplikaty (seed, budżet)'
    assert set(keys) == {(seed, budget) for seed in SEEDS for budget in ORDER}
    assert {row['budget'] for row in per_seed} == set(ORDER)
    assert {row['source'] for row in per_seed if row['budget'] in ORDER[:3]} == {'E045'}
    assert {row['source'] for row in per_seed if row['budget'] in ORDER[3:]} == {'E035'}
    assert sum(row['source'] == 'E045' for row in per_seed) == summary['new_assessments'] == 12
    assert sum(row['source'] == 'E035' for row in per_seed) == summary['reused_assessments'] == 8

    points = {point['budget']: point for point in summary['points']}
    assert list(points) == ORDER, 'kolejność budżetów'
    teacher = points['teacher']
    assert teacher['mean_top1_pct'] == 51.22 and teacher['mean_loss_pp'] == 0.0
    for budget in ORDER:
        point = points[budget]
        assert point['source'] == ('E045' if budget in ORDER[:3] else 'E035')
        rows = [row for row in per_seed if row['budget'] == budget]
        top1 = [float(row['top1_pct']) for row in rows]
        loss = [float(row['loss_pp']) for row in rows]
        assert abs(statistics.mean(top1) - point['mean_top1_pct']) < 1e-9
        assert abs(statistics.mean(loss) - point['mean_loss_pp']) < 1e-9
        assert abs(statistics.stdev(top1) - point['sd_top1_pct']) < 1e-9
        assert abs(statistics.stdev(loss) - point['sd_loss_pp']) < 1e-9
        assert point['loss_per_seed_pp'] == loss
        assert len({row['params'] for row in rows}) == 1
        assert len({row['mac'] for row in rows}) == 1
        assert all(float(row['samples']) == 10000 for row in rows)
    for budget in ORDER[1:]:
        params = points[budget]['params']
        mac = points[budget]['mac']
        assert abs((1 - params / teacher['params']) * 100 - points[budget]['p_reduction_pct']) < 1e-9
        assert abs((1 - mac / teacher['mac']) * 100 - points[budget]['mac_reduction_pct']) < 1e-9
    return summary, per_seed, audit


def bands(ax, x, mean, sd, color):
    ax.fill_between(x, [m - s for m, s in zip(mean, sd)],
                    [m + s for m, s in zip(mean, sd)],
                    color=color, alpha=0.10, linewidth=0, zorder=1)


def panel(ax, summaries, per_seed, xkey, ykey, ylabel, xlabel, title,
          ylim, xlim, ydigits, show_values):
    mean = [point[ykey] for point in summaries]
    sd = [point['sd_top1_pct' if ykey == 'mean_top1_pct' else 'sd_loss_pp']
          for point in summaries]
    x = [point[xkey] for point in summaries]
    bands(ax, x, mean, sd, '#666666')
    ax.plot(x, mean, color='#333333', lw=1.7, zorder=2)
    ax.axhline(0.0, color='#999999', lw=0.8, ls=(0, (2, 2)), zorder=0)

    by_budget = {}
    for row in per_seed:
        by_budget.setdefault(row['budget'], []).append(row)
    for index, (budget, point) in enumerate(zip(ORDER, summaries)):
        color = '#222222' if budget == 'teacher' else SOURCES[point['source']][0]
        rows = sorted(by_budget[budget], key=lambda row: int(row['seed']))
        if budget != 'teacher':
            xs = [point[xkey] + JITTER[i] for i in range(len(rows))]
        else:
            xs = [point[xkey]] * len(rows)
        ys = [float(row['top1_pct' if ykey == 'mean_top1_pct' else 'loss_pp'])
              for row in rows]
        ax.plot(xs, ys, linestyle='none', marker='o', ms=4.3, mfc='white',
                mec=color, mew=1.0, zorder=3, clip_on=False)
        ax.plot(point[xkey], point[ykey], marker=SOURCES[point['source']][1]
                if budget != 'teacher' else 'D', ms=8.8, color=color,
                markeredgecolor='white', markeredgewidth=0.7, zorder=4)
        if show_values:
            box = {'boxstyle': 'round,pad=0.12', 'facecolor': 'white',
                   'edgecolor': 'none', 'alpha': 0.78}
            ax.annotate(pl(point[ykey], ydigits), (point[xkey], point[ykey]),
                        xytext=(10, 9), textcoords='offset points', ha='left',
                        va='bottom', fontsize=7.0, color='#333333', zorder=6,
                        bbox=box)
        abox = {'boxstyle': 'round,pad=0.10', 'facecolor': 'white',
                'edgecolor': 'none', 'alpha': 0.72}
        ax.annotate(LABELS[budget], (point[xkey], point[ykey]),
                    xytext=(10, -9), textcoords='offset points',
                    ha='left', va='top', fontsize=6.6, color='#666666',
                    zorder=6, bbox=abox)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(pct_tick))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(pct_tick))
    ax.tick_params(labelsize=8)
    ax.grid(color='#EAEAEA', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.set_title(title, loc='left', fontsize=10.4, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=8.6)
    ax.set_xlabel(xlabel, fontsize=8.6)


def draw(summary, per_seed):
    summaries = summary['points']
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 8.0))
    fig.subplots_adjust(left=0.062, right=0.985, top=0.870, bottom=0.310, wspace=0.20)

    panel(axes[0], summaries, per_seed, 'p_reduction_pct', 'mean_top1_pct',
          'bezwzględny top-1 [%]', 'rzeczywista redukcja P [%]',
          'Jakość vs redukcja parametrów', (14.0, 55.0), (-1.5, 54.0), 2, True)
    panel(axes[1], summaries, per_seed, 'mac_reduction_pct', 'mean_loss_pp',
          'sparowana strata pełny−U [pp]', 'rzeczywista redukcja MAC [%]',
          'Strata vs redukcja MAC', (-1.5, 35.5), (-1.5, 54.0), 2, True)

    handles = [
        Line2D([], [], color='#333333', lw=1.7, marker='o', ms=8.8,
               mfc='#333333', mec='white', label='średnia 4 seedów (po CMA)'),
        Line2D([], [], color='none', marker='o', ms=4.3, mfc='white',
               mec='#555555', mew=1.0, label='cztery seedy 42–45'),
        Line2D([], [], color='none', marker='o', ms=8.8, mfc=SOURCES['E045'][0],
               mec='white', label=SOURCES['E045'][2]),
        Line2D([], [], color='none', marker='s', ms=8.8, mfc=SOURCES['E035'][0],
               mec='white', label=SOURCES['E035'][2]),
        Patch(facecolor='#666666', alpha=0.10, label='±1 SD po seedach'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.188),
               ncol=3, frameon=False, fontsize=8.0, handletextpad=0.4,
               columnspacing=1.6)

    fig.suptitle('E045 — łagodny UNIFORM na Tiny ImageNet dev: krzywa po CMA, 4 seedy',
                 fontsize=12.6, fontweight='bold', y=0.973)
    fig.text(0.5, 0.240,
             'P wyznacza budżet (dyskretyzacja kanałów), MAC jest mierzony na obrazie 64×64.',
             ha='center', va='center', fontsize=8.0, color='#444444')
    caption = (
        'Tiny ImageNet dev (10 000 indeksów E035), ResNet18/Tiny64, cztery końcowe modele E019 '
        '(seedy 42–45), identyczny reset BN + CMA.\n'
        'Nowe pomiary E045 (12 ocen): pełny model 0% oraz UNIFORM b10/b20; odniesienia U/b30 i U/b50 '
        '(8 ocen, *) pochodzą z zamrożonego dev E035 i nie są liczone ponownie.\n'
        'Rzeczywiste redukcje różnią się od nominalnych: b10 P/MAC 10,35/10,67%, b20 20,33/19,97%, '
        'b30 30,44/30,13%, b50 50,15/49,13%. Brak punktów val; audyt PASS (20 wierszy, 4 GPU exit 0).\n'
        'Strata to różnica top-1 pełny−U na tym samym seedzie; punkty seedów mają lekki rozrzut '
        'poziomy tylko dla czytelności.'
    )
    fig.text(0.062, 0.032, caption, ha='left', va='bottom', fontsize=7.4,
             color='#333333', linespacing=1.5)
    return fig


def main():
    summary, per_seed, _audit = load()
    fig = draw(summary, per_seed)
    png = HERE / 'e045_krzywa.png'
    pdf = HERE / 'e045_krzywa.pdf'
    fig.savefig(png, dpi=200, metadata={'Software': 'matplotlib'})
    fig.savefig(pdf, metadata={'Title': 'E045 — łagodny UNIFORM na Tiny dev',
                               'Author': 'E045', 'Subject': 'dev, 0/10/20 + reuse 30/50, 4 seedy',
                               'Creator': 'make_figure.py'})
    plt.close(fig)
    for path in (png, pdf):
        print(path.relative_to(ROOT), sha256(path))


if __name__ == '__main__':
    main()
