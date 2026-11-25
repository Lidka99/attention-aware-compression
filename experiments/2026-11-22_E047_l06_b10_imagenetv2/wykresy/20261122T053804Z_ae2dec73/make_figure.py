"""E047, figura publikacyjna b10: krzywa Tiny dev i wynik ImageNetV2.
Wejście: audytowane CSV i agregaty E045–E047; wyjście: PDF i PNG obok skryptu.
Zasady oceny: ../../OPIS.md; ślad i sposób odtworzenia: README.md.
--english-paper zapisuje figurę manuskryptu z różnicami po seedach w obu panelach.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = {
    'E045': (
        '20261122T031350Z_a58b56a3',
        'ca6a0d74db6974d52f3795e82f3c7803503c225adfe78fa203aa03024b4e8811',
        '486309557455840de25852f166e55e6c909cbb1aa23e477418f8ed3057a2a992',
    ),
    'E046': (
        '20261122T045655Z_0b0c53b7',
        '807c71466358f61551cb56228a2ae7a68278ff00934e1c821f44213c1b581a1a',
        '836b32fa10842acfa7bf241c8edb8dc8aeef197951255082fe51862998d97165',
    ),
    'E047': (
        '20261122T053804Z_ae2dec73',
        '44ca52b1d5c7cddef4f71ed04787b1af1d73746bb2ceac867311d79694266d09',
        '8b3d1f263730692353002f1bd4787684ec458f14cc565dd2994670638874d49e',
    ),
}
SEEDS = (42, 43, 44, 45)
CURVE = ('teacher', 'b10', 'b20', 'b30', 'b50')
COMPARE = ('teacher', 'b10', 'L06')
LABELS = {'teacher': 'Pełny', 'b10': 'UNIFORM b10', 'L06': 'L06 b10'}
COLORS = {'teacher': '#454c57', 'b10': '#215d8a', 'L06': '#ae4619'}
MARKERS = {'teacher': 's', 'b10': 'o', 'L06': 'D'}


def load(experiment, profiles, profile_key, hits_key, samples_key, n):
    run, *hashes = SOURCES[experiment]
    src = ROOT / 'runs' / experiment / run / 'analysis'
    blobs = []
    for name, sha in zip(('summary.json', 'per_seed.csv'), hashes):
        blob = (src / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == sha, f'Niezgodne wejście: {src / name}'
        blobs.append(blob)
    summary = json.loads(blobs[0])
    assert tuple(summary['seeds']) == SEEDS
    raw = list(csv.DictReader(io.StringIO(blobs[1].decode())))
    rows = {(row[profile_key], int(row['seed'])): row for row in raw}
    assert len(raw) == len(rows) and set(rows) == {(p, s) for p in profiles for s in SEEDS}
    assert all(int(row[samples_key]) == n for row in raw)
    metrics = {}
    for profile in profiles:
        sample = [rows[profile, s] for s in SEEDS]
        (params, mac), = {(int(r['params']), int(r['mac'])) for r in sample}
        hits = [int(r[hits_key]) for r in sample]
        assert all(0 <= k <= n for k in hits)
        top1 = [100 * k / n for k in hits]
        loss = [100 * (int(rows['teacher', s][hits_key]) - k) / n for s, k in zip(SEEDS, hits)]
        assert top1 == [float(r['top1_pct']) for r in sample]
        assert loss == [float(r['loss_pp']) for r in sample]
        metrics[profile] = dict(params=params, mac=mac, hits=hits, top1=st.mean(top1), loss=st.mean(loss))
    return summary, metrics


def inputs():
    e45, uniform = load('E045', CURVE, 'budget', 'correct', 'samples', 10000)
    e46, dev = load('E046', COMPARE, 'profile', 'correct', 'samples', 10000)
    e47, external = load('E047', COMPARE, 'model', 'hits', 'n', 2000)
    assert e45['dataset'] == e46['dataset'] == 'Tiny ImageNet dev'
    assert e47['dataset'].startswith('ImageNetV2 MatchedFrequency') and e47['samples'] == 2000
    assert all(uniform[p] == dev[p] for p in COMPARE[:2])
    costs = ('params', 'mac')
    assert all(dev[p][k] == external[p][k] for p in COMPARE for k in costs)
    assert all(999 * dev['b10'][k] <= 1000 * dev['L06'][k] <= 1000 * dev['b10'][k] for k in costs)
    for point in e45['points']:
        p = point['budget']
        assert point['source'] == ('E035' if p in ('b30', 'b50') else 'E045')
        assert point['mean_top1_pct'] == uniform[p]['top1']
        assert point['mean_loss_pp'] == uniform[p]['loss']
        assert all(point[k] == uniform[p][k] for k in costs)
    for summary, measured in ((e46, dev), (e47, external)):
        for p in COMPARE:
            assert all(summary['models'][p][k] == measured[p][k] for k in costs)
            assert summary['models'][p]['top1_pct']['mean'] == measured[p]['top1']
            assert summary['models'][p]['loss_pp']['mean'] == measured[p]['loss']
    deltas = [100 * (l - u) / 2000 for l, u in zip(external['L06']['hits'], external['b10']['hits'])]
    effect = e47['loss_uniform_minus_l06_pp']
    assert deltas == effect['per_seed'] and st.mean(deltas) == effect['mean']
    assert math.isclose(external['b10']['loss'] - external['L06']['loss'], effect['mean'])
    assert effect['positive'] == sum(d > 0 for d in deltas) and effect['primary_pass']
    return uniform, dev, external, effect


def pl(value, digits=2, signed=False):
    return format(value, f'{"+" if signed else ""}.{digits}f').replace('.', ',').replace('-', '−')


def labeled_point(ax, x, y, profile, **kwargs):
    ax.plot(x, y, linestyle='none', marker=MARKERS[profile], markersize=6.5,
            color=COLORS[profile], markeredgecolor='white', markeredgewidth=0.65,
            zorder=4, **kwargs)


def style(ax, axis):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#e1e7ed', linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#a1acb8')
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=3, width=.6, color='#a1acb8', labelsize=8.4)


def draw(uniform, dev, external, effect):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'text.color': '#202a35',
                         'axes.labelcolor': '#202a35', 'xtick.color': '#344352', 'ytick.color': '#344352'})
    fig = plt.figure(figsize=(8.8, 4.75), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=.083, right=.957, bottom=.135, top=.816, wspace=.43)
    ax = fig.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(2, 1, height_ratios=(.78, 1), hspace=.74)
    abs_ax = fig.add_subplot(right[0])
    pair_ax = fig.add_subplot(right[1])

    def header(panel, title, subtitle):
        x = panel.get_position().x0
        fig.text(x, .947, title, ha='left', fontsize=12, fontweight='bold')
        fig.text(x, .888, subtitle, ha='left', fontsize=8.5, color='#4b5966')

    header(ax, 'A  Tiny ImageNet — dev', '10 000 obrazów · krzywa jakości względem redukcji P')
    header(abs_ax, 'B  ImageNetV2 — test zewnętrzny', '2 000 obrazów · 200 klas Tiny · te same modele b10')

    def reduction(p):
        return 100 * (1 - p['params'] / dev['teacher']['params'])

    ax.plot([reduction(uniform[p]) for p in CURVE], [uniform[p]['top1'] for p in CURVE],
            color=COLORS['b10'], marker='o', markersize=4.8, linewidth=1.35, label='UNIFORM')
    labeled_point(ax, 0, dev['teacher']['top1'], 'teacher', label='Pełny')
    l06, u10 = dev['L06'], dev['b10']
    labeled_point(ax, reduction(l06), l06['top1'], 'L06', label='L06 b10')
    ax.annotate(f'L06 b10: {pl(l06["top1"])}%', (reduction(l06), l06['top1']),
                xytext=(9, 9), textcoords='offset points', color=COLORS['L06'], fontsize=8.6,
                fontweight='bold')
    ax.annotate(f'{pl(u10["top1"])}%', (reduction(u10), u10['top1']), xytext=(-8, -6),
                textcoords='offset points', ha='right', color=COLORS['b10'], fontsize=8.2)
    ax.set(xlim=(-2, 53.5), ylim=(17, 55), xlabel='Rzeczywista redukcja parametrów [%]',
           ylabel='Średni top-1 po CMA [%]')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    style(ax, 'both')
    ax.legend(loc='lower left', frameon=False, fontsize=8.4, handlelength=1.7)

    for pos, p in zip((2, 1, 0), COMPARE):
        x = external[p]['top1']
        labeled_point(abs_ax, x, pos, p)
        abs_ax.annotate(pl(x), (x, pos), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color=COLORS[p], fontweight='bold' if p == 'L06' else 'normal')
    abs_ax.set(xlim=(25.7, 31.9), ylim=(-.5, 2.5), yticks=(2, 1, 0),
               yticklabels=[LABELS[p] for p in COMPARE], xlabel='Średni top-1 po CMA [%]')
    abs_ax.xaxis.set_major_locator(MultipleLocator(1))
    style(abs_ax, 'x')

    pair_ax.set_title('L06 b10 − UNIFORM b10', loc='left', fontsize=9, fontweight='bold', pad=4)
    pair_ax.axvline(0, color='#6d7781', linewidth=.9, linestyle='--', zorder=1)
    for pos, value in zip((4, 3, 2, 1), effect['per_seed']):
        pair_ax.plot(value, pos, marker='o', color=COLORS['L06'], markersize=4.5, zorder=3)
        pair_ax.annotate(pl(value, signed=True), (value, pos), xytext=(5, 0),
                         textcoords='offset points', va='center', fontsize=8.1)
    mean = effect['mean']
    low, high = effect['ci95_pp']
    pair_ax.errorbar(mean, 0, xerr=[[mean-low], [high-mean]], fmt='D', markersize=5.4,
                     color=COLORS['L06'], capsize=3, elinewidth=1.5, zorder=4)
    pair_ax.annotate(pl(mean, signed=True), (high, 0), xytext=(6, 0), textcoords='offset points',
                     va='center', color=COLORS['L06'], fontweight='bold', fontsize=8.4)
    pair_ax.set(xlim=(-.18, 3.04), ylim=(-.57, 4.51), yticks=(4, 3, 2, 1, 0),
                yticklabels=[f'Seed {s}' for s in SEEDS] + ['Średnia'], xlabel='Sparowana różnica top-1 [pp]')
    pair_ax.xaxis.set_major_locator(MultipleLocator(.5))
    pair_ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: pl(val, 1) if val % 1 else pl(val, 0)))
    style(pair_ax, 'x')
    return fig


def draw_english(_uniform, dev, external, effect):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'text.color': '#202a35',
                         'axes.labelcolor': '#202a35', 'xtick.color': '#344352', 'ytick.color': '#344352'})
    fig = plt.figure(figsize=(7.65, 4.75), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=(.88, 1.43), left=.158, right=.983, bottom=.150,
                            top=.837, wspace=.49, hspace=.92)
    dev_effect = [100*(l-u)/10000 for l, u in zip(dev['L06']['hits'], dev['b10']['hits'])]
    names = ('Full model', 'Uniform folding', 'Frozen offsets')
    for i, (title, subtitle, scores, delta, limits) in enumerate((
        ('A  Tiny ImageNet development', '10,000 images', dev, dev_effect, (45.3, 53.2)),
        ('B  ImageNetV2 subset', '2,000 images; 200 Tiny ImageNet classes', external, effect['per_seed'], (25.5, 32.2)),
    )):
        top = fig.add_subplot(grid[0, i])
        bottom = fig.add_subplot(grid[1, i])
        x = top.get_position().x0
        fig.text(x, .952, title, ha='left', fontsize=11, fontweight='bold')
        fig.text(x, .905, subtitle, ha='left', fontsize=9, color='#4b5966')
        for y, p in zip((2, 1, 0), COMPARE):
            score = scores[p]['top1']
            labeled_point(top, score, y, p)
            top.annotate(f'{score:.4f}', (score, y), xytext=(6, 0), textcoords='offset points',
                         va='center', fontsize=8.8, color=COLORS[p], fontweight='bold' if p == 'L06' else 'normal')
        top.set(xlim=limits, ylim=(-.5, 2.5), yticks=(2, 1, 0), yticklabels=names,
                xlabel='Mean top-1 (%)')
        top.xaxis.set_major_locator(MultipleLocator(2))
        style(top, 'x')

        bottom.set_title('Gain over uniform folding', loc='left', fontsize=9.5, fontweight='bold', pad=6)
        bottom.axvline(0, color='#6d7781', linewidth=.9, linestyle='--', zorder=1)
        for y, value in zip((4, 3, 2, 1), delta):
            bottom.plot(value, y, 'o', color=COLORS['L06'], markersize=4.5, zorder=3)
            bottom.annotate(f'{value:+.2f}', (value, y), xytext=(5, 0), textcoords='offset points',
                            va='center', fontsize=8.3)
        avg = st.mean(delta)
        mark_at = avg
        if i == 1:
            low, high = effect['ci95_pp']
            bottom.errorbar(avg, 0, xerr=[[avg-low], [high-avg]], fmt='D', markersize=5.4,
                            color=COLORS['L06'], capsize=3, elinewidth=1.35, zorder=4)
            mark_at = high
            fig.text(top.get_position().x1, .026, f'95% image-bootstrap CI: [{low:+.3f}, {high:+.3f}] pp',
                     ha='right', fontsize=8.4, color='#4b5966')
        else:
            labeled_point(bottom, avg, 0, 'L06')
        bottom.annotate(f'{avg:+.4f}', (mark_at, 0), xytext=(6, 0), textcoords='offset points', va='center',
                        color=COLORS['L06'], fontweight='bold', fontsize=8.3)
        bottom.set(xlim=(-.23, 3.17), ylim=(-.53, 4.53), yticks=(4, 3, 2, 1, 0),
                   yticklabels=[f'Seed {s}' for s in SEEDS] + ['Mean'], xlabel='Paired top-1 gain (pp)')
        bottom.xaxis.set_major_locator(MultipleLocator(.5))
        bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        style(bottom, 'x')
    return fig


def main():
    if not __debug__:
        raise RuntimeError('Uruchom bez -O: kontrole wejść używają assert.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--english-paper', action='store_true', help='Wygeneruj angielską figurę manuskryptu.')
    english = parser.parse_args().english_paper
    fig = (draw_english if english else draw)(*inputs())
    stem = 'b10_tiny_imagenetv2_en' if english else 'b10_tiny_imagenetv2'
    png, pdf = (HERE / f'{stem}.{ext}' for ext in ('png', 'pdf'))
    fig.savefig(png, dpi=300 if english else 230, metadata={'Software': f'Matplotlib {matplotlib.__version__}'})
    title = 'Frozen Offset Profile at mild compression: Tiny ImageNet dev and ImageNetV2' if english else 'Folding CNN przy b10: Tiny dev i ImageNetV2'
    fig.savefig(pdf, metadata={'Title': title,
                               'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(f'PASS: zgodność źródeł, kosztów i sparowanych trafień; Matplotlib {matplotlib.__version__}.')
    for path in (png, pdf):
        print(path.relative_to(ROOT), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
