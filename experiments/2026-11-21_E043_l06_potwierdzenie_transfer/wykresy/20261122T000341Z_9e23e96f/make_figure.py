"""E043: figura rezerwy CIFAR-100 — top-1 po CMA i sparowany kontrast L06−U.

Wejście (tylko do odczytu): analysis/{per_seed.csv,summary.json,audit.json}
z zakończonego runu runs/E043/20261122T000341Z_9e23e96f. Wyjście: PNG i PDF
w tym katalogu. Bez treningu, danych, wag, inferencji i GPU; protokół:
../../OPIS.md. Skrypt sprawdza hashe wejść i zgodność liczb z summary.json.
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
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'runs/E043/20261122T000341Z_9e23e96f'
INPUTS = {
    'per_seed.csv': 'f64ea831be849ec70bbd8a7fc9081e01045e7bbc6fd91e3785b745b7a90b5e20',
    'summary.json': 'c130791164727cf9a366941e52af1f9886b21c41f1c4992c98792b9dbf06ca96',
    'audit.json': 'a2004ce35b783409250b2e00c7e780979383654ad7ffe79dae9e310af5a3c5e2',
}
SEEDS = [42, 43, 44, 45]
BUDGETS = {'b50': '~50% redukcji P', 'b30': '~30% redukcji P'}
MODELS = {
    'U': ('UNIFORM (U)', '#0072B2', 'o'),
    'T-proj': ('T-proj', '#E69F00', 's'),
    'L06': ('L06', '#009E73', 'D'),
}
TEACHER = ('bez kompresji', '#444444')
PL_PLACE = str.maketrans({'.': ','})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def pl_number(value, digits=3):
    return f'{value:.{digits}f}'.translate(PL_PLACE)


def pl_pp(value, digits=3):
    return f'{value:+.{digits}f}'.translate(PL_PLACE).replace('-', '−')


def pl_tick(value, _position):
    return f'{value:g}'.translate(PL_PLACE).replace('-', '−')


def load():
    for name, expected in INPUTS.items():
        actual = sha256(RUN / 'analysis' / name)
        assert actual == expected, f'{name}: {actual} != {expected}'
    per_seed = read_csv(RUN / 'analysis/per_seed.csv')
    summary = json.loads((RUN / 'analysis/summary.json').read_text(encoding='utf-8'))
    audit = json.loads((RUN / 'analysis/audit.json').read_text(encoding='utf-8'))

    assert audit['status'] == 'PASS'
    assert audit['run'] == 'runs/E043/20261122T000341Z_9e23e96f'
    assert audit['rows'] == len(per_seed) == 28
    assert audit['gpu_count'] == 4
    assert audit['e042_states_equal'] == audit['e042_states_checked'] == 16
    for check in ('four_gpu_exit0', 'all_paired_counts', 'BN_and_model_state'):
        assert check in audit['checks']

    assert summary['seeds'] == SEEDS
    assert summary['samples_per_seed'] == 5000
    assert summary['rule'] == {
        'primary_budget': 'b50',
        'transfer_budget': 'b30',
        'contrast': 'L06-U',
        'metric': 'top1_CMA',
        'min_gain_pp': 1.0,
        'min_sum_count_gain': 200,
        'min_strictly_positive_seeds': 3,
    }
    assert summary['primary_passed'] is True
    assert summary['transfer_passed'] is False
    return per_seed, summary


def prepare(per_seed, summary):
    cells = {}
    p_reduction = {}
    for row in per_seed:
        assert int(row['samples']) == 5000
        key = 'teacher' if row['key'] == 'teacher' else row['key']
        seed = int(row['seed'])
        assert seed in SEEDS and row['gpu'] == str(SEEDS.index(seed))
        correct = int(row['correct'])
        top1 = float(row['top1_pct'])
        assert abs(correct / 50.0 - top1) < 1e-9
        assert row['budget'] == ('-' if key == 'teacher' else key.split('/')[0])
        cells.setdefault(key, {})[seed] = correct
        if key.endswith('/L06'):
            p_reduction.setdefault(key.split('/')[0], []).append(float(row['p_reduction_pct']))
    assert set(cells) == {'teacher'} | {f'{budget}/{model}' for budget in BUDGETS for model in MODELS}
    assert all(sorted(values) == SEEDS for values in cells.values())
    assert {budget: round(statistics.mean(values), 9) for budget, values in p_reduction.items()} == \
        {'b50': 50.299007177, 'b30': 30.075225496}
    for values in p_reduction.values():
        assert max(values) - min(values) < 1e-9

    for budget in BUDGETS:
        means = summary['budgets'][budget]['mean_top1_pct']
        for model in MODELS:
            mean = statistics.mean(cells[f'{budget}/{model}'][s] for s in SEEDS) / 50.0
            assert abs(mean - means[model]) < 1e-12
            contrast = summary['budgets'][budget]['contrasts']['L06-U']
            differences = [(cells[f'{budget}/L06'][s] - cells[f'{budget}/U'][s]) / 50.0 for s in SEEDS]
            assert differences == contrast['per_seed_pp']
            assert abs(statistics.mean(differences) - contrast['mean_pp']) < 1e-12
            assert abs(statistics.stdev(differences) - contrast['sd_pp']) < 1e-12
            assert sum(value > 0 for value in differences) == contrast['positive']
            assert sum(d > 0 for d in differences) == contrast['positive']
            assert sum(int(cells[f'{budget}/L06'][s]) - int(cells[f'{budget}/U'][s])
                       for s in SEEDS) == contrast['sum_counts']

        loss = summary['budgets'][budget]['loss_vs_teacher']
        for model in MODELS:
            differences = [(cells['teacher'][s] - cells[f'{budget}/{model}'][s]) / 50.0 for s in SEEDS]
            assert abs(statistics.mean(differences) - loss[model]['mean_pp']) < 1e-12
            assert differences == loss[model]['per_seed_pp']

    teacher_mean = statistics.mean(cells['teacher'][s] for s in SEEDS) / 50.0
    for budget in BUDGETS:
        assert abs(summary['budgets'][budget]['mean_top1_pct']['teacher'] - teacher_mean) < 1e-12
    assert abs(summary['rule']['min_gain_pp'] - 1.0) < 1e-12
    return cells, teacher_mean, {b: statistics.mean(v) for b, v in p_reduction.items()}


def panel_absolute(ax, cells, teacher_mean):
    centers = {'b50': 1.9, 'b30': 4.15}
    positions = {}
    for budget, center in centers.items():
        offsets = {'U': -0.42, 'T-proj': 0.0, 'L06': 0.42}
        positions[budget] = {model: center + offsets[model] for model in MODELS}

    jitter = {seed: (-0.060, -0.020, 0.020, 0.060)[index] for index, seed in enumerate(SEEDS)}
    for seed in SEEDS:
        value = cells['teacher'][seed] / 50.0
        ax.plot(0.0 + jitter[seed], value, linestyle='none', marker='.', ms=5.0,
                color=TEACHER[1], alpha=0.55, zorder=3)
    ax.plot(0.0, teacher_mean, linestyle='none', marker='o', ms=9.5, color=TEACHER[1],
            markeredgecolor='white', markeredgewidth=1.0, zorder=4)
    ax.annotate(pl_number(teacher_mean, 2), (0.0, teacher_mean), xytext=(0, 9),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=8.4, fontweight='bold', color=TEACHER[1], zorder=5)

    for budget in BUDGETS:
        for model, (_, color, marker) in MODELS.items():
            x = positions[budget][model]
            values = [cells[f'{budget}/{model}'][s] / 50.0 for s in SEEDS]
            for seed, value in zip(SEEDS, values):
                ax.plot(x + jitter[seed], value, linestyle='none', marker='.',
                        ms=5.0, color=color, alpha=0.55, zorder=3)
            mean = statistics.mean(values)
            ax.plot(x, mean, linestyle='none', marker=marker, ms=9.5, color=color,
                    markeredgecolor='white', markeredgewidth=1.0, zorder=4)
            if model == 'U':
                xytext, va = (0, -10), 'top'
            else:
                xytext, va = (0, 9), 'bottom'
            ax.annotate(pl_number(mean), (x, mean), xytext=xytext,
                        textcoords='offset points', ha='center', va=va,
                        fontsize=8.4, fontweight='bold', color=color, zorder=5)

    for x in (0.95, 3.05):
        ax.axvline(x, color='#DDDDDD', lw=0.9, zorder=0)
    ax.set_xlim(-0.62, 4.95)
    ax.set_ylim(46.6, 79.0)
    ax.set_xticks([0.0, 1.9, 4.15])
    ax.set_xticklabels(['bez kompresji', 'b50\n~50% redukcji P', 'b30\n~30% redukcji P'],
                       fontsize=8.5)
    ax.set_ylabel('top-1 po CMA [%]', fontsize=9.0)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(pl_tick))
    ax.grid(axis='y', color='#EDEDED', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=8.0, length=3, colors='#333333')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.set_title('A. Bezwzględny top-1 (średnia 4 seedów, kropki = seedy)',
                 loc='left', fontsize=10.2, fontweight='bold', pad=8)


def panel_contrast(ax, cells, summary):
    centers = {'b50': 1.0, 'b30': 2.0}
    jitter = {seed: (-0.085, -0.028, 0.028, 0.085)[index] for index, seed in enumerate(SEEDS)}
    for budget, x in centers.items():
        contrast = summary['budgets'][budget]['contrasts']['L06-U']
        differences = contrast['per_seed_pp']
        assert differences == [(cells[f'{budget}/L06'][s] - cells[f'{budget}/U'][s]) / 50.0
                               for s in SEEDS]
        ax.vlines(x, min(differences), max(differences), color='#BBBBBB', lw=1.4, zorder=2)
        for seed, value in zip(SEEDS, differences):
            ax.plot(x + jitter[seed], value, linestyle='none', marker='o', ms=5.2,
                    mfc='white', mec='#222222', mew=1.0, zorder=4)
        mean = statistics.mean(differences)
        ax.plot(x, mean, linestyle='none', marker='D', ms=8.6, color='#111111',
                markeredgecolor='white', markeredgewidth=0.9, zorder=5)
        positive = sum(value > 0 for value in differences)
        if budget == 'b50':
            label = f'średnia {pl_pp(mean)} pp ({positive}/4 > 0)'
            ax.annotate(label, (x, mean), xytext=(0, 12), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.4, fontweight='bold',
                        color='#111111', zorder=6)
        else:
            shortfall = 1.0 - mean
            label = (f'średnia {pl_pp(mean)} pp ({positive}/4 > 0)\n'
                     f'o {pl_number(shortfall)} pp poniżej progu')
            ax.annotate(label, (x, mean), xytext=(0, -14), textcoords='offset points',
                        ha='center', va='top', fontsize=8.2, fontweight='bold',
                        color='#111111', zorder=6,
                        bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white',
                              'edgecolor': 'none', 'alpha': 0.92})

    ax.axhline(0.0, color='#555555', lw=0.9, zorder=1)
    ax.axhline(1.0, color='#C0392B', lw=1.1, ls=(0, (4, 2)), zorder=1)
    ax.text(0.55, 1.03, 'próg +1,00 pp', color='#C0392B', fontsize=7.6,
            va='bottom', ha='left', fontweight='bold',
            bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white',
                  'edgecolor': 'none', 'alpha': 0.9})
    ax.text(0.55, 0.04, 'brak różnicy', color='#666666', fontsize=7.2,
            va='bottom', ha='left')

    ax.set_xlim(0.52, 2.55)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(['b50', 'b30'], fontsize=8.6)
    ax.set_ylabel('L06 − U [pp]', fontsize=9.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FuncFormatter(pl_tick))
    ax.grid(axis='y', color='#EDEDED', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=8.0, length=3, colors='#333333')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.set_title('B. Sparowany kontrast L06 − U (punkty = seedy)',
                 loc='left', fontsize=10.2, fontweight='bold', pad=8)


def draw(cells, teacher_mean, p_reduction, summary):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 7.6),
                                     gridspec_kw={'width_ratios': [1.34, 1.0]})
    fig.subplots_adjust(left=0.062, right=0.983, top=0.815, bottom=0.270, wspace=0.16)
    panel_absolute(ax_a, cells, teacher_mean)
    panel_contrast(ax_b, cells, summary)

    handles = [Line2D([], [], linestyle='none', marker=marker, ms=8.0, color=color,
                      markeredgecolor='white', markeredgewidth=0.9, label=label)
               for model, (label, color, marker) in MODELS.items()]
    handles.append(Line2D([], [], linestyle='none', marker='o', ms=8.5, color=TEACHER[1],
                          markeredgecolor='white', markeredgewidth=0.9,
                          label=TEACHER[0]))
    handles.append(Line2D([], [], linestyle='none', marker='.', ms=7.0, color='#666666',
                          label='pojedynczy seed (42–45)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.893), ncol=5,
               frameon=False, fontsize=8.2, handletextpad=0.45, columnspacing=1.7)

    fig.suptitle('E043 — L06 na rezerwie CIFAR-100: top-1 po CMA i kontrast L06−U',
                 fontsize=12.8, fontweight='bold', y=0.977)
    fig.text(0.5, 0.947,
             'ResNet18, cztery checkpointy E015 (seedy 42–45), ta sama rezerwa 5 000 obrazów',
             ha='center', va='center', fontsize=8.8, color='#444444')

    c50 = summary['budgets']['b50']['contrasts']['L06-U']
    c30 = summary['budgets']['b30']['contrasts']['L06-U']
    l50 = summary['budgets']['b50']['loss_vs_teacher']['L06']
    l30 = summary['budgets']['b30']['loss_vs_teacher']['L06']
    threshold = summary['rule']['min_gain_pp']
    shortfall = threshold - c30['mean_pp']
    caption = (
        'Panel A: top-1 po CMA, średnia 4 seedów 42–45 (duże znaczniki); małe kropki to pojedyncze seedy. '
        'Bez kompresji = ten sam model bez foldingu.\n'
        f'Rzeczywista redukcja P przez L06: {pl_number(p_reduction["b50"])}% (b50) i '
        f'{pl_number(p_reduction["b30"])}% (b30); P i MAC profili U/T-proj/L06 mieszczą się '
        'w [99,9%, 100%] U.\n'
        'Panel B: różnica trafień L06 − U na tych samych 5 000 obrazach, w pp dla każdego seeda '
        '(puste kropki), oraz jej średnia (romb).\n'
        f'Kreska +{pl_number(threshold, 2)} pp to próg zamrożony w protokole; średnie '
        f'{pl_pp(c50["mean_pp"])} pp (b50, {c50["positive"]}/4 seedy > 0) oraz '
        f'{pl_pp(c30["mean_pp"])} pp (b30, {c30["positive"]}/4 > 0; o {pl_number(shortfall)} pp poniżej progu).\n'
        f'Strata L06 do nauczyciela: {pl_number(l50["mean_pp"])} pp (b50) i {pl_number(l30["mean_pp"])} pp (b30). '
        f'Run {RUN.name}, audyt PASS.\n'
        'Wartości są opisowe; figura nie zawiera testu istotności ani przedziałów ufności, '
        'a próg nie jest testem statystycznym.\n'
        'Źródło: analysis/per_seed.csv i summary.json (4 sparowane seedy, rezerwa użyta raz); '
        'figura liczona bez treningu, inferencji i GPU.'
    )
    fig.text(0.062, 0.030, caption, ha='left', va='bottom', fontsize=7.2,
             color='#333333', linespacing=1.55)
    return fig


def main():
    per_seed, summary = load()
    cells, teacher_mean, p_reduction = prepare(per_seed, summary)
    fig = draw(cells, teacher_mean, p_reduction, summary)
    png = HERE / 'e043_rezerwa.png'
    pdf = HERE / 'e043_rezerwa.pdf'
    fig.savefig(png, dpi=200, metadata={'Software': 'matplotlib'})
    fig.savefig(pdf, metadata={'Title': 'E043 — L06 na rezerwie CIFAR-100',
                               'Author': 'E043', 'Subject': 'top-1 po CMA i kontrast L06-U',
                               'Creator': 'make_figure.py'})
    plt.close(fig)
    for path in (png, pdf):
        print(path.relative_to(ROOT), sha256(path))
    for budget in BUDGETS:
        c = summary['budgets'][budget]['contrasts']['L06-U']
        means = summary['budgets'][budget]['mean_top1_pct']
        print(budget,
              f'teacher={means["teacher"]:.3f}', f'U={means["U"]:.3f}',
              f'T-proj={means["T-proj"]:.3f}', f'L06={means["L06"]:.3f}',
              f'L06-U mean={c["mean_pp"]:+.4f}',
              f'per_seed={[f"{v:+.2f}" for v in c["per_seed_pp"]]}',
              f'positive={c["positive"]}/4',
              f'L06 loss={summary["budgets"][budget]["loss_vs_teacher"]["L06"]["mean_pp"]:.4f}')


if __name__ == '__main__':
    main()
