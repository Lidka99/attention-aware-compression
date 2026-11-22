"""E042: wykres zapasu jakości 24 profili pod pułapami UNIFORM (dev).
Wejście: analysis/{profiles,per_seed}.csv i summary.json zakończonego runu
runs/E042/20261121T212257Z_207ae03f. Wyjście: dwa pliki tej figury (PNG, PDF).
Bez treningu, danych, wag i GPU; protokół i kryterium: ../../OPIS.md.
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
RUN = ROOT / 'runs/E042/20261121T212257Z_207ae03f'
INPUTS = {
    'profiles.csv': '670ff82e1859f8aa5bb407f798975ed566717a5473cadc11bc91d961cfcd9eaf',
    'per_seed.csv': '407dfa9fd3c23f36314db529b2e823b3ecd657b11b41f3240d7c5c5d6813c69a',
    'summary.json': '4950bfe1fc8344d134284b4b5b870624a2fb143b195427ee3e812cea750abc9b',
}
PAIRS = {
    'resnet_cifar100': ('CIFAR-100', 'główny'),
    'resnet_cifar10': ('CIFAR-10', 'kontrolny'),
}
FAMILIES = {
    'wewnatrz': ('wewnątrz etapu', '#0072B2', 'o'),
    'miedzy': ('między etapami', '#D55E00', 's'),
    'losowe': ('losowe', '#009E73', '^'),
}
PL_PLACE = str.maketrans({'.': ','})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=2):
    return f'{value:+.{digits}f}'.translate(PL_PLACE).replace('-', '−')


def pl_tick(value, _position):
    return f'{value:g}'.translate(PL_PLACE).replace('-', '−')


def pct(value):
    return f'{value:.3f}'.translate(PL_PLACE)


def load():
    for name, expected in INPUTS.items():
        actual = sha256(RUN / 'analysis' / name)
        assert actual == expected, f'{name}: {actual} != {expected}'
    profiles = read_csv(RUN / 'analysis/profiles.csv')
    per_seed = read_csv(RUN / 'analysis/per_seed.csv')
    summary = json.loads((RUN / 'analysis/summary.json').read_text(encoding='utf-8'))
    assert summary['phase'] == 'dev' and not summary['independent_validation']
    assert summary['evaluations'] == 216 and len(per_seed) == 216
    assert set(summary['contrasts']) == set(PAIRS)
    return profiles, per_seed, summary


def prepare(profiles, per_seed, summary):
    candidates, references, seeds = {}, {}, {}
    for pair in PAIRS:
        rows = [row for row in profiles if row['pair'] == pair]
        assert len(rows) == 26
        cand = [row for row in rows if row['family'] != 'referencja']
        assert len(cand) == 24
        assert {row['family'] for row in cand} == set(FAMILIES)
        assert sum(row['family'] == 'wewnatrz' for row in cand) == 8
        assert sum(row['family'] == 'miedzy' for row in cand) == 4
        assert sum(row['family'] == 'losowe' for row in cand) == 12
        stronger = summary['contrasts'][pair]['reference']
        assert stronger == 'U'
        assert [row['profile'] for row in rows if row['stronger_reference'] == 'True'] == [stronger]
        for row in rows:
            for key in ('params', 'mac', 'p_over_cap', 'mac_over_cap', 'top1',
                        'vs_stronger_ref_pp', 'sd_vs_stronger_ref_pp'):
                row[key] = float(row[key]) if key in ('p_over_cap', 'mac_over_cap', 'top1',
                                                      'vs_stronger_ref_pp', 'sd_vs_stronger_ref_pp') else int(row[key])
            row['positive_vs_stronger_ref'] = int(row['positive_vs_stronger_ref'])
            assert 0.999 <= row['p_over_cap'] <= 1.0
            assert 0.999 <= row['mac_over_cap'] <= 1.0
        assert max(row['p_over_cap'] for row in rows) <= 1.0
        assert max(row['mac_over_cap'] for row in rows) <= 1.0
        reference_row = next(row for row in rows if row['profile'] == stronger)
        assert reference_row['vs_stronger_ref_pp'] == 0.0

        per_pair = [row for row in per_seed if row['pair'] == pair]
        assert len(per_pair) == 27 * 4
        local = {}
        for row in per_pair:
            local.setdefault(row['profile'], {})[int(row['seed'])] = int(row['correct'])
        assert set(local) == {row['profile'] for row in rows} | {'teacher'}
        assert all(sorted(values) == [42, 43, 44, 45] for values in local.values())
        seeds[pair] = sorted(local[stronger])
        assert abs(statistics.mean(local['teacher'][s] for s in seeds[pair]) / 50 -
                   summary['contrasts'][pair]['teacher']) < 1e-12

        for row in rows:
            correct = local[row['profile']]
            assert abs(statistics.mean(correct[s] for s in seeds[pair]) / 50 - row['top1']) < 1e-12
            differences = [correct[s] - local[stronger][s] for s in seeds[pair]]
            assert abs(sum(differences) / 200 - row['vs_stronger_ref_pp']) < 1e-12
            assert sum(d > 0 for d in differences) == int(row['positive_vs_stronger_ref'])

        best = summary['contrasts'][pair]['best']
        best_row = next(row for row in rows if row['profile'] == best)
        assert best_row['best_candidate'] == 'True'
        assert sum(row['best_candidate'] == 'True' for row in rows) == 1
        assert best_row['vs_stronger_ref_pp'] == summary['contrasts'][pair]['mean_pp']
        assert best_row['positive_vs_stronger_ref'] == summary['contrasts'][pair]['positive']
        assert best_row['top1'] == summary['contrasts'][pair]['cma_best']
        assert next(row for row in rows if row['profile'] == 'U')['top1'] == summary['contrasts'][pair]['cma_U']
        assert next(row for row in rows if row['profile'] == 'T-proj')['top1'] == summary['contrasts'][pair]['cma_T_proj']
        assert summary['contrasts'][pair]['per_seed_pp'] == [
            (local[best][s] - local[stronger][s]) / 50 for s in seeds[pair]]
        assert [local[best][s] - local[stronger][s] for s in seeds[pair]] == \
            summary['contrasts'][pair]['per_seed_count']

        candidates[pair] = {row['profile']: row for row in cand}
        assert len(candidates[pair]) == 24
        references[pair] = {row['profile']: row for row in rows if row['family'] == 'referencja'}
    primary = candidates[summary['primary_pair']]
    order = sorted(primary, key=lambda name: (-primary[name]['vs_stronger_ref_pp'], name))
    for pair, by_id in candidates.items():
        assert set(by_id) == set(order)
        candidates[pair] = [by_id[name] for name in order]
    return candidates, references, seeds


def panel(ax, pair, rows, references, seeds, per_seed, title):
    n = len(rows)
    y = list(range(n))
    best = next(index for index, row in enumerate(rows) if row['best_candidate'] == 'True')
    ax.axhspan(best - 0.45, best + 0.45, color='#FFF1B8', zorder=0)
    for index, row in enumerate(rows):
        _, color, marker = FAMILIES[row['family']]
        ax.hlines(index, 0, row['vs_stronger_ref_pp'], color='#BDBDBD', lw=0.9, zorder=1)
        ax.plot(row['vs_stronger_ref_pp'], index, marker=marker, ms=6.4, color=color,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)
    best_row = rows[best]
    ax.plot(best_row['vs_stronger_ref_pp'], best, marker='o', ms=12, mfc='none',
            mec='#222222', mew=1.1, zorder=4)
    correct = {}
    for row in per_seed:
        if row['pair'] == pair:
            correct.setdefault(int(row['seed']), {})[row['profile']] = int(row['correct'])
    paired = [(correct[s][best_row['profile']] - correct[s]['U']) / 50 for s in seeds]
    strip = best - 0.28
    ax.hlines(strip, min(paired), max(paired), color='#777777', lw=0.8, zorder=3)
    ax.plot(paired, [strip] * len(paired), linestyle='none', marker='o', ms=4.0,
            mfc='white', mec='#222222', mew=0.8, zorder=5)
    ax.annotate(f'{fmt(best_row["vs_stronger_ref_pp"])} pp', (best_row['vs_stronger_ref_pp'], best),
                xytext=(0, -10), textcoords='offset points', va='top', ha='center',
                fontsize=7.5, fontweight='bold', color='#222222', zorder=6)

    values = [row['vs_stronger_ref_pp'] for row in rows]
    median = statistics.median(values)
    positive = sum(value > 0 for value in values)
    robust = sum(value > 0 and int(row['positive_vs_stronger_ref']) >= 3 for value, row in zip(values, rows))
    reference = references[pair]['U']
    details = (f'ref. U {reference["top1"]:.3f}%'.translate(PL_PLACE) +
               f' · mediana {fmt(median, 4) if abs(median) < 0.01 else fmt(median)}' +
               f' · {positive}/{n} średnich > 0 · {robust}/{n} ≥3/4 seedów')
    if pair == 'resnet_cifar100':
        details += f' · bramka: {best_row["profile"]} ({best_row["positive_vs_stronger_ref"]}/4)'

    ax.axvline(0.0, color='#444444', lw=0.9, zorder=2)
    ax.axvline(1.0, color='#C0392B', lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.axvline(median, color='#777777', lw=1.0, ls=(0, (1, 2)), zorder=2)
    ax.grid(axis='x', color='#E8E8E8', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-4.6, 2.1)
    ax.set_ylim(n - 0.4, -1.15)
    ax.set_yticks(y)
    ax.set_yticklabels([row['profile'] for row in rows], fontsize=7.4)
    for label, row in zip(ax.get_yticklabels(), rows):
        if row['best_candidate'] == 'True':
            label.set_fontweight('bold')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(pl_tick))
    ax.tick_params(axis='x', labelsize=7.4)
    ax.tick_params(axis='y', length=2.5, pad=1.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.set_title(f'{title}  (n={n})', loc='left', fontsize=10.2, fontweight='bold', pad=15)
    ax.text(0.0, 1.004, details, transform=ax.transAxes, fontsize=7.3,
            color='#444444', va='bottom', ha='left')
    ax.text(1.035, n - 0.06, '+1 pp', color='#C0392B', fontsize=7.0, va='center', ha='left')
    ax.text(median, -0.82, f'mediana {fmt(median, 4) if abs(median) < 0.01 else fmt(median)}',
            color='#666666', fontsize=6.8, va='center', ha='center',
            bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})


def draw(candidates, references, seeds):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.862, bottom=0.255, wspace=0.055)
    per_seed = read_csv(RUN / 'analysis/per_seed.csv')
    for ax, pair in zip(axes, PAIRS):
        name, role = PAIRS[pair]
        panel(ax, pair, candidates[pair], references, seeds[pair], per_seed,
              f'{name} — {role}')
    axes[1].tick_params(axis='y', labelleft=False)

    handles = [Line2D([], [], linestyle='none', marker=marker, ms=6.4, color=color,
                      markeredgecolor='white', markeredgewidth=0.5,
                      label=f'{label} ({sum(row["family"] == family for row in candidates["resnet_cifar100"])})')
               for family, (label, color, marker) in FAMILIES.items()]
    handles.append(Line2D([], [], linestyle='none', marker='o', ms=4.0, mfc='white',
                          mec='#222222', mew=0.8, label='L06: 4 seedy (Δ per seed)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.168), ncol=4,
               frameon=False, fontsize=7.8, handletextpad=0.5, columnspacing=1.8)

    fig.suptitle('E042 — zapas jakości pod pułapami UNIFORM: 24 profili, dev, 4 seedy',
                 fontsize=12.4, fontweight='bold', y=0.978)
    fig.text(0.5, 0.205, 'Δ średniej top-1 względem mocniejszej referencji [pp]',
             ha='center', va='center', fontsize=8.6, color='#222222')
    u100, u10 = references['resnet_cifar100']['U'], references['resnet_cifar10']['U']
    t100, t10 = references['resnet_cifar100']['T-proj'], references['resnet_cifar10']['T-proj']
    caption = (
        'Dev 5000, bez testu i rezerwy; Δ = średnia CMA po 4 sparowanych seedach (42–45) top-1 kandydata '
        'minus mocniejsza z referencji U/T-proj.\n'
        f'Mocniejsza referencja: U — CIFAR-100 {pct(u100["top1"])}%, CIFAR-10 {pct(u10["top1"])}% '
        f'(T-proj: {pct(t100["top1"])}% / {pct(t10["top1"])}%, czyli Δ = '
        f'{fmt(t100["vs_stronger_ref_pp"])} / {fmt(t10["vs_stronger_ref_pp"])} pp).\n'
        'Rzeczywiste P i MAC wszystkich 24 profili mieszczą się w [99,9%, 100%] pułapów U; '
        'kreska +1 pp jest progiem decyzyjnym (ilustracyjnym), nie testem istotności.\n'
        'Wybór najlepszego profilu na tym samym dev może zawyżać ocenę zapasu; wynik dotyczy tylko tej puli.'
    )
    fig.text(0.055, 0.030, caption, ha='left', va='bottom', fontsize=7.3,
             color='#333333', linespacing=1.45)
    return fig


def main():
    profiles, per_seed, summary = load()
    candidates, references, seeds = prepare(profiles, per_seed, summary)
    fig = draw(candidates, references, seeds)
    png = HERE / 'e042_zapas_dev.png'
    pdf = HERE / 'e042_zapas_dev.pdf'
    fig.savefig(png, dpi=200, metadata={'Software': 'matplotlib'})
    fig.savefig(pdf, metadata={'Title': 'E042 — zapas jakości pod pułapami UNIFORM (dev)',
                               'Author': 'E042', 'Subject': 'dev, 24 profile, 4 seedy',
                               'Creator': 'make_figure.py'})
    plt.close(fig)
    for path in (png, pdf):
        print(path.relative_to(ROOT), sha256(path))
    for pair in PAIRS:
        values = [row['vs_stronger_ref_pp'] for row in candidates[pair]]
        best = next(row for row in candidates[pair] if row['best_candidate'] == 'True')
        print(pair, f'median={statistics.median(values):+.4f}',
              f'positive={sum(v > 0 for v in values)}/24',
              f'best={best["profile"]} {best["vs_stronger_ref_pp"]:+.4f}',
              f'{best["positive_vs_stronger_ref"]}/4',
              f'p=[{min(row["p_over_cap"] for row in candidates[pair]):.6f},'
              f'{max(row["p_over_cap"] for row in candidates[pair]):.6f}]',
              f'mac=[{min(row["mac_over_cap"] for row in candidates[pair]):.6f},'
              f'{max(row["mac_over_cap"] for row in candidates[pair]):.6f}]')


if __name__ == '__main__':
    main()
