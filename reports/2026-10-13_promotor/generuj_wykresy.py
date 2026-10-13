"""Prezentacja podsumowania badań (2026-10-13): generuje wszystkie wykresy z zamrozonych wynikow.
Wejscia: runs/E001, E005, E007, E008, E009 (metryki jsonl/json) - tylko odczyt.
Wyjscia: wykresy/*.png w tym folderze. CPU-only, bez GPU i bez treningu.
E009 czyta najnowszy katalog runs/E009/2* (obsluguje stan czesciowy: brak pliku wariantu
= "w kolejce", epoka < 120 = "w toku"); po zakonczeniu biegu wykresy same sie odswieza.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/users/s224574/attention/badania_atencja_cnn')
OUT = HERE / 'wykresy'
OUT.mkdir(exist_ok=True)

E001 = ROOT / 'runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/metrics.jsonl'
E005 = ROOT / 'runs/E005/20261011T002030Z_3c93ed8_seed42_w7hggu4o/metrics.jsonl'
E007 = ROOT / 'runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80/metrics.jsonl'
E008_RUN = ROOT / 'runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim'
E008_S1 = E008_RUN / 'stage1/stage1_metrics.json'
E008_S2 = E008_RUN / 'stage2/stage2_metrics.json'

VARIANTS = ('BASE', 'PRE', 'POST', 'RANDOM')
COLORS = {'BASE': '#444444', 'PRE': '#1f77b4', 'POST': '#d62728', 'RANDOM': '#7f7f7f'}
E009_TOTAL_FALLBACK = 120


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print('saved', OUT / name)


def fmt_pl(value, nd=2):
    return f'{value:.{nd}f}'.replace('.', ',')


def find_e009_run():
    base = ROOT / 'runs/E009'
    candidates = sorted(p for p in base.iterdir()
                        if p.is_dir() and p.name[0].isdigit() and (p / 'command.sh').exists())
    if not candidates:
        raise RuntimeError('brak katalogu biegu E009')
    return candidates[-1]


def load_e009():
    """Zwraca stan E009 gotowy do wykresow i do odswiez_e009.py."""
    run = find_e009_run()
    total = E009_TOTAL_FALLBACK
    cfg_path = run / 'config.resolved.json'
    if cfg_path.exists():
        total = json.loads(cfg_path.read_text()).get('epochs', total)
    variants = {}
    for name in VARIANTS:
        path = run / f'metrics_{name}.jsonl'
        if path.exists():
            data = rows(path)
            variants[name] = {'epoch': data[-1]['epoch'], 'total': total,
                              'top1': data[-1]['dev_top1'], 'rows': data}
        else:
            variants[name] = {'epoch': 0, 'total': total, 'top1': None, 'rows': []}
    summary = None
    if (run / 'summary.json').exists():
        summary = json.loads((run / 'summary.json').read_text())
    finished = all(v['epoch'] >= v['total'] for v in variants.values())
    return {'run': run, 'variants': variants, 'summary': summary, 'finished': finished,
            'now_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}


def fig_e001_curve():
    e1 = rows(E001)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    epochs = [r['epoch'] for r in e1]
    ax.plot(epochs, [100 * r['dev_top1'] for r in e1], color='#1f77b4', label='dev top-1')
    ax.plot(epochs, [100 * r['train_correct'] / r['train_count'] for r in e1],
            color='#999999', label='train top-1')
    ax.annotate('dev 66,84% (best, ep. 177)', xy=(177, 66.84), xytext=(100, 82),
                arrowprops=dict(arrowstyle='->', color='#333333'), fontsize=9)
    ax.set_title('Baseline CIFAR-100, 96x96, 200 epok')
    ax.set_xlabel('epoka')
    ax.set_ylabel('top-1 [%]')
    ax.set_ylim(0, 100)
    ax.legend(loc='center right')
    ax.grid(alpha=0.3)
    save(fig, 'fig_baseline_cifar.png')


def fig_e005_e007_curves():
    e5 = rows(E005)
    e7 = rows(E007)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for data, label, color in ((e5, 'baseline 96x96 -> dev 51,53%', '#1f77b4'),
                               (e7, 'baseline 224x224 -> dev 58,49%', '#d62728')):
        ax.plot([r['epoch'] for r in data], [100 * r['dev_top1'] for r in data],
                color=color, label=label)
    ax.set_title('Tiny ImageNet: baseline 96 vs 224 (100 epok)')
    ax.set_xlabel('epoka')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(0, 65)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    save(fig, 'fig_baselines_tiny.png')


def fig_e003_raw():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    names = ['BASE', 'PRE', 'POST', 'RANDOM']
    values = [66.84, 54.94, 55.24, 49.48]
    bars = ax.bar(names, values, color=[COLORS[n] for n in names], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f'{value:.2f}',
                ha='center', fontsize=10)
    ax.set_title('Kompresja 10% na CIFAR-100, bez naprawy')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(0, 75)
    ax.grid(axis='y', alpha=0.3)
    save(fig, 'fig_kompresja_10_cifar.png')


def fig_repair():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    names = ['BASE', 'PRE', 'POST', 'RANDOM']
    values = [67.62, 66.96, 67.32, 67.30]
    bars = ax.bar(names, values, color=[COLORS[n] for n in names], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f'{value:.2f}',
                ha='center', fontsize=10)
    ax.set_title('CIFAR-100 po rownym dostrajaniu (120 epok)')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(64, 69)
    ax.grid(axis='y', alpha=0.3)
    ax.annotate('RANDOM dorownal: POST-PRE +0,36 pp,\nCI [-0,30; +1,02]',
                xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', fc='#fff7e6', ec='#cc9900'))
    save(fig, 'fig_dostrajanie_cifar.png')


def fig_se_schemat():
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.axis('off')
    boxes = [
        ('Mapy cech wejściowych\nC kanałów × H × W', '#eef3fb'),
        ('SQUEEZE\nśrednia po H×W (na kanał)\n→ C liczb: „ile kanał mówi"', '#e8f5e9'),
        ('EXCITATION\ndwie warstwy FC + sigmoid\n→ C wag w zakresie 0–1', '#fff7e6'),
        ('SCALE\nkanał × jego waga\n(dla każdego obrazu inaczej)', '#fdeaea'),
        ('Wyjście: te same mapy,\nważone „uwagą" dla tego obrazu', '#f3ecff'),
    ]
    width, gap = 0.178, 0.021
    x = 0.02
    for index, (text, color) in enumerate(boxes):
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.36), width, 0.40,
                     boxstyle='round,pad=0.012', linewidth=1.3,
                     edgecolor='#555555', facecolor=color))
        ax.text(x + width / 2, 0.56, text, ha='center', va='center', fontsize=9)
        if index < len(boxes) - 1:
            ax.annotate('', xy=(x + width + gap, 0.56), xytext=(x + width, 0.56),
                        arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.6))
        x += width + gap
    ax.text(0.5, 0.10, 'Bramki zależą od obrazu: ta sama sieć, inny obraz → inne wagi kanałów.',
            ha='center', fontsize=11, style='italic', color='#333333')
    ax.set_title('Czym jest blok SE (Squeeze-and-Excitation)?', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, 'fig_se_schemat.png')


def fig_e006():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    labels = ['rho(s_post, szkoda)', 'rho(g_pert, szkoda)', 'czastkowa g_pert | s_post']
    values = [-0.155, 0.044, 0.047]
    colors = ['#1f77b4', '#d62728', '#7f7f7f']
    bars = ax.barh(labels, values, color=colors)
    for bar, value in zip(bars, values):
        offset = 0.012 if value >= 0 else -0.012
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f'{value:+.3f}',
                va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-0.25, 0.22)
    ax.set_title('Korelacje rangowe (150 par, Tiny 96)')
    ax.grid(axis='x', alpha=0.3)
    ax = axes[1]
    ax.barh(['Delta rho = rho(g_pert) - rho(s_post)'], [0.199],
            xerr=[[0.199 - (-0.0203)], [0.4110 - 0.199]], color='#d62728',
            error_kw=dict(ecolor='black', capsize=8))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-0.15, 0.5)
    ax.set_title('Test glowny (95% CI) - dolna granica < 0')
    ax.grid(axis='x', alpha=0.3)
    save(fig, 'fig_bramki_se.png')


def fig_e008_stage1():
    stage1 = json.loads(E008_S1.read_text())['modes']
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ticks = ['BASE', '10_PRE', '10_POST', '10_RANDOM', '30_PRE', '30_POST', '30_RANDOM']
    values = [100 * stage1[t]['top1'] for t in ticks]
    palette = ['#444444', '#1f77b4', '#d62728', '#7f7f7f', '#1f77b4', '#d62728', '#7f7f7f']
    bars = ax.bar(range(len(ticks)), values, color=palette, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f'{value:.1f}',
                ha='center', fontsize=9)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=30, ha='right')
    ax.set_title('Tiny ImageNet 224: redukcja 10% i 30%, bez naprawy')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(0, 68)
    ax.grid(axis='y', alpha=0.3)
    save(fig, 'fig_kompresja_224_etap1.png')


def fig_e008_stage2():
    metrics = json.loads(E008_S2.read_text())['variants']
    names = ['BASE', 'PRE', 'POST', 'RANDOM']
    values = [100 * metrics[name]['last_row']['dev_top1'] for name in names]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(names, values, color=[COLORS[n] for n in names], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f'{value:.2f}',
                ha='center', fontsize=10)
    ax.set_title('E008: 30% redukcji po rownym dostrajaniu (60 epok)')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(min(values) - 2, max(values) + 2)
    ax.grid(axis='y', alpha=0.3)
    summary_path = E008_RUN / 'summary.json'
    if summary_path.exists():
        primary = json.loads(summary_path.read_text())['primary']
        ax.annotate('POST - PRE = %+.3f, CI [%.3f; %.3f]' % (
            primary['delta_top1'], primary['ci95_top1'][0], primary['ci95_top1'][1]),
            xy=(0.5, 0.03), xycoords='axes fraction', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', fc='#eef3fb', ec='#0b3d91'))
    save(fig, 'fig_etap2_wyniki.png')


def fig_e009_curves(state=None):
    state = state or load_e009()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in VARIANTS:
        v = state['variants'][name]
        if not v['rows']:
            continue
        epochs = [r['epoch'] for r in v['rows']]
        curve = [100 * r['dev_top1'] for r in v['rows']]
        label = f"{name}: {curve[-1]:.2f}% ({v['epoch']}/{v['total']})"
        style = '-' if v['epoch'] >= v['total'] else '--'
        ax.plot(epochs, curve, style, color=COLORS[name], label=label)
    ax.axvline(60, color='#cccccc', linestyle=':', linewidth=1.2)
    ymin, ymax = ax.get_ylim()
    ax.text(60.7, ymax - 0.4, 'budzet E008 (60 ep.)', fontsize=8, color='#777777', va='top')
    done = state['finished']
    title = ('E009: naprawa 30% redukcji, 120 epok - dev top-1'
             if done else
             f"E009 w toku (stan {state['now_utc']}): naprawa 120 epok - dev top-1")
    ax.set_title(title)
    ax.set_xlabel('epoka')
    ax.set_ylabel('dev top-1 [%]')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, 'fig_e009_krzywe.png')


def fig_e009_results(state=None):
    state = state or load_e009()
    e8 = json.loads(E008_S2.read_text())['variants']
    e8_values = {n: 100 * e8[n]['last_row']['dev_top1'] for n in VARIANTS}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.38
    xs = range(len(VARIANTS))
    for i, name in enumerate(VARIANTS):
        v = state['variants'][name]
        ax.bar(i - width / 2, e8_values[name], width, color='#d9d9d9',
               edgecolor='#999999')
        if v['top1'] is None:
            continue
        partial = v['epoch'] < v['total']
        ax.bar(i + width / 2, 100 * v['top1'], width, color=COLORS[name],
               hatch='//' if partial else None,
               edgecolor='white' if partial else None)
        value = 100 * v['top1']
        note = f"{value:.2f}%" if not partial else f"{value:.2f}%\n(ep. {v['epoch']}/{v['total']})"
        ax.text(i + width / 2, value + 0.08, note, ha='center', fontsize=9)
    for i, name in enumerate(VARIANTS):
        ax.text(i - width / 2, e8_values[name] + 0.08, f'{e8_values[name]:.2f}',
                ha='center', fontsize=8, color='#666666')
    title = '30% redukcji, rowna naprawa: 60 epok (E008) vs 120 epok (E009)'
    if not state['finished']:
        post = state['variants']['POST']
        status = (f"E009 w toku (stan {state['now_utc']})"
                  if post['top1'] is None or post['epoch'] >= post['total'] else
                  f"E009 w toku (stan {state['now_utc']}): POST ep. {post['epoch']}/{post['total']}")
        title += '\n' + status
    ax.set_title(title, fontsize=11)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel('dev top-1 [%]')
    ax.set_xlabel('jasne slupki = E008 (60 ep.) · kolorowe = E009 (120 ep.)'
                  + ('' if state['finished'] else '; // = w toku'), fontsize=9)
    ax.set_ylim(56, 61.5)
    ax.grid(axis='y', alpha=0.3)
    if state['summary'] is not None:
        primary = state['summary']['primary']
        ax.annotate('POST - PRE = %+.3f, CI [%.3f; %.3f]' % (
            primary['delta_top1'], primary['ci95_top1'][0], primary['ci95_top1'][1]),
            xy=(0.5, 0.03), xycoords='axes fraction', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', fc='#eef3fb', ec='#0b3d91'))
    save(fig, 'fig_e009_wyniki.png')


def generate_e009_figures(state=None):
    state = state or load_e009()
    fig_e009_curves(state)
    fig_e009_results(state)


def generate_all():
    for name in list(OUT.glob('fig_*.png')):
        name.unlink()
    fig_e001_curve()
    fig_e005_e007_curves()
    fig_e003_raw()
    fig_repair()
    fig_se_schemat()
    fig_e006()
    fig_e008_stage1()
    fig_e008_stage2()
    generate_e009_figures()
    print('PODSUMOWANIE_FIGURES=done')


if __name__ == '__main__':
    generate_all()
