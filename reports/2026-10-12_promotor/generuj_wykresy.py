"""Prezentacja dla promotora (2026-10-12): generuje wykresy z zamrozonych wynikow.
Wejscia: runs/E001, runs/E005, runs/E007, runs/E008 (metryki jsonl/json) - tylko odczyt.
Wyjscia: wykresy/*.png w tym folderze. CPU-only, bez GPU i bez treningu.
"""
import json
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
E008_BASE = E008_RUN / 'stage2/metrics_BASE.jsonl'

COLORS = {'BASE': '#444444', 'PRE': '#1f77b4', 'POST': '#d62728', 'RANDOM': '#7f7f7f'}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print('saved', OUT / name)


def fig_e001_curve():
    e1 = rows(E001)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    epochs = [r['epoch'] for r in e1]
    ax.plot(epochs, [100 * r['dev_top1'] for r in e1], color='#1f77b4', label='dev top-1')
    ax.plot(epochs, [100 * r['train_correct'] / r['train_count'] for r in e1],
            color='#999999', label='train top-1')
    ax.annotate('dev 66,84% (best, ep. 177)', xy=(177, 66.84), xytext=(100, 82),
                arrowprops=dict(arrowstyle='->', color='#333333'), fontsize=9)
    ax.set_title('E001: CIFAR-100, 96x96, 200 epok')
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
    for data, label, color in ((e5, 'E005: 96x96 -> dev 51,53%', '#1f77b4'),
                               (e7, 'E007: 224x224 -> dev 58,49%', '#d62728')):
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
    ax.set_title('E003: CIFAR-100, redukcja 10%, bez naprawy')
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
    ax.set_title('E004: CIFAR-100 po rownym dostrajaniu (120 epok)')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(64, 69)
    ax.grid(axis='y', alpha=0.3)
    ax.annotate('RANDOM dorownal: POST-PRE +0,36 pp,\nCI [-0,30; +1,02]',
                xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', fc='#fff7e6', ec='#cc9900'))
    save(fig, 'fig_dostrajanie_cifar.png')


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
    ax.set_title('E006: korelacje rangowe (150 par, Tiny 96)')
    ax.grid(axis='x', alpha=0.3)
    ax = axes[1]
    ax.barh(['Delta rho = rho(g_pert) - rho(s_post)'], [0.199],
            xerr=[[0.199 - (-0.0203)], [0.4110 - 0.199]], color='#d62728',
            error_kw=dict(ecolor='black', capsize=8))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-0.15, 0.5)
    ax.set_title('E006: test glowny (95% CI) - dolna granica < 0')
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
    ax.set_title('E008 etap 1: Tiny ImageNet 224, 10% i 30%, bez naprawy')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(0, 68)
    ax.grid(axis='y', alpha=0.3)
    save(fig, 'fig_kompresja_224_etap1.png')


def fig_e008_stage2_base():
    base = rows(E008_BASE)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    epochs = [r['epoch'] for r in base]
    ax.plot(epochs, [100 * r['dev_top1'] for r in base], marker='o', color='#444444',
            label='BASE (dostrajanie)')
    ax.axhline(58.49, color='#999999', linestyle='--', label='E007 (przed dostrajaniem): 58,49%')
    ax.set_title(f'E008 etap 2 w toku: BASE, epoka {epochs[-1]}/60 (12.10.2026)')
    ax.set_xlabel('epoka')
    ax.set_ylabel('dev top-1 [%]')
    ax.set_ylim(50, 62)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, 'fig_etap2_w_toku.png')


if __name__ == '__main__':
    for name in list(OUT.glob('fig_*.png')):
        name.unlink()
    fig_e001_curve()
    fig_e005_e007_curves()
    fig_e003_raw()
    fig_repair()
    fig_e006()
    fig_e008_stage1()
    fig_e008_stage2_base()
    print('E008_PROMOTOR_FIGURES=done')
