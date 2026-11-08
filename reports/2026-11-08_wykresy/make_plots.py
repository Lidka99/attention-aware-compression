"""Wykresy przekrojowe E035-E037; wejście analysis.json, wyjście PNG. Bez treningu."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAIRS = ('resnet_cifar10', 'resnet_cifar100', 'resnet_tiny', 'preact_cifar10', 'preact_cifar100')
LABELS = {'resnet_cifar10': 'ResNet/CIFAR-10', 'resnet_cifar100': 'ResNet/CIFAR-100',
          'resnet_tiny': 'ResNet/Tiny', 'preact_cifar10': 'PreAct/CIFAR-10',
          'preact_cifar100': 'PreAct/CIFAR-100'}
TAGS = ('b30', 'b50', 'b70')
X = [30, 50, 70]
COL = {'resnet_cifar10': '#1f77b4', 'resnet_cifar100': '#d62728', 'resnet_tiny': '#2ca02c',
       'preact_cifar10': '#9467bd', 'preact_cifar100': '#ff7f0e'}
read = lambda p: json.loads(Path(p).read_text())


def mean(vals):
    return statistics.mean(vals)


def plot_e035():
    d = read(ROOT / 'runs/E035/20261107T183831Z_8b0b14e0/analysis/analysis.json')
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=False)
    for ax, pair in zip(axes, PAIRS):
        for phase, style in (('dev', '--'), ('test', '-')):
            for profile, marker in (('U', 'o'), ('T', 's')):
                ys = []
                for tag in TAGS:
                    vals = [r['top1'] * 100 for r in d['rows'] if r['pair'] == pair and r['tag'] == tag
                            and r['phase'] == phase and r['profile'] == profile and r['mode'] == 'CMA']
                    ys.append(mean(vals))
                ax.plot(X, ys, style, marker=marker, color=COL[pair], alpha=.9,
                        label=f'{profile} {phase}')
        ax.set_title(LABELS[pair], fontsize=10)
        ax.set_xlabel('Redukcja parametrów [%]')
        ax.set_xticks(X)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Top-1 [%]')
    axes[-1].legend(fontsize=7, loc='best')
    fig.suptitle('E035: dev vs niezależna ocena (testy CIFAR, oficjalny val Tiny; CMA)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / 'e035_quality_dev_test.png', dpi=160)
    plt.close(fig)


def plot_e036():
    d = read(ROOT / 'runs/E036/20261107T231027Z_27bac186/analysis/analysis.json')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for pair in PAIRS:
        mac, thr = [], []
        for tag in TAGS:
            row = [r for r in d['averages'] if r['pair'] == pair and r['tag'] == tag
                   and r['name'] == 'T/U' and r['batch'] == 128][0]
            mac.append(row['mac_ratio'])
            thr.append(row['throughput_ratio'])
        axes[0].plot(X, mac, 'o-', color=COL[pair], label=LABELS[pair])
        axes[1].plot(X, thr, 'o-', color=COL[pair], label=LABELS[pair])
    axes[0].axhline(1, color='k', lw=1, ls=':')
    axes[0].set_title('MAC: TILT / UNIFORM')
    axes[0].set_ylabel('iloraz MAC')
    axes[1].axhline(1, color='k', lw=1, ls=':')
    axes[1].set_title('Przepustowość batch128: TILT / UNIFORM')
    axes[1].set_ylabel('iloraz przepustowości')
    for ax in axes:
        ax.set_xlabel('Redukcja parametrów [%]')
        ax.set_xticks(X)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle('E036: ten sam budżet parametrów, inny koszt obliczeń', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / 'e036_cost_ratios.png', dpi=160)
    plt.close(fig)


def plot_e037():
    d = read(ROOT / 'runs/E037/20261108T031759Z_855c0b33/analysis/analysis.json')
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=False)
    for ax, pair in zip(axes, PAIRS):
        for name, marker, style in (('U', 'o', '--'), ('T', 's', '-'), ('Up', '^', '-')):
            ys = []
            for tag in TAGS:
                vals = [a['top1'] * 100 for a in d['averages'] if a['pair'] == pair and a['tag'] == tag
                        and a['mode'] == 'CMA' and a['name'] == name]
                ys.append(mean(vals))
            label = {'U': 'UNIFORM', 'T': 'TILT', 'Up': "UNIFORM MAC-matched"}[name]
            ax.plot(X, ys, style, marker=marker, label=label)
        ax.set_title(LABELS[pair], fontsize=10)
        ax.set_xlabel('Nominalna redukcja parametrów [%]')
        ax.set_xticks(X)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Top-1 dev [%]')
    axes[-1].legend(fontsize=7)
    fig.suptitle('E037: przy dopasowanym MAC UNIFORM dostaje więcej parametrów', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / 'e037_matched_mac.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    labels = []
    values = []
    colors = []
    for pair in PAIRS:
        for tag in TAGS:
            r = [x for x in d['allocation'] if x['pair'] == pair and x['tag'] == tag
                 and x['mode'] == 'CMA' and x['contrast'] == 'T-Up'][0]
            labels.append(f"{LABELS[pair]}\n{tag[1:]}%")
            values.append(r['mean'])
            colors.append(COL[pair])
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
    ax.set_ylabel('TILT − UNIFORM(MAC) [pp]')
    ax.set_title('E037: różnica przy dopasowanym MAC (CMA, dev)')
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / 'e037_contrast.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    plot_e035(); plot_e036(); plot_e037()
    print('PLOTS', OUT)
