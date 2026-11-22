#!/usr/bin/env python3
"""E035 — schemat procedury do szkicu artykułu (tylko CPU, bez danych i wag).

Co robi: rysuje jednostronicowy schemat przepływu: checkpoint → legalne grupy
kanałów → profile UNIFORM/TILT przy docelowym P → wspólne operatory FOLD/MAG2
(znane elementy) → BN reset+CMA bez gradientów → ocena z licznikami P i MAC.
Rozdziela fazy: FOLD = ocena niezależna E035 (test CIFAR / val Tiny),
MAG2 = kontrola dev E038 (nie test E035).

Dane wejściowe: wyłącznie metadane planów `E035/frozen.json` (wektor szerokości
b50 ResNet18/CIFAR-10 i liczby parametrów) oraz wartości MAC T/U raportowane
w E036. Nie czytamy obrazów, checkpointów, wag ani wyników per seed.

Wyjście: `schemat_procedury.pdf` i `schemat_procedury.png` wyłącznie w tym
katalogu. Pełny opis: `README.md`; kontekst: OPIS.md i WYNIKI.md E035, E036,
E038.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
E035 = OUT.parents[1]
ROOT = OUT.parents[3]
FROZEN = E035 / "frozen.json"

PAIR = "resnet_cifar10"
BUDGET = "b50"

INK = "#1f2933"
MUTED = "#5b6770"
GRID = "#c9d1d8"
U_C = "#3b6ea5"
T_C = "#c1631f"
FOLD_C = "#2f6b4f"
MAG2_C = "#6f7378"
KNOWN_BG = "#f2f4f6"
FOLD_BG = "#e8f2ec"
MAG2_BG = "#f4f4f4"
INFRA_BG = "#eef3f7"

GROUP_SHORT = [
    "conv1", "L1.0.c1", "L1.1.c1", "L2.0.c1", "L2.0.c2", "L2.1.c1",
    "L3.0.c1", "L3.0.c2", "L3.1.c1", "L4.0.c1", "L4.0.c2", "L4.1.c1",
]
STAGE_COLORS = ["#dfe9f3", "#e3f0e4", "#f7ecd9", "#ece3f2", "#f6dede"]
STAGE_OF = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def load_plan():
    data = json.loads(FROZEN.read_text())
    plan = data["plans"][PAIR][BUDGET]
    n = np.array([64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512])
    u = np.array(plan["U"], dtype=int)
    t = np.array(plan["T"], dtype=int)
    if u.shape != n.shape or t.shape != n.shape:
        raise ValueError("Nieoczekiwany kształt planu w frozen.json")
    return u, t, n, int(plan["params"]["U"]), int(plan["params"]["T"])


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.9, ls="solid", r=0.9, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.0, ls="solid", z=3, style="-|>", ms=7):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=ms,
            color=color, linewidth=lw, linestyle=ls,
            shrinkA=0, shrinkB=0, zorder=z,
        )
    )


def text(ax, x, y, s, size=7.2, color=INK, weight="normal", ha="center", va="center",
         style="normal", z=5, family="DejaVu Sans", rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, family=family, rotation=rotation,
            linespacing=1.25)


def fmt_int(value):
    return f"{value:,}".replace(",", " ")


def fmt_pl(value, digits=5):
    return f"{value:.{digits}f}".replace(".", ",")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "axes.axisbelow": True,
    })


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=230, bbox_inches="tight", facecolor="white",
                metadata={"Software": None})
    fig.savefig(
        pdf, bbox_inches="tight", facecolor="white",
        metadata={"Creator": "E035 schemat procedury", "Producer": "Matplotlib",
                  "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return png, pdf


def panel_a(ax, widths):
    text(ax, 2.2, 90.8, "A. Checkpoint i legalne grupy kanałów", size=11.2,
         weight="bold", ha="left")
    text(ax, 2.2, 87.7,
         "Decyzja dotyczy grupy kanałów, a nie dowolnej pojedynczej warstwy.",
         size=7.4, color=MUTED, ha="left", style="italic")

    box(ax, 3.0, 80.8, 42.8, 4.6, fc=KNOWN_BG)
    text(ax, 24.4, 83.1,
         "Checkpoint po treningu źródłowym (4 seedy 42–45; od losowej inicjalizacji)\n"
         "ResNet18 / PreActResNet18; bez publicznych wag",
         size=7.0)

    # --- schemat bloku residualnego (oś producenta → konsumenci) ---
    y0, h = 72.6, 4.6
    x_in = 4.2
    chain = [
        (7.2, 6.0, "conv1\n(out×in)"),
        (14.4, 6.4, "BN+ReLU\n(oś producenta)"),
        (22.1, 6.0, "conv2\n(out×in)"),
        (29.4, 5.0, "BN\naffine"),
    ]
    for (cx, cw, label) in chain:
        box(ax, cx, y0, cw, h, fc="white")
        text(ax, cx + cw / 2, y0 + h / 2, label, size=6.1)

    circle_c = (38.1, y0 + h / 2)
    ax.add_patch(plt.Circle(circle_c, 1.15, facecolor="white", edgecolor=INK,
                            linewidth=0.9, zorder=4))
    text(ax, circle_c[0], circle_c[1] - 0.05, "+", size=9)

    arrow(ax, x_in, y0 + h / 2, 7.1, y0 + h / 2)
    arrow(ax, 13.2, y0 + h / 2, 14.3, y0 + h / 2)
    arrow(ax, 20.8, y0 + h / 2, 22.0, y0 + h / 2)
    arrow(ax, 28.1, y0 + h / 2, 29.3, y0 + h / 2)
    arrow(ax, 34.4, y0 + h / 2, 36.9, y0 + h / 2)
    arrow(ax, 39.3, y0 + h / 2, 43.2, y0 + h / 2)
    text(ax, 44.2, y0 + h / 2, "wyjście", size=6.4, ha="left")

    # skip z opcjonalnym conv 1×1 (downsampling)
    skip_y = 68.9
    ax.plot([x_in, x_in, circle_c[0], circle_c[0]],
            [y0, skip_y, skip_y, y0 + h / 2 - 1.18], color=INK, linewidth=1.0, zorder=3)
    box(ax, 22.4, skip_y - 1.7, 8.0, 3.4, fc="white", lw=0.8)
    text(ax, 26.4, skip_y, "conv 1×1\n(skrót)", size=5.8)

    text(ax, 2.8, 76.5, "wejście", size=5.8, ha="left", color=MUTED)
    text(ax, 14.4, 78.0, "oś producenta: kanały po ReLU", size=5.9, ha="left",
         color=MUTED, style="italic")
    text(ax, 8.6, 71.4, "konsumenci: fan-in conv2, affine BN, conv w skrócie",
         size=5.9, ha="left", color=MUTED, style="italic")

    text(ax, 3.0, 65.6,
         "k_g jest wspólne dla całego łańcucha: producent, fan-in kolejnych wag,\n"
         "afine BN i conv skrótu muszą zostać przebudowane razem.",
         size=6.7, ha="left")
    text(ax, 3.0, 62.9,
         "Przykład: grupa L3.0.c1 = wyjście layer3.0.conv1 → wejście layer3.0.conv2.",
         size=6.2, ha="left", color=MUTED, style="italic")

    # --- grupy w kolejności głębokości ---
    text(ax, 3.0, 60.2,
         "Grupy w kolejności głębokości (g = 0…G−1): ResNet18 = 12, PreActResNet18 = 13",
         size=6.8, ha="left", weight="bold")
    text(ax, 3.0, 58.5,
         "n = szerokość oryginalna; nazwy dla ResNet18 — PreActResNet18 dodaje grupę L1.0.c2.",
         size=6.2, ha="left", color=MUTED)

    chip_w, chip_h, gap = 6.25, 4.1, 0.75
    for row in range(2):
        for col in range(6):
            idx = row * 6 + col
            cx = 3.0 + col * (chip_w + gap)
            cy = 53.0 - row * (chip_h + 0.7)
            box(ax, cx, cy, chip_w, chip_h, fc=STAGE_COLORS[STAGE_OF[idx]], lw=0.7)
            text(ax, cx + chip_w / 2, cy + chip_h * 0.68, GROUP_SHORT[idx], size=5.6)
            text(ax, cx + chip_w / 2, cy + chip_h * 0.28, f"n={widths[idx]}", size=5.4)

    text(ax, 3.0, 45.6,
         "Kolejność grup jest osią profilu TILT; nie jest to lista niezależnych warstw.",
         size=6.2, ha="left", color=MUTED, style="italic")


def panel_b(ax, u, t, n, p_u, p_t):
    text(ax, 50.0, 90.8, "B. Profile UNIFORM i TILT przy docelowym P", size=11.2,
         weight="bold", ha="left")
    text(ax, 50.0, 87.7,
         "Przykład: b50 = 50% redukcji P (ResNet18/CIFAR-10, frozen.json); procent dotyczy P, nie kanałów.",
         size=7.0, color=MUTED, ha="left", style="italic")

    plot_ax = ax.figure.add_axes([0.505, 0.568, 0.462, 0.242])
    keep_u = 100.0 * u / n
    keep_t = 100.0 * t / n
    x = np.arange(len(n))
    plot_ax.fill_between(x, keep_u, keep_t, color=T_C, alpha=0.09, linewidth=0)
    plot_ax.plot(x, keep_u, color=U_C, lw=1.9, marker="o", ms=3.0,
                 label="UNIFORM (U)", zorder=4)
    plot_ax.plot(x, keep_t, color=T_C, lw=1.9, marker="s", ms=2.8,
                 label="TILT (α = 0,30)", zorder=4)
    plot_ax.set_xlim(-0.5, len(n) - 0.5)
    plot_ax.set_ylim(56, 94)
    plot_ax.set_ylabel("kanały zachowane k/n [%]", fontsize=8)
    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(GROUP_SHORT, rotation=0, ha="center", fontsize=5.3)
    plot_ax.tick_params(axis="y", labelsize=7)
    plot_ax.grid(axis="x", visible=False)
    plot_ax.legend(loc="upper right", frameon=True, fontsize=7.2,
                   framealpha=0.95, borderpad=0.4)

    plot_ax.annotate("ochrona wczesnych grup", xy=(1.0, keep_t[1]), xytext=(0.3, 92.0),
                     fontsize=6.4, color=MUTED, arrowprops=dict(arrowstyle="->",
                     color=MUTED, lw=0.7))
    plot_ax.annotate("mocniejsza kompresja późnych", xy=(10.0, keep_t[10]),
                     xytext=(5.4, 60.0), fontsize=6.4, color=MUTED,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    for side in ("top", "right"):
        plot_ax.spines[side].set_visible(False)

    # liczniki P i MAC
    box(ax, 50.0, 44.2, 47.4, 8.2, fc="#f7f9fa", ec=MUTED, lw=0.8)
    text(ax, 51.0, 50.7, "Liczniki planu b50", size=7.3, ha="left", weight="bold")
    text(ax, 51.0, 48.5,
         f"P:  U = {fmt_int(p_u)}   T = {fmt_int(p_t)}   T/U = {fmt_pl(p_t / p_u)}",
         size=7.2, ha="left", family="DejaVu Sans Mono")
    text(ax, 51.0, 46.6,
         "MAC:  T/U = 1,364 (E036)   → ten sam P, inny MAC",
         size=7.2, ha="left", family="DejaVu Sans Mono")
    text(ax, 51.0, 45.0,
         "P i MAC są liczone osobno; równy budżet P nie wymusza równego MAC.",
         size=6.7, ha="left", color=MUTED, style="italic")


def stage(ax, x, w, header, lines, fc="white", ec=INK, ls="solid", tag=None,
          tag_fc=KNOWN_BG):
    y, h = 12.7, 19.7
    box(ax, x, y, w, h, fc=fc, ec=ec, ls=ls, lw=1.0)
    text(ax, x + 0.9, y + h - 2.1, header, size=7.9, ha="left", weight="bold",
         color=ec if ec != INK else INK)
    yy = y + h - 4.9
    for line, weight, color, size in lines:
        text(ax, x + 0.9, yy, line, size=size, ha="left", weight=weight, color=color)
        yy -= 2.2
    if tag:
        box(ax, x + 0.7, y + 0.6, w - 1.4, 2.5, fc=tag_fc, ec=MUTED, lw=0.6, r=0.5)
        text(ax, x + w / 2, y + 1.85, tag, size=6.2, color=MUTED)
    return y, h


def pipeline(ax):
    text(ax, 1.6, 37.2, "Przebieg procedury i rozdzielenie faz", size=11.2,
         weight="bold", ha="left")
    text(ax, 1.6, 34.4,
         "FOLD i MAG2 to znane operatory; wkładem jest profil alokacji. Ocena niezależna E035 dotyczy wyłącznie FOLD.",
         size=7.2, color=MUTED, ha="left", style="italic")

    ymid = 22.55
    stage(ax, 1.6, 12.4, "1. Checkpoint", [
        ("wagi po treningu", "normal", INK, 6.6),
        ("źródłowym (4 seedy)", "normal", INK, 6.6),
        ("inicjalizacja losowa", "normal", MUTED, 6.2),
        ("bez publicznych wag", "normal", MUTED, 6.2),
        ("bez fine-tuningu", "normal", MUTED, 6.2),
    ], tag="źródło, nie wkład")

    stage(ax, 15.6, 17.4, "2. Profil alokacji k(P)", [
        ("UNIFORM: wspólne ratio", "normal", INK, 6.5),
        ("TILT: r_g = clip(b + αp_g,", "normal", INK, 6.5),
        ("0,10; 0,90), r_g = redukcja", "normal", INK, 6.5),
        ("b do P; p_g = 2g/(G−1) − 1", "normal", MUTED, 5.8),
        ("licznik P: parametry sieci", "normal", MUTED, 6.2),
        ("(bez buforów BN)", "normal", MUTED, 6.2),
    ], fc="#fbf7f1", ec=T_C, tag="nasza adaptacja znanej rodziny", tag_fc="#fbf0e6")

    # Stage 3: dwa operatory na tej samej geometrii
    x3, w3 = 34.6, 27.6
    y, h = 12.7, 19.7
    box(ax, x3, y, w3, h, fc="white", ec=INK, lw=1.0)
    text(ax, x3 + 0.9, y + h - 2.1, "3. Operator: wspólna geometria k", size=7.9,
         ha="left", weight="bold")
    text(ax, x3 + 0.9, y + h - 4.5,
         "Dla danego k: identyczne szerokości, P i MAC.", size=6.4, ha="left", color=MUTED)

    box(ax, x3 + 0.9, 21.4, w3 - 1.8, 5.6, fc=FOLD_BG, ec=FOLD_C, lw=1.0)
    text(ax, x3 + 1.8, 25.7, "FOLD — gęste scalenie w grupie", size=7.4, ha="left",
         weight="bold", color=FOLD_C)
    text(ax, x3 + 1.8, 23.8, "operator upstream (znany element)", size=6.4, ha="left")
    text(ax, x3 + 1.8, 22.3, "→ ścieżka oceny niezależnej E035", size=6.4, ha="left",
         color=FOLD_C, weight="bold")

    box(ax, x3 + 0.9, 14.6, w3 - 1.8, 5.6, fc=MAG2_BG, ec=MAG2_C, lw=1.0, ls=(0, (4, 2)))
    text(ax, x3 + 1.8, 18.9, "MAG2 — wycięcie kanałów o małej |w|", size=7.4, ha="left",
         weight="bold", color="#4d5156")
    text(ax, x3 + 1.8, 17.0, "baseline (znany element)", size=6.4, ha="left")
    text(ax, x3 + 1.8, 15.5, "→ tylko kontrola dev E038", size=6.4, ha="left",
         color="#4d5156", weight="bold")

    stage(ax, 64.4, 15.6, "4. BN bez gradientów", [
        ("reset BN + CMA", "bold", INK, 6.8),
        ("A ⊂ train: 4096 obrazów,", "normal", INK, 6.2),
        ("crop/flip, batch 128", "normal", INK, 6.2),
        ("zero aktualizacji wag", "bold", FOLD_C, 6.4),
        ("osobne kopie dla FOLD/MAG2", "normal", MUTED, 6.2),
        ("CMA = cumulative moving avg.", "normal", MUTED, 6.0),
    ], tag="standardowa procedura")

    stage(ax, 82.2, 16.2, "5. Ocena i liczniki", [
        ("E035: FOLD → oficjalny test", "normal", FOLD_C, 6.4),
        ("CIFAR-10/100 i val Tiny", "normal", FOLD_C, 6.4),
        ("E038: MAG2 → dev", "normal", "#4d5156", 6.4),
        ("(NIE test E035)", "bold", "#4d5156", 6.4),
        ("top1, NLL; 4 seedy", "normal", INK, 6.2),
        ("liczniki P oraz MAC osobno", "normal", INK, 6.2),
    ], tag="P i MAC to różne liczniki")

    for (sx, sw), (ex, _ew) in zip([(1.6, 12.4), (15.6, 17.4), (34.6, 27.6), (64.4, 15.6)],
                                   [(15.6, 0), (34.6, 0), (64.4, 0), (82.2, 0)]):
        arrow(ax, sx + sw + 0.15, ymid, ex - 0.25, ymid, lw=1.2, ms=9)

    # legenda faz / zakres
    box(ax, 1.6, 3.6, 62.0, 7.0, fc="#fbfcfc", ec=GRID, lw=0.7)
    text(ax, 2.6, 8.9, "Legenda faz", size=6.9, ha="left", weight="bold")
    ax.plot([2.8, 7.4], [6.9, 6.9], color=FOLD_C, lw=1.8, solid_capstyle="butt")
    text(ax, 7.9, 6.9, "ocena niezależna E035: tylko FOLD, test CIFAR / val Tiny",
         size=6.5, ha="left")
    ax.plot([2.8, 7.4], [5.0, 5.0], color="#4d5156", lw=1.8, ls=(0, (3, 2)))
    text(ax, 7.9, 5.0, "kontrola dev E038: MAG2, ten sam dev — bez testu E035",
         size=6.5, ha="left")

    text(ax, 65.0, 7.8,
         "Zakres twierdzenia: budżet parametrów, bez\n"
         "fine-tuningu; nie twierdzimy „compute-matched”.",
         size=6.6, ha="left", color=MUTED)
    text(ax, 65.0, 4.6,
         "FOLD i MAG2 są znane; wynik E035 nie jest\n"
         "oceną MAG2 ani nowego operatora.",
         size=6.6, ha="left", color=MUTED)


def main():
    if OUT.name != "schemat" or OUT.parent.name != "figures" or "E035" not in E035.name:
        raise RuntimeError(f"Nieoczekiwany katalog wyjściowy: {OUT}")
    if not FROZEN.exists():
        raise FileNotFoundError(FROZEN)

    style()
    u, t, n, p_u, p_t = load_plan()

    fig = plt.figure(figsize=(15.5, 9.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor("white")

    text(ax, 1.6, 97.1,
         "Schemat procedury: alokacja budżetu parametrów w foldingu bez fine-tuningu",
         size=14.0, weight="bold", ha="left")
    text(ax, 1.6, 94.2,
         "E035 (ocena niezależna, FOLD) i E038 (kontrola dev, MAG2) jako osobne fazy; "
         "liczniki P i MAC liczone oddzielnie.",
         size=7.6, color=MUTED, ha="left")

    ax.plot([1.6, 98.4], [92.9, 92.9], color=GRID, lw=0.8)
    ax.plot([1.6, 98.4], [39.5, 39.5], color=GRID, lw=0.8)

    panel_a(ax, n)
    panel_b(ax, u, t, n, p_u, p_t)
    pipeline(ax)

    text(ax, 98.4, 1.6,
         "Źródła metadanych: E035/frozen.json (plany b50) i raport MAC E036; "
         "schemat opisowy, bez obliczeń na modelach.",
         size=6.2, color=MUTED, ha="right")

    png, pdf = save(fig, "schemat_procedury")
    digest = hashlib.sha256((OUT / "generuj_schemat.py").read_bytes()).hexdigest()
    print(f"OK: {png.name} {png.stat().st_size} B; {pdf.name} {pdf.stat().st_size} B")
    print(f"skrypt sha256: {digest}")
    print(f"frozen.json sha256: {hashlib.sha256(FROZEN.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
