#!/usr/bin/env python3
"""E035: niezależna ocena CMA TILT vs UNIFORM (CPU only, deterministycznie).

Wejście: analysis.json oraz surowe summary.json dev/test z runu
20261107T183831Z_8b0b14e0. Tiny w fazie test jest oficjalnym val (cache_val),
dlatego na wykresach opisano go jako "oficjalny val", nie "test".
Wyjście: PNG + PDF + punkty.csv + sources.json + weryfikacja.json w figures/.
"""
import csv
import hashlib
import json
import os
import statistics as st
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/E035/20261107T183831Z_8b0b14e0"
ANALYSIS = RUN / "analysis/analysis.json"
TEST_DATA = RUN / "test_data.json"

SEEDS = (42, 43, 44, 45)
PAIRS = (
    "resnet_cifar10",
    "resnet_cifar100",
    "resnet_tiny",
    "preact_cifar10",
    "preact_cifar100",
)
TAGS = ("b30", "b50", "b70")
X = (30, 50, 70)
TAG_X = dict(zip(TAGS, X))
PHASES = ("dev", "test")
LABELS = {
    "resnet_cifar10": "ResNet/CIFAR-10",
    "resnet_cifar100": "ResNet/CIFAR-100",
    "resnet_tiny": "ResNet/Tiny",
    "preact_cifar10": "PreAct/CIFAR-10",
    "preact_cifar100": "PreAct/CIFAR-100",
}
SHORT = {
    "resnet_cifar10": "R/C10",
    "resnet_cifar100": "R/C100",
    "resnet_tiny": "R/Tiny",
    "preact_cifar10": "P/C10",
    "preact_cifar100": "P/C100",
}
COLORS = {
    "resnet_cifar10": "#0072B2",
    "resnet_cifar100": "#D55E00",
    "resnet_tiny": "#009E73",
    "preact_cifar10": "#CC79A7",
    "preact_cifar100": "#E69F00",
}
SEED_MARKERS = {42: "o", 43: "s", 44: "^", 45: "D"}
OFFSETS = {42: -0.72, 43: -0.24, 44: 0.24, 45: 0.72}

CSV_FIELDS = (
    "figura",
    "panel",
    "seria",
    "seed",
    "redukcja",
    "wartosc",
    "jednostka",
    "srednia",
    "sd",
    "typ",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_label(pair, phase):
    if pair == "resnet_tiny" and phase == "test":
        return "oficjalny val"
    return "dev" if phase == "dev" else "test"


def panel_label(pair, phase):
    return f"{LABELS[pair]}\n({phase_label(pair, phase)})"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": None},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": f"E035 {stem}",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def add_row(rows, figura, panel, seria, seed, reduction, value, unit, mean, sd, typ):
    rows.append(
        {
            "figura": figura,
            "panel": panel,
            "seria": seria,
            "seed": "" if seed is None else seed,
            "redukcja": reduction,
            "wartosc": f"{value:.10g}",
            "jednostka": unit,
            "srednia": f"{mean:.10g}",
            "sd": f"{sd:.10g}",
            "typ": typ,
        }
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT.name != "figures":
        raise RuntimeError(f"Nieoczekiwany katalog wyjściowy: {OUT}")

    summary_paths = [
        RUN / f"{phase}/seed_{seed}/{pair}/{tag}/summary.json"
        for phase in PHASES
        for seed in SEEDS
        for pair in PAIRS
        for tag in TAGS
    ]
    sources = [ANALYSIS, TEST_DATA, *summary_paths]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path.resolve()): sha256(path) for path in sources}

    # Surowe top-1 CMA dla U i T, per seed, w obu fazach.
    cells = {}
    for phase in PHASES:
        for pair in PAIRS:
            for tag in TAGS:
                per_seed = []
                for seed in SEEDS:
                    summary = read(RUN / f"{phase}/seed_{seed}/{pair}/{tag}/summary.json")
                    metrics = summary["metrics"]
                    u = metrics["U"]["CMA"]["top1"] * 100.0
                    t = metrics["T"]["CMA"]["top1"] * 100.0
                    per_seed.append({"seed": seed, "U": u, "T": t, "delta": t - u})
                cells[(phase, pair, tag)] = per_seed

    analysis = read(ANALYSIS)
    if len(analysis["rows"]) != 720 or len(analysis["contrasts"]) != 90:
        raise RuntimeError("Nieoczekiwana liczebność analysis.json")

    # Kontrola: surowe summary muszą odtworzyć analysis.json contrast (CMA).
    max_mean_diff = 0.0
    max_sd_diff = 0.0
    max_abs_diff = 0.0
    contrasts = {
        (row["phase"], row["pair"], row["tag"]): row
        for row in analysis["contrasts"]
        if row["mode"] == "CMA"
    }
    if len(contrasts) != 30:
        raise RuntimeError(f"Oczekiwano 30 kontrastów CMA, jest {len(contrasts)}")
    for phase in PHASES:
        for pair in PAIRS:
            for tag in TAGS:
                per_seed = cells[(phase, pair, tag)]
                deltas = [row["delta"] for row in per_seed]
                mean = st.mean(deltas)
                sd = st.stdev(deltas)
                ref = contrasts[(phase, pair, tag)]
                max_mean_diff = max(max_mean_diff, abs(mean - ref["mean"]))
                max_sd_diff = max(max_sd_diff, abs(sd - ref["sd"]))
                if abs(mean - ref["mean"]) > 1e-9 or abs(sd - ref["sd"]) > 1e-9:
                    raise RuntimeError(f"Niezgodność kontrastu {phase}/{pair}/{tag}")
                for got, want in zip(deltas, ref["per_seed"]):
                    max_abs_diff = max(max_abs_diff, abs(got - want))
                    if abs(got - want) > 1e-9:
                        raise RuntimeError(f"Niezgodność per_seed {phase}/{pair}/{tag}")
                for profile in ("U", "T"):
                    values = [row[profile] for row in per_seed]
                    abs_ref = ref["absolute"][profile]
                    max_abs_diff = max(
                        max_abs_diff,
                        abs(st.mean(values) - abs_ref["mean"]),
                        abs(st.stdev(values) - abs_ref["sd"]),
                    )
                    if abs(st.mean(values) - abs_ref["mean"]) > 1e-9:
                        raise RuntimeError(f"Niezgodność absolute {phase}/{pair}/{tag}/{profile}")
                    for got, want in zip(values, abs_ref["per_seed"]):
                        if abs(got - want) > 1e-9:
                            raise RuntimeError(f"Niezgodność absolute per_seed {phase}/{pair}/{tag}")

    test_data = read(TEST_DATA)
    tiny_val_confirmed = "data/tiny-imagenet/cache_val.npz" in test_data.get("files", {})
    if not tiny_val_confirmed:
        raise RuntimeError("Brak potwierdzenia, że Tiny używa oficjalnego cache_val")

    style()
    rows = []

    # Rysunek 1: niezależny top-1 CMA U i T.
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharey=False)
    axes_flat = axes.ravel()
    for index, pair in enumerate(PAIRS):
        ax = axes_flat[index]
        for profile, marker, linestyle, color, label in (
            ("U", "o", "--", "#0072B2", "UNIFORM (U)"),
            ("T", "s", "-", "#D55E00", "TILT (T)"),
        ):
            means = []
            sds = []
            for tag in TAGS:
                values = [row[profile] for row in cells[("test", pair, tag)]]
                means.append(st.mean(values))
                sds.append(st.stdev(values))
                add_row(
                    rows,
                    "e035_niezalezny_top1",
                    panel_label(pair, "test"),
                    label,
                    None,
                    TAG_X[tag],
                    means[-1],
                    "%",
                    means[-1],
                    sds[-1],
                    "mean",
                )
                for row in cells[("test", pair, tag)]:
                    add_row(
                        rows,
                        "e035_niezalezny_top1",
                        panel_label(pair, "test"),
                        label,
                        row["seed"],
                        TAG_X[tag],
                        row[profile],
                        "%",
                        means[-1],
                        sds[-1],
                        "seed",
                    )
            ax.errorbar(
                X,
                means,
                yerr=sds,
                fmt=marker + linestyle,
                color=color,
                lw=1.4,
                ms=5,
                capsize=3,
                label=label,
            )
        ax.set_title(panel_label(pair, "test"))
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.set_ylabel("Top-1 CMA [%]")
        ax.margins(y=0.12)
    axes_flat[5].axis("off")
    axes_flat[0].legend(loc="lower left")
    fig.suptitle("E035: niezależna ocena top-1 po CMA (CIFAR test; Tiny oficjalny val)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Średnia ± SD (ddof=1) z 4 seedów; nie CI. Tiny: oficjalny val, nie test.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs = list(save(fig, "e035_niezalezny_top1"))

    # Rysunek 2: sparowana różnica T-U per seed + średnia i SD.
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharey=False)
    axes_flat = axes.ravel()
    for index, pair in enumerate(PAIRS):
        ax = axes_flat[index]
        color = COLORS[pair]
        for seed in SEEDS:
            xs = [TAG_X[tag] + OFFSETS[seed] for tag in TAGS]
            ys = [
                next(row["delta"] for row in cells[("test", pair, tag)] if row["seed"] == seed)
                for tag in TAGS
            ]
            ax.plot(xs, ys, color="0.55", lw=0.7, alpha=0.55, zorder=1)
            ax.scatter(
                xs,
                ys,
                marker=SEED_MARKERS[seed],
                s=22,
                color=color,
                alpha=0.85,
                zorder=2,
                label=f"seed {seed}",
            )
        means = [st.mean([row["delta"] for row in cells[("test", pair, tag)]]) for tag in TAGS]
        sds = [st.stdev([row["delta"] for row in cells[("test", pair, tag)]]) for tag in TAGS]
        ax.errorbar(
            X,
            means,
            yerr=sds,
            fmt="D-",
            color="black",
            lw=1.4,
            ms=5,
            capsize=3,
            zorder=5,
            label="średnia ± SD",
        )
        for x, tag, mean, sd in zip(X, TAGS, means, sds):
            offset = 11 if mean >= 0 else -15
            ax.annotate(
                f"{mean:+.2f}",
                (x, mean),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=7,
            )
            add_row(
                rows,
                "e035_sparowane_delta",
                panel_label(pair, "test"),
                "T-U",
                None,
                x,
                mean,
                "pp",
                mean,
                sd,
                "mean",
            )
            for row in cells[("test", pair, tag)]:
                add_row(
                    rows,
                    "e035_sparowane_delta",
                    panel_label(pair, "test"),
                    "T-U",
                    row["seed"],
                    x,
                    row["delta"],
                    "pp",
                    mean,
                    sd,
                    "seed",
                )
        ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=0)
        ax.set_title(panel_label(pair, "test"))
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.set_ylabel("TILT − UNIFORM po CMA [pp]")
        ax.margins(y=0.18)
    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.suptitle("E035: sparowana różnica TILT − UNIFORM po CMA (niezależna ocena)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Punkty = 4 seedy; czarny romb = średnia, słupki = SD (ddof=1), nie CI. Zero = brak różnicy.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs.extend(save(fig, "e035_sparowane_delta"))

    # Rysunek 3 (kontrola spójności): średnia T-U na dev vs niezależna ocena.
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for index, pair in enumerate(PAIRS):
        ax = axes_flat[index]
        xs = []
        ys = []
        for tag in TAGS:
            dev_mean = st.mean([row["delta"] for row in cells[("dev", pair, tag)]])
            test_mean = st.mean([row["delta"] for row in cells[("test", pair, tag)]])
            xs.append(dev_mean)
            ys.append(test_mean)
            marker = {"b30": "o", "b50": "s", "b70": "^"}[tag]
            ax.scatter(
                dev_mean,
                test_mean,
                s=58,
                color=COLORS[pair],
                marker=marker,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                f"{tag[1:]}%",
                (dev_mean, test_mean),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
            for phase, series, value in (
                ("dev", "T-U_dev", dev_mean),
                ("test", "T-U_test", test_mean),
            ):
                add_row(
                    rows,
                    "e035_dev_vs_niezalezny",
                    f"{LABELS[pair]} {phase_label(pair, phase)}",
                    series,
                    None,
                    TAG_X[tag],
                    value,
                    "pp",
                    value,
                    0.0,
                    "mean",
                )
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = max(0.4, 0.12 * (hi - lo))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, zorder=1)
        ax.axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.7, zorder=0)
        ax.axvline(0.0, color="k", lw=0.8, ls=":", alpha=0.7, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{LABELS[pair]} ({phase_label(pair, 'test')})")
        ax.set_xlabel("Dev: średnia TILT − UNIFORM [pp]")
        ax.set_ylabel("Niezależna: średnia TILT − UNIFORM [pp]")
        ax.margins(0.02)
    axes_flat[5].axis("off")
    budget_handles = [
        Line2D([], [], marker=marker, ls="", color="0.35", label=f"{tag[1:]}%")
        for tag, marker in (("b30", "o"), ("b50", "s"), ("b70", "^"))
    ]
    budget_handles.append(Line2D([], [], ls="--", color="k", label="y = x"))
    axes_flat[5].legend(handles=budget_handles, loc="center", frameon=False, fontsize=9)
    fig.suptitle("E035: spójność dev vs niezależna ocena (CMA)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Punkty = średnie z 4 seedów. Tiny: niezależna = oficjalny val. To kontrola opisowa, nie test istotności.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs.extend(save(fig, "e035_dev_vs_niezalezny"))

    csv_path = OUT / "punkty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(csv_path)

    source_set = {Path(path).resolve() for path in sources}
    for path in outputs:
        resolved = Path(path).resolve()
        if OUT.resolve() not in resolved.parents:
            raise RuntimeError(f"Wynik poza figures/: {resolved}")
        if resolved in source_set:
            raise RuntimeError(f"Wynik nadpisałby wejście: {resolved}")

    after = {str(path.resolve()): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("Plik wejściowy zmienił się podczas generowania wykresów")

    sources_doc = {
        "eksperyment": "E035",
        "run": str(RUN.relative_to(ROOT)),
        "opis": "Niezależna ocena (test; Tiny oficjalny val) i dev; surowe summary oraz analysis.json do kontroli.",
        "skrypt": {
            "sciezka": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
        },
        "zrodla": [
            {
                "sciezka": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bajty": path.stat().st_size,
            }
            for path in sorted(sources, key=lambda p: str(p.relative_to(ROOT)))
        ],
        "zabezpieczenie": {
            "wejscia_niezmienione": True,
            "sprawdzone_po_generowaniu": True,
            "wyniki_tylko_w_figures": True,
        },
    }
    (OUT / "sources.json").write_text(
        json.dumps(sources_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    positive_test = sum(
        1
        for pair in PAIRS
        for tag in TAGS
        if st.mean([row["delta"] for row in cells[("test", pair, tag)]]) > 0
    )
    positive_dev = sum(
        1
        for pair in PAIRS
        for tag in TAGS
        if st.mean([row["delta"] for row in cells[("dev", pair, tag)]]) > 0
    )
    verification = {
        "eksperyment": "E035",
        "faza_niezalezna": "test; Tiny = oficjalny val (cache_val)",
        "seedy": list(SEEDS),
        "komorki": len(PAIRS) * len(TAGS),
        "wiersze_analysis": len(analysis["rows"]),
        "kontrasty_CMA": len(contrasts),
        "punkty_per_seed_test": len(PAIRS) * len(TAGS) * len(SEEDS),
        "dodatnie_srednie_test": positive_test,
        "dodatnie_srednie_dev": positive_dev,
        "tiny_val_potwierdzony": tiny_val_confirmed,
        "zgodnosc_z_analysis_json": {
            "max_abs_diff_sredniej_pp": max_mean_diff,
            "max_abs_diff_sd_pp": max_sd_diff,
            "max_abs_diff_absolute_lub_per_seed_pp": max_abs_diff,
        },
        "srednie_T_minus_U_test_pp": {
            f"{pair}/{tag}": {
                "mean": st.mean([row["delta"] for row in cells[("test", pair, tag)]]),
                "sd": st.stdev([row["delta"] for row in cells[("test", pair, tag)]]),
                "positive": sum(1 for row in cells[("test", pair, tag)] if row["delta"] > 0),
                "n": len(cells[("test", pair, tag)]),
                "etykieta": phase_label(pair, "test"),
            }
            for pair in PAIRS
            for tag in TAGS
        },
        "zabezpieczenie_wejsc": {
            "liczba_plikow": len(sources),
            "hash_zgodny_przed_i_po": True,
        },
    }
    (OUT / "weryfikacja.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("E035: zapisano", len(outputs) + 2, "plików w", OUT)
    print("komorki", len(PAIRS) * len(TAGS), "kontrasty CMA", len(contrasts))
    print("dodatnie średnie test", positive_test, "/15; dev", positive_dev, "/15")
    for pair in PAIRS:
        for tag in TAGS:
            values = [row["delta"] for row in cells[("test", pair, tag)]]
            print(
                f"{pair}/{tag} [{phase_label(pair, 'test')}]: "
                f"mean={st.mean(values):+.4f} pp sd={st.stdev(values):.4f} "
                f"positive={sum(v > 0 for v in values)}/4"
            )


if __name__ == "__main__":
    main()
