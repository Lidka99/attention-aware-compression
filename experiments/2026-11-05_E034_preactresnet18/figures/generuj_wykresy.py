#!/usr/bin/env python3
"""E034: wykresy CMA TILT-UNIFORM dla PreActResNet18 (dev, CPU only).

Wejście: analysis.json oraz surowe summary.json z runu fold_20261106T003625Z_9ec4f2ea.
Wyjście: PNG + PDF + punkty.csv + sources.json + weryfikacja.json w tym katalogu.
Skrypt nie pisze poza figures/ i sprawdza, że pliki wejściowe nie zmieniły się.
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

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/E034/fold_20261106T003625Z_9ec4f2ea"
ANALYSIS = RUN / "analysis_20261106T004001Z/analysis.json"

SEEDS = (42, 43, 44, 45)
DATASETS = ("cifar10", "cifar100")
TAGS = ("b30", "b50", "b70")
X = (30, 50, 70)
TAG_X = dict(zip(TAGS, X))
DATASET_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
COLORS = {"cifar10": "#0072B2", "cifar100": "#D55E00"}
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
            "Creator": f"E034 {stem}",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


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
        RUN / f"seed_{seed}/{dataset}/{tag}/summary.json"
        for seed in SEEDS
        for dataset in DATASETS
        for tag in TAGS
    ]
    sources = [ANALYSIS, *summary_paths]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    before = {str(path.resolve()): sha256(path) for path in sources}

    # Surowa różnica T-U po CMA, per seed, dla obu zbiorów i trzech budżetów.
    cells = {}
    for dataset in DATASETS:
        for tag in TAGS:
            per_seed = []
            for seed in SEEDS:
                summary = read(RUN / f"seed_{seed}/{dataset}/{tag}/summary.json")
                metrics = summary["metrics"]
                u = metrics["UP_U"]["CMA"]["top1"] * 100.0
                t = metrics["UP_T"]["CMA"]["top1"] * 100.0
                per_seed.append(
                    {
                        "seed": seed,
                        "U": u,
                        "T": t,
                        "delta": t - u,
                    }
                )
            cells[(dataset, tag)] = per_seed

    # Kontrola: surowe summary muszą odtworzyć analysis.json primary (CMA).
    analysis = read(ANALYSIS)
    primary = {
        (row["dataset"], row["tag"]): row
        for row in analysis["primary"]
        if row["mode"] == "CMA"
    }
    max_mean_diff = 0.0
    max_sd_diff = 0.0
    for dataset in DATASETS:
        for tag in TAGS:
            per_seed = cells[(dataset, tag)]
            deltas = [row["delta"] for row in per_seed]
            mean = st.mean(deltas)
            sd = st.stdev(deltas)
            ref = primary[(dataset, tag)]
            max_mean_diff = max(max_mean_diff, abs(mean - ref["mean"]))
            max_sd_diff = max(max_sd_diff, abs(sd - ref["sd"]))
            if abs(mean - ref["mean"]) > 1e-9 or abs(sd - ref["sd"]) > 1e-9:
                raise RuntimeError(f"Niezgodność z analysis.json dla {dataset}/{tag}")
            for got, want in zip(deltas, ref["per_seed"]):
                if abs(got - want) > 1e-9:
                    raise RuntimeError(f"Niezgodność per_seed dla {dataset}/{tag}")
    if len(primary) != 6:
        raise RuntimeError(f"Oczekiwano 6 komórek CMA, jest {len(primary)}")

    style()
    rows = []

    # Rysunek 1: sparowana różnica T-U, punkty per seed + średnia i SD.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        color = COLORS[dataset]
        for seed in SEEDS:
            xs = [TAG_X[tag] + OFFSETS[seed] for tag in TAGS]
            ys = [next(r["delta"] for r in cells[(dataset, tag)] if r["seed"] == seed) for tag in TAGS]
            ax.plot(xs, ys, color="0.55", lw=0.7, alpha=0.55, zorder=1)
            ax.scatter(
                xs,
                ys,
                marker=SEED_MARKERS[seed],
                s=24,
                color=color,
                alpha=0.85,
                zorder=2,
                label=f"seed {seed}",
            )
        means = [st.mean([r["delta"] for r in cells[(dataset, tag)]]) for tag in TAGS]
        sds = [st.stdev([r["delta"] for r in cells[(dataset, tag)]]) for tag in TAGS]
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
        for x, mean, sd in zip(X, means, sds):
            offset = 11 if mean >= 0 else -15
            ax.annotate(
                f"{mean:+.2f}",
                (x, mean),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=7,
            )
            add_row(rows, "e034_cma_t_minus_u", DATASET_LABEL[dataset], "T-U", None, x, mean, "pp", mean, sd, "mean")
            for row in cells[(dataset, tag_of_x(x))]:
                add_row(
                    rows,
                    "e034_cma_t_minus_u",
                    DATASET_LABEL[dataset],
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
        ax.set_title(f"{DATASET_LABEL[dataset]} (dev)")
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.margins(y=0.15)
    axes[0].set_ylabel("TILT − UNIFORM po CMA [pp]")
    axes[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    fig.suptitle("E034: sparowana różnica TILT − UNIFORM po CMA (dev, PreActResNet18)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Punkty = 4 seedy; czarny romb = średnia, słupki = SD (ddof=1), nie CI. Zero = brak różnicy.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    outputs = list(save(fig, "e034_cma_t_minus_u"))

    # Rysunek 2 (kontekst): bezwzględny top-1 CMA dla U i T, średnia ± SD.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        for profile, marker, linestyle, color, label in (
            ("U", "o", "--", "#0072B2", "UNIFORM (U)"),
            ("T", "s", "-", "#D55E00", "TILT (T)"),
        ):
            means = []
            sds = []
            for tag in TAGS:
                values = [r[profile] for r in cells[(dataset, tag)]]
                means.append(st.mean(values))
                sds.append(st.stdev(values))
                add_row(
                    rows,
                    "e034_cma_top1",
                    DATASET_LABEL[dataset],
                    label,
                    None,
                    TAG_X[tag],
                    means[-1],
                    "%",
                    means[-1],
                    sds[-1],
                    "mean",
                )
                for row in cells[(dataset, tag)]:
                    add_row(
                        rows,
                        "e034_cma_top1",
                        DATASET_LABEL[dataset],
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
        ax.set_title(f"{DATASET_LABEL[dataset]} (dev)")
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.set_ylabel("Top-1 CMA [%]")
        ax.margins(y=0.12)
    axes[0].legend(loc="lower left")
    fig.suptitle("E034: bezwzględny top-1 po CMA (dev, PreActResNet18)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Średnia ± SD (ddof=1) z 4 seedów; nie CI. Ten sam budżet parametrów dla U i T.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    outputs.extend(save(fig, "e034_cma_top1"))

    # CSV punktów (seed + średnia) w deterministycznej kolejności.
    csv_path = OUT / "punkty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(csv_path)

    # Zabezpieczenie: żaden wynik nie może leżeć poza figures/ ani być wejściem.
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
        "eksperyment": "E034",
        "run": str(RUN.relative_to(ROOT)),
        "opis": "Surowa różnica CMA T-U i top-1 CMA; dev, bez testu.",
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

    verification = {
        "eksperyment": "E034",
        "zakres": "dev; test i rezerwa nieużywane",
        "seedy": list(SEEDS),
        "komorki": len(primary),
        "punkty_per_seed": len(primary) * len(SEEDS),
        "zgodnosc_z_analysis_json": {
            "max_abs_diff_sredniej_pp": max_mean_diff,
            "max_abs_diff_sd_pp": max_sd_diff,
        },
        "srednie_T_minus_U_pp": {
            f"{dataset}/{tag}": {
                "mean": st.mean([r["delta"] for r in cells[(dataset, tag)]]),
                "sd": st.stdev([r["delta"] for r in cells[(dataset, tag)]]),
                "positive": sum(1 for r in cells[(dataset, tag)] if r["delta"] > 0),
                "n": len(cells[(dataset, tag)]),
            }
            for dataset in DATASETS
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

    print("E034: zapisano", len(outputs) + 2, "plików w", OUT)
    print("komorki", len(primary), "punkty per-seed", len(primary) * len(SEEDS))
    for dataset in DATASETS:
        for tag in TAGS:
            values = [r["delta"] for r in cells[(dataset, tag)]]
            print(
                f"{dataset}/{tag}: mean={st.mean(values):+.4f} pp "
                f"sd={st.stdev(values):.4f} positive={sum(v > 0 for v in values)}/4"
            )


def tag_of_x(x):
    for tag, value in TAG_X.items():
        if value == x:
            return tag
    raise KeyError(x)


if __name__ == "__main__":
    main()
