#!/usr/bin/env python3
"""E036: wykresy kosztu TILT vs UNIFORM (CPU only, deterministycznie).

Wejście: surowe pliki czasów *_bs{1,128}_s{0,1,2}.json oraz costs.json z runu
20261107T231027Z_27bac186; analysis.json służy tylko do kontroli zgodności.
Batch1: per-seed iloraz T median / U median i SD z czterech seedów.
Nie używamy SD z ręcznych TABLES (były błędne).
Wyjście: PNG + PDF + punkty.csv + sources.json + weryfikacja.json w figures/.
"""
import csv
import hashlib
import json
import math
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
RUN = ROOT / "runs/E036/20261107T231027Z_27bac186"
ANALYSIS = RUN / "analysis/analysis.json"

SEEDS = (42, 43, 44, 45)
PAIRS = (
    "resnet_cifar10",
    "resnet_cifar100",
    "resnet_tiny",
    "preact_cifar10",
    "preact_cifar100",
)
TAGS = ("b30", "b50", "b70")
BATCHES = (1, 128)
X = (30, 50, 70)
TAG_X = dict(zip(TAGS, X))
LABELS = {
    "resnet_cifar10": "ResNet/CIFAR-10",
    "resnet_cifar100": "ResNet/CIFAR-100",
    "resnet_tiny": "ResNet/Tiny",
    "preact_cifar10": "PreAct/CIFAR-10",
    "preact_cifar100": "PreAct/CIFAR-100",
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
            "Creator": f"E036 {stem}",
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

    timing_paths = []
    cost_paths = []
    for seed in SEEDS:
        for pair in PAIRS:
            cost_paths.append(RUN / f"seed_{seed}/{pair}/costs.json")
            for tag in TAGS:
                for profile in ("T", "U"):
                    for batch in BATCHES:
                        for series in range(3):
                            timing_paths.append(
                                RUN
                                / f"seed_{seed}/{pair}/{tag}_{profile}_bs{batch}_s{series}.json"
                            )
    sources = [ANALYSIS, *cost_paths, *timing_paths]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path.resolve()): sha256(path) for path in sources}

    analysis = read(ANALYSIS)
    ref_ratios = {
        (row["seed"], row["pair"], row["tag"], row["batch"]): row
        for row in analysis["ratios"]
    }
    if len(ref_ratios) != 120:
        raise RuntimeError("Nieoczekiwana liczba ratios w analysis.json")

    def times(seed, pair, tag, profile, batch):
        values = []
        for series in range(3):
            data = read(
                RUN / f"seed_{seed}/{pair}/{tag}_{profile}_bs{batch}_s{series}.json"
            )
            if len(data["times_ms"]) != 200:
                raise RuntimeError("Oczekiwano 200 pomiarów w serii")
            values.extend(data["times_ms"])
        if len(values) != 600 or not all(x > 0 and math.isfinite(x) for x in values):
            raise RuntimeError("Błędne czasy w plikach pomiarowych")
        return values

    # Per-seed ilorazy T/U liczone wprost z surowych czasów i costs.json.
    cells = {}
    max_ratio_diff = 0.0
    for pair in PAIRS:
        for tag in TAGS:
            per_seed = []
            for seed in SEEDS:
                costs = read(RUN / f"seed_{seed}/{pair}/costs.json")
                mac = costs[tag + "_T"]["mac"] / costs[tag + "_U"]["mac"]
                params = costs[tag + "_T"]["params"] / costs[tag + "_U"]["params"]
                t1 = times(seed, pair, tag, "T", 1)
                u1 = times(seed, pair, tag, "U", 1)
                t128 = times(seed, pair, tag, "T", 128)
                u128 = times(seed, pair, tag, "U", 128)
                latency = st.median(t1) / st.median(u1)
                throughput = (1000.0 * 128 / st.mean(t128)) / (
                    1000.0 * 128 / st.mean(u128)
                )
                per_seed.append(
                    {
                        "seed": seed,
                        "mac": mac,
                        "params": params,
                        "latency_b1": latency,
                        "throughput_b128": throughput,
                    }
                )
                ref = ref_ratios[(seed, pair, tag, 1)]
                ref128 = ref_ratios[(seed, pair, tag, 128)]
                for got, want in (
                    (latency, ref["latency_ratio"]),
                    (throughput, ref128["throughput_ratio"]),
                    (mac, ref["mac_ratio"]),
                    (params, ref["params_ratio"]),
                ):
                    max_ratio_diff = max(max_ratio_diff, abs(got - want))
                    if abs(got - want) > 1e-12:
                        raise RuntimeError(f"Niezgodność ratios {seed}/{pair}/{tag}")
            cells[(pair, tag)] = per_seed

    style()
    rows = []

    # Rysunek 1: MAC T/U oraz przepustowość batch128 T/U.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    mac_all = []
    throughput_all = []
    for pair in PAIRS:
        mac_values = [st.mean([row["mac"] for row in cells[(pair, tag)]]) for tag in TAGS]
        throughput_means = [
            st.mean([row["throughput_b128"] for row in cells[(pair, tag)]]) for tag in TAGS
        ]
        throughput_sds = [
            st.stdev([row["throughput_b128"] for row in cells[(pair, tag)]]) for tag in TAGS
        ]
        mac_all.extend(mac_values)
        throughput_all.extend(throughput_means)
        axes[0].plot(X, mac_values, "o-", color=COLORS[pair], lw=1.3, ms=4, label=LABELS[pair])
        axes[1].errorbar(
            X,
            throughput_means,
            yerr=throughput_sds,
            fmt="o-",
            color=COLORS[pair],
            lw=1.3,
            ms=4,
            capsize=3,
            label=LABELS[pair],
        )
        for tag, mac, thr_mean, thr_sd in zip(
            TAGS, mac_values, throughput_means, throughput_sds
        ):
            add_row(
                rows,
                "e036_mac_i_przepustowosc",
                LABELS[pair],
                "MAC T/U",
                None,
                TAG_X[tag],
                mac,
                "×",
                mac,
                0.0,
                "mean",
            )
            add_row(
                rows,
                "e036_mac_i_przepustowosc",
                LABELS[pair],
                "Przepustowość b128 T/U",
                None,
                TAG_X[tag],
                thr_mean,
                "×",
                thr_mean,
                thr_sd,
                "mean",
            )
            for row in cells[(pair, tag)]:
                add_row(
                    rows,
                    "e036_mac_i_przepustowosc",
                    LABELS[pair],
                    "MAC T/U",
                    row["seed"],
                    TAG_X[tag],
                    row["mac"],
                    "×",
                    mac,
                    0.0,
                    "seed",
                )
                add_row(
                    rows,
                    "e036_mac_i_przepustowosc",
                    LABELS[pair],
                    "Przepustowość b128 T/U",
                    row["seed"],
                    TAG_X[tag],
                    row["throughput_b128"],
                    "×",
                    thr_mean,
                    thr_sd,
                    "seed",
                )
    for ax, title, ylabel, reference in (
        (axes[0], "MAC: TILT / UNIFORM", "Iloraz MAC [×]", 1.0),
        (axes[1], "Przepustowość batch128: TILT / UNIFORM", "Iloraz przepustowości [×]", 1.0),
    ):
        ax.axhline(reference, color="black", lw=1.0, ls="--", zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(X)
        ax.margins(y=0.12)
    axes[0].legend(fontsize=7)
    fig.suptitle("E036: ten sam budżet parametrów, inny koszt obliczeń", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "MAC jest deterministyczny (bez SD); przepustowość = średnia ± SD (ddof=1) z 4 seedów, nie CI. "
        f"Zakres MAC T/U: {min(mac_all):.3f}–{max(mac_all):.3f}×; przepustowość: {min(throughput_all):.3f}–{max(throughput_all):.3f}×.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    outputs = list(save(fig, "e036_mac_i_przepustowosc"))

    # Rysunek 2: batch1, per-seed T median / U median + średnia i prawdziwa SD.
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), sharey=True)
    axes_flat = axes.ravel()
    all_values = []
    for pair in PAIRS:
        for tag in TAGS:
            all_values.extend([row["latency_b1"] for row in cells[(pair, tag)]])
    lo = min(all_values)
    hi = max(all_values)
    pad = max(0.01, 0.12 * (hi - lo))
    for index, pair in enumerate(PAIRS):
        ax = axes_flat[index]
        color = COLORS[pair]
        for seed in SEEDS:
            xs = [TAG_X[tag] + OFFSETS[seed] for tag in TAGS]
            ys = [
                next(row["latency_b1"] for row in cells[(pair, tag)] if row["seed"] == seed)
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
        means = [st.mean([row["latency_b1"] for row in cells[(pair, tag)]]) for tag in TAGS]
        sds = [st.stdev([row["latency_b1"] for row in cells[(pair, tag)]]) for tag in TAGS]
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
        for tag, mean, sd in zip(TAGS, means, sds):
            add_row(
                rows,
                "e036_batch1_ratio",
                LABELS[pair],
                "Latencja b1 T/U (mediana)",
                None,
                TAG_X[tag],
                mean,
                "×",
                mean,
                sd,
                "mean",
            )
            for row in cells[(pair, tag)]:
                add_row(
                    rows,
                    "e036_batch1_ratio",
                    LABELS[pair],
                    "Latencja b1 T/U (mediana)",
                    row["seed"],
                    TAG_X[tag],
                    row["latency_b1"],
                    "×",
                    mean,
                    sd,
                    "seed",
                )
        ax.axhline(1.0, color="black", lw=1.0, ls="--", zorder=0)
        ax.set_title(LABELS[pair])
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.set_ylim(lo - pad, hi + pad)
    axes_flat[5].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latencja batch1: T median / U median [×]")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.suptitle("E036: batch1 — per-seed T median / U median", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Punkty = 4 seedy; czarny romb = średnia, słupki = SD (ddof=1) z czterech per-seed T median / U median, nie CI. "
        "Zbliżone średnie nie dowodzą równoważności ani konkretnej przyczyny różnic.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs.extend(save(fig, "e036_batch1_ratio"))

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
        "eksperyment": "E036",
        "run": str(RUN.relative_to(ROOT)),
        "opis": "Koszt z surowych czasów i costs.json; analysis.json tylko do kontroli ilorazów.",
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

    mac_values_all = [row["mac"] for per_seed in cells.values() for row in per_seed]
    throughput_values_all = [
        row["throughput_b128"] for per_seed in cells.values() for row in per_seed
    ]
    throughput_means_all = [
        st.mean([row["throughput_b128"] for row in cells[(pair, tag)]])
        for pair in PAIRS
        for tag in TAGS
    ]
    latency_means = {
        f"{pair}/{tag}": {
            "mean": st.mean([row["latency_b1"] for row in cells[(pair, tag)]]),
            "sd": st.stdev([row["latency_b1"] for row in cells[(pair, tag)]]),
            "n": len(cells[(pair, tag)]),
        }
        for pair in PAIRS
        for tag in TAGS
    }
    verification = {
        "eksperyment": "E036",
        "seedy": list(SEEDS),
        "komorki": len(PAIRS) * len(TAGS),
        "pliki_czasow": len(timing_paths),
        "pliki_costs": len(cost_paths),
        "punkty_per_seed_na_metryke": len(PAIRS) * len(TAGS) * len(SEEDS),
        "zgodnosc_z_analysis_json": {"max_abs_diff_ratio": max_ratio_diff},
        "zakresy": {
            "mac_T_przez_U": [min(mac_values_all), max(mac_values_all)],
            "przepustowosc_b128_T_przez_U_mean": [
                min(throughput_means_all),
                max(throughput_means_all),
            ],
            "przepustowosc_b128_T_przez_U_per_seed": [
                min(throughput_values_all),
                max(throughput_values_all),
            ],
            "latencja_b1_T_median_przez_U_median_mean": [
                min(v["mean"] for v in latency_means.values()),
                max(v["mean"] for v in latency_means.values()),
            ],
            "latencja_b1_SD": [
                min(v["sd"] for v in latency_means.values()),
                max(v["sd"] for v in latency_means.values()),
            ],
            "params_T_przez_U": [
                min(row["params"] for per_seed in cells.values() for row in per_seed),
                max(row["params"] for per_seed in cells.values() for row in per_seed),
            ],
        },
        "sd_batch1": "SD z czterech per-seed T median / U median (ddof=1); nie z TABLES",
        "srednie_latencji_b1": latency_means,
        "zabezpieczenie_wejsc": {
            "liczba_plikow": len(sources),
            "hash_zgodny_przed_i_po": True,
        },
    }
    (OUT / "weryfikacja.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("E036: zapisano", len(outputs) + 2, "plików w", OUT)
    print("komorki", len(PAIRS) * len(TAGS), "pliki czasów", len(timing_paths))
    print(
        "MAC T/U", f"{min(mac_values_all):.3f}–{max(mac_values_all):.3f}",
        "przepustowość b128 T/U (średnie)",
        f"{min(throughput_means_all):.3f}–{max(throughput_means_all):.3f}",
        "(per-seed",
        f"{min(throughput_values_all):.3f}–{max(throughput_values_all):.3f})",
    )
    for pair in PAIRS:
        for tag in TAGS:
            values = [row["latency_b1"] for row in cells[(pair, tag)]]
            print(
                f"{pair}/{tag} batch1: mean={st.mean(values):.4f} "
                f"sd={st.stdev(values):.4f} (prawdziwe SD z 4 seedów)"
            )


if __name__ == "__main__":
    main()
