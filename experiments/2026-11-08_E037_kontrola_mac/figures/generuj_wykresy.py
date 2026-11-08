#!/usr/bin/env python3
"""E037: kontrola przy dopasowanym MAC (dev only, CPU only, deterministycznie).

Wejście: surowe summary.json E037 (U'), oryginalne summary.json dev E035 (U/T),
frozen.json E037 (plany i MAC) oraz analysis.json do kontroli.
Wyjście: PNG + PDF + punkty.csv + sources.json + weryfikacja.json w figures/.
Nie otwiera testu; E037 jest kontrolą dev.
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
RUN = ROOT / "runs/E037/20261108T031759Z_855c0b33"
RUN35 = ROOT / "runs/E035/20261107T183831Z_8b0b14e0"
ANALYSIS = RUN / "analysis/analysis.json"
VERIFICATION = RUN / "analysis/verification.json"
FROZEN = RUN / "frozen.json"

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
            "Creator": f"E037 {stem}",
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
        RUN / f"seed_{seed}/{pair}/{tag}/summary.json"
        for seed in SEEDS
        for pair in PAIRS
        for tag in TAGS
    ]
    dev_paths = [
        RUN35 / f"dev/seed_{seed}/{pair}/{tag}/summary.json"
        for seed in SEEDS
        for pair in PAIRS
        for tag in TAGS
    ]
    sources = [ANALYSIS, VERIFICATION, FROZEN, *summary_paths, *dev_paths]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path.resolve()): sha256(path) for path in sources}

    # U' z surowych summary E037; U/T z oryginalnych summary dev E035.
    cells = {}
    for pair in PAIRS:
        for tag in TAGS:
            per_seed = []
            for seed in SEEDS:
                summary = read(RUN / f"seed_{seed}/{pair}/{tag}/summary.json")
                if summary.get("phase") != "dev":
                    raise RuntimeError("E037 powinien być wyłącznie dev")
                up = summary["metrics"]["CMA"]["top1"] * 100.0
                dev = read(RUN35 / f"dev/seed_{seed}/{pair}/{tag}/summary.json")
                u = dev["metrics"]["U"]["CMA"]["top1"] * 100.0
                t = dev["metrics"]["T"]["CMA"]["top1"] * 100.0
                per_seed.append(
                    {
                        "seed": seed,
                        "U": u,
                        "T": t,
                        "Up": up,
                        "delta": t - up,
                    }
                )
            cells[(pair, tag)] = per_seed

    analysis = read(ANALYSIS)
    if len(analysis["rows"]) != 180:
        raise RuntimeError("Nieoczekiwana liczba rows w analysis.json")

    # Kontrola: surowe summary muszą odtworzyć analysis.json rows (CMA).
    rows_ref = {
        (row["pair"], row["tag"], row["seed"], row["mode"]): row
        for row in analysis["rows"]
    }
    max_abs_diff = 0.0
    for pair in PAIRS:
        for tag in TAGS:
            for row in cells[(pair, tag)]:
                ref = rows_ref[(pair, tag, row["seed"], "CMA")]
                for profile in ("U", "T", "Up"):
                    ref_pct = ref[profile] * 100.0
                    max_abs_diff = max(max_abs_diff, abs(row[profile] - ref_pct))
                    if abs(row[profile] - ref_pct) > 1e-12:
                        raise RuntimeError(
                            f"Niezgodność rows {pair}/{tag}/{row['seed']}/{profile}"
                        )

    # Kontrola: sparowana różnica T-U' względem analysis.json allocation.
    allocation = {
        (row["pair"], row["tag"]): row
        for row in analysis["allocation"]
        if row["mode"] == "CMA" and row["contrast"] == "T-Up"
    }
    if len(allocation) != 15:
        raise RuntimeError("Oczekiwano 15 kontrastów CMA T-Up")
    max_mean_diff = 0.0
    max_sd_diff = 0.0
    for pair in PAIRS:
        for tag in TAGS:
            deltas = [row["delta"] for row in cells[(pair, tag)]]
            mean = st.mean(deltas)
            sd = st.stdev(deltas)
            ref = allocation[(pair, tag)]
            max_mean_diff = max(max_mean_diff, abs(mean - ref["mean"]))
            max_sd_diff = max(max_sd_diff, abs(sd - ref["sd"]))
            if abs(mean - ref["mean"]) > 1e-9 or abs(sd - ref["sd"]) > 1e-9:
                raise RuntimeError(f"Niezgodność allocation {pair}/{tag}")
            for got, want in zip(deltas, ref["per_seed"]):
                if abs(got - want) > 1e-9:
                    raise RuntimeError(f"Niezgodność per_seed {pair}/{tag}")

    # Plany: U' ma więcej parametrów, ale MAC nie przekracza T.
    frozen = read(FROZEN)
    plan_ratios = {}
    for pair in PAIRS:
        for tag in TAGS:
            plan = frozen["plans"][pair][tag]
            params_ratio = plan["Uprime_params"] / plan["T_params"]
            mac_ratio = plan["Uprime_mac"] / plan["T_mac"]
            plan_ratios[(pair, tag)] = {
                "params": params_ratio,
                "mac": mac_ratio,
                "mac_matched": bool(plan["mac_matched"]),
            }
            if not plan["mac_matched"] or mac_ratio > 1.0 + 1e-12:
                raise RuntimeError(f"U' nie jest dopasowany MAC dla {pair}/{tag}")
            if params_ratio <= 1.0:
                raise RuntimeError(f"U' nie ma więcej parametrów dla {pair}/{tag}")

    verification = read(VERIFICATION)
    if not verification.get("dev_only") or verification.get("data_open_attempts") != 0:
        raise RuntimeError("E037 nie potwierdza dev-only / braku otwarcia testu")

    style()
    rows = []

    # Rysunek 1: top-1 dev CMA dla U, T, U'.
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharey=False)
    axes_flat = axes.ravel()
    for index, pair in enumerate(PAIRS):
        ax = axes_flat[index]
        for profile, marker, linestyle, color, label in (
            ("U", "o", "--", "#0072B2", "UNIFORM (U)"),
            ("T", "s", "-", "#D55E00", "TILT (T)"),
            ("Up", "^", "-", "#009E73", "UNIFORM dopasowany MAC (U')"),
        ):
            means = []
            sds = []
            for tag in TAGS:
                values = [row[profile] for row in cells[(pair, tag)]]
                means.append(st.mean(values))
                sds.append(st.stdev(values))
                add_row(
                    rows,
                    "e037_top1_u_t_up",
                    LABELS[pair],
                    label,
                    None,
                    TAG_X[tag],
                    means[-1],
                    "%",
                    means[-1],
                    sds[-1],
                    "mean",
                )
                for row in cells[(pair, tag)]:
                    add_row(
                        rows,
                        "e037_top1_u_t_up",
                        LABELS[pair],
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
        ax.set_title(f"{LABELS[pair]} (dev)")
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.set_ylabel("Top-1 CMA [%]")
        ax.margins(y=0.12)
    axes_flat[5].axis("off")
    axes_flat[0].legend(loc="lower left", fontsize=6.5)
    params_values = [value["params"] for value in plan_ratios.values()]
    mac_values = [value["mac"] for value in plan_ratios.values()]
    fig.suptitle("E037: top-1 dev po CMA — U, T i U' (UNIFORM dopasowany MAC)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        f"U' ma {min(params_values):.2f}–{max(params_values):.2f}× parametrów T i {min(mac_values):.3f}–{max(mac_values):.3f}× MAC T. "
        "Dev only; średnia ± SD (ddof=1) z 4 seedów, nie CI.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs = list(save(fig, "e037_top1_u_t_up"))

    # Rysunek 2: sparowana różnica T-U' per seed + średnia i SD.
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharey=False)
    axes_flat = axes.ravel()
    for index, pair in enumerate(PAIRS):
        ax = axes_flat[index]
        color = COLORS[pair]
        for seed in SEEDS:
            xs = [TAG_X[tag] + OFFSETS[seed] for tag in TAGS]
            ys = [
                next(row["delta"] for row in cells[(pair, tag)] if row["seed"] == seed)
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
        means = [st.mean([row["delta"] for row in cells[(pair, tag)]]) for tag in TAGS]
        sds = [st.stdev([row["delta"] for row in cells[(pair, tag)]]) for tag in TAGS]
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
                "e037_t_minus_up",
                LABELS[pair],
                "T-U'",
                None,
                x,
                mean,
                "pp",
                mean,
                sd,
                "mean",
            )
            for row in cells[(pair, tag)]:
                add_row(
                    rows,
                    "e037_t_minus_up",
                    LABELS[pair],
                    "T-U'",
                    row["seed"],
                    x,
                    row["delta"],
                    "pp",
                    mean,
                    sd,
                    "seed",
                )
        ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=0)
        ax.set_title(f"{LABELS[pair]} (dev)")
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.set_ylabel("TILT − UNIFORM(MAC) po CMA [pp]")
        ax.margins(y=0.18)
    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.suptitle("E037: sparowana różnica TILT − UNIFORM(MAC) po CMA (dev)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Dodatnie = TILT lepszy przy dopasowanym MAC; ujemne = U' lepszy. Punkty = 4 seedy; czarny romb = średnia, "
        "słupki = SD (ddof=1), nie CI. To nie dowód przyczynowy.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    outputs.extend(save(fig, "e037_t_minus_up"))

    # Rysunek 3 (kontekst kontroli): parametry i MAC U' względem T.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for pair in PAIRS:
        params = [plan_ratios[(pair, tag)]["params"] for tag in TAGS]
        mac = [plan_ratios[(pair, tag)]["mac"] for tag in TAGS]
        axes[0].plot(X, params, "o-", color=COLORS[pair], lw=1.3, ms=4, label=LABELS[pair])
        axes[1].plot(X, mac, "o-", color=COLORS[pair], lw=1.3, ms=4, label=LABELS[pair])
        for tag, param_value, mac_value in zip(TAGS, params, mac):
            add_row(
                rows,
                "e037_up_params_mac",
                LABELS[pair],
                "Parametry U'/T",
                None,
                TAG_X[tag],
                param_value,
                "×",
                param_value,
                0.0,
                "mean",
            )
            add_row(
                rows,
                "e037_up_params_mac",
                LABELS[pair],
                "MAC U'/T",
                None,
                TAG_X[tag],
                mac_value,
                "×",
                mac_value,
                0.0,
                "mean",
            )
    axes[0].axhline(1.0, color="black", lw=1.0, ls="--", zorder=0)
    axes[0].set_title("Parametry: U' / T")
    axes[0].set_ylabel("Iloraz parametrów [×]")
    axes[1].axhline(1.0, color="black", lw=1.0, ls="--", zorder=0)
    axes[1].set_title("MAC: U' / T")
    axes[1].set_ylabel("Iloraz MAC [×]")
    for ax in axes:
        ax.set_xlabel("Nominalna redukcja parametrów [%]")
        ax.set_xticks(X)
        ax.margins(y=0.12)
    axes[0].legend(fontsize=7)
    fig.suptitle("E037: budżet U' względem T (dev, plany zamrożone)", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "U' = UNIFORM dopasowany MAC: ma więcej parametrów, ale MAC nie przekracza T. To definicja kontroli, nie wynik jakości.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    outputs.extend(save(fig, "e037_up_params_mac"))

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
        "eksperyment": "E037",
        "run": str(RUN.relative_to(ROOT)),
        "opis": "Kontrola dev przy dopasowanym MAC: U' z E037, U/T z oryginalnych summary dev E035, plany z frozen.json.",
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

    verification_doc = {
        "eksperyment": "E037",
        "zakres": "dev only; test nie był otwierany",
        "dev_only_z_verification_json": bool(verification.get("dev_only")),
        "data_open_attempts": verification.get("data_open_attempts"),
        "seedy": list(SEEDS),
        "komorki": len(PAIRS) * len(TAGS),
        "wiersze_analysis": len(analysis["rows"]),
        "punkty_per_seed": len(PAIRS) * len(TAGS) * len(SEEDS),
        "zgodnosc_z_analysis_json": {
            "max_abs_diff_top1_pp": max_abs_diff,
            "max_abs_diff_sredniej_T_minus_Up_pp": max_mean_diff,
            "max_abs_diff_sd_T_minus_Up_pp": max_sd_diff,
        },
        "zakresy": {
            "params_Uprime_przez_T": [min(params_values), max(params_values)],
            "mac_Uprime_przez_T": [min(mac_values), max(mac_values)],
            "T_minus_Up_mean_pp": [
                min(st.mean([row["delta"] for row in cells[(pair, tag)]]) for pair in PAIRS for tag in TAGS),
                max(st.mean([row["delta"] for row in cells[(pair, tag)]]) for pair in PAIRS for tag in TAGS),
            ],
        },
        "srednie_T_minus_Up_pp": {
            f"{pair}/{tag}": {
                "mean": st.mean([row["delta"] for row in cells[(pair, tag)]]),
                "sd": st.stdev([row["delta"] for row in cells[(pair, tag)]]),
                "positive": sum(1 for row in cells[(pair, tag)] if row["delta"] > 0),
                "n": len(cells[(pair, tag)]),
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
        json.dumps(verification_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("E037: zapisano", len(outputs) + 2, "plików w", OUT)
    print("komorki", len(PAIRS) * len(TAGS), "punkty per-seed", len(PAIRS) * len(TAGS) * len(SEEDS))
    print(
        "params U'/T", f"{min(params_values):.4f}–{max(params_values):.4f}",
        "MAC U'/T", f"{min(mac_values):.4f}–{max(mac_values):.4f}",
    )
    for pair in PAIRS:
        for tag in TAGS:
            values = [row["delta"] for row in cells[(pair, tag)]]
            print(
                f"{pair}/{tag}: mean={st.mean(values):+.4f} pp "
                f"sd={st.stdev(values):.4f} positive={sum(v > 0 for v in values)}/4"
            )


if __name__ == "__main__":
    main()
