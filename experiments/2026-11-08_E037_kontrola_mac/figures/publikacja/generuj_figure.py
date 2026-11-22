#!/usr/bin/env python3
"""E035/E037: publikacyjna figura jakość-P/MAC dla U, T i U' (dev, po CMA).

Eksperyment 2026-11-08_E037_kontrola_mac; pełny protokół w OPIS.md.
Wejście: metryki dev E035 (U/T) i E037 (U'), plany P/MAC frozen.json E037,
kontrolne koszty nauczyciela z audytu E036 (check.json). Nie czytamy obrazów,
checkpointów ani wag; brak treningu, GPU i nowej oceny modeli.
Wyjście wyłącznie w tym podkatalogu: PDF/PNG, punkty_publikacja.csv,
sources.json i weryfikacja.json. Opis i podpis: README.md.
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
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent
EXP = OUT.parents[1]
ROOT = OUT.parents[3]
RUN37 = ROOT / "runs/E037/20261108T031759Z_855c0b33"
RUN35 = ROOT / "runs/E035/20261107T183831Z_8b0b14e0"
ANALYSIS = RUN37 / "analysis/analysis.json"
VERIFICATION = RUN37 / "analysis/verification.json"
FROZEN = RUN37 / "frozen.json"
E036_CHECK = ROOT / "experiments/2026-11-07_E036_audyt_kosztu/check.json"

SEEDS = (42, 43, 44, 45)
PAIRS = (
    "resnet_cifar10",
    "resnet_cifar100",
    "resnet_tiny",
    "preact_cifar10",
    "preact_cifar100",
)
TAGS = ("b30", "b50", "b70")
BUDGET = {"b30": 30, "b50": 50, "b70": 70}
PROFILES = ("U", "T", "Up")
PROFILE_NAMES = {"U": "U", "T": "T", "Up": "U'"}
PROFILE_META = {
    "U": {"label": "U (UNIFORM)", "color": "#0072B2", "ls": "--"},
    "T": {"label": "T (TILT)", "color": "#D55E00", "ls": "-"},
    "Up": {"label": "U' (MAC)", "color": "#009E73", "ls": "-."},
}
BUDGET_MARKERS = {"b30": "o", "b50": "s", "b70": "^"}
LABELS = {
    "resnet_cifar10": ("ResNet / CIFAR-10", "ResNet18", "CIFAR-10"),
    "resnet_cifar100": ("ResNet / CIFAR-100", "ResNet18", "CIFAR-100"),
    "resnet_tiny": ("ResNet / Tiny ImageNet", "ResNet18", "Tiny ImageNet"),
    "preact_cifar10": ("PreAct / CIFAR-10", "PreActResNet18", "CIFAR-10"),
    "preact_cifar100": ("PreAct / CIFAR-100", "PreActResNet18", "CIFAR-100"),
}

# Wartości z tabeli głównej WYNIKI.md E037 (top-1 dev CMA, %; ilorazy U'/T).
# Tolerancja obejmuje zaokrąglenie tabeli do 2 miejsc (i 3 dla MAC).
EXPECTED = {
    ("resnet_cifar10", "b30"): (93.49, 94.52, 94.16, 0.37, 1.10, 0.998),
    ("resnet_cifar10", "b50"): (89.44, 93.43, 92.95, 0.48, 1.36, 0.992),
    ("resnet_cifar10", "b70"): (77.31, 88.21, 90.22, -2.01, 1.79, 0.985),
    ("resnet_cifar100", "b30"): (66.34, 66.96, 69.67, -2.71, 1.10, 0.998),
    ("resnet_cifar100", "b50"): (53.85, 55.55, 65.66, -10.11, 1.37, 0.994),
    ("resnet_cifar100", "b70"): (31.40, 33.02, 56.71, -23.69, 1.80, 0.995),
    ("resnet_tiny", "b30"): (34.81, 38.59, 39.91, -1.32, 1.11, 0.998),
    ("resnet_tiny", "b50"): (19.93, 29.33, 34.17, -4.84, 1.37, 0.988),
    ("resnet_tiny", "b70"): (8.43, 16.18, 23.24, -7.06, 1.82, 0.999),
    ("preact_cifar10", "b30"): (93.59, 94.58, 94.55, 0.03, 1.10, 0.998),
    ("preact_cifar10", "b50"): (89.00, 93.88, 93.14, 0.74, 1.36, 0.996),
    ("preact_cifar10", "b70"): (78.60, 91.79, 90.00, 1.79, 1.75, 0.993),
    ("preact_cifar100", "b30"): (66.87, 71.31, 72.12, -0.81, 1.10, 0.998),
    ("preact_cifar100", "b50"): (56.84, 64.07, 68.58, -4.52, 1.37, 0.998),
    ("preact_cifar100", "b70"): (23.43, 46.52, 57.46, -10.94, 1.77, 0.998),
}

CSV_FIELDS = (
    "pair_code",
    "pair",
    "architecture",
    "dataset",
    "budget",
    "profile",
    "seed",
    "top1_pct",
    "top1_mean_pct",
    "top1_sd_pp",
    "params_million",
    "mac_million",
    "params_pct_teacher",
    "mac_pct_teacher",
    "teacher_params_million",
    "teacher_mac_million",
    "row_type",
    "source",
    "source_sha256",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def pl(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.8,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.4,
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
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": None},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": f"E035/E037 {stem}",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def formatter(value, _position):
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.1f}".replace(".", ",")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT.name != "publikacja" or OUT.parent.name != "figures" or "E037" not in EXP.name:
        raise RuntimeError(f"Nieoczekiwany katalog wyjściowy: {OUT}")
    if RUN37.name != "20261108T031759Z_855c0b33":
        raise RuntimeError(f"Nieoczekiwany run E037: {RUN37}")
    if RUN35.name != "20261107T183831Z_8b0b14e0":
        raise RuntimeError(f"Nieoczekiwany run E035: {RUN35}")

    summary37 = [
        RUN37 / f"seed_{seed}/{pair}/{tag}/summary.json"
        for seed in SEEDS
        for pair in PAIRS
        for tag in TAGS
    ]
    summary35 = [
        RUN35 / f"dev/seed_{seed}/{pair}/{tag}/summary.json"
        for seed in SEEDS
        for pair in PAIRS
        for tag in TAGS
    ]
    controls35 = [
        RUN35 / f"dev/seed_{seed}/{pair}/controls.json"
        for seed in SEEDS
        for pair in PAIRS
    ]
    sources = [ANALYSIS, VERIFICATION, FROZEN, E036_CHECK, *summary35, *summary37, *controls35]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    hash_before = {str(path.resolve()): sha256(path) for path in sources}

    verification = read(VERIFICATION)
    if verification.get("status") != "passed":
        raise RuntimeError("E037 verification.json nie ma statusu passed")
    if not verification.get("dev_only") or verification.get("data_open_attempts") != 0:
        raise RuntimeError("E037 nie potwierdza dev-only / braku otwarcia testu")
    if verification.get("rows") != 180 or verification.get("controls") != 60:
        raise RuntimeError("Nieoczekiwana liczebność w verification.json")
    if verification.get("analysis_sha256") != sha256(ANALYSIS):
        raise RuntimeError("Hash analysis.json nie zgadza się z verification.json")
    if verification.get("frozen_sha256") != sha256(FROZEN):
        raise RuntimeError("Hash frozen.json nie zgadza się z verification.json")

    analysis = read(ANALYSIS)
    if analysis.get("rows_n") != 180 or len(analysis.get("rows", [])) != 180:
        raise RuntimeError("Nieoczekiwana liczba rows w analysis.json")
    if analysis.get("controls_n") != 60 or len(analysis.get("controls", [])) != 60:
        raise RuntimeError("Nieoczekiwana liczba controls w analysis.json")
    cma_rows = [row for row in analysis["rows"] if row["mode"] == "CMA"]
    if len(cma_rows) != 60:
        raise RuntimeError("Oczekiwano 60 wierszy CMA (5 par x 3 budżety x 4 seedy)")
    analysis_cma = {}
    for row in cma_rows:
        key = (row["pair"], row["tag"], row["seed"])
        if key in analysis_cma:
            raise RuntimeError(f"Zduplikowany wiersz CMA: {key}")
        analysis_cma[key] = row
    expected_keys = {
        (pair, tag, seed) for pair in PAIRS for tag in TAGS for seed in SEEDS
    }
    if set(analysis_cma) != expected_keys:
        raise RuntimeError("Niepełna siatka wierszy CMA w analysis.json")

    source_hashes = analysis.get("source_hashes", {})
    expected_source_keys = {
        f"seed_{seed}/{pair}/{tag}/summary.json"
        for pair in PAIRS
        for tag in TAGS
        for seed in SEEDS
    }
    if set(source_hashes) != expected_source_keys:
        raise RuntimeError("Niepełna lista source_hashes w analysis.json")
    for key, digest in source_hashes.items():
        path = RUN37 / key
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Hash summary E037 nie zgadza się: {key}")

    frozen = read(FROZEN)
    plans = frozen.get("plans", {})
    if set(plans) != set(PAIRS):
        raise RuntimeError("Niepełna lista par w frozen.json")
    frozen_hashes = frozen.get("hashes", {})
    for path in [*summary35, *controls35]:
        key = rel(path)
        if key not in frozen_hashes:
            raise RuntimeError(f"Brak hasha E035 w frozen.json: {key}")
        if frozen_hashes[key] != sha256(path):
            raise RuntimeError(f"Hash E035 nie zgadza się: {key}")

    e036 = read(E036_CHECK)
    if e036.get("status") != "passed" or e036.get("data_open_attempts") != []:
        raise RuntimeError("E036 check.json nie potwierdza poprawnego audytu kosztu")
    e036_rows = {(row["pair"], row["name"]): row for row in e036["checks"]}
    teacher_cost = {}
    for pair in PAIRS:
        row = e036_rows.get((pair, "teacher"))
        if row is None or row["params"] <= 0 or row["mac"] <= 0:
            raise RuntimeError(f"Brak poprawnych kosztów nauczyciela: {pair}")
        teacher_cost[pair] = (int(row["params"]), int(row["mac"]))

    teacher = {}
    for pair in PAIRS:
        values = []
        for seed in SEEDS:
            controls_path = RUN35 / f"dev/seed_{seed}/{pair}/controls.json"
            controls = read(controls_path)
            if set(controls.get("metrics", {})) != {"PRE", "CMA", "UP_REPAIR"}:
                raise RuntimeError(f"Niepełne kontrole teacher: {controls_path}")
            cm = controls["metrics"]["CMA"]
            if not all(isinstance(cm[key], (int, float)) for key in ("top1", "nll")):
                raise RuntimeError(f"Błędne metryki teacher: {controls_path}")
            value = float(cm["top1"]) * 100.0
            values.append(value)
            ref = next(
                (
                    row
                    for row in analysis["controls"]
                    if row["pair"] == pair and row["seed"] == seed and row["mode"] == "CMA"
                ),
                None,
            )
            if ref is None or abs(value - float(ref["top1"]) * 100.0) > 1e-12:
                raise RuntimeError(f"Niezgodność teacher CMA: {pair}/seed{seed}")
        teacher[pair] = {
            "mean": st.mean(values),
            "sd": st.stdev(values),
            "per_seed": values,
        }

    cells = {}
    max_analysis_diff = 0.0
    for pair in PAIRS:
        teacher_params, teacher_mac = teacher_cost[pair]
        for tag in TAGS:
            plan = plans[pair][tag]
            if not plan.get("mac_matched") or plan["Uprime_mac"] > plan["T_mac"]:
                raise RuntimeError(f"U' nie jest dopasowany MAC: {pair}/{tag}")
            if plan["Uprime_params"] <= plan["T_params"]:
                raise RuntimeError(f"U' nie ma więcej parametrów: {pair}/{tag}")
            e36_u = e036_rows[(pair, f"{tag}_U")]
            e36_t = e036_rows[(pair, f"{tag}_T")]
            if e36_t["params"] != plan["T_params"] or e36_t["mac"] != plan["T_mac"]:
                raise RuntimeError(f"Niezgodność T z E036: {pair}/{tag}")
            if e36_u["mac"] != plan["U_mac"]:
                raise RuntimeError(f"Niezgodność MAC U z E036: {pair}/{tag}")

            per_seed = []
            controls_hash = None
            for seed in SEEDS:
                path35 = RUN35 / f"dev/seed_{seed}/{pair}/{tag}/summary.json"
                path37 = RUN37 / f"seed_{seed}/{pair}/{tag}/summary.json"
                controls_path = RUN35 / f"dev/seed_{seed}/{pair}/controls.json"
                controls_hash = sha256(controls_path)
                s35 = read(path35)
                s37 = read(path37)
                if s35.get("status") != "completed" or s35.get("phase") != "dev":
                    raise RuntimeError(f"E035 nie jest ukończonym dev: {path35}")
                if s37.get("status") != "completed" or s37.get("phase") != "dev":
                    raise RuntimeError(f"E037 nie jest ukończonym dev: {path37}")
                if (s35["pair"], s35["tag"], s35["seed"]) != (pair, tag, seed):
                    raise RuntimeError(f"Błędne przypisanie E035: {path35}")
                if (s37["pair"], s37["tag"], s37["seed"]) != (pair, tag, seed):
                    raise RuntimeError(f"Błędne przypisanie E037: {path37}")
                if s37.get("official_test_used") is not False:
                    raise RuntimeError(f"E037 oznacza użycie testu: {path37}")
                if s37["plan"] != plan:
                    raise RuntimeError(f"Plan w summary E037 różni się od frozen.json: {path37}")
                if s35["plan"]["params"]["T"] != plan["T_params"]:
                    raise RuntimeError(f"Niezgodność parametrów T: {path35}")
                if e36_u["params"] != s35["plan"]["params"]["U"]:
                    raise RuntimeError(f"Niezgodność parametrów U z E036: {pair}/{tag}")
                if s35["controls_sha256"] != controls_hash or s37["controls_sha256"] != controls_hash:
                    raise RuntimeError(f"Niezgodność controls_sha256: {pair}/{tag}/seed{seed}")

                values = {
                    "U": float(s35["metrics"]["U"]["CMA"]["top1"]) * 100.0,
                    "T": float(s35["metrics"]["T"]["CMA"]["top1"]) * 100.0,
                    "Up": float(s37["metrics"]["CMA"]["top1"]) * 100.0,
                }
                ref = analysis_cma[(pair, tag, seed)]
                for profile in PROFILES:
                    diff = abs(values[profile] - float(ref[profile]) * 100.0)
                    max_analysis_diff = max(max_analysis_diff, diff)
                    if diff > 1e-12:
                        raise RuntimeError(
                            f"Niezgodność summary z analysis.json: {pair}/{tag}/seed{seed}/{profile}"
                        )
                per_seed.append(
                    {
                        "seed": seed,
                        "top1_U": values["U"],
                        "top1_T": values["T"],
                        "top1_Up": values["Up"],
                        "params_U": int(s35["plan"]["params"]["U"]),
                        "params_T": int(plan["T_params"]),
                        "params_Up": int(plan["Uprime_params"]),
                        "mac_U": int(plan["U_mac"]),
                        "mac_T": int(plan["T_mac"]),
                        "mac_Up": int(plan["Uprime_mac"]),
                    }
                )

            for key in (
                "params_U",
                "params_T",
                "params_Up",
                "mac_U",
                "mac_T",
                "mac_Up",
            ):
                values = {row[key] for row in per_seed}
                if len(values) != 1:
                    raise RuntimeError(f"Koszt {key} zależy od seeda: {pair}/{tag}")

            cell = {
                "pair": pair,
                "tag": tag,
                "per_seed": per_seed,
                "teacher_params": teacher_params,
                "teacher_mac": teacher_mac,
                "teacher_top1": teacher[pair]["mean"],
                "teacher_sd": teacher[pair]["sd"],
            }
            for key in (
                "params_U",
                "params_T",
                "params_Up",
                "mac_U",
                "mac_T",
                "mac_Up",
            ):
                cell[key] = per_seed[0][key]
            for profile in PROFILES:
                values = [row[f"top1_{profile}"] for row in per_seed]
                cell[f"top1_{profile}_mean"] = st.mean(values)
                cell[f"top1_{profile}_sd"] = st.stdev(values)
            cells[(pair, tag)] = cell

    # Zgodność z tabelą WYNIKI.md E037 (top-1 i ilorazy kosztów).
    max_wyniki_diff = 0.0
    max_ratio_param_diff = 0.0
    max_ratio_mac_diff = 0.0
    for (pair, tag), expected in EXPECTED.items():
        cell = cells[(pair, tag)]
        for profile, value in zip(PROFILES, expected[:3]):
            diff = abs(cell[f"top1_{profile}_mean"] - value)
            max_wyniki_diff = max(max_wyniki_diff, diff)
            if diff > 0.0051:
                raise RuntimeError(
                    f"Niezgodność z WYNIKI.md: {pair}/{tag}/{profile} "
                    f"{cell[f'top1_{profile}_mean']:.4f} vs {value:.2f}"
                )
        diff = abs((cell["top1_T_mean"] - cell["top1_Up_mean"]) - expected[3])
        max_wyniki_diff = max(max_wyniki_diff, diff)
        if diff > 0.0051:
            raise RuntimeError(f"Niezgodność T-U' z WYNIKI.md: {pair}/{tag}")
        param_ratio = cell["params_Up"] / cell["params_T"]
        mac_ratio = cell["mac_Up"] / cell["mac_T"]
        diff = abs(round(param_ratio, 2) - expected[4])
        max_ratio_param_diff = max(max_ratio_param_diff, diff)
        if diff > 0.0051:
            raise RuntimeError(f"Niezgodność ilorazu parametrów z WYNIKI.md: {pair}/{tag}")
        diff = abs(round(mac_ratio, 3) - expected[5])
        max_ratio_mac_diff = max(max_ratio_mac_diff, diff)
        if diff > 0.00051:
            raise RuntimeError(f"Niezgodność ilorazu MAC z WYNIKI.md: {pair}/{tag}")

    # Sparowane seedy: allocation CMA T-Up w analysis.json.
    allocation = {
        (row["pair"], row["tag"]): row
        for row in analysis["allocation"]
        if row["mode"] == "CMA" and row["contrast"] == "T-Up"
    }
    if len(allocation) != 15:
        raise RuntimeError("Oczekiwano 15 kontrastów CMA T-Up")
    max_alloc_diff = 0.0
    max_alloc_sd_diff = 0.0
    for pair in PAIRS:
        for tag in TAGS:
            deltas = [row["top1_T"] - row["top1_Up"] for row in cells[(pair, tag)]["per_seed"]]
            mean = st.mean(deltas)
            sd = st.stdev(deltas)
            ref = allocation[(pair, tag)]
            max_alloc_diff = max(max_alloc_diff, abs(mean - ref["mean"]))
            max_alloc_sd_diff = max(max_alloc_sd_diff, abs(sd - ref["sd"]))
            if abs(mean - ref["mean"]) > 1e-9 or abs(sd - ref["sd"]) > 1e-9:
                raise RuntimeError(f"Niezgodność allocation: {pair}/{tag}")
            if any(abs(a - b) > 1e-9 for a, b in zip(deltas, ref["per_seed"])):
                raise RuntimeError(f"Niezgodność per_seed allocation: {pair}/{tag}")

    for pair in PAIRS:
        teacher_top1 = teacher[pair]["mean"]
        for tag in TAGS:
            cell = cells[(pair, tag)]
            if teacher_top1 <= max(cell[f"top1_{profile}_mean"] for profile in PROFILES):
                raise RuntimeError(f"Teacher nie jest najlepszy w parze: {pair}")

    # Rysunek: 5 par x 2 panele, oś x = koszt / koszt nauczyciela.
    style()
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(7.4, 8.6),
        sharex="col",
        sharey="row",
    )
    fig.subplots_adjust(left=0.165, right=0.985, top=0.900, bottom=0.105, hspace=0.30, wspace=0.10)

    for index, pair in enumerate(PAIRS):
        ax_p = axes[index, 0]
        ax_m = axes[index, 1]
        all_values = []
        for tag in TAGS:
            cell = cells[(pair, tag)]
            all_values.extend(
                cell[f"top1_{profile}_mean"] for profile in PROFILES
            )
        all_values.append(teacher[pair]["mean"])
        lo, hi = min(all_values), max(all_values)
        span = max(hi - lo, 1.0)
        for ax in (ax_p, ax_m):
            ax.set_ylim(lo - 0.06 * span, hi + 0.06 * span)
            ax.set_xlim(24, 102)
            ax.set_xticks([30, 50, 70, 90, 100])
            ax.xaxis.set_major_formatter(FuncFormatter(formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))
            ax.axvline(100, color="0.82", lw=0.45, ls=":", zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=2.2, width=0.6)

        # Osobno parametr i MAC, aby uniknąć pomyłki w mianowniku.
        for profile in PROFILES:
            meta = PROFILE_META[profile]
            teacher_p = teacher_cost[pair][0]
            teacher_m = teacher_cost[pair][1]
            xs_p = [100.0 * cells[(pair, tag)][f"params_{profile}"] / teacher_p for tag in TAGS]
            xs_m = [100.0 * cells[(pair, tag)][f"mac_{profile}"] / teacher_m for tag in TAGS]
            ys = [cells[(pair, tag)][f"top1_{profile}_mean"] for tag in TAGS]
            es = [cells[(pair, tag)][f"top1_{profile}_sd"] for tag in TAGS]
            for ax, xs in ((ax_p, xs_p), (ax_m, xs_m)):
                ax.plot(xs, ys, color=meta["color"], ls=meta["ls"], lw=1.0, zorder=2)
                for tag, x, y, e in zip(TAGS, xs, ys, es):
                    ax.errorbar(
                        x,
                        y,
                        yerr=e,
                        fmt=BUDGET_MARKERS[tag],
                        color=meta["color"],
                        ms=3.6,
                        capsize=1.4,
                        elinewidth=0.55,
                        capthick=0.55,
                        zorder=3,
                    )

        for ax in (ax_p, ax_m):
            ax.errorbar(
                100.0,
                teacher[pair]["mean"],
                yerr=teacher[pair]["sd"],
                fmt="*",
                color="0.55",
                mec="0.25",
                mew=0.4,
                ms=7.0,
                capsize=1.4,
                elinewidth=0.55,
                capthick=0.55,
                zorder=4,
            )

        if index == 0:
            ax_p.set_title("Top-1 vs P", pad=5)
            ax_m.set_title("Top-1 vs MAC", pad=5)
        ax_p.set_ylabel(LABELS[pair][0], labelpad=3, fontsize=7.0)
        if index == 4:
            ax_p.set_xlabel("P: % nauczyciela", labelpad=4)
            ax_m.set_xlabel("MAC: % nauczyciela", labelpad=4)
        if index < 4:
            ax_p.tick_params(labelbottom=False)
            ax_m.tick_params(labelbottom=False)

        teacher_p = teacher_cost[pair][0]
        teacher_m = teacher_cost[pair][1]
        ax_p.text(
            0.025,
            0.965,
            f"P_t={pl(teacher_p / 1e6, 1)} mln\nMAC_t={pl(teacher_m / 1e6, 0)} mln",
            transform=ax_p.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            color="0.45",
            linespacing=1.05,
        )

    fig.text(
        0.022,
        0.50,
        "Top-1 dev [%] po CMA",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
    )
    handles = [
        Line2D([0], [0], color=PROFILE_META[profile]["color"], ls=PROFILE_META[profile]["ls"], lw=1.2, label=PROFILE_META[profile]["label"])
        for profile in PROFILES
    ]
    handles.extend(
        Line2D([0], [0], marker=BUDGET_MARKERS[tag], color="0.25", ls="None", ms=4.0, label=f"{BUDGET[tag]}%")
        for tag in TAGS
    )
    handles.append(
        Line2D([0], [0], marker="*", color="0.55", mec="0.25", mew=0.4, ls="None", ms=8.0, label="nauczyciel")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=7,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.text(
        0.5,
        0.008,
        "Średnia z 4 seedów; słupki = SD. Dev, po CMA. Oś x: zmierzony P/MAC podzielony przez koszt nauczyciela; "
        "surowe P/MAC w punkty_publikacja.csv. Osie y mają różne zakresy dla par.",
        ha="center",
        fontsize=6.1,
        color="0.25",
    )
    outputs = list(save(fig, "e035_e037_jakosc_p_mac"))

    # CSV: punkty per seed i średnie, plus odniesienie nauczyciela.
    csv_rows = []
    for pair in PAIRS:
        pair_label, arch, dataset = LABELS[pair]
        teacher_params, teacher_mac = teacher_cost[pair]
        for tag in TAGS:
            cell = cells[(pair, tag)]
            for profile in PROFILES:
                for row in cell["per_seed"]:
                    csv_rows.append(
                        {
                            "pair_code": pair,
                            "pair": pair_label,
                            "architecture": arch,
                            "dataset": dataset,
                            "budget": BUDGET[tag],
                            "profile": PROFILE_NAMES[profile],
                            "seed": row["seed"],
                            "top1_pct": f"{row[f'top1_{profile}']:.6f}",
                            "top1_mean_pct": f"{cell[f'top1_{profile}_mean']:.6f}",
                            "top1_sd_pp": f"{cell[f'top1_{profile}_sd']:.6f}",
                            "params_million": f"{cell[f'params_{profile}'] / 1e6:.6f}",
                            "mac_million": f"{cell[f'mac_{profile}'] / 1e6:.6f}",
                            "params_pct_teacher": f"{100.0 * cell[f'params_{profile}'] / teacher_params:.6f}",
                            "mac_pct_teacher": f"{100.0 * cell[f'mac_{profile}'] / teacher_mac:.6f}",
                            "teacher_params_million": f"{teacher_params / 1e6:.6f}",
                            "teacher_mac_million": f"{teacher_mac / 1e6:.6f}",
                            "row_type": "seed",
                            "source": rel(
                                RUN35 / f"dev/seed_{row['seed']}/{pair}/{tag}/summary.json"
                                if profile in ("U", "T")
                                else RUN37 / f"seed_{row['seed']}/{pair}/{tag}/summary.json"
                            ),
                            "source_sha256": (
                                sha256(RUN35 / f"dev/seed_{row['seed']}/{pair}/{tag}/summary.json")
                                if profile in ("U", "T")
                                else sha256(RUN37 / f"seed_{row['seed']}/{pair}/{tag}/summary.json")
                            ),
                        }
                    )
                csv_rows.append(
                    {
                        "pair_code": pair,
                        "pair": pair_label,
                        "architecture": arch,
                        "dataset": dataset,
                        "budget": BUDGET[tag],
                        "profile": PROFILE_NAMES[profile],
                        "seed": "",
                        "top1_pct": f"{cell[f'top1_{profile}_mean']:.6f}",
                        "top1_mean_pct": f"{cell[f'top1_{profile}_mean']:.6f}",
                        "top1_sd_pp": f"{cell[f'top1_{profile}_sd']:.6f}",
                        "params_million": f"{cell[f'params_{profile}'] / 1e6:.6f}",
                        "mac_million": f"{cell[f'mac_{profile}'] / 1e6:.6f}",
                        "params_pct_teacher": f"{100.0 * cell[f'params_{profile}'] / teacher_params:.6f}",
                        "mac_pct_teacher": f"{100.0 * cell[f'mac_{profile}'] / teacher_mac:.6f}",
                        "teacher_params_million": f"{teacher_params / 1e6:.6f}",
                        "teacher_mac_million": f"{teacher_mac / 1e6:.6f}",
                        "row_type": "mean",
                        "source": "E035 dev summary (4 seedy)" if profile in ("U", "T") else "E037 summary (4 seedy)",
                        "source_sha256": "",
                    }
                )
        for seed, value in zip(SEEDS, teacher[pair]["per_seed"]):
            csv_rows.append(
                {
                    "pair_code": pair,
                    "pair": pair_label,
                    "architecture": arch,
                    "dataset": dataset,
                    "budget": 0,
                    "profile": "teacher",
                    "seed": seed,
                    "top1_pct": f"{value:.6f}",
                    "top1_mean_pct": f"{teacher[pair]['mean']:.6f}",
                    "top1_sd_pp": f"{teacher[pair]['sd']:.6f}",
                    "params_million": f"{teacher_params / 1e6:.6f}",
                    "mac_million": f"{teacher_mac / 1e6:.6f}",
                    "params_pct_teacher": "100.000000",
                    "mac_pct_teacher": "100.000000",
                    "teacher_params_million": f"{teacher_params / 1e6:.6f}",
                    "teacher_mac_million": f"{teacher_mac / 1e6:.6f}",
                    "row_type": "seed",
                    "source": rel(RUN35 / f"dev/seed_{seed}/{pair}/controls.json"),
                    "source_sha256": sha256(RUN35 / f"dev/seed_{seed}/{pair}/controls.json"),
                }
            )
        csv_rows.append(
            {
                "pair_code": pair,
                "pair": pair_label,
                "architecture": arch,
                "dataset": dataset,
                "budget": 0,
                "profile": "teacher",
                "seed": "",
                "top1_pct": f"{teacher[pair]['mean']:.6f}",
                "top1_mean_pct": f"{teacher[pair]['mean']:.6f}",
                "top1_sd_pp": f"{teacher[pair]['sd']:.6f}",
                "params_million": f"{teacher_params / 1e6:.6f}",
                "mac_million": f"{teacher_mac / 1e6:.6f}",
                "params_pct_teacher": "100.000000",
                "mac_pct_teacher": "100.000000",
                "teacher_params_million": f"{teacher_params / 1e6:.6f}",
                "teacher_mac_million": f"{teacher_mac / 1e6:.6f}",
                "row_type": "mean",
                "source": "E035 dev controls (4 seedy)",
                "source_sha256": "",
            }
        )
    csv_path = OUT / "punkty_publikacja.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    outputs.append(csv_path)

    # Zabezpieczenie: wyniki tylko w tym katalogu, wejścia bez zmian.
    source_set = {Path(path).resolve() for path in sources}
    for path in outputs:
        resolved = Path(path).resolve()
        if OUT.resolve() not in resolved.parents:
            raise RuntimeError(f"Wynik poza publikacja/: {resolved}")
        if resolved in source_set:
            raise RuntimeError(f"Wynik nadpisałby wejście: {resolved}")
    hash_after = {str(path.resolve()): sha256(path) for path in sources}
    if hash_before != hash_after:
        raise RuntimeError("Plik wejściowy zmienił się podczas generowania")

    outputs_meta = [
        {
            "sciezka": rel(path),
            "sha256": sha256(path),
            "bajty": Path(path).stat().st_size,
        }
        for path in outputs
    ]
    sources_doc = {
        "eksperyment": "E035/E037 (dev, po CMA)",
        "opis": "Figura jakość-P/MAC dla U, T i U' z porównywalnego dev E035/E037; koszty P/MAC z planów E037 i audytu E036.",
        "runy": {
            "E035": {
                "id": RUN35.name,
                "sciezka": rel(RUN35),
                "faza": "dev",
                "rola": "U i T po CMA; teacher CMA",
            },
            "E037": {
                "id": RUN37.name,
                "sciezka": rel(RUN37),
                "faza": "dev",
                "rola": "U' po CMA; plany P/MAC",
            },
            "E036": {
                "id": "audyt kosztu",
                "sciezka": rel(E036_CHECK),
                "rola": "kontrolne P/MAC nauczyciela i U/T",
            },
        },
        "skrypt": {
            "sciezka": rel(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "zrodla": [
            {
                "sciezka": rel(path),
                "sha256": sha256(path),
                "bajty": path.stat().st_size,
                "rola": (
                    "analysis E037"
                    if path == ANALYSIS
                    else "verification E037"
                    if path == VERIFICATION
                    else "frozen E037"
                    if path == FROZEN
                    else "E036 check (koszt teacher/U/T)"
                    if path == E036_CHECK
                    else "E035 dev summary (U/T)"
                    if "/dev/" in rel(path) and path.name == "summary.json"
                    else "E035 dev controls (teacher)"
                    if path.name == "controls.json"
                    else "E037 summary (U')"
                ),
            }
            for path in sorted(sources, key=rel)
        ],
        "wyjscia": outputs_meta,
        "zabezpieczenie": {
            "wejscia_niezmienione": True,
            "sprawdzone_po_generowaniu": True,
            "wyniki_tylko_w_publikacja": True,
            "bez_gpu_treningu_danych_checkpointow": True,
        },
    }
    (OUT / "sources.json").write_text(
        json.dumps(sources_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def top1_range(pair):
        values = [
            cells[(pair, tag)][f"top1_{profile}_mean"]
            for tag in TAGS
            for profile in PROFILES
        ]
        values.append(teacher[pair]["mean"])
        return [min(values), max(values)]

    verification_doc = {
        "eksperyment": "E035/E037 (dev, po CMA)",
        "zakres": "dev only; test nie był otwierany",
        "runy": {
            "E035": RUN35.name,
            "E037": RUN37.name,
            "E036_check": rel(E036_CHECK),
        },
        "seedy": list(SEEDS),
        "pary": [LABELS[pair][0] for pair in PAIRS],
        "komorki": len(PAIRS) * len(TAGS),
        "wiersze_analysis_CMA": len(cma_rows),
        "punkty_per_seed": len(PAIRS) * len(TAGS) * len(SEEDS),
        "kontroli_teacher": len(PAIRS) * len(SEEDS),
        "kompletnosc": {
            "summary_E035": len(summary35),
            "summary_E037": len(summary37),
            "controls_E035": len(controls35),
            "pary": len(PAIRS),
            "budzety": list(TAGS),
        },
        "dev_only": bool(verification.get("dev_only")),
        "data_open_attempts": verification.get("data_open_attempts"),
        "official_test_used_wszystkie_false": True,
        "zgodnosc_z_analysis_json": {
            "max_abs_diff_top1_pp": max_analysis_diff,
            "max_abs_diff_sredniej_T_minus_Up_pp": max_alloc_diff,
            "max_abs_diff_sd_T_minus_Up_pp": max_alloc_sd_diff,
        },
        "zgodnosc_z_WYNIKI_md": {
            "max_abs_diff_top1_pp": max_wyniki_diff,
            "max_abs_diff_ilorazu_parametrow": max_ratio_param_diff,
            "max_abs_diff_ilorazu_mac": max_ratio_mac_diff,
            "tolerancja_pp": 0.0051,
        },
        "zgodnosc_hashy": {
            "analysis_sha256": sha256(ANALYSIS),
            "verification_analysis_sha256": verification.get("analysis_sha256"),
            "frozen_sha256": sha256(FROZEN),
            "verification_frozen_sha256": verification.get("frozen_sha256"),
            "source_hashes_E037": len(source_hashes),
            "frozen_hashes_E035_summary_i_controls": len(summary35) + len(controls35),
            "wszystkie_zgodne": True,
        },
        "koszty": {
            "teacher_params_mln_zakres": [
                min(teacher_cost[pair][0] for pair in PAIRS) / 1e6,
                max(teacher_cost[pair][0] for pair in PAIRS) / 1e6,
            ],
            "teacher_mac_mln_zakres": [
                min(teacher_cost[pair][1] for pair in PAIRS) / 1e6,
                max(teacher_cost[pair][1] for pair in PAIRS) / 1e6,
            ],
            "params_pct_teacher_zakres": [
                min(
                    100.0 * cells[(pair, tag)][f"params_{profile}"] / teacher_cost[pair][0]
                    for pair in PAIRS
                    for tag in TAGS
                    for profile in PROFILES
                ),
                max(
                    100.0 * cells[(pair, tag)][f"params_{profile}"] / teacher_cost[pair][0]
                    for pair in PAIRS
                    for tag in TAGS
                    for profile in PROFILES
                ),
            ],
            "mac_pct_teacher_zakres": [
                min(
                    100.0 * cells[(pair, tag)][f"mac_{profile}"] / teacher_cost[pair][1]
                    for pair in PAIRS
                    for tag in TAGS
                    for profile in PROFILES
                ),
                max(
                    100.0 * cells[(pair, tag)][f"mac_{profile}"] / teacher_cost[pair][1]
                    for pair in PAIRS
                    for tag in TAGS
                    for profile in PROFILES
                ),
            ],
        },
        "top1_zakres_per_para": {
            LABELS[pair][0]: top1_range(pair) for pair in PAIRS
        },
        "zabezpieczenie_wejsc": {
            "liczba_plikow": len(sources),
            "hash_zgodny_przed_i_po": True,
        },
    }
    (OUT / "weryfikacja.json").write_text(
        json.dumps(verification_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("E035/E037: zapisano", len(outputs) + 2, "plików w", OUT)
    print("komorki", len(PAIRS) * len(TAGS), "punkty per-seed", len(PAIRS) * len(TAGS) * len(SEEDS))
    print(
        "P% nauczyciela",
        f"{verification_doc['koszty']['params_pct_teacher_zakres'][0]:.2f}–"
        f"{verification_doc['koszty']['params_pct_teacher_zakres'][1]:.2f}",
        "MAC% nauczyciela",
        f"{verification_doc['koszty']['mac_pct_teacher_zakres'][0]:.2f}–"
        f"{verification_doc['koszty']['mac_pct_teacher_zakres'][1]:.2f}",
    )
    print(
        "max diff analysis/WYNIKI",
        f"{max_analysis_diff:.2e}",
        f"{max_wyniki_diff:.2e}",
    )


if __name__ == "__main__":
    main()
