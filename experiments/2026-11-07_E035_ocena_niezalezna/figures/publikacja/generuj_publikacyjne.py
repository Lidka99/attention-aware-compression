#!/usr/bin/env python3
"""E035: ilustracje publikacyjne z metadanych niezależnej oceny (CPU only).

Wejście: E035/frozen.json oraz metadane i metryki runu
runs/E035/20261107T183831Z_8b0b14e0 (test: summary.json, controls.json,
analysis.json, manifest.json, test_data.json). Nie czytamy obrazów ani
checkpointów. Wyjście: PNG/PDF, CSV, weryfikacja.json, sources.json i README.md
wyłącznie w tym podkatalogu. --english-paper zapisuje tylko parę PDF/PNG
do manuskryptu. Opis: OPIS.md i WYNIKI.md E035.
"""
import argparse
import csv
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

OUT = Path(__file__).resolve().parent
E035 = OUT.parents[1]
ROOT = OUT.parents[3]
RUN = ROOT / "runs/E035/20261107T183831Z_8b0b14e0"
FROZEN = E035 / "frozen.json"

SEEDS = (42, 43, 44, 45)
PAIRS = (
    "resnet_cifar10",
    "resnet_cifar100",
    "resnet_tiny",
    "preact_cifar10",
    "preact_cifar100",
)
TAGS = ("b30", "b50", "b70")
PCT = {"b30": 30, "b50": 50, "b70": 70}
PROFILES = ("U", "T")
MODES = ("PRE", "CMA", "UP_REPAIR")

LABELS = {
    "resnet_cifar10": "ResNet18 / CIFAR-10",
    "resnet_cifar100": "ResNet18 / CIFAR-100",
    "resnet_tiny": "ResNet18 / Tiny ImageNet",
    "preact_cifar10": "PreActResNet18 / CIFAR-10",
    "preact_cifar100": "PreActResNet18 / CIFAR-100",
}
SHORT = {
    "resnet_cifar10": "R/C10",
    "resnet_cifar100": "R/C100",
    "resnet_tiny": "R/Tiny",
    "preact_cifar10": "P/C10",
    "preact_cifar100": "P/C100",
}
SPLIT = {pair: ("oficjalny val (cache_val)" if pair == "resnet_tiny" else "oficjalny test") for pair in PAIRS}
SPLIT_SHORT = {pair: ("val" if pair == "resnet_tiny" else "test") for pair in PAIRS}

# Wartości z tabel WYNIKI.md E035; tolerancja uwzględnia zaokrąglenie do 4 miejsc.
EXPECTED_DELTA = {
    ("resnet_cifar10", "b30"): 0.8100,
    ("resnet_cifar10", "b50"): 3.4125,
    ("resnet_cifar10", "b70"): 10.9875,
    ("resnet_cifar100", "b30"): 0.8275,
    ("resnet_cifar100", "b50"): 1.3025,
    ("resnet_cifar100", "b70"): 2.0600,
    ("resnet_tiny", "b30"): 3.5375,
    ("resnet_tiny", "b50"): 9.3600,
    ("resnet_tiny", "b70"): 7.4825,
    ("preact_cifar10", "b30"): 0.9600,
    ("preact_cifar10", "b50"): 4.9500,
    ("preact_cifar10", "b70"): 13.5775,
    ("preact_cifar100", "b30"): 4.4875,
    ("preact_cifar100", "b50"): 7.6425,
    ("preact_cifar100", "b70"): 22.7375,
}
EXPECTED_TOP1 = {
    ("resnet_cifar10", "b30"): (93.0400, 93.8500),
    ("resnet_cifar10", "b50"): (89.2800, 92.6925),
    ("resnet_cifar10", "b70"): (76.7925, 87.7800),
    ("resnet_cifar100", "b30"): (65.7675, 66.5950),
    ("resnet_cifar100", "b50"): (53.4275, 54.7300),
    ("resnet_cifar100", "b70"): (30.6200, 32.6800),
    ("resnet_tiny", "b30"): (34.5575, 38.0950),
    ("resnet_tiny", "b50"): (19.6400, 29.0000),
    ("resnet_tiny", "b70"): (8.3975, 15.8800),
    ("preact_cifar10", "b30"): (93.1275, 94.0875),
    ("preact_cifar10", "b50"): (88.5400, 93.4900),
    ("preact_cifar10", "b70"): (77.6875, 91.2650),
    ("preact_cifar100", "b30"): (66.4975, 70.9850),
    ("preact_cifar100", "b50"): (55.9050, 63.5475),
    ("preact_cifar100", "b70"): (23.0050, 45.7425),
}
EXPECTED_TEACHER = {
    "resnet_cifar10": 94.8050,
    "resnet_cifar100": 74.3725,
    "resnet_tiny": 50.7775,
    "preact_cifar10": 94.9025,
    "preact_cifar100": 76.4700,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_p(params):
    return f"{params / 1e6:.3f}M"


def group_short(first_axis):
    stem = first_axis.removesuffix(".weight")
    if stem == "conv1":
        return "conv1"
    return stem.replace("layer", "L").replace(".conv", ".c")


def group_permutation(arch, first_axis):
    """Nazwa osi permutacji upstream; first_axis to pierwszy parametr grupy."""
    stem = first_axis.removesuffix(".weight")
    if stem == "conv1":
        return "relu_conv"
    parts = stem.split(".")
    assert parts[0].startswith("layer") and len(parts) == 3, first_axis
    block = f"{parts[0]}.{parts[1]}"
    conv = parts[2]
    if arch == "resnet":
        if conv == "conv1":
            return f"{block}.relu1"
        assert conv == "conv2" and not block.startswith("layer1."), first_axis
        return f"{block}.relu2"
    assert arch == "preact", arch
    if conv == "conv1":
        return f"{block}.relu1"
    assert conv == "conv2", first_axis
    return f"relu{block.split('.')[0][5:]}_out"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 8.8,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 7.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem, english=False):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300 if english else 200, bbox_inches="tight", facecolor="white", metadata={"Software": None})
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "E035 manuscript figure" if english else "E035 ilustracje publikacyjne",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def draw_paper_delta(summaries):
    style()
    plt.rcParams.update({"axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.25, top=0.88, wspace=0.19)
    positions = np.arange(len(PAIRS))
    offsets = np.array((-0.36, -0.19, 0.19, 0.36))
    for ax, tag in zip(axes, TAGS):
        ax.set_title(f"{PCT[tag]}% parameter cut", pad=6, fontweight="bold")
        ax.set(xlim=(-1.2, 28.2), xticks=(0, 10, 20), ylim=(4.56, -0.56))
        ax.set_yticks(positions, [LABELS[pair].replace(" / ", "\n") for pair in PAIRS])
        ax.axvline(0, color="0.40", linestyle="--", linewidth=0.9)
        for y, pair in zip(positions, PAIRS):
            d = summaries[pair, tag]["delta"]
            ax.scatter(d, y + offsets, color="#0072B2", s=17, alpha=0.78, linewidths=0, zorder=3,
                       label="Checkpoint difference" if y == 0 else None)
            ax.scatter(d.mean(), y, color="#D55E00", marker="D", s=25, edgecolors="white",
                       linewidths=0.5, zorder=4, label="Mean" if y == 0 else None)
        ax.tick_params(length=2.5, axis="x")
        ax.tick_params(length=0, axis="y")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", visible=False)
    fig.supxlabel("Linear Depth Tilt minus uniform folding (top-1, pp)", y=0.116, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005),
               frameon=False, fontsize=9, columnspacing=1.3)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-paper", action="store_true", help="Wygeneruj angielską figurę manuskryptu.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT.name != "publikacja" or OUT.parent.name != "figures" or "E035" not in E035.name:
        raise RuntimeError(f"Nieoczekiwany katalog wyjściowy: {OUT}")
    if RUN.name != "20261107T183831Z_8b0b14e0":
        raise RuntimeError(f"Nieoczekiwany run E035: {RUN}")

    analysis_path = RUN / "analysis/analysis.json"
    per_seed_path = RUN / "analysis/per_seed.csv"
    manifest_path = RUN / "manifest.json"
    test_data_path = RUN / "test_data.json"
    summary_paths = [
        RUN / f"test/seed_{seed}/{pair}/{tag}/summary.json"
        for pair in PAIRS
        for tag in TAGS
        for seed in SEEDS
    ]
    controls_paths = [RUN / f"test/seed_{seed}/{pair}/controls.json" for pair in PAIRS for seed in SEEDS]
    inputs = [FROZEN, analysis_path, per_seed_path, manifest_path, test_data_path, *summary_paths, *controls_paths]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}

    frozen = read(FROZEN)
    if frozen != read(RUN / "frozen.json"):
        raise RuntimeError("E035/frozen.json różni się od frozen.json runu")
    manifest = read(manifest_path)
    if manifest["status"] != "completed" or manifest["phases"]["test"]["exit_codes"] != [0, 0, 0, 0]:
        raise RuntimeError("Manifest E035 nie potwierdza ukończonego testu na 4 GPU")
    if manifest["frozen_sha256"] != input_hashes[str(FROZEN.relative_to(ROOT))]:
        raise RuntimeError("Hash frozen.json nie zgadza się z manifestem")
    test_data = read(test_data_path)
    if test_data["cifar100_reserve_used"] is not False:
        raise RuntimeError("Rezerwa CIFAR-100 została oznaczona jako użyta")

    plans = frozen["plans"]
    teacher = {}
    group_meta = {}
    cells = {}
    summary_count = 0
    for pair in PAIRS:
        arch = pair.split("_")[0]
        for seed in SEEDS:
            controls_path = RUN / f"test/seed_{seed}/{pair}/controls.json"
            controls = read(controls_path)
            if set(controls["metrics"]) != set(MODES):
                raise RuntimeError(f"Niepełne kontrole: {controls_path}")
            cm = controls["metrics"]["CMA"]
            if not np.isfinite([cm["top1"], cm["nll"]]).all():
                raise RuntimeError(f"Nieskończone metryki teacher: {controls_path}")
            teacher[(pair, seed)] = dict(top1=float(cm["top1"]), nll=float(cm["nll"]))
        for tag in TAGS:
            plan = plans[pair][tag]
            records = []
            for seed in SEEDS:
                path = RUN / f"test/seed_{seed}/{pair}/{tag}/summary.json"
                summary = read(path)
                if summary["status"] != "completed" or summary["phase"] != "test":
                    raise RuntimeError(f"Nieukończona faza test: {path}")
                if (summary["pair"], summary["tag"], summary["seed"], summary["gpu"]) != (pair, tag, seed, seed - 42):
                    raise RuntimeError(f"Błędne przypisanie seed/GPU: {path}")
                if summary["plan"] != plan:
                    raise RuntimeError(f"Plan w summary różni się od frozen.json: {path}")
                if sha256(RUN / f"test/seed_{seed}/{pair}/controls.json") != summary["controls_sha256"]:
                    raise RuntimeError(f"controls_sha256 nie zgadza się: {path}")
                clusters = summary["clusters"]
                if set(clusters) != {"U", "T"}:
                    raise RuntimeError(f"Brak profili U/T: {path}")
                width = len(clusters["U"])
                if width != len(clusters["T"]) or width not in (12, 13):
                    raise RuntimeError(f"Błędna liczba grup: {path}")
                meta = []
                for index, (gu, gt) in enumerate(zip(clusters["U"], clusters["T"])):
                    if gu["n"] != gt["n"] or gu["first_axis"] != gt["first_axis"]:
                        raise RuntimeError(f"Niespójna geometria U/T: {path}")
                    if gu["k"] != plan["U"][index] or gt["k"] != plan["T"][index]:
                        raise RuntimeError(f"Szerokości klastrów różnią się od frozen.json: {path}")
                    if not (0 < gu["k"] <= gu["n"]) or not (0 < gt["k"] <= gt["n"]):
                        raise RuntimeError(f"Niepoprawne k/n: {path}")
                    meta.append((gu["first_axis"], int(gu["n"])))
                if pair in group_meta and group_meta[pair] != meta:
                    raise RuntimeError(f"Niespójne nazwy/szerokości grup: {path}")
                group_meta[pair] = meta
                metrics = summary["metrics"]
                if set(metrics) != {"U", "T"}:
                    raise RuntimeError(f"Brak profili metryk: {path}")
                for profile in PROFILES:
                    if set(metrics[profile]) != set(MODES):
                        raise RuntimeError(f"Niepełne tryby BN: {path}")
                    for mode in MODES:
                        values = [metrics[profile][mode]["top1"], metrics[profile][mode]["nll"]]
                        if not np.isfinite(values).all() or not 0 <= values[0] <= 1:
                            raise RuntimeError(f"Błędne metryki: {path}")
                if not (0.999 * plan["params"]["U"] <= plan["params"]["T"] <= plan["params"]["U"]):
                    raise RuntimeError(f"Niedopasowany budżet parametrów: {path}")
                if not np.isfinite([teacher[(pair, seed)]["top1"], teacher[(pair, seed)]["nll"]]).all():
                    raise RuntimeError(f"Błędny teacher: {pair}/{seed}")
                records.append(
                    dict(
                        seed=seed,
                        top1_U=float(metrics["U"]["CMA"]["top1"]),
                        top1_T=float(metrics["T"]["CMA"]["top1"]),
                        nll_U=float(metrics["U"]["CMA"]["nll"]),
                        nll_T=float(metrics["T"]["CMA"]["nll"]),
                        teacher=teacher[(pair, seed)]["top1"],
                        teacher_nll=teacher[(pair, seed)]["nll"],
                    )
                )
                summary_count += 1
            if len(records) != len(SEEDS):
                raise RuntimeError(f"Niepełna komórka: {pair}/{tag}")
            cells[(pair, tag)] = records
    if summary_count != 60 or len(cells) != 15 or len(teacher) != 20:
        raise RuntimeError("Nieoczekiwana liczebność danych E035")
    if set(group_meta) != set(PAIRS):
        raise RuntimeError("Brak geometrii którejś pary")

    analysis = read(analysis_path)
    if len(analysis["rows"]) != 720 or len(analysis["controls"]) != 120 or len(analysis["contrasts"]) != 90:
        raise RuntimeError("Nieoczekiwana liczebność analysis.json")
    test_cma = {
        (c["pair"], c["tag"]): c
        for c in analysis["contrasts"]
        if c["phase"] == "test" and c["mode"] == "CMA"
    }
    if len(test_cma) != 15:
        raise RuntimeError("Oczekiwano 15 kontrastów CMA w fazie test")

    summaries = {}
    max_analysis_diff = 0.0
    for (pair, tag), records in cells.items():
        u = np.array([100.0 * r["top1_U"] for r in records])
        t = np.array([100.0 * r["top1_T"] for r in records])
        te = np.array([100.0 * r["teacher"] for r in records])
        delta = t - u
        loss_u = te - u
        loss_t = te - t
        positive = int((delta > 0).sum())
        magnitude = bool(delta.mean() >= 2.0 - 1e-12 and positive >= 3) if tag != "b30" else None
        c = test_cma[(pair, tag)]
        for name, got, want in (
            ("mean", delta.mean(), c["mean"]),
            ("sd", delta.std(ddof=1), c["sd"]),
            ("teacher_mean", te.mean(), c["teacher"]["mean"]),
            ("U_mean", u.mean(), c["absolute"]["U"]["mean"]),
            ("T_mean", t.mean(), c["absolute"]["T"]["mean"]),
            ("U_loss_mean", loss_u.mean(), c["teacher_loss"]["U"]["mean"]),
            ("T_loss_mean", loss_t.mean(), c["teacher_loss"]["T"]["mean"]),
        ):
            max_analysis_diff = max(max_analysis_diff, abs(got - want))
            if abs(got - want) > 1e-9:
                raise RuntimeError(f"Niezgodność z analysis.json: {pair}/{tag}/{name}")
        if c["positive"] != positive or positive != 4 or c["direction"] is not True:
            raise RuntimeError(f"Błędny kierunek CMA: {pair}/{tag}")
        if c["magnitude"] != magnitude:
            raise RuntimeError(f"Błędna bramka wielkości efektu: {pair}/{tag}")
        if not np.allclose(delta, c["per_seed"], atol=1e-9, rtol=0):
            raise RuntimeError(f"Niezgodne różnice per seed: {pair}/{tag}")
        if abs(delta.mean() - EXPECTED_DELTA[(pair, tag)]) > 5e-4:
            raise RuntimeError(f"Średnia różni się od WYNIKI.md: {pair}/{tag}")
        if abs(u.mean() - EXPECTED_TOP1[(pair, tag)][0]) > 5e-4 or abs(t.mean() - EXPECTED_TOP1[(pair, tag)][1]) > 5e-4:
            raise RuntimeError(f"Top-1 różni się od WYNIKI.md: {pair}/{tag}")
        if abs(te.mean() - EXPECTED_TEACHER[pair]) > 5e-4:
            raise RuntimeError(f"Teacher różni się od WYNIKI.md: {pair}")
        summaries[(pair, tag)] = dict(
            U=u,
            T=t,
            teacher=te,
            delta=delta,
            loss_U=loss_u,
            loss_T=loss_t,
            positive=positive,
            magnitude=magnitude,
        )
    if not analysis["primary_pass"] or analysis["primary_pass_n"] != 15:
        raise RuntimeError("E035 primary_pass nie jest 15/15")
    if analysis["magnitude_pass_n"] != 9:
        raise RuntimeError("E035 magnitude_pass_n nie jest 9/10")
    if sum(1 for key, value in summaries.items() if value["magnitude"]) != 9:
        raise RuntimeError("Przeliczona bramka wielkości efektu nie daje 9/10")

    if args.english_paper:
        print("PASS: E035 held-out CMA contrasts and parameter matching (15 settings, four checkpoints each).")
        for path in save(draw_paper_delta(summaries), "e035_tilt_minus_uniform_en", english=True):
            print(path.relative_to(ROOT))
        return

    # Dokładne równości planów między parami (bez zaokrągleń).
    shared = []
    for tag in TAGS:
        for profile in PROFILES:
            by_widths = {}
            for pair in PAIRS:
                by_widths.setdefault(tuple(plans[pair][tag][profile]), []).append(pair)
            for widths, members in by_widths.items():
                if len(members) > 1:
                    shared.append(
                        dict(
                            budget=tag,
                            budget_pct=PCT[tag],
                            profile=profile,
                            pairs=";".join(members),
                            groups_n=len(widths),
                            comment=(
                                "Identyczne wektory szerokości; nie identyczne modele, "
                                "bo różnią się klasyfikator/parametry."
                            ),
                        )
                    )
    if not shared:
        raise RuntimeError("Brak wykrytych współdzielonych planów")
    for pair in PAIRS:
        for tag in TAGS:
            if plans[pair][tag]["U"] == plans[pair][tag]["T"]:
                raise RuntimeError(f"U i T są identyczne wewnątrz pary: {pair}/{tag}")

    # CSV: szerokości grup.
    width_rows = []
    for pair in PAIRS:
        arch = pair.split("_")[0]
        meta = group_meta[pair]
        for tag in TAGS:
            p = plans[pair][tag]
            for index, (first_axis, n) in enumerate(meta):
                width_rows.append(
                    dict(
                        pair=pair,
                        arch=arch,
                        dataset=pair.split("_", 1)[1],
                        budget=tag,
                        budget_pct=PCT[tag],
                        group_index=index,
                        group_first_axis=first_axis,
                        group_permutation=group_permutation(arch, first_axis),
                        group_short=group_short(first_axis),
                        n_channels=n,
                        k_U=p["U"][index],
                        k_T=p["T"][index],
                        keep_U_pct=100.0 * p["U"][index] / n,
                        keep_T_pct=100.0 * p["T"][index] / n,
                        delta_k_T_minus_U=p["T"][index] - p["U"][index],
                        params_U=p["params"]["U"],
                        params_T=p["params"]["T"],
                        param_ratio_T_over_U=p["params"]["T"] / p["params"]["U"],
                    )
                )
    write_csv(OUT / "szerokosci_grup.csv", list(width_rows[0]), width_rows)

    # CSV: metryki per seed.
    metric_rows = []
    for pair in PAIRS:
        for tag in TAGS:
            for r in cells[(pair, tag)]:
                metric_rows.append(
                    dict(
                        pair=pair,
                        arch=pair.split("_")[0],
                        dataset=pair.split("_", 1)[1],
                        split=SPLIT[pair],
                        budget=tag,
                        budget_pct=PCT[tag],
                        seed=r["seed"],
                        phase="test",
                        top1_U_cma=100.0 * r["top1_U"],
                        top1_T_cma=100.0 * r["top1_T"],
                        top1_teacher_cma=100.0 * r["teacher"],
                        delta_T_minus_U_pp=100.0 * (r["top1_T"] - r["top1_U"]),
                        loss_U_vs_teacher_pp=100.0 * (r["teacher"] - r["top1_U"]),
                        loss_T_vs_teacher_pp=100.0 * (r["teacher"] - r["top1_T"]),
                        nll_U_cma=r["nll_U"],
                        nll_T_cma=r["nll_T"],
                        nll_teacher_cma=r["teacher_nll"],
                    )
                )
    write_csv(OUT / "cma_teacher.csv", list(metric_rows[0]), metric_rows)

    # CSV: podsumowanie 15 komórek + szerokości/parametry.
    cell_rows = []
    for pair in PAIRS:
        arch = pair.split("_")[0]
        meta = group_meta[pair]
        sum_n = sum(n for _, n in meta)
        for tag in TAGS:
            s = summaries[(pair, tag)]
            p = plans[pair][tag]
            cell_rows.append(
                dict(
                    pair=pair,
                    arch=arch,
                    dataset=pair.split("_", 1)[1],
                    split=SPLIT[pair],
                    budget=tag,
                    budget_pct=PCT[tag],
                    groups_n=len(meta),
                    sum_n_channels=sum_n,
                    channel_keep_U_pct=100.0 * sum(p["U"]) / sum_n,
                    channel_keep_T_pct=100.0 * sum(p["T"]) / sum_n,
                    params_U=p["params"]["U"],
                    params_T=p["params"]["T"],
                    param_ratio_T_over_U=p["params"]["T"] / p["params"]["U"],
                    top1_U_mean=float(s["U"].mean()),
                    top1_U_sd=float(s["U"].std(ddof=1)),
                    top1_T_mean=float(s["T"].mean()),
                    top1_T_sd=float(s["T"].std(ddof=1)),
                    top1_teacher_mean=float(s["teacher"].mean()),
                    delta_T_minus_U_mean_pp=float(s["delta"].mean()),
                    delta_T_minus_U_sd_pp=float(s["delta"].std(ddof=1)),
                    positive_seeds=s["positive"],
                    loss_U_mean_pp=float(s["loss_U"].mean()),
                    loss_T_mean_pp=float(s["loss_T"].mean()),
                    magnitude_gate="" if s["magnitude"] is None else str(s["magnitude"]).lower(),
                )
            )
    write_csv(OUT / "komorki.csv", list(cell_rows[0]), cell_rows)
    write_csv(OUT / "wspoldzielone_plany.csv", list(shared[0]), shared)

    # Fig. 1: faktyczne k/n w grupach.
    style()
    colors = {"U": "#4D4D4D", "T": "#0072B2"}
    markers = {"U": "o", "T": "s"}
    fig, axes = plt.subplots(5, 3, figsize=(17.2, 13.8), sharey=True, constrained_layout=True)
    for row, pair in enumerate(PAIRS):
        meta = group_meta[pair]
        n = np.array([m[1] for m in meta])
        labels = [group_short(m[0]) for m in meta]
        x = np.arange(len(meta))
        for col, tag in enumerate(TAGS):
            ax = axes[row, col]
            p = plans[pair][tag]
            k_u = np.array(p["U"])
            k_t = np.array(p["T"])
            ax.plot(x, 100.0 * k_u / n, color=colors["U"], marker=markers["U"], ms=3.4, lw=1.15, label="UNIFORM (U)")
            ax.plot(x, 100.0 * k_t / n, color=colors["T"], marker=markers["T"], ms=3.4, lw=1.15, label="TILT (T)")
            ax.set_xlim(-0.5, len(meta) - 0.5)
            ax.set_ylim(0, 100)
            ax.set_xticks(x, labels, rotation=60, ha="right")
            ax.grid(axis="x", visible=False)
            keep_u = 100.0 * k_u.sum() / n.sum()
            keep_t = 100.0 * k_t.sum() / n.sum()
            ax.set_title(
                f"{LABELS[pair]} · {PCT[tag]}% redukcji P\n"
                f"P: U {fmt_p(p['params']['U'])} | T {fmt_p(p['params']['T'])} "
                f"({p['params']['T'] / p['params']['U']:.4f}×)",
                fontsize=8.2,
            )
            ax.text(
                0.01,
                0.02,
                f"kanały Σk/Σn: U {keep_u:.1f}% · T {keep_t:.1f}%",
                transform=ax.transAxes,
                fontsize=6.2,
                color="0.25",
            )
            if col == 0:
                ax.set_ylabel("k/n kanałów w grupie [%]")
            if row == 4:
                ax.set_xlabel("grupa (nazwa z osi permutacji upstream)")
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle(
        "E035: faktyczne szerokości grup U vs T (k/n), faza niezależna\n"
        "Budżet 30/50/70% dotyczy redukcji PARAMETRÓW, nie kanałów; k/n pokazuje osobno zachowane kanały. "
        "Tiny = oficjalny val.",
        fontsize=11.5,
    )
    save(fig, "fig1_szerokosci_grup")

    # Fig. 2: top-1 CMA U/T na tle teacher CMA, pary po seedach.
    fig, axes = plt.subplots(5, 3, figsize=(16.0, 13.2), sharey=True, constrained_layout=True)
    x = np.arange(len(SEEDS))
    for row, pair in enumerate(PAIRS):
        for col, tag in enumerate(TAGS):
            ax = axes[row, col]
            s = summaries[(pair, tag)]
            ax.plot(x, s["teacher"], color="0.25", ls="--", marker="o", ms=3.8, lw=1.0, label="teacher CMA")
            ax.plot(x - 0.06, s["U"], color=colors["U"], marker="o", ms=4.2, lw=0, label="U CMA")
            ax.plot(x + 0.06, s["T"], color=colors["T"], marker="s", ms=4.2, lw=0, label="T CMA")
            ax.set_xticks(x, [str(seed) for seed in SEEDS])
            ax.set_xlim(-0.35, len(SEEDS) - 0.65)
            ax.set_ylim(0, 100)
            ax.set_title(
                f"{LABELS[pair]} · {PCT[tag]}% P · {SPLIT_SHORT[pair]}\n"
                f"T-U {s['delta'].mean():+.2f} pp; strata T {s['loss_T'].mean():.2f} pp",
                fontsize=8.2,
            )
            if col == 0:
                ax.set_ylabel("top-1 CMA [%]")
            if row == 4:
                ax.set_xlabel("seed")
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(
        "E035: niezależna ocena CMA — U i T wobec teacher CMA (pary po seedach)\n"
        "Dla CIFAR-10/100 oficjalny test; dla ResNet/Tiny oficjalny val. Oś y wspólna 0–100%, bez sugerowania istotności.",
        fontsize=11.5,
    )
    save(fig, "fig2_teacher_cma_top1")

    # Fig. 3: utrata top-1 względem teacher (niżej = lepiej).
    fig, axes = plt.subplots(5, 3, figsize=(16.0, 13.2), constrained_layout=True)
    ymax = max(float(v["loss_U"].max()) for v in summaries.values())
    ymax = max(ymax, max(float(v["loss_T"].max()) for v in summaries.values())) * 1.08
    for row, pair in enumerate(PAIRS):
        for col, tag in enumerate(TAGS):
            ax = axes[row, col]
            s = summaries[(pair, tag)]
            ax.axhline(0, color="0.35", ls="--", lw=1.0, label="teacher CMA (0)")
            ax.plot(x - 0.06, s["loss_U"], color=colors["U"], marker="o", ms=4.2, lw=0, label="U CMA")
            ax.plot(x + 0.06, s["loss_T"], color=colors["T"], marker="s", ms=4.2, lw=0, label="T CMA")
            ax.set_xticks(x, [str(seed) for seed in SEEDS])
            ax.set_xlim(-0.35, len(SEEDS) - 0.65)
            ax.set_ylim(0, ymax)
            ax.set_title(
                f"{LABELS[pair]} · {PCT[tag]}% P · {SPLIT_SHORT[pair]}\n"
                f"utrata: U {s['loss_U'].mean():.2f} pp | T {s['loss_T'].mean():.2f} pp",
                fontsize=8.2,
            )
            if col == 0:
                ax.set_ylabel("utrata top-1 vs teacher CMA [pp]")
            if row == 4:
                ax.set_xlabel("seed")
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        "E035: utrata top-1 względem teacher CMA po CMA (niżej = lepiej)\n"
        "Różnica teacher - model; 0 = poziom teacher. Wspólna oś y obejmuje wszystkie 15 komórek.",
        fontsize=11.5,
    )
    save(fig, "fig3_teacher_cma_utrata")

    # Weryfikacja i źródła.
    verification = dict(
        status="PASS",
        run=str(RUN.relative_to(ROOT)),
        phase="test",
        tiny_split="oficjalny val (cache_val); pozostałe pary: oficjalny test",
        counts=dict(
            pairs=5,
            budgets=3,
            cells=15,
            seeds=4,
            summaries=summary_count,
            controls=len(teacher),
            cma_profile_points=120,
            cma_teacher_points=20,
            cma_contrasts=15,
        ),
        checks=dict(
            frozen_equal_run_copy=True,
            frozen_sha256_equal_manifest=True,
            all_cells_have_4_seeds=True,
            all_plans_equal_frozen=True,
            all_cluster_widths_equal_frozen=True,
            all_teacher_cma_present=True,
            all_15_cells_positive_4_of_4=True,
            magnitude_gate_9_of_10=True,
            analysis_json_consistent="max abs diff <= 1e-9",
            wyniki_md_means_consistent="tolerance 5e-4 pp",
            tiny_uses_official_val=True,
            cifar100_reserve_unused=True,
        ),
        max_abs_diff_vs_analysis_json=max_analysis_diff,
        shared_plans=shared,
        note=(
            "Szerokości pochodzą z E035/frozen.json. 30/50/70% to nominalna redukcja "
            "parametrów; k/n to osobna miara kanałów. U i T nie są utożsamiane."
        ),
    )
    dump(OUT / "weryfikacja.json", verification)

    key_hashes = {
        "experiments/.../frozen.json": input_hashes[str(FROZEN.relative_to(ROOT))],
        "runs/.../analysis/analysis.json": input_hashes[str(analysis_path.relative_to(ROOT))],
        "runs/.../manifest.json": input_hashes[str(manifest_path.relative_to(ROOT))],
        "runs/.../test_data.json": input_hashes[str(test_data_path.relative_to(ROOT))],
        "runs/.../analysis/per_seed.csv": input_hashes[str(per_seed_path.relative_to(ROOT))],
    }
    shared_table = "\n".join(
        f"| {row['budget']} | {row['profile']} | {row['pairs'].replace(';', ', ')} | {row['groups_n']} |"
        for row in shared
    )
    readme = f"""# E035 — ilustracje publikacyjne (CPU)

Odtwarzalne ilustracje z metadanych E035. **Nie czytają obrazów, checkpointów
ani wag**; korzystają wyłącznie z `frozen.json` i metryk `summary.json` /
`controls.json` runu `{RUN.name}`. Faza `test` oznacza niezależną ocenę:
oficjalny test CIFAR-10/100 oraz **oficjalny val** dla ResNet/Tiny.

## Pliki

| Plik | Co pokazuje |
|---|---|
| `fig1_szerokosci_grup.png/.pdf` | Faktyczne `k/n` kanałów w 12 (ResNet18) lub 13 (PreActResNet18) grupach, U vs T, 5 par × 30/50/70%. |
| `fig2_teacher_cma_top1.png/.pdf` | Top-1 CMA U i T oraz teacher CMA po seedach, 5 par × 3 budżety. |
| `fig3_teacher_cma_utrata.png/.pdf` | Utrata top-1 względem teacher CMA po seedach (niżej = lepiej). |
| `szerokosci_grup.csv` | `k`, `n`, `k/n`, nazwy grup, parametry U/T dla każdej grupy i komórki. |
| `cma_teacher.csv` | Metryki CMA per seed: U, T, teacher, różnica T−U, strata vs teacher, NLL. |
| `komorki.csv` | Podsumowanie 15 komórek: średnie, SD, 4/4, kanały, parametry. |
| `wspoldzielone_plany.csv` | Dokładnie równe wektory szerokości między parami. |
| `weryfikacja.json` | Kompletność, zgodność z `analysis.json`/`WYNIKI.md`, faza i teacher. |
| `sources.json` | SHA256 i rozmiary wszystkich wejść i wyjść. |

## Nazwy grup i budżet

Nazwy grup to pierwsza oś permutacji upstream (`group_first_axis` w CSV),
a `group_permutation` to nazwa osi. Grupa `relu_conv` jest współdzielona przez
wyjście stemu i bloki `layer1`. **30/50/70% to nominalna redukcja liczby
parametrów, nie odsetek kanałów.** Dlatego wykres pokazuje `k/n` osobno, a
`komorki.csv` podaje także `Σk/Σn` oraz faktyczne `params_U`/`params_T`
(T/U ≈ 0,999). Nie należy zastępować tym wykresem tabeli parametrów.

## Plany współdzielone (dokładna równość wektorów)

| Budżet | Profil | Pary o identycznych szerokościach | Grupy |
|---|---|---|---:|
{shared_table}

Poza tymi przypadkami wektory różnią się. W obrębie każdej pary U i T różnią
się przy każdym budżecie — T realokuje kanały. ResNet i PreAct mają różną
liczbę grup (12 vs 13), więc nie dzielą planu między architekturami.

## Co sprawdzono

- 15 komórek = 5 par × 3 budżety, po 4 seedy; 60 plików `summary.json` fazy
  `test`, 20 plików kontroli teacher.
- Każdy plan w `summary.json` jest równy `E035/frozen.json`; `k` klastrów jest
  równe szerokościom z `frozen.json`; `n` i nazwy grup są spójne.
- Teacher CMA istnieje dla każdej pary/seed; 15/15 średnich T−U dodatnich,
  każda komórka 4/4; bramka ≥2 pp przy 50/70% daje 9/10.
- Metryki odtwarzają `analysis.json` (maks. różnica ≤ 1e-9) i średnie z
  `WYNIKI.md` (tolerancja 5e-4 pp); rezerwa CIFAR-100 nieużyta.

## Uruchomienie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  python -B experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/generuj_publikacyjne.py
```

Skrypt jest deterministyczny (stała kolejność, brak losowości, metadane PDF
bez daty), zapisuje wyłącznie do tego katalogu i sprawdza hashe wejść.

## Hashe kluczowych wejść

| Wejście | SHA256 |
|---|---|
"""
    for name, digest in key_hashes.items():
        readme += f"| `{name}` | `{digest}` |\n"
    readme += """
Pełna lista hashy 85 plików wejściowych (5 bazowych + 60 `summary.json` + 20 `controls.json`) oraz wyjść: `sources.json`.

## Ograniczenia

To opisowa ilustracja, nie test istotności; cztery seedy to powtórzenia
treningu. Nie dodajemy wniosku o mechanizmie, uniwersalności ani przewadze
compute-matched. Test/val E035 zostały już użyte i nie są „niewidzianą
walidacją” przyszłego strojenia.
"""
    (OUT / "README.md").write_text(readme)

    generated = [
        "fig1_szerokosci_grup.png",
        "fig1_szerokosci_grup.pdf",
        "fig2_teacher_cma_top1.png",
        "fig2_teacher_cma_top1.pdf",
        "fig3_teacher_cma_utrata.png",
        "fig3_teacher_cma_utrata.pdf",
        "szerokosci_grup.csv",
        "cma_teacher.csv",
        "komorki.csv",
        "wspoldzielone_plany.csv",
        "weryfikacja.json",
        "README.md",
    ]
    sources = dict(
        run=str(RUN.relative_to(ROOT)),
        script=dict(
            path=str(Path(__file__).relative_to(ROOT)),
            sha256=sha256(Path(__file__)),
            bytes=Path(__file__).stat().st_size,
        ),
        inputs={name: dict(sha256=digest, bytes=(ROOT / name).stat().st_size) for name, digest in input_hashes.items()},
        outputs={name: dict(sha256=sha256(OUT / name), bytes=(OUT / name).stat().st_size) for name in generated},
    )
    dump(OUT / "sources.json", sources)

    print("PASS", len(cells), "komórek,", summary_count, "summary,", len(teacher), "teacher")
    for path in (OUT / "fig1_szerokosci_grup.png", OUT / "fig2_teacher_cma_top1.png", OUT / "fig3_teacher_cma_utrata.png"):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
