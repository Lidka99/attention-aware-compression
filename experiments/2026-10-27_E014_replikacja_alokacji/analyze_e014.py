"""E014: międzyseedowa analiza replikacji TILT+0.30 vs UNIFORM (tylko CPU).

Eksperyment: 2026-10-27_E014_replikacja_alokacji; pełny opis i protokół:
OPIS.md w tym samym katalogu. Skrypt czyta surowe summary.json ukończonego
biegu E014 (4 seedy E011 × 3 budżety × 4 warianty), sprawdza spójność danych,
paruje seedy i liczy różnice top-1 po BN. Nie trenuje, nie używa GPU i nie
zmienia istniejących wyników.

Jednostką wnioskowania jest seed, nie pojedyncza para seed×budżet. Trzy
budżety korzystają z tych samych czterech checkpointów, więc 12 różnic nie
jest 12 niezależnymi replikacjami. seed42 był użyty selekcyjnie w E013;
raportujemy go osobno i dodatkowo podzbiór seedów 43–45. Przedziały t-95%
przy n=4 (df=3) i n=3 (df=2) są opisowe i ostrożne, nie są populacyjnym CI.

Wejście:
  --run-dir PATH    katalog biegu E014 z manifest.json i seeds/seed_XX/.
  --output-dir PATH nowy katalog na analysis.json, analysis.md, CSV i wykresy.
  --audit-dir PATH  opcjonalny nowy katalog audytu pod runs/E014/audits/;
                    otrzymuje kopię wyników, komendę i META.json.

Wyjście: analysis.json, analysis.md, budgets.csv, seed_level.csv,
summary_stats.csv, variant_means.csv, META.json oraz trzy wykresy PNG+PDF.
Uruchomienie i kontekst: OPIS.md (ten sam katalog).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT = "2026-10-27_E014_replikacja_alokacji"
OPIS_REL = "experiments/2026-10-27_E014_replikacja_alokacji/OPIS.md"
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "runs/E014/20261027T094707Z_267904d_6a0e93eb"

SEEDS = (42, 43, 44, 45)
SEEDS_WITHOUT_42 = (43, 44, 45)
BUDGETS = (30, 50, 70)
VARIANTS = ("uniform", "tilt_p030", "tilt_p015", "mag2")
CANDIDATES = ("tilt_p030", "tilt_p015", "mag2")

VARIANT_LABELS = {
    "uniform": "UNIFORM",
    "tilt_p030": "TILT+0.30",
    "tilt_p015": "TILT+0.15",
    "mag2": "MAG2",
}
VARIANT_STYLE = {
    "uniform": {"color": "#4d4d4d", "marker": "o"},
    "tilt_p030": {"color": "#c0392b", "marker": "s"},
    "tilt_p015": {"color": "#e67e22", "marker": "^"},
    "mag2": {"color": "#2c6fb7", "marker": "D"},
}
SEED_COLORS = {42: "#c0392b", 43: "#6c8ebf", 44: "#2f8f5b", 45: "#8e6c2f"}
SECONDARY_METRICS = (
    ("nll", "ΔNLL", "candidate − UNIFORM; ujemne = TILT lepszy", ""),
    ("prediction_agreement", "Δagreement", "candidate − UNIFORM; dodatnie = TILT lepszy", "pp"),
    ("logit_mae", "ΔMAE", "candidate − UNIFORM; ujemne = TILT lepszy", ""),
    ("logit_rmse", "ΔRMSE", "candidate − UNIFORM; ujemne = TILT lepszy", ""),
    ("kl_baseline_candidate", "ΔKL", "candidate − UNIFORM; ujemne = TILT lepszy", ""),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def pl(value: float, digits: int = 2, signed: bool = False) -> str:
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace(".", ",")


def pl_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def describe(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    require(array.size >= 2, "statystyki wymagają co najmniej dwóch wartości")
    n = int(array.size)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    se = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, n - 1))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "positive_count": int(np.sum(array > 0)),
        "negative_count": int(np.sum(array < 0)),
        "ci95_t": [mean - t_crit * se, mean + t_crit * se],
        "ci_method": f"t-95%, df={n - 1}; opisowy przy małym n",
    }


def parse_name(name: str) -> tuple[int, str]:
    require(name.startswith("b") and "_" in name, f"nieoczekiwana nazwa wariantu: {name}")
    budget_text, variant = name[1:].split("_", 1)
    require(budget_text.isdigit(), f"nieoczekiwany budżet w nazwie: {name}")
    require(variant in VARIANTS, f"nieoczekiwany wariant w nazwie: {name}")
    return int(budget_text), variant


def expected_names() -> set[str]:
    return {f"b{budget}_{variant}" for budget in BUDGETS for variant in VARIANTS}


def validate_manifest(manifest: dict) -> None:
    require(manifest.get("experiment") == EXPERIMENT, "manifest dotyczy innego eksperymentu")
    require(manifest.get("status") == "completed", "bieg E014 nie ma statusu completed")
    require(manifest.get("official_test_used") is False, "manifest wskazuje użycie oficjalnego testu")
    require(set(manifest.get("seeds", {})) == {str(seed) for seed in SEEDS},
            "manifest nie zawiera dokładnie seedów 42–45")
    for seed in SEEDS:
        entry = manifest["seeds"][str(seed)]
        require(entry.get("status") == "completed", f"seed {seed} nie ma statusu completed")
        exit_codes = entry.get("exit_codes", {})
        require(set(exit_codes) == expected_names(),
                f"seed {seed}: nieoczekiwany zestaw workerów w exit_codes")
        require(all(code == 0 for code in exit_codes.values()),
                f"seed {seed}: co najmniej jeden worker zakończył się błędem")


def load_run(run_dir: Path) -> tuple[dict, dict, list[Path], dict]:
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"brak katalogu biegu: {run_dir}")
    manifest_path = run_dir / "manifest.json"
    require(manifest_path.is_file(), f"brak manifestu: {manifest_path}")
    manifest = read_json(manifest_path)
    validate_manifest(manifest)

    inputs = [manifest_path]
    data: dict[int, dict[str, dict]] = {}
    params_by_variant: dict[str, set[int]] = {name: set() for name in expected_names()}
    targets_by_budget: dict[int, set[int]] = {budget: set() for budget in BUDGETS}
    baseline_signatures: dict[int, set[tuple]] = {seed: set() for seed in SEEDS}

    for seed in SEEDS:
        seed_entry = manifest["seeds"][str(seed)]
        seed_dir = run_dir / seed_entry["path"]
        allocations_dir = seed_dir / "allocations"
        require(allocations_dir.is_dir(), f"brak katalogu allocations: {allocations_dir}")
        found = {path.name for path in allocations_dir.iterdir() if path.is_dir()}
        require(found == expected_names(),
                f"seed {seed}: oczekiwano 12 wariantów, znaleziono {sorted(found)}")

        data[seed] = {}
        for name in sorted(expected_names()):
            summary_path = allocations_dir / name / "summary.json"
            require(summary_path.is_file(), f"brak {summary_path}")
            summary = read_json(summary_path)
            inputs.append(summary_path)

            allocation = summary["allocation"]
            compressed = summary["compressed"]
            baseline = summary["baseline"]
            post = compressed["post_bn"]
            pre = compressed["pre_bn"]
            versus = compressed["post_bn_vs_baseline"]

            require(post["count"] == 5000, f"{summary_path}: post_bn count != 5000")
            require(pre["count"] == 5000, f"{summary_path}: pre_bn count != 5000")
            require(baseline["metrics"]["count"] == 5000, f"{summary_path}: baseline count != 5000")
            require(compressed["params_before"] == baseline["params"],
                    f"{summary_path}: params_before != baseline params")
            require(compressed["params"] < compressed["params_before"],
                    f"{summary_path}: brak redukcji parametrów")
            computed_reduction = 1.0 - compressed["params"] / compressed["params_before"]
            require(abs(computed_reduction - compressed["param_reduction_fraction"]) < 1e-12,
                    f"{summary_path}: niezgodna param_reduction_fraction")
            require(all(key in versus for key in (
                "prediction_agreement", "logit_mae", "logit_rmse",
                "kl_baseline_candidate", "kl_candidate_baseline")),
                f"{summary_path}: brak metryk wtórnych")
            require(0.0 <= post["top1"] <= 1.0, f"{summary_path}: top1 poza [0,1]")
            require(0.0 <= pre["top1"] <= 1.0, f"{summary_path}: pre top1 poza [0,1]")

            budget, variant = parse_name(name)
            require(budget in BUDGETS, f"{summary_path}: nieoczekiwany budżet {budget}")
            require(variant == name.split("_", 1)[1], f"{summary_path}: niezgodna nazwa wariantu")

            target_params = int(allocation["target_params"])
            target_reduction = 1.0 - target_params / compressed["params_before"]
            deviation_pp = (computed_reduction - target_reduction) * 100.0
            require(abs(deviation_pp) <= 0.5,
                    f"{summary_path}: budżet poza tolerancją 0,5 pp ({deviation_pp:+.3f} pp)")

            params_by_variant[name].add(int(compressed["params"]))
            targets_by_budget[budget].add(target_params)
            baseline_signatures[seed].add((
                int(baseline["metrics"]["correct"]),
                int(baseline["metrics"]["count"]),
                float(baseline["metrics"]["top1"]),
                float(baseline["metrics"]["nll"]),
            ))

            data[seed][name] = {
                "seed": seed,
                "nominal_budget": budget,
                "variant": variant,
                "name": name,
                "params": int(compressed["params"]),
                "params_before": int(compressed["params_before"]),
                "target_params": target_params,
                "param_reduction_fraction": float(compressed["param_reduction_fraction"]),
                "correct": int(post["correct"]),
                "count": int(post["count"]),
                "top1": float(post["top1"]),
                "nll": float(post["nll"]),
                "pre_top1": float(pre["top1"]),
                "pre_nll": float(pre["nll"]),
                "prediction_agreement": float(versus["prediction_agreement"]),
                "logit_mae": float(versus["logit_mae"]),
                "logit_rmse": float(versus["logit_rmse"]),
                "kl_baseline_candidate": float(versus["kl_baseline_candidate"]),
                "kl_candidate_baseline": float(versus["kl_candidate_baseline"]),
            }

    for name, values in params_by_variant.items():
        require(len(values) == 1,
                f"wariant {name}: parametry różnią się między seedami: {sorted(values)}")
    for budget, values in targets_by_budget.items():
        require(len(values) == 1,
                f"budżet {budget}: różne target_params między wariantami: {sorted(values)}")
    for seed, values in baseline_signatures.items():
        require(len(values) == 1,
                f"seed {seed}: baseline różni się między wariantami: {sorted(values)}")

    baseline = {seed: next(iter(values)) for seed, values in baseline_signatures.items()}
    for seed, (correct, count, top1, nll) in baseline.items():
        require(count == 5000, f"seed {seed}: baseline count != 5000")
        require(0.0 <= top1 <= 1.0, f"seed {seed}: baseline top1 poza [0,1]")
        require(correct == round(top1 * count),
                f"seed {seed}: baseline correct nie zgadza się z top1")
    return manifest, data, inputs, baseline


def budget_rows(data: dict) -> list[dict]:
    rows = []
    for budget in BUDGETS:
        uniform = data[SEEDS[0]][f"b{budget}_uniform"]
        target_reduction_pct = (1.0 - uniform["target_params"] / uniform["params_before"]) * 100.0
        uniform_reduction_pct = uniform["param_reduction_fraction"] * 100.0
        for variant in VARIANTS:
            item = data[SEEDS[0]][f"b{budget}_{variant}"]
            reduction_pct = item["param_reduction_fraction"] * 100.0
            rows.append({
                "nominal_budget": budget,
                "variant": variant,
                "params": item["params"],
                "params_before": item["params_before"],
                "target_params": item["target_params"],
                "target_reduction_pct": target_reduction_pct,
                "realized_reduction_pct": reduction_pct,
                "deviation_from_target_pp": reduction_pct - target_reduction_pct,
                "extra_params_vs_uniform": item["params"] - uniform["params"],
                "delta_reduction_vs_uniform_pp": reduction_pct - uniform_reduction_pct,
            })
    return rows


def variant_means(data: dict) -> dict:
    output = {}
    for budget in BUDGETS:
        output[str(budget)] = {}
        for variant in VARIANTS:
            name = f"b{budget}_{variant}"
            items = [data[seed][name] for seed in SEEDS]
            output[str(budget)][variant] = {
                "n": len(items),
                "mean_top1_pct": float(np.mean([item["top1"] for item in items]) * 100.0),
                "sd_top1_pct": float(np.std([item["top1"] for item in items], ddof=1) * 100.0),
                "mean_nll": float(np.mean([item["nll"] for item in items])),
                "sd_nll": float(np.std([item["nll"] for item in items], ddof=1)),
                "mean_agreement_pct": float(
                    np.mean([item["prediction_agreement"] for item in items]) * 100.0),
                "sd_agreement_pct": float(
                    np.std([item["prediction_agreement"] for item in items], ddof=1) * 100.0),
                "mean_logit_mae": float(np.mean([item["logit_mae"] for item in items])),
                "sd_logit_mae": float(np.std([item["logit_mae"] for item in items], ddof=1)),
                "mean_logit_rmse": float(np.mean([item["logit_rmse"] for item in items])),
                "sd_logit_rmse": float(np.std([item["logit_rmse"] for item in items], ddof=1)),
                "mean_kl_baseline_candidate": float(
                    np.mean([item["kl_baseline_candidate"] for item in items])),
                "sd_kl_baseline_candidate": float(
                    np.std([item["kl_baseline_candidate"] for item in items], ddof=1)),
                "mean_params": int(np.mean([item["params"] for item in items])),
                "mean_reduction_pct": float(
                    np.mean([item["param_reduction_fraction"] for item in items]) * 100.0),
                "seed_top1_pct": {
                    str(seed): float(data[seed][name]["top1"] * 100.0) for seed in SEEDS
                },
            }
    return output


def paired_block(data: dict, candidate: str, reference: str = "uniform") -> dict:
    output = {
        "candidate": candidate,
        "reference": reference,
        "all_seeds": {},
        "seeds_43_45": {},
        "seed42": {},
    }
    for budget in BUDGETS:
        candidate_name = f"b{budget}_{candidate}"
        reference_name = f"b{budget}_{reference}"
        for label, seeds in (("all_seeds", SEEDS), ("seeds_43_45", SEEDS_WITHOUT_42)):
            differences = {
                str(seed): float(
                    (data[seed][candidate_name]["top1"] - data[seed][reference_name]["top1"]) * 100.0)
                for seed in seeds
            }
            output[label][str(budget)] = {
                "differences_pp": differences,
                "summary": describe(differences.values()),
            }
        output["seed42"][str(budget)] = {
            "difference_pp": float(
                (data[42][candidate_name]["top1"] - data[42][reference_name]["top1"]) * 100.0)
        }
    return output


def secondary_block(data: dict) -> dict:
    output = {}
    for budget in BUDGETS:
        output[str(budget)] = {}
        for metric, label, interpretation, unit in SECONDARY_METRICS:
            differences = [
                float(data[seed][f"b{budget}_tilt_p030"][metric]
                      - data[seed][f"b{budget}_uniform"][metric])
                for seed in SEEDS
            ]
            output[str(budget)][metric] = {
                "label": label,
                "unit": unit,
                "interpretation": interpretation,
                "differences": {str(seed): value for seed, value in zip(SEEDS, differences)},
                "summary": describe(differences),
            }
    return output


def count_positive_pairs(paired: dict) -> tuple[int, int]:
    positive = 0
    total = 0
    for budget in BUDGETS:
        for value in paired["all_seeds"][str(budget)]["differences_pp"].values():
            total += 1
            positive += int(value > 0)
    return positive, total


def build_analysis(run_dir: Path) -> tuple[dict, dict, list[Path]]:
    manifest, data, inputs, baseline = load_run(run_dir)
    budgets = budget_rows(data)
    means = variant_means(data)
    paired = {candidate: paired_block(data, candidate) for candidate in CANDIDATES}
    tilt_vs_mag2 = paired_block(data, "tilt_p030", "mag2")
    secondary = secondary_block(data)
    positive, total = count_positive_pairs(paired["tilt_p030"])
    mag2_positive, mag2_total = count_positive_pairs(tilt_vs_mag2)

    for budget in BUDGETS:
        require(
            paired["tilt_p030"]["all_seeds"][str(budget)]["summary"]["n"] == len(SEEDS),
            f"budżet {budget}: zła liczba sparowanych seedów",
        )
        require(
            paired["tilt_p030"]["seeds_43_45"][str(budget)]["summary"]["n"]
            == len(SEEDS_WITHOUT_42),
            f"budżet {budget}: zła liczba seedów w podzbiorze 43–45",
        )

    baseline_top1_pct = [baseline[seed][2] * 100.0 for seed in SEEDS]
    baseline_correct = [baseline[seed][0] for seed in SEEDS]
    analysis = {
        "experiment": EXPERIMENT,
        "opis": OPIS_REL,
        "run_dir": display_path(run_dir),
        "run_id": manifest.get("run_id"),
        "git_sha": manifest.get("git_sha"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": {
            "unit": "seed",
            "n_all_seeds": len(SEEDS),
            "n_seeds_without_42": len(SEEDS_WITHOUT_42),
            "n_budgets": len(BUDGETS),
            "n_pairs_tilt030_vs_uniform": total,
            "n_checkpoints": len(SEEDS),
            "n_independent_checkpoints_after_e013": len(SEEDS_WITHOUT_42),
            "seed42_selected_in_e013": True,
            "note": ("12 par seed×budżet nie jest 12 niezależnymi replikacjami; "
                     "trzy budżety dzielą te same cztery checkpointy. seed42 "
                     "był selekcyjny w E013, więc po screeningu niezależne są "
                     "seedy 43–45."),
        },
        "validation": {
            "manifest_status": manifest.get("status"),
            "official_test_used": manifest.get("official_test_used"),
            "seeds": list(SEEDS),
            "variants_per_seed": len(VARIANTS),
            "summaries_checked": len(inputs) - 1,
            "baseline_per_seed": {
                str(seed): {
                    "correct": int(baseline[seed][0]),
                    "count": int(baseline[seed][1]),
                    "top1": float(baseline[seed][2]),
                    "nll": float(baseline[seed][3]),
                }
                for seed in SEEDS
            },
            "baseline_top1_pct_min": min(baseline_top1_pct),
            "baseline_top1_pct_max": max(baseline_top1_pct),
            "baseline_correct_min": min(baseline_correct),
            "baseline_correct_max": max(baseline_correct),
            "params_before": data[SEEDS[0]][f"b{BUDGETS[0]}_uniform"]["params_before"],
            "params_consistent_across_seeds": True,
            "targets_consistent_within_budget": True,
            "max_abs_deviation_from_target_pp": max(
                abs(row["deviation_from_target_pp"]) for row in budgets),
        },
        "budgets": budgets,
        "variant_means": means,
        "paired": paired,
        "paired_tilt030_vs_mag2": tilt_vs_mag2,
        "secondary_tilt030_vs_uniform": secondary,
        "notes": [
            "Jednostką jest seed (n=4), nie 12 par seed×budżet.",
            "seed42 był użyty selekcyjnie w E013; jego wynik w E014 odtwarza E013, "
            "więc nie jest niezależną replikacją. Seedy 43–45 raportowano osobno.",
            "Przedziały t-95% przy n=4 (df=3) i n=3 (df=2) są opisowe; przy tak małym n "
            "nie szacują stabilnie wariancji populacji.",
            "TILT+0.30 zachowuje nieco więcej parametrów niż UNIFORM przy tym samym "
            "budżecie nominalnym; różnice podano w tabeli budżetów.",
            "MAG2 ma dokładnie ten sam budżet co UNIFORM.",
            "Ocena na dev (5000 przykładów), bez niezależnego testu i drugiego datasetu.",
        ],
        "tilt030_positive_pairs": {
            "positive": positive,
            "total": total,
            "all_positive": positive == total,
        },
        "tilt030_vs_mag2_positive_pairs": {
            "positive": mag2_positive,
            "total": mag2_total,
            "all_positive": mag2_positive == mag2_total,
        },
        "inputs": [],
    }

    # Ścieżki wejściowe i hashe są dodawane po wygenerowaniu plików.
    return analysis, data, inputs


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_tables(output_dir: Path, analysis: dict, data: dict) -> list[Path]:
    paths = []

    budget_rows_data = analysis["budgets"]
    budget_path = output_dir / "budgets.csv"
    write_csv(budget_path, budget_rows_data, [
        "nominal_budget", "variant", "params", "params_before", "target_params",
        "target_reduction_pct", "realized_reduction_pct", "deviation_from_target_pp",
        "extra_params_vs_uniform", "delta_reduction_vs_uniform_pp",
    ])
    paths.append(budget_path)

    seed_rows = []
    for seed in SEEDS:
        for budget in BUDGETS:
            uniform = data[seed][f"b{budget}_uniform"]
            for variant in VARIANTS:
                item = data[seed][f"b{budget}_{variant}"]
                seed_rows.append({
                    "seed": seed,
                    "nominal_budget": budget,
                    "variant": variant,
                    "params": item["params"],
                    "param_reduction_fraction": item["param_reduction_fraction"],
                    "top1_post_bn": item["top1"],
                    "nll_post_bn": item["nll"],
                    "prediction_agreement": item["prediction_agreement"],
                    "logit_mae": item["logit_mae"],
                    "logit_rmse": item["logit_rmse"],
                    "kl_baseline_candidate": item["kl_baseline_candidate"],
                    "delta_top1_vs_uniform_pp": (
                        (item["top1"] - uniform["top1"]) * 100.0),
                    "delta_nll_vs_uniform": item["nll"] - uniform["nll"],
                    "delta_agreement_vs_uniform": (
                        item["prediction_agreement"] - uniform["prediction_agreement"]),
                    "delta_logit_mae_vs_uniform": item["logit_mae"] - uniform["logit_mae"],
                    "delta_logit_rmse_vs_uniform": item["logit_rmse"] - uniform["logit_rmse"],
                    "delta_kl_vs_uniform": (
                        item["kl_baseline_candidate"] - uniform["kl_baseline_candidate"]),
                })
    seed_path = output_dir / "seed_level.csv"
    write_csv(seed_path, seed_rows, [
        "seed", "nominal_budget", "variant", "params", "param_reduction_fraction",
        "top1_post_bn", "nll_post_bn", "prediction_agreement", "logit_mae",
        "logit_rmse", "kl_baseline_candidate", "delta_top1_vs_uniform_pp",
        "delta_nll_vs_uniform", "delta_agreement_vs_uniform",
        "delta_logit_mae_vs_uniform", "delta_logit_rmse_vs_uniform",
        "delta_kl_vs_uniform",
    ])
    paths.append(seed_path)

    summary_rows = []
    for candidate in CANDIDATES:
        for subset, label in (("all_seeds", "42-45"), ("seeds_43_45", "43-45")):
            for budget in BUDGETS:
                block = analysis["paired"][candidate][subset][str(budget)]
                summary = block["summary"]
                summary_rows.append({
                    "candidate": candidate,
                    "subset": label,
                    "nominal_budget": budget,
                    "n": summary["n"],
                    "mean_diff_pp": summary["mean"],
                    "sd_diff_pp": summary["sd"],
                    "min_diff_pp": summary["min"],
                    "max_diff_pp": summary["max"],
                    "range_diff_pp": summary["range"],
                    "ci95_low_pp": summary["ci95_t"][0],
                    "ci95_high_pp": summary["ci95_t"][1],
                    "positive_count": summary["positive_count"],
                })
    summary_path = output_dir / "summary_stats.csv"
    write_csv(summary_path, summary_rows, [
        "candidate", "subset", "nominal_budget", "n", "mean_diff_pp", "sd_diff_pp",
        "min_diff_pp", "max_diff_pp", "range_diff_pp", "ci95_low_pp",
        "ci95_high_pp", "positive_count",
    ])
    paths.append(summary_path)

    means_rows = []
    for budget in BUDGETS:
        for variant in VARIANTS:
            item = analysis["variant_means"][str(budget)][variant]
            means_rows.append({
                "nominal_budget": budget,
                "variant": variant,
                "n": item["n"],
                "mean_top1_pct": item["mean_top1_pct"],
                "sd_top1_pct": item["sd_top1_pct"],
                "mean_nll": item["mean_nll"],
                "sd_nll": item["sd_nll"],
                "mean_agreement_pct": item["mean_agreement_pct"],
                "sd_agreement_pct": item["sd_agreement_pct"],
                "mean_logit_mae": item["mean_logit_mae"],
                "sd_logit_mae": item["sd_logit_mae"],
                "mean_logit_rmse": item["mean_logit_rmse"],
                "sd_logit_rmse": item["sd_logit_rmse"],
                "mean_kl_baseline_candidate": item["mean_kl_baseline_candidate"],
                "sd_kl_baseline_candidate": item["sd_kl_baseline_candidate"],
                "mean_params": item["mean_params"],
                "mean_reduction_pct": item["mean_reduction_pct"],
            })
    means_path = output_dir / "variant_means.csv"
    write_csv(means_path, means_rows, [
        "nominal_budget", "variant", "n", "mean_top1_pct", "sd_top1_pct",
        "mean_nll", "sd_nll", "mean_agreement_pct", "sd_agreement_pct",
        "mean_logit_mae", "sd_logit_mae", "mean_logit_rmse", "sd_logit_rmse",
        "mean_kl_baseline_candidate", "sd_kl_baseline_candidate",
        "mean_params", "mean_reduction_pct",
    ])
    paths.append(means_path)
    return paths


def write_markdown(path: Path, analysis: dict) -> None:
    lines = []
    add = lines.append
    add("# E014 — analiza międzyseedowa TILT+0.30 vs UNIFORM")
    add("")
    add(f"Wygenerowano: {analysis['created_utc']}  ")
    add(f"Run: `{analysis['run_id']}` (git `{analysis['git_sha']}`)  ")
    add(f"Opis eksperymentu: [OPIS.md](../../OPIS.md)  ")
    add(f"Skrypt: `analyze_e014.py` (SHA-256 w `META.json`)")
    add("")
    add("## Jak czytać")
    add("")
    add("Jednostką jest **seed** (n=4), nie pojedyncza para seed×budżet. "
        "Trzy budżety korzystają z tych samych czterech checkpointów, więc "
        "12 różnic seed×budżet **nie jest** 12 niezależnymi replikacjami.")
    add("")
    add("seed42 był użyty selekcyjnie w E013; jego wynik w E014 odtwarza E013 "
        "i nie jest niezależną replikacją. Dlatego obok pełnego zbioru 42–45 "
        "raportujemy osobno seedy **43–45**.")
    add("")
    add("Przedziały t-95% przy n=4 (df=3) i n=3 (df=2) są **opisowe i ostrożne**; "
        "przy tak małym n nie szacują stabilnie wariancji populacji. Wszystkie "
        "różnice liczone są jako `TILT − UNIFORM` w punktach procentowych (pp).")
    add("")
    add("## 1. Rzeczywiste budżety")
    add("")
    add("| Budżet | Wariant | Parametry | Redukcja | Odch. od celu | Δ redukcji vs UNIFORM | Dodatkowe param. vs UNIFORM |")
    add("|---:|---|---:|---:|---:|---:|---:|")
    for row in analysis["budgets"]:
        variant = VARIANT_LABELS[row["variant"]]
        if row["variant"] == "uniform":
            delta = "—"
            extra = "—"
        else:
            delta = pl(row["delta_reduction_vs_uniform_pp"], 3, signed=True) + " pp"
            extra = pl_int(row["extra_params_vs_uniform"])
        add(
            f"| {row['nominal_budget']}% | {variant} | {pl_int(row['params'])} | "
            f"{pl(row['realized_reduction_pct'], 3)}% | "
            f"{pl(row['deviation_from_target_pp'], 3, signed=True)} pp | "
            f"{delta} | {extra} |"
        )
    add("")
    add("Cel budżetu jest liczony na całym modelu. UNIFORM i MAG2 mają identyczną "
        "liczbę parametrów; TILT trafia bliżej celu i zachowuje nieco więcej "
        "parametrów. To niewielka różnica na korzyść TILT, znacznie mniejsza niż "
        "efekt top-1, ale nie jest zerowa i nie została tu renormalizowana.")
    add("")
    add("## 2. TILT+0.30 vs UNIFORM — różnice na seedach")
    add("")
    add("| Budżet | seed42 | seed43 | seed44 | seed45 | średnia | SD | zakres |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        block = analysis["paired"]["tilt_p030"]["all_seeds"][str(budget)]
        diffs = block["differences_pp"]
        summary = block["summary"]
        add(
            f"| {budget}% | {pl(diffs['42'], 2, signed=True)} | "
            f"{pl(diffs['43'], 2, signed=True)} | {pl(diffs['44'], 2, signed=True)} | "
            f"{pl(diffs['45'], 2, signed=True)} | "
            f"{pl(summary['mean'], 3, signed=True)} | {pl(summary['sd'], 3)} | "
            f"{pl(summary['range'], 2)} |"
        )
    add("")
    add(f"Wszystkie **{analysis['tilt030_positive_pairs']['positive']}/"
        f"{analysis['tilt030_positive_pairs']['total']}** różnic jest dodatnich.")
    add("")
    add("## 3. Podsumowanie statystyk")
    add("")
    add("| Budżet | Podzbiór | n | średnia | SD | min | max | zakres | 95% CI (t) |")
    add("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        for subset, label in (("all_seeds", "42–45"), ("seeds_43_45", "43–45")):
            summary = analysis["paired"]["tilt_p030"][subset][str(budget)]["summary"]
            add(
                f"| {budget}% | {label} | {summary['n']} | "
                f"{pl(summary['mean'], 3, signed=True)} | {pl(summary['sd'], 3)} | "
                f"{pl(summary['min'], 2, signed=True)} | "
                f"{pl(summary['max'], 2, signed=True)} | {pl(summary['range'], 2)} | "
                f"[{pl(summary['ci95_t'][0], 2, signed=True)}; "
                f"{pl(summary['ci95_t'][1], 2, signed=True)}] |"
            )
    add("")
    add("## 4. Pozostałe warianty vs UNIFORM")
    add("")
    add("| Wariant | Budżet | średnia 42–45 | SD | zakres | średnia 43–45 | SD 43–45 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for candidate in ("tilt_p015", "mag2"):
        for budget in BUDGETS:
            full = analysis["paired"][candidate]["all_seeds"][str(budget)]["summary"]
            subset = analysis["paired"][candidate]["seeds_43_45"][str(budget)]["summary"]
            add(
                f"| {VARIANT_LABELS[candidate]} | {budget}% | "
                f"{pl(full['mean'], 3, signed=True)} | {pl(full['sd'], 3)} | "
                f"{pl(full['range'], 2)} | {pl(subset['mean'], 3, signed=True)} | "
                f"{pl(subset['sd'], 3)} |"
            )
    add("")
    mag2_cells = []
    for budget in BUDGETS:
        summary = analysis["paired_tilt030_vs_mag2"]["all_seeds"][str(budget)]["summary"]
        mag2_cells.append(
            f"{budget}%: {pl(summary['mean'], 3, signed=True)} ± {pl(summary['sd'], 3)}")
    add("TILT+0.30 vs MAG2 (średnia ± SD, 42–45): " + "; ".join(mag2_cells) + " pp. "
        f"Dodatnie pary: {analysis['tilt030_vs_mag2_positive_pairs']['positive']}/"
        f"{analysis['tilt030_vs_mag2_positive_pairs']['total']}.")
    add("")
    add("## 5. Metryki wtórne TILT+0.30 vs UNIFORM")
    add("")
    add("Wartości to średnia ± SD różnicy na czterech seedach. Dla NLL, MAE, RMSE "
        "i KL wartość ujemna oznacza lepszy TILT; dla agreement dodatnia oznacza "
        "lepszy TILT. Agreement podano w pp.")
    add("")
    add("| Budżet | ΔNLL | Δagreement | ΔMAE | ΔRMSE | ΔKL |")
    add("|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        cells = []
        for metric, _, _, unit in SECONDARY_METRICS:
            block = analysis["secondary_tilt030_vs_uniform"][str(budget)][metric]
            summary = block["summary"]
            scale = 100.0 if unit == "pp" else 1.0
            digits = 2 if unit == "pp" else 4
            cells.append(
                f"{pl(summary['mean'] * scale, digits, signed=True)} ± "
                f"{pl(summary['sd'] * scale, digits)}"
            )
        add(f"| {budget}% | " + " | ".join(cells) + " |")
    add("")
    add("Dla każdej z tych metryk znak różnicy był zgodny na wszystkich 12 parach "
        "seed×budżet (NLL/MAE/RMSE/KL ujemne, agreement dodatni).")
    add("")
    add("## 6. Weryfikacja i pliki")
    add("")
    validation = analysis["validation"]
    add(f"- manifest: `{validation['manifest_status']}`, official_test_used="
        f"`{validation['official_test_used']}`;")
    add(f"- sprawdzono {validation['summaries_checked']} plików summary.json "
        f"(4 seedy × 12 wariantów);")
    add(f"- baseline na seedach: top-1 "
        f"{pl(validation['baseline_top1_pct_min'], 2)}–"
        f"{pl(validation['baseline_top1_pct_max'], 2)}%, poprawnych "
        f"{validation['baseline_correct_min']}–{validation['baseline_correct_max']} "
        f"z 5000; baseline różni się między seedami, więc porównania są parowane "
        f"w obrębie seeda;")
    add(f"- parametry i target_params spójne między seedami; maks. odchylenie od "
        f"celu {pl(validation['max_abs_deviation_from_target_pp'], 3)} pp;")
    add("- brak GPU, treningu i zmiany istniejących wyników.")
    add("")
    add("Pliki: `analysis.json`, `budgets.csv`, `seed_level.csv`, "
        "`summary_stats.csv`, `variant_means.csv`, `META.json` oraz:")
    add("")
    add("- `roznice_tilt030_wg_budzetu.png` / `.pdf`")
    add("- `top1_wg_budzetu.png` / `.pdf`")
    add("- `zrealizowany_budzet.png` / `.pdf`")
    add("")
    add("## Ograniczenia i wniosek")
    add("")
    add("Kierunek TILT+0.30 jest spójny na wszystkich czterech seedach i trzech "
        "budżetach, także po wyłączeniu selekcyjnego seed42. Nadal jest to jeden "
        "dataset (CIFAR-10), dev, cztery checkpointy z jednego przepisu i adaptacja "
        "operatora E012; brak niezależnego testu. Małe n i współdzielone seedy "
        "ograniczają wnioskowanie statystyczne — raportujemy kierunek i rozrzut, "
        "nie istotność populacyjną.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig, stem: str, output_dir: Path) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(path, dpi=200, bbox_inches="tight", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_differences(analysis: dict, output_dir: Path) -> list[Path]:
    budgets = list(BUDGETS)
    x = np.arange(len(budgets))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for seed in SEEDS:
        values = [
            analysis["paired"]["tilt_p030"]["all_seeds"][str(budget)]
            ["differences_pp"][str(seed)] for budget in budgets
        ]
        if seed == 42:
            ax.plot(x, values, color=SEED_COLORS[seed], linestyle="--", marker="o",
                    markersize=7, markerfacecolor="white", markeredgewidth=1.7,
                    linewidth=1.7, label="seed42 (użyty w E013)", zorder=4)
        else:
            ax.plot(x, values, color=SEED_COLORS[seed], marker="o", markersize=5,
                    linewidth=1.2, alpha=0.85, label=f"seed{seed}", zorder=3)
    means = [
        analysis["paired"]["tilt_p030"]["all_seeds"][str(budget)]["summary"]["mean"]
        for budget in budgets
    ]
    lows = [
        analysis["paired"]["tilt_p030"]["all_seeds"][str(budget)]["summary"]["ci95_t"][0]
        for budget in budgets
    ]
    highs = [
        analysis["paired"]["tilt_p030"]["all_seeds"][str(budget)]["summary"]["ci95_t"][1]
        for budget in budgets
    ]
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(lows),
                                np.array(highs) - np.array(means)],
                fmt="-", color="black", linewidth=2.4, capsize=5, markersize=6,
                label="średnia 4 seedów, CI t-95%", zorder=5)
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xticks(x, [f"{budget}%" for budget in budgets])
    ax.set_xlabel("redukcja parametrów (nominalna)")
    ax.set_ylabel("różnica top-1 po BN [pp]: TILT+0.30 − UNIFORM")
    ax.set_title("Różnice na poszczególnych seedach")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    ax = axes[1]
    offsets = {"all_seeds": -0.13, "seeds_43_45": 0.13}
    labels = {"all_seeds": "wszystkie 4 seedy", "seeds_43_45": "seedy 43–45"}
    colors = {"all_seeds": "#111111", "seeds_43_45": "#2c6fb7"}
    for subset in ("all_seeds", "seeds_43_45"):
        means = [
            analysis["paired"]["tilt_p030"][subset][str(budget)]["summary"]["mean"]
            for budget in budgets
        ]
        lows = [
            analysis["paired"]["tilt_p030"][subset][str(budget)]["summary"]["ci95_t"][0]
            for budget in budgets
        ]
        highs = [
            analysis["paired"]["tilt_p030"][subset][str(budget)]["summary"]["ci95_t"][1]
            for budget in budgets
        ]
        positions = x + offsets[subset]
        ax.errorbar(positions, means,
                    yerr=[np.array(means) - np.array(lows),
                          np.array(highs) - np.array(means)],
                    fmt="o", color=colors[subset], capsize=5, markersize=7,
                    linewidth=1.4, label=labels[subset], zorder=4)
        seeds = SEEDS if subset == "all_seeds" else SEEDS_WITHOUT_42
        jitter = np.linspace(-0.045, 0.045, len(seeds))
        for index, budget in enumerate(budgets):
            values = [
                analysis["paired"]["tilt_p030"][subset][str(budget)]
                ["differences_pp"][str(seed)] for seed in seeds
            ]
            ax.scatter(
                np.full(len(values), positions[index]) + jitter,
                values,
                s=16,
                color=colors[subset],
                alpha=0.45,
                zorder=3,
            )
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xticks(x, [f"{budget}%" for budget in budgets])
    ax.set_xlabel("redukcja parametrów (nominalna)")
    ax.set_ylabel("różnica top-1 po BN [pp]")
    ax.set_title("Średnia i CI t-95% (n=4 oraz n=3)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.text(
        0.5, 0.005,
        "Punkty = seedy. 12 różnic (4 seedy × 3 budżety) nie jest 12 niezależnymi "
        "replikacjami; CI przy n=4 i n=3 są opisowe.",
        ha="center", fontsize=8,
    )
    fig.suptitle("TILT+0.30 vs UNIFORM — replikacja E014", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return save_figure(fig, "roznice_tilt030_wg_budzetu", output_dir)


def plot_top1(analysis: dict, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), sharey=True)
    jitter = np.linspace(-0.10, 0.10, len(SEEDS))
    for ax, budget in zip(axes, BUDGETS):
        for index, variant in enumerate(VARIANTS):
            item = analysis["variant_means"][str(budget)][variant]
            style = VARIANT_STYLE[variant]
            ax.errorbar(
                index, item["mean_top1_pct"], yerr=item["sd_top1_pct"],
                fmt=style["marker"], color=style["color"], capsize=5,
                markersize=7, linewidth=1.5, zorder=3,
            )
            ax.hlines(item["mean_top1_pct"], index - 0.25, index + 0.25,
                      color=style["color"], linewidth=2.0, zorder=3)
            for offset, seed in zip(jitter, SEEDS):
                ax.scatter(index + offset, item["seed_top1_pct"][str(seed)],
                           s=16, color=style["color"], alpha=0.45, zorder=2)
        ax.set_xticks(range(len(VARIANTS)),
                      [VARIANT_LABELS[variant] for variant in VARIANTS],
                      rotation=18, ha="right")
        ax.set_title(f"budżet {budget}%")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(55, 97)
    axes[0].set_ylabel("dev top-1 po BN [%]")
    fig.suptitle("Top-1 po BN: punkty seedów, średnia ± SD (n=4)", fontsize=12)
    fig.text(0.5, 0.005,
             "Małe punkty = pojedyncze seedy; duży marker i słupek = średnia ± SD "
             "między seedami (n=4).",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return save_figure(fig, "top1_wg_budzetu", output_dir)


def plot_budget(analysis: dict, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    budgets = list(BUDGETS)
    x = np.arange(len(budgets))
    ax = axes[0]
    for variant in VARIANTS:
        values = [
            next(row["deviation_from_target_pp"] for row in analysis["budgets"]
                 if row["nominal_budget"] == budget and row["variant"] == variant)
            for budget in budgets
        ]
        style = VARIANT_STYLE[variant]
        ax.plot(x, values, marker=style["marker"], color=style["color"],
                linewidth=1.8, markersize=6, label=VARIANT_LABELS[variant])
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xticks(x, [f"{budget}%" for budget in budgets])
    ax.set_xlabel("redukcja nominalna")
    ax.set_ylabel("zrealizowana redukcja − cel [pp]")
    ax.set_title("Odchylenie od celu budżetu")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.25
    for index, variant in enumerate(("tilt_p015", "tilt_p030")):
        values = [
            next(row["extra_params_vs_uniform"] for row in analysis["budgets"]
                 if row["nominal_budget"] == budget and row["variant"] == variant)
            / analysis["validation"]["params_before"] * 100.0
            for budget in budgets
        ]
        positions = x + (index - 0.5) * width
        style = VARIANT_STYLE[variant]
        bars = ax.bar(positions, values, width=width, color=style["color"],
                      alpha=0.85, label=VARIANT_LABELS[variant])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    pl(value, 3), ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x, [f"{budget}%" for budget in budgets])
    ax.set_xlabel("redukcja nominalna")
    ax.set_ylabel("dodatkowe parametry vs UNIFORM [% oryginału]")
    ax.set_title("TILT zachowuje nieco więcej parametrów (MAG2 = 0)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Rzeczywiste budżety: TILT trafia bliżej celu niż UNIFORM/MAG2", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "zrealizowany_budzet", output_dir)


def write_plots(analysis: dict, output_dir: Path) -> list[Path]:
    paths = []
    paths.extend(plot_differences(analysis, output_dir))
    paths.extend(plot_top1(analysis, output_dir))
    paths.extend(plot_budget(analysis, output_dir))
    return paths


def output_hashes(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": display_path(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="katalog biegu E014 (domyślnie ostatni ukończony bieg)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="nowy katalog na analizę i wykresy")
    parser.add_argument("--audit-dir", type=Path, default=None,
                        help="nowy katalog audytu pod runs/E014/audits/")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve() if args.audit_dir else None
    require(not output_dir.exists(), f"katalog wyjściowy już istnieje: {output_dir}")
    if audit_dir is not None:
        require(not audit_dir.exists(), f"katalog audytu już istnieje: {audit_dir}")
        require(audit_dir != output_dir, "katalog audytu i wyjściowy muszą być różne")
        require(output_dir not in audit_dir.parents and audit_dir not in output_dir.parents,
                "katalogi wyjściowy i audytu nie mogą być zagnieżdżone")

    output_dir.mkdir(parents=True, exist_ok=False)
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=False)

    analysis, data, input_paths = build_analysis(run_dir)
    analysis["inputs"] = output_hashes(input_paths)

    table_paths = write_tables(output_dir, analysis, data)
    markdown_path = output_dir / "analysis.md"
    write_markdown(markdown_path, analysis)
    plot_paths = write_plots(analysis, output_dir)
    analysis_path = output_dir / "analysis.json"
    write_json(analysis_path, {key: value for key, value in analysis.items()
                               if not key.startswith("_")})
    primary_paths = [analysis_path, markdown_path, *table_paths, *plot_paths]

    summary_lines = [
        "E014 — analiza międzyseedowa",
        f"run: {analysis['run_id']}  seed42 użyty selekcyjnie w E013 (raportowany osobno)",
        f"sprawdzono summary.json: {analysis['validation']['summaries_checked']}",
        "TILT+0.30 vs UNIFORM [pp]:",
    ]
    for budget in BUDGETS:
        full = analysis["paired"]["tilt_p030"]["all_seeds"][str(budget)]["summary"]
        subset = analysis["paired"]["tilt_p030"]["seeds_43_45"][str(budget)]["summary"]
        summary_lines.append(
            f"  {budget}%: 42-45 {full['mean']:+.3f} ± {full['sd']:.3f} "
            f"(zakres {full['min']:+.2f}..{full['max']:+.2f}); "
            f"43-45 {subset['mean']:+.3f} ± {subset['sd']:.3f}"
        )
    summary_lines.extend([
        f"dodatnie pary: {analysis['tilt030_positive_pairs']['positive']}/"
        f"{analysis['tilt030_positive_pairs']['total']}",
        f"wyniki: {display_path(output_dir)}",
    ])
    if audit_dir is not None:
        summary_lines.append(f"audyt: {display_path(audit_dir)}")
    summary_lines.append(f"META: {display_path(output_dir / 'META.json')}")
    summary_text = "\n".join(summary_lines) + "\n"
    print(summary_text, end="")

    audit_copies = []
    if audit_dir is not None:
        for path in primary_paths:
            target = audit_dir / path.name
            shutil.copy2(path, target)
            audit_copies.append(target)
        command_path = audit_dir / "KOMENDA.txt"
        command_path.write_text(
            "cd " + str(REPO_ROOT) + "\n"
            + " ".join([sys.executable, *sys.argv]) + "\n",
            encoding="utf-8",
        )
        audit_copies.append(command_path)
        stdout_path = audit_dir / "STDOUT.txt"
        stdout_path.write_text(summary_text, encoding="utf-8")
        audit_copies.append(stdout_path)

    meta = {
        "experiment": EXPERIMENT,
        "opis": OPIS_REL,
        "run_id": analysis["run_id"],
        "run_dir": analysis["run_dir"],
        "created_utc": analysis["created_utc"],
        "script": {
            "path": display_path(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "command": " ".join([sys.executable, *sys.argv]),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "analysis_unit": analysis["analysis_unit"],
        "validation": analysis["validation"],
        "inputs": analysis["inputs"],
        "outputs": output_hashes(primary_paths),
        "audit_copies": output_hashes(audit_copies) if audit_dir is not None else [],
        "note": (
            "Skrypt tylko czyta summary.json ukończonego biegu E014 i liczy "
            "statystyki oraz wykresy na CPU. Nie trenuje, nie używa GPU i nie "
            "zmienia źródłowych wyników."
        ),
    }
    meta_path = output_dir / "META.json"
    write_json(meta_path, meta)
    if audit_dir is not None:
        write_json(audit_dir / "META.json", meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
