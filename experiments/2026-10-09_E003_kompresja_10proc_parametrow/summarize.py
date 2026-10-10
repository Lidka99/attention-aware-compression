"""E003: analiza wyników i wykresy (tylko CPU, bez GPU, danych i checkpointów).

Eksperyment: **2026-10-09_E003_kompresja_10proc_parametrow**.
Pełny opis zakresu i wspólnego operatora: ``OPIS.md`` w tym samym katalogu.
Zamrożoną wersję protokołu (przygotowywaną osobno przez agenta głównego)
oznacza ``PROTOKOL_v1.md``.

Cel
---
Zebrać i zweryfikować wyniki gotowego biegu E003 oraz narysować wykresy
podsumowujące. Skrypt liczy sparowany bootstrap POST−PRE na tych samych
5000 obrazach dev, sprawdza zgodność surowych metryk z predykcjami i planem
parametrów oraz tworzy cztery rysunki (PNG/PDF). Nie trenuje, nie kompresuje,
nie uruchamia inferencji, nie czyta obrazów ani checkpointów, nie używa GPU.
Nie tworzy naukowej interpretacji ``WYNIKI.md`` — to należy do agenta głównego.

Wejścia (katalog ``--run-dir``)
-------------------------------
- ``config.resolved.json`` — rozwiązana konfiguracja biegu (tryby, liczby,
  seed i liczba powtórzeń bootstrapu, oczekiwane parametry, baseline).
- ``raw_metrics.json`` — ``{modes: {BASE/PRE/POST/RANDOM: {correct, count,
  top1, nll, parameters, model_sha256}}, count, unique_ids,
  baseline_reproduced, model_files: {mode: {bytes, sha256}},
  calibration_count, diagnostic_count}``.
- ``dev_predictions.npz`` — ``ids`` (int, N), ``targets`` (int, N),
  ``logits_BASE/PRE/POST/RANDOM`` (float32, N×100).
- ``block_errors.json`` — ``{PRE/POST/RANDOM: {blockpath:
  {branch_relative_mse: float|None, block_relative_mse: float|None}}}``.
- ``latency.json`` — lista ``{rank, gpu, mode, milliseconds: [50 float]}``.

Wyjścia
-------
- ``summary.json`` w katalogu biegu i w ``--plots-dir`` (dopiero po sukcesie
  wszystkich wykresów): status, tryby, pliki modeli, sparowane POST−PRE,
  latencje per rank, błędy bloków, kontrole i ograniczenia (po polsku).
- ``metrics.csv`` i ``latency_summary.csv`` w ``--plots-dir``.
- Cztery wykresy PNG i PDF: trafność czterech trybów, latencja per GPU i tryb,
  mapa błędów lokalnych gałęzi (log10) oraz liczba parametrów i rozmiar plików.

Uruchomienie
------------
Analiza istniejącego biegu::

    conda run -n attention python -B summarize.py \\
        --run-dir <KATALOG_BIEGU> --plots-dir <NOWY_KATALOG>

Kontrola funkcji czystej na danych syntetycznych (bez plików)::

    conda run -n attention python -B summarize.py --self-test

Odmawiamy nadpisania: ``--plots-dir`` nie może istnieć, a ``summary.json``
w katalogu biegu nie może już istnieć. Skrypt nie wykonuje i nie potwierdza
samodzielnie przebiegu kompresji — analizuje wyłącznie przekazane pliki.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_ID = "2026-10-09_E003_kompresja_10proc_parametrow"
OPIS_PATH = "OPIS.md"
PROTOKOL_V1_PATH = "PROTOKOL_v1.md"

MODES = ("BASE", "PRE", "POST", "RANDOM")
COMPRESSED_MODES = ("PRE", "POST", "RANDOM")
PAIR_PRE = "PRE"
PAIR_POST = "POST"

NUM_CLASSES = 100
DEV_COUNT = 5000
PER_CLASS = DEV_COUNT // NUM_CLASSES
BASE_PARAMETERS = 4135648
EXPECTED_COMPRESSED_PARAMETERS = 3721424
EXPECTED_REMOVED_PARAMETERS = 414224
BASELINE_CORRECT = 3342
CALIBRATION_COUNT = 500
DIAGNOSTIC_COUNT = 500

BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261009
CONFIDENCE = 0.95
NORM_ZERO_THRESHOLD = 1e-12
LATENCY_MEASUREMENTS = 50

COLORS = {"BASE": "#6b7280", "PRE": "#2563eb", "POST": "#dc2626", "RANDOM": "#16a34a"}

LIMITATIONS = (
    "Wyniki dotyczą jednego checkpointu E001 (seed 42); nie są niezależnym "
    "testem generalizacji ani dowodem przewagi metody.",
    "Dev brał udział w wyborze checkpointu E001, więc nie jest niezależnym "
    "zbiorem testowym; bootstrap opisuje zmienność obrazów przy jednym modelu, "
    "a nie zmienność między treningami.",
    "POST−PRE to różnica przy jednym wspólnym budżecie i jednym operatorze; "
    "samo wystąpienie różnicy nie dowodzi korzyści kryterium po SE.",
    "Cztery GPU to cztery procesy oceny tych samych modeli, nie cztery "
    "niezależne modele ani niezależne seedy.",
    "Analiza obejmuje wyłącznie pliki przekazane w --run-dir; skrypt nie "
    "wykonuje kompresji ani inferencji i nie potwierdza samodzielnie przebiegu "
    "eksperymentu.",
    "Zapis nie jest końcową interpretacją naukową; raport WYNIKI.md tworzy "
    "agent główny na podstawie tego podsumowania.",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Zapis JSON bez NaN/Infinity (allow_nan=False), odtwarzalny i z nową linią."""
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _display_log10(value):
    """Wartość do mapy: log10 z dolnym obcięciem do 1e-12; None -> NaN (maska)."""
    if value is None:
        return float("nan")
    number = float(value)
    if not math.isfinite(number):
        return float("nan")
    return math.log10(max(number, NORM_ZERO_THRESHOLD))


def paired_bootstrap(correctpre, correctpost, targets, repetitions=BOOTSTRAP_REPETITIONS,
                     seed=BOOTSTRAP_SEED):
    """Sparowany bootstrap POST−PRE, stratyfikowany po klasach (czysta funkcja).

    ``correctpre`` i ``correctpost`` to tablice logiczne poprawności dla tych
    samych obrazów; ``targets`` to etykiety klas. W każdej z ``repetitions``
    powtórzeń losujemy ze zwracaniem ``len(klasy)`` obrazów w każdej klasie
    (tu 50), liczymy różnicę średniej poprawności POST−PRE i zbieramy rozkład.
    Przedział to percentyle 2.5/97.5. Zwracamy też liczby rozbieżne (post_only,
    pre_only) oraz zgodne (both_correct, both_wrong). Funkcja nie mutuje wejść
    i nie używa GPU ani plików; jest deterministyczna dla danego seeda.

    Returns
    -------
    dict z kluczami ``delta_correct``, ``delta_top1``, ``ci95_correct``,
    ``ci95_top1``, ``confidence``, ``repetitions``, ``seed``, ``post_only``,
    ``pre_only``, ``both_correct``, ``both_wrong``, ``stratified_by``,
    ``per_class``, ``scope``.
    """
    pre = np.asarray(correctpre).astype(bool, copy=False).ravel()
    post = np.asarray(correctpost).astype(bool, copy=False).ravel()
    labels = np.asarray(targets).ravel()
    require(pre.shape == post.shape == labels.shape,
            "paired_bootstrap: niezgodne kształty correctpre/correctpost/targets")
    require(pre.size > 0, "paired_bootstrap: puste tablice")
    require(int(repetitions) > 0, "paired_bootstrap: repetitions musi być > 0")

    groups = [np.flatnonzero(labels == cls) for cls in np.unique(labels)]
    require(all(len(group) > 0 for group in groups), "paired_bootstrap: klasa bez próbek")
    sizes = {len(group) for group in groups}
    require(len(sizes) == 1, "paired_bootstrap: wymagane równe liczebności klas")
    per_class = int(next(iter(sizes)))
    total = int(pre.size)

    rng = np.random.default_rng(int(seed))
    distribution = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        difference = 0.0
        for group in groups:
            draw = group[rng.integers(0, group.size, group.size)]
            difference += float(post[draw].sum()) - float(pre[draw].sum())
        distribution[repetition] = difference / total

    delta_top1 = float(post.mean()) - float(pre.mean())
    delta_correct = int(post.sum()) - int(pre.sum())
    low, high = np.percentile(distribution, [2.5, 97.5])
    ci_top1 = [float(low), float(high)]
    ci_correct = [int(round(ci_top1[0] * total)), int(round(ci_top1[1] * total))]

    return {
        "delta_correct": delta_correct,
        "delta_top1": delta_top1,
        "ci95_correct": ci_correct,
        "ci95_top1": ci_top1,
        "confidence": CONFIDENCE,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "post_only": int(np.count_nonzero(~pre & post)),
        "pre_only": int(np.count_nonzero(pre & ~post)),
        "both_correct": int(np.count_nonzero(pre & post)),
        "both_wrong": int(np.count_nonzero(~pre & ~post)),
        "stratified_by": "class",
        "per_class": per_class,
        "scope": "one_checkpoint_not_generalization",
    }


def cfg_value(cfg, key, default):
    return cfg[key] if key in cfg else default


def verify_raw_metrics(raw):
    """Kontrole surowych metryk: liczebność, parametry, pliki modeli, baseline."""
    require(isinstance(raw, dict), "raw_metrics.json: oczekiwano obiektu")
    require(set(raw.get("modes", {})) == set(MODES),
            "raw_metrics.json: brakuje trybów BASE/PRE/POST/RANDOM")
    require(int(raw["count"]) == DEV_COUNT, "raw_metrics.json: count != 5000")
    require(int(raw["unique_ids"]) == DEV_COUNT, "raw_metrics.json: unique_ids != 5000")
    require(bool(raw["baseline_reproduced"]) is True,
            "raw_metrics.json: baseline_reproduced != true")
    require(int(raw.get("calibration_count", -1)) == CALIBRATION_COUNT,
            "raw_metrics.json: calibration_count != 500")
    require(int(raw.get("diagnostic_count", -1)) == DIAGNOSTIC_COUNT,
            "raw_metrics.json: diagnostic_count != 500")
    require(isinstance(raw.get("model_files"), dict), "raw_metrics.json: brak model_files")

    parameters = {}
    for mode in MODES:
        entry = raw["modes"][mode]
        for key in ("correct", "count", "top1", "nll", "parameters", "model_sha256"):
            require(key in entry, f"raw_metrics.json: modes[{mode}] bez '{key}'")
        require(int(entry["count"]) == DEV_COUNT, f"raw_metrics.json: {mode} count != 5000")
        expected = BASE_PARAMETERS if mode == "BASE" else EXPECTED_COMPRESSED_PARAMETERS
        require(int(entry["parameters"]) == expected,
                f"raw_metrics.json: {mode} parameters != {expected}")
        require(isinstance(entry["model_sha256"], str) and len(entry["model_sha256"]) == 64,
                f"raw_metrics.json: {mode} model_sha256 nieprawidłowy")
        files = raw["model_files"].get(mode)
        require(isinstance(files, dict) and "bytes" in files and "sha256" in files,
                f"raw_metrics.json: model_files[{mode}] niekompletne")
        parameters[mode] = int(entry["parameters"])

    require(int(raw["modes"]["BASE"]["correct"]) == BASELINE_CORRECT,
            "raw_metrics.json: BASE correct != 3342")
    compressed = {parameters[mode] for mode in COMPRESSED_MODES}
    require(compressed == {EXPECTED_COMPRESSED_PARAMETERS},
            "raw_metrics.json: skompresowane tryby nie mają wspólnego budżetu 3721424")
    return {
        "count": int(raw["count"]),
        "unique_ids": int(raw["unique_ids"]),
        "baseline_reproduced": True,
        "baseline_correct": int(raw["modes"]["BASE"]["correct"]),
        "parameters": parameters,
        "compressed_same_budget": True,
        "model_files_present": True,
    }


def verify_predictions(npz_path, raw):
    """Kontrole predykcji: ID, balans klas, logity i zgodność z raw_metrics."""
    with np.load(npz_path, allow_pickle=False) as handle:
        keys = set(handle.files)
        for required in ("ids", "targets", *(f"logits_{mode}" for mode in MODES)):
            require(required in keys, f"dev_predictions.npz: brak klucza '{required}'")
        ids = handle["ids"].copy()
        targets = handle["targets"].copy()
        logits = {mode: handle[f"logits_{mode}"].copy() for mode in MODES}

    require(ids.ndim == 1 and targets.ndim == 1, "dev_predictions.npz: ids/targets nie 1D")
    require(ids.shape == targets.shape, "dev_predictions.npz: ids/targets różne długości")
    require(ids.size == DEV_COUNT, "dev_predictions.npz: N != 5000")
    require(np.issubdtype(ids.dtype, np.integer), "dev_predictions.npz: ids nie int")
    require(np.issubdtype(targets.dtype, np.integer), "dev_predictions.npz: targets nie int")
    require(int(np.unique(ids).size) == DEV_COUNT, "dev_predictions.npz: ID nie są unikalne")

    counts = np.bincount(targets.astype(np.int64), minlength=NUM_CLASSES)
    require(counts.size == NUM_CLASSES, "dev_predictions.npz: numer klasy poza zakresem")
    require(bool(np.all(counts == PER_CLASS)),
            "dev_predictions.npz: brak balansu 50 obrazów na klasę")

    correctness, logits_finite, match_raw, top1_match = {}, {}, {}, {}
    for mode in MODES:
        block = logits[mode]
        require(block.shape == (DEV_COUNT, NUM_CLASSES),
                f"dev_predictions.npz: logits_{mode} zły kształt")
        require(block.dtype == np.float32, f"dev_predictions.npz: logits_{mode} nie float32")
        finite = bool(np.all(np.isfinite(block)))
        require(finite, f"dev_predictions.npz: logits_{mode} zawierają NaN/Inf")
        predictions = block.argmax(axis=1)
        correct = predictions == targets
        entry = raw["modes"][mode]
        require(int(entry["correct"]) == int(correct.sum()),
                f"{mode}: argmax nie zgadza się z raw_metrics.correct")
        rate = float(correct.mean())
        require(abs(float(entry["top1"]) - rate) <= 1e-9,
                f"{mode}: top1 nie zgadza się z argmax")
        correctness[mode] = correct
        logits_finite[mode] = finite
        match_raw[mode] = True
        top1_match[mode] = [float(entry["top1"]), rate]

    require(int(correctness["BASE"].sum()) == BASELINE_CORRECT,
            "dev_predictions.npz: BASE nie odtwarza 3342/5000")

    return {
        "correctness": correctness,
        "targets": targets,
        "checks": {
            "count": DEV_COUNT,
            "unique_ids": int(np.unique(ids).size),
            "class_balance": {"per_class": PER_CLASS,
                              "min": int(counts.min()), "max": int(counts.max())},
            "logits_finite": logits_finite,
            "predictions_match_raw": match_raw,
            "top1_match_raw": top1_match,
            "baseline_correct": int(correctness["BASE"].sum()),
        },
    }


def verify_latency(latency, cfg):
    """Kontrole pomiarów czasu i podsumowanie median/kwartyli per rank i tryb."""
    require(isinstance(latency, list) and latency, "latency.json: oczekiwano niepustej listy")
    measurements = int(cfg_value(cfg, "latency_measurements", LATENCY_MEASUREMENTS))
    by_key = {}
    ranks = set()
    for entry in latency:
        for key in ("rank", "gpu", "mode", "milliseconds"):
            require(key in entry, f"latency.json: wpis bez '{key}'")
        rank = int(entry["rank"])
        mode = str(entry["mode"])
        require(mode in MODES, f"latency.json: nieznany tryb {mode!r}")
        require((rank, mode) not in by_key, f"latency.json: duplikat rank={rank} mode={mode}")
        values = np.asarray(entry["milliseconds"], dtype=np.float64)
        require(values.ndim == 1 and values.size == measurements,
                f"latency.json: rank={rank} mode={mode} != {measurements} pomiarów")
        require(bool(np.all(np.isfinite(values))), "latency.json: NaN/Inf w pomiarach")
        require(bool(np.all(values >= 0.0)), "latency.json: ujemny czas")
        by_key[(rank, mode)] = (str(entry["gpu"]), values)
        ranks.add(rank)

    require(ranks == {0, 1, 2, 3}, "latency.json: wymagane ranki 0,1,2,3")
    for rank in sorted(ranks):
        for mode in MODES:
            require((rank, mode) in by_key, f"latency.json: brak rank={rank} mode={mode}")

    summary = []
    for rank in sorted(ranks):
        for mode in MODES:
            gpu, values = by_key[(rank, mode)]
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            summary.append({
                "rank": rank,
                "gpu": gpu,
                "mode": mode,
                "median_ms": float(median),
                "q25_ms": float(q25),
                "q75_ms": float(q75),
                "measurements": int(values.size),
            })
    return summary, {"ranks": sorted(ranks), "entries": len(summary),
                     "measurements_per_entry": measurements}


def verify_block_errors(block_errors):
    """Kontrole błędów bloków; None oznacza wartość nieokreśloną (stary przypadek)."""
    require(isinstance(block_errors, dict), "block_errors.json: oczekiwano obiektu")
    require(set(block_errors) == set(COMPRESSED_MODES),
            "block_errors.json: oczekiwane tryby PRE/POST/RANDOM")
    blocks = list(block_errors[COMPRESSED_MODES[0]].keys())
    require(blocks, "block_errors.json: brak bloków")
    defined = 0
    undefined = 0
    for mode in COMPRESSED_MODES:
        require(set(block_errors[mode]) == set(blocks),
                f"block_errors.json: {mode} obejmuje inny zestaw bloków")
        for block, metrics in block_errors[mode].items():
            require(isinstance(metrics, dict), f"block_errors.json: {mode}/{block} nie obiekt")
            for key in ("branch_relative_mse", "block_relative_mse"):
                require(key in metrics, f"block_errors.json: {mode}/{block} bez '{key}'")
                value = metrics[key]
                if value is None:
                    undefined += 1
                else:
                    require(not isinstance(value, bool), f"{mode}/{block}/{key} bool")
                    number = float(value)
                    require(math.isfinite(number) and number >= 0.0,
                            f"block_errors.json: {mode}/{block}/{key} nieprawidłowe")
                    defined += 1
    return {"modes": list(COMPRESSED_MODES), "blocks": len(blocks),
            "defined_values": defined, "undefined_none": undefined}


def save_figure(fig, plots_dir, stem, note):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.02, 0.01, note, fontsize=8)
    fig.savefig(Path(plots_dir) / f"{stem}.png", dpi=160)
    fig.savefig(Path(plots_dir) / f"{stem}.pdf")
    plt.close(fig)


def plot_accuracy(raw, plots_dir):
    """Wykres 1: trafność czterech trybów z jawnym mianownikiem 5000."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    correct = [int(raw["modes"][mode]["correct"]) for mode in MODES]
    total = [int(raw["modes"][mode]["count"]) for mode in MODES]
    heights = [100.0 * c / t for c, t in zip(correct, total)]
    bars = ax.bar(list(MODES), heights, color=[COLORS[mode] for mode in MODES])
    for bar, c, t, h in zip(bars, correct, total, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{c}/{t}\n{h:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Trafność top-1 dev [%]", ylim=(0, 100),
           title="E003: trafność dev na 5000 obrazach (BASE/PRE/POST/RANDOM)")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, plots_dir, "accuracy_by_mode",
                "Jeden checkpoint E001 (seed 42); mianownik każdego słupka to 5000 obrazów. "
                "Dev wcześniej wybierał checkpoint, więc nie jest niezależnym testem.")


def plot_latency(latency_summary, plots_dir):
    """Wykres 2: mediana czasu per GPU i tryb, 16 słupków (12+4), błąd kwartylowy."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ranks = sorted({entry["rank"] for entry in latency_summary})
    index = {(entry["rank"], entry["mode"]): entry for entry in latency_summary}
    width = 0.8 / len(MODES)
    centers = []
    for position, rank in enumerate(ranks):
        for mode_index, mode in enumerate(MODES):
            entry = index[(rank, mode)]
            x = position + (mode_index - (len(MODES) - 1) / 2) * width
            median = entry["median_ms"]
            lower = median - entry["q25_ms"]
            upper = entry["q75_ms"] - median
            ax.bar(x, median, width=width, color=COLORS[mode],
                   yerr=[[lower], [upper]], capsize=2, error_kw={"elinewidth": 1})
        centers.append(position)
        gpu = index[(rank, "BASE")]["gpu"]
        ax.text(position, -0.06 * max(index[(r, m)]["q75_ms"]
                for r in ranks for m in MODES), f"rank {rank}\n{gpu}",
                ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
    ax.set_xticks(centers)
    ax.set_xticklabels([f"rank {r}" for r in ranks])
    ax.set(ylabel="Czas forwardu [ms], mediana i kwartyle",
           title="E003: latencja per GPU i tryb (12 skompresowanych + 4 BASE)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[mode]) for mode in MODES]
    ax.legend(handles, MODES, fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, plots_dir, "latency_by_gpu_mode",
                "Batch 32, 10 rozgrzewek i 50 pomiarów na kartę. Cztery GPU to cztery "
                "procesy oceny, nie cztery niezależne modele; oszczędność parametrów "
                "nie musi oznaczać przyspieszenia.")


def plot_block_errors(block_errors, plots_dir):
    """Wykres 3: mapa log10 błędu lokalnej gałęzi; None maskowane, 0 obcinane do 1e-12."""
    blocks = list(block_errors[COMPRESSED_MODES[0]].keys())
    values = np.full((len(blocks), len(COMPRESSED_MODES)), np.nan)
    for column, mode in enumerate(COMPRESSED_MODES):
        for row, block in enumerate(blocks):
            values[row, column] = _display_log10(
                block_errors[mode][block]["branch_relative_mse"])
    masked = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(6, 9))
    image = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(COMPRESSED_MODES)))
    ax.set_xticklabels(COMPRESSED_MODES)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks, fontsize=8)
    for row in range(len(blocks)):
        for column in range(len(COMPRESSED_MODES)):
            if np.isnan(values[row, column]):
                ax.text(column, row, "—", ha="center", va="center", color="#9ca3af")
            else:
                ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(image, ax=ax, label="log10(branch_relative_mse)")
    ax.set_title("E003: błąd lokalnej gałęzi po blokach")
    save_figure(fig, plots_dir, "branch_error_heatmap",
                "Wartości pokazano jako log10; 0.0 obcięto do 1e-12, a None (cel zerowy, "
                "nieokreślone) maskowano. Oryginalne wartości zachowano w summary.json.")


def plot_parameters_filesize(raw, plots_dir):
    """Wykres 4: liczba parametrów i rozmiar plików modeli (te same budżety)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    parameters = [int(raw["modes"][mode]["parameters"]) for mode in MODES]
    files = [int(raw["model_files"][mode]["bytes"]) for mode in MODES]
    axes[0].bar(list(MODES), parameters, color=[COLORS[mode] for mode in MODES])
    axes[0].set_title("Liczba parametrów")
    axes[0].set_ylabel("parametry")
    for position, value in enumerate(parameters):
        axes[0].text(position, value, f"{value}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(list(MODES), files, color=[COLORS[mode] for mode in MODES])
    axes[1].set_title("Rozmiar pliku modelu")
    axes[1].set_ylabel("bajty")
    for position, value in enumerate(files):
        axes[1].text(position, value, f"{value}", ha="center", va="bottom", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, plots_dir, "parameters_and_filesize",
                "BASE: 4135648 parametrów; PRE/POST/RANDOM wspólny budżet 3721424 "
                "(`10.0159%` mniej, 414224 parametry mniej). To plan i pomiar biegu, "
                "nie deklaracja przewagi metody.")


def write_metrics_csv(raw, path):
    columns = ["mode", "correct", "count", "top1", "nll", "parameters",
               "model_sha256", "file_bytes", "file_sha256"]
    with Path(path).open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for mode in MODES:
            entry = raw["modes"][mode]
            files = raw["model_files"][mode]
            writer.writerow([mode, entry["correct"], entry["count"], entry["top1"],
                             entry["nll"], entry["parameters"], entry["model_sha256"],
                             files["bytes"], files["sha256"]])


def write_latency_csv(latency_summary, path):
    columns = ["rank", "gpu", "mode", "median_ms", "q25_ms", "q75_ms", "measurements"]
    with Path(path).open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for entry in latency_summary:
            writer.writerow([entry["rank"], entry["gpu"], entry["mode"], entry["median_ms"],
                             entry["q25_ms"], entry["q75_ms"], entry["measurements"]])


def run_analysis(run_dir, plots_dir):
    run_dir = Path(run_dir).resolve()
    plots_dir = Path(plots_dir).resolve()
    require(run_dir.is_dir(), f"--run-dir nie istnieje: {run_dir}")
    require(not (run_dir / "summary.json").exists(),
            "Odmowa: summary.json już istnieje w katalogu biegu")
    require(not plots_dir.exists(), f"Odmowa: katalog wykresów już istnieje: {plots_dir}")

    config_path = run_dir / "config.resolved.json"
    require(config_path.is_file(), f"Brak {config_path}")
    cfg = read_json(config_path)
    raw = read_json(run_dir / "raw_metrics.json")
    block_errors = read_json(run_dir / "block_errors.json")
    latency = read_json(run_dir / "latency.json")

    raw_checks = verify_raw_metrics(raw)
    prediction = verify_predictions(run_dir / "dev_predictions.npz", raw)
    latency_summary, latency_checks = verify_latency(latency, cfg)
    block_checks = verify_block_errors(block_errors)

    repetitions = int(cfg_value(cfg, "bootstrap_repetitions", BOOTSTRAP_REPETITIONS))
    seed = int(cfg_value(cfg, "bootstrap_seed", BOOTSTRAP_SEED))
    confidence = float(cfg_value(cfg, "bootstrap_confidence", CONFIDENCE))
    require(abs(confidence - CONFIDENCE) < 1e-12,
            "config.resolved.json: bootstrap_confidence != 0.95")

    paired = paired_bootstrap(prediction["correctness"][PAIR_PRE],
                              prediction["correctness"][PAIR_POST],
                              prediction["targets"], repetitions, seed)

    checks = {
        "raw_metrics": raw_checks,
        "predictions": prediction["checks"],
        "parameters_match_plan": raw_checks["parameters"] == {
            "BASE": BASE_PARAMETERS,
            "PRE": EXPECTED_COMPRESSED_PARAMETERS,
            "POST": EXPECTED_COMPRESSED_PARAMETERS,
            "RANDOM": EXPECTED_COMPRESSED_PARAMETERS,
        },
        "compressed_same_budget": raw_checks["compressed_same_budget"],
        "baseline_correct_3342": raw_checks["baseline_correct"] == BASELINE_CORRECT,
        "unique_ids_5000": prediction["checks"]["unique_ids"] == DEV_COUNT,
        "class_balanced_50": prediction["checks"]["class_balance"]["min"] == PER_CLASS
        and prediction["checks"]["class_balance"]["max"] == PER_CLASS,
        "latency": latency_checks,
        "block_errors": block_checks,
    }

    plots_dir.mkdir(parents=True, exist_ok=False)
    plot_accuracy(raw, plots_dir)
    plot_latency(latency_summary, plots_dir)
    plot_block_errors(block_errors, plots_dir)
    plot_parameters_filesize(raw, plots_dir)
    write_metrics_csv(raw, plots_dir / "metrics.csv")
    write_latency_csv(latency_summary, plots_dir / "latency_summary.csv")

    summary = {
        "status": "completed",
        "experiment": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "existing_run_inputs_only",
        "modes": raw["modes"],
        "model_files": raw["model_files"],
        "paired_post_pre": paired,
        "latency_by_rank": latency_summary,
        "block_errors": block_errors,
        "checks": checks,
        "bootstrap": {"repetitions": repetitions, "seed": seed,
                      "confidence": confidence, "stratified_by": "class",
                      "per_class": PER_CLASS},
        "plots": ["accuracy_by_mode", "latency_by_gpu_mode",
                  "branch_error_heatmap", "parameters_and_filesize"],
        "limitations": list(LIMITATIONS),
    }

    write_json(plots_dir / "summary.json", summary)
    write_json(run_dir / "summary.json", summary)
    print(f"SUMMARIZE=completed; delta_correct={paired['delta_correct']}; "
          f"delta_top1={paired['delta_top1']:.6f}; "
          f"ci95_top1=[{paired['ci95_top1'][0]:.6f}, {paired['ci95_top1'][1]:.6f}]; "
          f"latency_entries={len(latency_summary)}; plots={len(summary['plots'])}")


def _self_test():
    """Kontrole CPU czystej funkcji paired_bootstrap i mapy log10 na danych syntetycznych."""
    targets = np.repeat(np.arange(NUM_CLASSES), 5)
    rng = np.random.default_rng(20261009)

    pre = rng.random(targets.size) < 0.5
    post = pre.copy()
    same = paired_bootstrap(pre, post, targets, 200, 20261009)
    require(same["delta_top1"] == 0.0, "identyczne tablice: delta_top1 != 0")
    require(same["delta_correct"] == 0, "identyczne tablice: delta_correct != 0")
    require(same["ci95_top1"] == [0.0, 0.0], "identyczne tablice: CI != [0, 0]")
    require(same["ci95_correct"] == [0, 0], "identyczne tablice: CI liczby != [0, 0]")
    require(same["post_only"] == 0 and same["pre_only"] == 0,
            "identyczne tablice: brak rozbieżności")
    require(same["both_correct"] + same["both_wrong"] == targets.size,
            "identyczne tablice: zła suma zgodnych")

    pre_none = np.zeros(targets.size, dtype=bool)
    post_all = np.ones(targets.size, dtype=bool)
    full = paired_bootstrap(pre_none, post_all, targets, 200, 20261009)
    require(full["delta_top1"] == 1.0, "post wszystkie vs pre żadne: delta_top1 != 1")
    require(full["delta_correct"] == targets.size, "post wszystkie: delta_correct != N")
    require(full["ci95_top1"] == [1.0, 1.0], "post wszystkie: CI != [1, 1]")
    require(full["post_only"] == targets.size and full["pre_only"] == 0,
            "post wszystkie: złe liczby rozbieżne")

    require(math.isnan(_display_log10(None)), "None powinno dawać NaN (maskę)")
    require(abs(_display_log10(0.0) - math.log10(NORM_ZERO_THRESHOLD)) < 1e-12,
            "zero powinno być pokazane jako 1e-12")
    require(abs(_display_log10(1e-3) - (-3.0)) < 1e-12, "log10(1e-3) != -3")

    grouped = paired_bootstrap(
        np.array([True, False, True, False]), np.array([True, True, False, False]),
        np.array([0, 0, 1, 1]), 50, 7)
    require(grouped["both_correct"] == 1 and grouped["both_wrong"] == 1
            and grouped["post_only"] == 1 and grouped["pre_only"] == 1,
            "mała kontrola rozbieżności nie zgadza się")

    print("SUMMARIZE_SELF_TEST=passed; paired_bootstrap=identical[0,0];"
          "=post_all_vs_pre_none[1,1]; display_log10=masked_none,clipped_zero")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="katalog biegu E003 z config.resolved.json i wynikami")
    parser.add_argument("--plots-dir", type=Path,
                        help="nowy katalog na wykresy, CSV i summary.json")
    parser.add_argument("--self-test", action="store_true",
                        help="kontrole CPU na danych syntetycznych bez zapisu plików")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.run_dir is None and args.plots_dir is None:
        _self_test()
        return
    if args.run_dir is None or args.plots_dir is None:
        parser.error("--run-dir i --plots-dir należy podać razem")
    run_analysis(args.run_dir, args.plots_dir)


if __name__ == "__main__":
    main()
