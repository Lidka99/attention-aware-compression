"""E003: czytelne rysunki raportu z gotowego ``summary.json`` (tylko CPU, bez GPU).

Eksperyment: **2026-10-09_E003_kompresja_10proc_parametrow**.
To korekta prezentacji zakończonego biegu, nie nowy eksperyment i nie zmiana
wyników. Część zadania E003 „kompresja 10% parametrów całej sieci".

Cel
---
Z gotowego ``summary.json`` odtworzyć cztery rysunki (PNG i PDF) w czytelnym
układzie. Oryginalne rysunki mają długie stopki i ciasny/przycięty layout.
Ten skrypt używa jawnych marginesów ``fig.subplots_adjust`` (bez
``tight_layout``), krótkiej stopki zawijanej do dwóch linii i nie przycina
osi, tytułów ani etykiet. Nie liczy nic od nowa: korzysta wyłącznie
z wartości obecnych w ``summary.json``. Nie trenuje, nie kompresuje, nie
czyta modeli, obrazów ani checkpointów i nie używa GPU. Nie tworzy naukowej
interpretacji ``WYNIKI.md``.

Wejścia (``--summary``)
-----------------------
- ``summary.json`` biegu ``20261010T095706Z_5a2f233_seed42__0o_uckk``,
  SHA-256 ``ceeddb6ae36d488be5295c63cc60e686c087b194147e238d61af358f6b0759ad``.
  Odczytujemy tylko metadane:
  ``modes`` (trafność: ``correct``/``count``/``top1``), ``latency_by_rank``
  (``rank``/``gpu``/``mode``/``median_ms``/``q25_ms``/``q75_ms``),
  ``block_errors`` (``branch_relative_mse`` 3 tryby × 15 bloków) oraz
  ``model_files``/``modes`` (``parameters`` i ``bytes``). Skrypt sprawdza
  zgodność hasha źródła z oczekiwanym i odmawia pracy przy niezgodności.

Wyjścia (``--output-dir``)
--------------------------
Katalog ``--output-dir`` musi nie istnieć (odmawiamy nadpisania). Zapisujemy:
- cztery rysunki w PNG i PDF: ``accuracy_by_mode``, ``branch_error_heatmap``,
  ``latency_by_gpu_mode``, ``parameters_and_filesize``;
- ``META_RYSUNKOW.json``: ścieżka i hash źródła, SHA-256 tego skryptu,
  hashe wszystkich ośmiu plików rysunków, data UTC oraz jawne stwierdzenie
  ``nowe_obliczenia_naukowe = false``.

Uruchomienie
------------
::

    conda run -n attention python -B plot_report.py \\
        --summary /sciezka/do/summary.json --output-dir /nowy/katalog

Odniesienia
-----------
- Lokalny opis tego zadania rysunkowego: `OPIS.md` (w tym katalogu).
- Protokół eksperymentu: ``../../PROTOKOL_v1.md``.
- Pełny opis E003: ``../../OPIS.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENT_ID = "2026-10-09_E003_kompresja_10proc_parametrow"
RUN_ID = "20261010T095706Z_5a2f233_seed42__0o_uckk"
EXPECTED_SOURCE_SHA256 = (
    "ceeddb6ae36d488be5295c63cc60e686c087b194147e238d61af358f6b0759ad"
)

MODES = ("BASE", "PRE", "POST", "RANDOM")
COMPRESSED_MODES = ("PRE", "POST", "RANDOM")
COLORS = {"BASE": "#6b7280", "PRE": "#2563eb", "POST": "#dc2626", "RANDOM": "#16a34a"}

PLOT_STEMS = (
    "accuracy_by_mode",
    "branch_error_heatmap",
    "latency_by_gpu_mode",
    "parameters_and_filesize",
)

FOOTER_WIDTH = 100
FOOTER_TOP_Y = 0.115
FOOTER_STEP_Y = 0.055


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def wrap_lines(text, width):
    """Zawija tekst do linii o długości <= width (bez łamania wyrazów)."""
    lines = []
    current = ""
    for word in str(text).split():
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def add_footer(fig, text, width=FOOTER_WIDTH):
    """Krótka stopka zawijana do dwóch nieprzyciętych linii."""
    lines = wrap_lines(text, width)[:2]
    for index, line in enumerate(lines):
        fig.text(
            0.01,
            FOOTER_TOP_Y - index * FOOTER_STEP_Y,
            line,
            ha="left",
            va="bottom",
            fontsize=8,
        )


def save_figure(fig, output_dir, stem):
    """Zapis PNG i PDF bez tight_layout; marginesy ustawia sam wykres."""
    png = Path(output_dir) / f"{stem}.png"
    pdf = Path(output_dir) / f"{stem}.pdf"
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_accuracy(summary, output_dir):
    """Wykres 1: trafność czterech trybów, jawne liczniki i mianownik 5000."""
    modes = summary["modes"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    heights = [100.0 * int(modes[mode]["correct"]) / int(modes[mode]["count"]) for mode in MODES]
    bars = ax.bar(list(MODES), heights, color=[COLORS[mode] for mode in MODES])
    for bar, mode, height in zip(bars, MODES, heights):
        correct = int(modes[mode]["correct"])
        count = int(modes[mode]["count"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.6,
            f"{correct}/{count}\n{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set(
        ylabel="Trafność top-1 dev [%]",
        ylim=(0, 100),
        title="E003: trafność dev na 5000 obrazach (BASE/PRE/POST/RANDOM)",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.88, bottom=0.22)
    add_footer(
        fig,
        "Jeden checkpoint E001 (seed 42); mianownik każdego słupka to 5000 obrazów. "
        "Dev wcześniej wybierał checkpoint, więc nie jest niezależnym testem.",
    )
    return save_figure(fig, output_dir, "accuracy_by_mode")


def plot_latency(summary, output_dir):
    """Wykres 2: mediana czasu per GPU i tryb, pogrupowane 4 tryby × 4 GPU."""
    entries = summary["latency_by_rank"]
    ranks = sorted({int(entry["rank"]) for entry in entries})
    index = {(int(e["rank"]), e["mode"]): e for e in entries}
    for rank in ranks:
        for mode in MODES:
            require((rank, mode) in index, f"latency_by_rank: brak rank={rank} mode={mode}")

    fig, ax = plt.subplots(figsize=(10, 6.2))
    width = 0.8 / len(MODES)
    centers = []
    for position, rank in enumerate(ranks):
        for mode_index, mode in enumerate(MODES):
            entry = index[(rank, mode)]
            x = position + (mode_index - (len(MODES) - 1) / 2) * width
            median = float(entry["median_ms"])
            lower = median - float(entry["q25_ms"])
            upper = float(entry["q75_ms"]) - median
            ax.bar(
                x,
                median,
                width=width,
                color=COLORS[mode],
                yerr=[[lower], [upper]],
                capsize=2,
                error_kw={"elinewidth": 1},
            )
        centers.append(position)

    top = max(float(e["q75_ms"]) for e in entries)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [
            f"rank {rank}\n{index[(rank, 'BASE')]['gpu'].replace('NVIDIA GeForce ', '')}"
            for rank in ranks
        ],
        fontsize=8,
    )
    ax.set(
        ylabel="Czas forwardu [ms] — mediana i kwartyle",
        title="E003: latencja per GPU i tryb (12 skompresowanych + 4 BASE)",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[mode]) for mode in MODES]
    ax.legend(handles, MODES, fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.22)
    add_footer(
        fig,
        "Batch 32, 10 rozgrzewek i 50 pomiarów na kartę. Cztery GPU to cztery "
        "procesy oceny tych samych modeli, nie cztery niezależne seedy.",
    )
    return save_figure(fig, output_dir, "latency_by_gpu_mode")


def plot_block_errors(summary, output_dir):
    """Wykres 3: mapa log10 błędu lokalnej gałęzi, 3 tryby × 15 bloków."""
    import numpy as np

    block_errors = summary["block_errors"]
    blocks = list(block_errors[COMPRESSED_MODES[0]].keys())
    values = np.full((len(blocks), len(COMPRESSED_MODES)), np.nan)
    for column, mode in enumerate(COMPRESSED_MODES):
        for row, block in enumerate(blocks):
            raw = block_errors[mode][block]["branch_relative_mse"]
            if raw is None:
                continue
            number = float(raw)
            if number != number or number in (float("inf"), float("-inf")):
                continue
            values[row, column] = np.log10(max(number, 1e-12))
    masked = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(6.8, 9.2))
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
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
    fig.colorbar(image, ax=ax, label="log10(branch_relative_mse)", fraction=0.08, pad=0.03)
    ax.set_title("E003: błąd lokalnej gałęzi po blokach")
    fig.subplots_adjust(left=0.30, right=0.90, top=0.93, bottom=0.20)
    add_footer(
        fig,
        "Wartości jako log10(branch_relative_mse); 0.0 obcięto do 1e-12, None maskowano. "
        "Oryginalne wartości zachowano w summary.json.",
    )
    return save_figure(fig, output_dir, "branch_error_heatmap")


def plot_parameters_filesize(summary, output_dir):
    """Wykres 4: liczba parametrów i rozmiar plików modeli (dwa panele)."""
    modes = summary["modes"]
    files = summary["model_files"]
    parameters = [int(modes[mode]["parameters"]) for mode in MODES]
    sizes_mib = [int(files[mode]["bytes"]) / (1024 * 1024) for mode in MODES]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    axes[0].bar(list(MODES), parameters, color=[COLORS[mode] for mode in MODES])
    axes[0].set_title("Liczba parametrów")
    axes[0].set_ylabel("parametry")
    axes[0].set_ylim(0, max(parameters) * 1.12)
    for position, value in enumerate(parameters):
        axes[0].text(position, value, f"{value}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(list(MODES), sizes_mib, color=[COLORS[mode] for mode in MODES])
    axes[1].set_title("Rozmiar pliku modelu")
    axes[1].set_ylabel("Rozmiar pliku [MiB]")
    axes[1].set_ylim(0, max(sizes_mib) * 1.12)
    for position, value in enumerate(sizes_mib):
        axes[1].text(position, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.22, wspace=0.28)
    add_footer(
        fig,
        "BASE: 4135648 parametrów; PRE/POST/RANDOM wspólny budżet 3721424 "
        "(10.0159% mniej). To plan i pomiar biegu, nie deklaracja przewagi metody.",
    )
    return save_figure(fig, output_dir, "parameters_and_filesize")


def build_meta(summary_path, output_dir, script_path, produced):
    created = datetime.now(timezone.utc).isoformat()
    plot_hashes = {}
    for stem in PLOT_STEMS:
        png, pdf = produced[stem]
        plot_hashes[stem] = {
            "png": {"path": png.name, "sha256": sha256_file(png)},
            "pdf": {"path": pdf.name, "sha256": sha256_file(pdf)},
        }
    return {
        "experiment": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_path": str(Path(summary_path).resolve()),
        "source_sha256": sha256_file(summary_path),
        "script_path": str(Path(script_path).resolve()),
        "script_sha256": sha256_file(script_path),
        "created_utc": created,
        "nowe_obliczenia_naukowe": False,
        "uwaga": (
            "Skrypt tylko odczytuje summary.json i rysuje wykresy; nie wykonuje "
            "nowych obliczeń naukowych, pomiarów ani treningu (CPU, bez modelu)."
        ),
        "plots": list(PLOT_STEMS),
        "plot_sha256": plot_hashes,
    }


def generate(summary_path, output_dir, script_path):
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    require(summary_path.is_file(), f"brak pliku summary: {summary_path}")
    if output_dir.exists():
        raise RuntimeError(f"odmawiam nadpisania istniejącego katalogu: {output_dir}")

    actual = sha256_file(summary_path)
    require(
        actual == EXPECTED_SOURCE_SHA256,
        f"hash źródła niezgodny: {actual} != {EXPECTED_SOURCE_SHA256}",
    )
    summary = read_json(summary_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    produced = {
        "accuracy_by_mode": plot_accuracy(summary, output_dir),
        "branch_error_heatmap": plot_block_errors(summary, output_dir),
        "latency_by_gpu_mode": plot_latency(summary, output_dir),
        "parameters_and_filesize": plot_parameters_filesize(summary, output_dir),
    }
    meta = build_meta(summary_path, output_dir, script_path, produced)
    (output_dir / "META_RYSUNKOW.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="E003: czytelne rysunki raportu z gotowego summary.json (CPU)."
    )
    parser.add_argument("--summary", required=True, help="ścieżka do summary.json biegu")
    parser.add_argument(
        "--output-dir", required=True, help="nowy katalog wyjściowy (nie może istnieć)"
    )
    args = parser.parse_args(argv)
    try:
        meta = generate(args.summary, args.output_dir, Path(__file__))
    except Exception as error:  # noqa: BLE001 - jasny komunikat i kod wyjścia
        print(f"BŁĄD: {error}", file=sys.stderr)
        return 1
    print(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
