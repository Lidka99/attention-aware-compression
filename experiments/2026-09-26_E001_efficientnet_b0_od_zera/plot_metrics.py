"""E001: rysuje krzywe uczenia i zapisuje podsumowanie audytu (tylko CPU).

Eksperyment: 2026-09-26_E001_efficientnet_b0_od_zera. Protokół: AUDYT_OPIS_v1.md i OPIS.md
w tym samym katalogu. To audyt istniejących metryk, bez treningu, bez nowej
inferencji i bez zmiany wag.

Wejście:
  --metrics PATH     plik metrics.jsonl z ukończonego biegu E001 (200 ciągłych
                     epok, train_count 39936, dev_count 5000, spójne dev_top1).
  --output-dir PATH  nowy katalog wyjściowy; musi jeszcze nie istnieć.

Wyjście (w --output-dir):
  loss.png, loss.pdf, accuracy.png, accuracy.pdf, curve_summary.json, KRZYWE.md.
Wykresy obejmują wszystkie 200 epok bez wygładzania; zaznaczona jest najlepsza
epoka dev top-1 (pierwsze maksimum). Uruchomienie i kontekst: OPIS.md.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

EXPERIMENT_ID = "2026-09-26_E001_efficientnet_b0_od_zera"
EPOCHS_TOTAL = 200
TRAIN_COUNT_EXPECTED = 39936
DEV_COUNT_EXPECTED = 5000
DEV_TOP1_TOLERANCE = 1e-12
REQUIRED_FIELDS = (
    "epoch",
    "train_nll",
    "train_correct",
    "train_count",
    "dev_nll",
    "dev_correct",
    "dev_count",
    "dev_top1",
)
UWAGA_POMIAROWA = (
    "Uwaga: train mierzony online na obrazach augmentowanych w trybie train() "
    "z AMP; dev to ocena zamrożonego po epoce modelu w eval() w FP32. "
    "Warunki pomiaru nie są identyczne."
)
UWAGA_AUDYTU = (
    "Analiza istniejących metryk E001 na CPU; brak nowej inferencji i brak "
    "zmiany wag. Dev służył do wyboru checkpointu, nie jest niezależnym testem."
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics(path):
    records = []
    with open(path, encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                raise ValueError(f"pusta linia {lineno} w {path}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"niepoprawny JSON w linii {lineno}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"linia {lineno} nie jest obiektem JSON")
            records.append(record)
    return records


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_metrics(records):
    problems = []
    if len(records) != EPOCHS_TOTAL:
        problems.append(f"oczekiwano {EPOCHS_TOTAL} epok, otrzymano {len(records)}")

    expected_epochs = list(range(1, EPOCHS_TOTAL + 1))
    epochs = [record.get("epoch") for record in records]
    if epochs != expected_epochs:
        first = next(
            (i for i, (a, b) in enumerate(zip(epochs, expected_epochs)) if a != b),
            None,
        )
        if first is None:
            detail = "nadmiar lub brak epok"
        else:
            detail = f"pierwsza rozbieżność w pozycji {first + 1}: {epochs[first]!r}"
        problems.append(f"epoki nie tworzą ciągu 1..{EPOCHS_TOTAL} ({detail})")

    for index, record in enumerate(records):
        label = f"epoka {record.get('epoch', index + 1)!r}"
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            problems.append(f"{label}: brak pól {missing}")
            continue
        for field in ("train_nll", "dev_nll", "dev_top1"):
            value = record[field]
            if not is_number(value) or not math.isfinite(value):
                problems.append(f"{label}: {field} nie jest skończoną liczbą ({value!r})")
        for field in ("epoch", "train_count", "train_correct", "dev_count", "dev_correct"):
            if not isinstance(record[field], int) or isinstance(record[field], bool):
                problems.append(f"{label}: {field} nie jest liczbą całkowitą")
        if any(not isinstance(record[field], int) or isinstance(record[field], bool)
               for field in ("train_count", "train_correct", "dev_count", "dev_correct")):
            continue
        if record["train_count"] != TRAIN_COUNT_EXPECTED:
            problems.append(
                f"{label}: train_count={record['train_count']!r}, oczekiwano {TRAIN_COUNT_EXPECTED}"
            )
        if record["dev_count"] != DEV_COUNT_EXPECTED:
            problems.append(
                f"{label}: dev_count={record['dev_count']!r}, oczekiwano {DEV_COUNT_EXPECTED}"
            )
        if is_number(record["train_correct"]) and not (
            0 <= record["train_correct"] <= record["train_count"]
        ):
            problems.append(f"{label}: train_correct poza zakresem")
        if is_number(record["dev_correct"]) and not (
            0 <= record["dev_correct"] <= record["dev_count"]
        ):
            problems.append(f"{label}: dev_correct poza zakresem")
        if is_number(record["dev_top1"]) and is_number(record["dev_correct"]) and record["dev_count"]:
            expected_top1 = record["dev_correct"] / record["dev_count"]
            if abs(record["dev_top1"] - expected_top1) > DEV_TOP1_TOLERANCE:
                problems.append(
                    f"{label}: dev_top1={record['dev_top1']!r} niezgodne z "
                    f"dev_correct/dev_count={expected_top1!r}"
                )

    if problems:
        joined = "\n- ".join(problems)
        raise SystemExit(f"Walidacja metryk nie powiodła się:\n- {joined}")


def first_extreme(records, field, better):
    best = None
    for record in records:
        value = record[field]
        if best is None or better(value, best[0]):
            best = (value, record["epoch"])
    return best


def epoch_entry(record):
    return {
        "epoka": record["epoch"],
        "train_nll": record["train_nll"],
        "train_top1": record["train_correct"] / record["train_count"],
        "dev_nll": record["dev_nll"],
        "dev_top1": record["dev_top1"],
        "dev_correct": record["dev_correct"],
    }


def series(records):
    epochs = [record["epoch"] for record in records]
    train_nll = [record["train_nll"] for record in records]
    dev_nll = [record["dev_nll"] for record in records]
    train_acc = [record["train_correct"] / record["train_count"] for record in records]
    dev_acc = [record["dev_correct"] / record["dev_count"] for record in records]
    return epochs, train_nll, dev_nll, train_acc, dev_acc


def save_figure(fig, output_dir, stem):
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_dir / f"{stem}.png", dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_loss(records, output_dir, best_epoch):
    epochs, train_nll, dev_nll, _, _ = series(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_nll, color="#1f77b4", linewidth=1.5,
            label="Train NLL (online, augmentacja, AMP)")
    ax.plot(epochs, dev_nll, color="#d62728", linewidth=1.5,
            label="Dev NLL (eval, FP32)")
    ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2,
               label=f"Najlepsza epoka dev top-1 (ep. {best_epoch})")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("NLL (strata)")
    ax.set_title("E001: NLL train i dev, 200 epok, bez wygładzania")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.text(0.01, 0.01, "Train: online, augmentacja, train(), AMP. Dev: model po epoce, eval(), FP32.\nWarunki pomiaru nie są identyczne.", fontsize=8, color="#444444")
    save_figure(fig, output_dir, "loss")


def plot_accuracy(records, output_dir, best_epoch, best_value):
    epochs, _, _, train_acc, dev_acc = series(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, color="#2ca02c", linewidth=1.5,
            label="Train top-1 = train_correct/train_count (online, augmentacja, AMP)")
    ax.plot(epochs, dev_acc, color="#ff7f0e", linewidth=1.5,
            label="Dev top-1 = dev_correct/dev_count (eval, FP32)")
    ax.plot([best_epoch], [best_value], marker="o", markersize=7,
            color="#d62728",
            label=f"Najlepsza epoka dev top-1 (ep. {best_epoch})")
    ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Trafność top-1")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("E001: trafność top-1 train i dev, 200 epok, bez wygładzania")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.text(0.01, 0.01, "Train: online, augmentacja, train(), AMP. Dev: model po epoce, eval(), FP32.\nWarunki pomiaru nie są identyczne.", fontsize=8, color="#444444")
    save_figure(fig, output_dir, "accuracy")


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


def build_markdown(summary):
    first = summary["pierwsza_epoka"]
    last = summary["ostatnia_epoka"]
    min_nll = summary["min_dev_nll"]
    best = summary["najlepszy_dev_top1"]
    lines = [
        f"# Krzywe uczenia {EXPERIMENT_ID}",
        "",
        "Dokument wygenerowany automatycznie przez `plot_metrics.py` na podstawie "
        "istniejącego pliku metryk E001. To opis przebiegu uczenia, bez nowej "
        "inferencji, bez zmiany wag i bez ponownego treningu.",
        "",
        "## Źródło danych",
        "",
        f"- metryki: `{summary['metryki_sciezka']}`",
        f"- SHA-256 metryk: `{summary['metryki_sha256']}`",
        f"- skrypt: `{summary['skrypt_sciezka']}`",
        f"- SHA-256 skryptu: `{summary['skrypt_sha256']}`",
        f"- epoki: {summary['epoki_lacznie']} (ciągłe 1..{EPOCHS_TOTAL}), "
        f"train_count={summary['train_count']}, dev_count={summary['dev_count']}",
        f"- wygenerowano (UTC): {summary['utworzono_utc']}",
        "",
        "## Najważniejsze liczby",
        "",
        f"- Pierwsza epoka: train NLL {fmt(first['train_nll'])}, "
        f"train top-1 {first['train_top1'] * 100:.2f}%, "
        f"dev NLL {fmt(first['dev_nll'])}, dev top-1 {first['dev_top1'] * 100:.2f}%.",
        f"- Ostatnia epoka: train NLL {fmt(last['train_nll'])}, "
        f"train top-1 {last['train_top1'] * 100:.2f}%, "
        f"dev NLL {fmt(last['dev_nll'])}, dev top-1 {last['dev_top1'] * 100:.2f}%.",
        f"- Minimum dev NLL: {fmt(min_nll['wartosc'])}, epoka {min_nll['epoka']}.",
        f"- Najlepszy dev top-1: {best['wartosc'] * 100:.2f}% "
        f"({best['dev_correct']}/{summary['dev_count']}), epoka {best['epoka']} "
        f"(pierwsze maksimum; przy remisie zachowywane jest wcześniejsze).",
        "",
        "## Ostrzeżna interpretacja",
        "",
        "- Train jest mierzony online, na obrazach augmentowanych, w trybie `train()` "
        "z AMP, a dev to ocena zamrożonego po epoce modelu w `eval()` w FP32. "
        "Warunki te nie są identyczne, dlatego nie należy ich porównywać wprost.",
        "- Utrzymująca się różnica między train top-1 a dev top-1 oraz rosnąca "
        "dev NLL przy jednoczesnym wzroście dev top-1 mogą wskazywać na przeuczenie; "
        "ten opis nie ustala jego przyczyny.",
        "- Najlepszy dev top-1 i minimum dev NLL mogą przypadać na różne epoki: "
        f"top-1 w epoce {best['epoka']}, a minimum NLL w epoce {min_nll['epoka']}. "
         "NLL ocenia prawdopodobieństwo przypisane właściwej klasie i silnie karze "
         "pewne błędne odpowiedzi; nie jest samodzielnym pomiarem kalibracji. Top-1 mierzy trafność.",
        "- Dev był używany do wyboru checkpointu, więc nie jest niezależnym testem "
        "generalizacji. Nie oceniano tutaj rezerwy ani oficjalnego testu.",
        "- To audyt jednego zapisu metryk E001, a nie dowód korzyści atencji lub "
        "kompresji. Analiza nie wykonuje nowej inferencji i nie zmienia wag.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            f"{EXPERIMENT_ID}: rysuje krzywe E001 (PNG i PDF) oraz zapisuje "
            "curve_summary.json i KRZYWE.md. Tylko CPU, bez nowej inferencji."
        )
    )
    parser.add_argument("--metrics", required=True,
                        help="ścieżka do metrics.jsonl z ukończonego biegu E001")
    parser.add_argument("--output-dir", required=True,
                        help="nowy katalog wyjściowy (nie może istnieć)")
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        parser.error(f"plik metryk nie istnieje: {metrics_path}")
    script_path = Path(__file__).resolve()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"katalog wyjściowy już istnieje: {output_dir}")

    records = load_metrics(metrics_path)
    validate_metrics(records)

    output_dir.mkdir(parents=True, exist_ok=False)

    best_value, best_epoch = first_extreme(records, "dev_top1", lambda a, b: a > b)
    min_nll_value, min_nll_epoch = first_extreme(records, "dev_nll", lambda a, b: a < b)

    plot_loss(records, output_dir, best_epoch)
    plot_accuracy(records, output_dir, best_epoch, best_value)

    first = epoch_entry(records[0])
    last = epoch_entry(records[-1])
    generated_files = [
        "loss.png", "loss.pdf", "accuracy.png", "accuracy.pdf",
        "curve_summary.json", "KRZYWE.md",
    ]
    summary = {
        "experiment": EXPERIMENT_ID,
        "protokol": "AUDYT_OPIS_v1.md",
        "utworzono_utc": datetime.now(timezone.utc).isoformat(),
        "metryki_sciezka": str(metrics_path.resolve()),
        "metryki_sha256": sha256_file(metrics_path),
        "skrypt_sciezka": str(script_path),
        "skrypt_sha256": sha256_file(script_path),
        "epoki_lacznie": len(records),
        "train_count": TRAIN_COUNT_EXPECTED,
        "dev_count": DEV_COUNT_EXPECTED,
        "pierwsza_epoka": first,
        "ostatnia_epoka": last,
        "min_dev_nll": {"wartosc": min_nll_value, "epoka": min_nll_epoch},
        "najlepszy_dev_top1": {
            "wartosc": best_value,
            "epoka": best_epoch,
            "dev_correct": records[best_epoch - 1]["dev_correct"],
            "reguła_remisu": "pierwsze maksimum",
        },
        "pliki_wyjściowe": generated_files,
        "uwaga": UWAGA_AUDYTU,
        "pomiar": UWAGA_POMIAROWA,
    }

    summary_path = output_dir / "curve_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "KRZYWE.md").write_text(build_markdown(summary), encoding="utf-8")

    print(
        f"OK: {EXPERIMENT_ID}; epok={len(records)}, "
        f"najlepszy dev top1={best_value:.4f} (ep. {best_epoch}), "
        f"min dev NLL={min_nll_value:.6f} (ep. {min_nll_epoch}); "
        f"zapisano w {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
