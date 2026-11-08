#!/usr/bin/env python3
"""E036: przelicza TABLES.md wprost z surowych plików runu (CPU only).

Eksperyment 2026-11-07_E036_audyt_kosztu; pełny protokół w OPIS.md.
Wejście: runs/E036/20261107T231027Z_27bac186/{seed_*/*/costs.json,
*_bs{1,128}_s{0,1,2}.json} oraz frozen.json (jakość quality_CMA, reuse E035).
Wyjście: TABLES.md w katalogu eksperymentu. Runy pozostają niezmienione.
Batch1: SD z czterech sparowanych per-seed Tmedian/Umedian; przepustowość
b128: SD z czterech per-seed ilorazów (ddof=1). Nie używa starych SD.
"""
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RUN = ROOT / "runs/E036/20261107T231027Z_27bac186"
OUT = EXP / "TABLES.md"

SEEDS = (42, 43, 44, 45)
PAIRS = (
    "resnet_cifar10",
    "resnet_cifar100",
    "resnet_tiny",
    "preact_cifar10",
    "preact_cifar100",
)
TAGS = ("b30", "b50", "b70")
NAMES = {"U": "_U", "T": "_T"}
LABELS = {
    "resnet_cifar10": ("R", "C10"),
    "resnet_cifar100": ("R", "C100"),
    "resnet_tiny": ("R", "Tiny"),
    "preact_cifar10": ("P", "C10"),
    "preact_cifar100": ("P", "C100"),
}
REDUCTION = {"b30": "30%", "b50": "50%", "b70": "70%"}


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def n(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def signed(value, digits):
    return f"{value:+.{digits}f}".replace(".", ",")


def times(seed, pair, tag, profile, batch):
    values = []
    for series in range(3):
        data = read(RUN / f"seed_{seed}/{pair}/{tag}{NAMES[profile]}_bs{batch}_s{series}.json")
        if len(data["times_ms"]) != 200:
            raise RuntimeError("Oczekiwano 200 pomiarów w serii")
        values.extend(data["times_ms"])
    if len(values) != 600 or not all(x > 0 and math.isfinite(x) for x in values):
        raise RuntimeError("Błędne czasy w plikach pomiarowych")
    return values


def peak_mib(seed, pair, tag, profile, batch):
    peaks = []
    for series in range(3):
        data = read(RUN / f"seed_{seed}/{pair}/{tag}{NAMES[profile]}_bs{batch}_s{series}.json")
        peaks.append(data["peak_allocated"])
    return st.median(peaks) / 2**20


def main():
    timing_paths = []
    cost_paths = []
    for seed in SEEDS:
        for pair in PAIRS:
            cost_paths.append(RUN / f"seed_{seed}/{pair}/costs.json")
            for tag in TAGS:
                for profile in ("T", "U"):
                    for batch in (1, 128):
                        for series in range(3):
                            timing_paths.append(
                                RUN / f"seed_{seed}/{pair}/{tag}{NAMES[profile]}_bs{batch}_s{series}.json"
                            )
    frozen_path = RUN / "frozen.json"
    sources = [frozen_path, *cost_paths, *timing_paths]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path.resolve()): sha256(path) for path in sources}

    frozen = read(frozen_path)
    cells = {}
    quality = {}
    params_abs = {}
    mac_abs = {}

    for pair in PAIRS:
        for tag in TAGS:
            per_seed = []
            for seed in SEEDS:
                costs = read(RUN / f"seed_{seed}/{pair}/costs.json")
                cu = costs[tag + NAMES["U"]]
                ct = costs[tag + NAMES["T"]]
                t1 = times(seed, pair, tag, "T", 1)
                u1 = times(seed, pair, tag, "U", 1)
                t128 = times(seed, pair, tag, "T", 128)
                u128 = times(seed, pair, tag, "U", 128)
                ref = frozen["references"][str(seed)][pair][tag]
                per_seed.append(
                    {
                        "seed": seed,
                        "params_u": cu["params"],
                        "params_t": ct["params"],
                        "params_ratio": ct["params"] / cu["params"],
                        "mac_u": cu["mac"],
                        "mac_t": ct["mac"],
                        "mac_ratio": ct["mac"] / cu["mac"],
                        "lat_u": st.median(u1),
                        "lat_t": st.median(t1),
                        "lat_ratio": st.median(t1) / st.median(u1),
                        "thr_u": 128000.0 / st.mean(u128),
                        "thr_t": 128000.0 / st.mean(t128),
                        "thr_ratio": (128000.0 / st.mean(t128)) / (128000.0 / st.mean(u128)),
                        "peak_u": peak_mib(seed, pair, tag, "U", 128),
                        "peak_t": peak_mib(seed, pair, tag, "T", 128),
                        "quality_diff_pp": 100.0
                        * (ref["T"]["quality_CMA"]["top1"] - ref["U"]["quality_CMA"]["top1"]),
                    }
                )
            cells[(pair, tag)] = per_seed
            quality[(pair, tag)] = st.mean([r["quality_diff_pp"] for r in per_seed])

    def mean_of(pair, tag, key):
        return st.mean([r[key] for r in cells[(pair, tag)]])

    def sd_of(pair, tag, key):
        return st.stdev([r[key] for r in cells[(pair, tag)]])

    # Kontrola zgodności z analysis.json (tylko odczyt, nie źródło wartości).
    analysis_path = RUN / "analysis/analysis.json"
    max_ratio_diff = 0.0
    if analysis_path.is_file():
        analysis = read(analysis_path)
        ref_ratios = {
            (r["seed"], r["pair"], r["tag"], r["batch"]): r for r in analysis["ratios"]
        }
        for pair in PAIRS:
            for tag in TAGS:
                for row in cells[(pair, tag)]:
                    ref = ref_ratios[(row["seed"], pair, tag, 1)]
                    ref128 = ref_ratios[(row["seed"], pair, tag, 128)]
                    max_ratio_diff = max(
                        max_ratio_diff,
                        abs(row["lat_ratio"] - ref["latency_ratio"]),
                        abs(row["thr_ratio"] - ref128["throughput_ratio"]),
                        abs((row["mac_t"] / row["mac_u"]) - ref["mac_ratio"]),
                        abs((row["params_t"] / row["params_u"]) - ref["params_ratio"]),
                    )

    # Zakresy do opisowej noty.
    lat_means = [mean_of(pair, tag, "lat_ratio") for pair in PAIRS for tag in TAGS]
    lat_sds = [sd_of(pair, tag, "lat_ratio") for pair in PAIRS for tag in TAGS]
    thr_means = [mean_of(pair, tag, "thr_ratio") for pair in PAIRS for tag in TAGS]
    param_ratios = [
        st.mean([r["params_ratio"] for r in cells[(pair, tag)]])
        for pair in PAIRS
        for tag in TAGS
    ]

    lines = []
    lines.append("# E036 — tabele kosztu")
    lines.append("")
    lines.append(
        "Run `runs/E036/20261107T231027Z_27bac186`. Wszystkie liczby poniżej są "
        "przeliczone wprost z surowych `costs.json` i plików czasów przez "
        "`tables_from_raw.py` (tylko CPU). Wartości to średnie czterech seedów; "
        "każdy model/batch ma 600 zdarzeń (3 serie×200), mediana liczona per seed. "
        "Przepustowość =128000/średnia z600 zdarzeń; peak pamięci = mediana "
        "3 serii per seed, potem średnia4, `peak_allocated`/2^20. Jakość T−U "
        "pochodzi wyłącznie z zamrożonych referencji E036 `quality_CMA` "
        "(reuse E035), nie z nowej oceny. MAC: 1 multiply-add=1; bias/BN/ReLU "
        "poza MAC."
    )
    lines.append("")
    lines.append(
        "**Korekta:** wcześniejsze SD batch1 oraz część wartości parametrów "
        "wpisano ręcznie i nie odpowiadały temu runowi. "
        "Ta wersja liczy parametry, MAC, czasy, pamięć i SD wprost z surowych "
        "plików; starych SD nie używamy. SD to odchylenie z czterech sparowanych "
        "per-seed ilorazów (ddof=1), nie CI. Wykresy i opis: "
        "[figures/README.md](figures/README.md)."
    )
    lines.append("")
    lines.append("## Koszt bezwzględny")
    lines.append("")
    lines.append(
        "| Para | Red. | Param. U/T [M] | MAC U/T [M] | Lat. b1 U/T [ms] | "
        "Przepust. b128 U/T [obr./s] | Peak b128 U/T [MB] | Jakość T−U [pp] |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for pair in PAIRS:
        for tag in TAGS:
            p, d = LABELS[pair]
            lines.append(
                f"| {p}/{d} | {REDUCTION[tag]} | "
                f"{n(mean_of(pair, tag, 'params_u') / 1e6, 3)}/{n(mean_of(pair, tag, 'params_t') / 1e6, 3)} | "
                f"{n(mean_of(pair, tag, 'mac_u') / 1e6, 1)}/{n(mean_of(pair, tag, 'mac_t') / 1e6, 1)} | "
                f"{n(mean_of(pair, tag, 'lat_u'), 2)}/{n(mean_of(pair, tag, 'lat_t'), 2)} | "
                f"{n(mean_of(pair, tag, 'thr_u'), 0)}/{n(mean_of(pair, tag, 'thr_t'), 0)} | "
                f"{n(mean_of(pair, tag, 'peak_u'), 1)}/{n(mean_of(pair, tag, 'peak_t'), 1)} | "
                f"{signed(quality[(pair, tag)], 2)} |"
            )
    lines.append("")
    lines.append("## Ilorazy T/U")
    lines.append("")
    lines.append(
        "| Para | Red. | Param. | MAC | Przepust. b128 ±SD | Latencja b1 ±SD |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for pair in PAIRS:
        for tag in TAGS:
            p, d = LABELS[pair]
            lines.append(
                f"| {p}/{d} | {REDUCTION[tag]} | "
                f"{n(mean_of(pair, tag, 'params_ratio'), 4)} | "
                f"{n(mean_of(pair, tag, 'mac_ratio'), 3)} | "
                f"{n(mean_of(pair, tag, 'thr_ratio'), 3)} ±{n(sd_of(pair, tag, 'thr_ratio'), 3)} | "
                f"{n(mean_of(pair, tag, 'lat_ratio'), 3)} ±{n(sd_of(pair, tag, 'lat_ratio'), 3)} |"
            )
    lines.append("")
    lines.append(
        "Uwaga: średnie "
        f"ilorazy T/U bliskie1 ({n(min(lat_means), 3)}–{n(max(lat_means), 3)}) "
        "nie dowodzą równoważności metod ani dominacji narzutu uruchomienia; "
        f"SD z czterech seedów wynosi {n(min(lat_sds), 3)}–"
        f"{n(max(lat_sds), 3)}. Przepustowość batch128 pokazuje rzeczywisty "
        f"koszt: TILT jest wolniejszy o5–30% (iloraz {n(min(thr_means), 3)}–"
        f"{n(max(thr_means), 3)}). Parametry U i T nie są identyczne (iloraz "
        f"{n(min(param_ratios), 4)}–{n(max(param_ratios), 4)}), dlatego podano "
        "osobne wartości, a nie wspólną liczbę."
    )
    lines.append("")
    lines.append(
        "Kontrola: per-seed ilorazy zgodne z `analysis.json` (max |Δ| = "
        f"{max_ratio_diff:.1e}); pliki wejściowe niezmienione przed i po "
        "generowaniu."
    )
    lines.append("")

    after = {str(path.resolve()): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("Plik wejściowy zmienił się podczas generowania tabel")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("ZAPISANO", OUT)
    print("komorki", len(PAIRS) * len(TAGS), "pliki czasow", len(timing_paths))
    print("max |delta| vs analysis.json", max_ratio_diff)
    print("batch1 mean", n(min(lat_means), 3), "-", n(max(lat_means), 3),
          "SD", n(min(lat_sds), 3), "-", n(max(lat_sds), 3))
    print("przepustowosc b128 mean", n(min(thr_means), 3), "-", n(max(thr_means), 3))
    print("params ratio", n(min(param_ratios), 4), "-", n(max(param_ratios), 4))


if __name__ == "__main__":
    main()
