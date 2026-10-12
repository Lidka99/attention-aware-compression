# E007 — wynik treningu EfficientNet-B0 od zera na Tiny ImageNet 224×224

Data: 2026-10-12 UTC. **Status wykonania: completed (exit 0).**
Protokół przed uruchomieniem: [PROTOKOL_v1.md](PROTOKOL_v1.md), v1.0.
Bieg: **`runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80/`**,
commit **`4de51b3`**, dirty=false.

## Wynik końcowy

| Metryka | Wartość |
|---|---:|
| **Dev top-1 (epoka 100, wynik główny)** | **58,49%** (5849/10000) |
| Dev top-1 najlepszy (diagnostyka) | 58,58% (epoka 98) |
| Dev NLL (epoka 100) | 2,0001 |
| Train top-1 (epoka 100) | 99,50% (89531/89984) |
| Train NLL (epoka 100) | 0,1793 |
| Przebieg dev top-1 | 3,1% (ep. 1) → 46,8% (ep. 10) → 56,4% (ep. 50) → 58,5% (ep. 100) |

Start 2026-10-11T06:18:25Z, koniec 10:55:32Z — **4 h 37 min** na 4 GPU
(~165 s/epokę). Batch 32/GPU (globalny 128), AMP FP16, SyncBN, DDP,
deterministycznie; `artifacts.sha256`: 19/19 plików OK.

## Porównanie

| Baseline | Rozdzielczość | Dev top-1 |
|---|---:|---:|
| E005 (receptura własna) | 96×96 | 51,53% |
| **E007 (kotwica + wd 0,01)** | **224×224** | **58,49%** |
| Shahriar 2026 (kotwica; oficjalny val) | 224×224 | 60,54% |

- **+6,96 pp względem E005.** Do kotwicy brakuje 2,05 pp, ale porównanie
  jest jakościowe: inny zbiór oceny (ich oficjalny val, nasz dev), inny
  batch (32 vs nasz 128), wd (1e-4 vs 0,01) i dodany warmup.
- Przeuczenie nadal obecne (train 99,5%), ale luka train/dev jest dużo
  mniejsza niż w E005; dzięki label smoothing train NLL nie jest wprost
  porównywalny z E005 (0,18 vs 0,05 — głównie efekt wygładzenia).
- Dev top-1 rósł do końca (ep. 98 ≈ ep. 100), dev NLL płaski ~2,0 od
  połowy biegu. Bez wyboru epoki po fakcie: wynikiem jest `last.pt`.

## Dane i wykonanie

- Tiny ImageNet; podział własny seed 20261010: train 90 000 (450/klasę),
  dev 10 000 (50/klasę); oficjalny val = rezerwa **nietknięta**; test bez
  etykiet.
- Receptura: RRC(224, 0,80–1,00, ratio 0,90–1,10, bicubic) + ColorJitter
  (0,15/0,15/0,15/0,05) + hflip; normalizacja ImageNet; AdamW 3e-4,
  wd 0,01, warmup 5 epok, cosine do 3e-6, LS 0,1, clip 1,0; AMP FP16.
- `seed_all(seed + rank)` przed inicjalizacją modelu — naprawa błędu
  kolejności seedowania z E005 (wykrytego w audycie).
- Kontrola techniczna: smoke `runs/E007/smoke_20261011T061254Z`
  (50/klasę, 2 epoki: 20,6 s/epokę, peak CUDA 1515 MiB/rangę).
- `partial_checkpoint.pt` zapisywany co 5 epok (rolling, artefakt
  bezpieczeństwa); wznowienia nie używano.

## Artefakty i hashe

- `artifacts.sha256`: **19/19 plików OK** (`sha256sum -c`).

| Artefakt | SHA-256 |
|---|---|
| last.pt | `718bbb2621c1b136094749c3a7e5527839066d8b9e7a25bfccbc388bb3744408` |
| partial_checkpoint.pt | `ac53cebe59a3bf0da9f7e27b3a02ae182059ed9cf7d2d87fc83b0958c01ed33f` |
| metrics.json | `bcec8ace33d4c5bfee584d16ae5177445567edac068b74f97b47075464aa1de3` |
| metrics.jsonl | `d6763e4f1ee889bc7e36c0eaa0595c89b875ddccbf0431d3205a5b8e3752c593` |

## Interpretacja

- Baseline spełnia cel: 58,49% dev, wyraźnie mocniejszy i zdrowszy niż
  E005; jakościowo zgodny z kotwicą (60,54%). Wystarczający jako punkt
  odniesienia do badań kompresji na 224 i do publikacji.
- Odczytu nie traktujemy jako odtworzenia kotwicy 1:1 (inne warunki
  oceny i optymalizacji — wymienione wyżej).
- Model nadal się przeucza; ewentualne wzmocnienie receptury
  (mixup/cutmix, dłuższy trening) pozostaje opcją na później, nie
  warunkiem obecnych badań.

## Ograniczenia

Jeden seed i jeden bieg; 224×224 to adaptacja z natywnych 64 (bez nowego
detalu); brak mixup/cutmix świadomie (budżet 100 epok); ocena wyłącznie
na dev; porównanie z literaturą jakościowe. Oficjalny val i test
nietknięte.

## Następny krok

Kandydat **E008**: badanie kompresji na nowym baseline (pytanie: czy przy
większej redukcji parametrów selekcja kanałów — PRE/POST vs RANDOM,
wspólny operator — zaczyna mieć znaczenie po równym dostrajaniu).
Do rozstrzygnięcia przez użytkowniczkę: zakres (10% i/lub 30%),
kolejność etapów (najpierw pomiar szkody bez naprawy, potem równe
dostrajanie) oraz ewentualne dodatkowe seedy E007.
