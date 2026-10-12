# E008 — wersja wykonania 1.0

Eksperyment: `2026-10-12_E008_kompresja_224`.
Data zamrożenia: 2026-10-12 UTC. **Status protokołu: frozen**.
Status wykonania podaje manifest biegu (etapy `stage1` i `stage2`).

## Cel i pytanie

Na baseline E007 (EfficientNet-B0 z SE, Tiny ImageNet 224×224, dev 58,49%)
zmierzyć szkodę kompresji 10% i 30% parametrów dla kryteriów PRE/POST/
RANDOM (etap 1, bez naprawy) oraz sprawdzić po równym dostrajaniu przy
30% (etap 2), czy kryterium wyboru różnicuje końcową jakość. Kryterium
pierwszorzędne: **POST−PRE po naprawie** (sparowany bootstrap po obrazach,
stratyfikowany po klasach; sukces = dolna granica 95% CI > 0). Kontrola
RANDOM decyduje o sensowności samego kryterium. Szczegóły i ograniczenia:
[OPIS.md](OPIS.md).

## Ustalenia zamrożone

- Baseline: `runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80/last.pt`,
  SHA-256 `718bbb2621c1b136094749c3a7e5527839066d8b9e7a25bfccbc388bb3744408`,
  epoka 100, 4 263 748 parametrów; bramka stage1: dev top-1 5849/10000,
  |NLL − 2,0001186| ≤ 1e-5.
- Plany struktury (tylko architektura): `PLAN_STRUKTURY_10.json`
  (3 837 364 param.; 1048 kanałów; SHA-256 `1c23aad9…e15c9bd7`) i
  `PLAN_STRUKTURY_30.json` (2 981 828 param.; 3168 kanałów;
  SHA-256 `fa08dbcd…519a2f340`).
- Operator: `folding_e008.py` — fork E003, semantyka bez zmian
  (α do kolumny fc1, β do kolumny projekcji, usunięcie producenta pary).
- Dane: Tiny ImageNet; splits/cache hashe jak w manifeście E007
  (splits `9df906f7…`); A = B = 1000 obrazów train (5/klasę), rozłączne,
  seed 20261015; dev 10 000 (50/klasę); rezerwa i test **nietknięte**.
- Selekcja: `selection.py` z E003 importowany read-only
  (SHA-256 `53d2c251…3196cc`); seed RANDOM 20261015 + 1000·idx bloku;
  próg norm 1e-12; pary rozłączne, reprezentant mniejszy indeks.
- Etap 1 (4 GPU, jeden przebieg): kalibracja A[r::4] batch 32, ocena dev
  batch 64 FP32, diagnostyka B batch 32; momenty FP64; grupy zapisane
  przed oceną; BASE musi odtworzyć E007 inaczej failed.
- Etap 2 (4 GPU, warianty sekwencyjnie): BASE, PRE, POST, RANDOM; AdamW
  lr 3e-4, wd 0,01, warmup 3, cosine do min_lr 3e-6, LS 0,1, clip 1,0;
  aug. jak E007 (RRC 0,80–1,00 ratio 0,90–1,10 bicubic, ColorJitter
  0,15/0,15/0,15/0,05, hflip); batch 32/GPU (globalny 128); **60 epok**;
  wynik = ostatnia epoka; bez early stoppingu i wyboru epoki po fakcie.
- Bootstrap: 10 000 powtórzeń, seed 20261016, stratyfikacja po klasach;
  `paired_bootstrap` importowany z E003 `summarize.py`
  (SHA-256 `8007f0aa…8e8719`).

## Kontrole przed biegiem (techniczne, nie naukowe)

- [TESTY_CPU_v1.json](TESTY_CPU_v1.json): **passed** — plany i wzory
  parametrów (3 837 364 / 2 981 828), folding na prawdziwym modelu
  i strict reload grafów, selekcja syntetyczna (1048 i 3168 par na
  metodę), dane A/B/dev i hashe cache/splits zgodne z E007.
- `py_compile` wszystkich modułów: OK.

## Komenda

```bash
conda run --no-capture-output -n attention python experiments/2026-10-12_E008_kompresja_224/launch_e008.py
```

Launcher: czysty Git, frozen config, dokładnie 4 GPU; nowy katalog
`runs/E008/<UTC>_<git>_seed42_<unikalny>/` z podkatalogami `stage1/`
i `stage2/`; po etapie 1 uruchamia etap 2, potem `summarize_e008.py`;
na końcu `artifacts.sha256`. Wznowienie: brak (przerwany bieg
powtarzamy od zera w nowym katalogu).

## Ograniczenia

Jeden baseline i jeden seed naprawy; szkoda pełnej sieci po jednorazowym
foldingu wszystkich bloków (statystyki nauczyciela, błąd może się
kumulować); dev wybierał epokę E007 (nie jest niezależnym testem);
porównanie względne między wariantami w tym samym protokole; brak
magnitude (świadomie wąski zakres); rezerwa i test nietknięte.
