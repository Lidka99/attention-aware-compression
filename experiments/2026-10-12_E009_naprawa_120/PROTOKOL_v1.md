# E009 — wersja wykonania 1.0

Eksperyment: `2026-10-12_E009_naprawa_120`.
Data zamrożenia: 2026-10-12 UTC. **Status protokołu: frozen**.
Status wykonania podaje manifest biegu.

## Cel i pytanie

Czy przy **120 epokach równej naprawy** (jeden ciągły cosine) warianty
BASE / PRE / POST / RANDOM różnią się końcową jakością? E008 odpowiada dla 60 epok;
E009 izoluje efekt **dłuższego budżetu** na tych samych modelach etapu 1 (30% redukcji).
Kryterium pierwszorzędne: **POST−PRE po 120 epokach** (bootstrap 10 000, seed 20261017,
stratyfikacja po klasach); sukces = dolna granica 95% CI > 0. Pozostałe pary i każdy
wariant minus BASE — raportowane.

## Ustalenia zamrożone

- Wejścia (pinowane SHA-256 w configu): modele etapu 1 E008
  (`stage1/models/` BASE, 30_PRE, 30_POST, 30_RANDOM),
  `candidate_manifest_30.json`, `stage1_metrics.json`, `model_files.json`
  z biegu `runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim`;
  baseline E007 `718bbb26…3744408`; dane Tiny ImageNet (splits `9df906f7…`,
  cache jak w manifeście E007); kod zależny: `support_e008.py` `c32689fe…`,
  `folding_e008.py` `2ab8c90b…`.
- Receptura: AdamW lr 3e-4, wd 0,01, warmup 3 epoki, cosine do min_lr 3e-6,
  LS 0,1, clip 1,0; RRC(224, 0,80–1,00, ratio 0,90–1,10, bicubic) +
  ColorJitter(0,15/0,15/0,15/0,05) + hflip; batch 32/GPU (globalny 128); AMP FP16;
  **120 epok**; wynik = ostatnia epoka; dev co epokę; bez early stoppingu.
- Warianty sekwencyjnie na 4 GPU (DDP) — nie są to 4 niezależne powtórzenia.
- Bieg: `runs/E009/<UTC>_<git>_seed42_<id>/`; launcher wymaga, aby bieg E008
  miał status `completed` (E009 bez tego wejścia nie startuje).

## Kontrole przed biegiem (techniczne, nie naukowe)

- [TESTY_CPU_v1.json](TESTY_CPU_v1.json): **passed** — hashe wejść zgodne z configiem,
  2 981 828 / 4 263 748 parametrów, digesty modeli zgodne z etapem 1 E008,
  forward kontrolny 30_PRE OK.
- `py_compile` wszystkich modułów: OK.

## Komenda

```bash
conda run --no-capture-output -n attention python experiments/2026-10-12_E009_naprawa_120/launch_e009.py
```

Uruchamiany automatycznie po zakończeniu E008 (nadzorca operacyjny poza repozytorium
czeka na status `completed` i czyste drzewo Git, potem startuje w odłączonej sesji).
Po treningu launcher uruchamia `summarize_e009.py` i zapisuje `artifacts.sha256`.

## Ograniczenia

Jeden seed; dłuższy budżet dotyczy wyłącznie etapu naprawy (selekcja i kompresja
identyczne jak w E008); porównanie obowiązuje dla 120 epok i tego baseline;
rezerwa i test nietknięte.
