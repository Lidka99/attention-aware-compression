# 2026-10-12_E009_naprawa_120

Wersja protokołu: **1.0 (frozen)**. Status: **completed** (2026-10-14 UTC);
wyniki: [WYNIKI.md](WYNIKI.md). Bieg uruchamiany automatycznie po zakończeniu
biegu E008 (wejście).

## Pytanie i cel

Czy przy **dłuższym, równym budżecie naprawy (120 epok)** warianty wyboru kanałów
(BASE, PRE, POST, RANDOM) różnią się końcową jakością na baseline 224? E008 odpowiada
na to dla 60 epok; E009 odpowiada dla **120 epok jednym ciągłym harmonogramem cosine**
(analogia do E004, gdzie naprawa trwała 120 epok).

**Dlaczego osobny eksperyment:** protokół E008 jest zamrożony (60 epok) i nie zmieniamy
go w trakcie; dłuższy budżet to nowa próba badawcza z własnym protokołem i biegiem.
E009 startuje z **tych samych modeli etapu 1 E008** — różni się wyłącznie długością
naprawy, co izoluje efekt budżetu.

## Wejścia (zamrożone hashem)

- Modele etapu 1 E008:
  `runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim/stage1/models/`
  {BASE, 30_PRE, 30_POST, 30_RANDOM}.pt — fizyczne modele po redukcji 30%
  (2 981 828 parametrów; BASE = baseline E007, 4 263 748).
- Baseline E007 (SHA-256 `718bbb26…3744408`); dane Tiny ImageNet (splits/cache —
  hashe jak w E008); `candidate_manifest_30.json` i `stage1_metrics.json` E008.
- Kod zależny: `support_e008.py` i `folding_e008.py` (pinowane hashem w configu).

## Receptura (przedłużenie receptury E008)

- AdamW lr 3e-4, wd 0,01, warmup 3 epoki, cosine do min_lr 3e-6, LS 0,1, clip 1,0.
- Augmentacje jak w treningu bazowym: RRC(224, 0,80–1,00, ratio 0,90–1,10, bicubic),
  ColorJitter(0,15/0,15/0,15/0,05), hflip; normalizacja ImageNet.
- Batch 32/GPU (globalny 128), AMP FP16, SyncBN, deterministycznie.
- **120 epok**; wynik = ostatnia epoka; dev co epokę; bez early stoppingu i wyboru
  epoki po fakcie; zapasowy `partial_{VARIANT}.pt` co 10 epok.
- Warianty sekwencyjnie na 4 GPU (DDP), identyczny budżet: BASE, PRE, POST, RANDOM.

## Kryterium oceny

Porównania sparowane po obrazach (bootstrap 10 000, seed 20261017, stratyfikacja po
klasach): **POST−PRE (pierwszorzędne, jak w E008)**, pozostałe pary oraz każdy wariant
minus BASE. Sukces zjawiska: dolna granica 95% CI dla POST−PRE > 0. Bez progów
dobieranych po wyniku.

## Przebieg i czas

Jeden bieg: `runs/E009/<UTC>_<git>_seed42_<id>/`; launcher sprawdza, że bieg E008
ma status `completed` (E009 bez niego nie startuje). Szacunek: 4 × 120 × ~165 s
≈ **22 h** na 4 GPU. Uruchomienie automatyczne po zakończeniu E008 (nadzorca
operacyjny poza repozytorium, czeka też na czyste drzewo Git).

## Ograniczenia

Jeden seed; dłuższy budżet dotyczy wyłącznie etapu naprawy (selekcja i kompresja
identyczne jak w E008); porównanie obowiązuje dla 120 epok i tego baseline;
rezerwa i test nietknięte.

## Pliki

- [repair_120.py](repair_120.py) — trening naprawy; [launch_e009.py](launch_e009.py) — launcher;
  [summarize_e009.py](summarize_e009.py) — podsumowanie z bootstrapem;
  [support_e009.py](support_e009.py) — wejścia/kontrole; [check_cpu_e009.py](check_cpu_e009.py).
- Docelowo: `config.json` (frozen), `PROTOKOL_v1.md`, raport `WYNIKI.md`.
