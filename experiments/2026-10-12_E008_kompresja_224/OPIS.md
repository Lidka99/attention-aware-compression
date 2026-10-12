# 2026-10-12_E008_kompresja_224

Wersja protokołu: **0.1-draft**. Status: przygotowanie; pełny bieg
nieuruchomiony (po kontrolach CPU i zamrożeniu).

## Pytanie i zakres

Na mocniejszym baseline E007 (EfficientNet-B0 z SE, Tiny ImageNet 224×224,
dev 58,49%) sprawdzamy, czy wybór kanałów do scalenia ma znaczenie
(kryterium PRE/POST vs kontrola RANDOM, wspólny operator, α/β z A):

- **Etap 1 (bez naprawy):** jaką szkodę w dev top-1/NLL powoduje fizyczna
  redukcja **10%** i **30%** parametrów dla PRE, POST i RANDOM;
- **Etap 2 (naprawa, 30%):** czy po **równym dostrajaniu** (BASE, PRE,
  POST, RANDOM — identyczna receptura i budżet) kryterium różnicuje
  końcową jakość.

Kontekst E003/E004: na CIFAR-100 przy 10% różnice surowe znikały po równym
dostrajaniu, a RANDOM dorównał. Tu: mocniejszy baseline, dwa budżety i
decydujący etap naprawy przy 30% (głębsza szkoda = większa szansa, że
selekcja zacznie mieć znaczenie). Wynik zerowy/gorszy jest pełnoprawnym
wynikiem.

## Hipoteza i kryterium oceny (ustalone przed uruchomieniem)

POST (podobieństwo po SE, „attention-aware") zachowa jakość lepiej niż PRE
(przed SE) i RANDOM po równym dostrajaniu przy 30%. Kryterium
pierwszorzędne: **POST−PRE w etapie 2**, sparowany bootstrap po obrazach
stratyfikowany po klasach (10 000 powtórzeń, seed 20261016); sukces
zjawiska = dolna granica 95% CI > 0. Porównania z RANDOM raportowane jako
warunek sensowności (sama przewaga nad PRE bez przewagi nad RANDOM nie
wystarcza). Nie ustalamy progów po wynikach.

## Model i budżety (plany strukturalne, CPU)

- Baseline: `runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80/last.pt`
  (epoka 100, SHA-256 `718bbb26…3744408`, 4 263 748 parametrów, 200 klas).
  Bramka sanity w etapie 1: dev top-1 = 5849/10000,
  |NLL − 2,0001186| ≤ 1e-5 (odtworzenie wyniku E007).
- Plany tylko z architektury (`plan_structure_e008.py`, ten sam algorytm
  co E003: koszt kanału, kwantyzacja 8, limit połowy szerokości bloku,
  remisy po ścieżce):
  - **10%:** 4 263 748 → 3 837 364 (10,0002%; 1048 kanałów),
    `PLAN_STRUKTURY_10.json` SHA-256 `1c23aad9…e15c9bd7`;
  - **30%:** 4 263 748 → 2 981 828 (30,0656%; 3168 kanałów),
    `PLAN_STRUKTURY_30.json` SHA-256 `fa08dbcd…519a2f340`.
- Folding: lokalny fork `folding_e008.py` (semantyka E003 bez zmian):
  α do kolumny fc1, β do kolumny projekcji, fizyczne usunięcie producenta
  pary d; residuale, stem i klasyfikator nietknięte.

## Dane i kalibracja

- Tiny ImageNet, splits seed 20261010 (jak E007): train 90 000, dev
  10 000 (50/klasę); oficjalny val = rezerwa i test — **nietknięte**.
- Kalibracja A = 1000 obrazów train (5/klasę), B = 1000 (5/klasę),
  rozłączne, seed 20261015 (schemat E006). A: momenty i pary/α/β;
  B: diagnostyka błędów bloków na wejściach nauczyciela.
- Preprocessing deterministyczny: bicubic resize 224 + Normalize(ImageNet),
  bez augmentacji (A/B i dev).

## Wybór par i operator (`selection.py` z E003, hash przypiety)

- PRE: cosinusy h; POST: cosinusy z (po SE); RANDOM: permutacja
  kwalifikowanych kanałów, seed 20261015 + 1000·idx bloku. Pary rozłączne
  w bloku, reprezentant = mniejszy indeks; kwalifikacja: wszystkie normy
  h/z/g > 1e-12; za mało kandydatów → błąd bez zmiany budżetu.
- α = ⟨g_c,g_d⟩/⟨g_c,g_c⟩, β = ⟨z_c,z_d⟩/⟨z_c,z_c⟩ na A (FP64, bez
  clippingu). Momenty: sumy FP64 (akumulacja na GPU w FP64, agregacja
  ranków po CPU w stałej kolejności).

## Etap 1 — pomiar szkody (bez naprawy, 4 GPU)

Jeden przebieg rozproszony: rank r kalibruje A[r::4] (batch 32), rank0
agreguje momenty i wybiera grupy dla 10% i 30% (zapis
`candidate_manifest_{10,30}.json` **przed** oceną), wszystkie rangi
oceniają dev[r::4] (batch 64, FP32, deterministycznie) dla BASE + 6 modeli
oraz wykonują diagnostykę B. Zapis: momenty, `models/*.pt`
(inference-only, bez optymalizatora), predykcje dev per model,
`stage1_metrics.json`, `block_errors.json`. Bramka BASE musi przejść,
inaczej bieg failed.

## Etap 2 — równe dostrajanie przy 30% (4 GPU)

- Warianty sekwencyjnie na tych samych 4 GPU: BASE, PRE, POST, RANDOM.
- Identyczna receptura (kontynuacja receptury E007): AdamW lr 3e-4,
  wd 0,01, warmup 3 epoki, cosine do min_lr 3e-6, LS 0,1, clip 1,0;
  augmentacje jak E007 (RRC 0,80–1,00, ratio 0,90–1,10, bicubic;
  ColorJitter 0,15/0,15/0,15/0,05; hflip); batch 32/GPU (globalny 128);
  **60 epok**; bez early stoppingu i bez wyboru epoki po fakcie;
  dev co epokę; wynik = ostatnia epoka. Warianty startują dokładnie
  z modeli zmierzonych w etapie 1 (te same pliki i hashe).
- Zapis: `metrics_{VARIANT}.jsonl`, predykcje dev per wariant,
  `last_{VARIANT}.pt`, `stage2_metrics.json`.

## Cztery GPU

Etap 1: jeden pomiar podzielony na 4 rangi (nie 4 niezależne powtórzenia).
Etap 2: cztery warianty sekwencyjnie, każdy DDP na 4 GPU z identycznym
budżetem (nie 4 seedy).

## Ograniczenia z góry

Jeden baseline i jeden seed naprawy; wszystkie bloki zmieniane naraz ze
statystyk α/β nauczyciela (błąd może się kumulować — to samo założenie co
E003); dev wybierał epokę w E007 (nie jest niezależnym testem); ocena
względna między wariantami w tym samym protokole; rezerwa i test
nietknięte.

## Pliki

- [plan_structure_e008.py](plan_structure_e008.py) + `PLAN_STRUKTURY_{10,30}.json`
- [folding_e008.py](folding_e008.py) (fork E003), [support_e008.py](support_e008.py)
- [study_224.py](study_224.py) (etap 1), [finetune_224.py](finetune_224.py) (etap 2)
- [launch_e008.py](launch_e008.py), [summarize_e008.py](summarize_e008.py),
  [check_cpu_e008.py](check_cpu_e008.py)
- Docelowo: `config.json` (frozen), `PROTOKOL_v1.md`, raport `WYNIKI.md`.

## Kolejne kroki

1. Kontrole CPU: plany/folding/selekcja syntetyczna/ładowanie baseline/
   dane A/B/dev → `TESTY_CPU_v1.json`.
2. Zamrożenie `config.json` (protocol 1.0) + `PROTOKOL_v1.md` + commit.
3. Bieg: etap 1 (~10–20 min) i etap 2 (~11–12 h), jeden katalog runs/E008.
4. Raport `WYNIKI.md`: wyniki etapu 1, wynik główny etapu 2 (POST−PRE),
   decyzja o dalszym kierunku.
