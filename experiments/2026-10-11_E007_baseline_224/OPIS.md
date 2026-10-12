# 2026-10-11_E007_baseline_224

Wersja wykonania: **1.0** (zamrożona; config `protocol_version` 1.0;
szczegóły: [PROTOKOL_v1.md](PROTOKOL_v1.md)). Status: **pełny bieg
zakończony 2026-10-12 (completed, exit 0)** — [WYNIKI.md](WYNIKI.md).

Nowy, mocniejszy baseline 224×224 pod publikację: EfficientNet-B0 z SE od
zera na Tiny ImageNet, w recepturze opartej na kontrolowanym badaniu
Shahriar 2026 (arXiv:2607.01984). Zastępuje E005 (96×96, dev 51,53%) w roli
głównego punktu odniesienia do badań kompresji; E005 pozostaje jako baseline
96×96. Nie testujemy tutaj metod kompresji ani nowych kryteriów.

## Pytanie i cel

Czy w naszych warunkach (4×RTX 2080 Ti, DDP) odtworzymy jakość zbliżoną do
literatury: **~60% dev top-1** przy 224×224, 100 epokach, od losowych wag,
z słabszym przeuczeniem niż E005?

## Kotwica literaturowa i świadome różnice

- Shahriar 2026, **EfficientNet-B0 scratch, Tiny ImageNet, 224×224**:
  20 epok = 57,43%, **100 epok = 60,54%** (ich „test" to oficjalny val).
  Receptura 100-epokowa: AdamW lr 3e-4, wd 1e-4, label smoothing 0,1,
  clip gradientów 1,0, cosine (min lr 3e-6), batch 32, RandomResizedCrop
  0,80–1,00 (ratio 0,90–1,10, bicubic), color jitter (0,15/0,15/0,15/0,05),
  hflip, normalizacja ImageNet, AMP; **bez mixup/cutmix**.
- Nasze adaptacje (jawnie): globalny batch 128 (32/GPU), **wd 0,01**,
  warmup 5 epok, bez early stoppingu (wynik = `last.pt` po 100 epokach).
  Powód: wydajność na 4 GPU i spójność z E005; porównanie z literaturą
  ma charakter jakościowy, nie 1:1.

## Dane

Bez zmian względem E005: Tiny ImageNet (archiwum SHA-256
`6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`),
podział własny seed 20261010: train 90 000 (450/klasę), dev 10 000
(50/klasę); oficjalny val 10 000 to **rezerwa nieużywana**; test bez
etykiet. Cache `cache_train.npz` / `cache_val.npz`, `splits.json`.

## Protokół treningu (proponowany do zamrożenia)

- Model: `efficientnet_b0(weights=None, num_classes=200)`, inicjalizacja
  losowa; `seed_all(seed + rank)` **przed** budową modelu (naprawa błędu
  kolejności seedowania z E005).
- Wejście 224×224 (adaptacja z natywnych 64; porównanie z literaturą
  jakościowe). Train: RRC(224, 0,80–1,00, ratio 0,90–1,10, bicubic),
  ColorJitter(0,15/0,15/0,15/0,05), hflip, Normalize(ImageNet). Dev:
  deterministyczny resize bicubic 224, Normalize(ImageNet).
- AdamW lr 3e-4, wd 0,01, warmup 5 epok liniowo, potem cosine do
  min_lr 3e-6; gradient clipping 1,0; label smoothing 0,1; AMP FP16;
  SyncBN; DDP na dokładnie 4 GPU; batch 32/GPU (globalny 128);
  workers 3/rank. Deterministycznie (TF32 off, cuDNN deterministic).
- 100 epok; wynikiem jest `last.pt` (epoka 100); bez early stoppingu
  i bez wyboru epoki po fakcie; metryki train/dev co epokę.
- Zapasowy `partial_checkpoint.pt` (rolling, co 5 epok) — artefakt
  bezpieczeństwa (model bez optymalizatora); wznowienie biegu: od zera
  w nowym katalogu (jak w E005).

## Kontrole techniczne (nie wyniki naukowe)

- Smoke `runs/E007/smoke_20261011T061254Z` wykonany (2 epoki, 50/klasę =
  10 000 obrazów train, dev 1000): epoka 2 = 20,58 s (2,06 ms/obraz),
  dev top-1 0,4% → 1,1%, peak CUDA 1515 MiB/rangę (limit 11 264 MiB);
  brak błędów w `error_rank*.txt`.
- Ekstrapolacja pełnego biegu: ~185–250 s/epokę → ~5–7 h na 100 epok.

## Cztery GPU

Jeden model trenowany rozproszonie (DDP) na 4 GPU; rank `r` przetwarza
swoją 1/4 danych; agregacja `all_reduce`. To nie są cztery niezależne
powtórzenia. Batch 32/GPU → globalny 128 (bez akumulacji gradientu).

## Pliki

- [train_224.py](train_224.py) — trening DDP; `--smoke` do kontroli.
- [launch_224.py](launch_224.py) — launcher pełnego biegu.
- [config.draft.json](config.draft.json) — parametry proponowane.
- Docelowo: `config.json` (frozen), `PROTOKOL_v1.md`, raport `WYNIKI.md`.

## Ograniczenia z góry

- 224×224 to adaptacja (upscaling z 64), nie natywna rozdzielczość;
  nie tworzy nowego detalu, zwiększa tylko koszt i dopasowanie do
  protokołów literatury.
- Jeden seed i jeden bieg; brak mixup/cutmix świadomie (budżet 100 epok).
- Oficjalny val i test pozostają nietknięte; ocena wyłącznie na dev.

## Kolejne kroki

1. Smoke techniczny i korekta batch/workers na podstawie pomiaru.
2. Zamrożenie `config.json` (protocol 1.0) + `PROTOKOL_v1.md` + commit.
3. Pełny bieg ~15–18 h na 4 GPU (szacunek po smoke), w tle.
4. Raport `WYNIKI.md`: dev top-1 vs E005 (51,53%) i kotwica (60,54%).
