# 2026-10-10_E005_tiny_imagenet_od_zera

Wersja protokołu: **0.1-draft**. Status: dane i kod przygotowane; testy na
podzbiorze wykonane (techniczne); **pełny trening jeszcze nieuruchomiony**.
Nowy, niezależny baseline do dalszych badań kompresji — zastępuje CIFAR-100
jako główny zbiór po negatywnym wyniku kontrolnym E004.

## Pytanie i cel

Czy potrafimy od zera wytrenować EfficientNet-B0 z SE na **Tiny ImageNet**
(200 klas, prawdziwe zdjęcia 64×64) w warunkach powtarzalnych na 4 GPU,
tak aby uzyskać mocniejszy i mniej przeuczony punkt odniesienia do badania
selekcji kanałów niż przeuczony model E001 na CIFAR-100?

To baseline, nie test nowej metody. Nie używamy publicznych wag.

## Dane (zweryfikowane)

- Źródło: https://cs231n.stanford.edu/tiny-imagenet-200.zip,
  SHA-256 archiwum:
  `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`.
- Podział własny (seed 20261010): 450/50 na klasę z oficjalnego train:
  **train 90 000**, **dev 10 000**; oficjalny val (10 000, etykiety)
  pozostaje **rezerwą, nieużywaną**. Oficjalny test nie ma etykiet.
- Cache pamięciowy: `data/tiny-imagenet/cache_train.npz` (100k×64×64×3 + etykiety),
  `cache_val.npz`; `splits.json` z listami indeksów i hashami.

## Protokół treningu (proponowany, do zamrożenia)

- Efektywna rozdzielczość **96×96** (spójność z pipeline E001/E002/E003:
  mapy SE 6×6; adaptacja protokołu literatury z 224×224 — jawnie opisana).
- **100 epok** — budżet z kontrolowanego badania EfficientNet-B0 na Tiny
  ImageNet (Shahriar 2026, arXiv:2607.01984: 90k/10k, 100 epok, cosine;
  scratch ≈ 60,5% przy 224×224).
- AdamW lr 1e-3, wd 0,01, warmup 5 epok liniowo, potem cosine do zera;
  AMP FP16, SyncBN, DDP na dokładnie 4 GPU, batch 32/GPU (globalny 128).
- Augmentacja: RandomResizedCrop(96, scale 0,6–1,0), flip poziomy;
  normalizacja mean=std=0,5 (jak E001). Dev bez augmentacji.
- Determinizm i zapis: jak E001 (hashe, manifest, metrics.jsonl, seed 42).

## Kontrole techniczne wykonane 2026-10-10

- `runs/E005/smoke_20261010T235854Z` (3 epoki, 10 obrazów/klasę):
  pełny pipeline, zapis metryk i checkpointu OK.
- `runs/E005/pilot_20261011T000010Z` (10 epok, 50 obrazów/klasę = 10k):
  dev top-1 **0,55% → 8,0%**, train NLL 5,40 → 4,36 — nauka działa
  (losowo = 0,5%); ~14,5 s/epokę dla 10k obrazów.
- Wcześniejszy bieg `runs/E005/smoke_20261010T235751Z` przerwany z błędem
  indeksowania (dev z złego cache) — zachowany jako nieudany.
- Wykryte i naprawione przed pełnym biegiem: dev liczony wcześniej
  poczwórnie przez brak podziału rank::4 (zużyte tylko w testach).
- Ekstrapolacja pełnego biegu: ~130 s/epokę × 100 epok ≈ **3,5–4 h** na 4 GPU.

Testy na podzbiorze są wyłącznie kontrolą techniczną — nie są wynikiem
naukowym i nie zastępują pełnego treningu.

## Pliki

- [prepare_data_tiny.py](prepare_data_tiny.py) — listy + cache (odmawia nadpisania).
- [train_tiny.py](train_tiny.py) — trening DDP; `--smoke` do kontroli podzbioru.
- [config.draft.json](config.draft.json) — parametry proponowane.
- Docelowo: config.json (frozen), launcher, raport WYNIKI.md, protokół
  pełny z komendą i hashami środowiska.
