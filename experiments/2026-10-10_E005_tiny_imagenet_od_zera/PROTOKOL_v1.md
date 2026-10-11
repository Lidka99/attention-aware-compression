# E005 — wersja wykonania 1.0

Eksperyment: `2026-10-10_E005_tiny_imagenet_od_zera`.
Data zamrożenia: 2026-10-11 UTC. **Status protokołu: frozen**.
Status wykonania podaje manifest biegu.

## Cel i pytanie

Wytrenować od zera EfficientNet-B0 z SE na Tiny ImageNet (200 klas,
prawdziwe zdjęcia) jako nowy, mniej przeuczony baseline do badania
selekcji kanałów, po negatywnym wyniku kontrolnym E004 na CIFAR-100.
To nie test nowej metody; nie używamy publicznych wag.

## Ustalenia zamrożone

- Dane: Tiny ImageNet, archiwum SHA-256
  `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`;
  podział własny seed 20261010: train 90 000 (450/klasę), dev 10 000
  (50/klasę); oficjalny val 10 000 to **rezerwa nieużywana**.
  Do modelu trafiają wyłącznie obrazy train/dev; test i rezerwa nie.
- Rozdzielczość efektywna 96×96 (adaptacja protokołu literatury z 224;
  spójność map SE 6×6 z E001–E003 — jawnie opisana).
- Budżet: **100 epok** (Shahriar 2026, arXiv:2607.01984 — kontrolowane
  badanie EfficientNet-B0 na Tiny ImageNet: 90k/10k, 100 epok, cosine).
- AdamW lr 1e-3, wd 0,01; warmup 5 epok liniowo potem cosine do zera;
  AMP FP16; SyncBN; DDP na dokładnie 4 GPU; batch 32/GPU (globalny 128).
- Augmentacja: RandomResizedCrop(96, scale 0,6–1,0) + flip; normalizacja
  mean=std=0,5; dev bez augmentacji. Determinizm jak w E001.
- Wynik: dev top-1/NLL co epokę; `last.pt` po 100 epokach. Bez early
  stoppingu i bez wyboru epoki po fakcie.

## Kontrole przed biegiem (techniczne, nie naukowe)

- Smoke i pilot na podzbiorach (OPIS.md): pipeline działa, model się
  uczy (0,5%→8,0% w 10 epokach na 10k obrazów); naprawiony błąd
  poczwórnego liczenia dev. Ekstrapolacja: ~3,5–4 h.

## Komenda

```bash
conda run --no-capture-output -n attention python experiments/2026-10-10_E005_tiny_imagenet_od_zera/launch_tiny.py
```

Launcher: czysty Git, frozen config, 4 GPU, nowy katalog
`runs/E005/<UTC>_<git>_seed42_<unikalny>/` z manifestem, konfiguracją,
środowiskiem, komendą, logami; po zakończeniu `artifacts.sha256`.
Wznowienie: brak (przerwany bieg powtarzamy od zera w nowym katalogu).

## Ograniczenia

Jeden seed; dev z tego samego zbioru train (oficjalny val nietknięty);
wynik od zera będzie znacznie niższy niż liczby z pretreningiem ImageNet
(literatura: ~60% przy 224×224, u nas adaptacja 96×96). Nie porównujemy
tu metod kompresji — kolejny eksperyment (E006) użyje tego baseline.
