# E007 — wersja wykonania 1.0

Eksperyment: `2026-10-11_E007_baseline_224`.
Data zamrożenia: 2026-10-11 UTC. **Status protokołu: frozen**.
Status wykonania podaje manifest biegu.

## Cel i pytanie

Nowy, mocniejszy baseline pod publikację: EfficientNet-B0 z SE od zera na
Tiny ImageNet przy 224×224, receptura oparta na Shahriar 2026
(arXiv:2607.01984; ich scratch 100 ep = 60,54%), zaadaptowana do 4 GPU.
Nie testujemy tu metod kompresji. E005 (96×96, dev 51,53%) pozostaje jako
baseline pomocniczy.

## Ustalenia zamrożone

- Dane: Tiny ImageNet (archiwum SHA-256
  `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`);
  podział własny seed 20261010: train 90 000 (450/klasę), dev 10 000
  (50/klasę); oficjalny val 10 000 to **rezerwa nieużywana**; test bez
  etykiet; ocena wyłącznie na dev.
- Model: `efficientnet_b0(weights=None, num_classes=200)`, losowa
  inicjalizacja; `seed_all(seed + rank)` przed budową sieci (naprawa
  kolejności seedowania z E005).
- Wejście 224×224 (adaptacja z natywnych 64; porównanie z literaturą
  jakościowe). Train: RandomResizedCrop(224, 0,80–1,00, ratio 0,90–1,10,
  bicubic), ColorJitter(0,15/0,15/0,15/0,05), hflip, Normalize(ImageNet).
  Dev: deterministyczny resize bicubic 224.
- AdamW lr 3e-4, wd 0,01, warmup 5 epok liniowo, potem cosine do
  min_lr 3e-6; gradient clipping 1,0; label smoothing 0,1; AMP FP16;
  SyncBN; deterministycznie (TF32 off, cuDNN deterministic);
  DDP na dokładnie 4 GPU; batch 32/GPU (globalny 128, bez akumulacji);
  workers 3/rank.
- 100 epok; wynikiem jest `last.pt` (epoka 100); bez early stoppingu
  i bez wyboru epoki po fakcie; metryki train/dev co epokę;
  `partial_checkpoint.pt` (rolling, co 5 epok) jako artefakt
  bezpieczeństwa; wznowienie: od zera w nowym katalogu.

## Kontrole przed biegiem (techniczne, nie naukowe)

- Smoke `runs/E007/smoke_20261011T061254Z` (2026-10-11, commit przed
  zamrożeniem): 2 epoki, 10 000 obrazów train (50/klasę), dev 1000;
  epoka 2: **20,58 s** (2,06 ms/obraz), dev top-1 0,4% → 1,1%,
  **peak CUDA 1515 MiB/rangę** (limit 11 264); brak błędów.
- Ekstrapolacja pełnego biegu: ~185–250 s/epokę → **~5–7 h na 100 epok**
  na 4 GPU.
- `train_224.py` sprawdzony składniowo; transformacje i harmonogram LR
  przetestowane na CPU przed smoke.

## Komenda

```bash
conda run --no-capture-output -n attention python experiments/2026-10-11_E007_baseline_224/launch_224.py
```

Launcher: czysty Git, frozen config, dokładnie 4 GPU, nowy katalog
`runs/E007/<UTC>_<git>_seed42_<unikalny>/` z manifestem, konfiguracją,
środowiskiem, komendą i logami; po zakończeniu `artifacts.sha256`.
Wznowienie: brak (przerwany bieg powtarzamy od zera w nowym katalogu).

## Ograniczenia

224×224 to adaptacja (upscaling z 64 — bez nowego detalu); jeden seed;
brak mixup/cutmix świadomie (budżet 100 epok, zgodnie z kotwicą);
porównanie z literaturą jakościowe (ich ocena na oficjalnym val,
nasza na dev; inny batch/wd/warmup). Oficjalny val i test nietknięte.
