# 2026-10-27_E015_cifar100_alokacja

Status: **protokół 0.1 przygotowany; trening i folding nieuruchomione**.
Data utworzenia: 2026-10-27 (UTC). To nowa próba badawcza E015, nie powtórka
E014. Wyniki nie są jeszcze znane.

## Cel

Sprawdzić, czy nierówna alokacja budżetu foldingu (TILT+0.30) bije jednolity
folding (UNIFORM) na **drugim datasecie**: ResNet18/CIFAR-100, cztery
niezależne treningi od zera (seedy 42–45), a następnie **zamrożone
porównanie alokacji bez fine-tuningu**. E015 nie zmienia E011–E014 i nie
korzysta z attention_v2/attention_v3.

## Wejście i dane

- CIFAR-100 jest już obecny lokalnie w `data/cifar100/`; nie pobieramy danych.
- Używamy istniejącego podziału `data/cifar100/splits.json`
  (SHA-256 `ecb8804e41fbc4bee7c61fd0585bf1c7dcfc94ac8472333fc283d51dc6460e86`):
  **40 000 train / 5 000 dev / 5 000 rezerwy**, po 400/50/50 na klasę,
  seed podziału 20260926.
- Plik train CIFAR-100: SHA-256
  `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b`;
  meta: `a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9`;
  archiwum: `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.
- **Oficjalny test 10 000 obrazów nie jest wypakowywany, ładowany ani
  oceniany. Rezerwa 5 000 obrazów z train również pozostaje nietknięta.**
- Normalizacja jest stała i policzona wyłącznie na 40 000 train:
  mean `(0.507421, 0.486865, 0.441380)`,
  std `(0.267468, 0.256482, 0.275979)`. Nie estymujemy jej z dev/test.

## Trening od zera (4 niezależne modele)

Receptura jest adaptacją E011 do CIFAR-100: ten sam optymizator, liczba epok,
augmentacje i determinizm; zmieniają się dataset, liczba klas (100) i
normalizacja.

- Model: torchvision ResNet18 bez wag, `conv1` 3×3/stride1/padding1,
  `maxpool=Identity`, `fc=512→100`, **11 220 132 parametry**.
- Cztery niezależne treningi: GPU0 seed42, GPU1 seed43, GPU2 seed44,
  GPU3 seed45. Jeden proces = jedno GPU = jeden seed; bez DDP, bez
  akumulacji gradientów i bez uśredniania modeli.
- Batch 128 na GPU i na model; efektywny globalny batch **jednego przebiegu**
  to 128. Cztery GPU nie tworzą jednego większego batcha.
- SGD `lr=0.1`, `momentum=0.9`, `weight_decay=0.0005`, bez Nesterova.
  200 epok, cosine do 0, krok schedulera po epoce, bez warmupu.
- Cross entropy bez label smoothing, FP32, bez AMP. Algorytmy deterministyczne,
  TF32 wyłączone, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Train: `RandomCrop(32,pad4)`, HFlip, `RandomRotation(15)`, ToTensor,
  normalizacja CIFAR-100. Dev: `CenterCrop(32)`, ToTensor, ta sama
  normalizacja.
- Wynik główny: epoka 200. Brak early stoppingu i wyboru `best` po dev.
- **Retencja checkpointów:** tylko finalny `seed_<seed>/last.pt` (state_dict +
  metadane, bez optymizatora), zapisany atomowo raz na końcu. Brak
  checkpointów pośrednich i brak wznowienia; awaria oznacza nowy bieg.
  `metrics.jsonl` po każdej epoce zachowuje pełną krzywą.

## Zamrożone porównanie alokacji (bez fine-tuningu)

Po zakończeniu treningu `launch_folding.py` generuje dla każdego seeda
zamrożony config z hashem checkpointu i splitu oraz baseline top-1/NLL na dev.
Następnie uruchamia **48 workerów**: 4 seedy × 3 budżety × 4 warianty.

Budżety: **30%, 50%, 70%** redukcji parametrów. Warianty:

1. **UNIFORM** — ten sam keep ratio dla wszystkich 12 grup (baseline).
   Jego **faktyczna** liczba parametrów dla nominalnego budżetu jest wspólnym
   targetem pozostałych wariantów.
2. **TILT+0.30** — główna zamrożona hipoteza:
   `r_g = clip(base + 0.30·p_g, 0.10, 0.90)`, `p_g = 2g/11 − 1`;
   wczesne grupy mniej skompresowane, późne bardziej; `base` i repair
   szerokości dobierane do wspólnego targetu UNIFORM, bez przekraczania go.
3. **TILT+0.15** — wtórna, łagodniejsza wersja tego samego kierunku,
   dopasowana do tego samego targetu.
4. **MAG2** — magnitude pruning L2, uniform per warstwa, dopasowany do tego
   samego wspólnego targetu (faktyczny UNIFORM).

Operator foldingu i alokacji jest **importowany bez zmian** z tego repo:

- `experiments/2026-10-24_E012_folding_referencja/folding.py`
- `experiments/2026-10-24_E012_folding_referencja/evaluate.py`
- `experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py`
- `experiments/2026-10-27_E014_replikacja_alokacji/pruning.py`

Launcher zapisuje snapshoty tych plików i ich SHA-256 w katalogu biegu.
`launch_folding.py` odmawia startu, jeśli kod E015 albo używane pliki
E012/E014 zmieniły się po treningu względem snapshotu. Nie kopiujemy kodu z
attention_v2/attention_v3.

Wszystkie warianty: BN na pełnym train 40 000, seed BN `20261024`, bez
gradientów i bez aktualizacji wag; ewaluacja na 5 000 dev; batch 128;
fold seed 42; te same transformacje BN co w E014. „Bez fine-tuningu” oznacza
brak gradientów; korekta BN korzysta z train bez etykiet w sensie optymalizacji.

**Wspólny target i tolerancja (audyt 2026-10-27):** dla każdego budżetu
najpierw liczymy `nominal_target = round((1 − budget) · baseline_params)`.
UNIFORM jest rozwiązywany pod ten nominalny limit (cap), a jego **faktyczna**
liczba parametrów staje się wspólnym targetem TILT+0.30, TILT+0.15 i MAG2.
Dzięki temu żaden wariant nie dostaje dodatkowych parametrów względem
UNIFORM. Wcześniejsza wersja przygotowania celowała TILT w nominalny limit,
przez co przy 30/50/70% TILT miał o ok. 0,09/0,51/0,93% więcej parametrów
niż UNIFORM — to nie było uczciwe porównanie. Teraz raportujemy osobno
redukcję **nominalną** (z `budget`), redukcję **targetu** (UNIFORM) i redukcję
**faktyczną** każdego wariantu.

Realized params nie mogą przekroczyć wspólnego targetu i mogą być od niego
niższe o najwyżej 1,0%; w kontroli CPU odchylenie TILT/MAG2 od UNIFORM wynosi
0,000–0,003%. Nie renormalizujemy budżetu po wyniku i nie stroimy solvera.

## Kryterium replikacji (ustalone przed biegiem)

Dla **TILT+0.30 vs UNIFORM**:

- średnia różnica top-1 po BN > 0 na **każdym** budżecie,
- przy 50% i 70%: średnia różnica **≥ +2,0 pp** oraz dodatnia na **≥ 3/4**
  seedach,
- przy 30%: średnia > 0 oraz dodatnia na ≥ 3/4 seedach,
- TILT+0.30 nie gorszy od MAG2 o więcej niż 0,5 pp średnio.

Raportujemy też TILT+0.15, NLL, agreement, MAE/RMSE logitów, KL, czasy oraz
nominalne i faktyczne budżety (liczby parametrów, redukcję nominalną,
redukcję targetu UNIFORM i redukcję faktyczną). Nie zmieniamy progu po
wyniku. Jeśli replikacja się nie uda, raportujemy wynik negatywny.

## Uruchomienie

Z katalogu głównego repo, w środowisku Conda `attention`:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-27_E015_cifar100_alokacja/check_cpu_e015.py

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-27_E015_cifar100_alokacja/launch_training.py --smoke

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-27_E015_cifar100_alokacja/launch_training.py
```

Po zakończeniu treningu (katalog `runs/E015/<TRAIN_RUN>` wypisany przez
launcher):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-27_E015_cifar100_alokacja/launch_folding.py --train-run runs/E015/<TRAIN_RUN>

/home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-27_E015_cifar100_alokacja/analyze_e015.py --fold-run runs/E015/<FOLD_RUN> --output-dir runs/E015/analizy/<UTC>_<ID>
```

Launchery wymagają czterech wolnych GPU 0–3 (≥ 4 GiB wolnej VRAM, brak
aktywnych procesów), środowiska `attention`, CIFAR-100 i ≥ 2 GiB wolnego
dysku. `--smoke` to 2 epoki na 1000 train/200 dev; nie ma znaczenia
naukowego. Folding celowo odrzuca manifesty smoke (`smoke is False` oraz
`effective_epochs == 200`) w `launch_folding.py` i
`prepare_folding_config.py`. `launch_training.py --data-only` tylko waliduje
dane i zapisuje podział w katalogu biegu. `check_cpu_e015.py` sprawdza bez
GPU i bez ładowania obrazów dev wszystkie 12 kombinacji 3 budżetów ×
4 wariantów oraz odrzucanie smoke runów.

## Środowisko

Conda `attention`, Python 3.11, torch 2.12.0+cu130, torchvision 0.27.0+cu130,
numpy 2.4.6, `hkmeans`/`scikit-learn` dostępne, 4× RTX 2080 Ti. Launcher
zapisuje `environment.json`, `pip-freeze.txt`, `conda-explicit.txt` i
`nvidia-smi.txt` w katalogu biegu.

## Szacowane zasoby

- Trening: cztery procesy równolegle, ~1,1–1,5 h na seed (E011/CIFAR-10:
  ~4 480 s); wall-time ~1,5 h. VRAM ~1,7–2,0 GB na GPU (E011: 1 690 MB).
- Folding: 48 workerów falami po 4; ~30–60 s na worker, wall-time ~10–20 min;
  VRAM < 0,5 GB na GPU (E014: 231 MB).
- Dysk: ~180 MB finalnych checkpointów treningu (4 × ~45 MB) + logi,
  metryki i snapshoty (< 100 MB); folding z logits/diagnostics ~0,2–0,5 GB.
  Folding nie kopiuje checkpointów treningu — weryfikuje ich hashe pod
  ścieżkami biegu treningowego. Łącznie < 1 GB; przy 8,6 GB wolnego miejsca
  jest zapas. Launcher pilnuje minimum 2 GiB.
- Czasy są szacunkami z E011/E014, nie pomiarem E015.

## Co wolno będzie twierdzić po biegu

- Że na CIFAR-100/ResNet18, przy czterech checkpointach i ocenie na dev,
  alokacja depth-aware poprawia jakość po korekcie BN bez aktualizacji wag —
  **tylko jeśli kryterium zostanie spełnione**.
- O kierunku i rozrzucie między seedami (n=4), bez istotności populacyjnej.

## Czego nie wolno twierdzić

- Że metoda działa na ImageNet lub innych architekturach — brak takich danych.
- Że jesteśmy lepsi od *Cut Less, Fold More* 1:1 — to adaptacja operatora,
  nie oficjalny kod/liczby.
- Że metoda jest „data-free” — korekta BN używa train bez gradientów.
- Że TILT+0.30 jest optymalny — to zamrożona hipoteza, nie strojenie.
- Że dev jest niezależnym testem — to wydzielony zbiór walidacyjny;
  oficjalny test pozostaje nietknięty.

## Ograniczenia protokołu

- CIFAR-100 ma istniejący podział 40 000 train / 5 000 dev / 5 000 rezerwy
  (nie 45 000/5 000 jak E011 na CIFAR-10). Rezerwa jest świadomie nietknięta.
- Cztery checkpointy pochodzą z jednego przepisu treningowego; nie są
  niezależnymi zbiorami, tylko niezależnymi inicjalizacjami.
- Brak niezależnego testu i drugiej architektury w tym eksperymencie.
- Deterministyczność dotyczy tego samego sprzętu i środowiska; nie
  gwarantujemy bitowej zgodności między różnymi GPU/wersjami bibliotek.
- Folder E015 może być niecommitowany (zgodnie z poleceniem); źródłem prawdy
  jest snapshot `source/` i jego hashe, nie czystość git.

## Pliki

- `data_cifar100.py` — dane, podział, transformacje, model.
- `train_cifar100.py` — trening jednego seeda.
- `launch_training.py` — cztery treningi na czterech GPU.
- `prepare_folding_config.py` — baseline i zamrożony config seeda.
- `folding_worker.py` — alokacja, folding/MAG2, BN, ewaluacja.
- `launch_folding.py` — 48 workerów na czterech GPU.
- `analyze_e015.py` — analiza CPU po foldingu.
- `check_cpu_e015.py` — kontrole CPU danych, modelu, wspólnego targetu
  (wszystkie budżety i warianty) oraz odrzucania smoke runów.
- `TESTY.md` — wykonane kontrole przygotowawcze i audytowe na CPU.
- `configs/train.json`, `configs/allocation.json`.
