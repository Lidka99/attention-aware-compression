# E019 — alokacja TILT na Tiny ImageNet (protokół 1.0, zamrożony)

Status: **protokół zamrożony przed uruchomieniem treningu** (2026-10-29 UTC).
Trzeci zbiór dla najmocniejszego wyniku projektu: nierówna alokacja budżetu
foldingu (TILT) bez fine-tuningu.

## Pytanie
Czy przewaga TILT+0.30 nad UNIFORM, potwierdzona na CIFAR-10 (E014, 4 seedy),
a słabsza na CIFAR-100 (E015), przenosi się na Tiny ImageNet?

## Dane i model
- Tiny ImageNet 64×64, 200 klas; lokalne `cache_train.npz` + `splits.json`
  (pochodzenie E005). Train 90 000 / dev 10 000 / rezerwa 10 000
  (osobny `cache_val.npz`, oficjalny val).
- **Oficjalny val/test i rezerwa nie są ładowane ani oceniane.**
- Model: torchvision **ResNet18 ze standardowym stem** (7×7/stride2 + maxpool),
  fc 200 klas, od losowej inicjalizacji. 11 279 112 parametrów.
  To inna wariant architektury niż CIFAR (conv1 3×3, bez maxpool) —
  świadoma decyzja obliczeniowa; operator E012 działa bez zmian.
- Normalizacja: średnia/odchylenie policzone **wyłącznie z 90k train**.
- Transformacje: RandomResizedCrop(64, scale 0.4–1.0) + flip (train);
  resize 64 + normalizacja (dev).

## Trening
- Cztery niezależne seedy 42–45, po jednym na GPU 0–3, 200 epok.
- SGD lr 0.1, momentum 0.9, wd 5e-4, cosine, batch 128/GPU
  (globalny 128 na przebieg), bez AMP, bez label smoothing.
- Zapis: `metrics.jsonl` co epokę, finalny `last.pt` bez optymizatora,
  `summary.json`. Unikalny katalog `runs/E019/<UTC>_<git>_seeds42-45_<token>/`.

## Folding (po treningu)
- Operator E012/E014 bez zmian; `preserve_fc_bias=True`, `fold_seed=42`.
- Korekta BN: pełny train 90k bez etykiet i gradientów, `bn_seed=20261024`.
- Budżety **parametrów**: 30/50/70%; wspólny target = faktyczna liczba
  parametrów UNIFORM dla danego seeda i budżetu.
- Warianty: UNIFORM (baseline), TILT+0.30 (primary), TILT+0.15 (secondary),
  MAG2 (comparator).
- 4 seedy × 3 budżety × 4 warianty = **48 zadań**, fale po cztery na 4 GPU.
- Ocena wyłącznie na dev 10 000.

## Kryterium sukcesu (zamrożone)
TILT+0.30 uznajemy za potwierdzony na Tiny ImageNet, jeśli przy **50% i 70%**
redukcji parametrów:
1. średnia różnica względem UNIFORM (4 seedy) wynosi **≥ +1,0 pp**, oraz
2. różnica jest dodatnia na **co najmniej 3 z 4 seedów**.

30% raportujemy pomocniczo. Wynik spełniający kryterium oznacza spójność
z E014, nie publikację. Wynik niespełniający kryterium zapisujemy jako
negatywny — bez zmiany progu.

## Rewizja przed wynikami
- Pierwszy start treningu (`20261030T000316Z...`) przerwano po <1 epoce,
  bo normalizację policzono z 100k obrazów (z dev/rezerwy) zamiast z 90k
  train. Nowy bieg używa statystyk train-only. Przerwany bieg zachowany,
  bez wyników naukowych.
- Kod foldingu zapisano po starcie treningu; jego hashe są rejestrowane
  w manifeście foldingu. Żaden plik wejściowy treningu nie zmienił się.

## Ograniczenia
- Jedna architektura (ResNet18), jeden rozmiar 64×64, jeden przepis treningu.
- Brak niezależnego testu; dev był używany w E019 tylko do oceny.
- Porównanie z *Cut Less, Fold More* to adaptacja operatora, nie oficjalny kod 1:1.

## Odtworzenie
```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CONDA_DEFAULT_ENV=attention CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E019_tinyimagenet_alokacja/launch_training.py
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E019_tinyimagenet_alokacja/check_cpu_e019.py
CONDA_DEFAULT_ENV=attention CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E019_tinyimagenet_alokacja/launch_folding.py \
  --train-run runs/E019/<TRAIN_RUN>
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E019_tinyimagenet_alokacja/analyze_e019.py \
  --fold-run runs/E019/<FOLD_RUN> --output-dir runs/E019/analizy/<UTC>_<ID>
```
