# E015 — wykonane kontrole przygotowawcze i audytowe

Wszystko poniżej wykonano na CPU; **nie uruchamiano treningu ani GPU**.
Oficjalny test CIFAR-100 i rezerwa nie były ładowane. Kontrole nie zapisują
wyników naukowych.

## 0. Cel audytu (2026-10-27)

Wcześniejsze przygotowanie porównywało warianty przy nominalnym celu
`round((1 − budget) · baseline_params)`. UNIFORM i MAG2 schodziły poniżej
celu o ok. 0,09 / 0,51 / 0,93% przy 30 / 50 / 70%, a TILT trafiał prawie
dokładnie w cel. TILT miał więc o 7 401 / 28 502 / 31 144 parametrów więcej
niż UNIFORM. Poprawka: per budżet target = **faktyczna** liczba parametrów
UNIFORM, a TILT repair i MAG2 są dopasowane do tego samego targetu.

## 1. Składnia

```bash
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  -m py_compile experiments/2026-10-27_E015_cifar100_alokacja/*.py
```

Wynik: `SYNTAX_OK`.

## 2. Kontrola CPU wszystkich 3 budżetów × 4 wariantów

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-27_E015_cifar100_alokacja/check_cpu_e015.py
```

Wynik: `E015_CPU_CHECKS=passed`. Kontrola nie ładuje obrazów dev
(`dev_images_loaded=false`), nie używa GPU (`gpu_used=false`), a dla każdej
kombinacji sprawdza solver, faktyczny folding/MAG2, skończoność parametrów
i logitów oraz zgodność liczby parametrów z solverem.

| Budżet | Wariant | Cel nominalny | Target = UNIFORM | Parametry | Odchylenie od targetu |
|---:|---|---:|---:|---:|---:|
| 30% | UNIFORM | 7 854 092 | 7 846 668 | 7 846 668 | 0,000% |
| 30% | TILT+0.30 | 7 854 092 | 7 846 668 | 7 846 598 | −0,0009% |
| 30% | TILT+0.15 | 7 854 092 | 7 846 668 | 7 846 415 | −0,0032% |
| 30% | MAG2 | 7 854 092 | 7 846 668 | 7 846 668 | 0,000% |
| 50% | UNIFORM | 5 610 066 | 5 581 542 | 5 581 542 | 0,000% |
| 50% | TILT+0.30 | 5 610 066 | 5 581 542 | 5 581 434 | −0,0019% |
| 50% | TILT+0.15 | 5 610 066 | 5 581 542 | 5 581 523 | −0,0003% |
| 50% | MAG2 | 5 610 066 | 5 581 542 | 5 581 542 | 0,000% |
| 70% | UNIFORM | 3 366 040 | 3 334 845 | 3 334 845 | 0,000% |
| 70% | TILT+0.30 | 3 366 040 | 3 334 845 | 3 334 746 | −0,0030% |
| 70% | TILT+0.15 | 3 366 040 | 3 334 845 | 3 334 789 | −0,0017% |
| 70% | MAG2 | 3 366 040 | 3 334 845 | 3 334 845 | 0,000% |

Tolerancja: realized params nie przekraczają wspólnego targetu i nie schodzą
poniżej niego więcej niż 1,0%. Wszystkie 12 kombinacji przechodzi; faktyczne
odchylenie TILT/MAG2 od UNIFORM to 0,000–0,003%. MAG2 ma identyczny target
i identyczną liczbę parametrów jak UNIFORM na wszystkich trzech budżetach.
Dodatkowo: 100 klas, 11 220 132 parametry, podział 40 000/5 000/5 000,
normalizacja z 40k train, deterministyczny folding `ratio=0.5` daje
2 820 740 parametrów, `hkmeans`/`sklearn` dostępne.

## 3. Odrzucanie smoke train runów

Sprawdzono dwie ścieżki: `launch_folding.load_training_run` i
`prepare_folding_config.load_training_manifest`. Kontrola w
`check_cpu_e015.py` tworzy tymczasowy manifest ukończonego smoke runu; obie
funkcje rzucają `RuntimeError` z komunikatem „Folding wymaga pełnego
treningu, nie smoke” (`smoke_run_rejected_by_folding=true`).

Dodatkowo sprawdzono rzeczywisty ukończony smoke run
`runs/E015/20261027T180801Z_cb9ffaf_seeds42-45_86d9af16`
(`smoke=true`, `effective_epochs=2`): obie ścieżki go odrzucają. Smoke nie
jest więc materiałem do foldingu; potrzebny jest nowy pełny bieg 200 epok.

## 4. Analiza na syntetycznym katalogu 48 summary

`analyze_e015.py` uruchomiono na syntetycznym katalogu 48 summary w `/tmp`
z nowymi polami targetu. Wynik: kod 0, `TILT_P030_POSITIVE=12/12`;
`analysis.json`, `analysis.md`, `seed_level.csv` i `summary_stats.csv`
zawierają osobno redukcję nominalną, targetu UNIFORM i faktyczną. To tylko
test składni/ścieżek, nie wynik naukowy.

## 5. Nierozstrzygnięte przed biegiem

- Rzeczywisty czas treningu CIFAR-100 i zużycie VRAM.
- Czy baseline z checkpointu odtworzy się dokładnie na GPU (launcher wymaga
  top-1 bez różnicy i NLL w tolerancji 1e-5).
- Czy wynik TILT+0.30 spełni zamrożone kryterium — to rozstrzygnie dopiero bieg.
- **Bloker proceduralny:** pełny trening trzeba rozpocząć po tych zmianach.
  `launch_folding.py` porównuje hashe źródeł z manifestem treningu; bieg
  rozpoczęty przed audytem albo smoke run nie przejdzie tej kontroli.
