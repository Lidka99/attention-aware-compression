# E045 — krzywa łagodnego UNIFORM na Tiny dev

Figura do zakończonego runu `runs/E045/20261122T031350Z_a58b56a3`
(audyt PASS), tylko faza **dev**, po CMA. Bez treningu, obrazów, wag,
inferencji i GPU; bez punktów val. Protokół: [`../../OPIS.md`](../../OPIS.md).

## Pliki

| Plik | Zawartość |
|---|---|
| `e045_krzywa.png` | figura 200 dpi (2560×1600 px) |
| `e045_krzywa.pdf` | ta sama figura, wektorowa |
| `make_figure.py` | deterministyczny generator (tylko trzy pliki `analysis/`) |

## Co jest na wykresie

Dwa panele, Tiny ImageNet dev (10 000 indeksów E035), ResNet18/Tiny64,
cztery seedy 42–45, średnia i punkty seedów (±1 SD jako tło):

- **lewy:** bezwzględny top-1 [%] vs **rzeczywista redukcja P [%]**;
- **prawy:** sparowana strata pełny−U [pp] (ta sama seed para) vs
  **rzeczywista redukcja MAC [%]**.

Punkty niebieskie (0/10/20%) to nowe pomiary **E045** (12 ocen: pełny model
plus UNIFORM b10/b20). Punkty pomarańczowe `*` (b30/b50) to **reuse U z dev
E035** (8 ocen), nie liczone ponownie. P wyznacza budżet po dyskretyzacji
kanałów, MAC jest mierzony (64×64); dlatego rzeczywiste redukcje różnią się
od nominalnych: b10 P/MAC **10,35/10,67%**, b20 **20,33/19,97%**,
b30 **30,44/30,13%**, b50 **50,15/49,13%**.

| Punkt | top-1 [%] | strata pełny−U [pp] |
|---|---:|---:|
| pełny (0%) | 51,22 | 0,00 |
| b10 (E045) | 46,36 | 4,86 |
| b20 (E045) | 40,58 | 10,64 |
| b30 (E035 reuse) | 34,81 | 16,41 |
| b50 (E035 reuse) | 19,93 | 31,30 |

## Weryfikacja wejść

- `audit.json`: `status PASS`, `phase dev`, `val_used false`,
  `four_gpu_exit0 true`, `rows 20`, `teacher_exact_E035 4`.
- `per_seed.csv`: 20 wierszy, **0 duplikatów** `(seed, budżet)`,
  kompletne 4 seedy × 5 budżetów, po 10 000 próbek; 12 wierszy E045
  i 8 E035, zgodnie z `summary.json`.
- `summary.json`: 5 punktów; średnie, SD i straty per seed zgadzają się
  z `per_seed.csv`; redukcje P/MAC odtwarzają się z `params`/`mac`.

SHA256 wejść (sprawdzane na sztywno w generatorze):

```
ca6a0d74db6974d52f3795e82f3c7803503c225adfe78fa203aa03024b4e8811  summary.json
486309557455840de25852f166e55e6c909cbb1aa23e477418f8ed3057a2a992  per_seed.csv
fcca52a0fbd3020931b34d275f6fb84f278ac1d47f6575a850c5c0d3df20ad47  audit.json
```

SHA256 figur (powtarzalne przy ponownym uruchomieniu):

```
e3f897d1891e1a4f843f5e4e1ac08f86cd77875e9641c712f114fda7d8a38502  e045_krzywa.png
7ebee9a4599aee671c9c95921b8967c036830f06d0f5b79394ba1afd8a133f48  e045_krzywa.pdf
```

## Odtworzenie

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-22_E045_lagodny_uniform_tiny/wykresy/20261122T031350Z_a58b56a3/make_figure.py
```

CPU/matplotlib only. Ograniczenia: punkty b30/b50 to zamrożone odniesienia
E035 dev, a nie nowe oceny; figura nie zawiera val ani testu i nie służy
do oceny przewagi L06 przy b10/b20.
