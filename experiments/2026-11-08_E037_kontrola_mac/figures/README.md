# E037 — wykresy kontroli przy dopasowanym MAC (dev only)

Zestaw wygenerowany wyłącznie na CPU (matplotlib, Conda `attention`).
Dane pochodzą z surowych `summary.json` E037 (profil U') oraz oryginalnych
`summary.json` dev E035 (profile U i T). Plany i MAC pochodzą z
`frozen.json` E037; `analysis.json` służy do kontroli zgodności.
**E037 jest kontrolą dev — test nie był otwierany** (`dev_only=true`,
`data_open_attempts=0` w `analysis/verification.json`).

## Pliki

| Plik | Co pokazuje |
|---|---|
| `e037_top1_u_t_up.png/.pdf` | Top-1 CMA na dev dla U, T i U' (UNIFORM dopasowany MAC) przy 30/50/70%, pięć par. Średnia ± SD z 4 seedów. |
| `e037_t_minus_up.png/.pdf` | Sparowana różnica TILT − U' po CMA na dev. Punkty = 4 seedy, czarny romb = średnia, słupki = SD; linia zera. |
| `e037_up_params_mac.png/.pdf` | Budżet U' względem T: iloraz parametrów i iloraz MAC. Linia odniesienia = 1. |
| `punkty.csv` | Wszystkie punkty użyte na wykresach (per seed oraz średnie). |
| `sources.json` | SHA256 i rozmiary plików wejściowych oraz skryptu; potwierdzenie niezmienności wejść. |
| `weryfikacja.json` | Liczebności, średnie, SD, zakresy parametrów/MAC i maksymalne różnice względem `analysis.json`. |

## Dane i liczenie

Dla każdej pary, budżetu i seeda:

```text
U'  = summary E037, metrics.CMA.top1
U,T = oryginalne summary dev E035, metrics.{U,T}.CMA.top1
delta_pp(T−U') = (T − U') × 100
params U'/T    = frozen.plans[pair][tag].Uprime_params / T_params
MAC U'/T       = frozen.plans[pair][tag].Uprime_mac / T_mac
```

Średnia i SD są liczone po czterech seedach (42–45). **Słupki to SD
(ddof=1), nie CI.** Na wykresie różnic jest linia zera; zero = brak różnicy.
Nie dodawano oznaczeń istotności ani progów istotności.

## Odczyty

- U' ma **1,099–1,821× parametrów T** (w zaokrągleniu 1,10–1,82×) i
  **0,9849–0,9986× MAC T** (w zaokrągleniu 0,985–0,999×). To kontrola
  dopasowana po MAC, nie po rozmiarze.
- Przy dopasowanym MAC średnia T − U' jest dodatnia tylko w 5/15 komórek
  (głównie CIFAR-10 i PreAct/CIFAR-10); w pozostałych 10/15 przewagę ma U'.
  To wynik opisowy na dev, bez testu istotności.
- Nie należy twierdzić przyczynowo, że zysk przy stałych parametrach
  „wynika z compute”. E037 pokazuje jedynie, że porównanie przy stałych
  parametrach nie jest porównaniem przy stałym MAC. Nie rozstrzyga, jaka
  część różnicy pochodzi z alokacji, a jaka z kosztu obliczeń.

## Ograniczenia

- Dev był używany wcześniej; to kontrola mechanizmu, nie niezależna
  walidacja potwierdzająca.
- U' ma więcej parametrów niż T; porównanie dotyczy MAC, nie rozmiaru.
- MAC to nie latency; E036 nie rozstrzyga konkretnej przyczyny różnic
  opóźnienia batch1, a E037 nie mierzy czasu U'.
- Cztery seedy, dwie pokrewne architektury, jeden przepis treningu.
- Brak testu istotności; progi i liczby dodatnich komórek są opisowe.

## Odtworzenie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B experiments/2026-11-08_E037_kontrola_mac/figures/generuj_wykresy.py
```

Generator jest deterministyczny (stała kolejność, stałe offsety, brak
losowości, metadane PDF bez daty). Skrypt odmawia zapisu poza `figures/`
i sprawdza hashe wejść przed i po generowaniu.

## Weryfikacja

`weryfikacja.json` potwierdza: 15 komórek, 60 punktów per-seed, 180 wierszy
`analysis.json`, zerową różnicę względem surowych danych, zakresy parametrów
i MAC, `dev_only=true`, `data_open_attempts=0` oraz niezmienność hashy wejść.
