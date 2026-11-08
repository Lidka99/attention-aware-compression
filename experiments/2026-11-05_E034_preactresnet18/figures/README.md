# E034 — wykresy CMA TILT − UNIFORM (dev)

Zestaw wygenerowany wyłącznie na CPU (matplotlib, Conda `attention`).
Wszystkie liczby pochodzą z surowych `summary.json` runu
`runs/E034/fold_20261106T003625Z_9ec4f2ea`; `analysis.json` służy do kontroli
zgodności. Test i rezerwa nie były używane.

## Pliki

| Plik | Co pokazuje |
|---|---|
| `e034_cma_t_minus_u.png/.pdf` | Sparowana różnica TILT − UNIFORM po CMA dla CIFAR-10 i CIFAR-100 przy 30/50/70% redukcji. Punkty = 4 seedy, czarny romb = średnia, słupki = SD. |
| `e034_cma_top1.png/.pdf` | Kontekst: bezwzględny top-1 CMA dla U i T (średnia ± SD z 4 seedów). |
| `punkty.csv` | Wszystkie punkty użyte na wykresach (per seed oraz średnie) w formacie długim. |
| `sources.json` | SHA256 i rozmiary plików wejściowych oraz skryptu; potwierdzenie niezmienności wejść. |
| `weryfikacja.json` | Liczebności, średnie, SD i maksymalne różnice względem `analysis.json`. |

## Dane i liczenie

Dla każdego zbioru, budżetu i seeda:

```text
delta_pp = (summary.metrics.UP_T.CMA.top1 − summary.metrics.UP_U.CMA.top1) × 100
```

Średnia i SD są liczone po czterech seedach (42–45). **Podane słupki to SD
(ddof=1) sparowanych różnic, nie CI.** Na wykresie różnic jest linia zera;
zero oznacza brak różnicy. Nie dodawano gwiazdek, progów istotności ani
sugestii istotności statystycznej.

Kontrola: surowe `summary.json` odtwarzają wpisy `primary` (CMA) z
`analysis.json`; maksymalna różnica średniej i SD jest na poziomie błędu
maszynowego (szczegóły w `weryfikacja.json`).

## Odczyty

- We wszystkich 6 komórkach średnia T − U jest dodatnia, a dodatnie są 4/4
  seedy; różnica rośnie wraz z redukcją parametrów.
- Największa średnia różnica jest na CIFAR-100 przy 70% redukcji.
- To wynik opisowy na dev, z n = 4 seedami; nie jest testem istotności.

## Ograniczenia

- Tylko dev; test i rezerwa nietknięte.
- Cztery seedy to cztery powtórzenia treningu, nie niezależne replikacje.
- Wykresy nie rozstrzygają o mechanizmie ani o przyczynowej roli preaktywacji.
- Nie należy opisywać różnic jako „istotnych” bez formalnego testu.

## Odtworzenie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B experiments/2026-11-05_E034_preactresnet18/figures/generuj_wykresy.py
```

Generator jest deterministyczny: stała kolejność seedów, stałe offsety punktów,
brak losowości, metadane PDF bez daty. Ponowne uruchomienie daje te same
pliki. Skrypt odmawia zapisu poza `figures/` i sprawdza, że hashe wejść nie
zmieniły się po wygenerowaniu wykresów.

## Weryfikacja

`weryfikacja.json` potwierdza: 6 komórek, 24 punkty per-seed, 25 plików
wejściowych, zgodność z `analysis.json` oraz niezmienność hashy wejść.
