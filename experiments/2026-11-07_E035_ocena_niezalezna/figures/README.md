# E035 — wykresy niezależnej oceny CMA

Zestaw wygenerowany wyłącznie na CPU (matplotlib, Conda `attention`).
Dane pochodzą z surowych `summary.json` runu
`runs/E035/20261107T183831Z_8b0b14e0` (fazy dev i test); `analysis.json`
służy do kontroli zgodności.

## Faza niezależna i etykieta Tiny

Faza `test` w analizie E035 oznacza niezależną ocenę: dla CIFAR-10 i CIFAR-100
oficjalny test, a dla **ResNet/Tiny oficjalny val** (`cache_val`). Dlatego na
wykresach i w CSV panel Tiny jest opisany jako **„oficjalny val”**, a nie
„test”. Nie jest to twierdzenie, że val był wcześniej nigdy nieotwierany;
chodzi wyłącznie o poprawną nazwę zbioru.

## Pliki

| Plik | Co pokazuje |
|---|---|
| `e035_niezalezny_top1.png/.pdf` | Top-1 CMA dla UNIFORM (U) i TILT (T) w fazie niezależnej, pięć par × 30/50/70%. Średnia ± SD z 4 seedów. |
| `e035_sparowane_delta.png/.pdf` | Sparowana różnica TILT − UNIFORM po CMA w fazie niezależnej. Punkty = 4 seedy, czarny romb = średnia, słupki = SD; linia zera. |
| `e035_dev_vs_niezalezny.png/.pdf` | Kontrola spójności: średnia T − U na dev vs średnia na zbiorze niezależnym dla 15 komórek. |
| `punkty.csv` | Wszystkie punkty użyte na wykresach (per seed oraz średnie). |
| `sources.json` | SHA256 i rozmiary plików wejściowych oraz skryptu; potwierdzenie niezmienności wejść. |
| `weryfikacja.json` | Liczebności, średnie, SD, liczby dodatnich seedów i maksymalne różnice względem `analysis.json`. |

## Dane i liczenie

Dla każdej pary, budżetu i seeda:

```text
delta_pp = (metrics.T.CMA.top1 − metrics.U.CMA.top1) × 100
```

Średnia i SD są liczone po czterech seedach (42–45). **Słupki to SD
(ddof=1), nie CI.** Na wykresie różnic jest linia zera; zero = brak różnicy.
Nie dodawano oznaczeń istotności ani progów istotności.

Kontrola: surowe `summary.json` odtwarzają `analysis.json` (`rows` i
`contrasts` dla CMA) w granicach błędu maszynowego. `test_data.json`
potwierdza użycie `data/tiny-imagenet/cache_val.npz` dla Tiny.

## Odczyty

- W fazie niezależnej wszystkie 15 średnich T − U jest dodatnich, a w każdej
  komórce dodatnie są 4/4 seedy.
- Efekt jest mały dla ResNet/CIFAR-100 przy 50% i ma duży rozrzut przy 70%;
  nie należy opisywać go jako jednolitego, silnego efektu.
- Wykres dev vs niezależny służy tylko do kontroli spójności; nie jest testem
  istotności.

## Ograniczenia

- To ocena opisowa; cztery seedy to cztery powtórzenia treningu, a nie
  liczba obrazów testowych lub niezależnych stanowisk sprzętowych.
- Test/val zostały użyte w E035; po tej ocenie nie są „niewidzianą walidacją”
  dla przyszłego strojenia.
- Brak wniosku o mechanizmie, uniwersalności lub optymalności alokacji.
- Nie należy traktować SD jako CI ani dodawać pozornej istotności.

## Odtworzenie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B experiments/2026-11-07_E035_ocena_niezalezna/figures/generuj_wykresy.py
```

Generator jest deterministyczny (stała kolejność, stałe offsety, brak
losowości, metadane PDF bez daty). Skrypt odmawia zapisu poza `figures/`
i sprawdza hashe wejść przed i po generowaniu.

## Weryfikacja

`weryfikacja.json` potwierdza: 15 komórek, 120 pomiarów U/T po CMA
(2 profile×60 punktów seeda) i60 sparowanych różnic w fazie niezależnej,
30 kontrastów CMA (dev + niezależna), 15/15 dodatnich średnich
w fazie niezależnej, użycie oficjalnego val Tiny oraz niezmienność hashy wejść.
