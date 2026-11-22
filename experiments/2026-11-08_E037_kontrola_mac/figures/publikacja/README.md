# E035/E037 — figura publikacyjna jakość–P/MAC (dev, po CMA)

Figura pokazuje top-1 dev po CMA dla trzech profili: **U** (UNIFORM),
**T** (TILT) i **U'** (UNIFORM dopasowany MAC), dla pięciu par
model–zbiór i trzech nominalnych redukcji parametrów 30/50/70%.
Profile U/T pochodzą z dev E035, a U' z dev E037. To zestawienie
istniejących metryk, nie nowa ocena modeli.

## Pliki

| Plik | Co zawiera |
|---|---|
| `e035_e037_jakosc_p_mac.pdf/.png` | Figura: 5 wierszy (pary) × 2 panele (top-1 vs P, top-1 vs MAC). |
| `punkty_publikacja.csv` | Wszystkie punkty per seed i średnie, z surowymi P/MAC, ułamkami nauczyciela i SHA256 źródła. |
| `generuj_figure.py` | Reprodukowalne źródło figury i CSV; tylko CPU, bez GPU/treningu/danych/checkpointów. |
| `sources.json` | Pełna lista 144 wejść z SHA256, rozmiarami i rolami oraz hashe wyjść. |
| `weryfikacja.json` | Kompletność, zgodność z `analysis.json` i `WYNIKI.md`, zakresy, hashe. |

## Jak czytać wykres

- Lewy panel: top-1 vs liczba parametrów **P**. Prawy: top-1 vs **MAC**.
- Oś x to **zmierzony koszt podzielony przez koszt nauczyciela**;
  100% oznacza model nieskompresowany. Surowe P/MAC są w CSV.
- Kolor i styl linii: profil U/T/U'. Kształt punktu: 30/50/70%
  nominalnej redukcji parametrów (nie odsetek kanałów).
- Punkty to średnia z czterech seedów 42–45; słupki to SD (ddof=1),
  nie CI. Gwiazdka to teacher po CMA.
- Osie y mają **różne zakresy dla par**; pokazują zakres wyników,
  ale nie należy porównywać nachyleń między rzędami bez tego zastrzeżenia.
- To trzy konkretne profile przy trzech budżetach, nie optymalizacja
  ani wyznaczanie globalnej granicy osiągalnych rozwiązań.

## Pochodzenie i koszty

- E035: `runs/E035/20261107T183831Z_8b0b14e0`, wyłącznie `dev/`.
- E037: `runs/E037/20261108T031759Z_855c0b33`, wyłącznie `seed_*` dev.
- E036: `runs/E036/20261107T231027Z_27bac186`; kontrolne P/MAC
  nauczyciela i U/T z `experiments/2026-11-07_E036_audyt_kosztu/check.json`.

| Para | P nauczyciela [mln] | MAC nauczyciela [mln] | top-1 dev CMA [%] |
|---|---:|---:|---:|
| ResNet/CIFAR-10 | 11,174 | 555,423 | 77,3–95,4 |
| ResNet/CIFAR-100 | 11,220 | 555,469 | 31,4–75,1 |
| ResNet/Tiny ImageNet | 11,279 | 148,148 | 8,4–51,2 |
| PreAct/CIFAR-10 | 11,172 | 555,423 | 78,6–95,3 |
| PreAct/CIFAR-100 | 11,218 | 555,469 | 23,4–77,0 |

Zakresy rzeczywistych kosztów profili U/T/U': **P 3,33–8,68 mln**,
**MAC 45,4–426,9 mln**. Jako ułamek nauczyciela: **P 29,71–76,92%**,
**MAC 29,14–77,47%**. U' ma 1,10–1,82× więcej parametrów niż T,
ale MAC(U') ≤ MAC(T) w 15/15 komórkach.

P/MAC nauczyciela pochodzą z audytu E036 (`check.json`, status `passed`,
35 modeli, hooki Conv/Linear, 1 MAC = 1 multiply-add, BN/ReLU poza MAC).
U/T P/MAC z planów E037 są zgodne z E036 co do liczb całkowitych;
U' P/MAC pochodzą z zamrożonego `frozen.json` E037.

## Co sprawdzono

- Kompletność: 60 `summary.json` dev E035, 60 `summary.json` E037,
  20 `controls.json` E035; 15 komórek × 4 seedy = 60 wierszy CMA.
- Sparowane seedy 42–45; per-seed T−U' odtwarza `analysis.json`
  (maks. różnica 5×10⁻¹⁵ pp).
- Brak mieszania testu: `dev_only=true`, `data_open_attempts=0`;
  wszystkie 60 summary E037 mają `official_test_used=false`;
  wejścia E035 pochodzą wyłącznie z `dev/`, nie z `test/`.
- Zgodność z `analysis.json`: maks. różnica top-1 0;
  z tabelą `WYNIKI.md` E037: maks. 0,005 pp (zaokrąglenie tabeli do 2 miejsc).
- Hashe: 60/60 `source_hashes` E037 i 80/80 hashy dev E035
  (`summary.json` + `controls.json`) zgodne z `frozen.json` E037.
- Teacher CMA zgodny z `controls.json` E035; teacher jest najlepszy
  w każdej parze; `Uprime_mac ≤ T_mac` i `Uprime_params > T_params`
  w 15/15.

Kluczowe SHA256 wejść (pełna lista 144 plików: `sources.json`):

| Wejście | SHA256 |
|---|---|
| E037 `analysis/analysis.json` | `051b5133737efe4c0e1c0e3a6ed949089bc9e5b51780a436fcbea3d04dfb66de` |
| E037 `analysis/verification.json` | `046e375d20d39bc1a97eb8ecc9fdf4a9a2d058bcab9eb6cddb202c656fbf8ff7` |
| E037 `frozen.json` | `48832cdcc76cb3f4c99243ea707f3066a9bf219a6c855dad6689a2469924dec6` |
| E036 `check.json` | `4ba0282fb1a77f79f91a67d7cbb617e6ef2f07c1d747df1ae32f64fa3e7570b6` |

## Odtworzenie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -B experiments/2026-11-08_E037_kontrola_mac/figures/publikacja/generuj_figure.py
```

Skrypt jest deterministyczny (stała kolejność, brak losowości, metadane PDF
bez daty), zapisuje wyłącznie do tego katalogu i sprawdza hashe wejść
przed i po generowaniu.

## Ograniczenia

- Dev był używany wcześniej; to kontrola mechanizmu, nie niezależna
  walidacja potwierdzająca. Test E035 nie był czytany tutaj, ale został
  użyty poza tym zestawieniem.
- U' ma więcej parametrów niż T; porównanie dotyczy MAC, nie rozmiaru sieci.
- MAC to nie latency; E036 nie rozstrzyga przyczyn różnic opóźnienia.
- Cztery seedy, dwie pokrewne architektury, jeden przepis treningu;
  brak testu istotności, progi i liczby komórek są opisowe.
- Osie y mają różne zakresy dla par; figura nie jest wspólną skalą jakości.

## Podpis do rysunku

Jakość–koszt po CMA na dev (E035/E037): top-1 vs liczba parametrów P
(lewy) i vs MAC (prawy) dla U, T i U' przy 30/50/70% nominalnej redukcji
parametrów; pięć par model–zbiór. Punkty to średnia z 4 seedów
(słupki: SD), a oś x to zmierzony koszt podzielony przez koszt nauczyciela
(100% = model nieskompresowany); surowe P/MAC są w `punkty_publikacja.csv`.
U' to UNIFORM dopasowany MAC (1,10–1,82× więcej parametrów niż T);
to porównanie dev, nie niezależna walidacja, i nie wyznacza globalnej
granicy osiągalnych rozwiązań.
