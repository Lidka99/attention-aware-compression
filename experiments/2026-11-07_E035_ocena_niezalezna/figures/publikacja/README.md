# E035 — ilustracje publikacyjne (CPU)

Odtwarzalne ilustracje z metadanych E035. **Nie czytają obrazów, checkpointów
ani wag**; korzystają wyłącznie z `frozen.json` i metryk `summary.json` /
`controls.json` runu `20261107T183831Z_8b0b14e0`. Faza `test` oznacza niezależną ocenę:
oficjalny test CIFAR-10/100 oraz **oficjalny val** dla ResNet/Tiny.

## Pliki

| Plik | Co pokazuje |
|---|---|
| `fig1_szerokosci_grup.png/.pdf` | Faktyczne `k/n` kanałów w 12 (ResNet18) lub 13 (PreActResNet18) grupach, U vs T, 5 par × 30/50/70%. |
| `fig2_teacher_cma_top1.png/.pdf` | Top-1 CMA U i T oraz teacher CMA po seedach, 5 par × 3 budżety. |
| `fig3_teacher_cma_utrata.png/.pdf` | Utrata top-1 względem teacher CMA po seedach (niżej = lepiej). |
| `szerokosci_grup.csv` | `k`, `n`, `k/n`, nazwy grup, parametry U/T dla każdej grupy i komórki. |
| `cma_teacher.csv` | Metryki CMA per seed: U, T, teacher, różnica T−U, strata vs teacher, NLL. |
| `komorki.csv` | Podsumowanie 15 komórek: średnie, SD, 4/4, kanały, parametry. |
| `wspoldzielone_plany.csv` | Dokładnie równe wektory szerokości między parami. |
| `weryfikacja.json` | Kompletność, zgodność z `analysis.json`/`WYNIKI.md`, faza i teacher. |
| `sources.json` | SHA256 i rozmiary wszystkich wejść i wyjść. |

## Nazwy grup i budżet

Nazwy grup to pierwsza oś permutacji upstream (`group_first_axis` w CSV),
a `group_permutation` to nazwa osi. Grupa `relu_conv` jest współdzielona przez
wyjście stemu i bloki `layer1`. **30/50/70% to nominalna redukcja liczby
parametrów, nie odsetek kanałów.** Dlatego wykres pokazuje `k/n` osobno, a
`komorki.csv` podaje także `Σk/Σn` oraz faktyczne `params_U`/`params_T`
(T/U ≈ 0,999). Nie należy zastępować tym wykresem tabeli parametrów.

## Plany współdzielone (dokładna równość wektorów)

| Budżet | Profil | Pary o identycznych szerokościach | Grupy |
|---|---|---|---:|
| b30 | U | resnet_cifar10, resnet_cifar100 | 12 |
| b30 | U | preact_cifar10, preact_cifar100 | 13 |
| b30 | T | resnet_cifar10, resnet_cifar100 | 12 |
| b30 | T | preact_cifar10, preact_cifar100 | 13 |
| b50 | U | resnet_cifar100, resnet_tiny | 12 |
| b70 | U | resnet_cifar100, resnet_tiny | 12 |

Poza tymi przypadkami wektory różnią się. W obrębie każdej pary U i T różnią
się przy każdym budżecie — T realokuje kanały. ResNet i PreAct mają różną
liczbę grup (12 vs 13), więc nie dzielą planu między architekturami.

## Co sprawdzono

- 15 komórek = 5 par × 3 budżety, po 4 seedy; 60 plików `summary.json` fazy
  `test`, 20 plików kontroli teacher.
- Każdy plan w `summary.json` jest równy `E035/frozen.json`; `k` klastrów jest
  równe szerokościom z `frozen.json`; `n` i nazwy grup są spójne.
- Teacher CMA istnieje dla każdej pary/seed; 15/15 średnich T−U dodatnich,
  każda komórka 4/4; bramka ≥2 pp przy 50/70% daje 9/10.
- Metryki odtwarzają `analysis.json` (maks. różnica ≤ 1e-9) i średnie z
  `WYNIKI.md` (tolerancja 5e-4 pp); rezerwa CIFAR-100 nieużyta.

## Uruchomienie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B experiments/2026-11-07_E035_ocena_niezalezna/figures/publikacja/generuj_publikacyjne.py
```

Skrypt jest deterministyczny (stała kolejność, brak losowości, metadane PDF
bez daty), zapisuje wyłącznie do tego katalogu i sprawdza hashe wejść.

## Hashe kluczowych wejść

| Wejście | SHA256 |
|---|---|
| `experiments/.../frozen.json` | `0069e0126d472212f5cd18352301ef0319cc8875f45a7c50709b4bdab2321b78` |
| `runs/.../analysis/analysis.json` | `36709e058e9313df4237f82a35e898aa9514a726714aac0fa21b9564d2c0243b` |
| `runs/.../manifest.json` | `543f2ab7b3904399a7b7f5a53c46eb8e2ee46bbe36477b036b88976e45c21de5` |
| `runs/.../test_data.json` | `cafe6f31f80ae2d17d9387c589911b9b05885b0882db8fb89f079dd80afab075` |
| `runs/.../analysis/per_seed.csv` | `a40f28cfa186ce611846d25dc880ce0016908a98a199172bfe42e40cc8afff47` |

Pełna lista hashy 85 plików wejściowych (5 bazowych + 60 `summary.json` + 20 `controls.json`) oraz wyjść: `sources.json`.

## Ograniczenia

To opisowa ilustracja, nie test istotności; cztery seedy to powtórzenia
treningu. Nie dodajemy wniosku o mechanizmie, uniwersalności ani przewadze
compute-matched. Test/val E035 zostały już użyte i nie są „niewidzianą
walidacją” przyszłego strojenia.
