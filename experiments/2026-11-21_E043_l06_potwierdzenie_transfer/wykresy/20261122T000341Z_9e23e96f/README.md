# E043 — figura rezerwy CIFAR-100: top-1 po CMA i kontrast L06−U

Odtwarzalna figura publikacyjna do zakończonego runu
`runs/E043/20261122T000341Z_9e23e96f` (audyt PASS, 28/28 ocen, 4 GPU/exit 0).
Powstaje wyłącznie z `analysis/per_seed.csv`, `analysis/summary.json`
i `analysis/audit.json` — bez treningu, danych, wag, inferencji i GPU.
Protokół: [`../../OPIS.md`](../../OPIS.md).

## Pliki

| Plik | Zawartość |
|---|---|
| `e043_rezerwa.png` | figura 2600×1520 px (200 dpi) |
| `e043_rezerwa.pdf` | ta sama figura, wektorowa |
| `make_figure.py` | deterministyczny generator (tylko CSV/JSON z `analysis/`) |

## Co jest na figurze

- **Panel A — bezwzględny top-1 po CMA [%]:** U, T-proj i L06 przy b50
  (~50% redukcji P) i b30 (~30%), oraz model nieskompresowany (nauczyciel)
  poza budżetami. Duże znaczniki to średnie po seedach 42–45, małe kropki to
  pojedyncze seedy.
- **Panel B — sparowany kontrast L06 − U [pp]:** osobne punkty czterech
  seedów na tych samych 5 000 obrazach, ich średnia (romb), linia zera oraz
  **zamrożony próg +1,00 pp** (przerywana). Opisowo zaznaczono, że średnia
  b30 leży 0,055 pp poniżej progu.

Podpis figury nie zawiera twierdzeń o istotności; figura nie ma testów ani
przedziałów ufności.

## Najważniejsze liczby (rezerwa CIFAR-100)

| Budżet | U | T-proj | L06 | L06 − U (4 seedy) | > 0 | Strata L06 do nauczyciela |
|---|---:|---:|---:|---:|---:|---:|
| b50 | 53,255% | 53,405% | **54,470%** | **+1,215 pp** (2,10 / 0,98 / −0,06 / 1,84) | 3/4 | 19,970 pp |
| b30 | 66,075% | 66,070% | **67,020%** | **+0,945 pp** (1,30 / 1,34 / 0,04 / 1,10) | 4/4 | 7,420 pp |

Nauczyciel bez kompresji: **74,440%**. Rzeczywista redukcja P L06: 50,299%
(b50) i 30,075% (b30).

## Źródła i hashe (SHA256)

Wejścia (sprawdzane na sztywno w generatorze):

- `runs/E043/20261122T000341Z_9e23e96f/analysis/per_seed.csv`
  `f64ea831be849ec70bbd8a7fc9081e01045e7bbc6fd91e3785b745b7a90b5e20`
- `.../analysis/summary.json`
  `c130791164727cf9a366941e52af1f9886b21c41f1c4992c98792b9dbf06ca96`
- `.../analysis/audit.json`
  `a2004ce35b783409250b2e00c7e780979383654ad7ffe79dae9e310af5a3c5e2`
  (`status: PASS`, 28 wierszy, 16/16 stanów b50/teacher zgodnych z E042)

Wyjścia:

- `e043_rezerwa.png`
  `05d36632bde6ec75fe600756d2f0fc8a16589d9a5ecd8e92d129c4ca5cc8926e`
- `e043_rezerwa.pdf`
  `0930c0a61c36c6ba5d59a28f1202ba720550314eb0ad8ca685f259910ef2d3c5`

## Weryfikacja

Generator asertuje: `audit.json` ma `PASS`, run i 28 wierszy; 4 seedy
(42–45) na każdy z 7 modeli; zgodność średnich, kontrastów per seed, SD,
liczby dodatnich seedów i sum trafień z `summary.json`; stałą redukcję P
w obrębie budżetu; wartości top-1 równe `correct/5000`. Dwa kolejne
uruchomienia dają identyczne hashe PNG i PDF.

## Odtworzenie

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-11-21_E043_l06_potwierdzenie_transfer/wykresy/20261122T000341Z_9e23e96f/make_figure.py
```

CPU/matplotlib only (Python 3.11.15, matplotlib 3.10.9, DejaVu Sans,
`SOURCE_DATE_EPOCH=1795305600`). Obrazy rezerwy są rozłączne z dev użytym
do wyboru L06, lecz checkpointy są te same co w pilocie. Rezerwa została
użyta raz. Figura jest opisowa, a próg nie jest testem statystycznym.
