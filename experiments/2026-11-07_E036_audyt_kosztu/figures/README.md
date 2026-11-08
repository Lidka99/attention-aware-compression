# E036 — wykresy kosztu przy dopasowanych parametrach

Zestaw wygenerowany wyłącznie na CPU (matplotlib, Conda `attention`).
Wszystkie ilorazy policzono wprost z surowych plików czasów
`*_bs{1,128}_s{0,1,2}.json` oraz `costs.json` runu
`runs/E036/20261107T231027Z_27bac186`. `analysis.json` służy tylko do kontroli
zgodności ilorazów.

## Pliki

| Plik | Co pokazuje |
|---|---|
| `e036_mac_i_przepustowosc.png/.pdf` | MAC TILT/UNIFORM oraz przepustowość batch128 TILT/UNIFORM przy 30/50/70% redukcji, pięć par. |
| `e036_batch1_ratio.png/.pdf` | Batch1: per-seed iloraz T median / U median, średnia i **prawdziwa SD z czterech seedów**. |
| `punkty.csv` | Wszystkie punkty użyte na wykresach (per seed oraz średnie). |
| `sources.json` | SHA256 i rozmiary plików wejściowych oraz skryptu; potwierdzenie niezmienności wejść. |
| `weryfikacja.json` | Liczebności, zakresy, średnie batch1, SD i maksymalna różnica względem `analysis.json`. |

## Dane i liczenie

Dla każdego seeda, pary i budżetu:

```text
MAC T/U                 = costs[tag_T].mac / costs[tag_U].mac
przepustowość b128 T/U  = (128 / mean(T_bs128)) / (128 / mean(U_bs128))
                        = mean(U_bs128) / mean(T_bs128)
latencja b1 T/U         = median(T_bs1) / median(U_bs1)
```

Każdy plik serii ma 200 pomiarów; na model i batch łączymy 3 serie = 600
pomiarów. Średnia i SD ilorazów są liczone po czterech seedach (42–45).

**Batch1:** SD jest liczone z czterech per-seed ilorazów
`T median / U median` (ddof=1). **Nie używamy SD z ręcznych `TABLES.md`** —
te wartości były błędne (np. pokazywały ±0,001 przy rzeczywistym SD dochodzącym
do 0,100). Słupki to SD, nie CI. Linia odniesienia = 1; małe różnice średnich
nie dowodzą równoważności ani dominacji narzutu uruchomienia. Tabela została
skorygowana generatorem `../tables_from_raw.py`.

## Odczyty

- MAC T/U rośnie od ok. 1,105 do 1,823 przy 30→70%; TILT zużywa więcej MAC
  przy tym samym budżecie parametrów.
- Przepustowość batch128 T/U (średnie) mieści się w zakresie ok. 0,702–0,948;
  TILT jest wolniejszy, szczególnie na Tiny przy 50/70%.
- Batch1: średnie ok. 0,965–1,046, ale SD per-seed 0,019–0,100;
  brak podstaw do wniosku o równoważności lub konkretnej przyczynie różnic.
- E036 nie mierzy jakości; nie ma tu nowej oceny modeli.

## Ograniczenia

- Jeden serwer i cztery przypisania seed/GPU; nie jest to replikacja sprzętowa.
- Steady-state device-forward bez loadera i transferów; nie izolowano
  narzutu uruchomienia. Syntetyczne wejście `randn`.
- MAC to nie latency; nie należy utożsamiać tych miar.
- Brak wniosku o przyczynie zysku jakościowego; E036 jest audytem kosztu.

## Odtworzenie

```bash
source /home/users/s224574/miniconda3/etc/profile.d/conda.sh
conda activate attention
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B experiments/2026-11-07_E036_audyt_kosztu/figures/generuj_wykresy.py
```

Generator jest deterministyczny (stała kolejność, stałe offsety, brak
losowości, metadane PDF bez daty). Skrypt odmawia zapisu poza `figures/`
i sprawdza hashe wejść przed i po generowaniu.

## Weryfikacja

`weryfikacja.json` potwierdza: 15 komórek, 720 plików czasów, 20 plików
`costs.json`, 60 punktów per-seed na metrykę, zerową różnicę względem
`analysis.json`, zakresy MAC/przepustowości oraz prawdziwe SD batch1.
