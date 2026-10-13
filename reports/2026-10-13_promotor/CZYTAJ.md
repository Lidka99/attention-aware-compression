# Prezentacja podsumowania badań — 2026-10-13

Nowa wersja podsumowania badań E001–E009 (poprzednia, z 2026-10-12, została
w `reports/2026-10-12_promotor/` jako historia). 22 slajdy: „W skrócie" i wyjaśnienie
SE na początku, wyniki E001–E009, kontekst attention_v2 / attentionv3 na końcu.

## Pliki

- [prezentacja.html](prezentacja.html) — **slajdy** (otwórz w przeglądarce;
  Ctrl+P → PDF, orientacja pozioma, styl wydruku już ustawiony).
- [prezentacja.md](prezentacja.md) — ta sama treść w Markdown (podgląd/edycja).
- [notatki_prelegenta.md](notatki_prelegenta.md) — co mówić na slajdach,
  możliwe pytania i gotowe odpowiedzi.
- [generuj_wykresy.py](generuj_wykresy.py) — odtwarza wszystkie 10 rysunków (9 wykresów
  wynikowych + schemat SE) z wyników biegów (`runs/E001, E005, E007, E008, E009`);
  tylko odczyt, CPU, bez GPU.
- [odswiez_e009.py](odswiez_e009.py) — odświeża wyniki i wykresy E009 w prezentacji
  (podmienia bloki między znacznikami `<!-- E009:... -->`).
- `wykresy/` — 10 wygenerowanych rysunków PNG.

## Po zakończeniu E009 (~13.10 ok. 14:30 UTC)

Launcher sam zapisze `summary.json` w katalogu biegu. Wystarczy jedno polecenie:

```bash
cd reports/2026-10-13_promotor && conda run --no-capture-output -n attention python odswiez_e009.py
```

Skrypt sam: przeczyta najnowszy katalog `runs/E009/2*`, wygeneruje na nowo wykresy
E009 i podmieni liczby (status, wyniki, CI, werdykt) w `prezentacja.html` i
`prezentacja.md`. Można go uruchamiać także **w trakcie** biegu — pokaże stan
częściowy. Uruchomiony dwa razy pod rząd nie duplikuje treści.

Pełna regeneracja wszystkich wykresów (np. po zmianach w skryptach):

```bash
cd reports/2026-10-13_promotor && conda run --no-capture-output -n attention python generuj_wykresy.py
```

## Liczby źródłowe

- Raporty `WYNIKI.md` w `experiments/2026-*-E00*/`.
- Biegi: `runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim/` (summary.json),
  `runs/E009/20261012T155859Z_fef2836_seed42_jsagnut2/` (metryki jsonl, summary.json).
- Literatura zweryfikowana u źródła 13.10.2026 (arXiv/PMLR/Sci.Rep.), kotwica
  Shahriar 2026: 100 ep = 60,54% (potwierdzone w arXiv:2607.01984, Table 9).

## Uwagi

- Praca nad prezentacją nie używała GPU i nie dotykała katalogów wyników biegu
  (tylko odczyt).
- Status w prezentacji: E009 w toku — BASE 60,04% i PRE 59,18% po 120 epokach,
  POST w toku, RANDOM w kolejce (stan 13.10.2026 ~07:53 UTC).
- Macierz nowości (`literature/MATRIX.md`) ma status „lista startowa, nie ukończony
  audyt" — prezentacja uczciwie to sygnalizuje; przed publikacją wymaga domknięcia.
