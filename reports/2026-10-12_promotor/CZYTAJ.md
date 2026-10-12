# Prezentacja dla promotora — 2026-10-12

Podsumowanie badań E001–E008 w formie prezentacji.

## Pliki

- [prezentacja.html](prezentacja.html) — slajdy (do otwarcia w przeglądarce; Ctrl+P → PDF,
  orientacja pozioma; styl wydruku już ustawiony).
- [prezentacja.md](prezentacja.md) — ta sama treść w Markdown (podgląd/edycja).
- [generuj_wykresy.py](generuj_wykresy.py) — odtwarza wszystkie wykresy z wyników biegów
  (`runs/E001, E005, E007, E008`); tylko odczyt, CPU, bez GPU.
- `wykresy/` — 7 wygenerowanych rysunków PNG.

## Uwagi

- Bieg E008 (etap 2) trwa; slajd „E008 etap 2" pokazuje stan częściowy
  (BASE, epoka 10/60 na 04:58 UTC). Po zakończeniu biegu wystarczy ponownie uruchomić
  generator i ewentualnie dopisać wyniki etapu 2:

```bash
cd reports/2026-10-12_promotor && conda run --no-capture-output -n attention python generuj_wykresy.py
```

- Liczby źródłowe: raporty `WYNIKI.md` w folderach E001–E007 oraz pliki biegu E008:
  `runs/E008/<run>/stage1/stage1_metrics.json`, `runs/E008/<run>/stage2/metrics_*.jsonl`.
- Prezentacja powstała w trakcie treningu E008; praca nad nią nie używała GPU.
