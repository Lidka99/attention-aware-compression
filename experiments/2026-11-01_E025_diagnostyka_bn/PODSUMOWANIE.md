# E025 — źródło spadku po BN zidentyfikowane

176 nowych ocen +16 reuse,4GPU/4seedy, exit0,2,84min. Stałe modele
teacher/UNIFORM/TILT/SEARCH, bez treningu i nowego search.

- Crop/flip przy4096 obrazach i CMA odzyskuje **+5,735pp** teacher
  (69,405→75,140%, wyjściowo75,235%). Sam wzrost liczby próbek daje+0,135pp.
- EMA0.1 po resecie i32batchach załamuje modele do1%; pozostałe3,43%
  inicjalizacji wariancji dominuje w warstwach o małej wariancji.
  Audyt1280 statystyk potwierdza przewidywany efekt (max błąd4,26e−8).
- Z poprawioną BN stary SEARCH nadal słabszy od TILT o0,835pp, mimo
  lepszych KL/NLL. E024 nie zmienia werdyktu.

Rekomendacja do następnego protokołu: A4096+aug+CMA, ponownie zbadać
search z niezmienionym celem/limitem. Szczegóły: [WYNIKI.md](WYNIKI.md).
