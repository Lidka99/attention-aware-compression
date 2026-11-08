# E038 — dziennik techniczny

- 2026-11-08, przed oceną dev: pierwszy CPU check przerwany limitem narzędzia
  120s; powtórzony z limitem600s. Żadnych pomiarów jakości ani zadań GPU.
- Kolejny CPU check wykazał, że pomocnicze `ratio` planu PreAct/E035 nie
  realizuje dokładnie zapisanego wektora U (historyczny FOLD używał jawnego
  wektora szerokości). Poprawiono wyłącznie kontrolę zgodności upstream MAG:
  skalarny keep_ratio jest środkiem wspólnego przedziału dającego dokładnie U.
  Profile U/T, checkpointy, kryteria i implementacja MAG2 pozostały te same.
- Po runie dodano niezależną ścieżkę audytu T/U przez oryginalne `apply`
  upstream MAG2, z adapterem tylko liczby k i remisów. Pierwszy audyt
  techniczny źle zaliczał puste `Sequential` do projekcji skrótu; poprawiono
  listę oczekiwanych wywołań audytu. Operator pomiarowy i metryki bez zmian.
