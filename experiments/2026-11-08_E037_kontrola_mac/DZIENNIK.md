# E037 — dziennik wykonania

- Freeze15 profili U' PASS; wszystkie MAC(U')≤MAC(T), parametry U'>T.
- CPU check15 profili PASS.
- Run `20261108T030702Z_a7428da3` failed zaraz po starcie: monitor wywołał
  funkcję sprawdzającą bezczynność GPU, a workery już używały kart
  (600MiB/31–47%). Nie było to zadanie naukowe ani błąd modelu; procesy
  zatrzymane, GPU wróciły do1MiB/0%. Run zachowany bez wyników.
  Poprawka: monitor używa telemetrii bez asercji bezczynności; kontrola
  bezczynności pozostaje tylko przed startem. Bez zmian profili/kryteriów.
