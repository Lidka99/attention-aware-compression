# E037 — dziennik wykonania

- Freeze15 profili U' PASS; wszystkie MAC(U')≤MAC(T), parametry U'>T.
- CPU check15 profili PASS.
- Run `20261108T030702Z_a7428da3` failed zaraz po starcie: monitor wywołał
  funkcję sprawdzającą bezczynność GPU, a workery już używały kart
  (600MiB/31–47%). Nie było to zadanie naukowe ani błąd modelu; procesy
  zatrzymane, GPU wróciły do1MiB/0%. Run zachowany bez wyników.
  Poprawka: monitor używa telemetrii bez asercji bezczynności; kontrola
  bezczynności pozostaje tylko przed startem. Bez zmian profili/kryteriów.
- Finalny run `20261108T031759Z_855c0b33` completed,4×exit0,292,28s;
  180 ocen+60kontroli, audyt PASS. Primary negatywny. Po runie dodano
  agregaty `averages` i poprawkę nazwy klucza czasu `elapsed_seconds`
  w analyze; kod oceny i metryki bez zmian.

- Audyt zapisów i wykresów:135 kontrastów przeliczonych z raw E035 dev
  i E037 zgodnych. Historyczny failed run ma nadal manifest/statusy
  `running`, bo błąd obejmował sekcję końcową; PID-y są nieaktywne.
  Surowych plików nie nadpisano. Skorygowano nadmiernie przyczynową
  interpretację: U' zmienia parametry i MAC, więc nie izoluje wpływu
  compute od pojemności. Dodano figures/ z punktami i hashami źródeł.
