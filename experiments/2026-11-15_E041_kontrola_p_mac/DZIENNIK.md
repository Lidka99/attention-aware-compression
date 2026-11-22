# E041 — dziennik wykonania

## 2026-11-15 — przygotowanie etapu CPU

- Protokół i skrypt geometrii powstały przed nową oceną jakości.
- Pierwszy run `runs/E041/cpu_20261115T232140Z_9b168ae5` zakończył się
  błędem technicznym przed solverem (0,068s): metadane E039 używają
  aliasów `stem`, `*_hidden`, `stage*_out`, a asercja oczekiwała nazw
  upstream `relu_conv`, `*.relu1`, `*.relu2`.
- Poprawiono wyłącznie sprawdzanie jawnej listy aliasów. Kolejność,
  liczby szerokości, targety, progi dopasowania i odrębności nie zmieniły
  się. Pierwszy run i jego kod/protokół/manifest zachowano.
- Nie otwierano wag ani danych i nie używano GPU. Ten błąd nie jest
  negatywnym wynikiem bramki geometrii.

## 2026-11-15 — dodatnia geometria i kontrola operatora

- `runs/E041/cpu_20261115T232256Z_44602061`: completed, bramkaA PASS,
  8/8dopasowanych i różnych kontroli, 0,510s, bez wag/danych/GPU.
- Zamrożono geometryczne wejścia w `frozen.json`; źródła jakości służą
  wyłącznie przyszłemu odtworzeniu referencji, nie doborowi profili.
- Pierwszą kontrolę rzeczywistego operatora
  `runs/E041/check_20261115T234533Z_d7453f96` przerwał limit narzędzia120s
  po7potwierdzonych profilach. PID1352536 nie działa (sprawdzone `ps -p`).
  To przerwanie techniczne, nie wynik jakości ani ujemna bramka kosztów.
- Dodano bieżący zapis manifestu po każdym profilu i limit CPU300s;
  ponowna kontrola ma osobny run. Szerokości i kryteria pozostały bez zmian.
- Powtórna kontrola `runs/E041/check_20261115T234914Z_ab01e2db`: PASS,
  8/8profili,93,546s części pomiarowej. Rzeczywiste P/MAC zgodne ze wzorem,
  niepuste klastry, osie i bias poprawne. Bez oceny jakości i użycia GPU.
- Dwa przeglądy tylko do odczytu nie wskazały blokera geometrii ani
  runnera; gałąź GPU nie była w ramach tych przeglądów wykonywana.

## 2026-11-16 — ocena jakości po potwierdzeniu zasobów

- Po osobnym potwierdzeniu użytkowniczki odczytano `nvidia-smi` i `df`:
  4×RTX2080Ti wolne,1MiB/GPU,0%util, około4,5GiB wolnego dysku.
- Run `runs/E041/20261116T040008Z_9e34f1e9`, launcherPID1401803,
  workers1401879–1401882: completed,4×exit0,227,054s.
- Wspólna bariera odtworzenia teacher/U/T przeszła dla wszystkich seedów
  przed nową oceną. Telemetria potwierdza pracę4GPU i ich zwolnienie.
- Primary niespełnione łącznie: C10+0,11250pp,2/4; C100+2,10375pp,4/4.
  Nie zmieniano progów, profili ani kodu pomiarowego po starcie.

## 2026-11-19 — raport i dodatkowe przeliczenie

- `analyze` zakończone audytem PASS. Skrypt `analizy/tabele.py`, dodany
  po pomiarze i oddzielony od kodu wykonawczego, przeliczył30kontrastów
  z96wierszy CSV: PASS. Tabele i wykresy mają źródłowe SHA256.
- Tiny nie był częścią zamrożonego zakresu. Nie interpretujemy braku
  pomiaru jako wyniku negatywnego; nie włączamy go post hoc do bramki.
