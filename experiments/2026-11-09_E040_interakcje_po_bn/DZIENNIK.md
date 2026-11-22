# E040 — dziennik realizacji

## 2026-11-11 — przygotowanie i start

- Zgoda użytkowniczki na kontynuację badań i realizację pilota; późniejsze
  doprecyzowanie: stany między fazami w RAM, bez nowych plików wag.
- Cztery wolne RTX2080Ti potwierdzone przed startem. Conda attention,
  Python3.11.15, Torch2.12.0+cu130; OMP/MKL/OpenBLAS2, LOKY_MAX_CPU_COUNT2,
  FP32, TF32off, CUBLAS_WORKSPACE_CONFIG=:4096:8.
- CPU wykryło nieścisłość kolejności grup w notatce. Adapter mapuje po nazwach
  między semantyczną kolejnością E040 i upstream. Uzupełnienie wykonawcze
  zapisano **przed pomiarami jakości**. Nie zmieniono menu, limitów i bramek.
- `check_cpu.json`: PASS, 10kontroli operator×profil, w tym4FOLD i6MAG2;
  syntetyczne wejścia, własne wagi, bez obrazów i rekalibracji. MAG2 bitowo
  zgodny z E038, rzeczywiste MAC hooków zgodne z formułą, źródła niezmienione.
- `freeze` i `check`: PASS. SHA256 frozen.json:
  `996d624cd87e65b05084f93c903c767da4d6de79ec15ae6a6131f957b44abbba`.
  Hashe train i splitów zgodne z E035, checkpointy z E011/E015.
- Run: `runs/E040/20261111T032100Z_07ac54f7`, start03:21:00UTC.
  Supervisor PID1173669; GPU0/1/2/3 → seed42/43/44/45,
  worker PID1173695/1173696/1173697/1173698.
- Cztery procesy zachowują wybrane stany w RAM przez globalne bariery B/C.
  Dopiero pozytywna C na obu zbiorach może odblokować dev. Limit7200s.
  Brak automatycznego restartu i poszerzania próby.

## 2026-11-11 — zakończenie i audyt

- Koniec04:11:33UTC;3032,72s,4×exit0. Wszystkie79sond obu operatorów na obu
  zbiorach i4seedach ukończone. Globalna bariera B PASS przed odczytem C.
- C: iloraz MAE par/add1,764983(C10) i2,331635(C100), poprawa0/4 na każdym.
  Supervisor zapisał `C_GATE.json` z `passed=false` i prawidłowo zakończył
  próbę jako `stopped_C`. `DEV_GATE.json` nie powstał; dev nie oceniano.
- Audyt CPU zapisanych predykcji, wyborów i kosztów: **PASS**.
  `analysis/audit.json`, `analysis/AUDYT.md` i trzy CSV są w runie.
- Łącznie1558konstrukcji,1550CMA, bez wyjścia poza limit na checkpoint.
  Telemetria wykazuje niezerową aktywność każdego z4GPU; po runie GPU wolne.
- Stany terminalne pozostawały wyłącznie w RAM workerów i zostały zwolnione
  z ich normalnym zakończeniem. Nie zapisano nowych wag. Około79MiB w runie
  to kod, metryki, predykcje, diagnostyki i logi.
- Zamknięcie kierunku według prerejestracji: bez nowego wariantu i oceny
  niezależnej. [WYNIKI.md](WYNIKI.md); następny etap to artykuł E035–E039.
