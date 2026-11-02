# E015 — audyt rewizji `launch_folding.py`

Data: 2026-10-27 (UTC). Zakres: wyłącznie infrastruktura launchera foldingu.
Nie zmieniano kodu treningu, `folding_worker.py`, analizy ani configów.
Nie uruchamiano treningu ani foldingu.

## Przyczyna

Bieg `runs/E015/fold_20261027T204519Z_cb9ffaf_4ccfe035` zakończył się
`NameError("name 'os' is not defined")` w `environment_for`. Ten sam brak
dotyczył `terminate` (`os.killpg`, `os.getpgid`).

## Powiązanie starego i nowego źródła

- Stara wersja (snapshot treningu i nieudanego foldingu):
  `sha256 = 4654bf7bb44cbe6a7894a22756f84287a2c4ca689cd450047f430608a31957e4`
- Nowa wersja (naprawiony launcher):
  `sha256 = b4076943bd0e1e9e234a2ea302c15a50149486010cdb3d02261530a67c7280a9`
- Snapshot starej wersji:
  `runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12/source/launch_folding.py`
  ma dokładnie stary hash; ten sam hash jest w manifeście treningu.

Zmiana jest wyłącznie infrastrukturalna: dodano `import os`, jawną stałą
`LAUNCHER_REVISION` ze starym hashem oraz audytowalne pole
`source_revisions` w nowym manifeście foldingu. `verify_source_consistency`
nadal wymaga identyczności wszystkich pozostałych źródeł i configów;
wyjątek dotyczy tylko `launch_folding.py` i tylko wtedy, gdy stary hash
odpowiada powyższej znanej wersji, snapshot treningu się zgadza, a ten plik
zawiera oba hashe. Nie ma ogólnego mechanizmu pomijania kontroli.

## Kontrole CPU

Wykonano na CPU bez GPU i bez foldingu: składnię, wywołanie
`environment_for` dla różnych seedów/GPU, `terminate` dla procesu żywego,
zakończonego i po `ProcessLookupError`, oraz kontrolę niezdefiniowanych nazw.
Wyniki: `E015_LAUNCHER_CPU_CHECKS=passed`, `E015_CPU_CHECKS=passed`.
