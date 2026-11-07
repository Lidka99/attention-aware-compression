# E036 — dziennik wykonania

- Freeze źródeł PASS, bez otwierania data/.
- Pierwszy CPU check35 modeli przerwany przez domyślny timeout narzędzia120s,
  bez uruchomienia GPU/pomiarów i bez wyniku check.json. Proces zakończony.
  Dodano log postępu i zwiększono timeout samego CPU check do10min.
- CPU check35 modeli PASS po ponownym uruchomieniu; freeze PASS.
- Run `20261107T225229Z_1f5b7f3a` failed: dwa workery zatrzymały się na
  `memory_allocated()==0` po pierwszym modelu. Diagnoza: cuDNN zostawia
  trwały blok alokatora8,5MB; brak tensorów CUDA po `model.cpu()`.
  Poprawiono księgowanie (baseline po warmupie, tolerancja32MB na trwały
  workspace). Run zachowany, bez wyników do agregacji; brak ponownego treningu.
