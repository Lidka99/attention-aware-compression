# E035 — dziennik wykonania

- 2026-11-07 UTC: protokół zamrożony w f42c0b8 przed otwarciem testów.
  Sprawdzono cztery wolne RTX2080Ti (1MiB,0%).
- Pierwszy CPU check: FileNotFoundError dla E029/indices.json; E029 używa
  bezpośrednio indeksów E028/E024 i nie ma lokalnej kopii. Poprawiono ścieżki
  provenance i weryfikacji zgodności. Bez uruchomienia GPU i otwierania testu.
- CPU check po poprawce PASS10/10; implementacja5a7ec92. Run
  `20261107T183831Z_8b0b14e0`, launcher960106, cztery workery na etap.
- Dev490,39s,4exit0,360+60; gate top1max0 iNLLmax1,1921e-7, PASS.
- Po bramce odnaleziono lokalne archiwum CIFAR10, MD5 zgodny; rozpakowano
  wyłącznie test_batch. Bez pobierania i nadpisania train.
- Test515,72s,4exit0,360+60; pełny run1018,04s. Brak failed GPU runów.
- Analiza: primary15/15 (4/4 każda), próg2pp9/10, repair kierunek14/15.
  Audyt stanów modeli dev/test i pełnej siatki PASS. Dodatkowy audyt
  pochodzenia Tiny10000/10000 obrazy+etykiety zgodne z oficjalnym val.
- Po ocenie dodano wyłącznie audyt pochodzenia i formatter tabel;
  oryginalny check/kod oceny zachowane, bez zmiany progów/metryk.
