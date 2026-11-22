# E042 — podsumowanie

- **Pytanie:** czy któryś z 24 z góry określonych profili poprawia top‑1
  przy jednoczesnych pułapach liczby parametrów i MAC z UNIFORM/b50?
- **Wynik:** główny CIFAR-100 **+1,175 pp vs U (3/4)**, próg +1 pp
  spełniony. Kontrolny CIFAR-10 **+1,475 pp (4/4)**. Na obu wygrywa L06,
  ten sam wektor propozycji przed projekcją. T-proj nie poprawia U.
- **Granica:** ujemna mediana i 5/24 dodatnich średnich na CIFAR-100.
  Najlepszy profil wybrany na tym samym dev; to zapas zbadanej puli,
  nie zweryfikowany selektor. Model osiąga 55,03% vs 75,14% przed kompresją.
- **Wykonanie:** `runs/E042/20261121T212257Z_207ae03f`, 216 ocen, 4 GPU,
  723,49 s, audyt PASS; bez treningu, testu i rezerwy.
- **Dalej:** test doboru lub transferu z góry zamrożonego profilu w osobnym
  protokole, z nową oceną potwierdzającą. [Pełny wynik](WYNIKI.md).
