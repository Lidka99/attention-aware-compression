# E027 — czy winny był cel, czy wyszukiwanie?

**Completed, primary niespełnione o 0,065 pp.** 200 ocen,4GPU,4 seedy,19,3min.
Te same ustawienia co E026; zmiana: wybór po **disagreement** (argmax≠teacher)
zamiast KL. Trzy miary liczone z jednego forwardu na B.

NEW_SEARCH **56,485%** vs TILT55,550% → **+0,935pp, 4/4** (próg+1,0pp).
NEW−UNIFORM+2,630pp (4/4), NEW−stary SEARCH+1,770pp (4/4).
Zgodność miar z dev top1: disagreement0,634 > KL0,546 > teacher_ce0,544,
na każdym seedzie. Najlepszy z50 kandydatów bije TILT o1,66pp (4/4),
więc przestrzeń nie jest wąskim gardłem.

Wniosek: KL był złą surrogate; kierunek alokacji nadal żywy. Kryterium
niespełnione, więc wynik wymaga **przeniesienia na CIFAR-10** z tą samą
zamrożoną regułą, bez strojenia na użytym dev. [WYNIKI.md](WYNIKI.md).
