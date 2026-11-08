# E036 — koszt alokacji: TILT nie jest darmowy

**Completed, audyt opisowy.** Bez treningu, danych i nowej oceny jakości.
Run `runs/E036/20261107T231027Z_27bac186`:4GPU/exit0,789,50s,140 modeli,
840 plików czasów; audyt PASS. MAC sprawdzone ręcznie i hookami.

Przy dopasowanych parametrach TILT vs UNIFORM:
- **MAC T/U 1,105–1,823** (+10,5% do+82,3%);
- **przepustowość batch128 0,702–0,948** (−5,2% do−29,8%);
- latencja batch1 0,965–1,046 — w granicach narzutu;
- peak batch128 na Tiny/50–70% wyższy o68–82%.

**Wniosek:** zysk jakości TILT z E034/E035 jest przy stałej liczbie
parametrów, ale nie przy stałym koszcie obliczeń. Nie pisać „matched FLOPs”.
To ograniczenie zakresu twierdzenia, nie unieważnienie jakości.
Jakość w tabelach pochodzi wyłącznie z E035.

Pierwszy run `20261107T225229Z_1f5b7f3a` failed na zbyt silnym warunku
alokatora; poprawka tylko księgowania pamięci. [Pełny raport](WYNIKI.md),
[tabele](TABLES.md). Nowy trening nie był potrzebny.
