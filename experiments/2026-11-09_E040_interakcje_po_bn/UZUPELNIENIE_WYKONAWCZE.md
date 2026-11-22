# E040 — uzupełnienie wykonawcze przed pomiarami jakości

Zgoda użytkowniczki na realizację pilota: kontynuacja rozmowy po projekcie.
Bez nowego treningu i dostępu do testów E035/rezerwy CIFAR-100.
Decyzje metodyczne, budżety, zakres i bramki z OPIS.md pozostają ustalone.

## Korekta kolejności grup wykryta na CPU

Tabela §3 OPIS.md definiuje **kanoniczną kolejność semantyczną E040**.
Określenie jej jako „kolejności upstream” było nieścisłe. Upstream i E035
mają wewnątrz stage2/3/4 kolejność hidden0/output/hidden1, natomiast tabela
E040 ma hidden0/hidden1/output. Adapter E040 działa po nazwach grup;
wektor do upstream jest indeksowany permutacją:

`[0,1,2,3,5,4,6,8,7,9,11,10]`.

Zachowujemy kolejność i semantykę tabeli E040, również we wzorze d_g TILT.
Nie wolno podać kanonicznego wektora bezpośrednio jako wektora upstream.
Historyczny T/b50 E035 musi zostać przemapowany do tej samej kolejności:

- C10: `[57,57,57,115,115,115,222,194,208,360,304,332]`;
- C100: `[57,57,57,115,115,115,221,194,207,359,303,331]`.

Dlatego wcześniejsze liczby kosztu **projekcji TILT** w §5 pkt3 OPIS.md
nie są obowiązującym oczekiwaniem testowym. Poprawne koszty i szerokości
zostaną zapisane przez zamrożenie `frozen.json` po mapowaniu nazw, przed
pierwszym pomiarem jakości. U50, limity, menu L/H, anchor, liczba sond,
przestrzeń4096 i kryteria sukcesu nie zmieniają się. Nie jest to wybór
wariantu na podstawie wyniku jakości.

## Przechowywanie stanów

Zgodnie z doprecyzowaniem użytkowniczki: **bez nowych plików wag**.
Stany potrzebne do kolejnych faz pozostają w RAM żyjących workerów;
jednorazowe warianty sond można zwalniać po zapisaniu odpowiedzi. Na dysku
pozostają metryki, predykcje, wybory, hashe, manifesty i logi. Nie stosujemy
automatycznego restartu z częściowych wyników po awarii procesu.

Uzupełnienie podlega zahashowaniu i snapshotowi razem z OPIS.md i kodem.

Kontrola CPU implementacji2026-11-11 potwierdziła geometrię, forward na
syntetycznych wejściach i bitową zgodność MAG2 z E038. Po prawidłowym
mapowaniu nazw projekcja TILT daje akurat te same koszty co w notatce:
C10 P4 882 317/MAC276 237 070, C100 P4 487 553/MAC269 508 220.
Nie są to pomiary jakości. Run zapisze czas rozpoczęcia i środowisko osobno;
data2026-11-09 w nazwie folderu pozostaje datą utworzenia protokołu.
