# Status badań — 2026-10-27

**Kontynuacja w nowym czacie:** [NOTATKA_KONTYNUACJA.md](NOTATKA_KONTYNUACJA.md).

## Cel

Znaleźć metodę kompresji opartą na foldingu, która przy tym samym rozmiarze
sieci zachowuje więcej jakości **bez fine-tuningu**, i pokazać ją na tle
*Cut Less, Fold More* oraz MAG1/MAG2.

## Najważniejszy wynik

**E014 potwierdził na czterech seedach E011**, że nierówna alokacja budżetu
(TILT+0.30: mniej kompresji wczesnych grup, więcej późnych) bije jednolity
folding: **+0,87 pp** przy 30%, **+3,77 pp** przy 50%, **+11,27 pp** przy 70%.
Wszystkie 12 różnic seed×budżet dodatnie; MAG2 znacznie poniżej.

To nadal: jeden dataset (CIFAR-10), dev, cztery checkpointy z jednego przepisu
i adaptacja operatora E012. Brak niezależnego testu i drugiego datasetu.

## Eksperymenty — skrót

| ID | Pytanie | Wynik | Status |
|---|---|---|---|
| E001 | baseline EfficientNet+SE / CIFAR-100 | dev 66,84% | completed |
| E002 | czy podobieństwo kanałów przetrwa SE | obserwacja, bez dowodu korzyści | completed |
| E003 | kompresja 10% bez naprawy | POST−PRE +0,30 pp (CI 0); duża szkoda | completed |
| E004 | równe dostrojenie 120 epok | różnice znikają; RANDOM dorównuje | completed |
| E005 | baseline Tiny ImageNet 96×96 | dev 51,53% | completed |
| E006 | bramki SE jako predyktor szkody | brak wartości ponad `s_post` | completed |
| E007 | baseline Tiny ImageNet 224×224 | dev 58,49% | completed |
| E008 | kompresja 30% + 60 epok naprawy | POST−PRE +0,51 pp (CI 0) | completed |
| E009 | naprawa 120 epok | POST−PRE +0,28 pp (CI 0); różnice znikają | completed |
| E010 | 9 kryteriów selekcji | tylko TAYLOR, bez replikacji | completed |
| E011 | 4 checkpointy ResNet18/CIFAR-10 | dev 95,28–95,60%; średnia 95,46% | completed |
| E012 | folding bez fine-tuningu | po BN 94,20/89,30/70,50% (10/30/50%) | completed |
| E013 | alokacja budżetu, 1 seed | TILT+0.30: +0,58/+4,34/+10,70 pp | completed |
| E014 | replikacja na 4 seedach | +0,87/+3,77/+11,27 pp; 12/12 dodatnie | completed |

## Następny krok

1. Dodać do E014 wykresy i pełną analizę międzyseedową (wynik liczbowy jest
   w [WYNIKI.md](experiments/2026-10-27_E014_replikacja_alokacji/WYNIKI.md)).
2. Powtórzyć alokację na **CIFAR-100** i **Tiny ImageNet**.
3. Ustalić, czy metoda jest regułą depth-aware, czy wymaga predykcji
   wrażliwości; opisać mechanizm.
4. Dopiero potem decyzja o publikacji i ewentualny test na ImageNet.

## Jak czytać dokumentację

- **Ten plik** — aktualny stan, jedna strona.
- **`PODSUMOWANIE.md`** w folderze eksperymentu — cel, wynik, wniosek.
- **`WYNIKI.md` / `OPIS.md`** — pełne tabele, protokoły i ograniczenia.
- Pełne raporty zostają jako archiwum; nie są potrzebne do codziennego czytania.
