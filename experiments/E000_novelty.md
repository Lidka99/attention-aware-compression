# E000 — audyt nowości

Status: draft. Wersja: 0.1. Typ: przegląd literatury; brak obliczeń ML.

## Pytanie
Czy informacja atencji przy scalaniu kanałów wnosi nowy element względem istniejących metod oraz zwykłego podobieństwa aktywacji po atencji?

## Procedura
1. Zdefiniować proponowaną operację scalania i informację dostępną selektorowi.
2. Przeczytać pełne teksty i suplementy ECA-Net, Cut Less Fold More, GRAIL, ADSC i Pick-or-Mix; sprawdzić dalszych najbliższych poprzedników.
3. Szukać również neuron merging, activation-aware folding, kompresji SE/ECA i rekonstrukcji bloków. Zapisać dokładne zapytania, daty, narzędzia, kryteria doboru, wersje PDF i znalezione prace w `literature/MATRIX.md`.
4. Dla bliskich metod sprawdzić cel matematyczny i kod w określonym commicie. Oddzielić brak badania w artykule od braku rozwiązania w całej literaturze.
5. Sprawdzić, czy proponowany wynik jest równoważny selekcji opartej na z = a*h. Jeśli tak, odnotować to przed rozbudową implementacji.

## Wynik
Tabela źródeł, udokumentowany zakres wyszukiwania i zdanie: najbliższa metoda robi X; proponujemy Y, ponieważ przewidujemy problem Z. Nowość pozostaje hipotezą aż do porównania. Ręczny audyt odtwarza się z zapisanych źródeł i procedury; brak komendy ML.

## Bramka
Przejść do badania własnej metody dopiero po wskazaniu konkretnej różnicy algorytmicznej lub pytania mechanistycznego. Jeśli różnicy brak, poprawić hipotezę i wpisać decyzję. E001 można przygotowywać równolegle jako audyt infrastruktury.

Raport: `reports/E000_novelty.md` — jeszcze nie utworzony, brak wyniku.
