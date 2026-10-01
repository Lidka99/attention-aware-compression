# E002 — co chcemy teraz zobaczyć?

Data: 2026-10-01. Protokół i kod przygotowane; status pomiaru podaje manifest biegu.

Mamy sprawdzony model. Teraz zajrzymy do jednego bloku w jego środku:
`features.4.1`. Atencja SE waży tam 480 kanałów. Wybraliśmy go na podstawie
budowy sieci, zanim zobaczyliśmy wyniki kanałów. Nie zmieniamy modelu.

## Trzy rzeczy do odczytu

- **h:** mapy kanałów przed SE — reakcje na obraz.
- **a:** wagi przypisane przez SE temu obrazowi.
- **z=a*h:** mapy po SE.

Możemy odczytać te liczby podczas zwykłego przejścia obrazu przez sieć.
Hook to punkt odczytu, a nie nowa warstwa. Wyjście wewnętrznej projekcji
fc1 nie jest tutaj z; z to całe wyjście SE.

## Dlaczego nie wystarczy jeden pomiar podobieństwa?

Wektory [1,2] i [2,4] mają taki sam kierunek. Cosinus wynosi 1, chociaż
jeden kanał jest dwa razy silniejszy. Mnożenie pojedynczej mapy przez
dodatnią wagę SE nie zmienia tego kierunku.

Dlatego mierzymy zarówno kierunek, jak i różnicę wartości. Łączymy mapy
z wielu obrazów: jeśli wagi zmieniają się między obrazami inaczej dla
dwóch kanałów, ich wspólne podobieństwo może się zmienić. Oddzielnie
sprawdzamy zmienność stosunku wag. Wciąż nie jest to pomiar błędu scalenia.

## Mała i uporządkowana próba

1000 obrazów ze zbioru train E001: po 10 z każdej klasy. Połowa (A)
służy do wyboru 100 najbardziej podobnych par przed SE. Druga połowa (B)
pozwala zobaczyć, czy te same pary podobnie zachowują się na innych obrazach.
Nie wybieramy par według tego, jak duży efekt atencji dają. Zapiszemy także
wyniki wszystkich par, żeby nie opierać wniosku wyłącznie na przykładach.

Te obrazy były użyte w treningu modelu. B to sprawdzenie na innych obrazach
niż A, ale nie niezależny test generalizacji. Dev, test i rezerwa pozostają
poza tą próbą. Cztery GPU dostaną po 250 różnych obrazów — to jeden pomiar.

## Jak rozumiemy wynik?

Jeżeli kanały podobne przed SE są inaczej ważone, będziemy mieć obserwację
uzasadniającą dalsze pytania. Nie dowodzi to, że warto je łączyć ani że
osobne użycie wag atencji jest lepsze od podobieństwa aktywacji po SE.
Brak wyraźnych różnic lub niestabilność podobieństwa też są ważnymi wynikami.

Kod odczytu h/a/z i miar jest przygotowany. Kontrole na CPU przeszły:
miary zgadzają się z bezpośrednimi wzorami, hooki zachowują z=a*h i nie
zmieniają odpowiedzi ani stanu losowo inicjalizowanego modelu. Zapisano
listy A/B przed inferencją, z SHA-256
`338eb31862ab3d35495c22c8ae58109aed280a8218f1c32e234c941c3d1bae1d`.
To dowód kontroli technicznych, nie wynik hipotezy na CIFAR.

**Następny krok:** pomiar zamrożonego protokołu na dokładnie 4 GPU.
Nie zaczynamy całego programu badań ani nowego treningu.

Szczegóły miar, danych i kontroli: [OPIS.md](OPIS.md).
Wersja wykonania i komenda: [PROTOKOL_v1.md](PROTOKOL_v1.md).

## Czy uruchomimy trening na 4 GPU?

Obliczenia E002 mają wykorzystywać dokładnie 4 GPU, po 250 różnych obrazów
na kartę. Jest to odczyt aktywacji i wag atencji istniejącego modelu w
trybie oceny, bez aktualizacji parametrów. E001 był treningiem rozproszonym
na 4 GPU i jest już zakończony. Nowego treningu nie potrzebujemy do tej
obserwacji; jego ewentualny cel i protokół wymagałyby osobnego uzasadnienia.
Kod sprawdzono przed pomiarem; wynik rzeczywistego odczytu dokumentuje
osobny raport i manifest, nie samo istnienie skryptu.
