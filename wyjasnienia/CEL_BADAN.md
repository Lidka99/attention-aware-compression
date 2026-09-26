# Cel badań — do czego dążymy?

**Dążymy do opracowania metody, która zmniejsza sieć splotową z atencją, zachowując możliwie dużo jej skuteczności.** Chcemy sprawdzić, czy uwzględnienie działania atencji pomoże lepiej wybierać kanały do połączenia.

To kierunek badań, a nie potwierdzony pomysł na nową metodę. Dopiero eksperymenty i porównanie z literaturą pokażą, czy jest tu wartościowy wkład naukowy.

### 1. Co konkretnie chcemy zmniejszać?

Sieć splotowa tworzy wiele kanałów — map liczb opisujących przetwarzany obraz. Niektóre mogą przekazywać częściowo podobną informację.

Wyobraź sobie warstwę z 64 kanałami. Chcielibyśmy uzyskać na przykład 48 kanałów, odpowiednio przebudowując też połączone z nią elementy sieci. Liczby są tylko ilustracją, nie ustalonym budżetem eksperymentu.

Mniej kanałów może oznaczać mniej parametrów i obliczeń. Rzeczywiste przyspieszenie trzeba osobno zmierzyć, bo zależy również od sprzętu i implementacji.

### 2. Na czym polega łączenie kanałów?

Zamiast po prostu usuwać jeden z dwóch podobnych kanałów, próbujemy przenieść część jego wkładu do pozostającego kanału i odpowiednio zmienić dalsze obliczenia. Takie podejście nazywa się łączeniem kanałów lub foldingiem.

To przybliżenie: po zmianie sieć może działać gorzej. Naszym zadaniem będzie ograniczyć ten spadek.

### 3. Po co nam atencja?

Atencja kanałowa zmienia siłę kanałów zależnie od obrazu. Dwa kanały mogą wyglądać podobnie przed atencją, ale dostawać różne wagi.

Nasze główne pytanie brzmi:

**Czy informacja o tym, jak atencja traktuje kanały na różnych obrazach, pozwala lepiej przewidzieć, które kanały można połączyć z małą stratą jakości?**

Samo wykrycie różnych wag nie wystarczy. Musimy sprawdzić, czy ta informacja pomaga przewidywać rzeczywiste skutki połączenia.

### 4. Jak wyglądałby sukces?

Porównujemy kilka sposobów wybierania kanałów do połączenia. Wszystkie mają otrzymać taki sam docelowy rozmiar sieci, taki sam budżet danych i taką samą procedurę naprawy po kompresji.

Sukcesem byłaby sytuacja, w której nasz sposób wyboru daje mniejszy spadek jakości niż mocne metody porównawcze, a wynik powtarza się w kolejnych próbach i wybranych modelach.

Szczególnie ważne będzie porównanie z prostym pomysłem: **wybieraj kanały na podstawie ich aktywacji już po atencji**. Jeśli to wystarczy, bardziej złożona metoda oparta na wagach atencji może nie wnosić dodatkowej wartości.

Innowacyjność musi wynikać z czegoś konkretnego: nowego, użytecznego kryterium wyboru albo dobrze potwierdzonego wyjaśnienia, kiedy i dlaczego istniejące metody zawodzą. Samo połączenie słów „atencja” i „kompresja” nie zapewnia nowości.

### 5. Po co pierwszy eksperyment?

Pierwszy eksperyment jest małym krokiem diagnostycznym:

**Sprawdzamy, czy kanały podobne przed atencją są przez nią traktowane podobnie na różnych obrazach.**

To pomoże zrozumieć zachowanie modelu. Następnie trzeba będzie rzeczywiście łączyć wybrane kanały i mierzyć zmianę jakości. Dopiero ten kolejny etap sprawdzi, czy obserwowane różnice mają znaczenie dla kompresji.

Jeśli atencja nie dostarczy informacji przydatnej ponad proste metody porównawcze, będzie to argument za zmianą kierunku. Nie będziemy zakładać z góry, że hipoteza musi się potwierdzić.

### 6. Co robimy teraz?

**Najbliższy krok to wybór jednego modelu z atencją i dostępnymi wytrenowanymi wagami.** Na nim przygotujemy jeden mały, opisany i odtwarzalny pomiar.

Każdy eksperyment dostanie osobną kartę: pytanie, dokładny przepis, wersję kodu, model i dane, ustawienia, komendę uruchomienia oraz wyniki i wnioski. Na początku budujemy wiarygodny sposób sprawdzania pomysłu; dopiero na jego podstawie rozwijamy metodę.
