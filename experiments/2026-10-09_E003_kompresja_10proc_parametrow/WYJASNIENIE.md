# E003 — od obserwacji do rzeczywistej kompresji modelu

Data: 2026-10-09. **Projekt eksperymentu; wynik kompresji jeszcze niezmierzony.**

Użytkowniczka wybrała cel: **10% parametrów całej sieci**. Redukcja
jednego kanału nie jest naszym wynikiem badawczym. To może być tylko
wewnętrzny test poprawności kodu.

## Jak duża będzie zmiana?

Wyznaczono plan 15 bloków: z 4 135 648 do **3 721 424 parametrów**,
czyli **10,0159% mniej**. Łącznie 1016 kanałów wewnętrznych mniej,
przy zachowaniu wymiarów połączeń między blokami. Małe przekroczenie
10% wynika z dyskretnych szerokości, które zachowujemy jako wielokrotności 8.

To plan obliczony z budowy sieci, bez danych i bez ocen jakości.
Musimy go jeszcze potwierdzić po fizycznej zmianie wytrenowanego modelu.
Nie nazywamy tej statycznej tabeli wykonanym wynikiem kompresji.

## Jakie pytanie naukowe rozstrzygniemy?

Czy dla tej samej redukcji i tej samej procedury scalenia wybór par
**po atencji SE** zachowuje jakość lepiej niż wybór **przed SE**?

Porównamy:
- BASE — oryginalny model;
- PRE — pary wybrane według map przed SE;
- POST — pary wybrane według map po SE;
- RANDOM — wybór losowy jako kontrola.

Trzy mniejsze modele będą miały dokładnie te same szerokości i liczbę
parametrów. Wszystkie dostaną te same 500 obrazów do kalibracji i tę samą
kompensację, bez dostrajania i rekalibracji BatchNorm. Dzięki temu różnica
wyniku ma odpowiadać zmianie wyboru par, a nie większemu budżetowi naprawy.

## Co znaczy tutaj scalenie?

Z pary zostawiamy jedną ścieżkę obliczeń jako reprezentanta. Drugą
fizycznie usuwamy, a jej wkład wchłaniamy do wag SE i następnej projekcji.
Nie jest to maska na już obliczonym kanale. Dopasowanie dwóch stałych
współczynników jest częścią wspólnego operatora; model po zmianie ma
mniejszy graf. Atencja działa na tym grafie normalnie i może zmienić
wagi także innych kanałów — mierzymy rzeczywisty forward.

## Co zmierzymy?

Trafność i stratę na tych samych 5000 obrazów dev, różnicę POST−PRE
na tych samych obrazach, błąd wyjść bloków, liczbę parametrów, rozmiar
porównywalnych plików modelu i czas wykonania. Wynik gorszy albo brak
różnicy też odpowie na pytanie; nie będziemy dobierać po nim ustawień
w ramach tego samego zakończonego biegu.

Forward i ocena wykorzystają **4 GPU** z rozłącznymi obrazami. Analiza
statystyk może działać na CPU. Cztery karty nie oznaczają czterech
niezależnych modeli. Nie trenujemy nowego modelu od zera i nie używamy
publicznych wag — kompresujemy własny checkpoint E001 na nowych kopiach.

## Co będzie, a czego nie będzie można stwierdzić?

Uzyskamy rzeczywisty pomiar kompresji całej sieci przy określonym budżecie
i kontrolowane porównanie trzech sposobów wyboru par. Jeden checkpoint
nie pozwoli jeszcze uogólnić przewagi na wszystkie modele. Samo POST
nie jest nową metodą; przyszłe kryterium atencji musi wykazać korzyść
ponad takie odniesienie. Nie mamy obowiązku uzyskać pozytywnego wyniku.

Najbliższy krok to **implementacja wspólnego operatora i wykonanie tego
eksperymentu**, po kontrolach kodu i zamrożeniu wersji wykonania.
Nie dokładamy teraz kolejnych analiz E002 ani kolejnego treningu.

Szczegóły: [OPIS.md](OPIS.md), [PLAN_STRUKTURY.json](PLAN_STRUKTURY.json),
[config.draft.json](config.draft.json).

## Co oznacza SE i czy 10% to dużo? (2026-10-10)

SE to **Squeeze-and-Excitation**, moduł atencji kanałowej. Najpierw
uśrednia mapę każdego kanału po pozycjach przestrzennych (squeeze),
potem mała sieć wylicza wagę [0,1] dla każdego kanału (excitation).
Cała mapa danego kanału jest mnożona przez tę wagę. Wagi zależą od obrazu
i od reakcji wielu kanałów. Nie są bezpośrednim dowodem znaczenia dla
klasyfikacji ani możliwością pominięcia już wykonanej konwolucji.

10% parametrów jest **umiarkowaną, zauważalną redukcją**, nie skrajną
kompresją. W naszym planie znika 414224 parametrów z około 4,14 mln.
To około 1,66 MB samych danych wag FP32, bez buforów/metadanych pliku.
Rzeczywisty rozmiar zapisanych modeli zmierzymy. Redukcja parametrów
nie oznacza automatycznie takiej samej redukcji czasu czy kosztu forwardu,
bo warstwy pracują na różnych rozmiarach map i mają różne implementacje.

Do wniosków badawczych liczy się strata jakości i porównanie metod przy
tym samym budżecie, nie sam procent. E003 odpowie na pytanie PRE/POST
przy 10,0159%; nie dowodzi jeszcze nowości lub przewagi w innych budżetach.

Wykonanie: [PROTOKOL_v1.md](PROTOKOL_v1.md). Po naprawie środowiska
potwierdzono CUDA=True i **4×RTX2080Ti**. Kontrole kodu są na CPU,
forwardy badania na wszystkich czterech GPU, po jednym procesie na kartę.
Wczytywanie plików, sumy FP64 i raport nie wymagają ponownej pracy modelu
na GPU. Karty mogą być bez obciążenia podczas przygotowania, co nie
oznacza przejścia na mniejszą liczbę GPU w eksperymencie.
