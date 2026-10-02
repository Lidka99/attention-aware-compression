# Dlaczego sprawdzamy stałe tło kanałów?

Data: 2026-10-02. Dodatkowa analiza E002, opisana po pierwszej obserwacji.

W E002 znaleźliśmy 100 bardzo podobnych par, obejmujących 19 kanałów.
Zanim rozważymy ich łączenie, chcemy zrozumieć, **na czym polega podobieństwo**.

## Prosty przykład

Dwa kanały mogą mieć mapy [9,11] i [11,9]. Obie oscylują wokół tła 10,
więc ich surowy cosinus wynosi około 0,98. Po odjęciu tła dostajemy
[-1,1] i [1,-1]: cosinus wynosi -1, bo wzory są przeciwne.
To przykład rachunkowy, nie wynik modelu. Kontrole na takich sztucznych
mapach przeszły; dowody są w TESTY_CPU.json.

## Co dokładnie rozdzielamy?

1. Stałą średnią kanału, wspólną dla wszystkich obrazów i pozycji.
2. Zmiany średniej mapy między obrazami.
3. Szczegóły przestrzenne w obrębie każdej mapy.

Energia to suma kwadratów liczb w mapach. Możemy ją dokładnie rozdzielić
na te trzy części. Duży udział tła nie oznacza, że jest ono niepotrzebne:
dalsza warstwa także korzysta z tych liczb.

Sprawdzimy podobieństwo po odjęciu tylko stałej średniej oraz po odjęciu
średniej każdej mapy. Drugi wariant usuwa też informację o ogólnej sile
reakcji na dany obraz, więc nie traktujemy go jako automatycznie lepszego
kryterium kompresji. Nie zmieniamy map w modelu — przeliczamy zapisane liczby.

## Dane i wykonanie

Te same aktywacje h/z, te same obrazy A/B i te same 100 par. Nie wybieramy
nowych par na podstawie tego, który wynik pasuje do oczekiwania.
Główna prezentacja na B; A jako kontrola.

Pierwotny odczyt E002 wykorzystywał 4 GPU. Ta dodatkowa analiza działa
na CPU i nie wymaga ponownego odczytu modelu ani użycia GPU. Nie używamy
checkpointu, nowych obrazów, dev, rezerwy ani testu. Dotychczasowy bieg
i wykresy pozostają nienaruszone; nowe wyniki dostaną osobny katalog.

Po wyniku powiemy, czy podobieństwo utrzymuje się bez tła. To nadal
nie będzie dowód bezpieczeństwa połączenia ani skuteczności nowej metody.
Jest to analiza post hoc: pytanie sformułowaliśmy po obejrzeniu E002.

Protokół i dokładna komenda: [OPIS.md](OPIS.md).

## Wynik — analiza wykonana 2026-10-02

Raport i wykresy: [WYNIKI.md](WYNIKI.md). Stała średnia nie jest głównym
wyjaśnieniem wysokiego podobieństwa badanej grupy: przed SE ma medianowo
0,61% energii kanałów, po SE około 0,016%. Po odjęciu tej średniej
cosinus nadal jest wysoki: mediana 0,9476 przed SE i 0,9588 po SE.
Po odjęciu średniej każdej mapy też nie następuje załamanie podobieństwa.

Nie zmieniliśmy map, wag ani wyboru par; źródłowe pliki E002 zachowały
wszystkie hashe. Analizę wykonano na CPU, bez nowej inferencji.

Granica tej kontroli: odejmowaliśmy średnie, nie całą średnią mapę
przestrzenną po obrazach. Podobny wzór może powtarzać się niezależnie od
obrazu. Proponowany następny krok to oddzielenie takiego szablonu od zmian
odpowiedzi między obrazami, na tych samych zapisanych liczbach. Jeszcze
go nie wykonano. Wciąż nie jest to dowód bezpieczeństwa scalenia.
