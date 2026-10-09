# Co sprawdzamy teraz?

Data: 2026-10-07. Uzupełnienie E002, analiza post hoc.

Poprzednio odjęliśmy średnie wartości map. Kanały pozostały podobne.
Teraz sprawdzimy, czy nie tworzą podobnego **całego wzoru**, który
powtarza się niezależnie od obrazu.

Dla każdego kanału policzymy średnią mapę 6×6 po obrazach. To szablon
tej próby. Odejmując go od każdej mapy, otrzymamy odpowiedź różną od
typowego wzoru. Porównamy te zmiany w tych samych 100 parach, przed i po SE.
Osobno oddzielimy zmiany całej średniej obrazu od zmian samego układu
przestrzennego. Szablon obliczamy osobno na A i B; nie uczymy nim modelu.

Przykład: dwa kanały mają wspólny wzór [-10,10], ale zmieniają go w
przeciwnych kierunkach zależnie od obrazu. Surowy cosinus może wynosić
około 0,98, a cosinus zmian po odjęciu wzoru -1. To ilustracja rachunku,
nie wynik naszego checkpointu. Kontrole CPU na takich przykładach przeszły.

## Dlaczego komunikat „no model, no GPU”?

Komunikat dotyczy **testu kodu na sztucznych tablicach**: test nie wczytuje
wytrenowanego modelu i nie używa GPU. Sprawdza, czy wzory są poprawnie
zaimplementowane. Nie oznacza braku kart ani rezygnacji z ich używania.

Źródłowe E002 faktycznie wykorzystało **4 GPU po 250 innych obrazów**.
Mapy z wytrenowanego modelu są już zapisane. Dzisiejsza analiza czyta
te rzeczywiste aktywacje i porównuje je na CPU, bez kolejnego forwardu.
Do tych średnich, energii i cosinusów nie trzeba ponownie przepuszczać
obrazów przez sieć. Kod oddziela dowód kontroli na sztucznych danych
od wyniku na rzeczywistych zapisanych mapach.

W obu przypadkach nie aktualizujemy wag. Dev, test i rezerwa nie są
używane. Nie wybieramy nowych par pod wynik tej analizy.

Gdy zakończymy tę kontrolę, następnym krokiem będzie przygotowanie
pierwszej kontrolowanej próby scalenia i pomiaru skutków. Samo wysokie
podobieństwo zmian nie gwarantuje bezpieczeństwa połączenia.

Protokół i dokładna komenda: [OPIS.md](OPIS.md).

## Wynik na rzeczywistych zapisanych mapach — 2026-10-07

[Raport i wykresy](WYNIKI.md). Szablon ma medianowo 4,07% energii
wybranych kanałów przed SE i 1,72% po SE. Pozostała część map zmienia
się między obrazami. Po odjęciu całej średniej mapy podobieństwo tych
zmian nadal jest wysokie: mediana cosinusa 0,9520 przed SE i 0,9591 po SE.
Także same zmiany wzorów przestrzennych pozostają bardzo podobne.

To przemawia przeciw dominacji stałego szablonu w wysokim podobieństwie
tej grupy. Kanały podobnie zmieniają odpowiedzi na badanych obrazach.
Nie oznacza to jeszcze, że można je połączyć bez straty jakości.
Nie wykazaliśmy dodatkowej korzyści wag atencji ponad aktywacje po SE.

Domykamy te kontrole E002. Proponowany następny krok to **przygotowanie
protokołu jednej rzeczywistej próby scalenia** wraz z kontrolą zależnych
warstw i pomiarem skutków. Nie wykonano jej tutaj. Źródłowe E002 było
na 4 GPU; ta analiza na CPU wykorzystała zapisane aktywacje, bez nowych
obrazów, wczytywania modelu i aktualizacji jego wag.
