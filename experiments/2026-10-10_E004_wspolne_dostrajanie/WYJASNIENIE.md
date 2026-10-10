# E004 — co robimy i dlaczego?

Data: 2026-10-10. **Projekt; trening nieuruchomiony.** Protokół: [OPIS.md](OPIS.md).

E003 pokazało, że po odchudzeniu sieci o10% parametrów jakość mocno
spada: POST55,24% wobec66,84% modelu bazowego, bez żadnego treningu po
kompresji. To wynik „surowy”. Użytkowniczka słusznie zauważyła, że
prawdziwe eksperymenty kompresji zwykle dodają etap naprawy.

## Co zrobi E004?

Wszystkie cztery warianty — BASE, PRE, POST i RANDOM — przejdą dokładnie
to samo **dostrajanie gradientowe**: 120 epok, SGD z momentum0,9, LR0,01
z redukcją po60 i90 epokach, batch256. To protokół z publikacji
GReg (ICLR2021) dla CIFAR, jawnie zaadaptowany do naszego EfficientNet-B0
i obrazów96×96.

Dlaczego BASE też? Bo jeśli dostroimy tylko modele po kompresji, nie
wiemy, czy ich poprawa pochodzi z naprawy kompresji, czy po prostu
z dodatkowego treningu. BASE w tym samym budżecie jest kontrolą.

Dlaczego wszyscy dostają taki sam budżet? Bo uczciwe porównanie metod
wymaga tego samego czasu i danych. Żadna metoda nie dostanie więcej epok
ani lepszych ustawień; nie wybieramy też „najlepszej epoki” po fakcie —
wynik główny to epoka ostatnia (120).

## Czego się spodziewamy — a czego nie wiemy?

Zobaczymy, czy po naprawie:
- luka do BASE się zmniejsza;
- POST nadal wypada lepiej niż PRE (w E003 różnica15 trafień na5000 była
  nieprzekonująca);
- RANDOM zostaje z tyłu.

Nie przesądzamy odpowiedzi. Jeśli po równym budżecie różnica PRE/POST
zniknie, to też jest ważny wynik — oznaczałoby, że sama kompresja
przewagi nie daje. Jeden checkpoint i jedno dostrojenie nie wystarczą
na wniosek o nowej metodzie.

## Co technicznie?

- start: zapisane modele E003 (te same szerokości i selekcja),
- dane: 40000 train, 5000 dev z E001; bez testu i rezerwy,
- sprzęt: dokładnie4GPU, po jednym wariancie naraz (DDP), batch64/GPU,
- pomiar: trafność i NLL na dev co epokę; wynik = epoka120,
- zapis: nowy katalog runs/E004, raport i wykresy w tym folderze.

Czas: cztery treningi po120 epok. Zmierzymy i zapiszemy zamiast zgadywać.
Trening rozpocznie się dopiero po implementacji i kontrolach CPU protokołu.

## Wynik — 2026-10-10

[Raport](WYNIKI.md), run `20261010T140203Z_560deb2_seed42_zutghyj2`.
Po 120 epokach równego dostrajania: BASE67,62%, PRE66,96%, POST67,32%,
RANDOM67,30%. POST−PRE +0,36pp, przedział[−0,30;+1,02]pp — brak
wiarygodnej przewagi. Wybór losowy dorównał atencyjnemu, a wszystkie
skompresowane modele odzyskały jakość sprzed kompresji (start:49–55%).

Wniosek: przy 10% redukcji naprawa niweluje różnice między sposobami
wyboru kanałów. To ważny wynik kontrolny, nie porażka techniczna.
Następny ruch należy do użytkowniczki: większa redukcja (20–30%),
porównanie bez dostrajania, albo zamknięcie wątku na tym pilocie.
