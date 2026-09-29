# Aktualny plan doktoratu

Aktualizacja: 2026-09-29. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — jesteśmy tutaj.** Audyt integralności, ponowna ocena tych samych 5000 obrazów dev na 4 GPU, porównanie predykcji i wykresy.
3. Obserwacja podobieństwa kanałów oraz zachowania atencji.
4. Kontrolowane próby łączenia i pomiar rzeczywistej straty jakości.
5. Opracowanie i porównanie nowego kryterium, jeżeli wyniki to uzasadnią.

## Najbliższy krok

Przygotować i wykonać [audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_OPIS_v1.md). Audyt i wykresy pozostają w folderze E001; nowy numer przeznaczamy na nową próbę badawczą. Warunek przejścia dalej: zweryfikowane wejścia i zgodna ocena lub wyjaśnione rozbieżności. Nie używamy testu ani rezerwy, nie zmieniamy wag i nie rozpoczynamy kolejnego treningu. Cztery GPU oceniają cztery rozłączne części dev — nie stanowią czterech powtórzeń. Audyt oceny nie jest odtworzeniem całego treningu ani dowodem skuteczności kompresji.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
