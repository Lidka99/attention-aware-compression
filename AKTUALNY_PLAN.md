# Aktualny plan doktoratu

Aktualizacja: 2026-09-29. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — jesteśmy tutaj, etap zakończony.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia; etap 3 jeszcze nie rozpoczęty.
3. Obserwacja podobieństwa kanałów oraz zachowania atencji.
4. Kontrolowane próby łączenia i pomiar rzeczywistej straty jakości.
5. Opracowanie i porównanie nowego kryterium, jeżeli wyniki to uzasadnią.

## Najbliższy krok

Ustalić krótki protokół obserwacji podobieństwa kanałów przed/po SE i zmienności wag atencji w jednym bloku. Nie uruchomiono tego kroku. Audyt i wykresy pozostają w folderze E001; nowy numer przeznaczamy na nową próbę badawczą. Warunek techniczny przejścia dalej został spełniony. Test i rezerwa nie były oceniane, wagi są niezmienione. Cztery GPU oceniały cztery rozłączne części dev — nie stanowią czterech powtórzeń. Audyt oceny nie jest odtworzeniem całego treningu ani dowodem skuteczności kompresji.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
