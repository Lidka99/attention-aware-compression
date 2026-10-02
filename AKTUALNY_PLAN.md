# Aktualny plan doktoratu

Aktualizacja: 2026-10-02. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — jesteśmy tutaj.** [E002 wykonane](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU. Wybrane pary pozostają bardzo podobne po SE mimo zmiennych wag; obejmują tylko 19 kanałów. [Kontrola stałej średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) wykonana: mały udział tła, podobieństwo utrzymuje się po centrowaniu. To nie dowód skuteczności kompresji.
4. Kontrolowane próby łączenia i pomiar rzeczywistej straty jakości.
5. Opracowanie i porównanie nowego kryterium, jeżeli wyniki to uzasadnią.

## Najbliższy krok

Ustalić jedną uzupełniającą kontrolę E002: czy podobne wzory przestrzenne zmieniają się podobnie między obrazami, czy są głównie wspólnym powtarzalnym szablonem. Oddzielić średnią mapę po obrazach od odchyleń, na tych samych zapisanych aktywacjach i parach, bez nowego rankingu. Jeszcze jej nie wykonano. Analiza CPU na mapach wcześniej odczytanych na 4 GPU, bez nowej inferencji. Test, rezerwa i dev nie są używane. Nie rozpoczynamy jeszcze łączenia kanałów ani kolejnego treningu.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
