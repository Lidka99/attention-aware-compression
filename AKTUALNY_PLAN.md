# Aktualny plan doktoratu

Aktualizacja: 2026-10-07. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — jesteśmy tutaj, zakres E002 zakończony.** [E002](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU, 19 kanałów w wybranych parach. Kontrole [średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) i [szablonu przestrzennego](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md) wykonane: podobieństwo dotyczy również zmian odpowiedzi między obrazami. Nie wykazano skuteczności kompresji ani przewagi kryterium atencji.
4. Kontrolowane próby łączenia i pomiar rzeczywistej straty jakości — następny etap, jeszcze nieuruchomiony.
5. Opracowanie i porównanie nowego kryterium, jeżeli wyniki to uzasadnią.

## Najbliższy krok

Przygotować protokół pierwszej kontrolowanej próby scalenia jednej pary w features.4.1: operator fizycznej zmiany szerokości, zależne warstwy/SE, kontrole i pomiar skutków dla bloku oraz sieci. Nowa próba dostanie własny folder i budżet ustalony przed uruchomieniem. Porównania kryteriów muszą obejmować aktywacje po SE przy takim samym budżecie kompresji i naprawy. Dotychczas nie połączono kanałów ani nie uruchomiono nowego treningu. Ostatnie kontrole były analizą CPU map wcześniej odczytanych na 4 GPU, bez nowych obrazów.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
