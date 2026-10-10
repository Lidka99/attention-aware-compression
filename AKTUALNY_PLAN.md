# Aktualny plan doktoratu

Aktualizacja: 2026-10-09. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — zakres E002 zakończony.** [E002](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU, 19 kanałów w wybranych parach. Kontrole [średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) i [szablonu przestrzennego](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md) wykonane: podobieństwo dotyczy również zmian odpowiedzi między obrazami. Nie wykazano skuteczności kompresji ani przewagi kryterium atencji.
4. **Rzeczywista kompresja i pomiar straty jakości — jesteśmy tutaj, przygotowanie E003.** Użytkowniczka wybrała 10% parametrów całej sieci. [Projekt eksperymentu](experiments/2026-10-09_E003_kompresja_10proc_parametrow/OPIS.md): PRE/POST/RANDOM przy identycznym budżecie i operatorze, bez dostrajania. Kompresja jeszcze nieuruchomiona.
5. Opracowanie i porównanie nowego kryterium, jeżeli wyniki to uzasadnią.

## Najbliższy krok

Zaimplementować wspólny operator E003, sprawdzić go, zamrozić wersję wykonania i uruchomić rzeczywistą kompresję/ocenę na 4 GPU. Statyczny plan: 4 135 648→3 721 424 parametrów (10,0159% mniej), 1016 kanałów wewnętrznych mniej w 15 blokach. To plan, nie wynik zmierzonej kompresji. Jeden kanał jest wyłącznie testem technicznym, nie warunkiem badawczym. Następny wynik ma dotyczyć jakości, rozmiaru i wykonania całej mniejszej sieci; nie dokładamy kolejnych obserwacji E002 ani treningu.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
