# Aktualny plan doktoratu

Aktualizacja: 2026-10-10. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — zakres E002 zakończony.** [E002](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU, 19 kanałów w wybranych parach. Kontrole [średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) i [szablonu przestrzennego](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md) wykonane: podobieństwo dotyczy również zmian odpowiedzi między obrazami. Nie wykazano skuteczności kompresji ani przewagi kryterium atencji.
4. **Rzeczywista kompresja i pomiar straty jakości — wykonana, jesteśmy tutaj.** [E003, wyniki](experiments/2026-10-09_E003_kompresja_10proc_parametrow/WYNIKI.md): trzy modele fizycznie mniejsze o 10,0159% parametrów (4 135 648→3 721 424, 1016 kanałów w 15 blokach), PRE/POST/RANDOM ze wspólnym operatorem, bez dostrajania. Dev top-1: BASE 66,84%, PRE 54,94%, POST 55,24%, RANDOM 49,48%. POST−PRE +0,30 pp, przedział [−0,60; +1,22] pp — brak wyraźnej przewagi. Duża utrata jakości bez naprawy.
5. **Wspólne dostrajanie po kompresji — przygotowanie E004.** Równy budżet dla BASE/PRE/POST/RANDOM, oparty na opublikowanym protokole CIFAR-100, bez zmiany szerokości ani selekcji. [Szkic protokołu](experiments/2026-10-10_E004_wspolne_dostrajanie/OPIS.md).
6. Opracowanie i porównanie nowego kryterium, jeżeli wyniki to uzasadnią.

## Najbliższy krok

Dokończyć protokół E004 i zaimplementować wspólne dostrajanie na 4 GPU: SGD, momentum0,9, weight decay5e-4, LR 0,01→0,001 po60→0,0001 po90 epokach, 120 epok, globalny batch256 (adaptacja protokołu GReg, ICLR2021, dla CIFAR). Wynik końcowy bez wybierania epoki po fakcie. E003 pozostaje wynikiem przed naprawą; nie zmieniamy go. Rezerwy ani testu nie używamy.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
