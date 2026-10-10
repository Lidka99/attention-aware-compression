# Aktualny plan doktoratu

Aktualizacja: 2026-10-10. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — zakres E002 zakończony.** [E002](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU, 19 kanałów w wybranych parach. Kontrole [średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) i [szablonu przestrzennego](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md) wykonane: podobieństwo dotyczy również zmian odpowiedzi między obrazami. Nie wykazano skuteczności kompresji ani przewagi kryterium atencji.
4. **Rzeczywista kompresja i pomiar straty jakości — wykonana, jesteśmy tutaj.** [E003, wyniki](experiments/2026-10-09_E003_kompresja_10proc_parametrow/WYNIKI.md): trzy modele fizycznie mniejsze o 10,0159% parametrów (4 135 648→3 721 424, 1016 kanałów w 15 blokach), PRE/POST/RANDOM ze wspólnym operatorem, bez dostrajania. Dev top-1: BASE 66,84%, PRE 54,94%, POST 55,24%, RANDOM 49,48%. POST−PRE +0,30 pp, przedział [−0,60; +1,22] pp — brak wyraźnej przewagi. Duża utrata jakości bez naprawy.
5. **Wspólne dostrajanie po kompresji — wykonane, jesteśmy tutaj.** [E004, wyniki](experiments/2026-10-10_E004_wspolne_dostrajanie/WYNIKI.md): po 120 epokach równego dostrajania BASE 67,62%, PRE 66,96%, POST 67,32%, RANDOM 67,30%. POST−PRE +0,36 pp, przedział [−0,30; +1,02] pp. **Różnice między metodami wyboru zniknęły; wybór losowy dorównał atencyjnemu.** Przy 10% redukcji o wyniku decyduje naprawa, nie selekcja.
6. Opracowanie i porównanie nowego kryterium — wymaga nowego uzasadnienia; przy obecnym budżecie wyniki go nie wspierają.

## Najbliższy krok

Decyzja użytkowniczki, co dalej po wyniku kontrolnym E004. Możliwe kierunki (żadnego nie uruchamiam automatycznie): (a) ten sam protokół przy większej redukcji, np. 20–30%, gdzie wybór kanałów może zacząć odgrywać rolę; (b) porównanie przy braku dostrajania, gdzie widać sam efekt selekcji; (c) świadome zamknięcie wątku attention-aware na tym pilocie. E003 i E004 pozostają niezmienione. Rezerwy ani testu nie używamy.

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
