# Aktualny plan doktoratu

Aktualizacja: 2026-10-12. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zachowanie atencji kanałowej pomaga wybierać kanały do połączenia przy mniejszej stracie jakości. Kanały podobne przed atencją mogą być różnie ważone dla różnych obrazów. To hipoteza, nie potwierdzona luka badawcza. Wymagana jest kontrola oparta na aktywacjach **po atencji**, przy jednakowym budżecie kompresji i naprawy.

## Etapy

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — zakres E002 zakończony.** [E002](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU, 19 kanałów w wybranych parach. Kontrole [średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) i [szablonu przestrzennego](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md) wykonane: podobieństwo dotyczy również zmian odpowiedzi między obrazami. Nie wykazano skuteczności kompresji ani przewagi kryterium atencji.
4. **Rzeczywista kompresja i pomiar straty jakości — wykonana.** [E003, wyniki](experiments/2026-10-09_E003_kompresja_10proc_parametrow/WYNIKI.md): trzy modele fizycznie mniejsze o 10,0159% parametrów (4 135 648→3 721 424, 1016 kanałów w 15 blokach), PRE/POST/RANDOM ze wspólnym operatorem, bez dostrajania. Dev top-1: BASE 66,84%, PRE 54,94%, POST 55,24%, RANDOM 49,48%. POST−PRE +0,30 pp, przedział [−0,60; +1,22] pp — brak wyraźnej przewagi. Duża utrata jakości bez naprawy.
6. **Nowy baseline Tiny ImageNet — wykonany (E005).** [Wyniki](experiments/2026-10-10_E005_tiny_imagenet_od_zera/WYNIKI.md): 100 epok od zera, dev top-1 51,53% (3h10min na 4 GPU), nadal obecne przeuczenie (train 98,5%). Baseline wystarczający do E006; pod publikację planowany osobny trening 224×224 z mocniejszą recepturą.
5. **Wspólne dostrajanie po kompresji — wykonane.** [E004, wyniki](experiments/2026-10-10_E004_wspolne_dostrajanie/WYNIKI.md): po 120 epokach równego dostrajania BASE 67,62%, PRE 66,96%, POST 67,32%, RANDOM 67,30%. POST−PRE +0,36 pp, przedział [−0,30; +1,02] pp. **Różnice między metodami wyboru zniknęły; wybór losowy dorównał atencyjnemu.** Przy 10% redukcji o wyniku decyduje naprawa, nie selekcja.
6. **Kryterium z bramek SE — E006 zakończony (2026-10-11, wynik negatywny).** [Wyniki](experiments/2026-10-11_E006_kryterium_bramkowe_predykcja/WYNIKI.md): kontrola `s_post` działa kierunkowo (ρ = −0,155), ale `g_pert` nie przewiduje szkody ponad `s_post` (Δρ = +0,199, przedział [−0,020; +0,411]; cząstkowa +0,047). Wątek bramek jako sygnału ponad z zamknięty zgodnie z protokołem.
7. **Nowy baseline 224×224 — E007 zakończony (2026-10-12).** [Wyniki](experiments/2026-10-11_E007_baseline_224/WYNIKI.md): 100 epok od zera na 4 GPU, run `runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80` (commit `4de51b3`, protokół 1.0, ~4h37m, ~165 s/epokę). Dev top-1 (epoka 100, wynik główny) **58,49%** (5849/10000); najlepszy dev 58,58% (epoka 98); train 99,50%; dev NLL 2,0001. Powyżej E005 (51,53% przy 96×96), poniżej kotwicy Shahriar 2026 (60,54%; porównanie jakościowe, ich ocena na oficjalnym val, nasza na dev). Zastępuje E005 w roli głównego punktu odniesienia do badań kompresji.

## Najbliższy krok

E007 zakończony: mamy baseline **224×224 pod publikację** (dev 58,49%, [WYNIKI](experiments/2026-10-11_E007_baseline_224/WYNIKI.md)). Kolejny krok pozostaje **decyzją użytkowniczki**:

- (a) **szybki test 30% redukcji** na tym baseline (kandydat E008) — E003/E004 testowały tylko 10% na CIFAR-100 i po równym dostrajaniu różnice między metodami wyboru zniknęły; 30% to sygnał, czy selekcja kanałów zaczyna mieć znaczenie;
- (b) **inne warianty**, np. mocniejsza receptura przeciw przeuczeniu, dodatkowe seedy.

Otwieramy ją jako otwartą decyzję, bez przesądzania. E003/E004/E006/E007 pozostają niezmienione, rezerwy i testu nie używamy. [Rejestr decyzji](docs/DECISIONS.md).

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
