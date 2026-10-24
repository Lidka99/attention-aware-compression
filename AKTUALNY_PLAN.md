# Aktualny plan doktoratu

Aktualizacja: 2026-10-22. Pracujemy po jednym etapie.

## Cel

Sprawdzić, czy zmiana miary podobieństwa i grupowania kanałów poprawia jakość po **foldingu bez fine-tuningu**, przy tym samym rozmiarze skompresowanej sieci. Punkt odniesienia: *Cut Less, Fold More*. Najpierw odtworzenie referencji, potem zmiana grupowania. Korekta statystyk BN jest osobnym, jednakowym etapem dla porównywanych metod. Atencja pozostaje wcześniejszą hipotezą; nie jest wymogiem nowego kierunku.

[Audyt kodu i plan odtworzenia](literature/AUDYT_CUT_LESS_FOLD_MORE.md).

## Dotychczasowe etapy E001–E010

1. **Model bazowy — wykonany:** EfficientNet-B0 z SE, od losowych wag, CIFAR-100, seed 42, 200 epok, jeden trening rozproszony na 4 GPU. Zapisany wynik: 3342/5000 dev (66,84%), epoka 177. [Wyniki E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md).
2. **Kontrola checkpointu i przebiegu uczenia — zakończona.** [Audyt E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/AUDYT_WYNIKI.md): 3342/5000, zero różnic predykcji i NLL, cztery GPU po 1250 unikalnych obrazów. Wykresy potwierdzają silne oznaki przeuczenia.
3. **Obserwacja podobieństwa kanałów oraz zachowania atencji — zakres E002 zakończony.** [E002](experiments/2026-10-01_E002_obserwacja_se/WYNIKI.md): jeden blok SE, 1000 obrazów train na 4 GPU, 19 kanałów w wybranych parach. Kontrole [średniej](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-02_skladowa_stala/WYNIKI.md) i [szablonu przestrzennego](experiments/2026-10-01_E002_obserwacja_se/analizy/2026-10-07_szablon_przestrzenny/WYNIKI.md) wykonane: podobieństwo dotyczy również zmian odpowiedzi między obrazami. Nie wykazano skuteczności kompresji ani przewagi kryterium atencji.
4. **Rzeczywista kompresja i pomiar straty jakości — wykonana.** [E003, wyniki](experiments/2026-10-09_E003_kompresja_10proc_parametrow/WYNIKI.md): trzy modele fizycznie mniejsze o 10,0159% parametrów (4 135 648→3 721 424, 1016 kanałów w 15 blokach), PRE/POST/RANDOM ze wspólnym operatorem, bez dostrajania. Dev top-1: BASE 66,84%, PRE 54,94%, POST 55,24%, RANDOM 49,48%. POST−PRE +0,30 pp, przedział [−0,60; +1,22] pp — brak wyraźnej przewagi. Duża utrata jakości bez naprawy.
6. **Nowy baseline Tiny ImageNet — wykonany (E005).** [Wyniki](experiments/2026-10-10_E005_tiny_imagenet_od_zera/WYNIKI.md): 100 epok od zera, dev top-1 51,53% (3h10min na 4 GPU), nadal obecne przeuczenie (train 98,5%). Baseline wystarczający do E006; pod publikację planowany osobny trening 224×224 z mocniejszą recepturą.
5. **Wspólne dostrajanie po kompresji — wykonane.** [E004, wyniki](experiments/2026-10-10_E004_wspolne_dostrajanie/WYNIKI.md): po 120 epokach równego dostrajania BASE 67,62%, PRE 66,96%, POST 67,32%, RANDOM 67,30%. POST−PRE +0,36 pp, przedział [−0,30; +1,02] pp. **Różnice między metodami wyboru zniknęły; wybór losowy dorównał atencyjnemu.** Przy 10% redukcji o wyniku decyduje naprawa, nie selekcja.
6. **Kryterium z bramek SE — E006 zakończony (2026-10-11, wynik negatywny).** [Wyniki](experiments/2026-10-11_E006_kryterium_bramkowe_predykcja/WYNIKI.md): kontrola `s_post` działa kierunkowo (ρ = −0,155), ale `g_pert` nie przewiduje szkody ponad `s_post` (Δρ = +0,199, przedział [−0,020; +0,411]; cząstkowa +0,047). Wątek bramek jako sygnału ponad z zamknięty zgodnie z protokołem.
7. **Nowy baseline 224×224 — E007 zakończony (2026-10-12).** [Wyniki](experiments/2026-10-11_E007_baseline_224/WYNIKI.md): 100 epok od zera na 4 GPU, run `runs/E007/20261011T061825Z_4de51b3_seed42_24mzpz80` (commit `4de51b3`, protokół 1.0, ~4h37m, ~165 s/epokę). Dev top-1 (epoka 100, wynik główny) **58,49%** (5849/10000); najlepszy dev 58,58% (epoka 98); train 99,50%; dev NLL 2,0001. Powyżej E005 (51,53% przy 96×96), poniżej kotwicy Shahriar 2026 (60,54%; porównanie jakościowe, ich ocena na oficjalnym val, nasza na dev). Zastępuje E005 w roli głównego punktu odniesienia do badań kompresji.
8. **Kompresja na baseline 224 — E008 i E009 zakończone.** [Wyniki E008](experiments/2026-10-12_E008_kompresja_224/WYNIKI.md): bez naprawy selekcja > losowa (10%: 47,4/47,1 vs 40,8; 30%: 9,7/6,7 vs 1,0), ale POST < PRE; po 60 epokach naprawy BASE 59,45 / POST 58,32 / PRE 57,81 / RANDOM 57,55; POST−PRE +0,51 pp (CI [−0,18; +1,19]) — próg nieprzekroczony; POST istotnie ponad RANDOM na granicy (CI [+0,03; +1,50]). [Wyniki E009](experiments/2026-10-12_E009_naprawa_120/WYNIKI.md) (120 epok, te same modele startowe): BASE 60,04 / POST 59,46 / PRE 59,18 / RANDOM 59,21; POST−PRE +0,28 pp (CI [−0,41; +0,96]) — przewaga nie występuje; POST tylko 0,58 pp poniżej BASE (CI obejmuje 0). Pierwszy bieg E009 przerwany awarią 13.10 został wyłączony z analizy; pełny bieg zakończony 14.10.
9. **Benchmark kryteriów selekcji — E010 zakończony (2026-10-19).** [Wyniki](experiments/2026-10-18_E010_benchmark_kryteriow/WYNIKI.md): 9 kryteriów, Tiny 30% (primary), Tiny 10% (walidacja), CIFAR-100 10% (replikacja), T ≤ 20. Jedyny kandydat **TAYLOR**: Tiny 30%, T=5 +1,03 pp vs RANDOM (CI [+0,26; +1,79], Holm OK); brak replikacji na CIFAR-100 (+0,36 pp, CI [−0,42; +1,12]) → **`survivor_not_replicated`**. Pozostałe kryteria nie biją RANDOM; po naprawie różnice między selekcjami znikają.

## Najbliższy krok

Aktualizacja wykonania 2026-10-24: przygotowano **E011 — cztery checkpointy
CIFAR-10/ResNet18 od zera** (seed42–45, jeden model na GPU, 200 epok,
batch128 na model). CPU i smoke 4 GPU passed. To adaptacja wybranego punktu
siatki autorów z własnym dev, nie dokładne odtworzenie ich checkpointu.
[Protokół i komendy](experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/OPIS.md).
Następnie E012: kontrola foldingu bez fine-tuningu; potrzebne ukończone
wejście E011 i zweryfikowany operator.

**Odtworzenie referencyjnego foldingu na CIFAR-10/ResNet18.** Audyt statyczny kodu wykonany; do ustalenia konkretny checkpoint i środowisko. Repo autorów nie zawiera gotowych wag zwykłego ResNet18/CIFAR-10. Po ustaleniu wejść: testy operatora, zamrożenie protokołu, pomiar bez fine-tuningu przed i po korekcie BN. Następnie porównanie miar grupowania z oryginalnym foldingiem. Potwierdzenie TAYLORA na trzech seedach nie jest obecnie planowanym E011. Publiczne checkpointy pozostają możliwością do jawnego uzgodnienia względem wcześniejszej zasady treningu od zera; audyt nie zmienia tej zasady. [Szczegóły i kolejność](literature/AUDYT_CUT_LESS_FOLD_MORE.md).

[PLAN_DZIALANIA.md](PLAN_DZIALANIA.md) to historyczny szkic, nie bieżąca kolejność działań.
