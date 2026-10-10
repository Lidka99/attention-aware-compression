# E003 — wersja wykonania 1.0

Eksperyment: `2026-10-09_E003_kompresja_10proc_parametrow`.
Data zamrożenia: 2026-10-10 UTC. **Status protokołu: frozen**.
Status wykonania podaje manifest biegu, nie sam status konfiguracji.

Przyjmujemy pytanie, budżet, wybór par, operator, dane, kontrole i
ograniczenia z zachowanego [OPIS.md](OPIS.md), v0.1-draft. Ta wersja
uzupełnia go kodem i dokładną komendą. [config.json](config.json) ma te
same parametry naukowe co szkic (poza statusem i numerem wersji).
Nie podmieniamy ustawień po wyniku.

## Pytanie i kryteria

POST kontra PRE: sparowana różnica poprawnych odpowiedzi na dev=5000
i 95% przedział bootstrap po obrazach/klasach przy jednym checkpointcie.
RANDOM to kontrola. Wszystkie trzy mniejsze modele muszą mieć dokładnie
3721424 parametrów, BASE 4135648. Redukcja 414224 = 10,0159% całej sieci.
Wspólne szerokości 15 bloków, A=500 do par i współczynników, B=500 do
diagnostyki, brak treningu i rekalibracji BN. Nie używamy rezerwy/testu.

Sukces techniczny: hashe/ID poprawne, każdy obraz raz, kształty i parametry
rzeczywiście zmniejszone, baseline odtwarza 3342/5000 i wszystkie predykcje,
NLL różni się od E001 nie więcej niż 1e-6, wszystkie modele niezmienione
przed/po ocenie i identyczne między rankami, finite logits oraz komplet
artefaktów. Wynik jakości gorszy nie jest błędem technicznym. Nie dobieramy
progu przewagi po zobaczeniu wyników. Statystyki nie zastępują wielu seedów.

## Kod i kontrole

- [folding.py](folding.py): fizyczna zmiana producentów, BN, SE i projekcji,
  nowe kopie, kontrola budżetu i odrzucenie podwójnego foldingu.
- [selection.py](selection.py): wspólna kwalifikacja, PRE/POST/RANDOM,
  rozłączne pary, dopasowanie alpha/beta na A.
- [support.py](support.py): wejścia i dane, z niezmienionym helperem E002
  common.py jako jawną zależnością zapisywaną wraz z hashem.
- [study.py](study.py): kalibracja i ocena na dokładnie czterech GPU.
- [summarize.py](summarize.py): sparowane porównanie, bootstrap i wykresy CPU.
- [launch.py](launch.py): nowy bieg, commit/środowisko/komendy/logi/hashe.
- Kontrole: [TEST_FOLDING_CPU.json](TEST_FOLDING_CPU.json),
  [TESTY_CPU_v1.json](TESTY_CPU_v1.json). Testy są syntetyczne na CPU,
  nie oznaczają wykonania badania na CIFAR. Launcher odrzuca zmieniony kod
  lub konfigurację po dowodzie kontroli.

Sprawdzamy konstrukcję całego mniejszego modelu, algebraiczną tożsamość
na duplikowanych ścieżkach i proporcjonalnym podsystemie, odtworzenie
grafu przez materializację szerokości i strict load, hooki g/h/z,
kopię gałęzi przed inplace residual, niezmienność wyjścia bazowego,
bezpośrednie współczynniki LS, rozłączność par i bootstrap na identycznych
odpowiedziach. Nie dopasowujemy kodeksu testów pod jakość kompresji.

## Wykonanie czterech GPU

CUDA_VISIBLE_DEVICES=0,1,2,3, cztery procesy po jednej karcie. Wszystkie
forwardy kalibracji, oceny i pomiary czasu na GPU; sumy iloczynów/wybór
par i raport na CPU. A/B: po 125 obrazów/rank (3 batche po32 i końcówka29).
Dev: po1250/rank (39 batchy po32 i końcówka2). Bez dopełniania i drop_last.
To jeden checkpoint z czterema wariantami, nie cztery seedy.

Modele oceniane w kolejności BASE/PRE/POST/RANDOM, źródłowy checkpoint
niemodyfikowany. Parowanie i współczynniki są zapisane przed B/dev.
Każdy zapisany inference-only model jest materializowany i strict-loaded
przed oceną, bez ponownego foldingu. Hash stanu musi zgadzać się z zapisem.

Seedy i ustawienia zgodne ze szkicem: 42+rank, loader42+rank+100000,
RANDOM20261009+1000*index_bloku, bootstrap20261009, 2000 replikacji.
FP32 forward, FP64 statystyki, bez AMP/TF32. BLAS/OMP/MKL jeden wątek,
dwa workery danych/proces, in_order=True. Stany RNG przed/po w biegu.
Deterministyczne operacje wymagane, CUBLAS_WORKSPACE_CONFIG=:4096:8.

## Zmiana sterownika przed uruchomieniem

Po autoaktualizacji CUDA była niedostępna (0 GPU, błąd sterownika/NVML).
Użytkowniczka przywróciła karty. Przed przygotowaniem wykonania potwierdzono
4×RTX2080Ti, CUDA=True i sterownik **580.178.04**. E001 miał 580.173.02.
To jawna różnica środowiska, nie zmiana protokołu danych lub wag. Aktualne
wersje i porównanie do środowiska E001 zapisuje launcher. Baseline ma
odtworzyć predykcje i NLL w ustalonej tolerancji także po zmianie sterownika.

## Dokładna komenda i artefakty

Z głównego katalogu repo, z czystego commita kodu i tego protokołu:

```bash
conda run --no-capture-output -n attention python experiments/2026-10-09_E003_kompresja_10proc_parametrow/launch.py
```

Commit zapisuje się automatycznie w manifest.json przed pierwszym pomiarem.
Każde wykonanie ma nowy `runs/E003/<UTC>_<git>_seed42_<unikalny>/`.
Po awarii zachowujemy wynik failed i logi, nie nadpisujemy biegu ani nie
zmieniamy budżetu automatycznie. Launcher nie obsługuje wznowienia środka
fazy; powtórzenie ma nowy katalog, te same ID i ustalone kolejności.

Manifest/konfiguracja/środowisko/Conda/pip, źródła i zależności, hashe
wejść, statystyki A każdego ranku i agregacja0..3, candidate_manifest,
cztery modele inference-only, rankowe/globalne predykcje/logity,
błędy bloków B na wejściu nauczyciela, 50 czasów/tryb/GPU po10warmupach,
metryki oraz artifacts.sha256 po zamknięciu logów.

Wykresy PNG/PDF i małe tabelki pod `wykresy/<run-ID>/` przy E003:
accuracy, parametry/rozmiar, czasy perGPU oraz błędy gałęzi. Raport WYNIKI.md
po wykonaniu; duże modele, momenty i logity poza Git. Zewnętrzna kopia
tych artefaktów nadal nie jest skonfigurowana. Odtworzenie wymaga wejść
o hashach ze szkicu/manifestu, całego checkoutu z helperem E002 i Condy attention.

Granice: jedno źródło wag/seed, dane kalibracyjne train, wcześniej używany
dev, jeden poziom kompresji, brak dostrajania i testu końcowego. Nie
gwarantujemy 10% przyspieszenia ani nowości metody przez samą redukcję.
