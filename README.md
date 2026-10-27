# attention-aware-compression

Reproducible experiments on attention-aware channel merging and CNN compression.

**Aktualny stan:** [STATUS_BADAN.md](STATUS_BADAN.md) — jedna strona.
Najważniejszy wynik: E014 potwierdził na 4 seedach, że nierówna alokacja
budżetu foldingu bije jednolity folding o **+0,87 / +3,77 / +11,27 pp**
przy 30/50/70% redukcji parametrów, bez fine-tuningu.

## Od czego zacząć

- [STATUS_BADAN.md](STATUS_BADAN.md) — cel, wyniki, następny krok.
- [E014 — replikacja](experiments/2026-10-27_E014_replikacja_alokacji/WYNIKI.md),
  [PODSUMOWANIE](experiments/2026-10-27_E014_replikacja_alokacji/PODSUMOWANIE.md).
- [E013 — przesiew alokacji](experiments/2026-10-25_E013_alokacja_budzetu_foldingu/WYNIKI.md).
- [E012 — referencyjny folding](experiments/2026-10-24_E012_folding_referencja/WYNIKI.md).
- [E011 — checkpointy](experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/WYNIKI.md).

**Bieżący kierunek:** folding bez fine-tuningu; aktualny wynik i następny krok w [STATUS_BADAN.md](STATUS_BADAN.md).

## Eksperymenty — skróty

| ID | Temat | Podsumowanie |
|---|---|---|
| E001 | baseline EfficientNet+SE / CIFAR-100 | [E001](experiments/2026-09-26_E001_efficientnet_b0_od_zera/PODSUMOWANIE.md) |
| E002 | obserwacja SE | [E002](experiments/2026-10-01_E002_obserwacja_se/PODSUMOWANIE.md) |
| E003 | kompresja 10% | [E003](experiments/2026-10-09_E003_kompresja_10proc_parametrow/PODSUMOWANIE.md) |
| E004 | równe dostrojenie | [E004](experiments/2026-10-10_E004_wspolne_dostrajanie/PODSUMOWANIE.md) |
| E005 | baseline 96×96 | [E005](experiments/2026-10-10_E005_tiny_imagenet_od_zera/PODSUMOWANIE.md) |
| E006 | bramki SE | [E006](experiments/2026-10-11_E006_kryterium_bramkowe_predykcja/PODSUMOWANIE.md) |
| E007 | baseline 224×224 | [E007](experiments/2026-10-11_E007_baseline_224/PODSUMOWANIE.md) |
| E008 | kompresja 30% + 60 epok | [E008](experiments/2026-10-12_E008_kompresja_224/PODSUMOWANIE.md) |
| E009 | naprawa 120 epok | [E009](experiments/2026-10-12_E009_naprawa_120/PODSUMOWANIE.md) |
| E010 | benchmark kryteriów | [E010](experiments/2026-10-18_E010_benchmark_kryteriow/PODSUMOWANIE.md) |
| E011 | checkpointy ResNet18 | [E011](experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/PODSUMOWANIE.md) |
| E012 | folding referencyjny | [E012](experiments/2026-10-24_E012_folding_referencja/PODSUMOWANIE.md) |
| E013 | alokacja budżetu | [E013](experiments/2026-10-25_E013_alokacja_budzetu_foldingu/PODSUMOWANIE.md) |
| E014 | replikacja alokacji | [E014](experiments/2026-10-27_E014_replikacja_alokacji/PODSUMOWANIE.md) |

Pełne raporty: `WYNIKI.md` i `OPIS.md` w folderach eksperymentów.
[Sposób dokumentowania](docs/DOKUMENTOWANIE.md).

## Struktura

- `wyjasnienia/`: cel badań oraz proste objaśnienia, przekazywane też w rozmowie.
- `experiments/`: osobny folder dla każdego eksperymentu, wraz z jego audytami i wykresami.
- `experiments/_szablon/`: pusty wzór opisu eksperymentu.
- `experiments/_szkice/`: zachowane wcześniejsze propozycje, niezatwierdzone do wykonania.
- `configs/`: wcześniejsze szkice konfiguracji; nie są gotowe do uruchomienia.
- `literature/`: literatura i analiza nowości.
- `docs/`: zasady odtwarzania, zasoby, decyzje i kontekst.
- `reports/`: miejsce na przyszłe zestawienia przekrojowe.
- `runs/`: lokalne artefakty uruchomień, wyłączone z Git.
- `data/`, `checkpoints/`: lokalne duże zasoby, wyłączone z Git.

[Wcześniejszy plan badawczy](PLAN_DZIALANIA.md) pozostaje materiałem roboczym. Ustalamy kolejne kroki pojedynczo. [Zasady reprodukowalności](docs/REPRODUCIBILITY.md) wykorzystamy podczas przygotowywania konkretnego eksperymentu.

## Git i odtwarzanie

Repozytorium GitHub: [Lidka99/attention-aware-compression](https://github.com/Lidka99/attention-aware-compression). Gałąź robocza: main. Przed uruchomieniem eksperymentu zapiszemy wersję kodu w commicie oraz dokładne wersje danych, wag, środowiska i konfiguracji. Duże artefakty wymagają trwałego magazynu i identyfikatorów; samo wyłączenie z Git ich nie archiwizuje.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.

## E001 — zakończony model bazowy

[Opis eksperymentu](experiments/2026-09-26_E001_efficientnet_b0_od_zera/OPIS.md), [instrukcja uruchomienia](experiments/2026-09-26_E001_efficientnet_b0_od_zera/URUCHOMIENIE.md) i [wyniki](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md). Conda attention, 4×RTX 2080 Ti 11 GB, jeden model od losowych wag. CIFAR-100: best dev 66,84% (epoka 177), last dev 66,54%; oficjalnego testu nie oceniano. Audyt 2026-09-29 potwierdził ocenę best.pt; minimum dev NLL przypada na epokę 24, a krzywe wskazują na przeuczenie.
