# attention-aware-compression

Reproducible experiments on attention-aware channel merging and CNN compression.

**Aktualny stan (2026-11-05):** [STATUS_BADAN.md](STATUS_BADAN.md) — jedna strona.
**E031** rozszerzył audyt oficjalnego operatora na 30/70% kompresji:
zgodność operatorów **24/24**, kierunek TILT **11/12**
(CIFAR-100/70%/`repair_bn` tylko 2/4 seedów). **E032** potwierdził późną
wrażliwość CIFAR-100 na 4 seedach, ale nie wyjaśnił różnic zysku TILT.
**E033** nie potwierdził prostej hipotezy „chroń `layer4`”. **E034**
(PreActResNet18, CIFAR-10/100) jest w trakcie treningu — bez wyników.
Najmocniejsze wcześniejsze wyniki alokacji pozostają w E014/E019;
CIFAR-100 nadal jest słaby i nie deklarujemy publikowalności.

## Od czego zacząć

- [STATUS_BADAN.md](STATUS_BADAN.md) — cel, wyniki, następny krok.
- [E033 — interakcja późnych grup](experiments/2026-11-05_E033_interakcja_pozne_grupy/WYNIKI.md)
  (wynik negatywny; domyka prostą hipotezę mechanizmu).
- [E032 — replikacja wrażliwości](experiments/2026-11-04_E032_replikacja_wrazliwosci/WYNIKI.md).
- [E031 — oficjalny operator 30/70%](experiments/2026-11-02_E031_oficjalny_30_70/WYNIKI.md).
- [E029 — oficjalny operator](experiments/2026-11-01_E029_oficjalny_operator/WYNIKI.md)
  i [E030 — Tiny](experiments/2026-11-02_E030_oficjalny_tiny/WYNIKI.md).
- [E034 — PreActResNet18](experiments/2026-11-05_E034_preactresnet18/OPIS.md)
  (Phase 1: trening w toku, bez wyników).
- [E014 — replikacja](experiments/2026-10-27_E014_replikacja_alokacji/WYNIKI.md),
  [E013 — przesiew alokacji](experiments/2026-10-25_E013_alokacja_budzetu_foldingu/WYNIKI.md),
  [E012 — referencyjny folding](experiments/2026-10-24_E012_folding_referencja/WYNIKI.md),
  [E011 — checkpointy](experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/WYNIKI.md).

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
| E015 | alokacja CIFAR-100 | +0,81/+1,96/+1,60 pp; kryterium niepełne — [E015](experiments/2026-10-27_E015_cifar100_alokacja/PODSUMOWANIE.md) |
| E016 | wrażliwość grup i AUTO50 | S1 diagnostyka; S2 negatywny — [E016](experiments/2026-10-27_E016_wrazliwosc_foldingu/PODSUMOWANIE.md) |
| E017 | miary grupowania | wynik negatywny — [E017](experiments/2026-10-29_E017_miary_grupowania/PODSUMOWANIE.md) |
| E018 | replikacja cosine | wynik negatywny — [E018](experiments/2026-10-29_E018_replikacja_cosine/PODSUMOWANIE.md) |
| E019 | alokacja Tiny ImageNet | +3,71/+10,32/+7,60 pp; kryterium spełnione — [E019](experiments/2026-10-29_E019_tinyimagenet_alokacja/PODSUMOWANIE.md) |
| E020 | pilotaż mechanizmu | materiał uzupełniony przez E022 — [E020](experiments/2026-10-30_E020_mechanizm_alokacji) |
| E021 | metryki L1/L2/cosine/Ward | wynik negatywny — [E021](experiments/2026-10-31_E021_metryki_grupowania/PODSUMOWANIE.md) |
| E022 | mechanizm, BN, ANTI-TILT | mechanizm częściowy — [E022](experiments/2026-10-31_E022_mechanizm_alokacji/WYNIKI.md) |
| E023 | łagodne budżety | anulowany przed runem — [E023](experiments/2026-10-31_E023_lagodne_budzety) |
| E024 | search alokacji pełnej sieci | primary negatywny — [E024](experiments/2026-10-31_E024_optymalizacja_alokacji/PODSUMOWANIE.md) |
| E025 | diagnostyka BN | augmentacja odzyskuje teacher — [E025](experiments/2026-11-01_E025_diagnostyka_bn/PODSUMOWANIE.md) |
| E026 | search z poprawioną BN | primary negatywny — [E026](experiments/2026-11-01_E026_alokacja_poprawione_bn/PODSUMOWANIE.md) |
| E027 | disagreement jako cel | +0,935 pp, próg +1 pp niespełniony — [E027](experiments/2026-11-01_E027_diagnostyka_celu/PODSUMOWANIE.md) |
| E028 | transfer na CIFAR-10 | +0,220 pp, brak przewagi — [E028](experiments/2026-11-01_E028_transfer_cifar10/PODSUMOWANIE.md) |
| E029 | oficjalny operator 50% | +3,995/+1,695 pp, dodatni audyt — [E029](experiments/2026-11-01_E029_oficjalny_operator/WYNIKI.md) |
| E030 | oficjalny operator Tiny | +9,408/+9,178 pp, dodatni audyt — [E030](experiments/2026-11-02_E030_oficjalny_tiny/WYNIKI.md) |
| E031 | oficjalny operator 30/70% | zgodność 24/24; kierunek 11/12 — [E031](experiments/2026-11-02_E031_oficjalny_30_70/WYNIKI.md) |
| E032 | replikacja wrażliwości | H1/H2; nie wyjaśnia seed45 — [E032](experiments/2026-11-04_E032_replikacja_wrazliwosci/WYNIKI.md) |
| E033 | interakcja późnych grup | primary negatywny — [E033](experiments/2026-11-05_E033_interakcja_pozne_grupy/WYNIKI.md) |
| E034 | PreActResNet18 (druga architektura) | Phase 1: trening w toku, bez wyników — [OPIS](experiments/2026-11-05_E034_preactresnet18/OPIS.md) |

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

Repozytorium GitHub: [Lidka99/attention-aware-compression](https://github.com/Lidka99/attention-aware-compression). Gałąź robocza: main. Przed uruchomieniem eksperymentu zapisujemy wersję kodu w commicie oraz dokładne wersje danych, wag, środowiska i konfiguracji. Duże artefakty wymagają trwałego magazynu i identyfikatorów; samo wyłączenie z Git ich nie archiwizuje. **Push z tego środowiska jest zablokowany** (brak danych logowania do GitHub); commity lokalne są przygotowane, a `git push` wykonuje użytkownik.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.

## E001 — zakończony model bazowy

[Opis eksperymentu](experiments/2026-09-26_E001_efficientnet_b0_od_zera/OPIS.md), [instrukcja uruchomienia](experiments/2026-09-26_E001_efficientnet_b0_od_zera/URUCHOMIENIE.md) i [wyniki](experiments/2026-09-26_E001_efficientnet_b0_od_zera/WYNIKI.md). Conda attention, 4×RTX 2080 Ti 11 GB, jeden model od losowych wag. CIFAR-100: best dev 66,84% (epoka 177), last dev 66,54%; oficjalnego testu nie oceniano. Audyt 2026-09-29 potwierdził ocenę best.pt; minimum dev NLL przypada na epokę 24, a krzywe wskazują na przeuczenie.
