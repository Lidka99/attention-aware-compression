# Atencja kanałowa i kompresja CNN

Status: przygotowany kod i protokół E001; testy techniczne wykonane. Pełny trening nieuruchomiony. Nowość badawcza niepotwierdzona.

## Od czego zacząć

1. [Cel badań — do czego dążymy](wyjasnienia/CEL_BADAN.md).
2. [Wyjaśnienia krok po kroku](wyjasnienia/WYJASNIENIA_KROK_PO_KROKU.md).
3. [Zasady organizacji przyszłych eksperymentów](experiments/README.md).

## Struktura

- `wyjasnienia/`: cel badań oraz proste objaśnienia, przekazywane też w rozmowie.
- `experiments/`: w przyszłości osobny folder dla każdego eksperymentu.
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

Repozytorium jest lokalne; zdalna kopia nie jest skonfigurowana. Przed uruchomieniem eksperymentu zapiszemy wersję kodu w commicie oraz dokładne wersje danych, wag, środowiska i konfiguracji. Duże artefakty wymagają trwałego magazynu i identyfikatorów; samo wyłączenie z Git ich nie archiwizuje.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.

## E001 — przygotowany pierwszy pilot

[Opis eksperymentu](experiments/2026-09-26_E001_efficientnet_b0_od_zera/OPIS.md) i [instrukcja uruchomienia](experiments/2026-09-26_E001_efficientnet_b0_od_zera/URUCHOMIENIE.md). Używamy Condy attention. Dostępne 4×RTX 2080 Ti 11 GB. Trening od losowych wag; CIFAR-100 jest propozycją roboczą, pełnego treningu nie uruchomiono.
