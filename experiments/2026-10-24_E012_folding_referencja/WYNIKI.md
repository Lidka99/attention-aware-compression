# E012 — folding bez fine-tuningu: adaptacja referencyjnego operatora

Status: **completed**, cztery workery exit0.
Bieg: `runs/E012/20261024T235929Z_cca3cb2_a0b9b3f2`.
Kod/protokół: `cca3cb2`, wersja 1.0. Jeden checkpoint E011 seed42 epoka200.
Kontrola `sha256sum -c artifacts.sha256`: wszystkie wpisy OK.

## Wynik główny

Dev CIFAR-10, 5000 obrazów. `ref_behavior` odtwarza reguły operatora
i odbudowę fc z nowym biasem. To adaptacja, nie bitowa reprodukcja
całego programu autorów ani odtworzenie liczb na ich checkpointach.

| Usunięte kanały per grupa | Parametry | Redukcja parametrów | Top-1 przed BN | Top-1 po BN |
|---|---:|---:|---:|---:|
| 0% (BASE, bypass) | 11 173 962 | 0% | 95,28% | 95,48% |
| 10% | 9 018 358 | 19,29% | 92,04% | 94,20% |
| 30% | 5 459 866 | 51,14% | 44,88% | 89,30% |
| 50% | 2 797 610 | 74,96% | 15,32% | 70,50% |

BN: jeden przebieg 45000 train, bez gradientów i optymalizatora.
Hash uczonych parametrów identyczny przed/po BN; zmieniają się bufory.
Nie należy przedstawiać wyników po BN jako kompresji całkowicie bez danych.
Spadki po BN względem BASE po BN: 1,28 / 6,18 / 24,98 pp.

## Kontrola biasu klasyfikatora

Te same grupy i operator, jedyna różnica: kopiowanie oryginalnego fc.bias.

| Usunięte kanały | preserve_bias przed BN | preserve_bias po BN | Różnica po BN względem ref_behavior |
|---|---:|---:|---:|
| 10% | 92,02% | 94,18% | −0,02 pp |
| 30% | 44,86% | 89,22% | −0,08 pp |
| 50% | 15,40% | 70,48% | −0,02 pp |

Różnica mała opisowo; nie przeprowadzono testu równoważności. Ujawniony
problem biasu nie wyjaśnia dużej szkody kompresji w tym biegu. Nie dobieramy
polityki biasu po tym, która ma wyższą accuracy. Do nowych wariantów
grupowania proponujemy zachowanie biasu i identyczną politykę w referencji.

## Kontrole i ograniczenia

- Baseline odtworzony w każdym workerze: top1 dokładnie 4764/5000,
  NLL zgodna z E011 w tolerancji 1e-5.
- Cztery budżety przypisane GPU0–3; klasteryzacja na CPU. Nie są to
  cztery niezależne powtórzenia naukowe. Czas workerów ~24–58 s.
- Weryfikacja skończonych wag/logitów, niepustych klastrów, budżetów,
  strict reload i bitowo identycznych logitów syntetycznych po serializacji.
- 16 testów CPU, w tym rzeczywista integracja HKMeans, przeszło przed biegiem.
- Dev i train jak w E011; oficjalnego testu nie użyto.
- Implementacja zachowuje mapowanie grup, kolejność, conv sum/mean i BN mean.
  Różnice środowiska, CPU folding, lokalny RNG odbudowy modułów i bypass
  ratio0 są jawne w OPIS.md. Zgodność end-to-end z pełnym oficjalnym kodem
  nie została dowiedziona. Nie ogłaszamy reprodukcji artykułu 1:1.
- Jeden seed, brak porównania alternatywnych miar. To działający punkt
  odniesienia do kolejnego eksperymentu, nie dowód nowości.

## Następny krok

Przygotować osobny protokół porównania grupowania: oryginalne HKMeans/L2
vs warianty przy niezmienionych cechach, operatorze, budżecie i polityce BN.
Główne porównanie do foldingu referencyjnego, nie do RANDOM. Zamrozić
politykę biasu przed nowym biegiem. Jeśli potrzebne twierdzenie o wiernym
odtworzeniu artykułu, najpierw domknąć zgodność end-to-end z jego kodem.
