# 2026-11-22_E044_l06_tiny — czy zamrożony L06 przenosi się na Tiny ImageNet?

**Zakres:** przeniesienie jednej reguły alokacji z CIFAR na istniejący
ResNet18/Tiny64, pod oboma pułapami UNIFORM, bez fine-tuningu. Obecny etap
to protokół i kontrola CPU przed oceną jakości. [Odtworzenie](URUCHOMIENIE.md).

## Modele i dane

- Własne końcowe `last.pt` E019, seedy 42–45, torchvision ResNet18, stem
  7×7/stride 2 + maxpool, wejście 64×64, klasyfikator do 200 klas.
- Ocena L06 na oficjalnym `val` Tiny ImageNet: 10 000 obrazów w istniejącym
  `data/tiny-imagenet/cache_val.npz`. **To ten sam val, na którym E035
  porównał TILT i UNIFORM.** Tiny nie był użyty do wyboru L06 w E042, więc
  pomiar odpowiada na pytanie o transfer profilu poza konfiguracje wyboru.
  Raportujemy go jako transfer na inną konfigurację **z wcześniej użytym
  zbiorem oceny**. Nie jest to nowy holdout projektu ani izolacja wpływu
  samej rozdzielczości, danych lub stema.
- Rekalibracja na tej samej liście A4096 dla Tiny z E035, wyłącznie wybrane
  indeksy oficjalnego train Tiny. Wspólne widoki E030: `RandomResizedCrop(64,
  scale=(0.4,1.0))` i flip, seed dla obrazu `20261103 + indeks`, normalizacja
  stałymi z train E019. Sprawdzamy hash widoków A względem E035. Reset BN +
  CMA, 32 batche ×128, zmieniają się tylko bufory BN; `eval()` przy ocenie.
  Na val nie ma augmentacji ani aktualizacji BN. Taka sama procedura dla
  wszystkich metod, w tym pełnego modelu.

## Zamrożona geometria

U = historyczny UNIFORM danego budżetu (b50 i b30, nominalna **redukcja P**)
z E035. Pułapy to jego rzeczywiste P/MAC z E035/E037 na wejściu 64×64.
T-proj = historyczny TILT odpowiedniego budżetu z E035, rzutowany pod oba
pułapy U. Dla L06 bierzemy uporządkowane `offsets_pp` z zamrożonego E043,
zgodne z propozycją E042, i tworzymy kandydat całkowitoliczbowy
`clip(rint(U + offsets_pp * n / 100), n//10, 9*n//10)`. Stosujemy dokładnie
SLSQP z E042 (start U, cel obu kosztów 0,9995 U, odległość w `k/n`),
`rint` i deterministyczną korektę co najwyżej dwóch grup. Zmieniamy tylko
rachunek kosztów: Tiny ma inny stem i inną powierzchnię map cech niż CIFAR.
Kolejność, osie i rozmiary 12 grup potwierdza mapa operatora.

Sprawdzamy całkowite szerokości, odrębność profili, koszty Conv/Linear na
obrazie, a na CPU na seedzie 42 również rzeczywiste P, MAC z hooków na zerowym
wejściu, poprawność klastrów i biasu FC dla wszystkich sześciu modeli.
Oba koszty każdego profilu kompresji muszą być w **[99,9%, 100%] U**.
Pełny model jako punkt odniesienia jest poza pułapami. Jak w E030: upstream
CLFM `694691556e8314a835178a741938dd3d4d80c171` na naszym Tiny64, nie na
klasie modelu CIFAR. Na GPU weryfikujemy koszty ponownie dla każdego seeda.

## Werdykt i raportowanie

- **Główne:** sparowana różnica top‑1 L06−U po CMA w b50. Dodatni kierunek
  transferu w badanej konfiguracji: średnia >0 i ≥3 z 4 różnic >0, na
  niezaokrąglonych licznikach trafień. Kryterium kierunku jest opisowe;
  nie przenosimy umownego progu +1 pp z E043 jako warunku dla innego zadania.
- **Wtórne:** L06−U w b30, L06−T-proj, T-proj−U oraz strata do pełnego
  modelu dla obu budżetów. b30 nie zmienia werdyktu b50. Pokazujemy pełne
  wyniki per seed, średnią, SD próbki, liczbę dodatnich/remisów/ujemnych,
  trafność bezwzględną, P/MAC i rzeczywistą redukcję P. Bez dobierania
  budżetu, transformacji, referencji lub przesunięć na podstawie val.
- Zapisujemy binarną poprawność predykcji dla każdego obrazu i metody.
  Dla głównego kontrastu raportujemy dodatkowo 95% percentylowy bootstrap
  sparowany (10 000 losowań, PCG64 `20261122`, stratyfikacja po klasach;
  ten sam dobór obrazów w danym losowaniu dla wszystkich modeli i seedów).
  To niepewność obrazu warunkowo na czterech checkpointach i zadaniu,
  nie przedział zmienności między treningami. Jeśli przedział obejmuje 0,
  opisujemy dodatni kierunek jako niepewny na poziomie obrazów.

Późniejsza ocena: Conda `attention`, 4 GPU, seed 42/43/44/45→GPU 0/1/2/3.
Jeden proces na kartę, 7 modeli na seed (pełny + 3×b50 + 3×b30), **28 ocen**,
batch 128, FP32, deterministycznie, TF32 off. Wspólna bramka kosztów, stanów
i CMA przed punktacją; audyt przypisania GPU, hashy i 4 kodów zakończenia.
