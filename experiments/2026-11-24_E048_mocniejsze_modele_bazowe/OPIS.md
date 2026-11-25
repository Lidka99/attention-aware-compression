# 2026-11-24_E048_mocniejsze_modele_bazowe — mocniejsze ResNet18 na CIFAR-100 i Tiny ImageNet

**Decyzja 2026-11-24:** użytkowniczka zatwierdziła nowy trening od losowej inicjalizacji i ponowne porównania kompresji na obu trudniejszych zbiorach. Hipoteza E048 dotyczy trafności **modeli pełnych**. Zysk po foldingu sprawdzi osobny E049, na wszystkich nowych checkpointach.

## Protokół zamrażany przed uruchomieniem

- CIFAR-100: ta sama architektura i wejście 32×32 co E015. Tiny ImageNet: ResNet18, wejście 64×64 i 200 klas; stem Conv 3×3/stride 2, bez maxpool, zachowuje mapę 32×32 zamiast 16×16 przed pierwszym etapem. Oczekiwane P/MAC Tiny: 11 271 432/555 520 000 wobec starych 11 279 112/148 148 224. Ta zmiana zwiększa koszt pełnego Tiny; porównanie nowa–stara receptura na Tiny nie rozdziela efektu stema od treningu ani nie dowodzi lepszej efektywności względem starego stema.
- Te same train/dev i normalizacja train-only co E015/E019: 40 000/5 000 CIFAR-100, 90 000/10 000 Tiny. W E048 nie punktujemy oficjalnego testu, rezerwy CIFAR-100, oficjalnego val Tiny ani ImageNetV2. Historyczne testy pozostają wcześniej użytymi benchmarkami.
- **Jedna receptura**, cztery nowe treningi na zbiór, seedy 42–45. Losowa inicjalizacja i SGD Nesterov, momentum 0,9, WD 5e-4; 240 epok, LR 0,1, 5 epok liniowego warmupu i cosine do 0,001. FP16 autocast + GradScaler w treningu, ocena FP32. Batch 128 na GPU i 128 **na niezależny model**, bez akumulacji. Jeden seed na każdej z czterech RTX 2080 Ti, fala CIFAR-100, następnie fala Tiny na tych samych czterech GPU.
- CIFAR-100: crop z odbiciem + flip. Tiny: crop-resize z łagodniejszym dolnym ograniczeniem skali 0,65 (E019: 0,4) + flip. Na obu: RandAugment (2 operacje, magnitude 7), MixUp (α=0,2) lub CutMix (α=1) z równym prawdopodobieństwem dla 75% batchy przed ostatnimi 10 epokami; CE z label smoothing 0,05. W ostatnich 10 epokach brak mieszania batchy, nadal augmentacja obrazów. Inspiracja: [Wightman i in., §3 i tab. 2](https://www.alphaxiv.org/abs/2110.00476v1?utm_source=alphaxiv-mcp&utm_medium=mcp); lokalna adaptacja do dwóch małych zbiorów, nie reprodukcja A1–A3.
- Oceniamy **ostatnią epokę** każdego seeda. Top‑1 i NLL na starym dev bez augmentacji, przed rekalibracją BN, porównane ze starymi pełnymi modelami na tym samym dev. Dodatkowo wynik na stałej zbalansowanej próbce train (20 obrazów na klasę, seed 20261124), bez mieszania i augmentacji, opisuje lukę generalizacji. Smoke sprawdza wyłącznie wykonalność i nie wybiera receptury.

| Zbiór | Historyczny top‑1 dev (seedy 42–45) | Praktyczne kryterium E048 |
|---|---|---|
| CIFAR-100 | 74,94 / 75,10 / 75,04 / 75,86% (75,235%) | średnia ≥77,50% i ≥3/4 popraw względem sparowanego historycznego dev |
| Tiny ImageNet | 51,08 / 50,60 / 50,65 / 50,95% (50,82%) | średnia ≥59,00% i ≥3/4 popraw względem sparowanego historycznego dev |

Źródła wartości odniesienia: [E015](../2026-10-27_E015_cifar100_alokacja/WYNIKI.md), PRE dev zapisany w E035 dla tych samych checkpointów, i [E019](../2026-10-29_E019_tinyimagenet_alokacja/WYNIKI.md). Kryteria są praktyczne, nie są testami istotności. 4 seedy nie stanowią czterech prób danych.

## Związek z wkładem artykułu

Następny pomiar stosuje **ten sam operator FOLD i tę samą rekalibrację BN** do modeli z E048. Praktyczny test: czy jeden zamrożony profil (L06) przy nominalnym b10 zmniejsza stratę względem równomiernej alokacji przy obu pułapach UNIFORM na **każdym** z trudnych zbiorów? Raportujemy również pełny model, rzeczywiste P/MAC, wyniki per seed i czy strata do pełnego jest ≤3 pp. TILT przy porównywalnej liczbie P stanowi osobną kontrolę, nie porównanie przy równych MAC. Protokół oraz geometrie E049 będą zamrożone przed pierwszą punktacją foldingu. Kryterium E048 nie zmienia się po treningu. Dane z audytowanych, ukończonych treningów są podstawą E049 także przy niespełnieniu bramki jakości, z jawnym werdyktem E048.

Uruchomienie i odtworzenie: [URUCHOMIENIE.md](URUCHOMIENIE.md). Stan, hashe wejść/wyjść, czasy GPU i wartości top-1: manifest oraz `audit.json` pod `runs/E048/`.
