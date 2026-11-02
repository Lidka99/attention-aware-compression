# E019 — wyniki: alokacja TILT na Tiny ImageNet

## Uruchomienie
- Trening: `runs/E019/20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7`,
  **4/4 seedy completed**, 200 epok, cztery GPU 0–3, ok. 150 min.
- Folding: `runs/E019/fold_20261030T220139Z_cb9ffaf_51bc405f`,
  **48/48 zadań completed**, cztery GPU, exit 0.
- Analiza: `runs/E019/analizy/20261030T222400Z_final/`.
- Bez treningu wag po foldingu; oficjalny val/test i rezerwa nietknięte.
- Wcześniejszy trening `20261030T000316Z...` przerwano po <1 epoce
  (normalizacja z dev/rezerwy); zachowany, bez wyników naukowych.

## Baseline (dev top-1, po treningu)

| Seed | Top-1 |
|---|---:|
| 42 | 51,08% |
| 43 | 50,60% |
| 44 | 50,65% |
| 45 | 50,95% |
| **Średnia** | **50,82%** |

Model: ResNet18 standard stem, 64×64, 200 epok; train top-1 ok. 96%
(przeuczenie). To punkt odniesienia przed kompresją.

## Wynik po foldingu (dev top-1, średnia ± SD z 4 seedów)

| Budżet parametrów | UNIFORM | TILT+0.30 | TILT+0.15 | MAG2 |
|---:|---:|---:|---:|---:|
| 30% | 34,41 ± 0,95 | **38,12 ± 1,29** | 38,20 ± 1,03 | 26,34 ± 1,60 |
| 50% | 19,77 ± 1,70 | **30,08 ± 0,72** | 27,36 ± 0,83 | 10,34 ± 1,17 |
| 70% | 8,54 ± 0,65 | **16,14 ± 1,31** | 13,05 ± 0,98 | 2,73 ± 0,35 |

## TILT+0.30 vs UNIFORM — różnice per seed [pp]

| Budżet | seed42 | seed43 | seed44 | seed45 | średnia | SD | dodatnie |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | +2,10 | +2,92 | +5,56 | +4,25 | **+3,71** | 1,52 | 4/4 |
| 50% | +11,90 | +9,58 | +10,27 | +9,51 | **+10,32** | 1,11 | 4/4 |
| 70% | +6,30 | +6,90 | +9,05 | +8,14 | **+7,60** | 1,24 | 4/4 |

**12/12 par seed×budżet dodatnich.** Metryki wtórne mają zgodny kierunek:
NLL, MAE, RMSE i KL niższe, agreement wyższy na wszystkich 12 parach.

## Budżety
Wspólny target = faktyczna liczba parametrów UNIFORM. TILT różni się od
niego o **−137 / −39 / −85** parametrów (30/50/70%), czyli maks. 0,002%.
Faktyczna redukcja: 30,445 / 50,146 / 70,149%.
MAG2 ma dokładnie tyle parametrów co UNIFORM.

## Kryterium (zamrożone w OPIS.md)
TILT+0.30 ≥ **+1,0 pp** nad UNIFORM przy 50% i 70% oraz dodatni na ≥3/4 seedów.

- 50%: **+10,32 pp**, 4/4 dodatnich.
- 70%: **+7,60 pp**, 4/4 dodatnich.
- **Kryterium SPEŁNIONE.**

## Interpretacja
1. **Trzeci zbiór potwierdza kierunek E014:** depth-aware alokacja
   (mniej kompresji wczesnych grup, więcej późnych) poprawia folding
   bez fine-tuningu, tym razem także poza CIFAR.
2. Efekt jest duży i spójny między seedami: SD różnic 1,1–1,5 pp.
3. **TILT+0.15 jest lepszy tylko przy 30%** (+3,79 vs +3,71 pp, w granicach
   szumu); przy 50/70% wyraźnie gorszy. To wspiera +0.30 jako primary
   w reżimie średniej/wysokiej kompresji.
4. **MAG2 jest znacznie gorszy** (np. 10,34% vs 30,08% przy 50%).
5. **Ważne ograniczenie:** bezwzględna jakość po foldingu jest niska
   (UNIFORM 34,4/19,8/8,5%, TILT 38,1/30,1/16,1%). Folding na tym modelu
   i budżetach jest bardzo niszczący; wynik pokazuje przewagę względną,
   nie „dobrą” kompresję w sensie aplikacyjnym.
6. Wynik E015 (CIFAR-100) pozostaje słabszy: +0,81/+1,96/+1,60 pp
   i niespełnione kryterium. Obraz całości: CIFAR-10 mocno, Tiny ImageNet
   mocno, CIFAR-100 słabo.

## Ograniczenia
- Jedna architektura (ResNet18 standard stem), jeden rozmiar 64×64,
  jeden przepis treningu, cztery checkpointy.
- Ocena tylko na dev; brak niezależnego testu.
- Adaptacja operatora E012, nie oficjalny kod 1:1.
- Wysokie przeuczenie baseline (train ok. 96%, dev 50,8%); mechanizm
  wpływu tego przeuczenia na folding nie jest zbadany.

## Następny krok
Wynik domyka trzeci zbiór dla najmocniejszej tezy projektu. Kolejne kroki
do decyzji: (a) analiza mechanizmu (dlaczego wczesne grupy są wrażliwsze
i jak BN odzyskuje szkodę), (b) sprawdzenie hipotezy na drugiej architekturze
(np. PreActResNet18), (c) dopiero potem decyzja o publikacji.
