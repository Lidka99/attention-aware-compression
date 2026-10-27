# 2026-10-25_E013_alokacja_budzetu_foldingu

Status: **protokół 0.1 frozen przed biegiem**. Status wykonania w manifeście.
Wejście: E011 seed42. Operator: E012. Oficjalny test nietknięty.

## Dlaczego zmieniliśmy pytanie

Poprzedni szkic E013 porównywał L1/L2/cosine/korelację jako miary grupowania.
Przegląd literatury pokazał, że samo „grupowanie według przewidywanej szkody”
nie jest bezpieczną nowością:

- MTZ (He, Zhou, Thiele, NeurIPS 2018) używa Hessianu i funkcjonalnej
  różnicy neuronów do wyboru par do scalenia.
- Intra-Fusion (Theus et al., ICLR 2024) używa Optimal Transport i metryki
  ważności, żeby wchłonąć odrzucane neurony w zachowane; raportuje poprawę
  bez fine-tuningu i lepsze zachowanie wyjścia.
- Model Folding (Wang et al., 2025) używa k-means na wagach i korekty
  statystyk BN; Cut Less, Fold More (Saukh et al., 2026) porównuje folding
  z MAG1/MAG2 i używa **jednolitego współczynnika per warstwa**.

Autorzy Model Folding wprost piszą w ograniczeniach:
> „it does not optimize sparsity levels per layer, leaving this for future work.”

To jest nasza luka: **alokacja globalnego budżetu kompresji między sprzężone
grupy foldingu**. Nie twierdzimy, że sama miara podobieństwa jest nowa.

## Pytanie E013

Przy **tym samym łącznym budżecie parametrów** i bez fine-tuningu:

> Czy nierówna alokacja współczynników kompresji między grupy kanałów daje
> lepszą jakość po korekcie BN niż jednolity folding i niż MAG1/MAG2?

E013 jest **przesiewem**, nie gotową metodą. Ma odpowiedzieć, czy w ogóle
istnieje sygnał, zanim zbudujemy predykcyjną alokację.

## Warunki

- Checkpoint: E011 seed42 epoka200, hash jak w E012.
- Model: ResNet18/CIFAR-10, 11 173 962 parametry.
- Globalne budżety: redukcja **30%, 50%, 70%** wszystkich parametrów.
  Budżet jest liczony na całym modelu, nie per warstwa. Dokładna liczba
  parametrów po foldingu raportowana; wszystkie warianty w tolerancji
  ≤0,5% celu (najlepiej ≤0,1%).
- Podział danych i BN: jak E012 — 45 000 train, 5 000 dev, jeden przebieg
  BN, seed 20261024. Bez gradientów i optymalizatora. Test nietknięty.
- Cztery GPU: każdy worker dostaje jeden wariant alokacji; nie są to seedy.

## Warianty alokacji

Dla grup w kolejności depth: `g = 0..11`, pozycja znormalizowana
`p_g = 2g/11 - 1`.

1. **UNIFORM** — wszystkie grupy mają ten sam keep ratio `r`.
2. **TILT−0.30 / TILT−0.15 / TILT+0.15 / TILT+0.30** —
   `r_g = clip(base + tilt·p_g, 0.10, 0.90)`. `base` dobierany tak, by
   trafić w budżet. To tania hipoteza „więcej/ mniej kompresji na początku”.
3. **RANDOM×4** — losowe odchylenia od `base` o średniej zero, z
   zamrożonym seedem; kontrola, czy ewentualna poprawa nie wynika z przypadku.
4. **MAG1 / MAG2** — magnitude pruning po kanałach wyjściowych, L1/L2,
   jednolity współczynnik per warstwa dobrany do tego samego globalnego
   budżetu. To bezpośrednie porównanie z artykułem.

Wszystkie warianty dostają **identyczną** korektę BN i eval.

## Miary

- Dev top-1, NLL, liczba poprawnych.
- Zgodność predykcji z modelem źródłowym (prediction agreement).
- MAE/L1 i RMSE/L2 różnicy logitów względem źródła.
- KL(softmax(source) || softmax(candidate)) na dev.
- Parametry, redukcja, MACs/FLOPs Conv+Linear, czas foldingu i BN.
- Wyniki przed i po BN; główny wynik po BN.

## Kryterium sygnału

E013 nie ma „zwycięzcy” bez replikacji. Sygnałem do dalszej pracy jest:
- co najmniej jeden TILT lub wariant alokacji poprawia top-1 po BN o
  **≥0,5 pp** względem UNIFORM przy tym samym budżecie, oraz
- poprawa nie jest wyjaśniona przez RANDOM (średnia/rozrzut kontroli), oraz
- wariant nie jest gorszy od MAG1/MAG2 o więcej niż 0,5 pp.

Jeśli wszystkie alokacje są w szumie, zamykamy ten kierunek i wracamy
do innych osi (np. ważone scalanie, inne operatory), zamiast rozbudowywać
metodę bez sygnału.

## Ograniczenia

Jeden checkpoint i jeden seed; dev służy do screeningu, nie do wyboru
ostatecznej metody. Wynik dodatni wymaga E014: wielu checkpointów/seeds,
zamrożonej alokacji bez strojenia na dev, porównania z FOLD/MAG1/MAG2
i analizy kosztu. Nie obiecujemy publikacji z E013.