# 2026-10-27_E014_replikacja_alokacji

Status: **protokół 0.1 frozen przed biegiem**. Wejście: E011 seedy42–45.
Operator: E012/E013. Oficjalny test nietknięty.

## Cel

E013 (jeden checkpoint, seed42) dał bardzo duży sygnał: przy 50% redukcji
parametrów alokacja **TILT+0.30** osiągnęła 93,32% po BN wobec 88,98% dla
UNIFORM; przy 70% — 87,86% wobec 77,16%. Był to wynik jednego modelu.

E014 sprawdza, czy ten efekt **powtarza się na czterech niezależnych
checkpointach** E011 (seedy42–45), przy tym samym budżecie, operatorze,
korekcie BN i bez fine-tuningu. To replikacja, nie dobieranie nowego wariantu.

## Zamrożone warianty

Cztery warianty na każdym z trzech budżetów (30/50/70% redukcji parametrów):

1. **UNIFORM** — ten sam keep ratio dla wszystkich 12 grup (baseline).
2. **TILT+0.30** — `r_g = clip(base + 0.30·p_g, 0.10, 0.90)`,
   `p_g = 2g/11 - 1`; wczesne grupy mniej skompresowane, późne bardziej.
   `base` dobierany do budżetu; pojedyncze korekty szerokości tylko po to,
   by dojść do budżetu (repair), nie strojenie na dev.
3. **TILT+0.15** — łagodniejsza wersja tego samego kierunku.
4. **MAG2** — magnitude pruning L2 z artykułu, uniform per warstwa,
   dopasowany do tego samego globalnego budżetu.

Wszystkie warianty: BN na 45 000 train, seed20261024, bez gradientów;
ewaluacja na 5 000 dev; te same transformacje i batch128.

## Kryterium replikacji (ustalone przed biegiem)

Dla **TILT+0.30 vs UNIFORM**:

- średnia różnica top-1 po BN > 0 na **każdym** budżecie,
- przy 50% i 70%: średnia różnica **≥ +2,0 pp** oraz dodatnia na **≥3/4**
  seedach,
- przy 30%: średnia > 0 oraz dodatnia na ≥3/4 seedach (efekt może być mały),
- TILT+0.30 nie gorszy od MAG2 o więcej niż 0,5 pp średnio.

Dodatkowo raportujemy TILT+0.15, NLL, agreement, MAE/RMSE logitów, KL
i czasy. Nie zmieniamy progu po wyniku. Jeśli replikacja się nie uda,
raportujemy wynik negatywny i nie budujemy metody na jednym seedzie.

## Ograniczenia

Cztery checkpoints z jednego przepisu treningowego; dev nie jest niezależnym
testem; E013 użył tego samego dev do screeningu, więc E014 jest replikacją
na nowych wagach, nie na nowym zbiorze. Brak ImageNet/CIFAR-100. Nie
twierdzimy jeszcze, że metoda bije artykuł 1:1; porównanie jest adaptacją
ich operatora przy dopasowanym globalnym budżecie.

## Uruchomienie

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B experiments/2026-10-27_E014_replikacja_alokacji/launch_replication.py
```

Launcher wymaga czterech wolnych GPU, 2 GiB dysku i commitowanego folderu.
Każdy seed ma osobny manifest i katalog wyników w `runs/E014/<UTC>_<git>_<id>/seeds/seed_XX`.
W sumie 48 workerów: 4 seedy × 3 budżety × 4 warianty, fale po 4 procesy.