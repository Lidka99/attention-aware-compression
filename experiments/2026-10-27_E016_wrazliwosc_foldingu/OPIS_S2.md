# E016 — etap S2: automatyczna alokacja 50% (protokół 2.0)

Status: **protokół zamrożony przed uruchomieniem oceny S2**.
Data utworzenia: 2026-10-27 (UTC). Ten plik jest nowym, osobnym protokołem
etapu S2. Nie zmienia `OPIS.md`, `PLAN.md`, `WYNIKI.md` ani snapshotów S1.

## Historia rewizji protokołu (przed oceną S2)

- **2.0** — pierwotna reguła: stała baza 0.5 i siatka `beta 0..1.5`.
- **2.1** — **przed jakąkolwiek oceną dev w S2**, po CPU preflight 2.0:
  stała baza 0.5 nie kontrolowała globalnej liczby parametrów, a
  `repair_budget` musiał zmieniać ratio grup (np. 389 kroków), więc AUTO
  przestawał odpowiadać formule. Rewizja zamraża `beta=0.30` (ta sama siła
  co TILT+0.30) i rozwiązuje bazę `a` do targetu UNIFORM na rzeczywistych
  modelach; repair tylko przy luce > 0,5%. Nie użyto dev accuracy ani
  żadnej metryki oceny — decyzja opiera się wyłącznie na liczbach
  parametrów i primary predyktorze S1.

## Cel etapu

Sprawdzić, czy **automatyczna** alokacja globalnego budżetu 50% — oparta
wyłącznie na primary predyktorze zapisanym w S1
(`weight_reconstruction_rel_error`) — jest lepsza od UNIFORM i nie odstaje
od zamrożonej stałej reguły TILT+0.30 na istniejących checkpointach seed42
CIFAR-10 i CIFAR-100. To etap rozwojowy, nie replikacja i nie publikacja.

## Zakres i wejścia

- CIFAR-10/ResNet18: checkpoint E011 seed42
  (`runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012/seed_42/last.pt`,
  11 173 962 parametrów, dev top-1 95,28%).
- CIFAR-100/ResNet18: checkpoint E015 seed42
  (`runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12/seed_42/last.pt`,
  11 220 132 parametrów, dev top-1 74,94%).
- Splity i hashe: `configs/stage2.json`; przed każdym workerem sprawdzane są
  SHA-256 checkpointu i splitu.
- Tylko train/dev. **Oficjalny test CIFAR-10/CIFAR-100 i rezerwa 5 000
  obrazów CIFAR-100 pozostają nietknięte.**
- Brak nowego treningu, brak fine-tuningu, brak gradientów. Korekta BN jest
  jedyną adaptacją i używa pełnego train bez etykiet w optymalizacji.

## Operator, grupy i fold

Operator jest importowany bez zmian z tego repo:

- `experiments/2026-10-24_E012_folding_referencja/folding.py`,
- `experiments/2026-10-24_E012_folding_referencja/evaluate.py`,
- `experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py`,
- dane i model: E011 `train.py` oraz E015 `data_cifar100.py`,
- helpery S1: `e016_common.py` (bez modyfikacji).

Używamy 12 grup E012 w kolejności depth. Fold seed `42`, `preserve_fc_bias=True`,
`deterministic=False` (HKMeans z `random_state=42`). BN: pełny train
(45 000 dla CIFAR-10, 40 000 dla CIFAR-100), seed `20261024`, batch 128,
te same transformacje i ten sam loader dla wszystkich metod. Ewaluacja:
5 000 dev, batch 128, bez augmentacji losowej.

## Budżet 50% i wspólny target

Dla każdego datasetu najpierw liczymy:

```
N = liczba parametrów checkpointu
nominal_target = int(round(0.5 * N))
```

UNIFORM jest rozwiązywany przez
`folding_budget.solve_base_for_budget(model, 'uniform', nominal_target)`.
Jego **faktyczna** liczba parametrów `P_uniform` (zrealizowany model po
foldingu, nie suma oszczędności per grupa) staje się wspólnym targetem
TILT+0.30, TILT+0.15 i AUTO50. Żaden wariant nie może przekroczyć
`P_uniform`.

Tolerancja: realized params wariantu muszą spełniać
`0 <= P_uniform - P_variant <= 0.5% * P_uniform`. Przekroczenie targetu
jest błędem krytycznym. Faktyczna redukcja względem `N` jest raportowana
osobno od nominalnej; nie renormalizujemy budżetu po wyniku.

## Predyktor primary i reguła AUTO50

Primary predyktor jest zamrożony w S1: `weight_reconstruction_rel_error`.
Dla grupy `g` i poziomu S1 `l ∈ {0.25, 0.50, 0.75, 0.90}` niech `e_{g,l}`
będzie tym błędem z analizy S1. AUTO używa **tylko** tych wartości; nie
używa dev accuracy, dev NLL ani żadnej metryki oceny.

Aby poziom kompresji nie dominował porównania grup, standaryzujemy błąd
**w obrębie poziomu**, a dopiero potem uśredniamy po poziomach:

```
mu_l = mean_g e_{g,l}
sd_l = sqrt(mean_g (e_{g,l} - mu_l)^2)        # ddof=0
z_{g,l} = (e_{g,l} - mu_l) / sd_l
s_g = (1/4) * sum_l z_{g,l}                    # surowy wynik wrażliwości
z_g = (s_g - mean_h s_h) / sd_h(s_h)           # ddof=0, standaryzacja grup
```

Większy `z_g` oznacza większy błąd rekonstrukcji, więc większą wrażliwość.
Alokacja redukcji ma **stałą siłę tiltu** `beta = 0.30` (taką samą jak
zamrożona reguła TILT+0.30) i **bazę** `a` rozwiązywaną do wspólnego
targetu:

```
r_g(a) = clip(a - 0.30 * z_g, 0.10, 0.90)
```

Znak minus jest zamrożony: grupy o większym błędzie rekonstrukcji są mniej
kompresowane. `r_g` jest **docelowym** ratio usuniętych kanałów w grupie `g`.
Operator E012 realizuje szerokość `width = max(int(n*(1-r)), 1)`, więc
faktyczne ratio `1 - width/n` może różnić się od docelowego o najwyżej `1/n`
(maks. 1/64 = 0,015625). Raportujemy osobno `target_ratios` i realizowane
`ratios`; kontrola docelowych ratios jest ostra w [0,10, 0,90], a kontrola
realizowanych dopuszcza tylko tę dyskretyzację (0,02). Nie zmieniamy
szerokości po zobaczeniu wyników.

Siatka i optymalizacja są deterministyczne:

- stała baza `a` jest szukana na siatce `0.00, 0.01, ..., 0.95` (96 wartości),
- dla każdej `a` budujemy **rzeczywisty** model po foldingu z ratio
  `r(a)` i liczymy `P(a) = count_parameters(model)`. Nie sumujemy
  niezależnych oszczędności grup; parametry grup współzależnych są liczone
  na zrealizowanym modelu,
- zbiór dopuszczalny `F = {a : P(a) <= P_uniform}`,
- jeśli `F` niepusty: wybieramy `a*` z największym `P(a)` (najbliżej
  targetu bez przekroczenia; remis na mniejszą `a`),
- jeśli `F` pusty: `a* = 0.95` (maksymalna kompresja),
- następnie lokalne zawężenie wokół `a*`: `a ∈ [a*-0.01, a*+0.01]` co
  `0.001` (21 wartości) i ta sama reguła wyboru,
- `repair_budget(model, r(a*), P_uniform, min_ratio=0.10, max_ratio=0.90,
  max_steps=500)` jest uruchamiany **tylko wtedy**, gdy luka
  `100*(P_uniform - P(a*))/P_uniform` przekracza 0,5%. Dla mniejszej luki
  alokacja zostaje bez zmian (`repair_steps=0`), żeby repair nie zmieniał
  profilu ratio grup,
- finalna alokacja jest realizowana jako model i ponownie liczona przez
  `count_parameters`; musi być równa liczbie z solvera i nie większa niż
  `P_uniform`.

Kontrola jakości funkcji celu (przed oceną, na CPU): dla obu datasetów
sprawdzamy ciągłość siatki w sensie liczby parametrów, poprawność `P(a)` na
wybranych punktach przez niezależny fold niedeterministyczny, brak
przekroczenia targetu, lukę ≤ 0,5% oraz to, że `r_g` i `z_g` są skończone.
Nie zmieniamy formuły po zobaczeniu wyników S2.

### Czego AUTO nie robi

- Nie używa dev accuracy do selekcji grup, wyboru `beta`, naprawy budżetu
  ani siły kary.
- Nie wybiera poziomu S1 na podstawie korelacji z dev; używa wszystkich
  czterech poziomów symetrycznie, po standaryzacji w obrębie poziomu.
- Nie nazywa łącznej korelacji z 48 przebiegów S1 dowodem alokacji:
  korelacja łączna jest zmieszana z poziomem kompresji. W analizie S2
  raportujemy korelacje **przy stałym poziomie** jako kontrolę opisową,
  ale nie służą one do wyboru alokacji.

S1 dev był już oglądany w etapie rozwojowym. Dlatego S2 nie jest
niezależnym testem; to nowa, zamrożona ocena na tym samym seed42/dev.

## Warianty i koszty

Cztery warianty na dataset:

1. **UNIFORM** — wspólny keep ratio; jego faktyczne parametry są targetem.
2. **TILT+0.30** — `r_g = clip(base + 0.30*p_g, 0.10, 0.90)`,
   `p_g = 2g/11 - 1`; solver `base` pod target, potem repair.
3. **TILT+0.15** — jak wyżej z `0.15`.
4. **AUTO50** — reguła z tego protokołu, primary predictor, bez dev.

Wszystkie warianty: ten sam fold seed, ten sam BN seed, ten sam pełny train
BN, ta sama ewaluacja dev, ten sam batch size. Koszt kalibracji (liczba
próbek, batchy, czas, brak gradientów) jest raportowany per wariant.
Rzeczywiste koszty parametrów są raportowane jako `N`, `P_uniform`,
`P_variant`, `P_uniform - P_variant`, redukcja nominalna i faktyczna oraz
ratio per grupa.

## Uruchomienie i podział pracy

- 2 datasety × 4 warianty = **8 niezależnych workerów**.
- Cztery fizyczne GPU 0–3, po jednym workerze na GPU; dwie fale:
  CIFAR-10 (4 warianty), potem CIFAR-100 (4 warianty).
- Launcher przed startem workerów zapisuje zamrożone alokacje
  (`allocations_frozen.json`, specy per wariant) i audit alokacji.
  Kolejność jest zapisana w manifeście: `allocations_frozen_utc` musi być
  wcześniejsze niż start workerów.
- Launcher zapisuje snapshoty źródeł i ich SHA-256, `environment.json`,
  `pip-freeze.txt`, `conda-explicit.txt`, `nvidia-smi.txt`, przypisanie
  procesów do GPU, seedy, logi i `artifacts.sha256`.
- Launcher jawnie importuje `os`; nie powtarzamy błędu E015
  (`NameError: os`). Nie ma tu kontroli `expected_controls` z S1.
- Smoke: 8 wariantów na małym podzbiorze BN/dev (bez znaczenia naukowego),
  osobny katalog `runs/E016/s2_smoke_*`.
- Pełny bieg: osobny, unikalny katalog `runs/E016/s2_*`.

## Kryterium sukcesu (zamrożone)

Development criterion, nie publikacja:

- **G1:** AUTO50 post-BN top-1 > UNIFORM na **obu** datasetach.
- **G2:** AUTO50 nie gorszy od TILT+0.30 o więcej niż **0,5 pp** na obu
  datasetach.
- **G3:** wszystkie warianty completed, target nieprzekroczony, tolerancja
  dochowana, brak testu/rezerwy, BN identyczny dla metod.

Raportujemy też różnicę AUTO50 względem **najlepszej stałej reguły**
(TILT+0.30 albo TILT+0.15) wybranej opisowo po wynikach; nie służy ona do
zmiany kryterium. Sukces oznacza tylko rekomendację dalszych seedów, nie
publikowalność.

## Obsługa błędów

- Błąd infrastruktury (launcher, ścieżki, GPU, I/O) poprawiamy jawnie w
  nowych plikach S2, zachowujemy nieukończony run i uruchamiamy nowy.
- Błąd naukowy (np. AUTO nie przechodzi G1/G2, target nieosiągalny bez
  przekroczenia) raportujemy bez maskowania i bez zmiany formuły po wyniku.
- Nie modyfikujemy historycznych wyników, kodów ani snapshotów E011–E016 S1.
- Nie kopiujemy kodu z attention_v2/attention_v3.

## Ograniczenia

- Jeden seed42 na dataset; to etap rozwojowy.
- Dev był używany w S1; S2 nie jest niezależnym testem.
- Jeden przepis treningowy na dataset; checkpointy nie są niezależne.
- Ocena tylko na dev; test i rezerwa nietknięte.
- Pojedyncza architektura i dwa datasety; brak Tiny ImageNet/ImageNet.
- Deterministyczność dotyczy tego samego sprzętu i środowiska.
