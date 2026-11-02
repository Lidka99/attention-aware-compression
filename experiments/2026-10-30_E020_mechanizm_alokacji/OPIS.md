# E020 — mechanizm alokacji TILT: czy kierunek głębokości chroni grupy wrażliwe?

Status: **protokół 1.0 zamrożony przed uruchomieniem nowych pomiarów**
(2026-10-30 UTC). Eksperyment diagnostyczny; **bez treningu wag**, bez
fine-tuningu, bez zmiany wcześniejszych wyników.

## Pytanie

Dlaczego TILT+0.30 (rosnące ratio kompresji z głębokością) poprawia wynik
na CIFAR-10 (E014) i Tiny ImageNet (E019), a na CIFAR-100 (E015) prawie nie
poprawia? Czy kierunek „mniej kompresji wcześnie, więcej późno” rzeczywiście
chroni grupy o wysokiej zmierzonej szkodzie, czy działa z innego powodu?

Pytania szczegółowe (zgodne z `PLAN_2026-10-30_dalsze_kroki.md`, krok 1):

1. Jak szkoda zależy od głębokości grupy — przed i po korekcie BN?
2. Ile BN odzyskuje i które grupy pozostają krytyczne po BN?
3. Dlaczego CIFAR-100 wypada słabiej niż CIFAR-10 i Tiny ImageNet?
4. Czy TILT rzeczywiście chroni grupy o wysokiej zmierzonej szkodzie?
5. Czy błąd rekonstrukcji wag przewiduje rzeczywistą szkodę **przy stałym
   poziomie kompresji**?

## Zakres

Trzy zbiory, ResNet18, istniejące checkpointy:

| Zbiór | Checkpointy | Seedy | Uwaga |
|---|---|---|---|
| CIFAR-10 | E011 | 42–45 | stem 3×3, bez maxpool |
| CIFAR-100 | E015 | 42–45 | stem 3×3, bez maxpool |
| Tiny ImageNet 64×64 | E019 | 42–45 | standardowy stem 7×7 + maxpool |

- **Reanaliza bez GPU:** istniejące pomiary E016 S1 (pojedyncze grupy,
  CIFAR-10/CIFAR-100, seed42, poziomy 25/50/75/90%) oraz istniejące
  alokacje TILT/UNIFORM z E014/E015/E019 (wszystkie seedy i budżety).
- **Nowy pomiar GPU 1 (wrażliwość Tiny ImageNet):** dokładna replikacja
  protokołu E016 S1 na Tiny ImageNet, seed42: 12 grup × 4 poziomy
  + kontrola ratio 0 = **49 przebiegów**.
- **Ablacja GPU 2 (ANTI-TILT):** ten sam operator, BN i wspólny target
  parametrów co UNIFORM, ale alokacja z `tilt = −0.30` (odwrócony kierunek:
  więcej kompresji wcześnie, mniej późno). Budżety 30/50/70%, seedy 42–45,
  trzy zbiory = **36 przebiegów**. Wyniki UNIFORM i TILT+0.30 bierzemy
  z istniejących runów; nie powtarzamy ich.
- **Bez nowego treningu.** Oficjalny test i rezerwa pozostają nietknięte
  (Tiny ImageNet: `cache_val.npz` nieużywany).

Razem **85 nowych przebiegów** foldingu/oceny, cztery GPU, szacowany
wall-time 25–45 min, limit dysku 1,5 GiB.

## Operator i grupy

Bez zmian: 12 grup E012 w kolejności depth, operator
`experiments/2026-10-24_E012_folding_referencja/folding.py`, alokacja
`experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py`
(`allocation_from_tilt`, `solve_base_for_budget`, `repair_budget`).
`preserve_fc_bias=True`, `fold_seed=42`, korekta BN na pełnym train,
`bn_seed=20261024`, bez gradientów. Dla Tiny używamy modułu danych E019.

ANTI-TILT: `r_g = clip(base − 0.30 · (2g/(n−1) − 1))`, z `base` rozwiązanym
pod **faktyczną liczbę parametrów UNIFORM** dla danego seeda i budżetu,
następnie `repair_budget`. Te same granice ratio [0.1, 0.9] i tolerancja
budżetu 1% (nigdy powyżej targetu).

## Pomiary

Dla każdego przebiegu:

1. baseline na dev (top-1, NLL, logity),
2. folding,
3. **pre-BN** na dev,
4. korekta BN na pełnym train (CIFAR-10 45k, CIFAR-100 40k, Tiny 90k),
5. **post-BN** na dev.

Szkoda `damage_*_pp = 100 · (baseline_top1 − top1)`;
`bn_recovery_pp = 100 · (post_top1 − pre_top1)`;
analogicznie NLL. Zapisujemy logity (float32) i `summary.json` per item.

Dodatkowo, bez nowych przebiegów, liczymy:

- **szkodę znormalizowaną** `damage_pp / (100 · baseline_top1)` (frakcja
  trafności baseline utracona) oraz względny wzrost NLL,
- **koszt parametrów grupy** i `damage_pp` na 1% zaoszczędzonych parametrów,
- **profil klasowy** szkody dla CIFAR-10/CIFAR-100 z logitów E016 S1:
  korelacja szkody klasy z trafnością baseline klasy, koncentracja błędów,
- **interpolację szkody** w funkcji ratio grupy (0 przy ratio 0, liniowo
  między 25/50/75/90%) do porównania z ratio TILT.

## Hipotezy zamrożone przed nowymi pomiarami

Wszystkie są kierunkowe; raportujemy wsparcie lub sprzeczność, bez
twierdzenia o uniwersalności.

- **H1 (profil):** szkoda nie jest monotoniczna z głębokością i różni się
  między zbiorami. Na CIFAR-10 szczyty w grupach wczesno-średnich
  (`relu_conv`, `layer2.0.relu2`, `layer3.0.relu2`); na CIFAR-100 szczyt
  w późnych (`layer3.0.relu2`, `layer4.0.relu2`, `layer4.1.relu1`).
  Tiny ImageNet badamy eksploracyjnie (bez kierunku z góry).
- **H2 (dopasowanie TILT):** Spearman między ratio TILT a interpolowaną
  szkodą grupy jest **ujemny na CIFAR-10** (TILT kompresuje mniej tam,
  gdzie szkoda jest większa), a na **CIFAR-100 nieujemny lub słabo ujemny**.
  Różnica profili wyjaśnia słabszy wynik E015.
- **H3 (ablacja kierunku):** ANTI-TILT jest **gorszy od UNIFORM** na
  CIFAR-10 i Tiny ImageNet przy 50% (średnia różnica < 0, dodatnia u co
  najmniej 3/4 seedów), a na CIFAR-100 ANTI-TILT jest **nie gorszy od
  UNIFORM** i lepszy od TILT+0.30 (lub zbliżony). Jeżeli H3 nie zachodzi
  na CIFAR-10/Tiny, kierunek głębokości nie jest przyczyną przewagi TILT.
- **H4 (predyktor przy stałym poziomie):** w obrębie każdego poziomu
  kompresji `weight_reconstruction_rel_error` ma dodatnią korelację
  Spearmana ≥ 0,5 ze szkodą post-BN na **co najmniej 2 z 3** zbiorów.
- **H5 (CIFAR-100):** szkoda względna CIFAR-100 jest większa niż CIFAR-10
  przy tym samym poziomie; część różnicy współwystępuje z niższym
  baseline i większą liczbą klas, ale nie zakładamy z góry, że to ją
  wyjaśnia w całości.

## Kryteria interpretacji (zamrożone)

- **W1 — kompletność:** 49/49 pomiarów Tiny S1, kontrola ratio 0 w granicach
  0,05 pp i agreement ≥ 0,999999; 36/36 ablacji; brak użycia testu/rezerwy.
- **W2 — H2:** średnie ρ (po budżetach i seedach) dla CIFAR-10 ≤ −0,5 oraz
  ρ(CIFAR-100) ≥ −0,2 **albo** różnica ρ(CIFAR-10) − ρ(CIFAR-100) ≥ 0,3
  przy ujemnym ρ(CIFAR-10).
- **W3 — H3:** przy 50% średnia ANTI−UNIFORM < 0 na CIFAR-10 i Tiny
  (≥ 3/4 seedów ujemnych na każdym), a na CIFAR-100 ANTI−UNIFORM ≥ 0
  lub ANTI−TILT > 0.
- **W4 — H4:** jak wyżej, ≥ 2/3 zbiorów z ρ ≥ 0,5.
- **W5 — budżet:** wall-time ≤ 90 min, wyniki ≤ 1,5 GiB.

Niespełnienie W3 przy spełnionym W2 oznacza, że sama geometria szkody nie
wystarcza — raportujemy to jako ograniczenie mechanizmu, nie „naprawiamy”
wyniku po fakcie. Nie stroimy progów ani poziomów po zobaczeniu dev.

## Ograniczenia

- Pomiary pojedynczych grup nie są addytywne: TILT/ANTI-TILT kompresują
  wszystkie grupy naraz, więc profil szkody jest aproksymacją.
- Tiny ImageNet ma inną architekturę (standardowy stem) i 200 klas;
  porównania między zbiorami są jakościowe, nie izolują przyczyny.
- CIFAR-10/CIFAR-100: pomiar grupowy tylko seed42 (E016 S1); ablacja na
  4 seedach. Tiny: pomiar grupowy seed42, ablacja na 4 seedach.
- Ocena tylko na dev; brak niezależnego testu. To diagnostyka, nie
  publikacja ani nowa metoda.
- ANTI-TILT jest ablacją mechanizmu, nie proponowaną metodą; nie stroimy
  jego siły.

## Artefakty

- `configs/inventory.json` — zamrożone checkpointy, splity, baseline'y.
- `configs/sensitivity.json` — protokół pomiaru Tiny S1.
- `configs/ablation.json` — protokół ANTI-TILT.
- `e020_common.py` — wspólne helpery (dane, folding, BN, metryki).
- `sensitivity_worker.py`, `launch_sensitivity.py` — pomiar Tiny S1.
- `ablation_worker.py`, `launch_ablation.py` — ablacja ANTI-TILT.
- `analyze_e020.py` — analiza CPU (profile, ρ, ablacja, klasy).
- `check_cpu_e020.py` — preflight bez GPU.
- `runs/E020/...` — unikalne katalogi przebiegów i analiz.

## Odtworzenie (skrót)

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-30_E020_mechanizm_alokacji/check_cpu_e020.py
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-30_E020_mechanizm_alokacji/launch_sensitivity.py
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-30_E020_mechanizm_alokacji/launch_ablation.py
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-30_E020_mechanizm_alokacji/analyze_e020.py \
  --sensitivity-run runs/E020/<SENS> --ablation-run runs/E020/<ABL> \
  --output-dir runs/E020/analizy/<UTC>_final
```
