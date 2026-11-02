# E021 — metryki grupowania w foldingu: L1, L2, cosine, korelacja, Ward

Status: **protokół 1.0 zamrożony przed uruchomieniem** (2026-10-31 UTC).

## Pytanie
Czy przy **identycznym operatorze, budżecie i szerokościach warstw** inna
metryka/algorytm grupowania kanałów poprawia folding bez fine-tuningu?
Testujemy w szczególności **L1 (Manhattan)** i **korelację**, a także L2,
cosine, Ward i standaryzację — na **CIFAR-10, CIFAR-100 i Tiny ImageNet**.

## Dane i checkpointy (seed42, etap odkrywczy)
| Tag | Model | Dev top-1 |
|---|---|---:|
| cifar10_s42 | E011 ResNet18 | 95,28% |
| cifar100_s42 | E015 ResNet18 | 74,94% |
| tiny_s42 | E019 ResNet18 (standard stem, 64×64) | 51,08% |

Ścieżki i hashe w `configs/stage1.json`. Oficjalny test i rezerwa nietknięte.

## Budżety i stałe szerokości
Dla każdego datasetu i budżetu 30/50/70% parametrów najpierw rozwiązujemy
**UNIFORM** (`solve_base_for_budget`). Jego ratios są używane przez wszystkie
warianty, więc liczba klastrów `k` w każdej grupie i liczba parametrów po
foldingu są **identyczne**. Operator scalania/odbudowy E012 bez zmian;
`preserve_fc_bias=True`, `fold_seed=42`; korekta BN na pełnym train bez
etykiet i gradientów, `bn_seed=20261024`.

## Warianty (8)
1. **reference_hkmeans** — L2 na surowych cechach, PCA + HKMeans (jak E012).
2. **agg_l2** — agglomerative average, metryka euklidesowa.
3. **agg_l1** — agglomerative average, metryka Manhattan (L1).
4. **agg_cosine** — agglomerative average, metryka cosine.
5. **agg_correlation** — agglomerative average, metryka korelacji Pearsona.
6. **ward_l2** — agglomerative Ward, metryka euklidesowa.
7. **standardized_hkmeans** — standaryzacja współrzędnych + HKMeans.
8. **random_balanced** — etykiety referencyjne permutowane losowo
   (ten sam multizbiór rozmiarów klastrów); kontrola.

Wszystkie warianty używają tej samej liczby klastrów `k` i tego samego
operatora scalania. Różni się wyłącznie sposób przypisania kanałów do grup.

## Zakres obliczeń
3 datasety × 8 wariantów × 3 budżety = **72 zadania**, seed42.
Cztery GPU 0–3, fale po cztery zadania, unikalny katalog `runs/E021/`.
Bez treningu wag; brak zapisu pełnych logitów (oszczędność dysku).

## Kryterium (zamrożone)
Kandydat jest **STRONG**, jeśli przy **50% i 70%** parametrów na **wszystkich
trzech** zbiorach:
1. bije reference o ≥ **+0,50 pp**, oraz
2. bije random_balanced o ≥ **+0,50 pp**.

**PROMISING**, jeśli spełnia to na **≥2 z 3** zbiorów. 30% raportujemy
pomocniczo. STRONG/PROMISING oznacza wyłącznie rekomendację replikacji na
seedach 43–45 — nie publikację. Brak spełnienia = wynik negatywny.

## Ograniczenia
- Jeden seed na zbiór; brak CI i replikacji na tym etapie.
- Trzy architektury/przepisy: CIFAR ResNet18 i Tiny ResNet18 standard stem.
- Dev był używany wcześniej; to etap rozwojowy.
- Adaptacja operatora E012, nie oficjalny kod 1:1.

## Odtworzenie
```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-31_E021_metryki_grupowania/check_cpu_e021.py
CONDA_DEFAULT_ENV=attention CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/users/s224574/miniconda3/envs/attention/bin/python -u -B \
  experiments/2026-10-31_E021_metryki_grupowania/launch_e021.py
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-31_E021_metryki_grupowania/analyze_e021.py \
  --run-dir runs/E021/<RUN>
```
