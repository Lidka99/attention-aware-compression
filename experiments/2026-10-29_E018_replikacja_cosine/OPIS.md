# E018 — replikacja tropu cosine (protokół 1.0, zamrożony)

Status: **protokół zamrożony przed pierwszym uruchomieniem** (2026-10-29 UTC).
Eksperyment jest kontynuacją E017, ale **osobnym** etapem replikacji.

## Skąd hipoteza

W E017 (benchmark 18 komórek: 3 miary × 2 zbiory × 3 budżety, seed42)
jedyny duży efekt dodatni to **cosine przy 50% kanałów na CIFAR-10:
78,62% vs reference 70,48%, +8,14 pp** przy identycznej liczbie parametrów
(2 797 610). Ten wynik **nie spełnił** zamrożonego kryterium E017, bo nie
powtórzył się na CIFAR-100 ani przy 30%. Został znaleziony **post hoc**
przez przegląd siatki, więc wymaga niezależnej replikacji.

## Hipoteza H1

Grupowanie kanałów po **cosinusie** (normalizacja L2 wektorów kanałów przed
PCA+HKMeans) poprawia jakość foldingu względem reference przy **50% usuniętych
kanałów na CIFAR-10**, na **nowych seedach 43–45**.

Seedy 43–45 **nie brały udziału** w odkryciu efektu (był to seed42).

## Kryterium główne (zamrożone)

H1 uznajemy za **potwierdzoną** tylko jeśli na CIFAR-10, przy 50% kanałów:

1. różnica `cosine − reference` jest **dodatnia na każdym z trzech seedów**
   (43, 44, 45), oraz
2. **średnia różnica ≥ +2,0 pp** (konserwatywnie; obserwacja wynosiła +8,14).

Nie zmieniamy progu ani budżetu po wynikach.

## Kontrola specyficzności i budżet wtórny

- **CIFAR-100, 50%:** raportujemy średnią różnicę na seedach 43–45.
  Brak progu zaliczenia — to kontrola, czy efekt nie jest uniwersalny.
  Wynik dodatni nie szkodzi hipotezie, ale zmienia jej zakres.
- **CIFAR-10, 30%:** raportujemy średnią; jeśli główny test przejdzie,
  a 30% pogorszy się o >1,0 pp, wniosek ograniczamy do reżimu wysokiej
  kompresji.

## Co jest identyczne z E017 (kontrola)

- Operator scalania i odbudowy E012 bez zmian.
- Liczba klastrów `k = max(int(n*(1-ratio)), 1)` identyczna dla obu miar;
  liczba parametrów po foldingu **dokładnie równa** (weryfikacja w launcherze).
- PCА pełnego rzędu + HKMeans (`random_state=42`, `n_init=1`, `max_iter=10`,
  `n_jobs=1`).
- Korekta BN: pełny train bez etykiet i gradientów, `bn_seed=20261024`,
  batch 128, ta sama kolejność dla obu miar.
- `preserve_fc_bias=True`, `fold_seed=42`.
- Ocena dev 5 000; oficjalny test i rezerwa nietknięte. Brak treningu wag.

## Zakres obliczeń

- 6 checkpointów: CIFAR-10 seed43/44/45, CIFAR-100 seed43/44/45.
- 2 miary (reference, cosine) × 2 budżety (30%, 50%) = **24 zadania**.
- Cztery fizyczne GPU 0–3; fale po cztery zadania; unikalny katalog
  `runs/E018/<UTC>_<git>_<token>/`.
- Nadzór z twardym limitem czasu i zapisem exit code; bez restartów.

## Decyzja

- **Kryterium spełnione:** efekt cosine w reżimie wysokiej kompresji
  na CIFAR-10 uznajemy za **potwierdzony kierunkowo** na trzech niezależnych
  seedach. To nadal nie publikacja: jeden zbiór, jedna architektura,
  mechanizm nieustalony.
- **Kryterium niespełnione:** zamykamy trop cosine; nie stroimy miary
  ani progu po wynikach. Zapisujemy wynik negatywny.

## Odtworzenie

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E018_replikacja_cosine/check_cpu_e018.py
CONDA_DEFAULT_ENV=attention CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E018_replikacja_cosine/run_supervised.py \
  --timeout-seconds 3600 --log runs/E018/NOWY_UNIKALNY_LOG.log
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E018_replikacja_cosine/analyze_e018.py \
  --run-dir runs/E018/<RUN>
```
