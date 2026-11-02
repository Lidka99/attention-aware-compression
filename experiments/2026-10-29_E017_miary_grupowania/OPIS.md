# E017 — miary grupowania w foldingu (protokół 1.1, zamrożony)

Status: **protokół 1.1 zamrożony przed pierwszym uruchomieniem GPU** (2026-10-29 UTC).

## Historia rewizji (przed wynikami)

- **1.0** — pierwsza wersja.
- **1.1** — po kontroli CPU: budżety oznaczają **proporcję usuniętych
  kanałów w grupie**, nie parametrów. Operator E012 usuwa kanały, więc
  redukcja parametrów jest silniejsza (przy 50% kanałów ≈ 75% parametrów).
  Budżety zmieniono na 10/30/50% kanałów, zgodnie z zakresem E012;
  kryterium dotyczy 30% i 50%. Nie zmieniono żadnej miary ani operatora.
  Nie było jeszcze uruchomienia GPU ani wyników.
Ten eksperyment odpowiada na pytanie promotorów: przy **tym samym operatorze
foldingu i tych samych szerokościach warstw**, czy inna miara/grupowanie
kanałów poprawia jakość bez fine-tuningu?

## Cel i hipoteza

Operator E012 (adaptacja *Cut Less, Fold More*) grupuje kanały przez
klasteryzację surowych, konkatenowanych wag grupy po PCA (HKMeans).
Hipoteza: **reprezentacja kanału użyta do grupowania** ma znaczenie —
kierunek wektorów lub ich efektywna skala po BN mogą być lepszą podstawą
grupowania niż surowe wagi.

To pytanie o **grupowanie**, nie o alokację budżetu między warstwy.
Szerokości po kompresji są identyczne we wszystkich wariantach.

## Co jest zamrożone

- Operator scalania i odbudowy: funkcje E012 bez zmian
  (`merge_group_params`, `rebuild_module`, `_rebuild_group`).
- Liczba klastrów w grupie: `k = max(int(n * keep), 1)` dla każdej metody —
  identyczna. Liczba parametrów po foldingu zależy tylko od `k`, więc
  budżety są **dokładnie równe** (kontrola w launcherze).
- PCА z pełną liczbą składowych i HKMeans: `random_state=42`, `n_init=1`,
  `max_iter=10`, `n_jobs=1`.
- Korekta BN: pełny train bez etykiet i gradientów, `bn_seed=20261024`,
  batch 128, ten sam loader i kolejność dla wszystkich metod.
- `preserve_fc_bias=True`, `fold_seed=42`.
- Ocena: dev, 5 000 obrazów, bez augmentacji losowej.
- Oficjalny test i rezerwa CIFAR-100 **nietknięte**.

## Dane i checkpointy (seed42, etap rozwojowy)

| Zbiór | Checkpoint | Dev top-1 | Dev NLL |
|---|---|---:|---:|
| CIFAR-10 | E011 `seed_42/last.pt` | 95,28% | 0,16766352424621583 |
| CIFAR-100 | E015 `seed_42/last.pt` | 74,94% | 0,9906619615554809 |

Ścieżki i hashe SHA-256 są w `configs/stage1.json`. Split: E011
(45k train / 5k dev) i E015 (40k train / 5k dev / 5k rezerwy).
Baseline jest liczony w każdym zadaniu i musi zgadzać się z tabelą.

## Warianty grupowania (7)

Wszystkie używają tej samej liczby klastrów `k` w każdej grupie.

1. **reference** — surowe, konkatenowane cechy grupy (wagi konwolucji,
   wagi i biasy BN bez statystyk running), dokładnie jak E012.
2. **cosine** — każdy wektor kanału normalizowany do długości 1 (L2), potem
   ten sam pipeline PCA+HKMeans. Bada, czy liczy się kierunek, nie skala.
3. **standardized** — każda współrzędna cechy standaryzowana w poprzek
   kanałów (średnia 0, odchylenie 1), potem PCA+HKMeans. Bada wpływ różnych
   skal współrzędnych (oficjalny kod robi tak dla CLIP).
4. **bn_effective** — cechy nie-BN skalowane przez efektywną skalę kanału
   `s_c = |gamma_c| / sqrt(running_var_c + eps)`, uśredniona geometrycznie
   po wszystkich modułach BN w grupie (grupy mogą mieć kilka BN o tej samej
   szerokości); bez wag i biasu BN (są odbudowywane przez BN repair).
   Bada „funkcjonalną” skalę kanału. Grupa bez BN (fc) używa cech surowych —
   oznaczane flagą.
5. **random1/2/3** — kontrola: etykiety referencyjne permutowane losowo
   (ten sam multizbiór rozmiarów klastrów), trzy niezależne permutacje.
   Pokazuje, ile daje samo losowe przypisanie przy tym samym budżecie.

## Budżety

10%, 30%, 50% usuniętych **kanałów** w każdej grupie (semantyka E012).
Redukcja parametrów jest silniejsza i raportowana osobno jako
`reduction_fraction`; jest identyczna między miarami w obrębie budżetu,
bo zależy tylko od liczby klastrów `k`.

## Podział pracy i uruchomienie

- 2 zbiory × 7 wariantów × 3 budżety = **42 niezależne zadania**.
- Cztery fizyczne GPU 0–3; fale po cztery zadania, po jednym zadaniu na GPU.
- Każde uruchomienie ma unikalny katalog `runs/E017/<UTC>_<git>_<token>/`
  z manifestem, snapshotem źródeł, środowiskiem i hashami.
- Odporny nadzór (`run_supervised.py`): konkretny PID, limit czasu,
  `.status.json`, bez automatycznych restartów i bez `pgrep -f`.

## Kryterium sukcesu (zamrożone przed wynikami)

Wariant uznajemy za **potwierdzony kierunkowo** tylko jeśli na **obu**
zbiorach przy **30% i 50%** kanałów:

1. jest lepszy od `reference` o co najmniej **+0,50 pp**, oraz
2. jest lepszy od średniej `random1..3` o co najmniej **+0,50 pp**.

10% raportujemy pomocniczo. Ponieważ to jeden seed na zbiór, wynik spełniający
kryterium oznacza wyłącznie **rekomendację replikacji na pozostałych seedach**,
nie publikację. Wynik niespełniający kryterium zapisujemy jako negatywny.

## Kontrole poprawności (CPU)

- `reference` musi odtworzyć etykiety E012 grupa po grupie dla tego samego
  checkpointu i budżetu (test równoważności operatora).
- Ta sama `k` i ta sama liczba parametrów dla wszystkich 7 wariantów.
- Deterministyczność: dwa przebiegi dają identyczne etykiety.
- `random*` mają identyczny multizbiór rozmiarów klastrów jak `reference`.
- Brak dostępu do dev w trakcie grupowania; brak gradientów i treningu wag.

## Ograniczenia

- Jeden seed42 na zbiór; to etap rozwojowy.
- Dev był używany wcześniej (E011/E015), więc nie jest niezależnym testem.
- `bn_effective` korzysta z running stats BN z treningu, nie z dev.
- Miara uwzględniająca atencję (SE) to **osobny etap** po tym benchmarku;
  wymaga modelu z SE i osobnego protokołu. Nie jest częścią E017.
- Nie twierdzimy, że któraś miara jest „nowa” bez audytu literatury.

## Odtworzenie

```bash
cd /home/users/s224574/attention/badania_atencja_cnn
CUDA_VISIBLE_DEVICES='' /home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E017_miary_grupowania/check_cpu_e017.py
CONDA_DEFAULT_ENV=attention CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E017_miary_grupowania/run_supervised.py \
  --timeout-seconds 3600 --log runs/E017/NOWY_UNIKALNY_LOG.log
/home/users/s224574/miniconda3/envs/attention/bin/python -B \
  experiments/2026-10-29_E017_miary_grupowania/analyze_e017.py \
  --run-dir runs/E017/<RUN>
```
