# Krzywe uczenia 2026-09-26_E001_efficientnet_b0_od_zera

Dokument wygenerowany automatycznie przez `plot_metrics.py` na podstawie istniejącego pliku metryk E001. To opis przebiegu uczenia, bez nowej inferencji, bez zmiany wag i bez ponownego treningu.

## Źródło danych

- metryki: `/home/users/s224574/attention/badania_atencja_cnn/runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg/metrics.jsonl`
- SHA-256 metryk: `a5910ae927271f0d3b1b7087e385f2f9a432a646bb6b258a8aa6f615dc158e2e`
- skrypt: `/home/users/s224574/attention/badania_atencja_cnn/experiments/2026-09-26_E001_efficientnet_b0_od_zera/plot_metrics.py`
- SHA-256 skryptu: `8493b34ee9d68b508ea68147e166528043c7389ecd005fbc36e73f06a27e0f85`
- epoki: 200 (ciągłe 1..200), train_count=39936, dev_count=5000
- wygenerowano (UTC): 2026-09-29T02:55:32.658656+00:00

## Najważniejsze liczby

- Pierwsza epoka: train NLL 4.645315, train top-1 2.12%, dev NLL 4.590845, dev top-1 4.72%.
- Ostatnia epoka: train NLL 0.001631, train top-1 99.95%, dev NLL 2.490339, dev top-1 66.54%.
- Minimum dev NLL: 1.592417, epoka 24.
- Najlepszy dev top-1: 66.84% (3342/5000), epoka 177 (pierwsze maksimum; przy remisie zachowywane jest wcześniejsze).

## Ostrzeżna interpretacja

- Train jest mierzony online, na obrazach augmentowanych, w trybie `train()` z AMP, a dev to ocena zamrożonego po epoce modelu w `eval()` w FP32. Warunki te nie są identyczne, dlatego nie należy ich porównywać wprost.
- Utrzymująca się różnica między train top-1 a dev top-1 oraz rosnąca dev NLL przy jednoczesnym wzroście dev top-1 mogą wskazywać na przeuczenie; ten opis nie ustala jego przyczyny.
- Najlepszy dev top-1 i minimum dev NLL mogą przypadać na różne epoki: top-1 w epoce 177, a minimum NLL w epoce 24. NLL ocenia prawdopodobieństwo przypisane właściwej klasie i silnie karze pewne błędne odpowiedzi; nie jest samodzielnym pomiarem kalibracji. Top-1 mierzy trafność.
- Dev był używany do wyboru checkpointu, więc nie jest niezależnym testem generalizacji. Nie oceniano tutaj rezerwy ani oficjalnego testu.
- To audyt jednego zapisu metryk E001, a nie dowód korzyści atencji lub kompresji. Analiza nie wykonuje nowej inferencji i nie zmienia wag.
