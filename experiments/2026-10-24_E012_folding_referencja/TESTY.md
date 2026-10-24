# Testy E012 przed uruchomieniem

2026-10-24: 16/16 testów CPU passed, w tym rzeczywisty HKMeans ratio0.3
na syntetycznym ResNet18 (27 s). Singletony, bypass ratio0, mapowanie osi,
powiązania między stage, residuale, finitywność, serializacja i polityka biasu.

Pierwsza próba miała 15/16: błędnie wymagała równych rozmiarów klastrów
w małym przykładzie. HKMeans nie gwarantuje równych rozmiarów; poprawiono
test na niepuste klastry, bez zmiany algorytmu.

`check_reference.py` na checkpointcie E011 seed42: passed_single_operation.
Pominięcie biasu przy odbudowie Linear zmieniło logity o max0.065934;
kopiowanie biasu daje bitowo identyczne logity. To test jednej operacji,
nie pełnego pipeline'u referencyjnego przy ratio0.

Pełne porównanie end-to-end z oficjalną implementacją nie było wykonane.
Wynik E012 należy nazywać adaptacją referencyjnych reguł operatora.

Komenda z folderu E012:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' E012_RUN_HKMEANS=1 /home/users/s224574/miniconda3/envs/attention/bin/python -B -m unittest -v test_folding
```
