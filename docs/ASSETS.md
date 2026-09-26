# Zasoby — do ustalenia przed E001

| Zasób | Wymagany zapis | Status |
|---|---|---|
| Model CNN z SE/ECA | Implementacja, wersja, pełny graf, kolejność kanałów | UNSET |
| Checkpoint | Źródło, licencja, SHA-256, preprocessing, oczekiwana jakość | UNSET |
| Dane | Źródło, wersja, manifest plików/hash archiwum | UNSET |
| Podziały | Dokładne ID, hashe, role, audyt duplikatów | UNSET |
| Sprzęt | CPU/GPU/RAM, limit czasu i pamięci | UNSET |
| Środowisko ML | Python, lockfile, CUDA/cuDNN/sterownik | UNSET |
| Magazyn artefaktów | Trwałe URI, zasady dostępu i backupu | UNSET |
| Bloki pilota | Ścieżki modułów, wejścia/wyjścia, legalne szerokości | UNSET |

Decyzja o modelu powinna wynikać z dostępności wiarygodnego checkpointu i danych, nie tylko wygody implementacji. Nie zakładać, że istniejący checkpoint pasuje do nowego protokołu bez audytu.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.

## E001 — stan 2026-09-26

Sprzęt: 4×RTX 2080 Ti 11264 MiB; sterownik 580.173.02. Środowisko: Conda attention, Python 3.11.15, torch 2.12.0+cu130, torchvision 0.27.0+cu130. Eksport w configs/environment. Model: EfficientNet-B0, 100 klas, inicjalizacja losowa. Dane CIFAR-100 to roboczy wybór pilota, nie zostały pobrane. Trwały magazyn wyników pozostaje do ustalenia.
