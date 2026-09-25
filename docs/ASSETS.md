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
