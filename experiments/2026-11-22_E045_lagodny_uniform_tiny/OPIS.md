# 2026-11-22_E045_lagodny_uniform_tiny — ile jakości zachowuje łagodny folding?

**Pytanie:** jaka jest bezwzględna strata top-1 przy około 10% i 20% redukcji
parametrów (P) przez UNIFORM na ResNet18/Tiny64? To dwa z góry ustalone
punkty krzywej jakość–koszt. Raportujemy rozmiar straty, bez nowego progu
sukcesu. Punkty 0/30/50% są odniesieniem w tej samej fazie dev.

## Zamrożone ustawienia

- Cztery końcowe `last.pt` E019, seedy 42–45. Historyczny Tiny **dev** z
  E035: 10 000 indeksów w `cache_train.npz`, odrębne od train i A4096.
  Krzywa dotyczy dev; nie łączymy jej z oficjalnym val E035/E044.
- **UNIFORM:** dla b10/b20 wybieramy na CPU największy wspólny udział
  zachowanych kanałów (po dyskretyzacji), który spełnia `P(k) ≤ floor((1-b)P0)`
  przy `k_g = floor(n_g·keep)`. Sprawdzamy następny szerszy punkt sparametryzowany
  tym samym `keep` i podajemy rzeczywistą redukcję. Zakres `1 ≤ k_g ≤ n_g`
  pozwala przekroczyć 90% szerokości; nie stosujemy tu ograniczenia projekcji
  L06, które na Tiny wymusza minimum 19,21% redukcji P. Do CPU trafia tylko
  geometria, nie jakość nowych modeli. Liczymy P i Conv/Linear MAC na obraz
  64×64 z hooków. P wyznacza budżet; MAC jest mierzonym kosztem geometrii.
- Wyłącznie operator CLFM `694691556e8314a835178a741938dd3d4d80c171` na naszych
  modelach Tiny z zachowaniem biasu FC, jak E035. Pełny model i dwa nowe
  UNIFORM przechodzą osobno identyczny reset BN + CMA (te same widoki A4096
  z train, `e30.augment` i `e30.normalize`, batch 128, 32 partie). Bez
  aktualizacji gradientowych i bez zapisu nowych wag.
- **Bramka zgodności:** sprawdzić hashe wejść i widoków A, geometrię i stany
  CMA. Stan i liczba trafień pełnego modelu muszą dokładnie odtworzyć zapis
  E035 dev przed punktacją nowych profili. U/b30 i U/b50 z E035 dev służą
  wyłącznie jako punkty odniesienia z zamrożonych wyników.
- Conda `attention`, dokładnie 4 GPU, seed 42–45 → GPU 0–3, jeden proces na
  kartę, FP32/TF32 off, batch 128; **12 nowych ocen** (4 pełne, 8 nowych
  UNIFORM) i **8 odniesień** z E035. Wszystkie procesy kończą się przed
  odczytem zbiorczych wyników. Monitorujemy PID i kody zakończenia.

## Raport

Trafienia i top‑1 per seed, średnia i próbne SD; dla każdego budżetu strata
pełny−U na sparowanych checkpointach, P, MAC oraz redukcje względem
pełnego modelu. Figura: trafność oraz strata w funkcji kosztu. Wnioskujemy
o kompromisie dla UNIFORM na Tiny, bez oceny przewagi L06 przy b10/b20.
[Odtworzenie](URUCHOMIENIE.md).
