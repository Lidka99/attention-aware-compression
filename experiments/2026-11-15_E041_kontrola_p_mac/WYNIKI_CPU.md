# E041 — wykonalność dopasowania P i MAC potwierdzona

**Późniejsza aktualizacja:** etap GPU ukończono2026-11-16;
[wyniki jakości](WYNIKI.md). Poniżej zachowany raport przygotowawczy CPU.

**Etap przygotowawczy CPU ukończony, bez oceny jakości.** Otrzymano4różne
kontrole na każdym CIFAR-ze; wszystkie mają P oraz MAC nie większe niż
TILT i nie mniejsze niż99,9%jego kosztów. Rzeczywisty operator potwierdził
parametry i MAC8/8profili na CPU.

## Wykonanie

- Geometria: `runs/E041/cpu_20261115T232256Z_44602061`, completed,
  bramkaA PASS, **0,510s**. Tylko JSON-y planów, bez wag i obrazów.
- Operator: `runs/E041/check_20261115T234914Z_ab01e2db`, passed,
  **93,546s** części pomiarowej. Istniejące wagi seed42 dla obu zbiorów,
  syntetyczne wejścia do hooków MAC, bez obrazów/etykiet/inferencji dev.
- Conda `attention`, Python3.11.15, NumPy2.4.6, SciPy1.17.1; CPU,
  `CUDA_VISIBLE_DEVICES=''`, BLAS/OMP/MKL2.
- Zamrożone wejścia `frozen.json`, SHA256:
  `ddaacb15afe677681d832c85a87d9f02fd5bf10d429dd64b39bb07be56c36e16`.
- Check wiąże wersje `experiment.py`, `geometry.py`, OPIS i frozen SHA256.
  Niedokończone próby i poprawki techniczne: [DZIENNIK.md](DZIENNIK.md).
- Nie zapisano nowych plików wag. Etap GPU i jakość nie wynikają z tego raportu.

## Faktyczne dopasowanie

Ilorazy kosztów kontroli względem historycznego TILT b50 E035.
Tolerancja jest jednostronna `[0,999;1]`, a nie `±0,1%`.

| Zbiór | P kontroli / P(T) | MAC kontroli / MAC(T) | RMS różnic k/n od T |
|---|---|---|---|
| CIFAR-10 | 0,9995570–0,9999065 | 0,9990075–0,9999865 | 0,08246–0,13032 |
| CIFAR-100 | 0,9991875–0,9998932 | 0,9990880–0,9997529 | 0,08333–0,13074 |

Maksymalne niedowykorzystanie P to0,0813%, MAC to0,0993%.
Każda kontrola zmienia6–7grup o co najmniej5pp udziału zachowanych
kanałów; RMS parami między kontrolami wynosi0,07678–0,18524.
Osiem profili jest unikalnych. Źródła: `geometry.json`, `profiles.csv`
w runie geometrii oraz `check.json`/manifest kontroli operatora.

Wzór kosztu odtwarza także pełną sieć: C10 **11 173 962P /555 422 720MAC**,
C100 **11 220 132P /555 468 800MAC**; oraz historyczne U/T i propozycje
E039. Check operatora sprawdza niezależnie rzeczywiste kształty/forward,
P, MAC z hooków, mapowanie grup, niepuste klastry i zachowany bias FC.

## Wniosek i ograniczenia

**Porównanie różnych profili przy kontrolowanych P i MAC jest technicznie
wykonalne w tej konstrukcji.** Dodatni etap CPU nie mówi, który profil
będzie lepszy jakościowo. Nie ustanawia też optymalności TILT.

Kontrole pochodzą z czterech stałych propozycji E039 po lokalnej projekcji
geometrycznej. Nie zachowują histogramu ratio ani nie reprezentują całej
przestrzeni. Różnice skupiają się w późniejszych grupach, ponieważ
wczesne szerokości TILT są blisko górnych granic. Minimalnie większe P/MAC
TILT pozostaje jawnym ograniczeniem, mimo tolerancji≤0,1%.

Następny etap: [zamrożona kontrola jakości](OPIS.md),2zbiory×4seedy,
4GPU,168ocen (96nowych kontrolnych), limit30min/2GPU-h.
Przed pierwszą nową oceną wszystkie4procesy muszą odtworzyć teacher/U/T.
