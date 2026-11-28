# 2026-11-25_E050_wrazliwosc_bn_tiny — wpływ skali cropu w rekalibracji BN na b10

**Pytanie diagnostyczne po E049:** czy na Tiny ImageNet zmniejszenie siły
kadrowania przy rekalibracji pomaga L06 bardziej niż pełnemu modelowi i
UNIFORM? E049 dziedziczyło widoki E030 z `RandomResizedCrop` o zakresie
`[0,40; 1,00]`. E048 trenowało Tiny z dolną granicą 0,65. W E049 pełny
Tiny stracił na dev 1,7775 pp po CMA; nie przesądza to o źródle straty
względnej L06.

- Cztery `last.pt` E048, seedy 42–45, i niezmienione geometrie `full`,
  `b10_U`, `b10_Tproj`, `b10_L06` z E049. Ten sam operator upstream i adapter,
  bez aktualizacji parametrów. Na każdy seed przypada osobny proces/GPU.
- Dwie kalibracje na tych samych 4096 obrazach train, w tej samej kolejności,
  z tym samym deterministycznym ziarnem per obraz, resize, odbiciem,
  normalizacją i A4096/CMA (32 × 128). Zmieniamy tylko dolną granicę `scale`:
  0,40 → 0,65. Ten wariant przejmuje skalę cropu, nie cały pipeline E048.
- Punktacja tylko na tym samym 10 000-obrazowym Tiny dev, w FP32 bez TF32.
  Dla kontroli 0,40 wymagamy zgodności widoków, stanów po CMA oraz
  liczby poprawnych odpowiedzi z E049 (16/16). Potwierdzamy P/MAC oraz
  niezmienione parametry we wszystkich parach kalibracji.
- Raportujemy top‑1, zmianę po kalibracji dla każdej geometrii, zmianę
  straty do `full` i zmianę przewagi nad UNIFORM; również kontrast z T-proj.
  Punkt odniesienia to `full` i UNIFORM po **tej samej** kalibracji. Trop
  jest kandydatem do dalszego potwierdzenia, jeśli na dev wariant 0,65
  daje L06−UNIFORM ≥1 pp, ≥3/4 dodatnie pary i stratę L06 do `full` ≤3 pp,
  jednocześnie poprawia L06 względem 0,40 w ≥3/4 parach i nie obniża
  średniego top‑1 `full` ani UNIFORM. Progi są praktyczne.
- Kontrola zaprojektowana po obejrzeniu E049; wynik ma charakter
  diagnostyczny. Oficjalny val, rezerwa i ImageNetV2 są poza zakresem.
  Nie zmieniamy kryteriów ani wyników E049.
