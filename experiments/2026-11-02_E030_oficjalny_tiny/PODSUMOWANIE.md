# E030 — oficjalny operator na Tiny ImageNet

**Completed, kryteria spełnione.** Run `runs/E030/20261102T013740Z_609b7737`,
84 oceny+12 kontroli,4GPU/4seedy, exit0,2,0min. Bez treningu.
Niezmieniony operator upstream na naszym torchvision ResNet18/Tiny64.

- LEGACY E012/E014 **identyczny** z upstream (wagi, ARI1, logity≤2e−6).
  LOCAL E024 zgodny jakościowo (max0,36pp), ale nie bitowo: ARI0,66–0,76.
- TILT−UNIFORM na operatorze autorów: **+9,408pp** (CMA) i **+9,178pp**
  (ich repair_bn), **4/4** seedów w obu. Trzeci zbiór potwierdza kierunek.
- Bias FC: −0,025pp, nieistotny. Teacher pre50,82→CMA51,22%.
- Ograniczenie: nasz model Tiny, nie klasa modelu autorów (upstream nie ma
  wariantu64×64); jeden budżet i seed augmentacji.

Razem z E029: TILT−UNIFORM na operatorze upstream **+4,00/+1,70/+9,41pp**
(CIFAR-10/100/Tiny, CMA,~50%). [WYNIKI.md](WYNIKI.md).
