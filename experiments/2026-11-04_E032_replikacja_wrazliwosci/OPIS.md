# 2026-11-04_E032_replikacja_wrazliwosci

Status: **frozen przed runem**, 2026-11-04 UTC.

## Pytanie
Czy późna wrażliwość CIFAR-100 z E022/seed42 powtarza się na czterech
checkpointach i jest silniejsza niż na kontrolnym CIFAR-10? Czy opisowo
wiąże się ze zróżnicowanym zyskiem TILT w E031? Jest to osobna diagnostyka
na istniejących modelach, nie poszukiwanie nowej alokacji.

## Zamrożony zakres
- CIFAR-100 i kontrolny CIFAR-10, seedy42–45 z E015/E011.
- Każda z12 strukturalnych grup osobno: **75% redukcji kanałów tej grupy**,
  wszystkie pozostałe grupy zachowane. To nie70% globalnej redukcji parametrów.
  Grupy w kolejności `legacy.group_names()` (E014).
- Operator LEGACY E012/E014, seed42 klastrowania, zachowany bias FC.
  Niekompresowane grupy mają etykiety tożsamościowe; kontrola zero compression
  ma identyczne parametry i statystyki BN oraz logity; odbudowa BN zeruje
  liczniki `num_batches_tracked` (bez wpływu na eval, CMA/reset ani EMA0.1).
  Nie twierdzimy, że jest to nowy audyt operatora
  upstream na profilach pojedynczej grupy. E031 potwierdził zgodność LEGACY
  na pełnosieciowych profilach, nie na wszystkich możliwych profilach.
- PRE, CMA/reset oraz niezmienione `repair_bn` upstream commit
  `694691556e8314a835178a741938dd3d4d80c171`, klon E029 pozostaje czysty.
  Wszystkie oceniane modele to **nasza klasa torchvision CIFAR ResNet18**;
  od autorów używamy tutaj funkcji BN, nie klasy sieci. Opcjonalny stub CLIP
  dziedziczony z E029, ścieżka CLIP nieużywana.
- Identyczne A4096 crop/flip, normalizacja, dev5000 i batch128 jak E031.
  A/dev rozłączne; oficjalny test i rezerwa nieużywane. Bez nowych danych.
- Teacher PRE/CMA/repair_bn dla każdego checkpointu:24 nowe kontrole,
  porównanie do zapisanych kontroli E031 (top1 tolerancja1e-12).
  2 zbiory×4seedy×12grup×3stany = **288 ocen +24 kontrole**.
  Seed42 mierzymy ponownie, bo E022 używał innej korekty BN/danych kalibracyjnych.

## Miary i kryteria zamrożone
Szkoda [pp] = top1 teacher po danej procedurze minus top1 skompresowanego
modelu po tej samej procedurze. PRE porównujemy do teacher PRE.
Głębokość to indeks0–11. Wczesne grupy:0–2; późne:9–11 (layer4).
Różnica L−E = średnia szkoda późnych minus średnia szkoda wczesnych.

**H1 primary, CIFAR-100/CMA:** Spearman(głębokość, szkoda)≥0,5 i L−E>0
łącznie na co najmniej3/4 seedów. Jeśli nie, późna wrażliwość nie jest
replikowana według tego kryterium. Raportować też wynik na nowych seedach43–45.

**H2 kontrola specyficzności:** średnia po seedach różnica rho
(CIFAR-100−CIFAR-10)≥0,3 oraz rho wyższe na CIFAR-100 na≥3/4 sparowanych
seedów, po CMA. Te same numery seedów nie oznaczają identycznych reprezentacji
między datasetami. To porównanie opisowe, nie izolacja wpływu danych.

**Wtórne:** te same miary PRE/repair_bn, odzysk BN, pełne rankingi grup,
Spearman między12-elementowymi profilami różnych seedów. Związek L−E
z zyskiem TILT−UNIFORM przy70% z E031, osobno CMA/repair_bn:4 punkty,
bez p-value i bez kryterium sukcesu. Ujemny związek byłby zgodny z hipotezą
konfliktu, ale nie dowodzi przyczynowości ani addytywności szkód.

**Integralność:** hashe wag/kodu/danych/protokołu; strict load;12 grup,
dokładne szerokości i parametry; niepuste klastry; skończone logity/statystyki;
BN nie zmienia parametrów i brak gradientów; kontrola teacher zgodna z E031;
4GPU/exit0; kompletność288+24. W razie błędu zachować run i zgłosić przyczynę.
Progów ani miar primary nie zmieniamy po wynikach.

## Wykonanie
Conda attention, GPU0–3: seed42+i/GPUi, oba zbiory sekwencyjnie na każdym
workerze; folding CPU, BN i ocena GPU. Jeden launcher, monitoring konkretnych
PID, limit90min i0,5GiB artefaktów. Bez treningu, optimizerów, publicznych
wag,96×96 i checkpointów wyjściowych. Każdy run:
`runs/E032/<UTC>_<token>/` (manifest, source, pakiety, logi, SHA256).

Z katalogu projektu w Conda attention:
```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B experiments/2026-11-04_E032_replikacja_wrazliwosci/experiment.py check
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B experiments/2026-11-04_E032_replikacja_wrazliwosci/experiment.py run
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-04_E032_replikacja_wrazliwosci/experiment.py analyze --run-dir runs/E032/<ID>
```

## Ograniczenia i dalsza decyzja
75% kanałów jednej grupy to duża perturbacja, nie lokalna pochodna przy
profilu E031. Różne grupy zmieniają różną liczbę parametrów; późne/wczesne
mają inną rolę strukturalną. Nie rozdzielamy liczby klas, danych i treningu.
Dev używany wcześniej, tylko4seedy, jedna rodzina architektury. Próg H1/H2
opisowy, nie test istotności. Ani wynik dodatni, ani ujemny nie upoważnia
do wznowienia search czy nowego treningu bez osobnej decyzji.
