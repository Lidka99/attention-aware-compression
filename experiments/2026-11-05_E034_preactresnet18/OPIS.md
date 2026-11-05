# 2026-11-05_E034_preactresnet18

Status: **frozen przed treningiem**, 2026-11-05 UTC. Uruchomienie Phase 1
zatwierdzone przez użytkowniczkę (odpowiedź „Tak — CIFAR-10/100”).

## Cel
Sprawdzić, czy przewaga alokacji TILT nad UNIFORM utrzymuje się na **drugiej
architekturze** — PreActResNet18 — na CIFAR-10 i CIFAR-100, bez fine-tuningu.
To test ogólności kierunku alokacji, nie nowy search ani nowa metoda.

## Architektura i źródło
- PreActResNet18 z `models/preact_resnet.py` w `osaukh/folding_as_projection`,
  commit **694691556e8314a835178a741938dd3d4d80c171** (ten sam klon E029,
  `runs/E029/upstream_6946915`, pliki niezmienione). Architektura jest jawna
  i pinowana; trening od losowej inicjalizacji, bez publicznych wag.
- Operator foldingu: niezmieniony `PreActResNet18_ModelFolding` z tego commita.
  Mapa grup `get_axis_to_perm_PreActResNet18(override=False)` daje **13 grup**
  w kolejności topologicznej; TILT to nasz adapter `keep_ratio` per grupa,
  nie wariant autorów.
- GA `linear.bias` nie jest w mapie osi; po foldingu odtwarzamy bias z modelu
  źródłowego, jeśli operator go zgubił. To jawna kontrola, analogiczna do E029.

## Trening Phase 1
- CIFAR-10 i CIFAR-100, seedy42–45, 200 epok, SGD lr0.1, momentum0.9,
  weight_decay5e-4, CosineAnnealingLR do0, batch128/GPU (efektywny128),
  RandomCrop32/padding4 + horizontal flip, normalizacja jak E011/E015.
- Dane i podziały: istniejące E011 (CIFAR-10) i E015 (CIFAR-100), train/dev
  45k/5k i40k/5k. Bez nowych danych, testu, rezerwy i 96×96.
- 4 niezależne seedy na GPU0–3; każdy worker trenuje oba zbiory sekwencyjnie.
  Checkpointy `last.pt` tylko state_dict+metadata, po każdej epoce atomowo.
  Zapisujemy `metrics.jsonl`, `summary.json`, konfigurację, hashe i manifest.
- Sanity (nie kryterium naukowe): ukończone200 epok, dev top1 skończony,
  CIFAR-10≥90%, CIFAR-100≥65%. Baseline raportujemy, nie selekcjonujemy seeda.
- Koszt szacowany: ok.3 h na4 GPU (CIFAR-10 ~75 min, CIFAR-100 ~66 min
  dla ResNet18; PreAct może być nieco droższy), checkpointy<0,5 GB.
  Twardy limit launchera:5 h. Jeśli limit/dysk/GPU zawiodą, run failed
  zachowujemy i raportujemy, bez automatycznego restartu.

## Folding i ocena (po treningu)
- Budżety **30/50/70% redukcji parametrów** (pozostaje70/50/30%), UNIFORM
  i TILT+0.30; wspólny target = faktyczna liczba parametrów UNIFORM;
  TILT≤target, różnica≤0,1%; dokładne szerokości13 grup zapisane przed oceną.
- Ocena tylko dev: PRE, CMA/reset (A4096 crop/flip E031) i oryginalne
  `repair_bn` (retained EMA0.1) na tym samym A. Teacher PRE/CMA/repair_bn.
- 2 zbiory×3 budżety×4 seedy×2 profile×3 stany =144 oceny +24 kontrole.
  Bez gradientów, treningu po foldingu, testu i rezerwy.
- Koszt: ok.10–15 min na4 GPU; artefakty<0,5 GB.

## Kryteria zamrożone
1. **Trening:** 4/4 seedy×2 zbiory completed,200 epok, exit0; dev baseline
   spełnia sanity; brak testu/rezerwy.
2. **Folding:** strict load,13 grup, dokładne parametry, niepuste klastry,
   skończone modele,4GPU/exit0.
3. **Primary allocation:** TILT−UNIFORM na PreActResNet18, dla każdego
   zbioru osobno: przy50% i70% średnia≥+2,0 pp i≥3/4 dodatnich seedów,
   po CMA; przy30% średnia>0 i≥3/4. To samo raportujemy dla `repair_bn`.
4. **Secondary:** NLL, per-seed, różnica BN, krzywe; CIFAR-100 raportowany
   osobno i bez upiększania. Jeśli CIFAR-100 nie przejdzie, nie twierdzimy
   uniwersalności; jeśli przejdzie tylko CIFAR-10, wynik częściowy.
5. Nie zmieniamy progów po wynikach. Brak search, AUTO50, nowych miar
   grupowania i fine-tuningu.

## Ograniczenia
Jedna dodatkowa architektura, jeden przepis treningu,4seedy, dev używany
wcześniej; brak niezależnego testu i pełnej reprodukcji paperu1:1.
PreActResNet18 ma inny przepływ i13 grup, więc TILT nie jest tym samym
wektorem co w ResNet18; to test kierunku, nie identyczności profili.
Wynik nie jest dowodem optymalności ani gotowości publikacyjnej.

## Odtworzenie (Conda attention)
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-05_E034_preactresnet18/experiment.py check
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-05_E034_preactresnet18/experiment.py train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-05_E034_preactresnet18/experiment.py fold --train-run runs/E034/<train-ID>
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-05_E034_preactresnet18/experiment.py analyze --fold-run runs/E034/<fold-ID>
```
