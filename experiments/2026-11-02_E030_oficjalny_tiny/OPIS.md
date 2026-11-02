# 2026-11-02_E030_oficjalny_tiny

Status: **frozen**, 2026-11-02, przed oceną nowych modeli na dev.

## Cel
Rozszerzyć audyt E029 na **trzeci zbiór**: Tiny ImageNet64×64, budżet≈50%,
istniejące checkpointy E019, bez treningu. Pytanie: czy zgodność naszych
operatorów z upstream oraz dodatni efekt alokacji utrzymują się poza
CIFAR-10/100? Adresujemy otwarty problem budżetu per warstwa w foldingu.

E030 **nie** jest nową metodą ani ponownym search. To kontrola operatora.

## Architektura i uczciwa granica
Checkpointy E019 to torchvision ResNet18 ze standardowym stem7×7+maxpool,
wejście64×64, fc512→200 (11 279 112 parametrów). Upstream w sprawdzonym
commicie udostępnia klasę CIFAR ResNet18 (stem3×3, avg_pool2d4), więc
**nie używamy jego klasy modelu dla Tiny**. Używamy jego **niezmienionego
operatora** `ResNet18_ModelFolding.apply/compress_function` na naszym modelu:
mapa grup `get_axis_to_perm_ResNet18(override=False)` odwołuje się wyłącznie
do nazw conv1/bn1/layerX/downsample/fc, które są zgodne z torchvision.
To „oficjalny operator na naszym modelu”, nie „oficjalna architektura Tiny”.
CLIP stub jak w E029; żadnej zmiany matematyki CNN.

## Wejścia i dane
Commit `694691556e8314a835178a741938dd3d4d80c171`, klon E029 niezmieniony
i sprawdzany (git status czysty).
Checkpointy E019 seed42–45, hashe i baseline z inwentarza E020.
Dane: `data/tiny-imagenet/cache_train.npz` + `splits.json`, walidowane
modułem E019 (`load_tiny_train`, `load_split`), bez pobierania i bez testu.
Z train90k losujemy **A4096** (RNG20261103) tylko do BN; dev10k z pliku.
Rezerwa i oficjalny val/test nietknięte. A∩dev=∅.
Augmentacja BN: deterministyczny **RandomResizedCrop64 scale(0.4,1.0)** +
horizontal flip, RNG20261103+indeks obrazu, potem normalizacja Tiny.
To rodzaj zgodny z treningiem E019 (RRC+flip), inny niż crop CIFAR;
ta różnica jest jawna i wspólna dla wszystkich ramion. Plain B/C nie są
potrzebne, bo E030 nie prowadzi search.

## Ramiona i pomiar
7 ramion na seed, identyczne szerokości per profil:
UP_RAW_U (natywny z losowym biasem FC), UP_FIXED_U, LEGACY_U, LOCAL_U,
UP_FIXED_T, LEGACY_T, LOCAL_T. UP_FIXED_T to nasz adapter `keep_ratio`
per grupa na operatorze upstream; **nie** oficjalny wariant TILT.
Dla każdego ramienia: pre-BN oraz świeże kopie po CMA/reset i po
niezmienionym upstream `repair_bn` (retained EMA0.1) na tym samym A.
Teacher: pre/CMA/UP_REPAIR. Parametry niezmienione, bez gradientów.
7×3×4=84 oceny +12 kontroli teacher. Folding CPU, ocena/BN GPU.

## Kryteria zamrożone
1. **Zgodność jakości:** LEGACY i LOCAL kontra UP_FIXED przy tych samych
   profilach i CMA: średnia |Δtop1|≤0,5pp i maksimum≤1pp po4 seedach,
   osobno U i T.
2. **Kierunek alokacji:** UP_FIXED_T−UP_FIXED_U średnia>0 i≥3/4 dodatnich
   seedów, osobno dla CMA i UP_REPAIR.
3. **Integralność:** strict load, 12 grup, dokładne parametry, niepuste
   klastry, skończone modele, 4GPU/exit0, limity90min/0,5GiB.
4. Raportujemy również zgodność wag/logitów/partycji i efekt biasu FC,
   bez wybierania korzystnej procedury BN. Wynik negatywny zachowujemy.
Nie zmieniamy progów po dev.

## Wykonanie i ograniczenia
Conda attention,4×GPU0–3, seed42+i/GPUi, jeden launcher, bez treningu,
nowych danych, publicznych wag,96×96, checkpointów wyjściowych, testu
i rezerwy. Hashe kodu/protokołu/wejść, PID, pakiety i logi zapisane.
Ograniczenia: jeden budżet, jedna architektura Tiny (inny stem niż CIFAR),
jeden seed augmentacji, dev używany wcześniej w projekcie. To audyt
zgodności i kierunku na naszych wagach, nie reprodukcja paperu1:1.

## Odtworzenie
```
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-02_E030_oficjalny_tiny/check.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-02_E030_oficjalny_tiny/run.py
CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-02_E030_oficjalny_tiny/analyze.py --run-dir runs/E030/<run-ID>
```
