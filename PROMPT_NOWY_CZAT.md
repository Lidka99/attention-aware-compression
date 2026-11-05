Pracuj w /home/users/s224574/attention/badania_atencja_cnn.

Najpierw przeczytaj: AGENTS.md, STATUS_BADAN.md, NOTATKA_KONTYNUACJA.md,
README.md, CONTRIBUTION_STATEMENT_2026-10-30.md,
literature/AUDYT_ALOKACJA_FOLDING.md, WYNIKI E031, E032 i E033 oraz
OPIS E034.

## Cel badania
Adresujemy otwarty problem autorów foldingu (Wang et al. 2025): optymalizację
budżetu kompresji per warstwa. Cut Less, Fold More stosuje jednolite ratio
i tej osi nie bada. Zmienną jest alokacja budżetu między strukturalne grupy
ResNet18/PreActResNet18, przy dopasowanej globalnej liczbie parametrów
i bez fine-tuningu.

## Co jest już ustalone
- E031 — oficjalny operator autorów, CIFAR-10/100/Tiny, kompresja 30/70%:
  zgodność LEGACY/LOCAL z upstream **24/24**, kierunek TILT **11/12**.
  Wyjątek: CIFAR-100/70%/`repair_bn` tylko 2/4 seedów. CMA przechodzi 6/6.
  Wynik mieszany; nie uogólniać na wszystkie warunki BN.
- E032 — późna wrażliwość CIFAR-100 po BN replikuje się 4/4 seedów
  (rho 0,622–0,629; L−E +16,5 do +17,5 pp); na CIFAR-10 brak trendu.
  H1/H2 spełnione. Nie wyjaśnia jednak różnic zysku TILT między seedami
  (rho L−E vs zysk: 0,00 CMA / +0,20 repair_bn).
- E033 — interakcja wczesne/środkowe vs późne grupy przy profilach E031:
  **primary negatywny**. EP−T na CIFAR-100/b70/CMA +0,135 pp (2/4);
  repair_bn +0,95 pp (3/4). Na CIFAR-10/Tiny EP−T −7,1/−5,1 pp (0/4).
  Prosta hipoteza „CIFAR-100 przez layer4” niepotwierdzona; nie budować
  na niej nowego alokatora.
- E034 — druga architektura PreActResNet18, Phase 1 zatwierdzona przez
  użytkowniczkę: CIFAR-10/100, 4 seedy, 200 epok, SGD lr0.1/momentum0.9/
  wd5e-4, batch128/GPU, bez fine-tuningu. **Trening został uruchomiony**
  jako `runs/E034/train_20261105T014209Z_21b44f52` (start 2026-11-05
  01:42 UTC). Sprawdź manifest/status: może jeszcze działać albo być
  ukończony. Kod foldingu jest gotowy i przeszedł kontrolę CPU:
  `experiments/2026-11-05_E034_preactresnet18/fold_preact.py`.
- Zamknięte negatywnie: miary grupowania (E017/E018/E021), AUTO50
  (E016 S2), search KL/disagreement (E024–E028). E023 anulowany.
  Nie wracać do search bez nowej hipotezy i wyraźnej zgody.
- E014/E019 pozostają najmocniejszymi wynikami alokacji; CIFAR-100 jest
  słaby, a jakość bezwzględna przy 50–70% kompresji niska.

## Twoje zadanie
1. **Dokończ E034 Phase 1.**
   - Sprawdź `runs/E034/train_20261105T014209Z_21b44f52/manifest.json`
     i `seed_*/status.json`. Jeśli trening trwa: monitoruj konkretny PID
     i manifest, bez `pgrep -f`, bez równoległych zadań GPU. Nie uruchamiaj
     foldingu przed ukończeniem.
   - Jeśli run failed: zachowaj go, wskaż przyczynę, nie restartuj po cichu.
   - Jeśli completed: sprawdź 4/4 seedy×2 zbiory, 200 epok, exit0,
     dev top1 i brak testu/rezerwy. Uruchom folding:
     `CUDA_VISIBLE_DEVICES=0,1,2,3 python -B experiments/2026-11-05_E034_preactresnet18/fold_preact.py fold --train-run runs/E034/train_20261105T014209Z_21b44f52`
     Następnie analizę:
     `CUDA_VISIBLE_DEVICES='' python -B experiments/2026-11-05_E034_preactresnet18/fold_preact.py analyze --fold-run runs/E034/<fold-ID>`
   - Kryteria E034 (zamrożone w OPIS): oficjalny `PreActResNet18_ModelFolding`
     (commit 6946915), 13 grup, budżety 30/50/70% redukcji; TILT−UNIFORM
     przy 50/70 średnia ≥+2,0 pp i ≥3/4 dodatnich, przy 30% >0 i ≥3/4,
     po CMA oraz `repair_bn`; CIFAR-100 raportowany osobno. Jeśli CIFAR-100
     nie przejdzie, wynik częściowy, bez twierdzenia o uniwersalności.
   - Zapisz `WYNIKI.md` i `PODSUMOWANIE.md` w folderze E034; zaktualizuj
     `STATUS_BADAN.md`, `NOTATKA_KONTYNUACJA.md` i README. Uczciwie pokaż
     słaby CIFAR-100 i spadek jakości bezwzględnej.
   - Zrób lokalne commity. `git push` jest zablokowany brakiem danych
     logowania; push po stronie użytkownika. Nie przepisuj historii.

2. **Po E034:** przedstaw decyzję, czy Phase 2 (Tiny ImageNet / 96×96)
   ma sens, wraz z kosztem, przepisem i kryteriami. **Nie uruchamiaj
   nowego treningu bez wyraźnej zgody użytkowniczki.** Decyzja publikacyjna
   dopiero po wynikach; nie deklaruj publikowalności na podstawie tabel.

Zasady:
- 4 GPU, Conda attention, dysk ok. 3,5 GB wolne; bez nowych datasetów,
  bez 96×96, bez publicznych wag; oficjalny test i rezerwa nietknięte.
- Każdy run w unikalnym `runs/E034/<UTC>_<token>/`; manifest, hashe,
  source copy, logi. Nie zmieniaj E014–E033 ani nie usuwaj runów.
- Kod autorów tylko z przypiętego commita, repo upstream niezmienione.
  Raportuj, gdzie użyto naszego modelu zamiast ich klasy (Tiny).
- Podagenty tylko do małych, wydzielonych zadań; bez równoległych zmian
  tych samych plików i bez konkurencyjnych zadań GPU.
