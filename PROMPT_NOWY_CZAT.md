Pracuj w /home/users/s224574/attention/badania_atencja_cnn.

Najpierw przeczytaj: AGENTS.md, STATUS_BADAN.md, NOTATKA_KONTYNUACJA.md,
CONTRIBUTION_STATEMENT_2026-10-30.md, literature/AUDYT_ALOKACJA_FOLDING.md
oraz experiments/2026-11-01_E029_oficjalny_operator/WYNIKI.md i
experiments/2026-11-02_E030_oficjalny_tiny/WYNIKI.md.

## Cel badania
Adresujemy otwarty problem wskazany przez autorów foldingu (Wang et al. 2025):
optymalizację budżetu kompresji per warstwa. Cut Less, Fold More stosuje
jednolite ratio i tej osi nie bada. Zmienną jest alokacja budżetu między
12 strukturalnych grup ResNet18, przy dopasowanej globalnej liczbie
parametrów i bez fine-tuningu.

## Co jest ustalone i zamknięte
- TILT+0.30 bije UNIFORM na trzech zbiorach przy 30/50/70% parametrów:
  CIFAR-10 +0,87/+3,77/+11,27 pp; Tiny +3,71/+10,32/+7,60 pp;
  CIFAR-100 +0,81/+1,96/+1,60 pp (słabszy, kryterium E015 niespełnione).
- E022: mechanizm częściowy. BN odzyskuje 79/67/38% szkody przy 75%
  kompresji grupy; TILT nie chroni prosto grup o dużej szkodzie post-BN;
  ANTI-TILT katastrofalny. Profile grup tylko seed42.
- E025: korekta BN wymaga augmentacji. A4096 crop/flip + CMA/reset
  odtwarza teacher; brak augmentacji obniżał go o 5,83 pp. EMA0.1 po
  resecie przy 32 batchach zawodzi. To obowiązująca procedura.
- E029/E030: **oficjalny operator autorów** (commit 6946915) sprawdzony
  na naszych checkpointach CIFAR-10/100/Tiny przy ~50%. Nasza referencja
  E012/E014 jest identyczna z upstream. TILT−UNIFORM na operatorze
  autorów: +4,00/+1,70/+9,41 pp (CMA) oraz +3,75/+1,46/+9,18 pp przy
  ich repair_bn; wszędzie 4/4 seedów.
- **Zamknięte negatywnie:** miary grupowania (E017/E018/E021), AUTO50
  (E016 S2), search KL i disagreement jako nowy optymalizator
  (E024–E028). E027 dał +0,935 pp vs TILT na CIFAR-100 (4/4), ale poniżej
  zamrożonego progu +1 pp; E028 na CIFAR-10 tylko +0,22 pp. Nie wracać
  do search bez nowej hipotezy i zgody.
- E023 anulowany przed uruchomieniem.

## Zadanie
Zaproponuj krótki plan i wykonaj **E031: rozszerzenie audytu oficjalnego
operatora na budżety 30% i 70%** na istniejących checkpointach
(CIFAR-10/100/Tiny), bez treningu. Uzasadnienie: przy 70% kompresji
klasteryzacja jest bardziej niszcząca, więc zgodność operatora i kierunek
alokacji wymagają sprawdzenia również tam. Jeśli uznasz, że inny
bezkosztowy treningowo audyt jest ważniejszy, przedstaw uzasadnienie
przed uruchomieniem.

Zasady:
- Osobny folder experiments/<data>_E031_.../, OPIS i kryteria zamrożone
  PRZED runem; najpierw krótki plan w rozmowie.
- Tylko istniejące modele/dane; oficjalny test i rezerwa nietknięte.
- 4 GPU, Conda attention, dysk ~4,2 GB wolne; bez nowych datasetów,
  bez 96×96, bez publicznych wag.
- Nie zmieniaj wyników E014–E030 ani nie usuwaj runów. Każdy run ma
  unikalny katalog runs/E031/<UTC>_<token>/; hashe i manifest obowiązkowe.
- Jeśli używasz kodu autorów, przypnij commit i nie modyfikuj repo
  upstream; raportuj, gdzie użyto naszego modelu zamiast ich klasy.
- Podagenty tylko do małych, wydzielonych zadań; bez równoległych zmian
  tych samych plików i bez konkurencyjnych zadań GPU.
- Po zakończeniu: WYNIKI.md, PODSUMOWANIE.md, aktualizacja
  STATUS_BADAN.md i NOTATKA_KONTYNUACJA.md.

## Kolejność dalszych decyzji
1. E031 (bez treningu) — domknięcie audytu operatora na 30/70%.
2. **Druga architektura (np. PreActResNet18) wymaga nowego treningu
   i WYŁĄCZNIE Twojej wyraźnej zgody.** Najpierw przedstaw koszt, przepis
   i kryteria; nie startuj.
3. Decyzja publikacyjna po 1–2. Nie deklarować publikowalności na
   podstawie samych tabel; uczciwie pokazać słaby CIFAR-100 i spadek
   jakości bezwzględnej przy 50–70%.

## Stan repo
Commity lokalne są zrobione; `git push` blokowany brakiem danych
logowania do GitHub. Nie przepisuj historii; push po stronie użytkownika.
