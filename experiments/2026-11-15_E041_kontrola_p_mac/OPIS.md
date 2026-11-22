# 2026-11-15_E041_kontrola_p_mac — kontrola alokacji przy dopasowanych P i MAC

**Etap A: projekt i test geometrii CPU. Etap B: plan oceny jakości.**
Data protokołu: 2026-11-15 UTC. Punktem odniesienia jest historyczny TILT
E035 przy nominalnie 50% redukcji parametrów. Nowe pytanie względem E039:
czy jego przewaga nad kontrolami utrzymuje się po jednoczesnym dopasowaniu
liczby parametrów i MAC? To kontrola diagnostyczna, nie nowy optymalizator.

## 1. Zakres

- ResNet18 CIFAR-10/CIFAR-100, istniejące checkpointy E011/E015, seedy42–45.
- Jeden budżet: historyczne b50 E035; nie dobieramy nowej siły TILT.
- Cztery kontrolne profile na zbiór, wspólne dla czterech seedów modeli.
- Operator i BN jak E035/E039. Bez treningu i nowych plików wag.
- Faza jakości wyłącznie na istniejącym dev5000; testy E035 i rezerwa
  CIFAR-100 pozostają poza zakresem tej kontroli.

## 2. Etap A — geometryczne generowanie kontroli

Wejściem są wyłącznie plany szerokości z zamrożonych E035/E037/E039.
Pola metryk jakości nie uczestniczą w żadnej decyzji. Nie otwieramy
checkpointów, obrazów, etykiet ani predykcji.

Kolejność 12 grup: upstream, czyli w stage2/3/4 `hidden0 / output / hidden1`.
Nie używamy innej kolejności tabeli projektowej E040. P obejmuje wszystkie
parametry, bez buforów BN; MAC obejmuje Conv/Linear dla wejścia3×32×32.
Metadane geometrii E039 nazywają te same pozycje `stem`, `*_hidden`
i `stage*_out`; kod jawnie sprawdza tę listę aliasów. Nazwy upstream
pozostają zapisane obok, bez permutowania wektora szerokości.

| Zbiór | P(TILT) | MAC(TILT) |
|---|---:|---:|
| CIFAR-10 | 5 582 548 | 377 027 192 |
| CIFAR-100 | 5 580 606 | 376 140 124 |

**Uwaga:** targetem MAC jest TILT, nie znacznie niższy MAC UNIFORM z E040.

Propozycje początkowe to dokładnie cztery zapisane profile SHUFFLED E039,
permutacje101–104, przy b50. Nie losujemy i nie wybieramy nowych permutacji.
Dla każdej propozycji rozwiązujemy na CPU zadanie najbliższego profilu
w znormalizowanej odległości szerokości:

`min sum_g (k_g/n_g − proposal_g/n_g)^2`,

przy ciągłych równaniach `P(k)=0,9995 P(T)` i `MAC(k)=0,9995 MAC(T)` oraz
granicach `floor(0,1 n_g) ≤ k_g ≤ floor(0,9 n_g)`. Używamy SLSQP,
max500iteracji, ftol1e-12, ze startem z TILT. Następnie zaokrąglamy
i deterministycznie naprawiamy najwyżej dwie współrzędne, przeglądając
pary grup i dopuszczalne całkowite szerokości. Wybór naprawy minimalizuje
odległość do rozwiązania ciągłego, nie błąd klasyfikacji.
SLSQP jest solverem lokalnym; nie zgłaszamy gwarancji globalnie najbliższej
projekcji. Dowodem wykonalności są końcowe całkowite koszty, nie status solvera.

Wymagane dopasowanie końcowe, bez przekroczenia obu kosztów:

- `0,999 P(T) ≤ P(k) ≤ P(T)`;
- `0,999 MAC(T) ≤ MAC(k) ≤ MAC(T)`.

Kontrole muszą być odrębne geometrycznie:

- RMS różnic `k/n` względem TILT ≥0,05;
- co najmniej4grupy z bezwzględną zmianą `k/n` ≥0,05;
- parami między kontrolami RMS ≥0,03.

**Bramka A:** wszystkie4kontrole na każdym zbiorze spełniają warunki.
Niepowodzenie nie dowodzi nieistnienia innych dopasowanych profili;
odrzuca tę operacjonalizację kontroli. Maksymalnie8zdań SLSQP, limit
CPU120s. Zachowujemy także nieudane próby, bez zmiany progów po jakości.

Projekcja i zaokrąglenia zmieniają histogram ratio. Są to geometrycznie
dopasowane kontrole wywodzące się z przestawień, a nie czyste permutacje
ani próbka jednostajna całej przestrzeni dopuszczalnych architektur.

## 3. Etap B — plan oceny jakości po dodatniej bramce A

Przed oceną jakości zamrażamy listy szerokości, kod, OPIS i hashe źródeł.
Każdy profil musi przejść kontrolę rzeczywistego P/MAC i zgodności mapowania
grup. Dopasowanie samych wzorów nie zastępuje tej kontroli operatora.
Osobna kontrola operatora na CPU obejmuje8profili i ma limit300s;
nie jest oceną jakości, używa tylko syntetycznego wejścia do hooków MAC.

Podział: cztery niezależne procesy, seed42/43/44/45→GPU0/1/2/3.
Na każdym GPU oba zbiory kolejno. Każdy proces ocenia teacher, U, T
i cztery kontrole; batch128/GPU dla oceny i BN, bez akumulacji gradientów.
Cztery GPU oznaczają cztery seedy treningu źródłowego, a nie dodatkowe
seedy dla każdej kontroli. Gdy nie ma czterech urządzeń, etap B jest zablokowany.

Podstawowa BN: A4096 train, ustalone crop/flip, reset+CMA jak E035.
Wtórna: oryginalne `repair_bn` na tych samych widokach. PRE diagnostycznie.
Odtworzenie teacher/U/T musi zgadzać się z dev E035 (top1 i hash stanu,
NLL w tolerancji1e-5); brak zgodności zatrzymuje ocenę nowych kontroli.
Przed pierwszą nową oceną obowiązuje wspólna bariera odtworzenia wszystkich
czterech procesów na obu zbiorach. W RAM zachowujemy modele źródłowe
i dane dev; nie zapisujemy ich ponownie.

Primary, osobno dla każdego zbioru po CMA:

`D_seed = top1(TILT) − mean_4(top1(kontrola))` w pp.

Bramka kierunku: mean(D)>0 i≥3/4dodatnie seedy na obu zbiorach.
Próg praktyczny mean(D)≥1pp raportujemy osobno, bez ratowania bramki
średnią po zbiorach. Raportujemy wszystkie cztery kontrole i seedy,
TILT−każda kontrola, NLL i bezwzględne top1, P/MAC oraz odległości profili.
Średnia najpierw po kontrolach w obrębie seeda, potem po4seedach;
SD(ddof=1), bez traktowania16modeli jako niezależnych powtórzeń.

## 4. Koszt i reguła zatrzymania

Plan B: 2zbiory×4seedy×7modeli=56konstrukcji/odtworzeń, trzy stany
PRE/CMA/repair, czyli168ocen, w tym96ocen nowych kontroli. Dwa przebiegi
BN na każdy model (CMA i retained repair), łącznie112przebiegów A.
Nie zapisujemy checkpointów nowych profili.

E039 miał504oceny na4GPU w684,24s, w tym Tiny. Proporcja168/504 daje
orientacyjnie228,08s; nie jest gwarancją czasu E041. Twardy limit B:
30min na4GPU (≤2GPU-h rezerwacji), artefakty docelowo≤100MiB.
Monitorujemy konkretne PID/kody zakończenia i manifest, bez `pgrep -f`.

## 5. Interpretacja

Dodatni wynik wspiera wyłącznie przewagę TILT nad tą ustaloną średnią
czterech kontroli przy dopasowanych P/MAC. Nie dowodzi globalnego optimum,
nowości liniowej alokacji, przewagi nad najlepszym alokatorem ani równej
latencji/pamięci. Małe niedowykorzystanie kosztu kontroli raportujemy jawnie.
Przy tych targetach wczesne grupy TILT są blisko górnych granic szerokości;
różnorodność kontroli jest większa w późnych grupach. Nie jest to czysta
izolacja ochrony początku sieci ani jednorodna próbka wszystkich profili.

Ujemny wynik nie unieważnia E035 ani nie dowodzi równoważności. Dev był
już używany; to kontrola rozwojowa, nie nowa niezależna walidacja. Po tej
jednej kontroli wracamy do oceny zakresu artykułu, bez serii wariantów
ratunkowych i bez zmiany progu na podstawie jakości.
