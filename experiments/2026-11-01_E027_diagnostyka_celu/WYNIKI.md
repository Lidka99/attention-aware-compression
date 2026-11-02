# E027 — diagnostyka funkcji celu optymalizacji alokacji
Status: **completed, kryterium primary niespełnione o 0,065 pp**, 2026-11-01.
Pierwszy search, który bije TILT na **4/4 seedach** i średnio o +0,935 pp,
ale poniżej zamrożonego progu +1,0 pp. Nie zmieniamy progu po wyniku.

## Pytanie
E026 pokazał, że TILT ma **gorsze KL** od UNIFORM na 4/4 seedów, a **lepszą
trafność** na 4/4. Jeśli KL jest antyzgodne z top1, optymalizacja KL nie może
znaleźć dobrej alokacji. E027 rozstrzyga: ograniczeniem jest **funkcja celu**
czy **przestrzeń/budżet wyszukiwania**?

Nadal adresujemy otwarty problem budżetu kompresji per warstwa w foldingu.
To diagnostyka kryterium, nie miary grupowania i nie trening wag.

## Co zmieniono względem E026
Wszystko stałe: CIFAR-100, checkpointy, budżet50%, grupy, operator, PCA/HKMeans,
solver, pary donor→receiver, start z UNIFORM/TILT, 2×24, limit50, cache,
BN A4096+aug+CMA. Zmiana: dla każdego kandydata liczymy **trzy** miary z tego
samego forwardu na B2048 (bez etykiet):
- `kl` — jak E024/E026;
- `disagreement` — frakcja argmax≠argmax(teacher);
- `teacher_ce` — cross-entropy względem twardej etykiety teacher.
**Primary (zamrożone): wybór minimalizujący `disagreement`.**

## Wykonanie
- [OPIS.md](OPIS.md), frozen przed runem.
- Run: `runs/E027/20261101T143927Z_ce68f402`.
- 4/4 seedy, **200/200 ocen**, GPU0–3/seedy42–45, exit `[0,0,0,0]`.
- Wall-time **1157,48s (19,3min)**, artefakty <1MiB. Limit120min spełniony.
- Po wyborze i hashu `selection.json` dev oceniono dla **każdego** z50
  kandydatów. Dev nie brał udziału w search ani wyborze; służy wyłącznie
  do analizy post hoc selektora i pułapu puli.
- Referencje UNIFORM/TILT/OLD_SEARCH/E026_SEARCH pod tą samą BN z E025/E026.
- **Kontrola determinizmu:** pierwszy run `20261101T141932Z_6d45f5f4`
  przerwano po searchu na etapie składania referencji (błąd atrybutu
  w kodzie, nie w metodzie). Zachowany. Powtórny run dał **identyczne**
  szerokości, kolejność, wszystkie trzy miary (≤1e-12), wybory i dev
  wszystkich200 kandydatów; różniły się tylko pola czasu.
- Test i rezerwa nietknięte; brak treningu i nowych danych.

## Wynik: dev top1, średnia po4 seedach
| Metoda | Top1[%] mean±SD | NLL | KL_C |
|---|---:|---:|---:|
| UNIFORM | 53,855±1,509 | 2,1501 | 1,2011 |
| TILT | 55,550±1,105 | 2,2198 | 1,2503 |
| OLD_SEARCH (E024) | 54,715±0,879 | 2,0843 | 1,0639 |
| E026_SEARCH (KL+BN) | 55,155±0,689 | 2,0455 | 1,0517 |
| **NEW_SEARCH (disagreement)** | **56,485±0,699** | 2,1445 | 1,1747 |

| Seed | NEW−TILT | NEW−UNIFORM | NEW−OLD | NEW−E026 |
|---|---:|---:|---:|---:|
| 42 | +0,60 | +2,78 | +1,48 | +1,92 |
| 43 | +1,42 | +2,12 | +0,16 | −0,28 |
| 44 | +0,56 | +1,42 | +2,54 | +1,66 |
| 45 | +1,16 | +4,20 | +2,90 | +2,02 |
| Średnia±SD | **+0,935±0,424** | **+2,630±1,185** | **+1,770±1,231** | **+1,330±1,084** |
| Dodatnie | **4/4** | **4/4** | **4/4** | 3/4 |

### Kryteria zamrożone
| Kryterium | Wynik |
|---|---|
| Primary: NEW−TILT≥+1pp i≥3/4 | **NIE — +0,935pp (4/4), brakuje0,065pp** |
| NEW−UNIFORM średnio>0 | TAK |
| Cel lepszy: alignment(disagreement) > KL i > teacher_ce | TAK, 4/4 seedów |
| Pułap puli: oracle−TILT≥+1pp | TAK: **+1,660±0,208pp, 4/4** |

## Co wynika z diagnostyki
1. **Funkcja celu miała znaczenie.** Średnia zgodność z dev top1 (alignment,
   ujemny Spearman odwrócony): `disagreement` **0,634**, `kl` 0,546,
   `teacher_ce` 0,544. Na **każdym** seedzie disagreement wyprzedza oba
   pozostałe. To wyjaśnia, dlaczego E024/E026 nie mogły dogonić TILT:
   optymalizowały miarę, która rankingu trafności nie odtwarza.
2. **Pula zawiera alokacje lepsze od TILT.** Najlepszy z50 odwiedzonych
   kandydatów bije TILT o **1,66pp** średnio, 4/4 seedy. Przestrzeń
   i mechanizm propozycji **nie są** wąskim gardłem przy tym budżecie.
   To pułap optymistyczny na użytym dev, nie metoda.
3. **Selektor nadal ma rezerwę.** Regret wyboru: disagreement 0,73pp,
   KL 0,86pp średnio. Poprawa selektora może dać kolejne dziesiąte pp.
4. **Cel zmienia kompromis:** NEW ma najlepszy top1, ale gorsze NLL/KL niż
   E026. To spójne — disagreement optymalizuje decyzje, nie rozkład.
   Nie łączymy tych metryk w jeden wynik.

## Ograniczenia
- Primary **niespełnione**; nie zaokrąglamy do sukcesu i nie proponujemy
  obniżenia progu.
- Dev był użyty do diagnostyki post hoc; dalsze strojenie na tym dev byłoby
  zanieczyszczeniem. Następny test musi być **przeniesieniem** na ustawienie,
  którego nie użyto do projektowania kryterium (np. CIFAR-10/50%), z tą samą
  zamrożoną regułą.
- Pula jest adaptacyjna i zbudowana przez disagreement, więc porównanie
  selektorów i oracle dotyczy tej puli, nie dowolnego search.
- Jeden zbiór i budżet,4checkpointy, jedna architektura.

## Wniosek
Kierunek „optymalizacja budżetu per warstwa” **nie jest wyczerpany**:
zmiana bezetykietowej funkcji celu z KL na zgodność decyzji dała pierwszy
search bijący TILT na wszystkich seedach. Ale kryterium sukcesu nie zostało
spełnione, a wynik wymaga przeniesienia poza CIFAR-100, zanim nazwiemy go
metodą. Żaden kolejny eksperyment nie został uruchomiony w E027.
