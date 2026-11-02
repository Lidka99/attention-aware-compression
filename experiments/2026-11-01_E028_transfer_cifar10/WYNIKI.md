# E028 — transfer reguły E027 na CIFAR-10
Status: **completed, transfer nieudany**, 2026-11-01. Reguła wybrana na
CIFAR-100 (disagreement) na CIFAR-10 **wyrównuje TILT**, nie bije go.
Ważny wynik poboczny: na CIFAR-10 TILT jest **blisko pułapu** odwiedzonej
puli alokacji, więc brak przewagi nie wynika wyłącznie ze słabego selektora.

## Pytanie i co było stałe
Po E027 (CIFAR-100: +0,935pp vs TILT, 4/4, poniżej progu+1pp) sprawdzamy
**przeniesienie** tej samej zamrożonej reguły na zbiór nieużyty do jej
projektowania. Bez zmian: operator, 12 grup, PCA/HKMeans, solver, pary
donor→receiver, RNG, start UNIFORM/TILT,2×24, limit50, cache, stop1e-6,
BN A4096+aug+CMA, wybór po `disagreement`. Zmienia się tylko zbiór
(CIFAR-10, ResNet18 3×3/no-maxpool, fc10) i checkpointy E011.

Budżet: nominalnie50%, target=faktyczny UNIFORM **5 584 580** parametrów
(teacher11 173 962). Wszystkie profile≤target.

## Wykonanie i integralność
- [OPIS.md](OPIS.md), frozen przed runem.
- Run: `runs/E028/20261101T154700Z_086b311e`.
- 4/4 seedy, **176 ocen** (seed43 zatrzymał się po26 — brak poprawy),
  GPU0–3/seedy42–45, exit `[0,0,0,0]`.
- Wall-time **1143,73s (19,1min)**, artefakty <1MiB.
- Dev oceniany dla każdego odwiedzonego kandydata dopiero po hashu
  `selection.json`; nie wpływa na search ani wybór.
- UNIFORM/TILT policzone od nowa pod tą samą BN; nie kopiowano tabel E014.
- **Kontrola BN na CIFAR-10:** teacher raw95,46% → teacher+BN95,44%
  (per seed różnice ≤0,16pp). To niezależne potwierdzenie, że poprawiona
  procedura z E025 nie psuje modelu na tym zbiorze.
- Teacher raw zgodny z historycznym baseline E011 na wszystkich seedach.
- Test i rezerwa nietknięte; brak treningu.

## Wynik
| Metoda | Top1[%] mean±SD | NLL | KL_C |
|---|---:|---:|---:|
| UNIFORM | 89,435±0,630 | 0,3504 | 0,1305 |
| TILT | 93,430±0,322 | 0,2311 | 0,0416 |
| NEW_SEARCH | **93,650±0,305** | **0,2250** | **0,0401** |

| Seed | NEW−TILT | NEW−UNIFORM | Pułap puli−TILT |
|---|---:|---:|---:|
| 42 | −0,04 | +4,50 | +0,50 |
| 43 | 0,00* | +4,00 | +0,12 |
| 44 | +0,78 | +4,66 | +1,08 |
| 45 | +0,14 | +3,70 | +0,48 |
| Średnia±SD | **+0,220±0,381** | **+4,215±0,444** | **+0,545±0,397** |
| Dodatnie | **2/4** | **4/4** | **4/4** |

\* seed43: search nie poprawił TILT i wybrał dokładnie profil TILT po26 ocenach.

### Kryteria zamrożone
| Kryterium | Wynik |
|---|---|
| Primary: NEW−TILT≥+1pp i≥3/4 | **NIE:** +0,220pp, 2/4 |
| NEW−UNIFORM średnio>0 | TAK: +4,215pp, 4/4 |
| Alignment(disagreement) najwyższy | TAK: 0,661 vs KL0,573 i CE0,571 |
| Pułap puli−TILT≥+1pp | **NIE:** +0,545pp |

**Transfer nieudany.** Reguła działa, ale nie daje przewagi nad TILT.

## Co to znaczy
1. **TILT jest bardzo mocny na CIFAR-10.** Jego przewaga nad UNIFORM
   wynosi **+3,995pp** i odtwarza kierunek E014 (+3,77pp przy innej BN).
   Search nie ma tu pola do poprawy: najlepszy z50 kandydatów bije TILT
   tylko o0,55pp.
2. **Brak przewagi to nie tylko wina selektora.** Regret disagreement
   wynosi0,33pp średnio, a pułap0,55pp. Na CIFAR-100 pułap był+1,66pp,
   więc **headroom zależy od zbioru**.
3. **Zmiana celu KL→disagreement nie jest uniwersalna.** Na CIFAR-100 dała
   najlepszy dotąd wynik search (+0,935pp, 4/4, nadal <+1pp); na CIFAR-10
   kończy się remisem z TILT. To osłabia tezę „nowy optymalizator bije TILT”.
4. **Stabilny wynik projektu pozostaje po stronie alokacji, nie search.**
   TILT: CIFAR-10 +3,995pp, CIFAR-100 +1,96pp, Tiny +10,32pp nad UNIFORM
   przy50% (ostatnie dwa z wcześniejszych, zamrożonych protokołów).
5. **Cel bezetykietowy nadal ma sens tam, gdzie jest headroom.** E027 pokazał
   +1,66pp pułapu na CIFAR-100; E028 pokazuje, że na CIFAR-10 go prawie nie ma.
   Dalsze ulepszanie selektora ma sens tylko na zbiorach z headroomem,
   a dev jest już zanieczyszczony diagnostyką.

## Ograniczenia
Jeden budżet/architektura,4checkpointy. CIFAR-10 dev był używany wcześniej
w projekcie, choć nie do projektowania tej reguły. Pułap puli jest
optymistyczny i zależy od użytego dev. Wynik dotyczy tej heurystyki,
tego operatora i tego kosztu; nie dowodzi, że nie istnieje lepsza alokacja.
Nie zmieniano progu po wyniku.

## Decyzja
**Zatrzymujemy linię „nowy optymalizator KL/disagreement” jako kierunek
główny.** Dwa zbiory, dwa cele, brak stabilnej przewagi nad TILT.
Wkład pozostaje: alokacja per warstwa jako odpowiedź na otwarty problem
autorów foldingu + prosta reguła + kontrole negatywne + rola BN.
Następne kroki powinny być publikacyjne, nie kolejne warianty search:
oficjalny baseline1:1 i druga architektura. Żaden nowy eksperyment ani
trening nie został uruchomiony w E028.
