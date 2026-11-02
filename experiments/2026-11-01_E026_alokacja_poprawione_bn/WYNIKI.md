# E026 — optymalizacja alokacji z poprawioną korektą BN

Status: **completed, kryterium primary niespełnione**, 2026-11-01.
Poprawiona BN podczas wyszukiwania zmienia wybrane profile i poprawia
średni top1 względem starego SEARCH, lecz nowy SEARCH nadal nie pokonuje
TILT. KL i NLL są lepsze; nie zmienia to zamrożonego werdyktu primary.

## Cel i kontrast
Adresujemy otwarty problem optymalizacji budżetu foldingu per warstwa.
E026 izoluje wpływ poprawki BN z E025 na **wybór alokacji**: te same
checkpointy CIFAR-100, globalny budżet około50%, grupy, operator, cel KL,
reguła wyszukiwania, seedy par i limit50 ocen jak E024. Jedyna zmiana
naukowa: A4096 do BN ma crop/flip zamiast obrazów bez augmentacji.

UNIFORM, TILT i OLD_SEARCH mają **identyczne poprawione BN** jak NEW_SEARCH.
Ich wyniki dev/KL_C pochodzą z E025, nie z gorszej procedury E024.
Nie powtarzano search E024 ani oceny gotowych referencji. Nowe KL_B
UNIFORM/TILT było konieczne do wyboru punktu startowego; E025 go nie zapisywał.

## Wykonanie i kontrole
- Zamrożony przed runem [OPIS.md](OPIS.md).
- Run: `runs/E026/20261101T133903Z_bbd75e05`.
- **4/4 seedy completed, 50 ocen/seed, 200/200 ocen search**, cztery
  GPU0–3/seedy42–45, exit codes `[0,0,0,0]`. Cztery nowe oceny finalnie
  wybranych profili; referencje UNIFORM/TILT/OLD_SEARCH/teacher reuse.
- Wall-time **578,79s (9,65min)**; artefakty około656KiB, poniżej0,5GiB.
  4×RTX2080Ti, Conda attention, batch128/GPU/przebieg; wersje w manifest
  i `pip-freeze.txt`. Brak treningu wag, checkpointów wyjściowych i nowych danych.
- Kontrola CPU odtworzyła **pełną sekwencję50 kandydatów i końcowy wybór
  E024 na każdym seedzie**, podając zapisane oceny starego celu. Potwierdza
  zgodność reguły wyszukiwania, nie wymaga nowej oceny starych modeli.
- Widoki A mają identyczny hash z E025. A4096/B2048/C2048/dev5000 jak
  wcześniej, rozłączne; B/C/dev/rezerwa poza BN. BN CMA/reset,32batchy,
  parametry niezmienione, brak gradientów.
- Search widzi tylko nieetykietowane A/B. Profil zapisany i zahashowany
  przed odczytem C/dev oraz referencyjnych metryk do porównania. OLD_SEARCH
  nie był dodatkowym startem ani kandydatem dopisanym poza limitem.
- Hashe protokołu, kodu, checkpointów, splitów i referencji zgodne.
  Powtórne KL_B wybranego modelu przy finalnej ocenie zgodne z search.
  Test i rezerwa nietknięte.
- Analiza: `analysis_20261101T135139Z/{analysis.json,metrics.csv,pairs.csv,TABLES.md}`.
  Per seed: pełne `trace.json`, `selection.json`, `summary.json`, diagnostyki.

## Wynik na dev — średnia po4 seedach

| Metoda | Top1[%], mean±SD | NLL | KL na C (holdout wobec search) |
|---|---:|---:|---:|
| UNIFORM | 53,855±1,509 | 2,1501 | 1,2011 |
| TILT | **55,550±1,105** | 2,2198 | 1,2503 |
| OLD_SEARCH, profil E024 z poprawioną BN | 54,715±0,879 | 2,0843 | 1,0639 |
| NEW_SEARCH, wybór pod poprawioną BN | 55,155±0,689 | **2,0455** | **1,0517** |

Teacher po poprawionej BN z E025:75,140%, wyjściowy75,235%. Wyniki
bezwzględne skompresowanych sieci nadal wyraźnie gorsze od teacher.

### Różnice top1 per seed [pp]

| Seed | NEW−TILT | NEW−UNIFORM | NEW−OLD_SEARCH |
|---|---:|---:|---:|
| 42 | −1,32 | +0,86 | −0,44 |
| 43 | +1,70 | +2,40 | +0,44 |
| 44 | −1,10 | −0,24 | +0,88 |
| 45 | −0,86 | +2,18 | +0,88 |
| Średnia±SD | **−0,395±1,409** | **+1,300±1,231** | **+0,440±0,622** |
| Dodatnie | **1/4** | **3/4** | **3/4** |

## Zamrożone kryteria

| Kryterium | Wynik |
|---|---|
| NEW−TILT≥+1pp i≥3/4 dodatnich seedów | **NIE:**−0,395pp,1/4 |
| NEW−UNIFORM średnia>0 | TAK:+1,300pp |
| KL_C mniejsze od TILT na≥3/4 seedów | TAK:4/4 |
| Wszystkie warunki jednocześnie | **NIE — primary negatywny** |

Nie zwiększano limitu ani nie zmieniano celu po wyniku. Poprawa NLL/KL
nie jest zastępczym kryterium sukcesu top1.

## Co zmieniła korekta BN podczas wyszukiwania?

1. **Profile zmieniły się na4/4 seedów.** Nie jest to tylko ponowna ocena
   tej samej alokacji. Nadal start na wszystkich seedach wybrano z UNIFORM
   na podstawie KL_B; OLD_SEARCH nie dopisywano jako kandydat.
2. W porównaniu z OLD_SEARCH pod **tą samą** poprawioną BN nowy wybór
   daje średnio **+0,44pp**, dodatnio3/4. Zatem lepsza procedura oceny
   wpłynęła na wybór i średnią jakość, lecz efekt jest za mały wobec TILT.
3. KL_C względem OLD_SEARCH poprawia się średnio o0,01216, lecz tylko
   na2/4 seedów. Nie twierdzimy, że nowy wybór stabilnie dominuje stary
   na wszystkich kryteriach i powtórzeniach.
4. **Rozbieżność KL i top1 pozostała:** NEW ma mniejsze KL_C niż TILT
   na4/4 seedów, a lepszy top1 tylko na1/4. Już w punktach startowych
   UNIFORM ma niższe KL_B niż TILT na4/4 seedów, mimo gorszego top1.
   Sama naprawa BN nie usuwa więc problemu zgodności celu z trafnością.
5. To nie dowodzi, że jedynym ograniczeniem jest KL: skończony budżet
   wyszukiwania i sposób tworzenia kandydatów również mogą ograniczać wynik.
   Nie wykazaliśmy optymalności żadnej alokacji.

## Budżety i pochodzenie
Target UNIFORM:5 581 542 parametrów; TILT5 580 606.
NEW_SEARCH42–45: **5 581 454 / 5 581 454 / 5 581 178 / 5 581 369**.
Wszystkie pod wspólnym targetem, różnice znacznie poniżej0,1%.
Dokładne12 szerokości dla każdej metody w `analysis.json` i `selection.json`.
Referencje z E025:12 ocen alokacji oraz4 kontrole teacher+BN. Wyjściowe
teacher i pre-BN referencje z E024, bez ponownego obliczania.

## Decyzja
Pozostajemy przy **optymalizacji budżetu per grupa w foldingu**, ale
**nie skalujemy obecnego KL-search** na nowe zbiory i nie zwiększamy
automatycznie liczby ocen. E024 i E026 nie pokonały TILT w primary.

Następna uzasadniona hipoteza dotyczy **funkcji celu**, nie miary grupowania:
czy kryterium zachowania decyzji teacher na nieetykietowanym train lepiej
odpowiada poprawie top1 niż odtwarzanie całego rozkładu KL? To wymaga osobnego
protokołu z tym samym poprawionym BN, budżetem i kosztem search. Teacher też
popełnia błędy, więc zgodność decyzji nie gwarantuje trafności względem etykiet.
Żaden taki eksperyment ani trening nie został uruchomiony.

Ograniczenia: jeden zbiór/budżet,4checkpointy, jeden seed augmentacji,
dev i C wielokrotnie używane, wybór poprawki BN na podstawie E025.
To kontrolowany pilotaż, nie niezależna walidacja ani dowód przewagi nad
oficjalnym kodem1:1 lub innymi optymalizatorami alokacji.
