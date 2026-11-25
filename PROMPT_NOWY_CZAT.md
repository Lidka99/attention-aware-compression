# Prompt do nowego czatu — E048, E049 i wkład artykułu

## Do skopiowania — bieżący stan

```text
Kontynuuję pracę nad artykułem o foldingu CNN w repozytorium
/home/users/s224574/attention/badania_atencja_cnn. Mam około tygodnia do
przedstawienia artykułu promotorowi. Potrzebuję konkretnego, uczciwego
wkładu na CIFAR-100 i Tiny ImageNet; CIFAR-10 nie wystarczy. Rozmawiaj ze
mną po polsku, rzeczowo i krytycznie. Manuskrypt jest po angielsku.

Najpierw przeczytaj STATUS_BADAN.md i AGENTS.md, aktualny rozdział
PLAN_PUBLIKACJI.md oraz WYNIKI.md E048/E049. Źródło manuskryptu: paper/main.tex
i paper/abstract.tex; istnieje paper/main.pdf. Nie odtwarzaj całej historii.

Sprawdź faktyczny stan istniejących uruchomień:
- E048: runs/E048/train_20261124T232822Z_89dfc621 — trening od zera na 4 GPU,
  4 seedy na zbiór, 240 epok na seed; najpierw CIFAR-100, następnie Tiny.
- E049: runs/E049/eval_20261124T234559Z_b4ca081b — oczekuje na ukończenie
  E048; uruchamia się automatycznie po audycie 8/8 checkpointów i zwolnieniu GPU.
Odczytaj manifest.json i w razie potrzeby sprawdź konkretny PID oraz logi.
Nie uruchamiaj drugiej kopii ani konkurencyjnych zadań GPU. Jeśli są wyniki,
odczytaj audit.json, a dla E049 także WYNIKI.md i per_seed.csv w katalogu runu.

E048: cele dev to ≥77,50% na CIFAR-100 i ≥59,00% na Tiny, z poprawą na ≥3/4
par seedów względem dawnych baz (75,235% i 50,82%). Tiny ma nowy stem i inną
recepturę; koszt pełnego modelu rośnie z 148,15 do 555,52 mln MAC/obraz.

E049: porównanie pełny / UNIFORM / L06 / T-proj przy b10 na obu zbiorach.
Ten sam operator foldingu i kalibracja BN, bez dostrajania wag. Główna bramka
na każdym zbiorze: ≥9,5% mniej P i MAC względem własnego pełnego modelu,
L06−U średnio ≥1 pp i ≥3/4 dodatnich par oraz strata L06 do pełnego ≤3 pp.
T-proj to dodatkowy nierównomierny punkt odniesienia pod tymi samymi pułapami.
Porównania U/TILT przy b30/b50/b70 mają dopasowane P, nie MAC. Benchmarki były
używane wcześniej; nie stroimy profili ani progów na ich wynikach.

Zacznij od krótkiego meldunku o E048/E049. Jeśli trening jeszcze trwa,
kontynuujmy dyskusję: zaproponuj jedno zdanie o wkładzie i układ głównej
tabeli dla promotora. Rozróżnij lepszą bazę od wkładu empirycznego: czy raz
wybrany profil zachowuje więcej top‑1 przy obu limitach, także względem
T-proj? L06 nie jest uniwersalnym optymalizatorem. Jeśli wyniki są gotowe,
oceń uczciwie, który wariant tezy potwierdzają, i co konkretnie trzeba poprawić
w abstrakcie, tabelach i wykresach. Do artykułu wybieramy porównania potrzebne
do rozstrzygnięcia tezy i pokazujemy jej istotne ograniczenia.
```

## Historyczny prompt po E047 — angielski manuskrypt LaTeX po szkicu 0.9

```text
Pracuj w /home/users/s224574/attention/badania_atencja_cnn. Celem jest
pełny angielski artykuł LaTeX do recenzji wewnętrznej, z gotowymi figurami
i tabelami. Artykuł, podpisy, osie, legendy i tabele pisz po angielsku;
ze mną rozmawiaj po polsku. Wykonaj poniższe kroki.

1. Przeczytaj STATUS_BADAN.md, AGENTS.md, SZKIC_ARTYKULU.md (0.9),
   sekcję „Po E047” w PLAN_PUBLIKACJI.md i szkic w paper/abstract.tex.
   Szczegóły sprawdzaj w raportach WYNIKI i audytowanych artefaktach runs/.
   Ten etap opiera się na istniejących wynikach; nowy trening lub ocena
   nowych danych wymaga osobnej zgody.

2. Zredaguj abstrakt i wstęp wokół wkładu. Wang et al. (2025, s. 13)
   zostawiają dobór kompresji per warstwa jako dalszą pracę; w głównych
   porównaniach CNN w Cut Less, Fold More (rys. 1 i 8) ratio jest jednolite.
   My badamy, jak alokacja szerokości sprzężonych grup kanałów zmienia
   trafność przy tym samym operatorze foldingu i rekalibracji A4096/CMA.
   To badanie empiryczne bez dostrajania wag, z kalibracją na danych.
   TILT to adaptacja znanej reguły liniowej; L06 to odrębny profil z puli
   E042, wybrany na CIFAR-100 dev i zamrożony przed dalszymi ocenami.
   Nie deklaruj pierwszeństwa, ogólnego optymalizatora ani data-free.

3. Napisz kompletny paper/main.tex z bibliografią paper/references.bib.
   Dołącz paper/abstract.tex, opisz metody, wyniki, dyskusję, ograniczenia
   i wnioski. Poprowadź wyniki w tej kolejności:
   - E035: przy dopasowanej liczbie parametrów (P) TILT poprawia top-1
     względem UNIFORM w 15/15 konfiguracji po CMA (4/4 seedy w każdej). To faza
     niezależna: oficjalne testy CIFAR i Tiny val.
   - E036/E037/E041: TILT ma 1.105–1.823× MAC UNIFORM przy dopasowanych P.
     Na dev szerszy UNIFORM U′ o zbliżonym MAC wygrywa w 10/15 konfiguracji
     i ma więcej parametrów. E041 dopasowuje oba koszty do TILT i daje
     wynik zależny od zbioru. Rozróżnij liczbę MAC i zmierzoną latencję.
   - E046/E047: przy b10 L06 mieści się pod oboma pułapami UNIFORM,
     redukując względem pełnego modelu P o 10.41%, a MAC o 10.70%.
     Zysk wynosi +1.1425 pp na Tiny dev i +1.3500 pp na ImageNetV2
     MatchedFrequency (4/4 w obu). Na ImageNetV2 sparowana strata do
     pełnego modelu spada z 3.35 do 2.00 pp. Sparowany 95% przedział
     bootstrap [+0.700, +2.025] pp dla zysku dotyczy obrazów przy ustalonych
     modelach. To 2,000 nowych obrazów tych samych 200 klas, ocenionych
     na tych samych czterech checkpointach co w E046. E043/E044 są
     uzupełnieniem: osobno traktuj
     rezerwę CIFAR-100, wcześniej użyty Tiny val, Tiny dev i ImageNetV2.

4. Wygeneruj trzy figury: (a) sparowane różnice TILT–UNIFORM przy
   dopasowanych P z E035; (b) top-1 i rzeczywiste koszty U/TILT/U′ na
   wspólnym dev E037; (c) b10 z osobnymi panelami Tiny dev i ImageNetV2,
   z punktami po seedach i przedziałem E047. Wykorzystaj istniejące
   generatory z figures/publikacja/ E035 i E037 oraz z
   experiments/2026-11-22_E047_l06_b10_imagenetv2/wykresy/20261122T053804Z_ae2dec73/.
   Angielskie figury zapisz w katalogach eksperymentów jako wektorowy
   PDF i PNG. Dołącz je do artykułu i podaj instrukcje odtworzenia.

5. Przygotuj i dołącz tabele LaTeX w paper/tables/ oraz zapisz dane
   tabel jako CSV: TILT–UNIFORM (E035), kontrole kosztu i b10. Dla b10
   podaj Full/UNIFORM/L06, rzeczywiste P i MAC per image, top-1 oraz
   sparowaną stratę do pełnego modelu, osobno dla obu prób. Dla E046/E047
   bierz liczby z analysis/summary.json i analysis/per_seed.csv.

6. Sprawdź zgodność liczb, podpisów, cytowań i odsyłaczy. Skompiluj PDF
   i sprawdź czytelność figur i tabel. Jeśli brakuje silnika LaTeX, podaj
   stan kompilacji i polecenie do odtworzenia. Na końcu daj ścieżki do
   gotowych źródeł LaTeX, PDF, figur i tabel.
```

## Historyczny prompt po E045

```text
Kontynuuj badania nad foldingiem CNN bez fine-tuningu w repozytorium
/home/users/s224574/attention/badania_atencja_cnn. Najpierw przeczytaj
AGENTS.md, STATUS_BADAN.md, NOTATKA_KONTYNUACJA.md, sekcję „Po E045”
w PLAN_PUBLIKACJI.md i raport E045. Aktualny SZKIC_ARTYKULU.md ma wersję 0.6.

Priorytetem jest zachowanie trafności; 50% redukcji parametrów nie jest
wymogiem. Na Tiny dev po CMA pełny model ma 51,2200%. W E045 UNIFORM przy
10,35% redukcji P i 10,67% redukcji MAC osiąga 46,3625% (strata 4,8575 pp,
per seed 4,38/4,85/4,60/5,60 pp). Przy 20,33% P i 19,97% MAC to 40,5800%
(strata 10,6400 pp). U/b30 i U/b50 na wykresie pochodzą z tego samego dev E035.
E045: runs/E045/20261122T031350Z_a58b56a3, 12/12 ocen na 4 GPU, audyt PASS.

L06 wybrano na dev w E042. W E043 poprawia top‑1 względem U o +1,215 pp
przy ~50% redukcji P na rezerwie CIFAR-100 i +0,945 pp przy ~30%.
W E044 na Tiny uzyskał +2,100 pp (4/4) i +0,255 pp (3/4), na wcześniej
użytym val. To dowody dla zamrożonego profilu pod oboma pułapami, nie wynik
automatycznego selektora. E045 nie badał L06 przy b10/b20.

Najbliższe pytanie: czy w b10 można zmniejszyć stratę jednym zamrożonym
profilem, np. L06, bez przekraczania rzeczywistych P i MAC U z E045?
Na początek zaproponuj jeden mały protokół i wykonaj kontrolę CPU geometrii.
Dotychczasowy limit k≤floor(0,9n) wymusza na Tiny ≥19,21% redukcji P;
jeśli zmieniasz górny limit, zapisz to jawnie jako zmianę reguły projekcji.
Nie mierz trafności przed ustaleniem porównania i kryteriów. Jeśli geometria
nie przejdzie, opisz ograniczenie bez poszerzania puli.

Używaj istniejących checkpointów E019, seedów 42–45, tego samego A4096/CMA
i odniesienia do pełnego modelu; pomiary GPU w Conda attention na czterech
kartach, jeden seed na GPU. Rozdzielaj Tiny dev, wcześniej użyty val E035/E044
i wykorzystaną rezerwę CIFAR-100 z E043. Nowy trening oraz nowa ocena na
nieużytych danych testowych wymagają odrębnej zgody. Nie wznawiaj search
E024–E028 ani nie rozbudowuj E042. Zapisz protokół w nowym folderze eksperymentu.
```

## Archiwum promptu sprzed E034

> **Notatka historyczna — doprecyzowanie 2026-11-11.** Poniżej zachowano prompt
> sprzed zakończenia E034. Bieżący stan: STATUS_BADAN.md i NOTATKA_KONTYNUACJA.md.
> „Adresujemy otwarty problem” oznacza podjęcie badania, nie wykazane rozwiązanie
> ogólnej optymalizacji alokacji. Po E035–E040 zakres wkładu opisuje
> [OCENA_WKLADU_I_ILUSTRACJI.md](OCENA_WKLADU_I_ILUSTRACJI.md).

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
