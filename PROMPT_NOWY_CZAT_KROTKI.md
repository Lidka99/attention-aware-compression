Pracuj w /home/users/s224574/attention/badania_atencja_cnn. Przeczytaj
STATUS_BADAN.md, NOTATKA_KONTYNUACJA.md oraz WYNIKI E029 i E030.

Cel: otwarty problem autorów foldingu — alokacja budżetu per warstwa przy
dopasowanych parametrach, bez fine-tuningu. TILT+0.30 bije UNIFORM na
CIFAR-10/Tiny/CIFAR-100. Oficjalny operator autorów (commit 6946915)
sprawdzony na trzech zbiorach przy ~50%: TILT−UNIFORM +4,00/+1,70/+9,41 pp.
Search KL/disagreement (E024–E028) zamknięty jako niestabilny; miary
grupowania, AUTO50 i E023 zamknięte. Procedura BN: A4096 crop/flip + CMA.

Zadanie: E031 — rozszerzyć audyt oficjalnego operatora na budżety 30% i 70%
na istniejących checkpointach, bez treningu. Zamroź OPIS/kryteria przed
runem, najpierw krótki plan. 4 GPU, conda attention, ~4,2 GB dysku,
bez nowych danych i 96×96, test/rezerwa nietknięte. Nie zmieniaj starych
wyników. Po zakończeniu WYNIKI.md, PODSUMOWANIE.md, STATUS i NOTATKA.

Nowy trening (druga architektura) tylko po wyraźnej zgodzie użytkowniczki.
Push do GitHub wymaga danych logowania — commity są lokalne.
