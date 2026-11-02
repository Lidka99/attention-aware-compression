Pracuj w /home/users/s224574/attention/badania_atencja_cnn.

Najpierw przeczytaj: AGENTS.md, STATUS_BADAN.md, PLAN_2026-10-30_dalsze_kroki.md,
literature/AUDYT_ALOKACJA_FOLDING.md oraz CONTRIBUTION_STATEMENT_2026-10-30.md.
Nie powtarzaj wcześniejszych eksperymentów ani nie proponuj kolejnych miar
grupowania — ten wątek jest zamknięty (E017, E018, E021 negatywne).

Stan: TILT+0.30 (nasza metoda) bije UNIFORM na CIFAR-10 (+0,87/+3,77/+11,27 pp),
Tiny ImageNet (+3,71/+10,32/+7,60 pp, 12/12) i słabo na CIFAR-100
(+0,81/+1,96/+1,60 pp). Audyt potwierdził lukę: Wang et al. (2025) wprost
zostawiają optymalizację sparsity per warstwa jako future work, a
Cut Less, Fold More używa jednolitych ratio.

Zadanie: zaimplementuj i uruchom E022 — analizę mechanizmu alokacji TILT
na istniejących checkpointach E011/E015/E019. Bez nowego treningu.
Odpowiedz konkretnie:
1. Jak szkoda zależy od głębokości grupy (pre i post BN)?
2. Ile od szkody odzyskuje korekta BN i które grupy pozostają krytyczne?
3. Dlaczego CIFAR-100 wypada słabiej niż CIFAR-10 i Tiny ImageNet?
4. Czy TILT rzeczywiście chroni grupy o wysokiej zmierzonej szkodzie?
5. Czy błąd rekonstrukcji wag przewiduje rzeczywistą szkodę przy stałym
   poziomie kompresji?

Zasady:
- Osobny folder experiments/<data>_E022_mechanizm_alokacji/, zamrożony OPIS
  i kryteria PRZED uruchomieniem; najpierw przedstaw krótki plan.
- Tylko istniejące modele; oficjalny test i rezerwa nietknięte.
- 4 GPU; dysk ma 4,4 GB wolne — bez nowych datasetów i bez 96×96.
- Conda attention; nie zmieniaj wcześniejszych wyników ani nie usuwaj runów.
- Podagenty tylko do małych, wydzielonych zadań; bez równoległych zmian tych
  samych plików i bez konkurencyjnych zadań GPU.
- Po zakończeniu: WYNIKI.md, aktualizacja STATUS_BADAN.md i NOTATKA_KONTYNUACJA.md.

Nie zaczynaj nowego treningu bez mojej zgody. Najpierw mechanizm, potem decyzja.
