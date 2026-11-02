# E015 — status

- Protokół 0.1 zamrożony; [OPIS.md](OPIS.md) niezmieniony.
- Trening: **completed**, 4/4 seedy, run
  `runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12`; dev top-1
  74,94/75,10/75,04/75,86%, średnia **75,24%**. Test i rezerwa nietknięte.
- Folding: **completed**, udany run
  `runs/E015/fold_20261027T210324Z_cb9ffaf_36a01249`; 48/48 alokacji,
  GPU0–3. Pierwszy run `fold_20261027T204519Z_cb9ffaf_4ccfe035` failed
  (`NameError: os`); poprawka i audyt w
  [AUDYT_NAPRAWY_LAUNCHERA.md](AUDYT_NAPRAWY_LAUNCHERA.md).
- Budżety: wspólny target = faktyczne parametry UNIFORM; TILT maks.
  **253 parametry mniej**, MAG2 równy UNIFORM.
- Wynik: TILT+0.30 **+0,81 / +1,96 / +1,60 pp**; **główne kryterium
  replikacji NIE spełnione** (50/70% poniżej +2,0 pp). TILT+0.15 lepszy
  wtórnie, bez zastępowania primary.
- Następny krok: analiza mechanizmu i siły tiltu na istniejących wynikach;
  brak nowych uruchomień.
- Pełne tabele i CI: [WYNIKI.md](WYNIKI.md).
