# E019 — status

- Protokół 1.0: [OPIS.md](OPIS.md); wyniki: [WYNIKI.md](WYNIKI.md).
- **Completed:** trening 4/4, folding 48/48, cztery GPU.
- **Kryterium SPEŁNIONE:** TILT+0.30 vs UNIFORM **+3,71/+10,32/+7,60 pp**
  przy 30/50/70% parametrów; **12/12** par dodatnich.
- Baseline dev 50,82%; po foldingu UNIFORM 34,41/19,77/8,54%,
  TILT 38,12/30,08/16,14% (30/50/70%).
- MAG2 znacznie gorszy; TILT+0.15 lepszy tylko przy 30%.
- Test/rezerwa nietknięte; brak treningu wag po foldingu.
- Runy: trening `20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7`,
  folding `fold_20261030T220139Z_cb9ffaf_51bc405f`.
