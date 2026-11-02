# E026 — poprawione BN podczas wyboru alokacji

**Completed, primary negatywny.** CIFAR-100/~50%,4seedy,200 ocen search,
4GPU/exit0,9,65min. Jedyna zmiana względem E024: crop/flip na A do BN;
cel KL i limit50 ocen bez zmian. Referencje z E025 ponownie użyte.

NEW_SEARCH55,155%, TILT55,550%, UNIFORM53,855%, OLD_SEARCH54,715%.
NEW−TILT **−0,395pp (1/4)**; NEW−UNIFORM+1,300pp (3/4);
NEW−OLD_SEARCH+0,440pp (3/4). Profile zmienione4/4. KL_C i NLL lepsze
od TILT, lecz top1 nie. Naprawa BN pomaga, nie rozwiązuje celu alokacji.

Następna hipoteza: zgodność decyzji teacher zamiast KL, w osobnym protokole;
nie zwiększać automatycznie limitu ani nie wracać do miar grupowania.
Run `runs/E026/20261101T133903Z_bbd75e05`; [WYNIKI.md](WYNIKI.md).
