# E028 — transfer reguły E027 na CIFAR-10

**Completed, transfer nieudany.** 176 ocen,4GPU,4 seedy,19,1min, exit0.
Ta sama zamrożona reguła (wybór po disagreement), ten sam operator i BN,
tylko CIFAR-10 i checkpointy E011.

NEW_SEARCH **93,650%**, TILT93,430%, UNIFORM89,435%.
NEW−TILT **+0,220pp (2/4)**, próg+1pp niespełniony. NEW−UNIFORM+4,215pp.
Pułap puli−TILT tylko+0,545pp: na CIFAR-10 TILT jest blisko granicy
osiągalnej alokacji w tej przestrzeni. Kontrola BN: teacher95,46→95,44%.

Wniosek: linia „nowy optymalizator search” nie daje stabilnej przewagi
nad TILT na dwóch zbiorach. Stabilny wkład to alokacja per warstwa,
prosta reguła TILT i kontrole negatywne. Następne kroki: oficjalny
baseline1:1 i druga architektura, nie kolejne warianty search.
[WYNIKI.md](WYNIKI.md).
