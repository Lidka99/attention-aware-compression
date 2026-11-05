# E032 — replikacja wrażliwości grup

**Completed, H1/H2 spełnione.** CIFAR-100/CIFAR-10,4seedy,12grup,
75% kompresji kanałów jednej grupy, PRE/CMA/repair_bn.
288 ocen+24kontrole,4GPU/exit0,149,79s; bez treningu/testu/rezerwy.

- **CIFAR-100 późna wrażliwość po CMA replikuje się4/4** (nowe seedy3/3):
  rho głębokość–szkoda0,622–0,629; późne−wczesne+16,48 do+17,54pp.
- Na kontrolnym CIFAR-10 rho−0,224 do+0,077, późne−wczesne ujemne4/4.
  Kontrast rho między zbiorami średnio+0,677: H2 spełnione.
- Po CMA późne grupy tracą średnio20,64pp na CIFAR-100 i0,96pp na CIFAR-10.
  `layer4.0.relu2`:30,36pp vs1,46pp.
- **Nie wyjaśnia to seed45:** profile CIFAR-100 niemal identyczne rangowo
  (rho między seedami0,958–0,986), związek L−E z zyskiem TILT E031:
  **rho0,00 CMA /+0,20 repair_bn**, tylko4punkty, opisowo.

Potwierdzono powtarzalną różnicę profili, nie przyczynowy mechanizm całej
alokacji. ANTI-TILT/AUTO50 i słabe wyniki CIFAR-100 pozostają obowiązujące.
Run `runs/E032/20261104T235711Z_fc9d5eaa`, analiza
`analysis_20261105T000640Z/`; [pełny raport](WYNIKI.md).
