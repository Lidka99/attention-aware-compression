# E017 — kontrole przed uruchomieniem

## CPU (`check_cpu_e017.py`, bez GPU i bez dev)

- **Równoważność z E012:** przy 50% kanałów `reference` odtwarza
  **bit w bit** model E012 (`state_dict`) i etykiety wszystkich 12 grup.
- **Determinizm:** drugi przebieg daje identyczny model i etykiety.
- **Równy budżet:** wszystkie 7 miar dają **2 797 610** parametrów
  (bo liczba klastrów `k` jest identyczna; liczba parametrów zależy od `k`,
  nie od rozmiarów klastrów).
- **Kontrola losowa:** permutacja zachowuje multizbiór rozmiarów klastrów
  (test syntetyczny); sprawdzane też sumy i liczba klastrów w realnych grupach.
- **bn_effective:** używa średniej geometrycznej skal wszystkich BN w grupie;
  co najmniej jedna grupa faktycznie użyła skali.
- Wynik: `E017_CPU_CHECKS=passed`, `TESTY_WYNIK.json`.

## Poprawki przed pierwszym uruchomieniem GPU

- Protokół 1.0 → 1.1: budżety oznaczają proporcję **kanałów** (10/30/50%),
  nie parametrów. Nie zmieniono miar ani operatora; nie było wyników.
- W trakcie kontroli poprawiono założenie o równych rozmiarach klastrów
  między miarami (nie jest wymagane; wymagana jest równa `k`).

## Czego te testy nie sprawdzają

- Jakości na dev (to robi bieg GPU).
- Wpływu miary na zbieżność BN — każda metoda ma identyczny BN.
