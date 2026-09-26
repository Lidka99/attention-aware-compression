# Rejestr decyzji

## D001 — 2026-09-25: oddzielne repo i dokumentacja przed obliczeniami

Utworzono lokalne repo w istniejącym folderze `badania_atencja_cnn`. Zachowano wcześniejszy plan. Prace dzielimy na E000–E003. Wszystkie pozostają `draft`; danych i architektury nie wybrano. Nowość jest warunkiem przejścia do badania własnego kryterium.

Każdy późniejszy wpis: data UTC, decyzja, uzasadnienie, dostępne wtedy wyniki, alternatywy, zmienione wersje protokołów. Decyzji po obejrzeniu wyników nie przedstawiać jako wcześniejszych założeń.

## Niezależny projekt — ustalenie 2026-09-26

Eksperymenty piszemy od nowa w tym repozytorium. Nie kopiujemy ani nie importujemy kodu, konfiguracji, wag czy wyników z attention_v2 ani attentionv3. Nie tworzymy zależności od tych folderów. Wcześniejsza propozycja użycia lokalnego checkpointu SE jest wycofana.

Dobór modelu i danych przeprowadzamy od początku. Możemy korzystać z udokumentowanych bibliotek i znanych architektur. Aktualna decyzja: trenujemy od losowych wag na 4 GPU w środowisku Conda attention, bez publicznych checkpointów.
