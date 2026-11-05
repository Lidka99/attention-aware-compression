Pracuj w /home/users/s224574/attention/badania_atencja_cnn. Przeczytaj
STATUS_BADAN.md, NOTATKA_KONTYNUACJA.md, README.md, OPIS E034 oraz
WYNIKI E031, E032 i E033.

Cel: alokacja budżetu per warstwa w foldingu bez fine-tuningu.
E031: oficjalny operator 30/70%, zgodność 24/24, kierunek 11/12
(CIFAR-100/70%/repair_bn tylko 2/4). E032: późna wrażliwość CIFAR-100
replikuje się 4/4, ale nie wyjaśnia różnic zysku TILT. E033: primary
negatywny, prosta hipoteza „layer4” zamknięta. Search, AUTO50 i miary
grupowania zamknięte.

Zadanie: dokończ E034 Phase 1 (PreActResNet18, CIFAR-10/100, 4 seedy).
Run `runs/E034/train_20261105T014209Z_21b44f52`; sprawdź manifest.
Jeśli trening trwa, monitoruj bez równoległych zadań GPU. Jeśli completed,
uruchom `fold_preact.py fold --train-run ...`, potem `analyze`, oceń
TILT−UNIFORM przy 30/50/70% (50/70 ≥+2 pp i ≥3/4; CIFAR-100 osobno),
zapisz WYNIKI.md/PODSUMOWANIE.md i zaktualizuj STATUS, NOTATKĘ i README.
Zrób lokalne commity. Nowy trening tylko po wyraźnej zgodzie; push do
GitHub wymaga danych logowania użytkownika.
