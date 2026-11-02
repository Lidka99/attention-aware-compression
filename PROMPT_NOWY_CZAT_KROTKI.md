Pracuj w /home/users/s224574/attention/badania_atencja_cnn. Przeczytaj
STATUS_BADAN.md, PLAN_2026-10-30_dalsze_kroki.md i
literature/AUDYT_ALOKACJA_FOLDING.md.

Kontynuujemy od E022: analiza mechanizmu alokacji TILT na istniejących
checkpointach E011/E015/E019, bez nowego treningu. Odpowiedz: jak szkoda
zależy od głębokości (pre/post BN), ile odzyskuje BN, czemu CIFAR-100 jest
słabszy, czy TILT chroni wrażliwe grupy, czy błąd rekonstrukcji wag
przewiduje rzeczywistą szkodę.

Zasady: osobny folder E022 z zamrożonym OPIS; najpierw krótki plan; 4 GPU;
4,4 GB dysku; conda attention; nie zmieniaj starych wyników; bez nowego
treningu bez zgody; po zakończeniu WYNIKI.md i aktualizacja STATUS_BADAN.md.
