# E011 — stan uruchomienia

Aktualizacja: **completed**, koniec 2026-10-24 20:37 UTC, exit0.
[Wyniki końcowe](WYNIKI.md). Poniżej historyczny zapis startu.

2026-10-24 19:23 UTC: **running**, cztery procesy treningowe na GPU0–3.
Bieg główny: `runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012`.
Commit kodu: `018f3f2`. Launcher PID 266021; workery 266191–266194.
Log launchera: `runs/E011/launcher_detached_20261024.log`.

Potwierdzono 93–94% wykorzystania wszystkich czterech RTX 2080 Ti,
~2040 MiB na GPU. Seed42: epoka 1, train 45000/dev 5000, 22,46 s,
dev top-1 41,0%. To wyłącznie kontrola startu, nie wynik końcowy.
Plan: 200 epok, niezależne seedy 42/43/44/45, ostatnia epoka.
Przy utrzymaniu tempa szacunek ~75–90 min od startu, nie gwarancja czasu.

Stan bieżący: `manifest.json`; postęp: `seed_*/metrics.jsonl`;
końcowe metryki: `seed_*/summary.json`. Launcher zakończy się statusem
completed dopiero po sprawdzeniu wszystkich czterech wyników.
Manifest git_dirty może być true z powodu wcześniejszych nieśledzonych
notatek E010/literature; kod E011 był commitowany i snapshotowany.

Poprzednia próba głównego startu
`20261024T192136Z_018f3f2_seeds42-45_15180f99` została przerwana przez
limit czasu powłoki przed uruchomieniem workerów. Nie jest aktywna.
Aktualny launcher uruchomiono przez Popen(start_new_session=True), z logiem
plikowym i odłączonym stdin. Próby przerwane pozostają do audytu.
