# E035 — niezależna ocena ukończona

**Primary CMA15/15; wszystkie60 sparowanych różnic dodatnie.**
Pięć par, seedy42–45, redukcje30/50/70%, U/TILT+0.30, PRE/CMA/repair_bn.
Oficjalne testy CIFAR10/100 po10k i oficjalny val Tiny10k. Bez treningu.

TILT−UNIFORM po CMA przy30/50/70% [pp]:
- ResNet18/CIFAR10: **+0,8100/+3,4125/+10,9875**;
- ResNet18/CIFAR100: **+0,8275/+1,3025/+2,0600**;
- ResNet18/Tiny: **+3,5375/+9,3600/+7,4825**;
- PreActResNet18/CIFAR10: **+0,9600/+4,9500/+13,5775**;
- PreActResNet18/CIFAR100: **+4,4875/+7,6425/+22,7375**.

Osobny próg≥2pp przy50/70: **9/10**, nie R/CIFAR100/50.
Secondary repair_bn14/15; R/CIFAR100/70 tylko2/4 dodatnie.
R/CIFAR100/50 i70 ma gorsze NLL TILT po CMA. Duże straty bezwzględne
przy70% pozostają: T top1 R/C10032,68%, Tiny15,88%, P/C10045,7425%.
Historyczne progi i werdykty E034/E031 bez zmian.

Run `runs/E035/20261107T183831Z_8b0b14e0`:360+60 dev i360+60 test,
4GPU/exit0 na etap,1018,04s, audyty PASS. Dev top1 odtworzone dokładnie;
test dopiero po globalnej bramce. Protokół f42c0b8, implementacja5a7ec92.
CIFAR100 reserve nieużyta. Tiny wcześniej otwierany do kształtu E005/E007,
nie oceniany; E035 używa reserve z osobnego cache_val. CIFAR10 test_batch
wydobyto ze zweryfikowanego lokalnego archiwum, bez nadpisania train.

Wynik potwierdza kierunek zamrożonej reguły, nie uniwersalność ani optymalność.
Testy/val są odtąd użyte. [Pełny raport, SD, per-seed i audyt](WYNIKI.md).
