# E041 — kontrola P i MAC: wynik zależny od zbioru

**Cel:** sprawdzić TILT względem czterech innych profili przy jednocześnie
dopasowanych parametrach i MAC, na ResNet18/CIFAR-10/100/b50.

**Stan:** completed,4GPU/exit0,227,054s. Każda kontrola ma99,9–100%P/MAC
TILT i zmienia6–7grup. Primary poCMA: **C10+0,11250pp,2/4;
C100+2,10375pp,4/4**. Łączna bramka niespełniona przez CIFAR-10.
Po repair analogicznie+0,12750/+1,96375pp. Kontrola104 naC10 bije TILT
o0,3850pp,4/4poCMA. Tiny nie był objęty tym pilotem.

**Wniosek:** częściowe potwierdzenie znaczenia profilu przy P/MAC, bez
ogólnej przewagi TILT. Próba zamknięta; wynik do wąskiej pracy empirycznej.
Bez treningu, testu/rezerwy i nowych plików wag; audyt PASS.

Szczegóły: [wyniki](WYNIKI.md), [wyniki CPU](WYNIKI_CPU.md), [protokół](OPIS.md),
[uruchomienie](URUCHOMIENIE.md).
