# E034 — PreActResNet18: kryteria spełnione na obu CIFAR

**Phase 1 completed, 2026-11-06.** Trening8/8 checkpointów po200 epok,
4GPU/exit0; baseline95,215% CIFAR-10 i76,905% CIFAR-100.
Folding144 ocen+24kontrole,4GPU/exit0,179,07s; niezależny audyt PASS.

TILT−UNIFORM przy30/50/70% **redukcji parametrów**:

| Zbiór | CMA [pp] | repair_bn [pp] |
|---|---|---|
| CIFAR-10 | +0,985 / +4,880 / +13,190 | +1,020 / +4,920 / +13,230 |
| CIFAR-100 | +4,435 / +7,225 / +23,095 | +4,600 / +7,175 / +23,310 |

**Każda komórka4/4 dodatnie; primary6/6, secondary6/6.** Reguła przeniosła
się na drugą architekturę, również na CIFAR-100. Nie dowodzi uniwersalności
ani przyczynowej roli preaktywacji. CIFAR-100/70 TILT46,52% wobec
baseline76,905% — duża przewaga względna, nadal duża utrata jakości.
Test/rezerwa nietknięte, bez fine-tuningu; używany wcześniej dev.

Run fold `runs/E034/fold_20261106T003625Z_9ec4f2ea`, analiza
`analysis_20261106T004001Z/`. [Pełny raport, poprawki i odtworzenie](WYNIKI.md).
Następny krok: zamrożony protokół niezależnej oceny i zakres twierdzeń
artykułu. Nowy trening wymaga zgody; zamkniętych kierunków nie otwieramy.
