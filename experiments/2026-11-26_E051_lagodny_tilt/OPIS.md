# 2026-11-26_E051_lagodny_tilt — łagodny TILT przy dopasowanym P

**Pytanie.** Czy liniowa alokacja z nachyleniem 0,30 odzyskuje top‑1
względem UNIFORM również przy ok. 10% redukcji parametrów? W E049
porównanie liniowe przy samym P zaczynało się od 30%; w b10 oceniano
**T-proj**, profil rzutowany na dwa pułapy. E051 bada brakujący wariant
przy stałym P. Dodatkowy, osobny punkt b5 pokazuje, jak zmienia się strata
do pełnego modelu przy łagodniejszej kompresji.

1. **Zamrożone geometrie:** b10 UNIFORM, T-proj i Frozen offsets z E049.
   Dla b5 UNIFORM używamy największego wspólnego ratio na siatce
   całkowitoliczbowej pod $\lfloor0{,}95 P_{\mathrm{full}}\rfloor$.
   TILT w obu budżetach: `mild_linear` z E049 (pochylenie `0.30`, redukcje
   ograniczone do `[0, 0.90]`, żeby móc zostawić całą grupę), dopasowany
   **tylko do P** UNIFORM. Jeśli zaokrąglenie nie daje 99,9–100% P UNIFORM,
   zwiększamy szerokości o jeden kanał, najwyżej do oryginalnej, zgodnie z
   największym wykonalnym przyrostem P, przy remisie najniższy indeks.
   MAC mierzymy jawnie dla każdej geometrii, bez pułapu na MAC TILT.
2. **Kontrole i ocena:** pełny model oraz U/T dla b5 i b10, T-proj i Frozen
   offsets dla b10. Te same osiem checkpointów E048, operator FOLD i adapter
   E049. Dla wszystkich wariantów w danym zbiorze A4096/CMA, FP32 bez TF32:
   CIFAR-100 widoki E049, Tiny crop min 0,65 z kontroli E050. Seedy 42–45,
   po jednym procesie na każdym z czterech GPU. Ocena tylko na dev; żadnego
   treningu, dobierania nachylenia ani otwierania benchmarków.
3. **Audyt:** rzeczywiste P/MAC przez forward CPU, parametry bez zmian w trakcie
   rekalibracji i punktacji, odtworzenie 32 kontroli (`full`, `b10_U`, `b10_Tproj`,
   `b10_L06` na obu zbiorach; zgodne widoki, hashe stanów, liczby trafień).
4. **Rozstrzygnięcie b10 na dev:** ≥9,5% redukcji P, TILT−UNIFORM ≥1 pp
   w średniej oraz ≥3/4 dodatnich par, strata TILT do pełnego po CMA ≤3 pp,
   oddzielnie na każdym zbiorze. Pokazujemy też TILT−Frozen offsets i
   TILT−T-proj, z ich rzeczywistym MAC. b5 jest opisowym, osobnym budżetem.
   E051 zaprojektowano po E049/E050; to pomiar diagnostyczny na wcześniej
   użytym dev i nie zmienia on oceny E049 pod oboma pułapami.
