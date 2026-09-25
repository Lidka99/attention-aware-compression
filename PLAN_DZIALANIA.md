# Plan badań: atencja kanałowa a scalanie kanałów CNN

Data: 25 września 2026 r.
Status: propozycja pilota i audytu nowości. Nie potwierdzono luki badawczej ani skuteczności proponowanej metody. Nie przeprowadzono nowych eksperymentów w ramach tej notatki.

## 1. Cel

Sprawdzić, czy informacja o tym, jak atencja waży kanały dla różnych wejść, pomaga wybrać grupy do scalenia i uzyskać fizycznie mniejszy CNN przy mniejszej utracie jakości.

To osobny kierunek do oceny obok foldingu z kwantyzacją opisanego w [poprzedniej notatce](docs/context/FOLDING_KWANTYZACJA.md). Pierwszy pilot dotyczy atencji i scalania w pełnej precyzji. Kwantyzację można dołączyć po wykazaniu użyteczności samego kryterium.

## 2. Pytanie i hipoteza

**Pytanie:** czy kanały podobne pod względem wag lub aktywacji przed atencją można bezpiecznie scalić, jeśli atencja wykorzystuje je inaczej dla różnych wejść?

**Hipoteza:** uwzględnienie zachowania kanałów po ważeniu przez atencję pozwala lepiej przewidywać błąd scalenia i wybierać grupy niż podobieństwo wag lub aktywacji przed atencją.

Dla kanału c oznaczmy aktywację przed atencją przez h_c(x), wagę atencji przez a_c(x), a aktywację po atencji przez z_c(x) = a_c(x) h_c(x). Podobieństwo h_c i h_d nie musi oznaczać podobieństwa z_c i z_d. Samo podobieństwo a_c i a_d również nie wystarcza.

Ważna kontrola: jeśli z_c zawiera już całą użyteczną informację, oddzielne dodawanie wag a_c do kryterium może być zbędne. Nowa metoda musi pokazać przewagę także nad prostym scalaniem według aktywacji po atencji.

## 3. Kontekst dotychczasowych badań

Wcześniejsze raporty projektu nie potwierdziły przewagi kilku badanych kontrolerów i wariantów semantic feedback. Wyniki te odnoszą się do konkretnych implementacji i protokołów, nie do wszystkich metod atencji.

Zachować dwie lekcje:

- Maska nałożona po obliczeniu konwolucji nie dowodzi oszczędności obliczeń. W tym pilocie wynikiem ma być rzeczywiście zmniejszony graf.
- Złożoność kryterium musi dawać korzyść ponad proste kontrole przy takim samym rozmiarze modelu, dostępie do danych i budżecie naprawy.

## 4. Etap A — literatura i audyt nowości

Przeczytać najbliższe metody i przygotować tabelę: mechanizm, dane, funkcja celu, sposób wyboru kanałów, naprawa, architektury, rzeczywisty koszt, kod i ograniczenia.

Kolejność lektury:

1. ECA-Net — działanie taniej atencji kanałowej.
2. Cut Less, Fold More i GRAIL — scalanie oraz kompensacja po zmianie struktury.
3. ADSC — atencja jako kryterium strukturalnego pruning’u.
4. Pick-or-Mix — dynamiczny wybór i łączenie kanałów.
5. MRLA oraz praca o dekompozycji dynamicznej konwolucji — sąsiednie mechanizmy, potrzebne do wyznaczenia zakresu nowości.
6. DANCE — dodatkowa lektura, jeśli później wrócimy do pomijania obliczeń w zależności od wejścia.

Poszukać również kompresji sieci SE/ECA/CBAM, neuron merging, activation-aware folding i rekonstrukcji bloków. Uwzględnić starsze prace. Samo określenie „attention-aware” nie stanowi nowości.

ADSC wymaga krytycznej lektury: w opisanym celu dodatni składnik wariancji, jeśli jest minimalizowany z dodatnim współczynnikiem, zmniejsza wariancję ocen, choć tekst przypisuje mu zwiększanie różnorodności. Zweryfikować pełny cel i kod przed reprodukcją.

**Rezultat:** jedna konkretna różnica względem najbliższej metody oraz lista wymaganych porównań. Jeśli proponowane kryterium już istnieje, odpowiednio zmienić hipotezę przed implementacją.

## 5. Etap B — jeden model i poprawne scalanie

Wybrać dostępny, wiarygodny checkpoint CNN z atencją kanałową, np. wariant ResNet z SE lub ECA. Sprawdzić dostępność wag i zgodność danych przed wyborem. Nie rozpoczynać od trenowania wielu nowych architektur.

Wybrać blok, w którym można zmniejszyć wewnętrzną szerokość bez zmiany zewnętrznego wymiaru połączenia residual. Opisać wszystkie operacje zależne od liczby kanałów: konwolucje, BatchNorm, moduł atencji i warstwę odbierającą jego wyjście.

Scalanie zmienia wejście modułu atencji, a więc może zmienić także jego wagi a_c(x). Kryterium trzeba oceniać na rzeczywistym zmodyfikowanym bloku, nie wyłącznie na zamrożonych aktywacjach nauczyciela. Dla ECA sprawdzić dodatkowo wpływ zmiany kolejności i sąsiedztwa kanałów; dla SE — zależności wymiarów obu projekcji.

Sprawdzić poprawność przebiegu, wymiary, liczbę parametrów i jakość checkpointu. Dla identycznych lub proporcjonalnych kanałów w prostym kontrolowanym przykładzie sprawdzić algebraiczną poprawność implementacji tam, gdzie zakładamy dokładne scalenie.

**Rezultat:** działający operator fizycznego scalania jednego bloku oraz jawny opis przybliżeń.

## 6. Etap C — wykrycie zjawiska przed projektowaniem algorytmu

Na kilku blokach przygotować ograniczoną pulę par lub grup do scalenia. Każdy kandydat ma prowadzić do takiej samej szerokości wynikowej.

Porównać kryteria:

| Kryterium | Informacja wykorzystywana do wyboru |
|---|---|
| Podobieństwo wag | Parametry kanałów i jawnie ustalona reprezentacja |
| Podobieństwo aktywacji przed atencją | h_c(x) |
| Podobieństwo aktywacji po atencji | z_c(x) |
| Zachowanie wag atencji | a_c(x), jako kontrola diagnostyczna |
| Proponowane kryterium | Dodatkowa informacja o zależności odpowiedzi i ważenia od wejścia |
| Losowy wybór | Ten sam rozmiar i budżet ocen |

Każde scalenie ocenić przez błąd wyjścia całego bloku i zmianę jakości modelu. Użyć jednakowej kompensacji dla wszystkich kryteriów; oddzielnie zachować wyniki bez naprawy. Jeśli GRAIL wymaga adaptacji do konkretnego bloku z atencją, opisać tę adaptację.

Sprawdzić nie tylko korelację ocen z błędem, lecz także jakość faktycznie wybranego kandydata. Punktem odniesienia może być najlepszy bezpośrednio oceniony kandydat z tej samej puli, nie nieznane globalne optimum.

**Rezultat:** odpowiedź, czy informacja z atencji zmienia decyzję o scaleniu w sposób istotny dla jakości.

## 7. Etap D — prosta metoda, jeśli zjawisko się potwierdzi

Opracować najprostsze kryterium, które wyjaśnia obserwowaną korzyść. Na początku wykorzystać atencję istniejącego modelu. Osobny uczony kontroler wymaga dodatkowego uzasadnienia.

Sprawdzić, czy kryterium działa na innych grupach i blokach niż użyte do jego opracowania. Zmierzyć koszt zbierania aktywacji, wyznaczania grup i kompensacji. Porównać z bezpośrednią oceną kandydatów przy takim samym budżecie.

**Rezultat:** metoda z jasno określoną różnicą algorytmiczną i kosztem.

## 8. Etap E — cały model i kontrole

Dopiero po pozytywnym wyniku lokalnym przejść do kompresji całej sieci.

- Porównać identyczne szerokości wynikowe, ten sam dostęp do danych i równy budżet naprawy.
- Uwzględnić standardowy folding, folding z kompensacją, scalanie według aktywacji po atencji oraz mocny structured pruning.
- Raportować jakość bez dostrajania i po jednakowym krótkim dostrajaniu.
- Mierzyć rzeczywisty rozmiar artefaktu, czas wykonania, pamięć oraz koszt samego kompresowania.
- Potwierdzić efekt dla kilku losowań danych kalibracyjnych. Dla szerszych wniosków wykorzystać niezależne checkpointy; losowania kalibracji nie zastępują seedów treningu.
- Drugą architekturę i większy benchmark dodać po pozytywnym pilocie.

## 9. Dane i kryteria decyzji

Rozdzielić dane do wyznaczania kryterium i kompensacji, walidację do wyboru wariantu oraz test końcowy. Oprócz podziału obrazów wydzielić grupy scalenia lub bloki niewykorzystane podczas projektowania kryterium. Uwzględnić wcześniejsze audyty duplikatów i podziałów w projekcie.

Kontynuować, jeśli korzyść:

1. pozostaje względem prostego podobieństwa aktywacji po atencji;
2. występuje przy równym rozmiarze, liczbie ocen i budżecie naprawy;
3. powtarza się poza warunkami użytymi do projektowania kryterium;
4. ma znaczenie dla jakości końcowej i uzasadnia koszt kompresowania.

Przed porównaniem finalnych wariantów ustalić minimalny interesujący efekt na podstawie celu zastosowania oraz zmienności pomiarów. Nie dobierać tego progu po zobaczeniu wyników.

Zamknąć lub zawęzić hipotezę, jeśli informacja atencji jest zbędna względem z_c(x), przewaga znika po wyrównaniu naprawy, albo nowa metoda jest równoważna istniejącemu rozwiązaniu. Taki pilot nie gwarantuje samodzielnej publikacji.

## 10. Najbliższe działania

Najpierw wykonać etapy A–C. Orientacyjnie przeznaczyć kilka dni na lekturę i audyt oraz około dwóch tygodni na lokalny pilot, zależnie od dostępności checkpointu i trudności zmian grafu. Wynikiem ma być krótka notatka z tabelą porównań i decyzją o dalszej pracy.

Aktualizacja: folder jest osobnym repozytorium Git. Opisy etapów znajdują się w `experiments/`, konfiguracje w `configs/`, literatura w `literature/`, raporty w `reports/`, a kontrakt reprodukowalności w `docs/REPRODUCIBILITY.md`. Eksperymentów jeszcze nie uruchomiono.

## 11. Literatura i biblioteka alphaXiv

W alphaXiv utworzono folder **„Atencja w CNN — kompresja”**, zawierający pięć poniższych prac o atencji i dynamicznych CNN. ADSC nie udało się zapisać przez MCP pod adresem wydawcy.

1. **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**, CVPR 2020. [alphaXiv](https://www.alphaxiv.org/abs/1910.03151).
2. **Pick-or-Mix: Dynamic Channel Sampling for ConvNets**, 2024. [alphaXiv](https://www.alphaxiv.org/abs/2406.10935).
3. **Cross-Layer Retrospective Retrieving via Layer Attention**, ICLR 2023. [alphaXiv](https://www.alphaxiv.org/abs/2302.03985).
4. **Revisiting Dynamic Convolution via Matrix Decomposition**, 2021. [alphaXiv](https://www.alphaxiv.org/abs/2103.08756).
5. **DANCE: Dynamic 3D CNN Pruning: Joint Frame, Channel, and Feature Adaptation for Energy Efficiency on the Edge**, preprint, 2026. [alphaXiv](https://www.alphaxiv.org/abs/2603.17275).
6. **Learnable attention driven structured compression technique for neural networks**, Scientific Reports, 2026. [Wydawca](https://www.nature.com/articles/s41598-026-49893-2).
7. **Cut Less, Fold More: Model Compression through the Lens of Projection Geometry**, ICLR 2026. [alphaXiv](https://www.alphaxiv.org/abs/2602.18116).
8. **GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks**, CPAL 2026. [Proceedings](https://proceedings.mlr.press/v328/tang26a.html), [alphaXiv](https://www.alphaxiv.org/abs/2602.23795).

Przegląd pozostaje wstępny. Uwagi o nowości dotyczą kandydackiej hipotezy, a nie potwierdzonego braku wcześniejszych rozwiązań.
