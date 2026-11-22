"""E040: geometria i zamrożony alokator par; pełny protokół: OPIS.md.
Wejście: szerokości i liczby niezgodnych decyzji na B; wyjście: plany i wybory.
Bez modeli, etykiet, GPU i adaptacyjnych ocen kandydatów.
"""
import hashlib
import itertools
import time
from fractions import Fraction

import numpy as np

NAMES = (
    'relu_conv', 'layer1.0.relu1', 'layer1.1.relu1',
    'layer2.0.relu1', 'layer2.1.relu1', 'layer2.0.relu2',
    'layer3.0.relu1', 'layer3.1.relu1', 'layer3.0.relu2',
    'layer4.0.relu1', 'layer4.1.relu1', 'layer4.0.relu2',
)
SIZES = (64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512)
LOW = tuple(3 * n // 5 for n in SIZES)
HIGH = tuple(4 * n // 5 for n in SIZES)
PERM = (0, 1, 2, 3, 5, 4, 6, 8, 7, 9, 11, 10)
UP_NAMES = tuple(NAMES[i] for i in PERM)
DATASETS = ('cifar10', 'cifar100')
SEEDS = (42, 43, 44, 45)
CAPS = {'cifar10': (5584580, 276383716), 'cifar100': (5581542, 275335732)}
UNIFORM = {
    'cifar10': (45, 45, 45, 90, 90, 90, 181, 181, 181, 362, 362, 362),
    'cifar100': (45, 45, 45, 90, 90, 90, 180, 180, 180, 361, 361, 361),
}
PAIRS = tuple(itertools.combinations(range(12), 2))
ALL_BITS = tuple(itertools.product((0, 1), repeat=12))
PROBES = ((0,) * 12,) + tuple(
    tuple(int(j == g) for j in range(12)) for g in range(12)
) + tuple(tuple(int(j in pair) for j in range(12)) for pair in PAIRS)


def bits(z):
    assert len(z) == 12 and set(z) <= {0, 1}
    return ''.join(map(str, z))


def cost(widths, dataset):
    """Dokładne P i Conv/Linear MAC dla semantycznej kolejności NAMES."""
    w = tuple(map(int, widths))
    assert len(w) == 12 and all(1 <= k <= n for k, n in zip(w, SIZES))
    classes = 10 if dataset == 'cifar10' else 100
    assert dataset in DATASETS
    p, mac, previous = 29 * w[0] + classes, 1024 * 27 * w[0], w[0]
    for stage, (out_index, hidden) in enumerate(
        ((0, (1, 2)), (5, (3, 4)), (8, (6, 7)), (11, (9, 10)))
    ):
        out, area = w[out_index], (32 // (2 ** stage)) ** 2
        for block, hi in enumerate(hidden):
            incoming, h = (previous if block == 0 else out), w[hi]
            p += 9 * incoming * h + 2 * h + 9 * h * out + 2 * out
            mac += area * (9 * incoming * h + 9 * h * out)
            if stage > 0 and block == 0:
                p += incoming * out + 2 * out
                mac += area * incoming * out
        previous = out
    return int(p + classes * previous), int(mac + classes * previous)


def feasible(widths, dataset):
    return all(a <= b for a, b in zip(cost(widths, dataset), CAPS[dataset]))


def distance(widths, target):
    # Wszystkie n dzielą512: całkowitoliczbowa wersja znormalizowanego L2.
    return sum(((int(k) - int(t)) * (512 // n)) ** 2
               for k, t, n in zip(widths, target, SIZES))


def projection(menu, target, coordinate='v'):
    return min(menu, key=lambda r: (distance(r['widths'], target),
                                   -r['params'], r['mac'], tuple(r[coordinate])))


def tilt_profile(dataset, beta):
    """Enumeracja dokładnych punktów zmiany w a, bez zaokrągleń float."""
    beta = Fraction(beta)
    depth = tuple(Fraction(2 * g - 11, 11) for g in range(12))
    boundaries = {Fraction(-1), Fraction(2)}
    for n, d in zip(SIZES, depth):
        boundaries.update(1 - Fraction(k, n) - beta * d for k in range(n + 1))
    ordered = sorted(boundaries)
    points = ordered + [(a + b) / 2 for a, b in zip(ordered, ordered[1:])]
    seen, best = set(), None
    for a in points:
        widths = []
        for n, d in zip(SIZES, depth):
            retain = 1 - min(Fraction(9, 10), max(Fraction(1, 10), a + beta * d))
            widths.append((n * retain).numerator // (n * retain).denominator)
        w = tuple(widths)
        if w in seen:
            continue
        seen.add(w)
        p, m = cost(w, dataset)
        if p <= CAPS[dataset][0] and m <= CAPS[dataset][1]:
            key = (sum(w), p, -m, tuple(-k for k in w))
            if best is None or key > best[0]:
                best = (key, w)
    assert best is not None
    return best[1]


def make_plan(dataset, historical_tilt):
    menu = []
    for v in ALL_BITS:
        w = tuple(LOW[g] if v[g] else HIGH[g] for g in range(12))
        p, m = cost(w, dataset)
        if p <= CAPS[dataset][0] and m <= CAPS[dataset][1]:
            menu.append(dict(v=list(v), widths=list(w), params=p, mac=m))
    anchor = projection(menu, UNIFORM[dataset])
    for row in menu:
        row['z'] = [a ^ b for a, b in zip(row['v'], anchor['v'])]
    eligible = [r for r in menu if sum(r['z']) >= 3
                and 10 * r['params'] >= 9 * CAPS[dataset][0]
                and 10 * r['mac'] >= 9 * CAPS[dataset][1]]

    def mask_key(row):
        z = bits(row['z'])
        return hashlib.sha256(f'20261109|{dataset}|{z}'.encode()).digest(), z

    diagnostics = sorted(eligible, key=mask_key)[:16]
    assert len(diagnostics) == 16
    refs = []
    for step in range(7):
        w = tilt_profile(dataset, Fraction(step, 20))
        p, m = cost(w, dataset)
        refs.append(dict(name='U' if step == 0 else f'TILT_{step * 5:02d}',
                         beta=float(Fraction(step, 20)), widths=list(w), params=p, mac=m))
    assert tuple(refs[0]['widths']) == UNIFORM[dataset]
    proj_t = projection(menu, historical_tilt, coordinate='z')
    refs.append(dict(name='TILT_PROJ', widths=proj_t['widths'], params=proj_t['params'], mac=proj_t['mac']))
    result = dict(dataset=dataset, caps=dict(params=CAPS[dataset][0], mac=CAPS[dataset][1]),
                  names=list(NAMES), sizes=list(SIZES), low=list(LOW), high=list(HIGH),
                  anchor=anchor, references=refs, diagnostics=diagnostics,
                  diagnostic_pool=len(eligible), feasible_count=len(menu),
                  historical_tilt=list(historical_tilt))
    expected = (1857, 564, 5246333, 275824206) if dataset == 'cifar10' else (1759, 565, 5426105, 273062716)
    assert (len(menu), len(eligible), anchor['params'], anchor['mac']) == expected
    return result


def widths_at(plan, z):
    return tuple(k + (l + h - 2 * k) * int(b)
                 for k, l, h, b in zip(plan['anchor']['widths'], LOW, HIGH, z))


def coefficients(counts):
    """Liczby niezgodności, nie float: dokładne remisy i kontrasty na B2048."""
    assert len(counts) == 79
    d0 = int(counts[0])
    a = [int(counts[1 + g]) - d0 for g in range(12)]
    interaction = [int(counts[13 + i]) - int(counts[1 + g]) - int(counts[1 + h]) + d0
                   for i, (g, h) in enumerate(PAIRS)]
    return dict(constant=d0, unary=a, pair=interaction, denominator=2048)


def predict_counts(coef, z, order=2):
    q = int(coef['constant']) + sum(a * b for a, b in zip(coef['unary'], z))
    if order == 2:
        q += sum(i * z[g] * z[h] for i, (g, h) in zip(coef['pair'], PAIRS))
    else:
        assert order == 1
    return q


def allocate(plan, coef, order=2):
    start = time.monotonic()
    rows = []
    for z in ALL_BITS:
        w = widths_at(plan, z)
        p, m = cost(w, plan['dataset'])
        if p <= plan['caps']['params'] and m <= plan['caps']['mac']:
            q = predict_counts(coef, z, order)
            rows.append((q, -p, m, z, w))
    chosen = min(rows)
    assert time.monotonic() - start < 60
    return dict(widths=list(chosen[4]), z=list(chosen[3]), q_count=chosen[0],
                params=-chosen[1], mac=chosen[2], order=order,
                feasible_count=len(rows), predicted_count_range=[min(r[0] for r in rows), max(r[0] for r in rows)],
                out_of_range=sum(r[0] < 0 or r[0] > 2048 for r in rows),
                seconds=time.monotonic() - start)


def prediction_gate(rows):
    """Wspólna bramka na obu zbiorach; wiersze zawierają sumy16błędów licznikowych."""
    cells = []
    for dataset in DATASETS:
        vals = sorted([r for r in rows if r['dataset'] == dataset], key=lambda r: r['seed'])
        assert [r['seed'] for r in vals] == list(SEEDS)
        s1, s2 = sum(r['error1_sum'] for r in vals), sum(r['error2_sum'] for r in vals)
        positive = sum(r['error2_sum'] < r['error1_sum'] for r in vals)
        passed = s1 > 0 and 5 * s2 <= 4 * s1 and positive >= 3
        cells.append(dict(dataset=dataset, passed=passed, positive=positive,
                          mae1=float(s1 / (4 * 16 * 2048)), mae2=float(s2 / (4 * 16 * 2048)),
                          ratio=float(s2 / s1) if s1 else None, rows=vals))
    return dict(passed=all(r['passed'] for r in cells), cells=cells)
