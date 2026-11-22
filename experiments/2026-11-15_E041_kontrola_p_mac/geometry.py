"""2026-11-15_E041_kontrola_p_mac: dopasowanie geometrii P/MAC na CPU.
Wejście: zamrożone plany E035/E037/E039, bez wag i danych; wyjście: profile,
koszty i bramka wykonalności w unikalnym runs/E041/cpu_*. Pełny opis: OPIS.md.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NAMES = (
    'relu_conv', 'layer1.0.relu1', 'layer1.1.relu1',
    'layer2.0.relu1', 'layer2.0.relu2', 'layer2.1.relu1',
    'layer3.0.relu1', 'layer3.0.relu2', 'layer3.1.relu1',
    'layer4.0.relu1', 'layer4.0.relu2', 'layer4.1.relu1',
)
GEOMETRY_NAMES = (
    'stem', 'layer1.0_hidden', 'layer1.1_hidden',
    'layer2.0_hidden', 'stage2_out', 'layer2.1_hidden',
    'layer3.0_hidden', 'stage3_out', 'layer3.1_hidden',
    'layer4.0_hidden', 'stage4_out', 'layer4.1_hidden',
)
SIZES = np.array((64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512))
LOW, HIGH = SIZES // 10, 9 * SIZES // 10
PERMUTATIONS = (101, 102, 103, 104)
PAIRS = ('resnet_cifar10', 'resnet_cifar100')
SOURCES = {
    'E035': ROOT / 'runs/E035/20261107T183831Z_8b0b14e0/frozen.json',
    'E037': ROOT / 'experiments/2026-11-08_E037_kontrola_mac/frozen.json',
    'E039': ROOT / 'experiments/2026-11-08_E039_przestawienia_tilt/frozen.json',
}


def utc():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def costs(widths, classes):
    """P i Conv/Linear MAC; ostatni wymiar to jawna kolejność upstream NAMES."""
    w = np.asarray(widths)
    assert w.shape[-1] == len(NAMES)
    p = 29 * w[..., 0] + classes
    mac = 1024 * 27 * w[..., 0]
    previous = w[..., 0]
    for stage, (output, hidden) in enumerate(((0, (1, 2)), (4, (3, 5)),
                                            (7, (6, 8)), (10, (9, 11)))):
        out = w[..., output]
        area = (32 // (2 ** stage)) ** 2
        for block, index in enumerate(hidden):
            incoming = previous if block == 0 else out
            h = w[..., index]
            conv = 9 * incoming * h + 9 * h * out
            p = p + conv + 2 * h + 2 * out
            mac = mac + area * conv
            if stage and block == 0:
                p = p + incoming * out + 2 * out
                mac = mac + area * incoming * out
        previous = out
    return np.stack((p + classes * previous, mac + classes * previous), axis=-1)


def inside(cost, target):
    cost = np.asarray(cost, dtype=np.int64)
    return bool(np.all(cost <= target) and np.all(1000 * cost >= 999 * target))


def rms(a, b):
    return float(np.sqrt(np.mean(((np.asarray(a) - b) / SIZES) ** 2)))


def repair(continuous, target, classes):
    """Naprawa całkowitoliczbowa, bez ocen jakości i bez zmiany >2grup."""
    rounded = np.clip(np.rint(continuous), LOW, HIGH).astype(np.int64)
    if inside(costs(rounded, classes), target):
        return rounded, []
    minimum = (999 * target + 999) // 1000
    best = None
    for i, j in itertools.combinations(range(12), 2):
        us = np.arange(LOW[i], HIGH[i] + 1, dtype=np.int64)
        probe = np.tile(rounded, (len(us), 1))
        probe[:, i], probe[:, j] = us, 0
        c0 = costs(probe, classes)
        probe[:, j] = 1
        slope = costs(probe, classes) - c0
        assert np.all(slope > 0)
        lower = np.maximum(LOW[j], np.max(-((c0 - minimum) // slope), axis=1))
        upper = np.minimum(HIGH[j], np.min((target - c0) // slope, axis=1))
        valid = lower <= upper
        if not np.any(valid):
            continue
        rows = probe[valid].copy()
        rows[:, j] = np.clip(int(np.rint(continuous[j])), lower[valid], upper[valid])
        distances = np.sum(((rows - continuous) / SIZES) ** 2, axis=1)
        for row, distance in zip(rows, distances):
            actual = costs(row, classes)
            assert inside(actual, target)
            key = (float(distance), tuple(map(int, row)))
            if best is None or key < best[0]:
                best = (key, row.copy(), [i, j])
    return (best[1], best[2]) if best else (None, [])


def project(proposal, tilt, target, classes):
    desired = np.asarray(proposal, dtype=float) / SIZES
    center = .9995 * target

    def objective(x):
        return float(np.sum((x - desired) ** 2))

    def residual(x):
        return costs(x * SIZES, classes) / center - 1

    start = time.monotonic()
    result = minimize(objective, tilt / SIZES, jac=lambda x: 2 * (x - desired),
                      method='SLSQP', bounds=list(zip(LOW / SIZES, HIGH / SIZES)),
                      constraints={'type': 'eq', 'fun': residual},
                      options={'maxiter': 500, 'ftol': 1e-12})
    evidence = dict(success=bool(result.success), message=str(result.message),
                    iterations=int(result.nit), residual=residual(result.x).tolist(),
                    continuous_widths=(result.x * SIZES).tolist())
    if not result.success or np.max(np.abs(residual(result.x))) > 1e-7:
        return None, dict(evidence, seconds=time.monotonic() - start)
    widths, changed = repair(result.x * SIZES, target, classes)
    evidence.update(repaired_coordinates=changed, seconds=time.monotonic() - start)
    return widths, evidence


def run(parent):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Wymagane jawne ukrycie GPU'
    started = time.monotonic()
    out = parent / f'cpu_{utc()}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=False, exist_ok=False)
    manifest = dict(status='running', utc=utc(), pid=os.getpid(), phase='geometry_cpu',
                    python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    gpu_used=False, weights_opened=False, images_opened=False,
                    quality_used_for_selection=False,
                    sources={str(p.relative_to(ROOT)): digest(p) for p in SOURCES.values()},
                    code_sha256=digest(Path(__file__)), protocol_sha256=digest(HERE / 'OPIS.md'))
    write(out / 'manifest.json', manifest)
    shutil.copy2(__file__, out / 'geometry.py')
    shutil.copy2(HERE / 'OPIS.md', out / 'OPIS.md')
    # Zamrożone pliki zawierają też dawne referencje jakości; niżej odczytujemy
    # wyłącznie pola plans/shuffled/costs. Nie wykorzystujemy references/metrics.
    inputs = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    results, csv_rows = {}, []
    try:
        for pair in PAIRS:
            classes = 10 if pair == 'resnet_cifar10' else 100
            base = inputs['E035']['plans'][pair]['b50']
            ref = inputs['E037']['plans'][pair]['b50']
            metadata = inputs['E039']['costs'][pair]['b50']
            assert tuple(metadata['group_names']) == GEOMETRY_NAMES
            assert metadata['group_sizes'] == SIZES.tolist()
            tilt = np.asarray(base['T'], dtype=np.int64)
            target = np.array((ref['T_params'], ref['T_mac']), dtype=np.int64)
            assert base['T'] == ref['T']
            assert np.array_equal(costs(tilt, classes), target)
            assert np.array_equal(costs(base['U'], classes), (base['params']['U'], ref['U_mac']))
            rows = []
            for permutation in PERMUTATIONS:
                assert time.monotonic() - started < 120, 'Limit CPU120s'
                old = inputs['E039']['shuffled'][pair]['b50'][str(permutation)]
                assert np.array_equal(costs(old['widths'], classes), (old['params'], old['mac']))
                widths, solver = project(old['widths'], tilt, target, classes)
                row = dict(permutation=permutation, proposal=old['widths'], solver=solver,
                           matched=False, diverse=False)
                if widths is not None:
                    actual = costs(widths, classes)
                    changed = int(np.sum(np.abs((widths - tilt) / SIZES) >= .05 - 1e-12))
                    distance = rms(widths, tilt)
                    row.update(widths=widths.tolist(), params=int(actual[0]), mac=int(actual[1]),
                               ratios=(actual / target).tolist(), distance_from_tilt=distance,
                               changed_groups_5pp=changed, matched=inside(actual, target),
                               diverse=distance >= .05 and changed >= 4)
                    csv_rows.append(dict(pair=pair, permutation=permutation, params=int(actual[0]),
                                         mac=int(actual[1]), params_ratio=float(actual[0] / target[0]),
                                         mac_ratio=float(actual[1] / target[1]), rms_tilt=distance,
                                         changed_groups_5pp=changed,
                                         **dict(zip(NAMES, map(int, widths)))))
                rows.append(row)
                print(pair, permutation, json.dumps(row, ensure_ascii=False), flush=True)
            distances = [dict(a=a['permutation'], b=b['permutation'], rms=rms(a['widths'], b['widths']))
                         for a, b in itertools.combinations(rows, 2) if 'widths' in a and 'widths' in b]
            passed = all(r['matched'] and r['diverse'] for r in rows) and \
                len(distances) == 6 and all(d['rms'] >= .03 for d in distances)
            results[pair] = dict(tilt=tilt.tolist(), target=target.tolist(), controls=rows,
                                 pairwise_distances=distances, passed=passed)
        passed = all(r['passed'] for r in results.values())
        write(out / 'geometry.json', dict(names=NAMES, geometry_names=GEOMETRY_NAMES,
                                         sizes=SIZES.tolist(), results=results,
                                         passed=passed, matching_tolerance=.001))
        if csv_rows:
            with (out / 'profiles.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        manifest.update(status='completed', gate_A_passed=passed)
        print('GATE_A', 'PASS' if passed else 'FAIL', 'RUN', out.relative_to(ROOT), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest.update(finished_utc=utc(), seconds=time.monotonic() - started)
        write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'runs/E041')
    run(parser.parse_args().parent)
