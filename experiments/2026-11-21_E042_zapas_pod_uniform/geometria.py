"""E042: projekcje na P/MAC UNIFORM wyłącznie z metadanych; por. OPIS.md.
Z wczytanych planów E035/E037 zapisuje pulę, próby geometrii i manifest CPU.
"""

import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs/E042'
E041 = HERE.parent / '2026-11-15_E041_kontrola_p_mac'
spec = importlib.util.spec_from_file_location('_e042_e041_geometry', E041 / 'geometry.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
R35 = g.SOURCES['E035'].parent
PAIRS = g.PAIRS
CAPS = {PAIRS[0]: (5584580, 276383716), PAIRS[1]: (5581542, 275335732)}
HIDDEN = ((1, 2), (3, 5), (6, 8), (9, 11))
DISTANCE = .025


def read(path):
    return json.loads(path.read_text())


def classes(pair):
    return {PAIRS[0]: 10, PAIRS[1]: 100}[pair]


def moves():
    for stage, (a, b) in enumerate(HIDDEN, 1):
        for suffix, sign in (('p', 1), ('m', -1)):
            offsets = np.zeros(12, dtype=int)
            offsets[a], offsets[b] = 12 * sign, -12 * sign
            yield f'W{stage}{suffix}', 'wewnatrz', offsets
    for a, b in ((1, 4), (2, 3)):
        for suffix, sign in (('p', 1), ('m', -1)):
            offsets = np.zeros(12, dtype=int)
            offsets[3*(a-1):3*a], offsets[3*(b-1):3*b] = 16 * sign, -16 * sign
            yield f'M{a}{b}{suffix}', 'miedzy', offsets
    rng = np.random.Generator(np.random.PCG64(20261121))
    for i, offsets in enumerate(rng.integers(-18, 19, size=(48, 12)), 1):
        yield f'L{i:02}', 'losowe', offsets


def profile(name, family, widths, pair):
    widths = np.asarray(widths, dtype=np.int64)
    target = np.array(CAPS[pair], dtype=np.int64)
    cost = g.costs(widths, classes(pair))
    assert np.all(g.LOW <= widths) and np.all(widths <= g.HIGH)
    assert g.inside(cost, target), (name, cost, target)
    return dict(profile=name, family=family, widths=widths.tolist(),
                params=int(cost[0]), mac=int(cost[1]))


def validate_bank(bank):
    assert bank['passed'] and tuple(bank['names']) == g.NAMES and bank['sizes'] == g.SIZES.tolist()
    assert set(bank['pairs']) == set(PAIRS)
    for pair, menu in bank['pairs'].items():
        assert tuple(menu['caps']) == CAPS[pair]
        rows = menu['references'] + menu['candidates']
        assert [r['profile'] for r in menu['references']] == ['U', 'T-proj']
        assert len({r['profile'] for r in rows}) == len(rows) <= 26
        assert (rows[0]['params'], rows[0]['mac']) == CAPS[pair]
        families = [r['family'] for r in menu['candidates']]
        assert 6 <= families.count('wewnatrz') <= 8 and 2 <= families.count('miedzy') <= 4
        assert families.count('losowe') == 12 and set(families) == {'wewnatrz', 'miedzy', 'losowe'}
        for i, row in enumerate(rows):
            assert row == profile(row['profile'], row['family'], row['widths'], pair)
            assert all(g.rms(row['widths'], prev['widths']) >= DISTANCE for prev in rows[:i])


def build(pair, base, previous):
    target = np.array(CAPS[pair], dtype=np.int64)
    uniform = np.array(base['U'], dtype=np.int64)
    assert base['T'] == previous['T']
    assert tuple(target) == (base['params']['U'], previous['U_mac'])
    assert tuple(g.costs(base['T'], classes(pair))) == (base['params']['T'], previous['T_mac'])
    assert previous['T_mac'] > target[1]
    t, solver = g.project(base['T'], uniform, target, classes(pair))
    assert t is not None and g.rms(t, uniform) >= DISTANCE, solver
    refs = [profile('U', 'referencja', uniform, pair), profile('T-proj', 'referencja', t, pair)]
    candidates, trials = [], []
    for name, family, offsets in moves():
        proposal = np.clip(np.rint(uniform + offsets * g.SIZES / 100), g.LOW, g.HIGH).astype(int)
        widths, solver_data = g.project(proposal, uniform, target, classes(pair))
        trial = dict(profile=name, family=family, offsets_pp=offsets.tolist(),
                     proposal=proposal.tolist(), solver=solver_data, accepted=False)
        if widths is None:
            trial['reason'] = 'projekcja'
        else:
            row = profile(name, family, widths, pair)
            min_distance = min(g.rms(widths, r['widths']) for r in refs + candidates)
            trial.update(result=row, min_rms=min_distance)
            if min_distance >= DISTANCE:
                trial['accepted'] = True
                candidates.append(row)
            else:
                trial['reason'] = 'odleglosc'
        trials.append(trial)
        if sum(r['family'] == 'losowe' for r in candidates) == 12:
            break
    return dict(caps=target.tolist(), references=refs, candidates=candidates,
                projection=solver, proposals=trials, historical_t=base['T'],
                historical_t_mac=previous['T_mac'])


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    start = time.monotonic()
    out = RUNS / f'cpu_{g.utc()}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True)
    sources = [g.SOURCES['E035'], g.SOURCES['E037'], E041 / 'geometry.py',
               HERE / 'geometria.py', HERE / 'OPIS.md']
    hashes = {str(p.relative_to(ROOT)): g.digest(p) for p in sources}
    manifest = dict(status='running', pid=os.getpid(), phase='geometry_cpu',
                    started_utc=g.utc(), hashes=hashes, python=sys.version,
                    numpy=np.__version__, scipy=g.scipy.__version__, quality_used=False)
    g.write(out / 'manifest.json', manifest)
    shutil.copy2(HERE / 'geometria.py', out / 'geometria.py')
    shutil.copy2(E041 / 'geometry.py', out / 'geometry_E041.py')
    shutil.copy2(HERE / 'OPIS.md', out / 'OPIS.md')
    try:
        e35, e37 = [read(p)['plans'] for p in sources[:2]]
        bank = dict(passed=True, pairs={}, names=list(g.NAMES), sizes=g.SIZES.tolist())
        for pair in PAIRS:
            menu = build(pair, e35[pair]['b50'], e37[pair]['b50'])
            bank['pairs'][pair] = menu
        g.write(out / 'bank.json', bank)
        validate_bank(bank)
        manifest.update(status='completed', bank_sha256=g.digest(out / 'bank.json'))
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest.update(finished_utc=g.utc(), seconds=time.monotonic()-start)
        g.write(out / 'manifest.json', manifest)
    for pair, menu in bank['pairs'].items():
        counts = {f: sum(r['family'] == f for r in menu['candidates'])
                  for f in ('wewnatrz', 'miedzy', 'losowe')}
        print(pair, 'CAPS', menu['caps'], 'CANDIDATES', counts, 'T-PROJ', menu['references'][1], flush=True)
    print('GEOMETRY PASS', out.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    run()
