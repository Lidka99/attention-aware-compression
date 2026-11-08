"""E037: niezależna kontrola U' i agregacji; wyjście verification.json. OPIS.md."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PAIRS = ('resnet_cifar10', 'resnet_cifar100', 'resnet_tiny', 'preact_cifar10', 'preact_cifar100')
TAGS = ('b30', 'b50', 'b70')
SEEDS = tuple(range(42, 46))
MODES = ('PRE', 'CMA', 'UP_REPAIR')
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(run):
    run = Path(run)
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed' and [w['exit_code'] for w in m['workers']] == [0] * 4
    frozen = read(run / 'frozen.json')
    assert digest(run / 'frozen.json') == m['frozen_sha256']
    assert digest(Path('runs/E035/20261107T183831Z_8b0b14e0/frozen.json')) == frozen['e35_frozen_sha256']
    checks = []
    for pair in PAIRS:
        for tag in TAGS:
            plan = frozen['plans'][pair][tag]
            assert plan['Uprime_mac'] <= plan['T_mac']
            assert plan['Uprime_params'] >= plan['T_params']
            assert len(plan['Uprime']) == len(plan['T'])
            checks.append(f'{pair}/{tag}/mac')
        for seed in SEEDS:
            status = read(run / f'seed_{seed}/status.json')
            assert status['status'] == 'completed' and len(status['completed']) == 15
            assert status.get('data_open_attempts') == []
            ctrl = read(run / f'seed_{seed}/{pair}/controls.json')
            assert len(ctrl['metrics']) == 3
            for tag in TAGS:
                plan = frozen['plans'][pair][tag]
                s = read(run / f'seed_{seed}/{pair}/{tag}/summary.json')
                assert s['status'] == 'completed' and s['phase'] == 'dev'
                assert s['plan'] == plan
                assert [d['k'] for d in s['clusters']] == plan['Uprime']
                for mode in MODES:
                    val = s['metrics'][mode]
                    assert math.isfinite(val['top1']) and math.isfinite(val['nll']) and 0 <= val['top1'] <= 1
                    ev = s['evidence'][mode]
                    assert ev['weights_unchanged'] and ev['no_grad']
                checks.append(f'{seed}/{pair}/{tag}/summary')
    analysis = read(run / 'analysis/analysis.json')
    rows = analysis['rows']
    assert len(rows) == 180 and len(analysis['controls']) == 60 and len(analysis['averages']) == 135
    for r in rows:
        for k in ('U', 'T', 'Up'):
            assert 0 <= r[k] <= 1
    # Niezależne przeliczenie primary z rows.
    gate_ok = []
    for pair in PAIRS:
        for tag in TAGS:
            d = [100 * (r['T'] - r['Up']) for r in rows if r['pair'] == pair and r['tag'] == tag and r['mode'] == 'CMA']
            mean = float(np.mean(d)); pos = sum(x > 0 for x in d)
            gate = (mean >= 2 and pos >= 3) if tag in ('b50', 'b70') else (mean > 0 and pos >= 3)
            gate_ok.append(gate)
            found = [x for x in analysis['allocation'] if x['pair'] == pair and x['tag'] == tag and x['mode'] == 'CMA' and x['contrast'] == 'T-Up'][0]
            assert abs(found['mean'] - mean) < 1e-9 and found['positive'] == pos and found['gate'] == gate
    assert all(gate_ok) == analysis['primary_pass']
    out = dict(status='passed', utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
               checks=len(checks), rows=len(rows), controls=len(analysis['controls']),
               primary_pass=analysis['primary_pass'], gates=gate_ok,
               frozen_sha256=digest(run / 'frozen.json'), analysis_sha256=digest(run / 'analysis/analysis.json'),
               data_open_attempts=0, dev_only=True)
    (run / 'analysis/verification.json').write_text(json.dumps(out, indent=1) + '\n')
    print('PASS', out)


if __name__ == '__main__':
    main(sys.argv[1])
