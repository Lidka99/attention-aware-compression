"""E036: niezależna kontrola runu, surowych czasów i agregacji; wyjście verification.json. OPIS.md."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PAIRS = ('resnet_cifar10', 'resnet_cifar100', 'resnet_tiny', 'preact_cifar10', 'preact_cifar100')
TAGS = ('b30', 'b50', 'b70')
SEEDS = tuple(range(42, 46))
NAMES = ('teacher',) + tuple(t + '_' + p for t in TAGS for p in ('U', 'T'))
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(run):
    run = Path(run)
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed'
    assert [w['exit_code'] for w in m['workers']] == [0] * 4
    assert digest(run / 'source/benchmark.py') == m['code_hashes']['experiments/2026-11-07_E036_audyt_kosztu/benchmark.py']
    frozen = read(run / 'frozen.json')
    checks = []
    timing_files = 0
    for seed in SEEDS:
        status = read(run / f'seed_{seed}/status.json')
        assert status['status'] == 'completed' and len(status['completed']) == 210
        assert not status.get('data_open_attempts'), status.get('data_open_attempts')
        for pair in PAIRS:
            costs = read(run / f'seed_{seed}/{pair}/costs.json')
            ref = frozen['references'][str(seed)][pair]
            for name in NAMES:
                expected = ref['teacher'] if name == 'teacher' else ref[name[:3]][name[-1]]
                assert costs[name]['state_sha256'] == expected['state_sha256']
                assert costs[name]['params'] > 0 and costs[name]['mac'] > 0
                assert costs[name]['independent_mac_check'] and len(costs[name]['layers']) > 0
                checks.append(f'{seed}/{pair}/{name}/state')
            for tag in TAGS:
                u = costs[tag + '_U']['params']; t = costs[tag + '_T']['params']
                assert t <= u and t >= 0.999 * u, (pair, tag, u, t)
            for batch in (1, 128):
                for name in NAMES:
                    times = []
                    for series in range(3):
                        p = run / f'seed_{seed}/{pair}/{name}_bs{batch}_s{series}.json'
                        d = read(p)
                        timing_files += 1
                        assert d['batch'] == batch and d['name'] == name and d['series'] == series
                        assert d['seed'] == seed and d['pair'] == pair and d['state_unchanged']
                        assert len(d['times_ms']) == 200
                        assert all(x > 0 and math.isfinite(x) for x in d['times_ms'])
                        assert d['after_allocated'] - d['empty_allocated'] <= 32 * 1024**2
                        assert d['peak_allocated'] >= d['baseline_allocated']
                        times.extend(d['times_ms'])
                    checks.append(f'{seed}/{pair}/{name}/bs{batch}/times')
                    if seed == 42 and name == 'teacher' and batch == 128:
                        rows = read(run / 'analysis/analysis.json')['rows']
                        row = [r for r in rows if r['seed'] == 42 and r['pair'] == pair and r['name'] == name and r['batch'] == 128][0]
                        assert abs(row['median'] - float(np.median(times))) < 1e-9
                        assert row['events'] == 600
    assert timing_files == 4 * 5 * 7 * 2 * 3
    analysis = read(run / 'analysis/analysis.json')
    assert len(analysis['rows']) == 280 and len(analysis['ratios']) == 120
    assert len(analysis['memory']) == 280
    for q in analysis['quality']:
        assert len(q['per_seed']) == 4
        for row in q['per_seed']:
            ref = frozen['references'][str(row['seed'])][q['pair']][q['tag']]
            assert row['U'] == ref['U']['quality_CMA']['top1']
            assert row['T'] == ref['T']['quality_CMA']['top1']
    out = dict(status='passed', utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
               timing_files=timing_files, checks=len(checks), rows=len(analysis['rows']),
               state_hash_matches=sum(1 for c in checks if c.endswith('/state')),
               frozen_sha256=digest(run / 'frozen.json'),
               analysis_sha256=digest(run / 'analysis/analysis.json'),
               source_benchmark_sha256=digest(run / 'source/benchmark.py'),
               data_open_attempts=0, no_new_quality=True)
    (run / 'analysis/verification.json').write_text(json.dumps(out, indent=1) + '\n')
    print('PASS', out)


if __name__ == '__main__':
    main(sys.argv[1])
