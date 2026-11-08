"""2026-11-08_E039_przestawienia_tilt: audyt ocen i agregacji; OPIS.md.
Wejście: completed run; wyjście: verification.json, bez GPU i danych testowych.
"""
import sys
from pathlib import Path

from experiment import PAIRS, PERM_SEEDS, rt


def audit(run):
    manifest, frozen = rt.verify_run(run)
    count = replay_count = controls_count = 0
    for pair in PAIRS:
        for seed in rt.SEEDS:
            ctrl = rt.read(run / f'seed_{seed}/{pair}/controls.json')
            ref = frozen['references'][pair]['b50'][str(seed)]
            rt.check_reference(ctrl['metrics'], ctrl['evidence'], ref['controls']['metrics'], ref['controls']['evidence'])
            controls_count += 3
            replay = rt.read(run / f'seed_{seed}/{pair}/replay_T.json')
            rt.check_reference(replay['metrics'], replay['evidence'], ref['metrics']['T'], ref['evidence']['T'])
            replay_count += 3
            for tag in rt.TAGS:
                for permutation in PERM_SEEDS:
                    s = rt.read(run / f'seed_{seed}/{pair}/{tag}/perm_{permutation}.json')
                    assert s['status'] == 'completed' and not s['official_test_used'] and s['bias_preserved']
                    assert s['phase'] == 'dev' and s['seed'] == seed and s['gpu'] == seed - 42
                    assert s['permutation'] == permutation
                    assert s['plan'] == frozen['shuffled'][pair][tag][str(permutation)]
                    assert s['views_sha256'] == ref['views_sha256'] == ctrl['views_sha256']
                    assert s['checkpoint_sha256'] == rt.digest(rt.e35.checkpoint(pair, seed))
                    assert [d['k'] for d in s['clusters']] == s['plan']['widths']
                    for d in s['clusters']:
                        assert sorted(set(d['labels'])) == list(range(d['k']))
                    for mode in rt.MODES:
                        rt.verify_metric(s['metrics'][mode], s['evidence'][mode])
                        count += 1
    assert (count, replay_count, controls_count) == (432, 36, 36)
    analysis = rt.read(run / 'analysis/analysis.json')
    assert len(analysis['rows']) == 432 and len(analysis['contrasts']) == 135
    for row in analysis['contrasts']:
        if row['comparator'] != 'mean4':
            continue
        expected = []
        for seed in rt.SEEDS:
            values = [r for r in analysis['rows'] if (r['pair'], r['tag'], r['mode'], r['seed']) ==
                      (row['pair'], row['tag'], row['mode'], seed)]
            assert len(values) == 4
            expected.append(values[0]['T'] - sum(v['S'] for v in values) / 4)
        assert rt.np.allclose(expected, row['per_seed'], atol=1e-12, rtol=0)
    for p, h in analysis['source_hashes'].items():
        assert rt.digest(run / p) == h
    size = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert size < 1 << 30
    rt.write(run / 'analysis/verification.json', dict(status='passed', utc=rt.utc(), new_shuffled=count,
        fold_replay=replay_count, teacher_controls=controls_count, bytes=size,
        gpu_exit_codes=[w['exit_code'] for w in manifest['workers']], data_open_attempts=rt.DENIED,
        primary_pass=analysis['primary_pass']))
    print('AUDIT PASS', count, replay_count, controls_count)


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
