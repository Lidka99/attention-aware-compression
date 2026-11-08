"""2026-11-08_E038_operator_alokacja: audyt integralności pomiarów; OPIS.md.
Wejście: zakończony run; wyjście: verification.json, bez GPU i danych testowych.
"""
import sys
from pathlib import Path

import runtime as rt


def audit(run):
    manifest, frozen = rt.verify_run(run)
    count, replay_count, controls_count = 0, 0, 0
    for pair in rt.PAIRS:
        for seed in rt.SEEDS:
            ctrl = rt.read(run / f'seed_{seed}/{pair}/controls.json')
            expected_ctrl = frozen['references'][pair]['b50'][str(seed)]['controls']
            rt.check_reference(ctrl['metrics'], ctrl['evidence'], expected_ctrl['metrics'], expected_ctrl['evidence'])
            controls_count += 3
            for tag in rt.TAGS:
                s = rt.read(run / f'seed_{seed}/{pair}/{tag}/summary.json')
                ref = frozen['references'][pair][tag][str(seed)]
                assert s['status'] == 'completed' and s['plan'] == frozen['plans'][pair][tag]
                assert s['seed'] == seed and s['gpu'] == seed - 42
                assert s['phase'] == 'dev' and not s['official_test_used'] and s['bias_preserved']
                assert s['checkpoint_sha256'] == rt.digest(rt.e35.checkpoint(pair, seed))
                assert s['views_sha256'] == ref['views_sha256'] == ctrl['views_sha256']
                for profile in ('U', 'T'):
                    diag = s['diagnostics'][profile]
                    assert [d['k'] for d in diag] == s['plan'][profile]
                    for d in diag:
                        assert len(d['keep']) == len(set(d['keep'])) == d['k']
                        assert min(d['keep']) >= 0 and max(d['keep']) < d['n']
                    for mode in rt.MODES:
                        rt.verify_metric(s['metrics'][profile][mode], s['evidence'][profile][mode])
                        count += 1
                    if tag == 'b50':
                        r = s['replay'][profile]
                        rt.check_reference(r['metrics'], r['evidence'], ref['metrics'][profile], ref['evidence'][profile])
                        replay_count += 3
    assert (count, replay_count, controls_count) == (360, 120, 60)
    analysis = rt.read(run / 'analysis/analysis.json')
    assert len(analysis['rows']) == 180 and len(analysis['contrasts']) == 135
    for row in analysis['rows']:
        assert abs(row['I'] - ((row['FT'] - row['FU']) - (row['MT'] - row['MU']))) < 1e-12
    for p, h in analysis['source_hashes'].items():
        assert rt.digest(run / p) == h
    size = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert size < 1 << 30
    rt.write(run / 'analysis/verification.json', dict(status='passed', utc=rt.utc(),
        new_mag=count, fold_replay=replay_count, teacher_controls=controls_count,
        gpu_exit_codes=[w['exit_code'] for w in manifest['workers']], bytes=size,
        data_open_attempts=rt.DENIED, primary_pass=analysis['primary_pass']))
    print('AUDIT PASS', count, replay_count, controls_count)


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
