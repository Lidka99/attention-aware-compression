"""E042: audyt zapisów dev, wybór jednej geometrii i sparowane wyniki.
Wejście: zakończony run E042; wyjście: audyt i CSV. Kryterium: OPIS.md.
"""

import argparse
import math
from pathlib import Path
from statistics import mean, stdev

from experiment import (DATA_FILES, HERE, PAIRS, ROOT, RUNS, SEEDS,
                        compare_reference, design, e35, rt, split_evidence)


def summarize(items, plans, baseline, best):
    def pp_diff(profile, reference):
        differences = [a['correct'] - b['correct']
                       for a, b in zip(items[profile], items[reference])]
        return dict(per_seed_count=differences, per_seed_pp=[d / 50 for d in differences],
                    total_count=sum(differences), mean_pp=sum(differences) / 200,
                    sd_pp=stdev(differences) / 50, positive=sum(d > 0 for d in differences))

    u = plans['U']
    rows = []
    for name, observations in items.items():
        if name == 'teacher':
            continue
        plan = plans[name]
        vs_ref = pp_diff(name, baseline)
        rows.append(dict(profile=name, family=plan['family'], params=plan['params'], mac=plan['mac'],
            p_over_cap=plan['params']/u['params'], mac_over_cap=plan['mac']/u['mac'],
            top1=mean(o['correct'] for o in observations)/50,
            nll=mean(o['metric']['nll'] for o in observations),
            loss_vs_teacher_pp=-pp_diff(name, 'teacher')['mean_pp'],
            vs_U_pp=pp_diff(name, 'U')['mean_pp'], vs_T_proj_pp=pp_diff(name, 'T-proj')['mean_pp'],
            vs_stronger_ref_pp=vs_ref['mean_pp'], sd_vs_stronger_ref_pp=vs_ref['sd_pp'],
            positive_vs_stronger_ref=vs_ref['positive'], best_candidate=name == best,
            stronger_reference=name == baseline))
    return rows, pp_diff(best, baseline)


def audit(run):
    run = run.resolve()
    assert run.parent == RUNS
    manifest, frozen = rt.verify_run(run)
    assert manifest['settings']['phase'] == 'dev' and manifest['settings']['mode'] == 'CMA'
    assert not manifest['settings']['training'] and manifest['elapsed_seconds'] <= 1800
    assert manifest['code_hashes'] == rt.hashes(HERE)
    assert rt.read(run / 'source/check.json')['status'] == 'passed'
    assert split_evidence(frozen['indices']) == frozen['splits']
    design.validate_bank(frozen['bank'])
    assert not any(p.suffix in ('.pt', '.pth', '.ckpt', '.safetensors') for p in run.rglob('*'))

    opened = {rt.rel(p) for p in DATA_FILES}
    gate_time = max(rt.read(run / f'seed_{s}/reference_gate.json')['utc'] for s in SEEDS)
    for seed in SEEDS:
        folder = run / f'seed_{seed}'
        assert rt.read(folder / 'reference_gate.json')['passed']
        status = rt.read(folder / 'status.json')
        assert status['original_states_unchanged'] and status['barrier_passed_utc'] >= gate_time
        assert set(status['data_opened']) == opened and not status['data_open_attempts']
        expected = [f'{pair}/{r["profile"]}' for pair in PAIRS
                    for r in frozen['bank']['pairs'][pair]['references'][1:] +
                             frozen['bank']['pairs'][pair]['candidates']]
        assert status['completed'] == expected

    telemetry = [rt.json.loads(line) for line in (run / 'monitor.jsonl').read_text().splitlines()]
    active = set()
    for event in telemetry:
        for line in event['gpu'].splitlines():
            gpu, _, _, memory, utilization = line.split(',')
            if int(memory) >= 256 or int(utilization) > 0:
                active.add(int(gpu))
    assert active == {0, 1, 2, 3} and telemetry[-1]['exit_codes'] == [0] * 4

    all_rows, all_seeds, contrasts, hashes = [], [], {}, {}
    for pair in PAIRS:
        menu = frozen['bank']['pairs'][pair]
        plans = {p['profile']: p for p in menu['references'] + menu['candidates']}
        items = {k: [] for k in ['teacher', *plans]}
        for seed in SEEDS:
            folder = run / f'seed_{seed}' / pair
            assert {p.stem for p in folder.glob('*.json')} == set(items)
            s, c = (rt.read(ROOT / p) for p in frozen['reference_paths'][pair][str(seed)])
            checkpoint = frozen['hashes'][rt.rel(e35.checkpoint(pair, seed))]
            for name in items:
                path = folder / f'{name}.json'
                entry = rt.read(path)
                assert entry['pair'] == pair and entry['seed'] == seed and entry['phase'] == 'dev'
                assert not entry['official_test_used'] and not entry['reserve_used']
                assert entry['views_sha256'] == s['views_sha256'] and entry['checkpoint_sha256'] == checkpoint
                metric, ev = entry['metric'], entry['evidence']
                rt.verify_metric(metric, ev)
                assert ev['samples'] == 5000 and ev['calibration']['samples'] == 4096
                assert ev['calibration']['params_unchanged'] and ev['calibration']['estimator'] == 'CMA_reset'
                assert ev['calibration']['parameters_sha256'] == ev['weights_sha256']
                assert math.isclose(entry['correct'] / 5000, metric['top1'], rel_tol=0, abs_tol=1e-12)
                if name == 'teacher':
                    assert entry['profile'] == name
                    compare_reference(entry, c['metrics']['CMA'], c['evidence']['CMA'])
                else:
                    plan, diag = plans[name], entry['clusters']
                    assert entry['plan'] == plan and entry['actual'] == {k: plan[k] for k in ('params', 'mac')}
                    assert [d['k'] for d in diag] == plan['widths'] and [d['n'] for d in diag] == frozen['bank']['sizes']
                    assert all(len(d['labels']) == d['n'] and set(d['labels']) == set(range(d['k'])) for d in diag)
                    assert tuple(design.g.costs(plan['widths'], design.classes(pair))) == tuple(entry['actual'].values())
                    if name == 'U':
                        compare_reference(entry, s['metrics']['U']['CMA'], s['evidence']['U']['CMA'])
                items[name].append(entry)
                hashes[rt.rel(path)] = rt.digest(path)
                all_seeds.append(dict(pair=pair, seed=seed, profile=name, correct=entry['correct'],
                    top1=entry['correct']/50, nll=metric['nll'], params=plans.get(name, {}).get('params'),
                    mac=plans.get(name, {}).get('mac')))

        assert all(len(observations) == len(SEEDS) for observations in items.values())
        rank = lambda name: (-sum(v['correct'] for v in items[name]), plans[name]['mac'], plans[name]['params'], name)
        reference = min(('U', 'T-proj'), key=rank)
        best = min((p['profile'] for p in menu['candidates']), key=rank)
        rows, effect = summarize(items, plans, reference, best)
        averages = {name: sum(v['correct'] for v in scores) / 200 for name, scores in items.items()}
        gate = effect['total_count'] >= frozen['primary']['minimum_count_gain'] and \
               effect['positive'] >= frozen['primary']['minimum_positive']
        contrasts[pair] = dict(best=best, candidate_count=len(menu['candidates']),
            reference=reference, teacher=averages['teacher'], cma_U=averages['U'], cma_T_proj=averages['T-proj'],
            cma_best=averages[best], loss_vs_teacher_pp=averages['teacher']-averages[best],
            best_cost=dict(params=plans[best]['params'], mac=plans[best]['mac']),
            gate=gate, **effect)
        all_rows.extend(dict(pair=pair, **row) for row in rows)

    assert set(contrasts) == set(PAIRS) and len(hashes) == len(all_seeds) <= 216
    expected_evaluations = len(SEEDS) * sum(len(m['candidates']) + 3 for m in frozen['bank']['pairs'].values())
    assert len(hashes) == expected_evaluations
    out = run / 'analysis'
    out.mkdir()
    primary = frozen['primary']['pair']
    rt.csv_write(out / 'per_seed.csv', all_seeds)
    rt.csv_write(out / 'profiles.csv', all_rows)
    rt.write(out / 'summary.json', dict(phase='dev', primary_pair=primary, primary_pass=contrasts[primary]['gate'],
        contrasts=contrasts, evaluations=expected_evaluations, independent_validation=False))
    rt.write(out / 'audit.json', dict(passed=True, active_gpus=sorted(active),
        checks=['source_hashes', 'checkpoint_parameters', 'reference_barrier', 'A_and_dev_no_reserve',
                'full_profile_grid', 'both_costs', 'paired_metric_counts', 'four_gpu_exits'],
        manifest_sha256=rt.digest(run / 'manifest.json'), source_hashes=hashes))
    print('AUDIT PASS', expected_evaluations, 'evaluations')
    for pair, result in contrasts.items():
        print(pair, 'BEST', result['best'], 'VS', result['reference'],
              f'{result["mean_pp"]:+.4f} pp', f'{result["positive"]}/4', 'GATE', result['gate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    audit(parser.parse_args().run)
