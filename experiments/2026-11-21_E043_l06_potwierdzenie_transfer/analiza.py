"""2026-11-21_E043_l06_potwierdzenie_transfer: audyt i sparowana agregacja.
Czyta wyłącznie zapisane metadane i 28 liczników trafień; zapisuje audit.json,
summary.json i per_seed.csv, bez inferencji. Pełny protokół: OPIS.md.
"""

import argparse
import csv
import json
import math
import os
import statistics as st

import kontrola_cpu as cpu
from ocena_rezerwy import AUTHORIZATION, FROZEN_SHA, ROWS, RUNS, read


def audit(run):
    manifest_path = run / 'manifest.json'
    m = read(manifest_path)
    assert m['status'] == 'completed' and m['phase'] == 'reserve' and m['assessments'] == 28
    assert m['approval']['request'] == AUTHORIZATION and m['reserve_used'] and m['reserve_inference_released']
    assert not any(m[k] for k in ('training', 'dev_used', 'official_test_used'))
    assert m['environment']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3' and m['environment']['CONDA_DEFAULT_ENV'] == 'attention'
    workers = m['workers']
    assert [(w['seed'], w['gpu'], w['exit_code']) for w in workers] == [(s, i, 0) for i, s in enumerate(cpu.SEEDS)]
    assert len({w['uuid'] for w in workers}) == len({w['pid'] for w in workers}) == 4
    assert not m['gpu_end']['compute_apps']
    gate = read(run / 'inference_gate.json')
    assert gate['run'] == m['run'] == cpu.rel(run)
    assert gate['pid_gpu_uuid'] == {str(w['pid']): w['uuid'] for w in workers}
    assert cpu.digest(run / 'frozen.json') == gate['frozen_sha256'] == m['frozen_sha256'] == FROZEN_SHA
    f = read(run / 'frozen.json')
    for name, h in m['source_sha256'].items():
        assert cpu.digest(run / 'source' / name) == h, name
    assert m['source_sha256']['OPIS.md'] == f['protocol_sha256']
    assert cpu.digest(cpu.HERE / 'analiza.py') == m['source_sha256']['analiza.py']
    expected_files = {f'seed_{s}/{n}.json' for s in cpu.SEEDS for n in ('status', 'ready', 'metrics')}
    assert set(m['output_sha256']) == expected_files
    for name, h in m['output_sha256'].items():
        assert cpu.digest(run / name) == h, name

    n = f['planned_evaluation']['samples']
    data = f['data']
    eq_states = 0
    rows = []
    for w in workers:
        seed, gpu = w['seed'], w['gpu']
        folder = run / f'seed_{seed}'
        s, ready, result = [read(folder / f'{x}.json') for x in ('status', 'ready', 'metrics')]
        assert all((v['seed'], v['gpu']) == (seed, gpu) for v in (s, ready, result))
        assert ready['pid'] == w['pid'] == s['pid'] and ready['cuda_visible_devices'] == str(gpu)
        assert ready['frozen_sha256'] == FROZEN_SHA and ready['source_state_unchanged']
        assert s['status'] == 'completed' and s['reserve_sliced'] and s['assessments'] == len(ROWS)
        assert s['classes']*s['reserve_count_per_class'] == n and (s['classes'], s['reserve_count_per_class']) == (100, 50)
        assert not s['denied'] and set(s['files_opened']) == {
            data['split'], data['index_space'], data['checkpoints'][str(seed)]['path']}
        assert ready['train_file_sha256'] == data['expected_train_sha256']
        assert ready['checkpoint_sha256'] == data['checkpoints'][str(seed)]['sha256']
        assert ready['a_indices_sha256'] == data['index_list_sha256']['A']
        assert ready['a_views_sha256'] == data['expected_A_views_sha256']
        assert ready['reserve_indices_sha256'] == result['reserve_indices_sha256'] == data['index_list_sha256']['reserve']
        assert m['output_sha256'][f'seed_{seed}/ready.json'] == gate['ready_sha256'][str(seed)]
        assert ready['utc'] <= gate['utc'] == result['release_utc'] <= s['finished_utc']
        assert set(ready['models']) == set(result['models']) == set(ROWS)
        for key in ROWS:
            b, r = ready['models'][key], result['models'][key]
            assert b['cma_mode'] == f['planned_evaluation']['mode'] and (b['cma_samples'], b['cma_batches'], b['cma_last_batch']) == (4096, 32, 128)
            assert b['bn_counters'] and all(v == [0, 32] for v in b['bn_counters'].values())
            assert r['samples'] == n and type(r['correct']) is int and 0 <= r['correct'] <= n
            assert math.isclose(r['top1_pct'], 100*r['correct']/n, abs_tol=1e-12) and r['state_unchanged_in_eval']
            if key == 'teacher':
                budget, profile = '-', key
                expected, widths = f['teacher'], f['sizes']
                caps = expected
            else:
                budget, profile = key.split('/')
                expected = f['budgets'][budget]['profiles'][profile]
                widths, caps = expected['widths'], f['budgets'][budget]['caps']
            assert b['widths'] == widths and len(widths) == 12
            assert b['actual'] == {k: expected[k] for k in ('params', 'mac')}
            for k in b['actual']:
                assert 999*caps[k] <= 1000*b['actual'][k] <= 1000*caps[k]
            if budget != 'b30':
                assert b['e042_state_sha256'] and b['e042_state_equal'] == (b['e042_state_sha256'] == b['state_after_cma_sha256'])
                eq_states += b['e042_state_equal']
            rows.append(dict(seed=seed, gpu=gpu, key=key, budget=budget, profile=profile,
                             correct=r['correct'], samples=n, top1_pct=r['top1_pct'], **b['actual'],
                             p_pct_of_U=100*b['actual']['params']/caps['params'], mac_pct_of_U=100*b['actual']['mac']/caps['mac'],
                             p_reduction_pct=100*(1-b['actual']['params']/f['teacher']['params'])))
    proof = dict(status='PASS', run=m['run'], rows=len(rows), gpu_count=4, e042_states_equal=eq_states,
                 e042_states_checked=16, manifest_sha256=cpu.digest(manifest_path), frozen_sha256=FROZEN_SHA,
                 checks=['authorization', 'frozen_and_source_hashes', 'four_gpu_exit0', 'calibration_barrier',
                         'all_roles_and_hashes', 'params_mac_and_widths', 'BN_and_model_state', 'all_paired_counts'])
    return f, rows, proof


def summarize(frozen, rows):
    by = {(r['seed'], r['key']): r for r in rows}
    assert len(by) == 28
    n = frozen['planned_evaluation']['samples']
    rule = frozen['decision']
    assert rule['contrast'] == 'L06-U' and rule['metric'] == 'top1_CMA'

    def pair(a, b):
        counts = [by[s, a]['correct'] - by[s, b]['correct'] for s in cpu.SEEDS]
        pp = [100*c/n for c in counts]
        return dict(per_seed_counts=counts, per_seed_pp=pp, sum_counts=sum(counts), mean_pp=st.mean(pp),
                    sd_pp=st.stdev(pp), positive=sum(c > 0 for c in counts), tied=counts.count(0),
                    negative=sum(c < 0 for c in counts))

    results = {}
    for tag in (rule['primary_budget'], rule['transfer_budget']):
        contrasts = {f'{a}-{b}': pair(f'{tag}/{a}', f'{tag}/{b}') for a, b in
                     (('L06', 'U'), ('L06', 'T-proj'), ('T-proj', 'U'))}
        main = contrasts['L06-U']
        passed = main['sum_counts'] >= rule['min_sum_count_gain'] and main['positive'] >= rule['min_strictly_positive_seeds']
        absolute = {p: st.mean(by[s, key]['top1_pct'] for s in cpu.SEEDS) for p, key in
                    [('teacher', 'teacher'), *((p, f'{tag}/{p}') for p in cpu.PROFILES)]}
        results[tag] = dict(passed=passed, mean_top1_pct=absolute, contrasts=contrasts,
                            loss_vs_teacher={p: pair('teacher', f'{tag}/{p}') for p in cpu.PROFILES})
    return dict(seeds=list(cpu.SEEDS), samples_per_seed=n, rule=rule, budgets=results,
                primary_passed=results[rule['primary_budget']]['passed'], transfer_passed=results[rule['transfer_budget']]['passed'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=cpu.Path)
    run = p.parse_args().run.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and run.parent == RUNS
    assert not cpu.sys.flags.optimize and cpu.sys.dont_write_bytecode
    frozen, rows, proof = audit(run)
    summary = summarize(frozen, rows)
    out = run / 'analysis'
    out.mkdir()
    cpu.write(out / 'audit.json', proof)
    cpu.write(out / 'summary.json', summary)
    with (out / 'per_seed.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print('AUDYT PASS', proof, flush=True)


if __name__ == '__main__':
    main()
