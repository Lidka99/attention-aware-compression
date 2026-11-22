"""2026-11-22_E044_l06_tiny: niezależny audyt i agregacja oceny na Tiny val.
Wejścia: zakończony run E044 (manifest, frozen, ready/status, correctness.npz).
Wyjścia: analysis/audit.json, summary.json, per_seed.csv; bez GPU i treningu.
Protokół i kryterium: OPIS.md.
"""

import argparse
import csv
import json
import math
import os
import statistics as st
import sys

import numpy as np

import kontrola_cpu as cpu


CPU_RUN = cpu.ROOT / 'runs/E044/cpu_20261122T020819Z_00821a3f'
FROZEN_SHA = 'b145cf9d1c0e7e6785963dfe425a4c3df694db6ef8f4422d6ec1bdf70c191908'
RUNS = CPU_RUN.parent
ROWS = ('teacher', *(f'{t}/{p}' for t in cpu.TAGS for p in cpu.PROFILES))
ORDER = {k: i for i, k in enumerate(ROWS)}
DRAWS, SEED, CLASSES, PER_CLASS = 10000, 20261122, 200, 50


def read(path):
    return json.loads(path.read_bytes())


def audit(run):
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed' and m['phase'] == 'tiny_val' and m['assessments'] == 28
    assert not m['training'] and not m['dev_used'] and not m['fresh_project_test']
    assert m['run'] == cpu.rel(run) and m['frozen_sha256'] == FROZEN_SHA
    workers = m['workers']
    assert [(w['seed'], w['gpu']) for w in workers] == [(s, i) for i, s in enumerate(cpu.SEEDS)]
    assert all(w['exit_code'] == 0 for w in workers)
    assert len({w['uuid'] for w in workers}) == len({w['pid'] for w in workers}) == 4
    assert not m['gpu_end']['compute_apps']
    assert m['gpu_end'].get('inference_released', m.get('inference_released')) is True

    f = read(run / 'frozen.json')
    assert cpu.digest(run / 'frozen.json') == FROZEN_SHA
    assert f['pair'] == cpu.PAIR and f['planned_evaluation'] == m['settings']
    assert f['data']['evaluation']['fresh_project_test'] is False
    assert f['data']['tiny_used_for_selecting_L06'] is False
    for name, h in m['source_sha256'].items():
        assert cpu.digest(run / 'source' / name) == h, name
    assert m['source_sha256']['OPIS.md'] == f['protocol_sha256']
    if 'analiza.py' in m['source_sha256']:
        assert cpu.digest(cpu.HERE / 'analiza.py') == m['source_sha256']['analiza.py']

    expected = {f'seed_{s}/{n}' for s in cpu.SEEDS for n in ('ready.json', 'status.json', 'correctness.npz')}
    assert expected <= set(m['output_sha256'])
    for name, h in m['output_sha256'].items():
        assert not cpu.Path(name).is_absolute() and cpu.digest(run / name) == h, name

    gate = read(run / 'inference_gate.json')
    assert gate['run'] == m['run']
    assert gate['pid_gpu_uuid'] == {str(w['pid']): w['uuid'] for w in workers}
    assert set(gate['ready_sha256']) == {str(s) for s in cpu.SEEDS}
    for s in cpu.SEEDS:
        name = f'seed_{s}/ready.json'
        assert gate['ready_sha256'][str(s)] == m['output_sha256'][name] == cpu.digest(run / name)

    data, rows, corrects, labels0 = f['data'], [], {}, None
    costs = 0
    for w in workers:
        seed, gpu = w['seed'], w['gpu']
        folder = run / f'seed_{seed}'
        status, ready = read(folder / 'status.json'), read(folder / 'ready.json')
        assert (status['seed'], status['gpu'], status['pid']) == (seed, gpu, w['pid'])
        assert (ready['seed'], ready['gpu'], ready['pid']) == (seed, gpu, w['pid'])
        assert ready['frozen_sha256'] == FROZEN_SHA and ready['source_state_unchanged'] is True
        assert ready['checkpoint_sha256'] == data['checkpoints'][str(seed)]['sha256']
        assert ready['a_views_sha256'] == data['calibration']['views_sha256']
        assert status['status'] == 'completed' and status['assessments'] == 7
        assert status['model_states_unchanged'] is True and not status['denied']
        assert status['val_cache_sha256'] == data['evaluation']['sha256']
        assert set(status['files_opened']) == {
            data['calibration']['cache'], data['evaluation']['cache'], data['checkpoints'][str(seed)]['path']}
        assert set(ready['models']) == set(ROWS)

        meta = {}
        for key in ROWS:
            b = ready['models'][key]
            if key == 'teacher':
                budget, profile, expected, widths, caps = '-', key, f['teacher'], f['sizes'], f['teacher']
            else:
                budget, profile = key.split('/')
                expected = f['budgets'][budget]['profiles'][profile]
                widths, caps = expected['widths'], f['budgets'][budget]['caps']
            assert b['widths'] == widths and len(widths) == 12
            assert (b['params'], b['mac']) == (expected['params'], expected['mac'])
            assert 999 * caps['params'] <= 1000 * b['params'] <= 1000 * caps['params']
            assert 999 * caps['mac'] <= 1000 * b['mac'] <= 1000 * caps['mac']
            assert (b['cma_samples'], b['cma_batches']) == (4096, 32)
            assert b['bn_counters_all_0_to_32'] is True
            assert isinstance(b['state_sha256'], str) and len(b['state_sha256']) == 64
            meta[key] = dict(b=b, budget=budget, profile=profile, caps=caps)
            costs += 1
        assert len({meta[k]['b']['state_sha256'] for k in ROWS}) == 7

        with np.load(folder / 'correctness.npz') as z:
            assert set(z.files) == {'models', 'labels', 'correct', 'indices'}, z.files
            models, labels, correct, indices = z['models'], z['labels'], z['correct'], z['indices']
        n = int(status['samples'])
        assert [str(x) for x in models] == list(ROWS)
        assert indices.dtype.kind in 'iu' and np.array_equal(indices, np.arange(n))
        assert labels.shape == (n,) and np.issubdtype(labels.dtype, np.integer)
        assert np.array_equal(np.bincount(labels, minlength=CLASSES), np.full(CLASSES, PER_CLASS))
        assert correct.shape == (len(ROWS), n) and correct.dtype == np.bool_
        assert n == data['evaluation']['count'] == 10000
        if labels0 is None:
            labels0 = labels.copy()
        else:
            assert np.array_equal(labels, labels0)

        corrects[seed] = correct
        for key in ROWS:
            b, caps = meta[key]['b'], meta[key]['caps']
            hits = int(correct[ORDER[key]].sum())
            rows.append(dict(seed=seed, gpu=gpu, key=key, budget=meta[key]['budget'],
                             profile=meta[key]['profile'], correct=hits, samples=n,
                             top1_pct=100 * hits / n, params=b['params'], mac=b['mac'],
                             p_pct_of_U=100 * b['params'] / caps['params'],
                             mac_pct_of_U=100 * b['mac'] / caps['mac'],
                             p_reduction_pct=100 * (1 - b['params'] / f['teacher']['params'])))

    assert len(rows) == costs == 28 and len(corrects) == 4
    proof = dict(status='PASS', run=m['run'], rows=len(rows), gpu_count=4,
                 gpu_mapping=[{k: w[k] for k in ('seed', 'gpu', 'pid', 'uuid', 'exit_code')} for w in workers],
                 cma_models=28, bn_counters_all_0_to_32=True, costs_verified=costs,
                 manifest_sha256=cpu.digest(run / 'manifest.json'), frozen_sha256=FROZEN_SHA,
                 checks=['frozen_and_sources', 'output_hashes', 'inference_gate', 'four_gpu_exit0',
                         'ready_and_status', 'costs_and_widths', 'cma_state', 'correctness_npz',
                         'labels_stratification'])
    return f, rows, corrects, labels0, proof


def bootstrap(delta, labels, draws=DRAWS, seed=SEED):
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    total = np.zeros(int(draws), dtype=np.float64)
    for c in range(CLASSES):
        group = np.flatnonzero(labels == c)
        counts = rng.multinomial(group.size, np.full(group.size, 1.0 / group.size), size=int(draws))
        total += counts @ delta[group]
    means = 100 * total / delta.size
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(estimate_pp=100 * float(delta.mean()), ci95_pp=[float(lo), float(hi)],
                draws=int(draws), seed=int(seed), stratified_by='labels')


def paired_bootstrap(frozen, corrects, labels):
    rule = frozen['planned_evaluation']
    tag, contrast = rule['primary_budget'], rule['contrast']
    a, b = contrast.split('-')
    assert a in cpu.PROFILES and b in cpu.PROFILES
    left = np.stack([corrects[s][ORDER[f'{tag}/{a}']].astype(np.int8) for s in cpu.SEEDS])
    right = np.stack([corrects[s][ORDER[f'{tag}/{b}']].astype(np.int8) for s in cpu.SEEDS])
    delta = (left - right).mean(axis=0)
    return bootstrap(delta, labels, rule['bootstrap_draws'], rule['bootstrap_seed'])


def summarize(frozen, rows, boot):
    by = {(r['seed'], r['key']): r for r in rows}
    assert len(by) == 28 and len({r['samples'] for r in rows}) == 1
    n = rows[0]['samples']
    rule = frozen['planned_evaluation']
    assert rule['contrast'] == 'L06-U' and rule['metric'] == 'top1'

    def pair(a, b):
        counts = [by[s, a]['correct'] - by[s, b]['correct'] for s in cpu.SEEDS]
        pp = [100 * c / n for c in counts]
        return dict(per_seed_counts=counts, per_seed_pp=pp, sum_counts=sum(counts),
                    mean_pp=st.mean(pp), sd_pp=st.stdev(pp), positive=sum(c > 0 for c in counts),
                    tied=counts.count(0), negative=sum(c < 0 for c in counts))

    def key(tag, profile):
        return 'teacher' if profile == 'teacher' else f'{tag}/{profile}'

    budgets = {}
    for tag in (rule['primary_budget'], rule['transfer_budget']):
        contrasts = {f'{a}-{b}': pair(f'{tag}/{a}', f'{tag}/{b}')
                     for a, b in (('L06', 'U'), ('L06', 'T-proj'), ('T-proj', 'U'))}
        top1 = {p: [by[s, key(tag, p)]['top1_pct'] for s in cpu.SEEDS] for p in ('teacher', *cpu.PROFILES)}
        budgets[tag] = dict(
            mean_top1_pct={p: st.mean(v) for p, v in top1.items()},
            sd_top1_pct={p: st.stdev(v) for p, v in top1.items()},
            contrasts=contrasts,
            loss_vs_teacher={p: pair('teacher', f'{tag}/{p}') for p in cpu.PROFILES})

    main = budgets[rule['primary_budget']]['contrasts'][rule['contrast']]
    passed = main['mean_pp'] > 0 and main['positive'] >= rule['minimum_strictly_positive_seeds']
    assert math.isclose(boot['estimate_pp'], main['mean_pp'], abs_tol=1e-9)
    transfer = budgets[rule['transfer_budget']]['contrasts'][rule['contrast']]
    return dict(seeds=list(cpu.SEEDS), samples_per_seed=n, rule=rule,
                gpu_to_seed=rule.get('gpu_to_seed'), budgets=budgets,
                primary=dict(budget=rule['primary_budget'], contrast=rule['contrast'],
                             meets_primary_criterion=passed, bootstrap=boot, **main),
                transfer=dict(budget=rule['transfer_budget'], contrast=rule['contrast'], **transfer),
                primary_passed=passed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=cpu.Path)
    run = p.parse_args().run.resolve()
    assert run.parent == RUNS and run.is_dir()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not sys.flags.optimize and sys.dont_write_bytecode
    frozen, rows, corrects, labels, proof = audit(run)
    boot = paired_bootstrap(frozen, corrects, labels)
    summary = summarize(frozen, rows, boot)
    out = run / 'analysis'
    out.mkdir()
    cpu.write(out / 'audit.json', proof)
    cpu.write(out / 'summary.json', summary)
    with (out / 'per_seed.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('AUDYT PASS', proof, flush=True)


if __name__ == '__main__':
    main()
