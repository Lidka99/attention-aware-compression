"""2026-11-22_E047_l06_b10_imagenetv2: jednorazowa ocena zewnętrzna i audyt.
Wejścia: zatwierdzona próba ImageNetV2, stany E046 i A4096 z Tiny train.
Wyjścia: bramka 12 stanów, wyniki sparowane i bootstrap. Zob. OPIS.md.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
import traceback
from pathlib import Path

import przygotuj as data

e46, cpu = data.e46, data.cpu
ROOT, HERE, RUNS = data.ROOT, data.HERE, data.RUNS
SEEDS, TAGS, LIMIT = e46.SEEDS, e46.TAGS, e46.LIMIT
read, write, rel, digest, utc = e46.read, e46.write, e46.rel, e46.digest, e46.utc
DATA_RUN = RUNS / 'cpu_20261122T053249Z_169cdc5b'
DATA_MANIFEST_SHA = 'c21776e538e245f4c96684929580f1744adc4ccdc53ee7a2c895b8dbf29afb37'
TEST = DATA_RUN / data.DATA_FILE
RELEASE = RUNS / 'imagenetv2_b33_first_use.json'
R46 = ROOT / 'runs/E046/20261122T045655Z_0b0c53b7'
R46_SHA = 'bea790ff30c8ef2d04f87575cee954687e2f4f35399a5789c7ae7d92b7a9f30d'
AUDIT46_SHA = 'd7389cccbdd866f23638b344b24f9add03c8b2d6e71b8c34a28b3490a8a60610'
EVIDENCE = ('params', 'mac', 'state_sha256', 'cma_batches')
DRAW_COUNT, BOOT_SEED = 10000, 20261122
SOURCES = ('ocena_v2.py', 'przygotuj.py', 'OPIS.md', 'URUCHOMIENIE.md')


def verified_inputs():
    hashes = {}
    dm = cpu.read(DATA_RUN / 'manifest.json', DATA_MANIFEST_SHA, hashes)
    assert dm['status'] == 'passed' and not dm['image_evaluation'] and not dm['denied']
    assert dm['model_plan_sha256'] == e46.FROZEN_SHA
    assert all(digest(HERE / n) == h for n, h in dm['source_sha256'].items())
    for name, sha in dm['output_sha256'].items():
        if name != data.DATA_FILE:
            cpu.verify(DATA_RUN / name, sha, hashes)
    audit_data = read(DATA_RUN / 'audit.json')
    cohort = read(DATA_RUN / 'cohort.json')
    assert audit_data['status'] == 'PASS' and audit_data['after'] == dm['samples']
    assert cohort['counts'] == audit_data['counts'] and min(cohort['counts']) >= data.MIN_CLASS
    assert len(cohort['counts']) == 200 and sum(cohort['counts']) == dm['samples'] >= data.MIN_TOTAL
    selected = [r for r in cohort['entries'] if r['retained']]
    assert len(selected) == dm['samples'] and len({r['id'] for r in selected}) == dm['samples']
    assert all(r['min_tiny_hamming'] > data.MAX_HAMMING and not r['tiny_exact_match']
               and not r['similar_within'] for r in selected)

    f, inputs45 = e46.load_plan()
    run = cpu.read(R46 / 'manifest.json', R46_SHA, hashes)
    aud = cpu.read(R46 / 'analysis/audit.json', AUDIT46_SHA, hashes)
    assert run['status'] == 'completed' and run['assessments'] == 12
    assert aud['status'] == 'PASS' and aud['four_gpu_exit0'] and aud['new_L06_scores'] == 4
    assert run['frozen_sha256'] == aud['frozen_sha256'] == e46.FROZEN_SHA
    assert digest(R46 / 'frozen.json') == e46.FROZEN_SHA
    for n in ('ocena_dev.py', 'sprawdz_geometrie.py'):
        cpu.verify(data.E046 / n, run['source_sha256'][n], hashes)
    refs = {}
    for seed in SEEDS:
        name = f'seed_{seed}/metrics.json'
        old = cpu.read(R46 / name, run['output_sha256'][name], hashes)
        assert old['seed'] == seed and old['views_sha256'] == f['calibration']['views_sha256']
        assert set(old['models']) == set(TAGS)
        refs[str(seed)] = {tag: {k: old['models'][tag][k] for k in EVIDENCE} for tag in TAGS}
    inputs = dict(expected_models=refs, uniform_plan=inputs45['uniform_plan'], counts=cohort['counts'],
                  samples=dm['samples'], test_path=rel(TEST), test_sha256=dm['output_sha256'][data.DATA_FILE],
                  cohort_sha256=dm['output_sha256']['cohort.json'], input_sha256=hashes,
                  reference_run=rel(R46), draws=DRAW_COUNT, bootstrap_seed=BOOT_SEED)
    return f, inputs


def protect_test(can_open):
    opened, denied = [], []

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        if Path(os.fsdecode(args[0])).resolve() != TEST:
            return
        if not can_open() or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
            denied.append(rel(TEST))
            raise RuntimeError('E047: próba dostępu do testu poza bramką')
        opened.append(utc())

    sys.addaudithook(guard)
    return opened, denied


def worker(run, gpu):
    m = read(run / 'manifest.json')
    assert run.parent == RUNS and m['status'] == 'running' and m['pid'] == os.getppid()
    f, inputs = e46.phase_inputs(run, m)
    seed = f['measurement']['gpu_to_seed'][str(gpu)]
    allowed = {data.TRAIN, cpu.CKPT[seed]}
    opened, denied = e46.guard_files(allowed)
    test_allowed = False
    test_reads, test_denied = protect_test(lambda: test_allowed)
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', gpu=gpu, seed=seed, pid=os.getpid(), started_utc=utc())
    write(out / 'status.json', status)
    try:
        rt = cpu.load(cpu.E37, '_e047_e037')
        e35, torch, np = rt.e35, rt.torch, rt.np
        count_cost, _ = e46.helpers()
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu) and torch.cuda.device_count() == 1
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        cal = f['calibration']
        with np.load(data.TRAIN, allow_pickle=False) as cache:
            aug = e35.e30.augment(cache['images'], cal['A'])
        view = hashlib.sha256(aug.contiguous().numpy().tobytes()).hexdigest()
        assert view == cal['views_sha256']
        a = e35.e30.normalize(aug)
        del aug
        base = e35.model_source(e46.PAIR, seed)
        state_before = e35.state_hash(base)
        geometry = f['geometry']
        models, proof = {}, {}
        for tag in TAGS:
            if tag == 'teacher':
                model, widths = copy.deepcopy(base), geometry['sizes']
            elif tag == 'b10':
                widths = geometry['uniform']
                model, diag = e35.fold(e46.PAIR, base, inputs['uniform_plan'], 'U')
            else:
                widths = geometry['candidate']['widths']
                model, diag = rt.fold_widths(e46.PAIR, base, widths)
            if tag != 'teacher':
                assert [d['k'] for d in diag] == widths and [d['n'] for d in diag] == geometry['sizes']
                assert torch.equal(model.fc.bias, base.fc.bias)
            model.to('cuda:0')
            cost = count_cost(torch, model)
            meta = e35.e29.calibrate(model, a, 'CMA')
            assert (meta['samples'], meta['batches'], meta['last_batch']) == (4096, 32, 128)
            assert meta['estimator'] == 'CMA_reset' and meta['params_unchanged'] and meta['layers']
            assert all((b['counter_start'], b['counter_end']) == (0, 32) for b in meta['layers'])
            proof[tag] = dict(**cost, state_sha256=e35.state_hash(model), cma_batches=32)
            models[tag] = model
        assert proof == inputs['expected_models'][str(seed)]
        assert e35.state_hash(base) == state_before and not test_reads and not test_denied
        del a, base

        ready = out / 'ready.json'
        write(ready, dict(pid=os.getpid(), gpu=gpu, seed=seed, models=proof, views_sha256=view, utc=utc()))
        gate = run / 'gate.json'
        t0 = time.monotonic()
        while not gate.is_file():
            assert time.monotonic()-t0 < LIMIT and os.getppid() == m['pid']
            time.sleep(.5)
        g = read(gate)
        assert g['ready_sha256'][str(seed)] == digest(ready)
        assert g['release_sha256'] == digest(RELEASE) and read(RELEASE)['run'] == rel(run)
        test_allowed = True
        assert digest(TEST) == inputs['test_sha256']
        with np.load(TEST, allow_pickle=False) as cache:
            raw, labels = cache['images'], cache['labels']
        n = inputs['samples']
        assert raw.shape == (n, 64, 64, 3) and raw.dtype == np.uint8
        assert labels.shape == (n,) and np.bincount(labels, minlength=200).tolist() == inputs['counts']
        x = e35.e30.normalize(torch.from_numpy(raw).permute(0, 3, 1, 2).float()/255)
        y = torch.from_numpy(labels)
        hits, vectors = {}, []
        for tag, model in models.items():
            logits = e35.bn.predict(model, x)
            assert logits.shape == (n, 200) and torch.isfinite(logits).all()
            assert e35.state_hash(model) == proof[tag]['state_sha256']
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            correct = (logits.argmax(1) == y).numpy()
            hits[tag] = int(correct.sum())
            vectors.append(correct)
        assert opened == allowed and not denied and test_reads and not test_denied
        np.savez_compressed(out / 'correctness.npz', labels=labels, correct=np.stack(vectors))
        write(out / 'metrics.json', dict(seed=seed, n=n, hits=hits, models=proof, views_sha256=view,
                                         test_sha256=inputs['test_sha256']))
        status.update(status='completed', assessments=3)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=utc(), opened=sorted(map(rel, opened)), denied=denied+test_denied,
                      test_first_open=test_reads[0] if test_reads else None)
        write(out / 'status.json', status)


def launch():
    env = {k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                                         'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    assert list(env.values()) == ['attention', '0,1,2,3', 'PCI_BUS_ID', ':4096:8', '2', '2', '2']
    assert not RELEASE.exists(), 'Próba została już otwarta do punktacji; użyj analizy zapisanego runu.'
    allowed = {data.TRAIN, cpu.SPLIT, *cpu.CKPT.values()}
    opened, denied = e46.guard_files(allowed)
    f, inputs = verified_inputs()
    assert digest(data.TRAIN) == f['calibration']['sha256']
    import torch
    _, gpu_info = e46.helpers()
    pre = gpu_info()
    assert torch.cuda.device_count() == 4 and not pre['compute_apps']
    assert [g['index'] for g in pre['gpus']] == list('0123')
    assert all(int(g['memory_mib']) < 256 and int(g['utilization']) == 0 for g in pre['gpus'])
    run = data.new_run()
    source = data.snapshot(run, SOURCES)
    shutil.copyfile(R46 / 'frozen.json', run / 'frozen.json')
    write(run / 'references.json', inputs)
    m = dict(status='running', run=rel(run), phase='imagenetv2_b33_tiny200', pid=os.getpid(), started_utc=utc(),
             command=[sys.executable, '-B', *sys.argv], source_sha256=source, data_run=rel(DATA_RUN),
             frozen_sha256=e46.FROZEN_SHA, references_sha256=digest(run / 'references.json'),
             python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
             environment=env, gpu_start=pre, workers=[], training=False)
    write(run / 'manifest.json', m)
    print('RUN', rel(run), flush=True)
    procs, logs, t0 = [], [], time.monotonic()
    try:
        for gpu, seed in enumerate(SEEDS):
            logs.append((run / f'worker_{gpu}.log').open('w'))
            cmd = [sys.executable, '-B', sys.argv[0], 'worker', str(run), str(gpu)]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=logs[-1], stderr=subprocess.STDOUT)
            procs.append(proc)
            m['workers'].append(dict(gpu=gpu, seed=seed, pid=proc.pid, uuid=pre['gpus'][gpu]['uuid'], command=cmd))
        write(run / 'manifest.json', m)
        gate = run / 'gate.json'
        ready = {s: run / f'seed_{s}/ready.json' for s in SEEDS}
        devices = {str(w['pid']): w['uuid'] for w in m['workers']}
        with (run / 'monitor.jsonl').open('w') as log:
            while True:
                codes, telemetry = [p.poll() for p in procs], gpu_info()
                log.write(json.dumps(dict(**telemetry, exit_codes=codes))+'\n')
                log.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0]*4:
                    break
                assert time.monotonic()-t0 < LIMIT, 'Limit 30 min'
                if not gate.exists() and all(p.is_file() for p in ready.values()):
                    assert telemetry['compute_apps'] == devices
                    for w in m['workers']:
                        s, r = w['seed'], read(ready[w['seed']])
                        assert (r['pid'], r['gpu'], r['seed']) == (w['pid'], w['gpu'], s)
                        assert r['models'] == inputs['expected_models'][str(s)]
                        assert r['views_sha256'] == f['calibration']['views_sha256']
                    assert digest(TEST) == inputs['test_sha256']
                    event = dict(run=rel(run), utc=utc(), test_sha256=inputs['test_sha256'], samples=inputs['samples'])
                    with RELEASE.open('x') as marker:
                        json.dump(event, marker)
                    write(gate, dict(**event, exact_states=12, pid_gpu_uuid=devices, release_sha256=digest(RELEASE),
                                     ready_sha256={str(s): digest(p) for s, p in ready.items()}))
                    print('Bramka PASS: 12/12 stanów E046, 4 GPU; otwarto próbę zewnętrzną.', flush=True)
                time.sleep(2)
        assert gate.is_file() and digest(TEST) == inputs['test_sha256'] and not denied and opened == allowed
        assert source == {n: digest(HERE / n) for n in SOURCES}
        files = ['gate.json', 'monitor.jsonl', *(f'worker_{i}.log' for i in range(4))]
        for s in SEEDS:
            folder = run / f'seed_{s}'
            assert read(folder / 'status.json')['status'] == 'completed'
            files.extend(f'seed_{s}/{n}' for n in ('ready.json', 'status.json', 'metrics.json', 'correctness.npz'))
        m.update(status='completed', assessments=12, output_sha256={name: digest(run / name) for name in files})
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        raise
    finally:
        for w, p in zip(m['workers'], procs):
            w['exit_code'] = p.poll()
        for log in logs:
            log.close()
        m.update(seconds=time.monotonic()-t0, finished_utc=utc(), opened=sorted(map(rel, opened)), denied=denied)
        try:
            m['gpu_end'] = gpu_info()
        finally:
            write(run / 'manifest.json', m)
    print('COMPLETED', rel(run), flush=True)


def analyze(run):
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and run.parent == RUNS
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed' and m['phase'] == 'imagenetv2_b33_tiny200' and m['assessments'] == 12
    assert not m['training'] and not m['denied'] and not m['gpu_end']['compute_apps']
    f, inputs = e46.phase_inputs(run, m)
    for name, h in m['source_sha256'].items():
        assert digest(run / 'source' / name) == h
    for name, h in m['output_sha256'].items():
        assert digest(run / name) == h
    assert [(w['gpu'], w['seed'], w['exit_code']) for w in m['workers']] == [(i, s, 0) for i, s in enumerate(SEEDS)]
    gate = read(run / 'gate.json')
    e = read(RELEASE)
    assert gate['release_sha256'] == digest(RELEASE) and e['run'] == rel(run)
    assert gate['exact_states'] == 12 and gate['test_sha256'] == e['test_sha256'] == inputs['test_sha256']
    devices = {str(w['pid']): w['uuid'] for w in m['workers']}
    assert gate['pid_gpu_uuid'] == devices and len(set(devices.values())) == 4
    assert any(json.loads(line)['compute_apps'] == devices for line in (run / 'monitor.jsonl').read_text().splitlines())
    labels_ref = np.repeat(np.arange(200), inputs['counts'])
    rows, all_correct, gains = [], [], []
    n = inputs['samples']
    assert labels_ref.shape == (n,)
    for s in SEEDS:
        folder = run / f'seed_{s}'
        status, result = read(folder / 'status.json'), read(folder / 'metrics.json')
        assert status['status'] == 'completed' and status['assessments'] == 3 and not status['denied']
        assert status['test_first_open'] >= gate['utc']
        assert set(status['opened']) == {rel(data.TRAIN), f['checkpoints'][str(s)]['path']}
        assert digest(folder / 'ready.json') == gate['ready_sha256'][str(s)]
        assert read(folder / 'ready.json')['utc'] <= gate['utc']
        assert result['seed'] == s and result['n'] == n and result['test_sha256'] == inputs['test_sha256']
        assert result['views_sha256'] == f['calibration']['views_sha256']
        assert result['models'] == inputs['expected_models'][str(s)]
        with np.load(folder / 'correctness.npz', allow_pickle=False) as pack:
            correct = pack['correct']
            assert np.array_equal(pack['labels'], labels_ref)
        assert correct.shape == (3, n) and correct.dtype == np.bool_
        counts = dict(zip(TAGS, map(int, correct.sum(axis=1))))
        assert counts == result['hits']
        all_correct.append(correct)
        gain = counts['L06'] - counts['b10']
        gains.append(gain)
        for tag in TAGS:
            params, mac, _, _ = [result['models'][tag][k] for k in EVIDENCE]
            rows.append(dict(seed=s, model=tag, hits=counts[tag], n=n, params=params, mac=mac,
                             top1_pct=100*counts[tag]/n, loss_pp=100*(counts['teacher']-counts[tag])/n,
                             gain_vs_uniform_pp=100*gain/n if tag == 'L06' else None))

    def describe(values):
        return dict(mean=st.mean(values), sd=st.stdev(values), minimum=min(values), maximum=max(values))

    points = {}
    for tag in TAGS:
        sub = [r for r in rows if r['model'] == tag]
        points[tag] = {k: sub[0][k] for k in ('params', 'mac')}
        assert all(all(r[k] == v for k, v in points[tag].items()) for r in sub)
        points[tag].update({k: describe([r[k] for r in sub]) for k in ('top1_pct', 'loss_pp')})
    margins = {k: points['b10'][k]-points['L06'][k] for k in ('params', 'mac')}
    assert all(0 <= 1000*d <= points['b10'][k] for k, d in margins.items())
    correct = np.stack(all_correct)
    delta = (correct[:, 2].astype(np.int8) - correct[:, 1]).mean(axis=0)
    assert int(4*delta.sum()) == sum(gains)
    assert (inputs['draws'], inputs['bootstrap_seed']) == (DRAW_COUNT, BOOT_SEED)
    rng, draws = np.random.default_rng(BOOT_SEED), np.zeros(DRAW_COUNT)
    for label in range(200):
        within = delta[labels_ref == label]
        indices = rng.integers(0, len(within), size=(DRAW_COUNT, len(within)))
        draws += within[indices].sum(axis=1)
    ci = (np.quantile(draws, [.025, .975]) * (100/n)).tolist()
    positives = sum(g > 0 for g in gains)
    direction = sum(gains) > 0 and positives >= 3
    contrast = dict(**describe([100*g/n for g in gains]), per_seed=[100*g/n for g in gains],
                    ci95_pp=ci, positive=positives, direction_replicated=direction,
                    primary_pass=direction and ci[0] > 0)
    summary = dict(dataset='ImageNetV2 MatchedFrequency b-33 / Tiny 200 klas', samples=n,
                   seeds=list(SEEDS), models=points, loss_uniform_minus_l06_pp=contrast,
                   margins_under_uniform=margins)
    out = run / 'analysis'
    if out.is_dir():
        assert read(out / 'summary.json') == summary and read(out / 'audit.json')['status'] == 'PASS'
        assert digest(out / 'per_seed.csv') == read(out / 'audit.json')['output_sha256']['per_seed.csv']
    else:
        out.mkdir()
        write(out / 'summary.json', summary)
        with (out / 'per_seed.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write(out / 'audit.json', dict(status='PASS', exact_E046_models=12, assessments=len(rows), four_gpu_exit0=True,
                                       fold_plan_sha256=e46.FROZEN_SHA, test_sha256=inputs['test_sha256'],
                                       output_sha256={p: digest(out / p) for p in ('summary.json', 'per_seed.csv')}))
    print(json.dumps(summary, indent=2), '\nAUDYT PASS', flush=True)


def main():
    assert not sys.flags.optimize and sys.dont_write_bytecode
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and '/envs/attention/' in sys.executable
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('run')
    sub.add_parser('analyze').add_argument('run', type=Path)
    w = sub.add_parser('worker')
    w.add_argument('run', type=Path)
    w.add_argument('gpu', type=int, choices=range(4))
    args = p.parse_args()
    if args.action == 'run':
        launch()
    elif args.action == 'worker':
        worker(args.run.resolve(), args.gpu)
    else:
        analyze(args.run.resolve())


if __name__ == '__main__':
    main()
