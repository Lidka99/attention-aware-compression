"""2026-11-22_E045_lagodny_uniform_tiny: budżety UNIFORM i ocena Tiny dev.
Wejścia: E019/E035/E044; wyjścia: zamrożone plany CPU, run 4 GPU, agregaty.
Zakres i algorytm w OPIS.md.
"""

import argparse
import copy
import csv
import json
import math
import os
import platform
import shutil
import statistics as st
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
E44 = HERE.parent / '2026-11-22_E044_l06_tiny'
sys.path.insert(0, str(E44))
import kontrola_cpu as cpu
import ocena as previous


RUNS = cpu.ROOT / 'runs/E045'
CURRENT = ('teacher', 'b10', 'b20')
HISTORY = ('b30', 'b50')
SOURCE = ('eksperyment.py', 'OPIS.md', 'URUCHOMIENIE.md')
LIMIT = 1800
N = 10000
read, utc, cost = previous.read, previous.utc, previous.count_cost


def new_run(prefix=''):
    ident = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    run = RUNS / (prefix + ident)
    (run / 'source').mkdir(parents=True)
    for n in SOURCE:
        shutil.copyfile(HERE / n, run / 'source' / n)
    return run


def snapshot(run):
    return {n: cpu.digest(run / 'source' / n) for n in SOURCE}


def count(top1):
    rounded = round(N * top1)
    assert 0 <= rounded <= N and abs(N * top1 - rounded) < 1e-8
    return rounded


def uniform(geom, p0, percent):
    ns = list(geom.sizes)
    cap = (100 - percent) * p0 // 100
    denom = math.lcm(*ns)
    widths = lambda i: [i * n // denom for n in ns]
    tick = next(i for i in range(denom - 1, 0, -1)
                if min(widths(i)) >= 1 and geom.count(widths(i)) <= cap)
    w = widths(tick)
    ratio = 1 - (tick + 0.5) / denom
    nxt = geom.count(widths(tick + 1))
    assert [int(n * (1 - ratio)) for n in ns] == w
    assert geom.count(w) <= cap < nxt
    return dict(U=w, ratio=ratio, params={'U': geom.count(w)}, nominal_reduction_pct=percent,
                max_params=cap, next_uniform_params=nxt)


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    run = new_run('cpu_')
    start = time.monotonic()
    m = dict(status='running', phase='cpu', run=cpu.rel(run), command=[sys.executable, '-B', *sys.argv],
             source_sha256=snapshot(run), quality_evaluated=False, started_utc=utc())
    cpu.write(run / 'manifest.json', m)
    print('CPU RUN', cpu.rel(run), flush=True)
    # Jawna blokada wszystkich plików obrazowych; wolno czytać przypięte metadane i wagi.
    sys.addaudithook(cpu.guard)
    try:
        e44 = previous.verify_inputs(check_cache=False)
        f35 = read(cpu.R35 / 'frozen.json')
        cal = e44['data']['calibration']
        idx = f35['indices']['tiny']
        assert idx['A'] == cal['A'] and idx['source'] == cal['cache'] and idx['source_sha256'] == cal['sha256']
        train, dev = [read(cpu.SPLIT)['splits'][k] for k in ('train', 'dev')]
        assert len(set(dev)) == len(dev) == N and dev == idx['dev']
        assert set(cal['A']) <= set(train) and not set(cal['A']) & set(dev)

        e35 = cpu.load(previous.E35, '_e045_e035_cpu')
        torch = e35.torch
        assert not torch.cuda.is_available()
        torch.set_num_threads(2)
        base = e35.model_source(cpu.PAIR, 42)
        geom = e35.bn.Geometry(base)
        teacher = cost(torch, base)
        assert teacher == e44['teacher'] and list(geom.sizes) == e44['sizes']
        assert geom.count(list(geom.sizes)) == teacher['params']
        base_state = e35.state_hash(base)
        plans, costs = {}, {'teacher': teacher}
        for tag, percent in (('b10', 10), ('b20', 20)):
            plans[tag] = uniform(geom, teacher['params'], percent)
            model, diag = e35.fold(cpu.PAIR, base, plans[tag], 'U')
            costs[tag] = cost(torch, model)
            assert [d['k'] for d in diag] == plans[tag]['U'] and costs[tag]['params'] == plans[tag]['params']['U']
            assert torch.equal(model.fc.bias, base.fc.bias)
            print(tag, plans[tag]['U'], costs[tag], flush=True)
        assert e35.state_hash(base) == base_state and not cpu.DENIED

        inputs = {**e44['input_sha256'], cpu.rel(previous.CPU_RUN / 'frozen.json'): previous.FROZEN_SHA,
                  cpu.rel(E44 / 'ocena.py'): cpu.digest(E44 / 'ocena.py'),
                  cpu.rel(E44 / 'kontrola_cpu.py'): cpu.digest(E44 / 'kontrola_cpu.py')}
        expected = {}
        for tag in HISTORY:
            costs[tag] = {k: e44['budgets'][tag]['profiles']['U'][k] for k in ('params', 'mac')}
        for s in cpu.SEEDS:
            old = cpu.R35 / f'dev/seed_{s}/{cpu.PAIR}'
            p = old / 'controls.json'
            ctrl = read(p)
            assert cpu.digest(p) == inputs[cpu.rel(p)] and ctrl['views_sha256'] == cal['views_sha256']
            assert ctrl['evidence']['CMA']['samples'] == N
            expected[str(s)] = {'teacher': dict(correct=count(ctrl['metrics']['CMA']['top1']),
                                                 state_sha256=ctrl['evidence']['CMA']['state_sha256'])}
            for tag in HISTORY:
                p = old / tag / 'summary.json'
                result = read(p)
                assert result['phase'] == 'dev' and result['status'] == 'completed' and result['seed'] == s
                assert result['plan'] == e44['budgets'][tag]['uniform_plan']
                assert result['views_sha256'] == cal['views_sha256'] and result['evidence']['U']['CMA']['samples'] == N
                assert result['checkpoint_sha256'] == e44['data']['checkpoints'][str(s)]['sha256']
                inputs[cpu.rel(p)] = cpu.digest(p)
                expected[str(s)][tag] = dict(correct=count(result['metrics']['U']['CMA']['top1']), source=cpu.rel(p))

        frozen = dict(plans=plans, costs=costs, references=expected, calibration=cal, dev=dev,
                      checkpoints=e44['data']['checkpoints'], protocol_sha256=m['source_sha256']['OPIS.md'],
                      input_sha256=inputs, settings=dict(seeds=list(cpu.SEEDS), batch_per_gpu=128, mode='CMA_reset'))
        cpu.write(run / 'frozen.json', frozen)
        m.update(status='passed', frozen_sha256=cpu.digest(run / 'frozen.json'), torch=str(torch.__version__))
        print('CPU PASS', cpu.rel(run), m['frozen_sha256'], flush=True)
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        m.update(seconds=time.monotonic()-start, finished_utc=utc(), data_access_denied=cpu.DENIED,
                 data_and_weights_opened=sorted(map(cpu.rel, cpu.OPENED)))
        cpu.write(run / 'manifest.json', m)


def worker(run, gpu):
    m = read(run / 'manifest.json')
    assert run.parent == RUNS and m['status'] == 'running' and os.getppid() == m['pid']
    assert cpu.digest(run / 'frozen.json') == m['frozen_sha256']
    f = read(run / 'frozen.json')
    seed = f['settings']['seeds'][gpu]
    out = run / f'seed_{seed}'
    out.mkdir()
    stt = dict(status='running', pid=os.getpid(), seed=seed, gpu=gpu, started_utc=utc())
    opened, denied = set(), []
    allow = {cpu.TRAIN, cpu.CKPT[seed]}

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(cpu.ROOT / 'data') or (p.is_relative_to(cpu.ROOT) and p.suffix in ('.pt', '.pth', '.ckpt', '.safetensors')):
            if p not in allow or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
                denied.append(cpu.rel(p))
                raise RuntimeError(f'E045: zabroniony odczyt/zapis {p}')
            opened.add(p)

    sys.addaudithook(guard)
    cpu.write(out / 'status.json', stt)
    try:
        e35 = cpu.load(previous.E35, '_e045_e035_worker')
        np, torch, bn, e30 = e35.np, e35.torch, e35.bn, e35.e30
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu) and torch.cuda.device_count() == 1
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        with np.load(cpu.TRAIN, allow_pickle=False) as cache:
            raw, labels = cache['images'], cache['labels']
        assert raw.shape == (100000, 64, 64, 3) and raw.dtype == np.uint8
        aug = e30.augment(raw, f['calibration']['A'])
        view = cpu.sha(aug.contiguous().numpy().tobytes())
        assert view == f['calibration']['views_sha256']
        a = e30.normalize(aug)
        x = e30.normalize(torch.from_numpy(raw[f['dev']]).permute(0, 3, 1, 2).float()/255)
        y = torch.tensor(labels[f['dev']], dtype=torch.long)
        assert torch.equal(torch.bincount(y), torch.full((200,), 50))
        del raw, labels, aug
        source = e35.model_source(cpu.PAIR, seed)
        state_before = e35.state_hash(source)

        def prepare_model(tag):
            if tag == 'teacher':
                model = copy.deepcopy(source)
            else:
                model, diag = e35.fold(cpu.PAIR, source, f['plans'][tag], 'U')
                assert [d['k'] for d in diag] == f['plans'][tag]['U'] and all(set(d['labels']) == set(range(d['k'])) for d in diag)
                assert torch.equal(model.fc.bias, source.fc.bias)
            model.to('cuda:0')
            actual = cost(torch, model)
            assert actual == f['costs'][tag]
            bnmeta = e35.e29.calibrate(model, a, 'CMA')
            assert (bnmeta['samples'], bnmeta['batches'], bnmeta['last_batch']) == (4096, 32, 128)
            assert bnmeta['estimator'] == 'CMA_reset' and bnmeta['params_unchanged']
            assert bnmeta['layers'] and all((b['counter_start'], b['counter_end']) == (0, 32) for b in bnmeta['layers'])
            return model, dict(**actual, state_sha256=e35.state_hash(model), cma_batches=32)

        def score(model, evidence):
            logits = bn.predict(model, x)
            assert logits.shape == (N, 200) and torch.isfinite(logits).all()
            assert e35.state_hash(model) == evidence['state_sha256']
            assert all(not t.requires_grad and t.grad is None for t in model.parameters())
            return dict(**evidence, correct=int((logits.argmax(1) == y).sum()), samples=N)

        teacher, proof = prepare_model('teacher')
        ref = f['references'][str(seed)]['teacher']
        assert proof['state_sha256'] == ref['state_sha256']
        results = {'teacher': score(teacher, proof)}
        assert results['teacher']['correct'] == ref['correct']
        ready = dict(seed=seed, gpu=gpu, pid=os.getpid(), views_sha256=view, teacher=results['teacher'])
        cpu.write(out / 'ready.json', ready)
        gate = run / 'gate.json'
        t0 = time.monotonic()
        while not gate.is_file():
            assert time.monotonic() - t0 < LIMIT and os.getppid() == m['pid']
            time.sleep(.5)
        assert read(gate)['ready_sha256'][str(seed)] == cpu.digest(out / 'ready.json')
        for tag in CURRENT[1:]:
            model, proof = prepare_model(tag)
            results[tag] = score(model, proof)
            del model
        assert opened == allow and not denied and e35.state_hash(source) == state_before
        cpu.write(out / 'metrics.json', dict(seed=seed, models=results, views_sha256=view))
        stt.update(status='completed', assessments=len(results))
        print(f'seed {seed}, GPU {gpu}: {len(results)}/3 zakończone', flush=True)
    except BaseException:
        stt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        stt.update(finished_utc=utc(), files_opened=sorted(map(cpu.rel, opened)), denied=denied)
        cpu.write(out / 'status.json', stt)


def launch(cpu_run):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert cpu_run.parent == RUNS and cpu_run.name.startswith('cpu_')
    m0 = read(cpu_run / 'manifest.json')
    p = cpu_run / 'frozen.json'
    sha = cpu.digest(p)
    assert m0['status'] == 'passed' and not m0['quality_evaluated'] and sha == m0['frozen_sha256']
    f = read(p)
    assert m0['source_sha256'] == {n: cpu.digest(HERE / n) for n in SOURCE}
    assert cpu.digest(HERE / 'OPIS.md') == f['protocol_sha256']
    previous.verify_inputs(check_cache=False)
    for path, h in f['input_sha256'].items():
        assert cpu.digest(cpu.ROOT / path) == h, path
    expected = f['calibration']['sha256']
    assert cpu.digest(cpu.TRAIN) == expected
    import torch
    assert torch.cuda.device_count() == 4
    pre = previous.gpu_info()
    assert not pre['compute_apps'] and [g['index'] for g in pre['gpus']] == list('0123')
    assert all(int(g['memory_mib']) < 256 and int(g['utilization']) == 0 for g in pre['gpus'])

    run = new_run()
    shutil.copyfile(p, run / 'frozen.json')
    m = dict(status='running', phase='tiny_dev', run=cpu.rel(run), pid=os.getpid(), started_utc=utc(),
             source_sha256=snapshot(run), command=[sys.executable, '-B', *sys.argv], frozen_sha256=sha,
             cpu_run=cpu.rel(cpu_run), training=False, official_val_used=False, gpu_start=pre,
             python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version(), train_cache_sha256=expected, settings=f['settings'], workers=[],
             environment={k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                          'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')})
    cpu.write(run / 'manifest.json', m)
    print('RUN', cpu.rel(run), flush=True)
    procs, logs, start = [], [], time.monotonic()
    try:
        for gpu, seed in enumerate(cpu.SEEDS):
            logs.append((run / f'worker_{gpu}.log').open('w'))
            cmd = [sys.executable, '-B', sys.argv[0], 'worker', str(run), str(gpu)]
            proc = subprocess.Popen(cmd, cwd=cpu.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=logs[-1], stderr=subprocess.STDOUT)
            procs.append(proc)
            m['workers'].append(dict(gpu=gpu, seed=seed, pid=proc.pid, uuid=pre['gpus'][gpu]['uuid'], command=cmd))
        ready = {s: run / f'seed_{s}/ready.json' for s in cpu.SEEDS}
        devices = {str(w['pid']): w['uuid'] for w in m['workers']}
        gate = run / 'gate.json'
        cpu.write(run / 'manifest.json', m)
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in procs]
                monitor.write(json.dumps(dict(**previous.gpu_info(), exit_codes=codes)) + '\n')
                monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0]*4:
                    break
                assert time.monotonic() - start < LIMIT, 'Limit 30 min'
                if not gate.exists() and all(p.is_file() for p in ready.values()):
                    assert previous.gpu_info()['compute_apps'] == devices
                    for w in m['workers']:
                        s = w['seed']
                        r = read(ready[s])
                        ref = f['references'][str(s)]['teacher']
                        assert (r['seed'], r['gpu'], r['pid']) == (s, w['gpu'], w['pid'])
                        assert all(r['teacher'][k] == v for k, v in ref.items())
                    cpu.write(gate, dict(utc=utc(), pid_gpu_uuid=devices,
                                         ready_sha256={str(s): cpu.digest(p) for s, p in ready.items()}))
                    print('Bramka 4 GPU + replay E035 dev PASS', flush=True)
                time.sleep(2)
        assert gate.is_file() and cpu.digest(cpu.TRAIN) == expected
        assert m['source_sha256'] == {n: cpu.digest(HERE / n) for n in SOURCE}
        files = ['gate.json']
        for s in cpu.SEEDS:
            folder = run / f'seed_{s}'
            status = read(folder / 'status.json')
            assert status['status'] == 'completed' and status['assessments'] == 3
            files.extend([f'seed_{s}/{n}' for n in ('ready.json', 'status.json', 'metrics.json')])
        m.update(status='completed', assessments=12, output_sha256={n: cpu.digest(run / n) for n in files})
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()
                pr.wait()
        raise
    finally:
        for w, pr in zip(m['workers'], procs):
            w['exit_code'] = pr.poll()
        for log in logs:
            log.close()
        m.update(seconds=time.monotonic()-start, finished_utc=utc())
        try:
            m['gpu_end'] = previous.gpu_info()
        finally:
            cpu.write(run / 'manifest.json', m)
    print('COMPLETED', cpu.rel(run), flush=True)


def analyze(run):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and run.parent == RUNS
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed' and m['phase'] == 'tiny_dev' and m['assessments'] == 12
    assert not m['official_val_used'] and not m['training'] and not m['gpu_end']['compute_apps']
    assert m['source_sha256'] == snapshot(run)
    assert m['frozen_sha256'] == cpu.digest(run / 'frozen.json')
    f = read(run / 'frozen.json')
    for n, h in m['output_sha256'].items():
        assert cpu.digest(run / n) == h, n
    gate = read(run / 'gate.json')
    assert gate['pid_gpu_uuid'] == {str(w['pid']): w['uuid'] for w in m['workers']}
    assert [(w['seed'], w['gpu'], w['exit_code']) for w in m['workers']] == [(s, i, 0) for i, s in enumerate(cpu.SEEDS)]
    rows = []
    for s in cpu.SEEDS:
        folder = run / f'seed_{s}'
        stt, metrics = [read(folder / n) for n in ('status.json', 'metrics.json')]
        assert stt['status'] == 'completed' and not stt['denied']
        assert set(stt['files_opened']) == {cpu.rel(cpu.TRAIN), f['checkpoints'][str(s)]['path']}
        assert gate['ready_sha256'][str(s)] == cpu.digest(folder / 'ready.json')
        assert metrics['seed'] == s and set(metrics['models']) == set(CURRENT)
        assert metrics['views_sha256'] == f['calibration']['views_sha256']
        ref = f['references'][str(s)]
        base = metrics['models']['teacher']['correct']
        assert base == ref['teacher']['correct'] and metrics['models']['teacher']['state_sha256'] == ref['teacher']['state_sha256']
        for tag in CURRENT + HISTORY:
            c = f['costs'][tag]
            if tag in CURRENT:
                result = metrics['models'][tag]
                assert result['samples'] == N and result['cma_batches'] == 32
                assert {k: result[k] for k in c} == c
                hits = result['correct']
            else:
                p = ref[tag]['source']
                assert cpu.digest(cpu.ROOT / p) == f['input_sha256'][p]
                hits = ref[tag]['correct']
            assert type(hits) is int and 0 <= hits <= N
            rows.append(dict(seed=s, budget=tag, source='E045' if tag in CURRENT else 'E035', **c,
                             correct=hits, samples=N, top1_pct=100 * hits / N, loss_pp=100 * (base - hits) / N,
                             p_reduction_pct=100*(1-c['params']/f['costs']['teacher']['params']),
                             mac_reduction_pct=100*(1-c['mac']/f['costs']['teacher']['mac'])))
    points = []
    for tag in CURRENT+HISTORY:
        r = [v for v in rows if v['budget'] == tag]
        assert len(r) == 4
        points.append(dict(**{k: r[0][k] for k in ('budget', 'source', 'params', 'mac', 'p_reduction_pct', 'mac_reduction_pct')},
                           mean_top1_pct=st.mean(v['top1_pct'] for v in r), sd_top1_pct=st.stdev(v['top1_pct'] for v in r),
                           mean_loss_pp=st.mean(v['loss_pp'] for v in r), sd_loss_pp=st.stdev(v['loss_pp'] for v in r),
                           loss_per_seed_pp=[v['loss_pp'] for v in r]))
    out = run / 'analysis'
    out.mkdir()
    cpu.write(out / 'summary.json', dict(dataset='Tiny ImageNet dev', seeds=list(cpu.SEEDS), new_assessments=12,
                                         reused_assessments=8, points=points))
    with (out / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cpu.write(out / 'audit.json', dict(status='PASS', rows=len(rows), phase='dev', four_gpu_exit0=True,
                                       teacher_exact_E035=4, val_used=False, frozen_sha256=m['frozen_sha256']))
    print(json.dumps(points, indent=2), '\nAUDYT PASS', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    for action in ('run', 'worker', 'analyze'):
        arg = sub.add_parser(action)
        arg.add_argument('path', type=Path)
        if action == 'worker':
            arg.add_argument('gpu', type=int, choices=range(4))
    args = p.parse_args()
    assert '/envs/attention/' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert not sys.flags.optimize and sys.dont_write_bytecode
    if args.action == 'prepare':
        prepare()
    elif args.action == 'run':
        launch(args.path.resolve())
    elif args.action == 'worker':
        worker(args.path.resolve(), args.gpu)
    else:
        analyze(args.path.resolve())


if __name__ == '__main__':
    main()
