"""2026-11-22_E046_l06_b10_tiny: pomiar zamrożonego L06 na Tiny dev, zob. OPIS.md.
Wejścia: bramka CPU E046, referencje E045, E019 i cache train Tiny.
Wyjścia: 8 odtworzeń kontroli i 4 oceny L06 na czterech GPU, audyt i kontrast.
"""

import argparse
import copy
import csv
import json
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

import sprawdz_geometrie as cpu

HERE, ROOT, SEEDS, PAIR = cpu.HERE, cpu.ROOT, cpu.SEEDS, cpu.PAIR
RUNS = ROOT / 'runs/E046'
CPU_RUN = RUNS / 'cpu_20261122T042524Z_aa692377'
FROZEN_SHA = 'e7b4ffb9b839bb69cf35d35487afd032e1e661ec5ba23f66163d25533a8c7d84'
CACHE = ROOT / 'data/tiny-imagenet/cache_train.npz'
E44 = HERE.parent / '2026-11-22_E044_l06_tiny'
SOURCES = ('ocena_dev.py', 'sprawdz_geometrie.py', 'OPIS.md', 'URUCHOMIENIE_GPU.md')
CONTROL, TAGS = ('teacher', 'b10'), ('teacher', 'b10', 'L06')
N, LIMIT = 10000, 1800
digest, write, rel = cpu.digest, cpu.write, cpu.rel


def read(path):
    return json.loads(path.read_bytes())


def utc():
    return datetime.now(timezone.utc).isoformat()


def helpers():
    # Te same liczniki Conv/Linear i telemetria co w E045.
    sys.path.insert(0, str(E44))
    from ocena import count_cost, gpu_info
    return count_cost, gpu_info


def guard_files(allow):
    opened, denied = set(), []

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(ROOT / 'data') or p.suffix in ('.pt', '.pth', '.ckpt', '.safetensors'):
            if p not in allow or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
                denied.append(str(p))
                raise RuntimeError(f'E046 dev: niedozwolony dostęp do {p}')
            opened.add(p)

    sys.addaudithook(guard)
    return opened, denied


def load_plan():
    m = read(CPU_RUN / 'manifest.json')
    assert m['status'] == 'passed' and m['gate_passed'] and not m['quality_evaluated']
    assert len(m['model_checks']) == 12
    path = CPU_RUN / 'frozen.json'
    assert digest(path) == m['frozen_sha256'] == FROZEN_SHA
    f = read(path)
    assert f['source_sha256'] == m['source_sha256']
    assert all(digest(HERE / name) == h for name, h in f['source_sha256'].items())
    assert f['pair'] == PAIR and f['budget'] == 'b10' and f['profile'] == 'L06'
    assert f['geometry']['passed'] and rel(CACHE) == f['calibration']['cache']
    assert f['measurement'] == dict(seeds=list(SEEDS), batch_per_gpu=128, mode='CMA_reset',
                                    gpu_to_seed={str(i): s for i, s in enumerate(SEEDS)},
                                    new_candidate_scores=4, control_replays=8)
    for name, h in f['input_sha256'].items():
        assert digest(ROOT / name) == h, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cpu.UP, text=True).strip() == f['upstream_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=cpu.UP, text=True).strip()
    f45 = read(cpu.R45 / 'frozen.json')
    assert f['reference_run'] == rel(cpu.R45) and f['reference_frozen_sha256'] == cpu.F45_SHA
    assert f['calibration'] == f45['calibration'] and f['dev'] == f45['dev']
    assert f['checkpoints'] == f45['checkpoints'] and f['geometry']['uniform'] == f45['plans']['b10']['U']
    for helper in ('kontrola_cpu.py', 'ocena.py'):
        p = E44 / helper
        assert digest(p) == f45['input_sha256'][rel(p)]
    refs = {}
    for seed in SEEDS:
        name = f'seed_{seed}/metrics.json'
        path = cpu.R45 / name
        assert digest(path) == f['reference_metrics_sha256'][name]
        old = read(path)
        assert old['seed'] == seed and old['views_sha256'] == f['calibration']['views_sha256']
        refs[str(seed)] = {tag: old['models'][tag] for tag in CONTROL}
    return f, dict(models=refs, uniform_plan=f45['plans']['b10'])


def phase_inputs(run, m):
    assert m['frozen_sha256'] == FROZEN_SHA == digest(run / 'frozen.json')
    assert m['references_sha256'] == digest(run / 'references.json')
    return read(run / 'frozen.json'), read(run / 'references.json')


def worker(run, gpu):
    m = read(run / 'manifest.json')
    assert run.parent == RUNS and m['status'] == 'running' and m['pid'] == os.getppid()
    f, refs = phase_inputs(run, m)
    seed = f['measurement']['gpu_to_seed'][str(gpu)]
    allowed = {CACHE, cpu.CKPT[seed]}
    opened, denied = guard_files(allowed)
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', seed=seed, gpu=gpu, pid=os.getpid(), started_utc=utc())
    write(out / 'status.json', status)
    try:
        rt = cpu.load(cpu.E37, '_e046_worker_e037')
        e35, torch, np = rt.e35, rt.torch, rt.np
        e30, bn = e35.e30, e35.bn
        count_cost, _ = helpers()
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu) and torch.cuda.device_count() == 1
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(seed)
        np.random.seed(seed)

        cal = f['calibration']
        with np.load(CACHE, allow_pickle=False) as cache:
            raw, labels = cache['images'], cache['labels']
        assert raw.shape == (100000, 64, 64, 3) and raw.dtype == np.uint8
        aug = e30.augment(raw, cal['A'])
        view = cpu.hashlib.sha256(aug.contiguous().numpy().tobytes()).hexdigest()
        assert view == cal['views_sha256']
        a = e30.normalize(aug)
        x = e30.normalize(torch.from_numpy(raw[f['dev']]).permute(0, 3, 1, 2).float()/255)
        y = torch.tensor(labels[f['dev']], dtype=torch.long)
        assert torch.equal(torch.bincount(y), torch.full((200,), 50))
        del raw, labels, aug
        source = e35.model_source(PAIR, seed)
        before = e35.state_hash(source)
        geometry = f['geometry']

        def evaluate(tag):
            if tag == 'teacher':
                model, expected = copy.deepcopy(source), geometry['teacher']
            elif tag == 'b10':
                model, diag = e35.fold(PAIR, source, refs['uniform_plan'], 'U')
                expected = geometry['caps']
            else:
                expected = geometry['candidate']
                model, diag = rt.fold_widths(PAIR, source, expected['widths'])
            if tag != 'teacher':
                widths = geometry['uniform'] if tag == 'b10' else expected['widths']
                assert [d['k'] for d in diag] == widths and [d['n'] for d in diag] == geometry['sizes']
                assert all(set(d['labels']) == set(range(d['k'])) for d in diag)
                assert torch.equal(model.fc.bias, source.fc.bias)
            model.to('cuda:0')
            cost = count_cost(torch, model)
            assert cost == {k: expected[k] for k in cost}
            meta = e35.e29.calibrate(model, a, 'CMA')
            assert (meta['samples'], meta['batches'], meta['last_batch']) == (4096, 32, 128)
            assert meta['estimator'] == 'CMA_reset' and meta['params_unchanged'] and meta['layers']
            assert all((b['counter_start'], b['counter_end']) == (0, 32) for b in meta['layers'])
            state = e35.state_hash(model)
            logits = bn.predict(model, x)
            assert logits.shape == (N, 200) and torch.isfinite(logits).all()
            assert e35.state_hash(model) == state and all(not p.requires_grad and p.grad is None for p in model.parameters())
            return dict(**cost, state_sha256=state, cma_batches=32, correct=int((logits.argmax(1) == y).sum()), samples=N)

        results = {tag: evaluate(tag) for tag in CONTROL}
        assert results == refs['models'][str(seed)], (seed, 'Niezgodność odtworzenia E045', results)
        ready = out / 'ready.json'
        write(ready, dict(pid=os.getpid(), seed=seed, gpu=gpu, models=results, views_sha256=view))
        gate = run / 'gate.json'
        started = time.monotonic()
        while not gate.is_file():
            assert time.monotonic() - started < LIMIT and os.getppid() == m['pid']
            time.sleep(.5)
        assert read(gate)['ready_sha256'][str(seed)] == digest(ready)
        results['L06'] = evaluate('L06')
        assert e35.state_hash(source) == before and opened == allowed and not denied
        write(out / 'metrics.json', dict(seed=seed, views_sha256=view, models=results))
        status.update(status='completed', assessments=len(results))
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=utc(), opened=sorted(map(rel, opened)), denied=denied)
        write(out / 'status.json', status)


def launch():
    env = {k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                                         'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    assert list(env.values()) == ['attention', '0,1,2,3', 'PCI_BUS_ID', ':4096:8', '2', '2', '2']
    allowed = {CACHE, cpu.SPLIT, *cpu.CKPT.values()}
    opened, denied = guard_files(allowed)
    f, refs = load_plan()
    cache_sha = f['calibration']['sha256']
    assert digest(CACHE) == cache_sha
    import torch
    _, gpu_info = helpers()
    pre = gpu_info()
    assert not pre['compute_apps'] and [g['index'] for g in pre['gpus']] == list('0123')
    assert torch.cuda.device_count() == 4
    assert all(int(g['memory_mib']) < 256 and int(g['utilization']) == 0 for g in pre['gpus'])

    run = RUNS / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    (run / 'source').mkdir(parents=True)
    for name in SOURCES:
        shutil.copyfile(HERE / name, run / 'source' / name)
    source_hashes = {n: digest(run / 'source' / n) for n in SOURCES}
    shutil.copyfile(CPU_RUN / 'frozen.json', run / 'frozen.json')
    write(run / 'references.json', refs)
    m = dict(run=rel(run), phase='tiny_dev', pid=os.getpid(), status='running', started_utc=utc(),
             command=[sys.executable, '-B', *sys.argv], cpu_run=rel(CPU_RUN), frozen_sha256=FROZEN_SHA,
             source_sha256=source_hashes, references_sha256=digest(run / 'references.json'), train_cache_sha256=cache_sha,
             python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version(), environment=env, gpu_start=pre, workers=[],
             training=False, official_val_used=False, cifar100_reserve_used=False)
    write(run / 'manifest.json', m)
    print('RUN', rel(run), flush=True)
    procs, logs, started = [], [], time.monotonic()
    try:
        for gpu, seed in enumerate(SEEDS):
            logs.append((run / f'worker_{gpu}.log').open('w'))
            cmd = [sys.executable, '-B', sys.argv[0], 'worker', str(run), str(gpu)]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=logs[-1], stderr=subprocess.STDOUT)
            procs.append(proc)
            m['workers'].append(dict(gpu=gpu, seed=seed, pid=proc.pid, uuid=pre['gpus'][gpu]['uuid'], command=cmd))
        write(run / 'manifest.json', m)
        ready = {s: run / f'seed_{s}/ready.json' for s in SEEDS}
        devices = {str(w['pid']): w['uuid'] for w in m['workers']}
        gate = run / 'gate.json'
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes, telemetry = [p.poll() for p in procs], gpu_info()
                monitor.write(json.dumps(dict(**telemetry, exit_codes=codes)) + '\n')
                monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0] * 4:
                    break
                assert time.monotonic() - started < LIMIT, 'Limit 30 min'
                if not gate.exists() and all(p.is_file() for p in ready.values()):
                    assert telemetry['compute_apps'] == devices
                    for w in m['workers']:
                        s, r = w['seed'], read(ready[w['seed']])
                        assert (r['seed'], r['gpu'], r['pid']) == (s, w['gpu'], w['pid'])
                        assert r['models'] == refs['models'][str(s)]
                        assert r['views_sha256'] == f['calibration']['views_sha256']
                    write(gate, dict(utc=utc(), pid_gpu_uuid=devices,
                                     ready_sha256={str(s): digest(p) for s, p in ready.items()}))
                    print('PASS: odtworzenia E045 8/8, aktywne 4 GPU; start punktacji L06', flush=True)
                time.sleep(2)
        assert gate.is_file() and digest(CACHE) == cache_sha and opened == allowed and not denied
        assert source_hashes == {n: digest(HERE / n) for n in SOURCES}
        files = ['gate.json', 'monitor.jsonl', *(f'worker_{i}.log' for i in range(4))]
        for s in SEEDS:
            p = run / f'seed_{s}'
            status = read(p / 'status.json')
            assert status['status'] == 'completed' and status['assessments'] == 3
            files.extend(f'seed_{s}/{n}' for n in ('ready.json', 'metrics.json', 'status.json'))
        m.update(status='completed', assessments=12, candidate_scores=4, control_replays=8,
                 output_sha256={n: digest(run / n) for n in files})
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        for w, proc in zip(m['workers'], procs):
            w['exit_code'] = proc.poll()
        for log in logs:
            log.close()
        m.update(seconds=time.monotonic() - started, finished_utc=utc(),
                 opened=sorted(map(rel, opened)), denied=denied)
        try:
            m['gpu_end'] = gpu_info()
        finally:
            write(run / 'manifest.json', m)
    print('COMPLETED', rel(run), flush=True)


def analyze(run):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and run.parent == RUNS
    m = read(run / 'manifest.json')
    assert m['phase'] == 'tiny_dev' and m['status'] == 'completed' and m['assessments'] == 12
    assert m['candidate_scores'] == 4 and m['control_replays'] == 8 and not m['denied']
    assert not any(m[k] for k in ('training', 'official_val_used', 'cifar100_reserve_used'))
    assert not m['gpu_end']['compute_apps']
    f, refs = phase_inputs(run, m)
    for name, h in m['source_sha256'].items():
        assert digest(run / 'source' / name) == h
    for name, h in m['output_sha256'].items():
        assert digest(run / name) == h
    assert [(w['gpu'], w['seed'], w['exit_code']) for w in m['workers']] == [(i, s, 0) for i, s in enumerate(SEEDS)]
    gate = read(run / 'gate.json')
    devices = {str(w['pid']): w['uuid'] for w in m['workers']}
    assert gate['pid_gpu_uuid'] == devices and len(set(devices.values())) == 4
    assert any(json.loads(line)['compute_apps'] == devices for line in (run / 'monitor.jsonl').read_text().splitlines())
    costs = dict(zip(TAGS, (f['geometry'][k] for k in ('teacher', 'caps', 'candidate'))))
    rows, gains = [], []
    for s in SEEDS:
        p = run / f'seed_{s}'
        status, metrics = read(p / 'status.json'), read(p / 'metrics.json')
        assert status['status'] == 'completed' and not status['denied']
        assert set(status['opened']) == {rel(CACHE), f['checkpoints'][str(s)]['path']}
        assert gate['ready_sha256'][str(s)] == digest(p / 'ready.json')
        assert metrics['seed'] == s and metrics['views_sha256'] == f['calibration']['views_sha256']
        result = metrics['models']
        assert set(result) == set(TAGS) and {k: result[k] for k in CONTROL} == refs['models'][str(s)]
        base = result['teacher']['correct']
        gain = result['L06']['correct'] - result['b10']['correct']
        gains.append(gain)
        for tag in TAGS:
            r = result[tag]
            hits = r['correct']
            assert type(hits) is int and 0 <= hits <= N and (r['samples'], r['cma_batches']) == (N, 32)
            params, mac = (r[k] for k in ('params', 'mac'))
            assert (params, mac) == (costs[tag]['params'], costs[tag]['mac'])
            rows.append(dict(seed=s, profile=tag, correct=hits, samples=N,
                             top1_pct=100*hits/N, loss_pp=100*(base-hits)/N, params=params, mac=mac,
                             p_reduction_pct=100*(1 - params/costs['teacher']['params']),
                             mac_reduction_pct=100*(1 - mac/costs['teacher']['mac']),
                             gain_vs_uniform_pp=100*gain/N if tag == 'L06' else None))

    def describe(values):
        return dict(mean=st.mean(values), sd=st.stdev(values), minimum=min(values), maximum=max(values))

    points = {}
    for tag in TAGS:
        subset = [r for r in rows if r['profile'] == tag]
        points[tag] = {k: subset[0][k] for k in ('params', 'mac', 'p_reduction_pct', 'mac_reduction_pct')}
        points[tag].update({k: describe([r[k] for r in subset]) for k in ('top1_pct', 'loss_pp')})
    positive = sum(g > 0 for g in gains)
    contrast = dict(**describe([100*g/N for g in gains]), positive=positive,
                    per_seed=[100*g/N for g in gains], primary_pass=sum(gains) > 0 and positive >= 3)
    margins = {k: costs['b10'][k] - costs['L06'][k] for k in ('params', 'mac')}
    assert all(0 <= 1000*d <= costs['b10'][k] for k, d in margins.items())
    summary = dict(dataset='Tiny ImageNet dev', seeds=list(SEEDS), models=points,
                   loss_uniform_minus_l06_pp=contrast, margins_under_uniform=margins)
    out = run / 'analysis'
    out.mkdir()
    write(out / 'summary.json', summary)
    with (out / 'per_seed.csv').open('w', newline='') as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write(out / 'audit.json', dict(status='PASS', assessments=len(rows), phase='dev', exact_E045_controls=8,
                                   four_gpu_exit0=True, new_L06_scores=4, frozen_sha256=FROZEN_SHA,
                                   analyzed_sha256={name: digest(out / name) for name in ('summary.json', 'per_seed.csv')}))
    print(json.dumps(summary, indent=2), '\nAUDYT PASS', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('run')
    sub.add_parser('analyze').add_argument('run', type=Path)
    w = sub.add_parser('worker')
    w.add_argument('run', type=Path)
    w.add_argument('gpu', type=int, choices=range(4))
    args = p.parse_args()
    assert not sys.flags.optimize and sys.dont_write_bytecode
    assert '/envs/attention/' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    if args.action == 'run':
        launch()
    elif args.action == 'worker':
        worker(args.run.resolve(), args.gpu)
    else:
        analyze(args.run.resolve())


if __name__ == '__main__':
    main()
