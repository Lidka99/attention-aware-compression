"""2026-11-22_E044_l06_tiny: 28 ocen na Tiny val, cztery GPU, bez treningu.
Wejścia: frozen z kontroli CPU, cztery wagi E019 i cache Tiny z E035.
Wyjścia: manifest, dowody geometrii/CMA i poprawność per obraz. Zob. OPIS.md.
"""

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import kontrola_cpu as cpu


CPU_RUN = cpu.ROOT / 'runs/E044/cpu_20261122T020819Z_00821a3f'
FROZEN_SHA = 'b145cf9d1c0e7e6785963dfe425a4c3df694db6ef8f4422d6ec1bdf70c191908'
RUNS = CPU_RUN.parent
RELEASE = RUNS / 'tiny_val_release.json'
E35 = cpu.HERE.parent / '2026-11-07_E035_ocena_niezalezna/experiment.py'
ROWS = ('teacher', *(f'{t}/{p}' for t in cpu.TAGS for p in cpu.PROFILES))
FILES = ('ocena.py', 'analiza.py', 'kontrola_cpu.py', 'OPIS.md', 'URUCHOMIENIE.md')
LIMIT = 1800


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def gpu_info():
    cmd = ['nvidia-smi', '--format=csv,noheader,nounits']
    raw = subprocess.check_output([*cmd, '--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu'], text=True)
    gpus = [dict(zip(('index', 'name', 'uuid', 'driver', 'memory_mib', 'utilization'),
                     [s.strip() for s in line.split(',')])) for line in raw.splitlines()]
    raw = subprocess.check_output([*cmd, '--query-compute-apps=pid,gpu_uuid'], text=True).strip()
    apps = {p.strip(): g.strip() for p, g in (line.split(',') for line in raw.splitlines())}
    return dict(utc=utc(), gpus=gpus, compute_apps=apps)


def verify_inputs(*, check_cache):
    report = read(CPU_RUN / 'manifest.json')
    p = CPU_RUN / 'frozen.json'
    assert report['status'] == 'passed' and not report['quality_evaluated'] and len(report['checks']) == 7
    assert cpu.digest(p) == report['frozen_sha256'] == FROZEN_SHA
    f = read(p)
    assert f['pair'] == cpu.PAIR and not f['data']['evaluation']['fresh_project_test']
    assert f['data']['evaluation']['previously_used_by'] == 'E035' and not f['data']['tiny_used_for_selecting_L06']
    assert cpu.digest(cpu.HERE / 'OPIS.md') == f['protocol_sha256']
    assert cpu.digest(cpu.HERE / 'kontrola_cpu.py') == f['cpu_code_sha256']
    for path, h in f['input_sha256'].items():
        assert cpu.digest(cpu.ROOT / path) == h, path
    old = read(cpu.R42 / 'frozen.json')['hashes'] | read(cpu.R42 / 'manifest.json')['code_hashes']
    assert cpu.rel(E35) in old
    for path, h in old.items():
        if path.endswith('.py'):
            assert cpu.digest(cpu.ROOT / path) == h, path
    up = cpu.ROOT / 'runs/E029/upstream_6946915'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=up, text=True).strip() == f['upstream_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=up, text=True).strip()
    if check_cache:
        for role, path in (('calibration', cpu.TRAIN), ('evaluation', cpu.VAL)):
            assert cpu.rel(path) == f['data'][role]['cache']
            assert cpu.digest(path) == f['data'][role]['sha256'], path
    return f


def count_cost(torch, model):
    mac = 0

    def hook(layer, _x, y):
        nonlocal mac
        mac += y.numel() * layer.weight[0].numel()

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    try:
        with torch.inference_mode():
            z = model(torch.zeros((1, 3, 64, 64), device=next(model.parameters()).device))
        assert z.shape == (1, 200) and torch.isfinite(z).all()
    finally:
        for h in hooks:
            h.remove()
    return dict(params=sum(p.numel() for p in model.parameters()), mac=mac)


def frozen_model(e35, spec, base, key):
    if key == 'teacher':
        return copy.deepcopy(base), spec['sizes'], spec['teacher']
    tag, profile = key.split('/')
    budget = spec['budgets'][tag]
    expected = budget['profiles'][profile]
    if profile == 'U':
        model, diag = e35.fold(cpu.PAIR, base, budget['uniform_plan'], 'U')
    else:
        model, diag = e35.e29.official(base, 0.5, expected['widths'])
        model.fc.bias.data.copy_(base.fc.bias)
        model.eval().requires_grad_(False)
    assert [d['k'] for d in diag] == expected['widths'] and [d['n'] for d in diag] == spec['sizes']
    assert all(set(d['labels']) == set(range(d['k'])) for d in diag)
    caps = budget['caps']
    for field in ('params', 'mac'):
        assert 999*caps[field] <= 1000*expected[field] <= 1000*caps[field]
    return model, expected['widths'], {k: expected[k] for k in ('params', 'mac')}


def worker(run, gpu):
    assert run.parent == RUNS
    m = read(run / 'manifest.json')
    assert m['status'] == 'running' and os.getppid() == m['pid']
    assert cpu.digest(run / 'frozen.json') == FROZEN_SHA
    f = read(run / 'frozen.json')
    seed = f['planned_evaluation']['gpu_to_seed'][str(gpu)]
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', seed=seed, gpu=gpu, pid=os.getpid(), started_utc=utc(), val_opened=False)
    cpu.write(out / 'status.json', status)
    allowed = {cpu.TRAIN, cpu.VAL, cpu.CKPT[seed]}
    opened, denied = set(), []

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = cpu.Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(cpu.ROOT / 'data') or (p.is_relative_to(cpu.ROOT) and p.suffix in ('.pt', '.pth', '.safetensors', '.ckpt')):
            if p not in allowed or (p == cpu.VAL and not status['val_opened']) or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
                denied.append(cpu.rel(p))
                raise RuntimeError(f'E044: zabroniony dostęp do {p}')
            opened.add(p)

    sys.addaudithook(guard)
    try:
        e35 = cpu.load(E35, '_e044_e035_ops')
        np, torch, bn, e30 = e35.np, e35.torch, e35.bn, e35.e30
        assert torch.cuda.device_count() == 1 and os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(seed)
        np.random.seed(seed)

        data, cal = f['data'], f['data']['calibration']
        with np.load(cpu.TRAIN, allow_pickle=False) as cache:
            train = cache['images']
        assert train.shape == (100000, 64, 64, 3) and train.dtype == np.uint8
        aug = e30.augment(train, cal['A'])
        view = cpu.sha(aug.contiguous().numpy().tobytes())
        assert view == cal['views_sha256']
        a = e30.normalize(aug)
        assert a.shape == (4096, 3, 64, 64) and torch.isfinite(a).all()
        del train, aug
        base = e35.model_source(cpu.PAIR, seed)
        before = e35.state_hash(base)
        assert cpu.rel(e35.checkpoint(cpu.PAIR, seed)) == data['checkpoints'][str(seed)]['path']
        models, evidence = {}, {}
        for key in ROWS:
            model, widths, expected = frozen_model(e35, f, base, key)
            assert torch.equal(model.fc.bias, base.fc.bias)
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            model.to('cuda:0')
            costs = count_cost(torch, model)
            assert costs == expected, (seed, key, costs, expected)
            params = bn.weights_hash(model)
            result = e35.e29.calibrate(model, a, 'CMA')
            assert result['estimator'] == 'CMA_reset' and result['params_unchanged']
            assert (result['samples'], result['batches'], result['last_batch']) == (4096, 32, 128)
            assert params == result['parameters_sha256'] == bn.weights_hash(model)
            assert result['layers'] and all((l['counter_start'], l['counter_end']) == (0, 32) for l in result['layers'])
            assert all(not mod.training for mod in model.modules())
            evidence[key] = dict(**costs, widths=widths, cma_samples=result['samples'], cma_batches=result['batches'],
                                 bn_counters_all_0_to_32=True, state_sha256=e35.state_hash(model), params_sha256=params)
            models[key] = model
        assert e35.state_hash(base) == before and opened == {cpu.TRAIN, cpu.CKPT[seed]} and not denied
        del a, base
        stamp = dict(seed=seed, gpu=gpu, pid=os.getpid(), utc=utc(), frozen_sha256=FROZEN_SHA,
                     device=torch.cuda.get_device_name(0), a_views_sha256=view,
                     checkpoint_sha256=data['checkpoints'][str(seed)]['sha256'],
                     source_state_unchanged=True, models=evidence)
        cpu.write(out / 'ready.json', stamp)
        print(f'seed {seed}, GPU {gpu}: 7 modeli po CMA, czekam na wspólną bramkę', flush=True)
        gate = run / 'inference_gate.json'
        t0 = time.monotonic()
        while not gate.is_file():
            assert time.monotonic() - t0 < LIMIT and os.getppid() == m['pid'], 'Brak bramki inferencji Tiny val'
            time.sleep(.5)
        g = read(gate)
        assert g == read(RELEASE) and g['run'] == cpu.rel(run)
        assert g['ready_sha256'][str(seed)] == cpu.digest(out / 'ready.json')

        status['val_opened'] = True
        cpu.write(out / 'status.json', status)
        assert data['evaluation']['indices'] == '0..9999' and data['evaluation']['count'] == 10000
        with np.load(cpu.VAL, allow_pickle=False) as cache:
            raw, labels = cache['images'], cache['labels']
        assert raw.shape == (10000, 64, 64, 3) and raw.dtype == np.uint8
        assert labels.shape == (10000,) and labels.dtype.kind in 'iu'
        assert np.array_equal(np.bincount(labels), np.full(200, 50))
        x = e30.normalize(torch.from_numpy(raw).permute(0, 3, 1, 2).float()/255)
        y = torch.tensor(labels, dtype=torch.long)
        assert torch.isfinite(x).all()
        del raw
        correct = []
        for key, model in models.items():
            z = bn.predict(model, x)
            assert z.shape == (10000, 200) and torch.isfinite(z).all()
            assert e35.state_hash(model) == evidence[key]['state_sha256']
            correct.append((z.argmax(1) == y).numpy())
        assert opened == allowed and not denied
        np.savez_compressed(out / 'correctness.npz', models=np.asarray(ROWS), labels=labels,
                            correct=np.stack(correct), indices=np.arange(10000))
        status.update(status='completed', inference_utc=g['utc'], assessments=len(correct), samples=len(labels),
                      val_cache_sha256=data['evaluation']['sha256'], model_states_unchanged=True)
        print(f'seed {seed}, GPU {gpu}: {len(correct)}/7 zakończone', flush=True)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=utc(), files_opened=sorted(map(cpu.rel, opened)), denied=denied)
        cpu.write(out / 'status.json', status)


def launch():
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and '/envs/attention/' in sys.executable
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3'
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert not RELEASE.exists(), 'Ten pomiar E044 był już otwarty'
    import torch
    assert torch.cuda.device_count() == 4
    pre = gpu_info()
    assert [g['index'] for g in pre['gpus']] == list('0123') and not pre['compute_apps']
    assert all(int(g['memory_mib']) < 256 and int(g['utilization']) == 0 for g in pre['gpus'])
    f = verify_inputs(check_cache=True)
    now = datetime.now(timezone.utc)
    run = RUNS / f'{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    (run / 'source').mkdir(parents=True)
    sources = {}
    for n in FILES:
        p = cpu.HERE / n
        sources[n] = cpu.digest(p)
        shutil.copyfile(p, run / 'source' / n)
    shutil.copyfile(CPU_RUN / 'frozen.json', run / 'frozen.json')
    cache_hashes = {f['data'][k]['cache']: f['data'][k]['sha256'] for k in ('calibration', 'evaluation')}
    m = dict(status='running', phase='tiny_val', run=cpu.rel(run), pid=os.getpid(), started_utc=utc(),
             frozen_sha256=FROZEN_SHA, cpu_run=cpu.rel(CPU_RUN), source_sha256=sources,
             settings=f['planned_evaluation'], cache_sha256=cache_hashes, training=False, dev_used=False,
             fresh_project_test=False, inference_released=False, workers=[], gpu_start=pre,
             python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version(), limit_seconds=LIMIT, command=[sys.executable, '-B', *sys.argv],
             environment={k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                         'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    cpu.write(run / 'manifest.json', m)
    print('RUN', cpu.rel(run), 'PID', os.getpid(), flush=True)
    processes, logs, start = [], [], time.monotonic()
    try:
        for gpu, seed in enumerate(cpu.SEEDS):
            logs.append((run / f'worker_{gpu}.log').open('w'))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, '-B', sys.argv[0], 'worker', '--run', str(run), '--gpu', str(gpu)]
            p = subprocess.Popen(cmd, cwd=cpu.ROOT, env=env, stdout=logs[-1], stderr=subprocess.STDOUT)
            processes.append(p)
            m['workers'].append(dict(seed=seed, gpu=gpu, pid=p.pid, uuid=pre['gpus'][gpu]['uuid'], command=cmd))
        cpu.write(run / 'manifest.json', m)
        map_gpu = {str(w['pid']): w['uuid'] for w in m['workers']}
        ready = {s: run / f'seed_{s}/ready.json' for s in cpu.SEEDS}
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in processes]
                monitor.write(json.dumps(dict(**gpu_info(), exit_codes=codes)) + '\n')
                monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0] * 4:
                    break
                assert time.monotonic() - start < LIMIT, 'Limit E044: 30 minut'
                if not m['inference_released'] and all(p.is_file() for p in ready.values()):
                    assert gpu_info()['compute_apps'] == map_gpu
                    for w in m['workers']:
                        r = read(ready[w['seed']])
                        assert (r['seed'], r['gpu'], r['pid']) == (w['seed'], w['gpu'], w['pid'])
                        assert r['device'] == pre['gpus'][w['gpu']]['name'] and tuple(r['models']) == ROWS
                        assert r['a_views_sha256'] == f['data']['calibration']['views_sha256']
                    gate = dict(run=cpu.rel(run), utc=utc(), pid_gpu_uuid=map_gpu, frozen_sha256=FROZEN_SHA,
                                ready_sha256={str(s): cpu.digest(p) for s, p in ready.items()})
                    with RELEASE.open('x') as mark:
                        json.dump(gate, mark, indent=2)
                        mark.write('\n')
                    m['inference_released'] = True
                    cpu.write(run / 'inference_gate.json', gate)
                    cpu.write(run / 'manifest.json', m)
                    print('Bramka 4 GPU/CMA PASS — start punktacji Tiny val', flush=True)
                time.sleep(2)
        assert m['inference_released']
        assert all(cpu.digest(cpu.HERE / n) == h for n, h in sources.items())
        assert all(cpu.digest(cpu.ROOT / p) == h for p, h in cache_hashes.items())
        files = ['inference_gate.json']
        for s in cpu.SEEDS:
            status = read(run / f'seed_{s}/status.json')
            assert status['status'] == 'completed' and status['assessments'] == 7
            files += [f'seed_{s}/{n}' for n in ('ready.json', 'status.json', 'correctness.npz')]
        m.update(status='completed', assessments=28, output_sha256={n: cpu.digest(run / n) for n in files})
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        raise
    finally:
        for w, p in zip(m['workers'], processes):
            w['exit_code'] = p.poll()
        for log in logs:
            log.close()
        m.update(finished_utc=utc(), seconds=time.monotonic()-start)
        try:
            m['gpu_end'] = gpu_info()
        finally:
            cpu.write(run / 'manifest.json', m)
    print('COMPLETED', cpu.rel(run), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('run')
    w = sub.add_parser('worker')
    w.add_argument('--run', type=cpu.Path, required=True)
    w.add_argument('--gpu', type=int, choices=range(4), required=True)
    args = p.parse_args()
    assert not sys.flags.optimize and sys.dont_write_bytecode
    if args.action == 'run':
        launch()
    else:
        worker(args.run.resolve(), args.gpu)


if __name__ == '__main__':
    main()
