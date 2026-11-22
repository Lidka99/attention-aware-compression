"""2026-11-21_E043_l06_potwierdzenie_transfer: jednorazowa ocena na 4 GPU.
Wejścia: frozen.json z bramki CPU, checkpointy E015, A i rezerwa CIFAR-100.
Wyjścia: manifest, dowód CMA/identyczności modeli, liczniki top-1 per seed.
Pełny protokół: OPIS.md.
"""

import argparse
import copy
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import kontrola_cpu as cpu


CPU_RUN = cpu.ROOT / 'runs/E043/cpu_20261121T230452Z_8b558297'
FROZEN_SHA = 'f84a40c4d4b435495904723aef61664ef11b359de4bee12ecc9160d6d81ded81'
RUNS = CPU_RUN.parent
USED = RUNS / 'reserve_release.json'
LIMIT = 1800
ROWS = ('teacher', *(f'{t}/{p}' for t in cpu.TAGS for p in cpu.PROFILES))
AUTHORIZATION = 'kontynuuj i uruchom'
PACKAGE_PATHS = ('kontrola_cpu.py', 'ocena_rezerwy.py', 'analiza.py', 'OPIS.md', 'URUCHOMIENIE_GPU.md')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def info():
    cmd = ['nvidia-smi', '--format=csv,noheader,nounits']
    fields = '--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu'
    rows = subprocess.check_output([*cmd, fields], text=True).strip().splitlines()
    gpus = [dict(zip(('index', 'name', 'uuid', 'driver', 'memory_mib', 'utilization'),
                     [s.strip() for s in row.split(',')])) for row in rows]
    rows = subprocess.check_output([*cmd, '--query-compute-apps=pid,gpu_uuid'], text=True).strip().splitlines()
    apps = {p.strip(): g.strip() for p, g in (row.split(',') for row in rows)}
    return dict(utc=utc(), gpus=gpus, compute_apps=apps)


def audit_hook(seed, opened, denied):
    allowed = {cpu.SPLIT, cpu.TRAIN, cpu.CHECKPOINTS[seed]}

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = cpu.Path(os.fsdecode(args[0])).resolve()
        protected = path.is_relative_to(cpu.ROOT / 'data') or path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors')
        if protected and path.is_relative_to(cpu.ROOT):
            if path not in allowed or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
                denied.append(cpu.rel(path))
                raise RuntimeError(f'E043: niedozwolony dostęp do {path}')
            opened.add(cpu.rel(path))

    sys.addaudithook(guard)
    return {cpu.rel(p) for p in allowed}


def load_payload(data, np):
    # Hash dokładnie tych bajtów, które zostaną zdeserializowane.
    img_bytes = cpu.TRAIN.read_bytes()
    assert cpu.sha256(img_bytes) == data['expected_train_sha256']
    payload = pickle.loads(img_bytes, encoding='latin1')
    raw = payload['data']
    assert raw.shape == (50000, 3072) and raw.dtype == np.uint8
    return payload


def load_ids(data):
    split_blob = cpu.SPLIT.read_bytes()
    assert cpu.sha256(split_blob) == data['split_sha256']
    split = json.loads(split_blob)['splits']
    a = read(cpu.ROOT / data['calibration_source'])['A']
    reserve = split['reserve']
    for key, ids in (split | {'A': a}).items():
        assert cpu.index_hash(ids) == data['index_list_sha256'][key]
        assert len(set(ids)) == len(ids) == data['sizes'][key]
    assert set(a) <= set(split['train']) and not set(reserve) & set(split['train'] + split['dev'] + a)
    return a, reserve


def historical_states(seed):
    """Tylko hashe stanów po CMA z E042, bez odczytu jakości jako kryterium."""
    audit = read(cpu.R42 / 'analysis/audit.json')
    states = {}
    for name in ('teacher', *cpu.PROFILES):
        path = cpu.R42 / f'seed_{seed}/{cpu.PAIR}/{name}.json'
        blob = path.read_bytes()
        assert cpu.sha256(blob) == audit['source_hashes'][cpu.rel(path)]
        key = name if name == 'teacher' else f'b50/{name}'
        states[key] = json.loads(blob)['evidence']['state_sha256']
    return states


def worker(run, gpu):
    assert run.parent == RUNS and run.is_dir()
    manifest = read(run / 'manifest.json')
    assert manifest['status'] == 'running' and manifest['pid'] == os.getppid()
    assert manifest['approval']['request'] == AUTHORIZATION
    assert cpu.digest(run / 'frozen.json') == FROZEN_SHA
    frozen = read(run / 'frozen.json')
    seed = frozen['planned_evaluation']['gpu_to_seed'][str(gpu)]
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', seed=seed, gpu=gpu, pid=os.getpid(), started_utc=utc(), reserve_sliced=False)
    cpu.write(out / 'status.json', status)
    opened, denied = set(), []
    allowed = audit_hook(seed, opened, denied)
    try:
        cpu.load(cpu.E42 / 'geometria.py', 'geometria')
        e42 = cpu.load(cpu.E42 / 'experiment.py', '_e043_e042_operator')
        e35, rt = e42.e35, e42.rt
        torch, np, bn = rt.torch, rt.np, e35.bn
        rt.configure(gpu)
        assert torch.cuda.current_device() == 0
        data = frozen['data']
        a_ids, reserve = load_ids(data)
        ckpt = data['checkpoints'][str(seed)]
        assert cpu.rel(e35.checkpoint(cpu.PAIR, seed)) == ckpt['path']
        assert cpu.digest(cpu.ROOT / ckpt['path']) == ckpt['sha256']
        payload = load_payload(data, np)
        aug = bn.augment_uint8(payload['data'], a_ids)
        view_sha = bn.tensor_hash(aug)
        assert view_sha == data['expected_A_views_sha256']
        a = bn.normalize(aug)
        assert a.shape == (4096, 3, 32, 32) and torch.isfinite(a).all()
        del aug

        base = e35.model_source(cpu.PAIR, seed)
        geo, source_state = e42.verify_group_map(cpu.PAIR, base), e35.state_hash(base)
        assert list(geo.sizes) == frozen['sizes']
        former = historical_states(seed)
        models, checks = {}, {}
        for key in ROWS:
            if key == 'teacher':
                model, widths = copy.deepcopy(base), frozen['sizes']
                cost = dict(params=sum(p.numel() for p in model.parameters()), mac=rt.e37.mac(model, cpu.PAIR))
                assert cost == frozen['teacher']
            else:
                tag, name = key.split('/')
                budget = frozen['budgets'][tag]
                model, cost, diag = e42.build(cpu.PAIR, base, budget['profiles'][name], budget['UNIFORM_plan'], geo)
                widths = [d['k'] for d in diag]
                for k in ('params', 'mac'):
                    assert 999 * budget['caps'][k] <= 1000 * cost[k] <= 1000 * budget['caps'][k]
            model.to('cuda:0')
            calib = e35.e29.calibrate(model, a, 'CMA')
            assert calib['estimator'] == 'CMA_reset' and calib['params_unchanged']
            assert (calib['samples'], calib['batches'], calib['last_batch']) == (4096, 32, 128)
            layers = calib['layers']
            assert layers and all(l['counter_start'] == 0 and l['counter_end'] == 32 for l in layers)
            state = e35.state_hash(model)
            params = bn.weights_hash(model)
            assert params == calib['parameters_sha256']
            assert all(p.device.type == 'cuda' and not p.requires_grad and p.grad is None for p in model.parameters())
            assert all(not m.training for m in model.modules())
            checks[key] = dict(actual=cost, widths=widths, cma_samples=calib['samples'], cma_batches=calib['batches'],
                               cma_last_batch=calib['last_batch'], cma_mode=calib['estimator'],
                               bn_counters={l['name']: [l['counter_start'], l['counter_end']] for l in layers},
                               params_sha256=params, state_after_cma_sha256=state,
                               e042_state_sha256=former.get(key),
                               e042_state_equal=(state == former[key]) if key in former else None)
            models[key] = model
        assert e35.state_hash(base) == source_state and not denied and not e42.denied()
        assert opened == allowed
        ready = dict(seed=seed, gpu=gpu, pid=os.getpid(), device=torch.cuda.get_device_name(0),
                     cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], checkpoint_sha256=ckpt['sha256'],
                     a_indices_sha256=cpu.index_hash(a_ids), a_views_sha256=view_sha,
                     reserve_indices_sha256=cpu.index_hash(reserve), train_file_sha256=data['expected_train_sha256'],
                     frozen_sha256=FROZEN_SHA, models=checks, source_state_unchanged=True, utc=utc())
        cpu.write(out / 'ready.json', ready)
        print(f'seed {seed}, GPU {gpu}: 7 modeli po CMA; czekam na wspólną bramkę', flush=True)
        gate = run / 'inference_gate.json'
        t0 = time.monotonic()
        while not gate.is_file():
            if time.monotonic() - t0 > LIMIT or os.getppid() != manifest['pid']:
                raise RuntimeError('Brak wspólnej bramki do inferencji rezerwy')
            time.sleep(0.5)
        release = read(gate)
        assert release == read(USED) and release['run'] == cpu.rel(run)
        assert release['ready_sha256'][str(seed)] == cpu.digest(out / 'ready.json')

        status['reserve_sliced'] = True
        cpu.write(out / 'status.json', status)
        raw = payload['data'][reserve].reshape(-1, 3, 32, 32)
        y = torch.tensor(np.asarray(payload['fine_labels'])[reserve], dtype=torch.long)
        assert len(y) == len(raw) == 5000 and torch.equal(torch.bincount(y), torch.full((100,), 50))
        x = bn.normalize(torch.from_numpy(raw))
        assert torch.isfinite(x).all()
        del payload, raw
        result = {}
        for key, model in models.items():
            z = bn.predict(model, x)
            assert z.shape == (5000, 100) and torch.isfinite(z).all()
            assert all(not m.training for m in model.modules())
            assert e35.state_hash(model) == checks[key]['state_after_cma_sha256']
            n = int((z.argmax(1) == y).sum().item())
            result[key] = dict(correct=n, samples=len(y), top1_pct=100*n/len(y), state_unchanged_in_eval=True)
        assert not denied and not e42.denied()
        cpu.write(out / 'metrics.json', dict(seed=seed, gpu=gpu, release_utc=release['utc'],
                                            reserve_indices_sha256=ready['reserve_indices_sha256'], models=result))
        status.update(status='completed', assessments=len(result), classes=100, reserve_count_per_class=50)
        print(f'seed {seed}, GPU {gpu}: 7/7 ocen zakończonych', flush=True)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=utc(), files_opened=sorted(opened), denied=denied)
        cpu.write(out / 'status.json', status)


def verify_sources():
    cpu_manifest = read(CPU_RUN / 'manifest.json')
    fpath = CPU_RUN / 'frozen.json'
    assert cpu_manifest['status'] == 'passed' and not cpu_manifest['quality_evaluated']
    assert cpu.digest(fpath) == cpu_manifest['frozen_sha256'] == FROZEN_SHA
    frozen = read(fpath)
    assert len(cpu_manifest['checks']) == frozen['planned_evaluation']['rows'] == 28
    assert cpu.digest(cpu.HERE / 'OPIS.md') == frozen['protocol_sha256']
    assert cpu.digest(cpu.HERE / 'kontrola_cpu.py') == frozen['cpu_code_sha256']
    assert frozen['pair'] == cpu.PAIR and frozen['b50_exact_E042']
    for path, h in frozen['input_sha256'].items():
        assert cpu.digest(cpu.ROOT / path) == h, path
    f42 = read(cpu.R42 / 'frozen.json')
    sources = read(cpu.R42 / 'manifest.json')['code_hashes']
    for path, h in (f42['hashes'] | sources).items():
        if path.endswith('.py'):
            assert cpu.digest(cpu.ROOT / path) == h, path
    up = cpu.ROOT / 'runs/E029/upstream_6946915'
    assert subprocess.check_output(['git', '-C', str(up), 'rev-parse', 'HEAD'], text=True).strip() == frozen['upstream_commit']
    assert not subprocess.check_output(['git', '-C', str(up), 'status', '--porcelain'], text=True).strip()
    return frozen


def launch():
    assert 'envs/attention/' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3'
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert not USED.exists(), 'Jednorazowa bramka rezerwy E043 już otwarta'
    frozen = verify_sources()
    import torch
    assert torch.cuda.device_count() == 4
    gpu_start = info()
    assert [g['index'] for g in gpu_start['gpus']] == list('0123') and not gpu_start['compute_apps']
    assert all(int(g['memory_mib']) < 256 and int(g['utilization']) == 0 for g in gpu_start['gpus'])

    run = RUNS / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    (run / 'source').mkdir(parents=True)
    source_hashes = {}
    for name in PACKAGE_PATHS:
        p = cpu.HERE / name
        source_hashes[name] = cpu.digest(p)
        shutil.copyfile(p, run / 'source' / name)
    shutil.copyfile(CPU_RUN / 'frozen.json', run / 'frozen.json')
    manifest = dict(status='running', phase='reserve', pid=os.getpid(), run=cpu.rel(run), started_utc=utc(),
                    approval=dict(request=AUTHORIZATION, scope='jednorazowa ocena E043, 4 GPU, CIFAR-100 reserve'),
                    cpu_run=cpu.rel(CPU_RUN), frozen_sha256=FROZEN_SHA, source_sha256=source_hashes,
                    settings=frozen['planned_evaluation'], training=False, reserve_inference_released=False,
                    official_test_used=False, dev_used=False,
                    command=[sys.executable, '-B', *sys.argv],
                    environment={k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES',
                                 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')},
                    python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), limit_seconds=LIMIT, gpu_start=gpu_start, workers=[])
    cpu.write(run / 'manifest.json', manifest)
    print('RUN', cpu.rel(run), 'PID', os.getpid(), flush=True)
    processes, logs = [], []
    started = time.monotonic()
    try:
        for gpu, seed in enumerate(cpu.SEEDS):
            log = (run / f'worker_{gpu}.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, '-B', sys.argv[0], 'worker', '--run', str(run), '--gpu', str(gpu)]
            p = subprocess.Popen(cmd, cwd=cpu.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append(p)
            manifest['workers'].append(dict(pid=p.pid, seed=seed, gpu=gpu, uuid=gpu_start['gpus'][gpu]['uuid'], command=cmd))
        cpu.write(run / 'manifest.json', manifest)
        expected_map = {str(r['pid']): r['uuid'] for r in manifest['workers']}
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in processes]
                telemetry = info()
                monitor.write(json.dumps(dict(**telemetry, exit_codes=codes)) + '\n')
                monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0]*4:
                    break
                if time.monotonic() - started > LIMIT:
                    raise TimeoutError('Limit E043: 30 min')
                paths = {s: run / f'seed_{s}/ready.json' for s in cpu.SEEDS}
                if not manifest['reserve_inference_released'] and all(p.is_file() for p in paths.values()):
                    # Aktualny pomiar GPU: telemetry z początku iteracji mógł wyprzedzić ready.
                    assert info()['compute_apps'] == expected_map, 'Nie potwierdzono czterech przypisań PID/GPU'
                    ready = [read(paths[s]) for s in cpu.SEEDS]
                    for r, w in zip(ready, manifest['workers']):
                        assert (r['seed'], r['gpu'], r['pid']) == (w['seed'], w['gpu'], w['pid'])
                        assert r['device'] == gpu_start['gpus'][w['gpu']]['name'] and tuple(r['models']) == ROWS
                        assert r['a_views_sha256'] == frozen['data']['expected_A_views_sha256']
                    release = dict(run=cpu.rel(run), utc=utc(), frozen_sha256=FROZEN_SHA, pid_gpu_uuid=expected_map,
                                   ready_sha256={str(s): cpu.digest(paths[s]) for s in cpu.SEEDS})
                    with USED.open('x') as ledger:
                        json.dump(release, ledger, indent=2)
                        ledger.write('\n')
                    manifest['reserve_inference_released'] = True
                    cpu.write(run / 'inference_gate.json', release)
                    cpu.write(run / 'manifest.json', manifest)
                    print('Bramka 4 GPU/CMA PASS — rozpoczęto ocenę na rezerwie', flush=True)
                time.sleep(2)
        assert manifest['reserve_inference_released']
        assert all(source_hashes[n] == cpu.digest(cpu.HERE / n) for n in PACKAGE_PATHS)
        outputs = {}
        for s in cpu.SEEDS:
            folder = run / f'seed_{s}'
            status = read(folder / 'status.json')
            assert status['status'] == 'completed' and status['assessments'] == len(ROWS) and status['reserve_sliced']
            assert tuple(read(folder / 'metrics.json')['models']) == ROWS
            for n in ('ready.json', 'status.json', 'metrics.json'):
                outputs[str((folder / n).relative_to(run))] = cpu.digest(folder / n)
        manifest.update(status='completed', assessments=28, reserve_used=True, output_sha256=outputs)
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
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
        for w, p in zip(manifest['workers'], processes):
            w['exit_code'] = p.poll()
        for log in logs:
            log.close()
        manifest.update(finished_utc=utc(), seconds=time.monotonic() - started)
        try:
            manifest['gpu_end'] = info()
        finally:
            cpu.write(run / 'manifest.json', manifest)
    print('COMPLETED', cpu.rel(run), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    go = sub.add_parser('run')
    go.add_argument('--zgoda-rezerwa', action='store_true', required=True)
    w = sub.add_parser('worker')
    w.add_argument('--run', type=cpu.Path, required=True)
    w.add_argument('--gpu', type=int, choices=range(4), required=True)
    args = p.parse_args()
    assert not sys.flags.optimize and sys.dont_write_bytecode, 'Uruchom z -B, bez -O'
    if args.action == 'run':
        launch()
    else:
        worker(args.run.resolve(), args.gpu)


if __name__ == '__main__':
    main()
