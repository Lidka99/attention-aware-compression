"""E042: zamrożenie puli, kontrola operatora i 4 procesy dev; por. OPIS.md.
Wejścia: bank geometrii, checkpointy E011/E015, indeksy E035; wyjścia: logi i metryki.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import geometria as design

HERE, ROOT, RUNS = design.HERE, design.ROOT, design.RUNS
sys.path.insert(0, str(HERE.parent / '2026-11-08_E038_operator_alokacja'))
import runtime as rt

PAIRS, SEEDS = design.PAIRS, rt.SEEDS
e35 = rt.e35
SPLITS = {'cifar10': e35.e31.d10.SPLIT, 'cifar100': e35.bn.SPLIT}
TRAIN_FILES = {
    'cifar10': {e35.e31.d10.DATA / f'data_batch_{i}' for i in range(1, 6)},
    'cifar100': {ROOT / 'data/cifar100/cifar-100-python/train'},
}
DATA_FILES = set(SPLITS.values()).union(*TRAIN_FILES.values())
CHECKPOINTS = {e35.checkpoint(pair, seed) for pair in PAIRS for seed in SEEDS}
LIMIT_SECONDS = 1800
DENIED, DATA_OPENED = [], set()


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT):
        return
    if path.is_relative_to(ROOT / 'data') or path in DATA_FILES:
        allowed = path in DATA_FILES
        if allowed:
            DATA_OPENED.add(rt.rel(path))
    elif path.suffix in ('.pt', '.pth', '.safetensors', '.ckpt'):
        allowed = path in CHECKPOINTS
    else:
        return
    flags = args[2]
    if not allowed or flags & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
        DENIED.append(rt.rel(path))
        raise RuntimeError(f'E042: niedozwolony dostęp: {path}')


sys.addaudithook(guard)


def denied():
    return DENIED + rt.DENIED + rt.e37.DENIED


def cpu_only():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    rt.torch.set_num_threads(2)


def split_evidence(indices):
    evidence = {}
    for ds, path in SPLITS.items():
        parts = rt.read(path)
        if ds == 'cifar100':
            parts = parts['splits']
        a, dev = (set(indices[ds][name]) for name in ('A', 'dev'))
        assert len(a) == len(indices[ds]['A']) == 4096
        assert len(dev) == len(indices[ds]['dev']) == 5000
        assert indices[ds]['dev'] == parts['dev'] and a.issubset(parts['train'])
        assert not a & dev and not (a | dev) & set(parts.get('reserve', []))
        evidence[ds] = dict(A=len(a), dev=len(dev), reserve_in_A_or_dev=0)
    return evidence


def freeze(cpu_run):
    cpu_only()
    frozen_path = HERE / 'frozen.json'
    assert not frozen_path.exists(), 'Pula jest już zamrożona'
    cpu_run = cpu_run.resolve()
    assert cpu_run.parent == RUNS and cpu_run.name.startswith('cpu_')
    manifest = rt.read(cpu_run / 'manifest.json')
    assert manifest['status'] == 'completed' and not manifest['quality_used']
    assert manifest['bank_sha256'] == rt.digest(cpu_run / 'bank.json')
    e35.validate_hashes(manifest['hashes'])
    bank = rt.read(cpu_run / 'bank.json')
    design.validate_bank(bank)
    history = rt.read(design.R35 / 'frozen.json')
    prov = history['provenance']
    # Historyczne hashe plików kodu i wyłącznie używanych danych/8 wag.
    input_paths = DATA_FILES | CHECKPOINTS
    hashes = {p: h for p, h in prov.items() if p.endswith('.py')}
    hashes.update({rt.rel(p): prov[rt.rel(p)] for p in input_paths})
    hashes.update(manifest['hashes'])
    for path in (Path(rt.__file__), Path(rt.e37.__file__), Path(e35.__file__),
                 cpu_run / 'bank.json', cpu_run / 'manifest.json'):
        hashes[rt.rel(path)] = rt.digest(path)
    reference_paths = {}
    for pair in PAIRS:
        reference_paths[pair] = {}
        for seed in SEEDS:
            folder = design.R35 / f'dev/seed_{seed}/{pair}'
            s, c = folder / 'b50/summary.json', folder / 'controls.json'
            saved = rt.read(s)
            assert saved['phase'] == 'dev' and saved['status'] == 'completed'
            assert saved['plan'] == history['plans'][pair]['b50']
            assert saved['checkpoint_sha256'] == hashes[rt.rel(e35.checkpoint(pair, seed))]
            assert saved['controls_sha256'] == rt.digest(c)
            for p in (s, c):
                hashes[rt.rel(p)] = rt.digest(p)
            reference_paths[pair][str(seed)] = [rt.rel(s), rt.rel(c)]
    indices = {ds: history['indices'][ds] for ds in SPLITS}
    splits = split_evidence(indices)
    e35.validate_hashes(hashes)
    assert not denied()
    rt.write(frozen_path, dict(created_utc=rt.utc(), bank=bank, geometry_run=rt.rel(cpu_run),
        indices=indices, splits=splits, hashes=hashes, reference_paths=reference_paths,
        plans={pair: history['plans'][pair]['b50'] for pair in PAIRS},
        primary=dict(pair='resnet_cifar100', minimum_count_gain=200, minimum_positive=3)))
    print('FREEZE PASS', rt.digest(frozen_path), flush=True)


def verify_group_map(pair, base):
    axes = e35.e29.get_axis_to_perm_ResNet18(override=False)
    groups = e35.e29.axes2perm_to_perm2axes(axes)
    geo = rt.e37.geometry(pair, base)
    assert tuple(groups) == design.g.NAMES and tuple(geo.names) == design.g.GEOMETRY_NAMES
    assert list(geo.sizes) == design.g.SIZES.tolist()
    for components, name in zip(groups.values(), geo.names):
        assert components == geo.members[name]
    return geo


def build(pair, base, row, uniform_plan, geo):
    if row['profile'] == 'U':
        model, diag = e35.fold(pair, base, uniform_plan, 'U')
    else:
        model, diag = rt.e37.fold_widths(pair, base, row['widths'])
    assert [d['k'] for d in diag] == row['widths']
    assert [d['n'] for d in diag] == list(geo.sizes)
    for d, name in zip(diag, geo.names):
        assert d['first_axis'] == geo.members[name][0][0]
        assert len(d['labels']) == d['n'] and set(d['labels']) == set(range(d['k']))
    actual = dict(params=sum(p.numel() for p in model.parameters()), mac=rt.e37.mac(model, pair))
    assert actual == {k: row[k] for k in ('params', 'mac')}
    assert actual['params'] == geo.count(row['widths'])
    assert rt.torch.equal(model.fc.bias, base.fc.bias)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return model, actual, diag


def check():
    cpu_only()
    frozen_path = HERE / 'frozen.json'
    frozen = rt.read(frozen_path)
    design.validate_bank(frozen['bank'])
    assert split_evidence(frozen['indices']) == frozen['splits']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=e35.e29.UP, text=True).strip() == e35.e29.COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=e35.e29.UP, text=True).strip()
    out = RUNS / f'check_{design.g.utc()}_{uuid.uuid4().hex[:8]}'
    out.mkdir()
    started = time.monotonic()
    state = dict(status='running', checks=[], run=rt.rel(out), pid=os.getpid(), started_utc=rt.utc(),
                 code_hashes=rt.hashes(HERE), frozen_sha256=rt.digest(frozen_path), quality_evaluated=False)
    rt.write(out / 'manifest.json', state)
    try:
        for pair in PAIRS:
            base = e35.model_source(pair, 42)
            geo = verify_group_map(pair, base)
            before = e35.state_hash(base)
            menu = frozen['bank']['pairs'][pair]
            rows = menu['references'] + [next(r for r in menu['candidates'] if r['family'] == family)
                                         for family in ('wewnatrz', 'miedzy', 'losowe')]
            for row in rows:
                model, actual, _ = build(pair, base, row, frozen['plans'][pair], geo)
                state['checks'].append(dict(pair=pair, profile=row['profile'], **actual))
                rt.write(out / 'manifest.json', state)
                del model
                assert time.monotonic() - started < 300, 'Limit CPU 300 s'
            assert before == e35.state_hash(base)
            del base
        assert len(state['checks']) == 10 and not denied()
        assert DATA_OPENED == {rt.rel(p) for p in SPLITS.values()}
        state['status'] = 'passed'
    except BaseException:
        state.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        state.update(finished_utc=rt.utc(), seconds=time.monotonic()-started,
                     data_opened=sorted(DATA_OPENED), data_open_attempts=denied())
        rt.write(out / 'manifest.json', state)
    rt.write(HERE / 'check.json', state)
    print('CPU OPERATOR PASS', len(state['checks']), state['run'], flush=True)


def metric(pair, model, a, x, y):
    values, evidence = e35.evaluate(pair, model, a, x, y, 'CMA')
    rt.verify_metric(values, evidence)
    cal = evidence['calibration']
    assert evidence['samples'] == 5000 and cal['samples'] == 4096 and cal['batches'] == 32
    count = round(5000 * values['top1'])
    assert abs(values['top1'] - count / 5000) < 1e-12
    return dict(metric=values, evidence=evidence, correct=count)


def compare_reference(item, values, evidence):
    assert item['metric']['top1'] == values['top1']
    assert abs(item['metric']['nll'] - values['nll']) <= 1e-5
    assert item['evidence']['state_sha256'] == evidence['state_sha256']


def worker(run, gpu):
    rt.configure(gpu)
    seed = SEEDS[gpu]
    out = run / f'seed_{seed}'
    out.mkdir()
    frozen = rt.read(run / 'frozen.json')
    assert split_evidence(frozen['indices']) == frozen['splits']
    status = dict(status='running', seed=seed, gpu=gpu, pid=os.getpid(), started_utc=rt.utc(), completed=[])
    rt.write(out / 'status.json', status)
    contexts = {}
    try:
        for pair in PAIRS:
            ds = pair.split('_')[1]
            a, x, y, views = e35.e31.data(ds, frozen['indices'][ds])
            base = e35.model_source(pair, seed)
            geo, before = verify_group_map(pair, base), e35.state_hash(base)
            s, c = (rt.read(ROOT / p) for p in frozen['reference_paths'][pair][str(seed)])
            assert views == s['views_sha256'] == c['views_sha256']
            teacher = metric(pair, base, a, x, y)
            compare_reference(teacher, c['metrics']['CMA'], c['evidence']['CMA'])
            u = frozen['bank']['pairs'][pair]['references'][0]
            model, actual, diag = build(pair, base, u, frozen['plans'][pair], geo)
            result_u = metric(pair, model, a, x, y)
            compare_reference(result_u, s['metrics']['U']['CMA'], s['evidence']['U']['CMA'])
            del model
            ckpt_hash = frozen['hashes'][rt.rel(e35.checkpoint(pair, seed))]
            metadata = dict(phase='dev', pair=pair, seed=seed, views_sha256=views,
                            checkpoint_sha256=ckpt_hash, official_test_used=False, reserve_used=False)
            rt.write(out / pair / 'teacher.json', dict(metadata, profile='teacher', **teacher))
            rt.write(out / pair / 'U.json', dict(metadata, plan=u, actual=actual, clusters=diag, **result_u))
            contexts[pair] = (base, geo, before, a, x, y, metadata)
        rt.write(out / 'reference_gate.json', dict(passed=True, seed=seed, utc=rt.utc()))
        start = time.monotonic()
        gates = [run / f'seed_{s}/reference_gate.json' for s in SEEDS]
        while not all(p.is_file() for p in gates):
            if time.monotonic() - start > 600:
                raise TimeoutError('Brak wspólnej bramki odtworzenia')
            time.sleep(1)
        assert all(rt.read(p)['passed'] for p in gates)
        status['barrier_passed_utc'] = rt.utc()
        for pair in PAIRS:
            base, geo, before, a, x, y, metadata = contexts[pair]
            menu = frozen['bank']['pairs'][pair]
            for row in menu['references'][1:] + menu['candidates']:
                model, actual, diag = build(pair, base, row, frozen['plans'][pair], geo)
                values = metric(pair, model, a, x, y)
                rt.write(out / pair / f'{row["profile"]}.json',
                         dict(metadata, plan=row, actual=actual, clusters=diag, **values))
                status['completed'].append(f'{pair}/{row["profile"]}')
                rt.write(out / 'status.json', status)
                del model
            assert before == e35.state_hash(base)
            print(seed, pair, 'COMPLETED', len(menu['candidates']), 'candidates', flush=True)
        assert not denied()
        status.update(status='completed', original_states_unchanged=True)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        contexts.clear()
        status.update(finished_utc=rt.utc(), data_opened=sorted(DATA_OPENED), data_open_attempts=denied())
        rt.write(out / 'status.json', status)


def launch():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3' and rt.torch.cuda.device_count() == 4
    frozen = rt.validate_check(HERE)
    design.validate_bank(frozen['bank'])
    info = rt.telemetry()
    rows = [row.split(',') for row in info.splitlines()]
    assert len(rows) == 4 and all(int(r[0]) == i and int(r[3]) < 256 and int(r[4]) == 0
                                  for i, r in enumerate(rows)), info
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                         '--format=csv,noheader'], text=True).strip()
    run = RUNS / f'{design.g.utc()}_{uuid.uuid4().hex[:8]}'
    (run / 'source').mkdir(parents=True)
    for path in HERE.iterdir():
        if path.suffix in ('.py', '.md', '.json'):
            shutil.copy2(path, run / 'source' / path.name)
    for path in frozen['hashes']:
        source = ROOT / path
        if source.suffix == '.py':
            dest = run / 'source/dependencies' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    shutil.copy2(HERE / 'frozen.json', run / 'frozen.json')
    manifest = dict(status='running', pid=os.getpid(), run=rt.rel(run), started_utc=rt.utc(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
        frozen_sha256=rt.digest(run / 'frozen.json'), code_hashes=rt.hashes(HERE), workers=[],
        gpu_start=info, python=sys.version, executable=sys.executable, torch=rt.torch.__version__,
        cuda=rt.torch.version.cuda, limit_seconds=LIMIT_SECONDS,
        command=[sys.executable, '-B', rt.rel(HERE / 'experiment.py'), 'run'],
        environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CONDA_DEFAULT_ENV',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')},
        settings=dict(phase='dev', pairs=PAIRS, batch_per_gpu=128, mode='CMA', training=False, dtype='FP32', tf32=False))
    (run / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    rt.write(run / 'manifest.json', manifest)
    processes, logs = [], []
    start = time.monotonic()
    print('RUN', rt.rel(run), 'PID', os.getpid(), flush=True)
    try:
        for gpu, seed in enumerate(SEEDS):
            log = (run / f'worker_{gpu}.log').open('w')
            logs.append(log)
            process = subprocess.Popen([sys.executable, '-B', str(HERE / 'experiment.py'), 'worker',
                    '--run', str(run), '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT,
                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            processes.append(process)
            manifest['workers'].append(dict(gpu=gpu, seed=seed, pid=process.pid))
        rt.write(run / 'manifest.json', manifest)
        with (run / 'monitor.jsonl').open('w') as log:
            while True:
                codes = [p.poll() for p in processes]
                seconds = time.monotonic() - start
                log.write(json.dumps(dict(utc=rt.utc(), elapsed=seconds, exit_codes=codes, gpu=rt.telemetry())) + '\n')
                log.flush()
                assert all(code in (None, 0) for code in codes), codes
                if codes == [0] * 4:
                    break
                if seconds >= LIMIT_SECONDS:
                    raise TimeoutError('E042: limit oceny 30 min')
                time.sleep(5)
        assert manifest['code_hashes'] == rt.hashes(HERE)
        e35.validate_hashes(frozen['hashes'])
        assert not denied()
        manifest['status'] = 'completed'
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for row, process in zip(manifest['workers'], processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            row['exit_code'] = process.poll()
        for log in logs:
            log.close()
        manifest.update(elapsed_seconds=time.monotonic()-start, finished_utc=rt.utc())
        try:
            manifest['gpu_end'] = rt.telemetry()
        except Exception as error:
            manifest['gpu_end_error'] = repr(error)
        rt.write(run / 'manifest.json', manifest)
        print(manifest['status'], rt.rel(run), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'check', 'run', 'worker'))
    parser.add_argument('--cpu-run', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.cpu_run)
    elif args.action == 'check':
        check()
    elif args.action == 'run':
        launch()
    else:
        worker(args.run, args.gpu)
