"""E036: syntetyczny audyt kosztu wag/profili E035, bez danych i treningu.
Wejście frozen/checkpointy; wyjście czasy, MAC, pamięć i manifest. OPIS.md.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = str(ROOT / 'data') + '/'
DENIED = []


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = str(Path(os.fsdecode(args[0])).resolve())
        if path.startswith(DATA):
            DENIED.append(path)
            raise RuntimeError('E036: dostęp do danych zabroniony: ' + path)


sys.addaudithook(guard)
import numpy as np
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e35 = load(ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna/experiment.py', '_e036_e35')
R35 = ROOT / 'runs/E035/20261107T183831Z_8b0b14e0'
read, write, digest = e35.read, e35.write, e35.digest
PAIRS, TAGS = e35.PAIRS, e35.TAGS


def forbidden(*args, **kwargs):
    raise RuntimeError('E036: helper danych/oceny jest wyłączony')


# Nie wywołujemy wrapperów, które mogłyby budować loadery.
for module in (e35, e35.e31, e35.e34, e35.e29, e35.e30, e35.bn, e35.e31.d10, e35.e31.dt):
    for name in ('data', 'independent_data', 'prepare_test', 'evaluate', 'calibrate',
                 'Images', 'load_split', 'load_data', 'make_loaders', 'predict'):
        if hasattr(module, name):
            setattr(module, name, forbidden)
torch.utils.data.DataLoader = forbidden


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def synthetic(size, batch):
    generator = torch.Generator(device='cpu').manual_seed(360000 + size + batch)
    return torch.randn(batch, 3, size, size, generator=generator, dtype=torch.float32)


def validate(mapping):
    for path, sha in mapping.items():
        assert digest(ROOT / path) == sha, path


def freeze():
    assert not torch.cuda.is_available()
    frozen = read(R35 / 'frozen.json')
    # Nie hashujemy ponownie danych; kopiujemy tylko model/kod/raport provenance.
    hashes = {p: h for p, h in frozen['provenance'].items() if not p.startswith('data/')}
    for p in (R35 / 'frozen.json', Path(e35.__file__), HERE / 'OPIS.md'):
        hashes[rel(p)] = digest(p)
    refs = {}
    for seed in range(42, 46):
        refs[str(seed)] = {}
        for pair in PAIRS:
            parent = R35 / f'test/seed_{seed}/{pair}'
            cp = parent / 'controls.json'
            c = read(cp)
            hashes[rel(cp)] = digest(cp)
            row = {'teacher': {'state_sha256': c['evidence']['PRE']['state_sha256'],
                               'quality_CMA': c['metrics']['CMA']}}
            for tag in TAGS:
                p = parent / tag / 'summary.json'
                s = read(p)
                hashes[rel(p)] = digest(p)
                row[tag] = {k: {'state_sha256': s['evidence'][k]['PRE']['state_sha256'],
                               'quality_CMA': s['metrics'][k]['CMA'],
                               'clusters': s['clusters'][k]} for k in ('U', 'T')}
            refs[str(seed)][pair] = row
    validate(hashes)
    obj = dict(plans=frozen['plans'], references=refs, hashes=hashes,
               e035_frozen_sha256=digest(R35 / 'frozen.json'), created_utc=utc())
    write(HERE / 'frozen.json', obj)
    print('FREEZE PASS:20 checkpointów,120 profili, zero danych', flush=True)


def manual_cost(model, size, preact):
    """Niezależny graf geometrii: wyłącznie kształty wag i wzór konwolucji."""
    layers = {}
    bn_elements = 0
    relu_elements = 0

    def conv(name, shape):
        m = model.get_submodule(name)
        c, h, w = shape
        co, ci, kh, kw = m.weight.shape
        assert ci * m.groups == c
        ho = (h + 2*m.padding[0] - m.dilation[0]*(kh-1) - 1)//m.stride[0] + 1
        wo = (w + 2*m.padding[1] - m.dilation[1]*(kw-1) - 1)//m.stride[1] + 1
        layers[name] = dict(input=[1, c, h, w], output=[1, co, ho, wo],
                            mac=int(ho*wo*co*ci*kh*kw))
        return co, ho, wo

    shape = conv('conv1', (3, size, size))
    if not preact:
        bn_elements += int(np.prod(shape)); relu_elements += int(np.prod(shape))
        pool = getattr(model, 'maxpool', None)
        if isinstance(pool, torch.nn.MaxPool2d):
            c, h, w = shape
            shape = (c, (h+2*pool.padding-pool.kernel_size)//pool.stride+1,
                     (w+2*pool.padding-pool.kernel_size)//pool.stride+1)
    for stage in range(1, 5):
        for block in range(2):
            prefix = f'layer{stage}.{block}'
            mod = model.get_submodule(prefix)
            original = shape
            if preact:
                bn_elements += int(np.prod(shape)); relu_elements += int(np.prod(shape))
            shape = conv(prefix+'.conv1', shape)
            bn_elements += int(np.prod(shape)); relu_elements += int(np.prod(shape))
            shape = conv(prefix+'.conv2', shape)
            if not preact:
                bn_elements += int(np.prod(shape)); relu_elements += int(np.prod(shape))
            skip = getattr(mod, 'shortcut', getattr(mod, 'downsample', None))
            shortcut_shape = original
            if skip is not None:
                for name, layer in skip.named_children():
                    if isinstance(layer, torch.nn.Conv2d):
                        key = 'shortcut' if hasattr(mod, 'shortcut') else 'downsample'
                        shortcut_shape = conv(prefix+'.'+key+'.'+name, shortcut_shape)
                    elif isinstance(layer, torch.nn.BatchNorm2d):
                        bn_elements += int(np.prod(shortcut_shape))
            assert shape == shortcut_shape, (prefix, shape, shortcut_shape)
    if preact:
        bn_elements += int(np.prod(shape)); relu_elements += int(np.prod(shape))
    name = 'linear' if preact else 'fc'
    co, ci = model.get_submodule(name).weight.shape
    assert ci == shape[0]
    layers[name] = dict(input=[1, ci], output=[1, co], mac=int(ci*co))
    return layers, bn_elements, relu_elements


def cost(model, size, preact):
    manual, bn_count, relu_count = manual_cost(model, size, preact)
    observed = {}
    observed_bn = []
    hooks = []

    def hook(name):
        def record(m, inputs, output):
            if isinstance(m, torch.nn.Conv2d):
                value = output.numel() * m.weight[0].numel()
            else:
                value = output.numel() * inputs[0].shape[-1]
            observed[name] = dict(input=list(inputs[0].shape), output=list(output.shape), mac=value)
        return record

    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook(name)))
        elif isinstance(m, torch.nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(lambda m, i, o: observed_bn.append(o.numel())))
    with torch.inference_mode():
        output = model(synthetic(size, 1))
    for h in hooks:
        h.remove()
    assert torch.isfinite(output).all()
    assert observed == manual, (observed, manual)
    assert sum(observed_bn) == bn_count
    return dict(params=sum(p.numel() for p in model.parameters()),
                mac=sum(x['mac'] for x in manual.values()), layers=manual,
                bn_elements=bn_count, relu_elements=relu_count, independent_mac_check=True)


def gpu_processes():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    return [line.strip() for line in text.splitlines() if line.strip()]


def check():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    frozen = read(HERE/'frozen.json')
    validate(frozen['hashes'])
    rows = []
    for pair in PAIRS:
        base = e35.model_source(pair, 42)
        for name in ['teacher'] + [t+'_'+p for t in TAGS for p in ('U', 'T')]:
            if name == 'teacher':
                model = base
                ref = frozen['references']['42'][pair]['teacher']
            else:
                model, diag = e35.fold(pair, base, frozen['plans'][pair][name[:3]], name[-1])
                ref = frozen['references']['42'][pair][name[:3]][name[-1]]
                assert diag == ref['clusters']
            assert e35.state_hash(model) == ref['state_sha256']
            stats = cost(model, 64 if pair.endswith('tiny') else 32, pair.startswith('preact'))
            rows.append(dict(pair=pair, name=name, mac=stats['mac'], params=stats['params']))
            print('CPU', pair, name, stats['mac'], flush=True)
    assert not DENIED
    write(HERE/'check.json', dict(status='passed', utc=utc(), checks=rows,
          frozen_sha256=digest(HERE/'frozen.json'), code_sha256=digest(HERE/'benchmark.py'), data_open_attempts=DENIED))
    print('CPU PASS', len(rows), flush=True)


def telemetry():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)


def measure(cpu_model, pair, size, batch):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    empty = torch.cuda.memory_allocated()
    assert empty == 0, empty
    model = cpu_model.to('cuda')
    if pair.startswith('preact'):
        model.normalize.mu = model.normalize.mu.to('cuda')
        model.normalize.std = model.normalize.std.to('cuda')
    model.eval().requires_grad_(False)
    xcpu = synthetic(size, batch)
    input_hash = hashlib.sha256(xcpu.numpy().tobytes()).hexdigest()
    x = xcpu.to('cuda')
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(50):
            y = model(x)
            del y
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(200):
            start.record()
            y = model(x)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
            del y
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    assert torch.cuda.memory_allocated() == baseline
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    model.cpu()
    if pair.startswith('preact'):
        model.normalize.mu = model.normalize.mu.cpu()
        model.normalize.std = model.normalize.std.cpu()
    del x, xcpu, start, end
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == 0
    return dict(times_ms=timings, input_sha256=input_hash, batch=batch,
                baseline_allocated=baseline, peak_allocated=peak, delta_allocated=peak-baseline,
                baseline_reserved=reserved, peak_reserved=peak_reserved, empty_allocated=empty)


def worker(run, gpu):
    seed = 42 + gpu
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu) and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    frozen = read(run / 'frozen.json')
    out = run / f'seed_{seed}'
    status = dict(status='running', pid=os.getpid(), gpu=gpu, seed=seed, started_utc=utc(), completed=[])
    write(out / 'status.json', status)
    try:
        for pair in PAIRS:
            size = 64 if pair.endswith('tiny') else 32
            ref = frozen['references'][str(seed)][pair]
            base = e35.model_source(pair, seed)
            models = {'teacher': base}
            for tag in TAGS:
                for profile in ('U', 'T'):
                    model, diag = e35.fold(pair, base, frozen['plans'][pair][tag], profile)
                    assert diag == ref[tag][profile]['clusters']
                    models[tag+'_'+profile] = model
            costs = {}
            for name, model in models.items():
                expected = ref['teacher'] if name == 'teacher' else ref[name[:3]][name[-1]]
                state = e35.state_hash(model)
                assert state == expected['state_sha256'], (pair, name, 'PRE state mismatch')
                costs[name] = cost(model, size, pair.startswith('preact'))
                costs[name]['state_sha256'] = state
            write(out / pair / 'costs.json', costs)
            sequence = 0
            for batch in (1, 128):
                for series in range(3):
                    order = ('U', 'T') if (gpu+series) % 2 == 0 else ('T', 'U')
                    names = ['teacher'] + [tag+'_'+p for tag in TAGS for p in order]
                    for name in names:
                        before = utc()
                        result = measure(models[name], pair, size, batch)
                        assert e35.state_hash(models[name]) == costs[name]['state_sha256']
                        result.update(pair=pair, seed=seed, gpu=gpu, name=name, series=series,
                                      sequence=sequence, started_utc=before, finished_utc=utc(), state_unchanged=True)
                        write(out / pair / f'{name}_bs{batch}_s{series}.json', result)
                        sequence += 1
                        status['completed'].append(f'{pair}/{name}/bs{batch}/s{series}')
                        write(out / 'status.json', status)
                        print(seed, pair, name, batch, series, 'median_ms', float(np.median(result['times_ms'])), flush=True)
            del models, base, model
        assert not DENIED
        status.update(status='completed', finished_utc=utc(), data_open_attempts=DENIED)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc(), data_open_attempts=DENIED)
        raise
    finally:
        write(out / 'status.json', status)


def run():
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3'
    assert torch.cuda.device_count() == 4 and not gpu_processes()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Commit przed runem'
    frozen = read(HERE / 'frozen.json')
    validate(frozen['hashes'])
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + os.urandom(4).hex()
    path = ROOT / 'runs/E036' / name
    (path / 'source').mkdir(parents=True)
    for p in HERE.iterdir():
        if p.suffix in ('.py', '.md', '.json'):
            shutil.copy2(p, path / 'source' / p.name)
    shutil.copy2(HERE / 'frozen.json', path / 'frozen.json')
    manifest = dict(status='running', pid=os.getpid(), started_utc=utc(), run=rel(path), limit_seconds=5400,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    python=platform.python_version(), executable=sys.executable, torch=torch.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu_start=telemetry(),
                    settings=dict(precision='FP32', tf32=False, cudnn_benchmark=False, cudnn_deterministic=True,
                                  deterministic_algorithms=True, batch=[1,128], warmup=50, samples=200, series=3,
                                  threads=2, cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
                    code_hashes={rel(p):digest(p) for p in HERE.glob('*.py')}, frozen_sha256=digest(HERE/'frozen.json'),
                    workers=[])
    (path / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    processes = []
    logs = []
    start = time.monotonic()
    print('RUN', path, 'PID', os.getpid(), flush=True)
    try:
        for gpu in range(4):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log = (path / f'worker_{gpu}.log').open('w'); logs.append(log)
            p = subprocess.Popen([sys.executable, '-B', str(HERE/'benchmark.py'), 'worker', '--run', str(path), '--gpu', str(gpu)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p)
            manifest['workers'].append(dict(gpu=gpu, seed=42+gpu, pid=p.pid))
        write(path/'manifest.json', manifest)
        with (path/'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in processes]
                apps = gpu_processes()
                foreign = [line for line in apps if int(line.split(',')[0]) not in [p.pid for p in processes]]
                monitor.write(json.dumps(dict(utc=utc(), elapsed=time.monotonic()-start, exit_codes=codes,
                                              processes=apps, gpu=telemetry()))+'\n'); monitor.flush()
                assert not foreign, ('Obcy proces GPU', foreign)
                assert all(c in (None, 0) for c in codes), codes
                if all(c == 0 for c in codes):
                    break
                if time.monotonic()-start > 5400:
                    raise TimeoutError('Limit90min')
                time.sleep(10)
        validate(frozen['hashes']); validate(manifest['code_hashes'])
        manifest['status'] = 'completed'
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        raise
    finally:
        for row, p in zip(manifest['workers'], processes):
            row['exit_code'] = p.poll()
        for log in logs:
            log.close()
        manifest.update(finished_utc=utc(), elapsed_seconds=time.monotonic()-start, gpu_end=telemetry())
        write(path/'manifest.json', manifest)
        print(manifest['status'], rel(path), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'check', 'run', 'worker'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'check':
        check()
    elif args.command == 'run':
        run()
    else:
        worker(args.run, args.gpu)
