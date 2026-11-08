"""E037: kontrola TILT vs UNIFORM przy dopasowanym MAC na dev; OPIS.md.
Wejście: plany/checkpointy E035; wyjście profile U', metryki dev i analysis, bez treningu.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DENIED = []
TEST_MARKERS = ('/test_batch', '/cifar-100-python/test', '/cache_val.npz', '/test_data.json')


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = str(Path(os.fsdecode(args[0])).resolve())
        if any(marker in path for marker in TEST_MARKERS):
            DENIED.append(path)
            raise RuntimeError('E037: dostęp do testu zabroniony: ' + path)


sys.addaudithook(guard)
import numpy as np
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e35 = load(ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna/experiment.py', '_e037_e35')
e31, e34, e29, bn = e35.e31, e35.e34, e35.e29, e35.bn
read, write, digest = e35.read, e35.write, e35.digest
PAIRS, TAGS, SEEDS, MODES = e35.PAIRS, e35.TAGS, e35.SEEDS, e35.MODES
R35 = ROOT / 'runs/E035/20261107T183831Z_8b0b14e0'


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def geometry(pair, model):
    return e34.Geometry(model) if pair.startswith('preact') else bn.Geometry(model)


def fold_widths(pair, base, widths):
    if pair.startswith('preact'):
        model, diag = e34.official(base, 0.5, widths)
    else:
        model, diag = e29.official(base, 0.5, widths)
        model.fc.bias.data.copy_(base.fc.bias)
    model.eval().requires_grad_(False)
    assert [d['k'] for d in diag] == list(widths)
    assert sum(p.numel() for p in model.parameters()) == geometry(pair, base).count(widths)
    bias = 'linear' if pair.startswith('preact') else 'fc'
    assert torch.equal(getattr(model, bias).bias, getattr(base, bias).bias)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return model, diag


def mac(model, pair):
    size = 64 if pair.endswith('tiny') else 32
    total = 0
    hooks = []

    def hook(m, inputs, output):
        nonlocal total
        if isinstance(m, torch.nn.Conv2d):
            total += output.numel() * m.weight[0].numel()
        else:
            total += output.numel() * inputs[0].shape[-1]

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, size, size))
    for h in hooks:
        h.remove()
    assert torch.isfinite(out).all()
    return int(total)


def mac_plan(pair, base, geo):
    size = 64 if pair.endswith('tiny') else 32
    shapes = {}
    hooks = []

    def record(name):
        def hook(m, inputs, output):
            shapes[name] = tuple(output.shape[2:]) if isinstance(m, torch.nn.Conv2d) else (inputs[0].shape[-1],)
        return hook

    for name, module in base.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(record(name)))
    with torch.inference_mode():
        out = base(torch.zeros(1, 3, size, size))
    for hook in hooks:
        hook.remove()
    assert torch.isfinite(out).all()

    def group(pname, axis):
        if hasattr(geo, 'axis_group'):
            return geo.axis_group.get((pname, axis))
        axes = getattr(geo, 'axes', {}).get(pname)
        return axes[axis] if axes is not None else None

    plan = []
    for name, module in base.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pname = name + '.weight'
            plan.append(('conv', name, group(pname, 0), group(pname, 1),
                         module.weight.shape[0], module.weight.shape[1],
                         module.weight.shape[2], module.weight.shape[3],
                         shapes[name][0], shapes[name][1], module.groups))
        elif isinstance(module, torch.nn.Linear):
            pname = name + '.weight'
            plan.append(('linear', name, group(pname, 0), group(pname, 1),
                         module.weight.shape[0], module.weight.shape[1]))

    def compute(widths):
        assignment = dict(zip(geo.names, widths))
        total = 0
        for item in plan:
            if item[0] == 'conv':
                _, _, out_g, in_g, out0, in0, kh, kw, h, w, groups = item
                out_c = assignment.get(out_g, out0)
                in_c = assignment.get(in_g, in0)
                total += h * w * out_c * (in_c // groups) * kh * kw
            else:
                _, _, out_g, in_g, out0, in0 = item
                out_c = assignment.get(out_g, out0)
                in_c = assignment.get(in_g, in0)
                total += out_c * in_c
        return int(total)

    return compute


def solve_up(base, pair, mac_target, tag):
    geo = geometry(pair, base)
    compute = mac_plan(pair, base, geo)
    max_widths = [int(0.9 * n) for n in geo.sizes]
    max_params = geo.count(max_widths)
    max_mac = compute(max_widths)
    if max_mac <= mac_target:
        return max_params, max_widths, max_mac, False
    lo, hi = 1, max_params
    best = None
    for _ in range(32):
        mid = (lo + hi) // 2
        widths = geo.profile(mid, 0)
        params = geo.count(widths)
        value = compute(widths)
        if value <= mac_target:
            best = (params, widths, value, True)
            lo = mid + 1
        else:
            hi = mid - 1
    assert best is not None, (pair, tag)
    return best


def freeze():
    assert not torch.cuda.is_available()
    ref = read(R35 / 'frozen.json')
    plans = {pair: {} for pair in PAIRS}
    hashes = {rel(R35 / 'frozen.json'): digest(R35 / 'frozen.json')}
    for pair in PAIRS:
        base = e35.model_source(pair, 42)
        geo = geometry(pair, base)
        for tag in TAGS:
            plan = ref['plans'][pair][tag]
            original_t, _ = fold_widths(pair, base, plan['T'])
            original_u, _ = fold_widths(pair, base, plan['U'])
            mac_t = mac(original_t, pair)
            mac_u = mac(original_u, pair)
            params, widths, mac_up, matched = solve_up(base, pair, mac_t, tag)
            assert mac_up <= mac_t, (pair, tag, mac_up, mac_t)
            assert params >= plan['params']['T'], (pair, tag)
            plans[pair][tag] = dict(T=plan['T'], U=plan['U'], Uprime=widths,
                                    target=plan['params']['U'], T_params=plan['params']['T'],
                                    Uprime_params=params, T_mac=mac_t, U_mac=mac_u,
                                    Uprime_mac=mac_up, ratio=plan['ratio'],
                                    mac_matched=matched, mac_gap=mac_t - mac_up)
        for seed in SEEDS:
            cp = e35.checkpoint(pair, seed)
            hashes[rel(cp)] = digest(cp)
    for pair in PAIRS:
        for seed in SEEDS:
            for tag in TAGS:
                p = R35 / f'dev/seed_{seed}/{pair}/{tag}/summary.json'
                hashes[rel(p)] = digest(p)
            p = R35 / f'dev/seed_{seed}/{pair}/controls.json'
            hashes[rel(p)] = digest(p)
    # Usuń placeholdery None (gdyby któraś ścieżka nie istniała).
    hashes = {k: v for k, v in hashes.items() if v is not None}
    for path, sha in hashes.items():
        assert digest(ROOT / path) == sha, path
    obj = dict(plans=plans, indices=ref['indices'], references=ref['references'],
               e35_frozen_sha256=digest(R35 / 'frozen.json'), hashes=hashes, created_utc=utc())
    write(HERE / 'frozen.json', obj)
    print('FREEZE PASS', len(PAIRS) * len(TAGS), 'profiles', flush=True)


def check():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    frozen = read(HERE / 'frozen.json')
    e35.validate_hashes(frozen['hashes'])
    rows = []
    for pair in PAIRS:
        base = e35.model_source(pair, 42)
        for tag in TAGS:
            plan = frozen['plans'][pair][tag]
            t, _ = fold_widths(pair, base, plan['T'])
            u, _ = fold_widths(pair, base, plan['U'])
            up, diag = fold_widths(pair, base, plan['Uprime'])
            assert mac(up, pair) <= mac(t, pair)
            assert sum(p.numel() for p in up.parameters()) == plan['Uprime_params']
            assert len(diag) in (12, 13)
            rows.append(dict(pair=pair, tag=tag, T_mac=mac(t, pair), U_mac=mac(u, pair),
                             Uprime_mac=mac(up, pair), Uprime_params=plan['Uprime_params'],
                             T_params=plan['T_params']))
            print('CPU', pair, tag, rows[-1], flush=True)
    assert not DENIED
    write(HERE / 'check.json', dict(status='passed', utc=utc(), rows=rows,
          frozen_sha256=digest(HERE / 'frozen.json'), code_sha256=digest(HERE / 'experiment.py'),
          data_open_attempts=DENIED))
    print('CPU PASS', len(rows), flush=True)


def worker(run, gpu):
    seed = 42 + gpu
    assert torch.cuda.device_count() == 1 and os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed); np.random.seed(seed)
    frozen = read(run / 'frozen.json')
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', pid=os.getpid(), gpu=gpu, seed=seed, started_utc=utc(), completed=[])
    write(out / 'status.json', status)
    try:
        for pair in PAIRS:
            arch, ds = pair.split('_')
            a, x, y, views = e31.data(ds, frozen['indices'][ds])
            base = e35.model_source(pair, seed)
            controls = {}
            control_evidence = {}
            for mode in MODES:
                controls[mode], control_evidence[mode] = e35.evaluate(pair, base, a, x, y, mode)
            write(out / pair / 'controls.json', dict(metrics=controls, evidence=control_evidence, views_sha256=views))
            for tag in TAGS:
                plan = frozen['plans'][pair][tag]
                folded, diag = fold_widths(pair, base, plan['Uprime'])
                metrics = {}; evidence = {}
                for mode in MODES:
                    metrics[mode], evidence[mode] = e35.evaluate(pair, folded, a, x, y, mode)
                write(out / pair / tag / 'summary.json', dict(status='completed', pair=pair, tag=tag, seed=seed,
                    gpu=gpu, phase='dev', plan=plan, metrics=metrics, evidence=evidence, clusters=diag,
                    views_sha256=views, checkpoint_sha256=digest(e35.checkpoint(pair, seed)),
                    bias_preserved=True, controls_sha256=digest(out / pair / 'controls.json'),
                    official_test_used=False))
                status['completed'].append(pair + '/' + tag)
                write(out / 'status.json', status)
                print(seed, pair, tag, {k: round(metrics[k]['top1'] * 100, 3) for k in MODES}, flush=True)
                del folded
            del a, x, y, base
        assert not DENIED
        status.update(status='completed', finished_utc=utc(), data_open_attempts=DENIED)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc(), data_open_attempts=DENIED)
        raise
    finally:
        write(out / 'status.json', status)


def telemetry():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)


def gpu_info():
    text = telemetry()
    assert len(text.strip().splitlines()) == 4
    for i, line in enumerate(text.strip().splitlines()):
        r = [v.strip() for v in line.split(',')]
        assert int(r[0]) == i and int(r[2]) < 256 and int(r[3]) == 0, text
    return text


def run():
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3'
    assert torch.cuda.device_count() == 4
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    checked = read(HERE / 'check.json')
    assert checked['status'] == 'passed' and checked['frozen_sha256'] == digest(HERE / 'frozen.json')
    assert checked['code_sha256'] == digest(HERE / 'experiment.py')
    frozen = read(HERE / 'frozen.json')
    e35.validate_hashes(frozen['hashes'])
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(4)
    path = ROOT / 'runs/E037' / name
    (path / 'source').mkdir(parents=True)
    for p in HERE.iterdir():
        if p.suffix in ('.py', '.md', '.json'):
            shutil.copy2(p, path / 'source' / p.name)
    shutil.copy2(HERE / 'frozen.json', path / 'frozen.json')
    manifest = dict(status='running', pid=os.getpid(), started_utc=utc(), run=rel(path), limit_seconds=5400,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    python=platform.python_version(), executable=sys.executable, torch=torch.__version__,
                    cuda=torch.version.cuda, gpu_start=gpu_info(),
                    settings=dict(phase='dev', modes=list(MODES), seeds=list(SEEDS),
                                  batch=128, samples=5000, architecture='E035 five pairs'),
                    code_hashes={rel(p): digest(p) for p in HERE.glob('*.py')},
                    frozen_sha256=digest(HERE / 'frozen.json'), workers=[])
    (path / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    processes = []; logs = []; start = time.monotonic()
    print('RUN', path, 'PID', os.getpid(), flush=True)
    try:
        for gpu in range(4):
            log = (path / f'worker_{gpu}.log').open('w'); logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            p = subprocess.Popen([sys.executable, '-B', str(HERE / 'experiment.py'), 'worker',
                                  '--run', str(path), '--gpu', str(gpu)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p)
            manifest['workers'].append(dict(gpu=gpu, seed=42 + gpu, pid=p.pid))
        write(path / 'manifest.json', manifest)
        with (path / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in processes]
                monitor.write(json.dumps(dict(utc=utc(), elapsed=time.monotonic() - start, exit_codes=codes,
                                              gpu=telemetry())) + '\n'); monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if all(c == 0 for c in codes):
                    break
                if time.monotonic() - start > 5400:
                    raise TimeoutError('limit90min')
                time.sleep(10)
        e35.validate_hashes(manifest['code_hashes'])
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
        manifest.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - start, gpu_end=telemetry())
        write(path / 'manifest.json', manifest)
        print(manifest['status'], rel(path), flush=True)


def analyze(run):
    run = Path(run)
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed' and [w['exit_code'] for w in m['workers']] == [0] * 4
    frozen = read(run / 'frozen.json')
    e35.validate_hashes(frozen['hashes'])
    rows = []; controls = []
    for pair in PAIRS:
        for seed in SEEDS:
            c = read(run / f'seed_{seed}/{pair}/controls.json')
            for mode in MODES:
                controls.append(dict(pair=pair, seed=seed, mode=mode, **c['metrics'][mode]))
            for tag in TAGS:
                s = read(run / f'seed_{seed}/{pair}/{tag}/summary.json')
                assert s['status'] == 'completed' and s['plan'] == frozen['plans'][pair][tag]
                ref = frozen['references'][pair][tag][str(seed)]
                for mode in MODES:
                    rows.append(dict(pair=pair, tag=tag, seed=seed, mode=mode,
                                     U=ref['metrics']['U'][mode]['top1'],
                                     T=ref['metrics']['T'][mode]['top1'],
                                     Up=s['metrics'][mode]['top1'],
                                     U_nll=ref['metrics']['U'][mode]['nll'],
                                     T_nll=ref['metrics']['T'][mode]['nll'],
                                     Up_nll=s['metrics'][mode]['nll']))
    assert len(rows) == 180 and len(controls) == 60
    allocation = []
    for pair in PAIRS:
        for tag in TAGS:
            for mode in MODES:
                vals = [r for r in rows if r['pair'] == pair and r['tag'] == tag and r['mode'] == mode]
                for a, b in (('T', 'Up'), ('T', 'U'), ('Up', 'U')):
                    d = [100 * (r[a] - r[b]) for r in vals]
                    st = dict(pair=pair, tag=tag, mode=mode, contrast=f'{a}-{b}',
                              mean=float(np.mean(d)), sd=float(np.std(d, ddof=1)),
                              positive=sum(x > 0 for x in d), per_seed=d)
                    if mode == 'CMA' and a == 'T' and b == 'Up':
                        st['gate'] = (st['mean'] >= 2.0 and st['positive'] >= 3) if tag in ('b50', 'b70') else (st['mean'] > 0 and st['positive'] >= 3)
                    allocation.append(st)
    with_seed = [r for r in allocation if r['contrast'] == 'T-Up' and r['mode'] == 'CMA']
    averages = []
    for pair in PAIRS:
        for tag in TAGS:
            for mode in MODES:
                vals = [r for r in rows if r['pair'] == pair and r['tag'] == tag and r['mode'] == mode]
                for name, key, nkey in (('U', 'U', 'U_nll'), ('T', 'T', 'T_nll'), ('Up', 'Up', 'Up_nll')):
                    averages.append(dict(pair=pair, tag=tag, mode=mode, name=name,
                                         top1=float(np.mean([r[key] for r in vals])),
                                         top1_sd=float(np.std([r[key] for r in vals], ddof=1)),
                                         nll=float(np.mean([r[nkey] for r in vals])),
                                         nll_sd=float(np.std([r[nkey] for r in vals], ddof=1))))
    result = dict(allocation=allocation, averages=averages, rows=rows, controls=controls,
                  primary=with_seed, primary_pass=all(r.get('gate', False) for r in with_seed),
                  note='dev only; U/T reused E035; Uprime evaluated on existing checkpoints',
                  source_hashes={str(p.relative_to(run)): digest(p) for p in run.glob('seed_*/*/*/summary.json')},
                  wall_seconds=m['elapsed_seconds'], rows_n=len(rows), controls_n=len(controls))
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(result, indent=1) + '\n')
    import csv
    with (out / 'per_seed.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('ANALYSIS', out, 'primary_pass', result['primary_pass'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'check', 'run', 'worker', 'analyze'])
    parser.add_argument('--run', type=Path); parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'check': check()
    elif args.action == 'run': run()
    elif args.action == 'analyze': analyze(args.run)
    else: worker(args.run, args.gpu)
