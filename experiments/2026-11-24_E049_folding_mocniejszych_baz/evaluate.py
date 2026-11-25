"""E049: zamrożony folding E048 na CIFAR-100 i Tiny (OPIS.md).

Wejście: audytowanych osiem last.pt E048, A4096, dev i wcześniejsze benchmarki.
Wyjście: manifest, per_seed.csv, sparowane kontrasty i werdykt pod runs/E049/.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT/'runs/E049'
E48 = ROOT/'experiments/2026-11-24_E048_mocniejsze_modele_bazowe/train.py'
E31 = ROOT/'experiments/2026-11-02_E031_oficjalny_30_70/common.py'
E41 = ROOT/'experiments/2026-11-15_E041_kontrola_p_mac/geometry.py'
INDEX_FILE = ROOT/'runs/E031/20261104T212951Z_6c3d7356/indices.json'
BENCHMARK_PROVENANCE = ROOT/'runs/E035/20261107T183831Z_8b0b14e0/test_data.json'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load(E48, '_e049_train')
old = load(E31, '_e049_e31')
projector = load(E41, '_e049_geometry')
e29 = old.e29
read, write, digest, utc, require = train.read, train.write, train.digest, train.utc, train.require
DATASETS = ('cifar100', 'tiny')
SEEDS = (42, 43, 44, 45)
OFFSETS = np.array([2, -15, -14, 13, -1, -5, -6, 9, -12, -13, 17, -6])
VARIANTS = ('full_pre', 'full', 'b10_U', 'b10_L06', 'b10_Tproj',
            'b30_U', 'b30_T', 'b50_U', 'b50_T', 'b70_U', 'b70_T')
BENCHMARKS = {
    'cifar100': ('data/cifar100/cifar-100-python/test', '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864'),
    'tiny': ('data/tiny-imagenet/cache_val.npz', 'fd0c5fc8e45cff072f45f878b60a33d8f0e0af6340fc11d8f937e1c6c75bf07b'),
}


def source_hashes():
    # Reużywamy tylko plików kodu i indeksów; nie otwieramy benchmarków tutaj.
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=e29.UP, text=True).strip() == e29.COMMIT and
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=e29.UP, text=True).strip() == '', 'Zmieniony upstream')
    files = [Path(__file__), HERE/'OPIS.md', HERE/'URUCHOMIENIE.md', E31, E41, INDEX_FILE, BENCHMARK_PROVENANCE,
             *(Path(mod.__file__) for mod in (old.e30, e29, e29.bn, e29.d10, e29.legacy, old.dt)),
             *(p for p in e29.UP.rglob('*.py'))]
    files.extend(ROOT/p for p in train.hashes())
    files.extend(ROOT/p for p in ['experiments/2026-10-31_E024_optymalizacja_alokacji/folding.py',
                                 'experiments/2026-10-31_E024_optymalizacja_alokacji/data.py',
                                 'experiments/2026-10-24_E012_folding_referencja/folding.py'])
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def new_run(prefix):
    run = RUNS/f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True, exist_ok=False)
    return run


def uniform(geom, cap):
    denom = math.lcm(*geom.sizes)
    return next(w for i in range(denom-1, 0, -1)
                if geom.count(w := [i*n//denom for n in geom.sizes]) <= cap and min(w) >= 1)


def mild_linear(geom, cap):
    n = np.array(geom.sizes)
    depth = np.linspace(-1., 1., len(n))
    widths = lambda b: np.floor(n*(1-np.clip(b+.30*depth, 0., .90))).astype(np.int64)
    lo, hi = -.30, 1.30
    for _ in range(60):
        mid = (lo+hi)/2
        if geom.count(widths(mid)) > cap:
            lo = mid
        else:
            hi = mid
    return widths(hi)


def build_plans(base, classes):
    geom = e29.bn.Geometry(base)
    n = np.array(geom.sizes)
    require(np.array_equal(n, projector.SIZES) and tuple(geom.names) == projector.GEOMETRY_NAMES, 'Inne grupy')
    p0, m0 = map(int, projector.costs(n, classes))
    u = np.array(uniform(geom, 90*p0//100))
    caps = projector.costs(u, classes)
    projector.HIGH = n.copy()
    proposals = {'b10_L06': np.clip(np.rint(u + OFFSETS*n/100), projector.LOW, n).astype(np.int64),
                 'b10_Tproj': mild_linear(geom, int(caps[0]))}
    widths = {'full_pre': n, 'full': n, 'b10_U': u}
    solver = {}
    for key, proposal in proposals.items():
        value, evidence = projector.project(proposal, u, caps, classes)
        require(value is not None and projector.inside(projector.costs(value, classes), caps), f'{key}: niewykonalne pułapy')
        widths[key], solver[key] = value, evidence
    require(len({tuple(widths[v]) for v in ('b10_U','b10_L06','b10_Tproj')}) == 3, 'Nierozróżnialne profile b10')
    for budget in (30, 50, 70):
        u = geom.profile(round((1-budget/100)*p0), 0)
        t = geom.profile(geom.count(u), .30)
        require(999*geom.count(u) <= 1000*geom.count(t) <= 1000*geom.count(u), f'b{budget}: niedopasowanie P')
        widths[f'b{budget}_U'], widths[f'b{budget}_T'] = u, t
    require(tuple(widths) == VARIANTS, 'Błędna siatka E049')
    rows = {}
    for key, value in widths.items():
        p, mac = map(int, projector.costs(value, classes))
        require(p == geom.count(value) and np.all((np.asarray(value) >= 1) & (np.asarray(value) <= n)), f'{key}: niespójna geometria')
        if key.startswith('b10_'):
            require(1000*p <= 905*p0 and 1000*mac <= 905*m0, f'{key}: niewystarczająca redukcja obu kosztów')
        rows[key] = {'widths': list(map(int, value)), 'parameters': p, 'mac': mac}
    return {'profiles': rows, 'solver': solver}


def fold(base, key, plan):
    if key in ('full', 'full_pre'):
        return copy.deepcopy(base).eval().requires_grad_(False)
    model, diag = e29.official(base, .5, plan['widths'])
    with torch.no_grad():
        model.fc.bias.copy_(base.fc.bias)
    require([r['k'] for r in diag] == plan['widths'] and torch.equal(model.fc.bias, base.fc.bias), 'Adapter nie odtworzył profilu')
    return model


def validated_indices():
    indices = read(INDEX_FILE)
    for ds in DATASETS:
        split = train.DATA[ds].load_split()['splits']
        require(len(set(indices[ds]['A'])) == len(indices[ds]['A']) == 4096 and
                set(indices[ds]['A']) <= set(split['train']) and len(indices[ds]['dev']) == len(split['dev']) and
                set(indices[ds]['dev']) == set(split['dev']), 'Niezgodne A4096/dev')
    return {ds:indices[ds] for ds in DATASETS}


def check(smoke=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available(), 'CPU check wymaga ukrycia GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = read(train.CONFIG)
    cfg_hash = digest(train.CONFIG)
    run = new_run('cpu')
    print(f'CHECK={run.relative_to(ROOT)}', flush=True)
    m = {'status': 'running', 'utc': utc(), 'sources': source_hashes(), 'train_config_sha256': cfg_hash,
         'operator': e29.COMMIT, 'benchmark_opened': False, 'checks': []}
    write(run/'manifest.json', m)
    try:
        prior = read(BENCHMARK_PROVENANCE)['files']
        require(all(len(h) == 64 and prior.get(p) == h for p,h in BENCHMARKS.values()), 'Złe referencje hashy E035')
        validated_indices()
        if smoke is not None:
            require(read(smoke/'manifest.json')['status'] == 'completed' and
                    read(smoke/'manifest.json')['config_sha256'] == cfg_hash and
                    read(smoke/'audit.json')['status'] == 'passed', 'Niezgodny smoke E048')
        plans = {}
        for ds in DATASETS:
            spec = cfg['datasets'][ds]
            base = train.model_for(ds, cfg)
            if smoke is not None:
                path = smoke/ds/'seed_42/last.pt'
                summary = read(path.parent/'summary.json')
                require(digest(path) == summary['checkpoint_sha256'], 'Zmieniony smoke checkpoint')
                base.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'], strict=True)
            base.eval().requires_grad_(False)
            plans[ds] = build_plans(base, spec['classes'])
            for key in VARIANTS[1:]:
                plan = plans[ds]['profiles'][key]
                model = fold(base, key, plan)
                actual = train.size_cost(model, spec['input_size'], spec['classes'])
                require(actual == (plan['parameters'], plan['mac']), f'{ds}/{key}: niezgodne faktyczne P/MAC {actual}')
                m['checks'].append({'dataset': ds, 'variant': key, 'actual_p_mac': list(actual)})
                print(ds, key, actual, flush=True)
        write(run/'geometry.json', plans)
        m.update(status='passed', geometry_sha256=digest(run/'geometry.json'), smoke_checkpoint_check=smoke is not None)
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        m['finished_utc'] = utc()
        write(run/'manifest.json', m)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def proc_alive(pid, marker):
    try:
        cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        return os.fsencode(marker) in cmd
    except FileNotFoundError:
        return False


def wait_training(status):
    path = Path(status['training'])
    deadline = time.monotonic() + 25*3600
    while True:
        m = read(path/'manifest.json')
        require(m['mode'] == 'train' and m['config_sha256'] == status['train_config_sha256'], 'Inny protokół E048')
        if m['status'] == 'completed':
            a = read(path/'audit.json')
            require(a['status'] == 'passed' and a['validated_checkpoints'] == 8, 'Niepełny audyt E048')
            return a
        require(m['status'] == 'running' and proc_alive(m['launcher_pid'], str(path)), 'E048 nie jest aktywny')
        require(time.monotonic() < deadline, 'Limit oczekiwania na E048')
        time.sleep(30)


def load_model(source, record, ds, seed, cfg):
    path = source/ds/f'seed_{seed}/last.pt'
    require(digest(path) == record['sha256'], 'Zmieniony checkpoint E048')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta = payload['metadata']
    require(meta['config_sha256'] == record['config_sha256'] and meta['dataset'] == ds and meta['seed'] == seed and
            meta['epoch'] == cfg['epochs'] and not meta['smoke'], 'Zły checkpoint E048')
    base = train.model_for(ds, cfg)
    base.load_state_dict(payload['state_dict'], strict=True)
    return base.eval().requires_grad_(False)


def benchmark_data(ds, release):
    rel, expected = BENCHMARKS[ds]
    path = ROOT/rel
    require(release['files'][rel] == expected and digest(path) == expected, 'Zmieniony benchmark')
    if ds == 'cifar100':
        with path.open('rb') as f:
            data = pickle.load(f, encoding='latin1')
        x = e29.bn.normalize(torch.from_numpy(data['data'].reshape(-1,3,32,32)))
        y = torch.tensor(data['fine_labels'], dtype=torch.long)
    else:
        with np.load(path, allow_pickle=False) as data:
            x = old.e30.normalize(torch.from_numpy(data['images']).permute(0,3,1,2).float()/255)
            y = torch.tensor(data['labels'], dtype=torch.long)
    c = 100 if ds == 'cifar100' else 200
    require(len(x) == len(y) == 10000 and torch.equal(torch.bincount(y), torch.full((c,), 10000//c)), 'Błędne przykłady benchmarku')
    return x,y


def score(model, x, y, expected):
    model = model.cuda()
    z = e29.bn.predict(model, x)
    require(torch.isfinite(z).all() and state_hash(model) == expected, 'Zmiana stanu przy punktacji')
    correct = int((z.argmax(1) == y).sum())
    result = {'correct': correct, 'n': len(y), 'top1_pct': 100*correct/len(y), 'nll': float(F.cross_entropy(z, y))}
    model.cpu()
    return result


def worker(run, gpu):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1, 'Błędne przypisanie GPU')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = SEEDS[gpu]
    out = run/f'seed_{seed}'
    out.mkdir(exist_ok=False)
    m = read(run/'manifest.json')
    source = Path(m['training'])
    cfg = read(source/'config.json')
    frozen = read(run/'frozen.json')
    require(digest(run/'frozen.json') == m['frozen_sha256'] and digest(source/'config.json') == m['train_config_sha256'], 'Zmieniony plan')
    bank, records = {}, []
    for ds in DATASETS:
        base = load_model(source, frozen['checkpoints'][ds][str(seed)], ds, seed, cfg)
        base_state = state_hash(base)
        spec = cfg['datasets'][ds]
        a, x, y, aug_sha = old.data(ds, frozen['indices'][ds])
        bank[ds] = {}
        for key in VARIANTS:
            plan = frozen['geometry'][ds]['profiles'][key]
            model = fold(base, key, plan)
            require(train.size_cost(model, spec['input_size'], spec['classes']) == (plan['parameters'], plan['mac']), 'Zły koszt rzeczywistego modelu')
            if key != 'full_pre':
                repair = e29.calibrate(model.cuda(), a, 'CMA')
                require(repair['params_unchanged'] and repair['samples'] == 4096 and repair['batches'] == 32, 'Niespójna CMA')
            model_state = state_hash(model)
            row = {'dataset': ds, 'seed': seed, 'variant': key, 'parameters': plan['parameters'], 'mac': plan['mac'],
                   'calibration_views_sha256': aug_sha, 'state_sha256': model_state, 'dev': score(model, x, y, model_state)}
            bank[ds][key] = model
            records.append(row)
            print(ds, key, 'dev', row['dev']['top1_pct'], flush=True)
        require(state_hash(base) == base_state, 'Zmienione wejściowe wagi')
    write(out/'ready.json', {'seed': seed, 'gpu': gpu, 'pid': os.getpid(), 'records': records,
                            'frozen_sha256': m['frozen_sha256'], 'benchmark_opened': False})
    until = time.monotonic() + 1800
    path = run/'benchmark_release.json'
    while not path.is_file():
        require(time.monotonic() < until and proc_alive(m['launcher_pid'], str(run)), 'Brak bramki dev E049')
        time.sleep(2)
    release = read(path)
    require(release['passed'] is True and release['frozen_sha256'] == m['frozen_sha256'] and
            release['ready_sha256'][str(seed)] == digest(out/'ready.json'), 'Zmieniona bramka E049')
    for ds in DATASETS:
        x,y = benchmark_data(ds, release)
        for row in records:
            if row['dataset'] == ds:
                row['benchmark'] = score(bank[ds][row['variant']], x, y, row['state_sha256'])
                print(ds, row['variant'], 'benchmark', row['benchmark']['top1_pct'], flush=True)
        bank.pop(ds)
    write(out/'metrics.json', {'seed': seed, 'records': records, 'release_sha256': digest(path), 'finished_utc': utc()})


def release_benchmark(run, frozen_hash):
    ready = {}
    for seed in SEEDS:
        path = run/f'seed_{seed}/ready.json'
        row = read(path)
        require(row['seed'] == seed and row['gpu'] == seed-42 and not row['benchmark_opened'] and row['frozen_sha256'] == frozen_hash and
                [(r['dataset'],r['variant']) for r in row['records']] == [(d,k) for d in DATASETS for k in VARIANTS], 'Niepełny dev')
        ready[str(seed)] = digest(path)
    # Tu po raz pierwszy otwieramy pliki benchmarków, bez dobierania wynikami dev.
    files = {rel: digest(ROOT/rel) for rel, _ in BENCHMARKS.values()}
    require(files == dict(BENCHMARKS.values()), 'Niezgodne hashe referencji E035')
    write(run/'benchmark_release.json', {'passed': True, 'utc': utc(), 'frozen_sha256': frozen_hash,
                                        'ready_sha256': ready, 'files': files, 'benchmark_previously_used': True})


def audit(run, baseline):
    result = {'status': 'passed', 'benchmark_previously_used': True, 'baseline_audit': baseline, 'datasets': {}}
    notes = ['# E049 — wyniki na modelach E048', '',
             'Benchmark: oficjalny test CIFAR-100, oficjalny val Tiny. Obie próby wcześniej użyte w projekcie.',
             'Top‑1 i straty: średnie czterech sparowanych checkpointów. Full oznacza model pełny po CMA.', '']
    rows = []
    for seed in SEEDS:
        data = read(run/f'seed_{seed}/metrics.json')
        ready = read(run/f'seed_{seed}/ready.json')
        require(data['seed'] == ready['seed'] == seed and data['release_sha256'] == digest(run/'benchmark_release.json'), 'Niezgodne artefakty per seed')
        require(len(data['records']) == len(ready['records']) == len(DATASETS)*len(VARIANTS), 'Niepełna siatka')
        for row, before in zip(data['records'], ready['records'], strict=True):
            require({k:v for k,v in row.items() if k != 'benchmark'} == before and row['benchmark']['n'] == 10000,
                    'Benchmark nie użył zablokowanego stanu')
            rows.append(row)
    for ds in DATASETS:
        notes += [f'## {ds}', '', '| Profil | Top‑1 benchmark [%] | Strata do pełnego [pp] | Parametry | MAC/obraz |',
                  '|---|---:|---:|---:|---:|']
        subset = {key:[r for r in rows if r['dataset'] == ds and r['variant'] == key] for key in VARIANTS}
        subsets_by_split = {}
        for split in ('dev','benchmark'):
            scores = {key:[r[split]['top1_pct'] for r in group] for key,group in subset.items()}
            contrasts = {}
            for a,b in [('b10_L06','b10_U'),('b10_L06','b10_Tproj'),
                        *((f'b{d}_T',f'b{d}_U') for d in (30,50,70))]:
                diff = [x-y for x,y in zip(scores[a],scores[b],strict=True)]
                contrasts[f'{a}-{b}'] = {'mean_pp': statistics.mean(diff), 'sd_pp': statistics.stdev(diff),
                                          'per_seed_pp': diff, 'positive_pairs': sum(x>0 for x in diff)}
            subsets_by_split[split] = {'top1_mean_pct': {key:statistics.mean(values) for key,values in scores.items()},
                                       'loss_to_full_pp': {key:statistics.mean(x-y for x,y in zip(scores['full'],values,strict=True))
                                                           for key,values in scores.items()}, 'contrasts': contrasts}
        b = subsets_by_split['benchmark']
        for key,group in subset.items():
            loss = '—' if key == 'full_pre' else f'{b["loss_to_full_pp"][key]:.4f}'
            notes.append(f'| {key} | {b["top1_mean_pct"][key]:.4f} | {loss} | {group[0]["parameters"]:,} | {group[0]["mac"]:,} |')
        main = b['contrasts']['b10_L06-b10_U']
        loss = b['loss_to_full_pp']['b10_L06']
        hits = {key:sum(r['benchmark']['correct'] for r in group) for key,group in subset.items()}
        passed = (hits['b10_L06']-hits['b10_U'] >= 400 and main['positive_pairs'] >= 3 and
                  hits['full']-hits['b10_L06'] <= 1200)
        result['datasets'][ds] = {**subsets_by_split, 'passed': passed,
                                  'E048_quality_gate_met': baseline['results'][ds]['quality_gate_met']}
        notes += ['', f'**B10 L06−U: {main["mean_pp"]:+.4f} ± {main["sd_pp"]:.4f} pp** '
                  f'({main["positive_pairs"]}/4); strata L06 {loss:.4f} pp; kryterium E049: **{"TAK" if passed else "NIE"}**.', '']
    result['joint_gate_met'] = all(d['passed'] for d in result['datasets'].values())
    notes += [f'Łączna bramka obu zbiorów: **{"TAK" if result["joint_gate_met"] else "NIE"}**. '
              'Techniczny audyt nie oznacza spełnienia bramki trafności. Pełne kontrasty per seed w `audit.json`.']
    with (run/'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset','seed','variant','parameters','mac','split','correct','n','top1_pct','nll','state_sha256'])
        writer.writeheader()
        for row in rows:
            for split in ('dev','benchmark'):
                writer.writerow({**{k:row[k] for k in ('dataset','seed','variant','parameters','mac','state_sha256')},
                                 'split':split, **row[split]})
    (run/'WYNIKI.md').write_text('\n'.join(notes)+'\n')
    write(run/'audit.json', result)


def supervise(run):
    m = read(run/'manifest.json')
    procs = []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        m.update(launcher_pid=os.getpid(), status='waiting_e048')
        write(run/'manifest.json', m)
        baseline = wait_training(m)
        require(m['sources'] == source_hashes() and m['geometry_sha256'] == digest(run/'geometry.json'), 'Zmieniony kod lub geometria E049')
        m['gpus'] = train.require_idle_gpus()
        source = Path(m['training'])
        checkpoints = {}
        for ds in DATASETS:
            checkpoints[ds] = {}
            for seed in SEEDS:
                path = source/ds/f'seed_{seed}'
                summary = read(path/'summary.json')
                sha = digest(path/'last.pt')
                require(not summary['smoke'] and sha == summary['checkpoint_sha256'] and
                        summary['config_sha256'] == m['train_config_sha256'], 'Zmienione wagi E048')
                checkpoints[ds][str(seed)] = {'sha256':sha, 'config_sha256':m['train_config_sha256']}
        write(run/'frozen.json', {'geometry':read(run/'geometry.json'), 'checkpoints':checkpoints,
                                 'benchmark_expectations':BENCHMARKS, 'indices':validated_indices()})
        m.update(status='running', started_utc=utc(), frozen_sha256=digest(run/'frozen.json'), workers=[])
        write(run/'manifest.json', m)
        start = time.monotonic()
        for gpu,seed in enumerate(SEEDS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            with (run/'logs'/f'seed_{seed}.log').open('w') as log:
                p = subprocess.Popen([sys.executable,'-B',str(HERE/'evaluate.py'),'worker','--run-dir',str(run),'--gpu',str(gpu)],
                                     cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            procs.append(p)
            m['workers'].append({'seed':seed, 'gpu':gpu, 'pid':p.pid})
        write(run/'manifest.json', m)
        while True:
            codes = [p.poll() for p in procs]
            require(not any(c not in (None,0) for c in codes), f'Worker E049 failed: {codes}')
            if codes == [0]*4:
                break
            release = run/'benchmark_release.json'
            if not release.exists() and all((run/f'seed_{s}/ready.json').is_file() for s in SEEDS):
                release_benchmark(run, m['frozen_sha256'])
                m['benchmark_used'] = True
                write(run/'manifest.json', m)
            require(time.monotonic()-start < 1800, 'Przekroczono limit E049')
            time.sleep(2)
        audit(run, baseline)
        m.update(status='completed', wall_seconds=time.monotonic()-start)
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()
        m.update(finished_utc=utc(), exit_codes=[p.returncode for p in procs])
        write(run/'manifest.json', m)


def queue(training, checked):
    c = read(checked/'manifest.json')
    config_hash = read(training/'manifest.json')['config_sha256']
    require(read(training/'manifest.json')['mode'] == 'train' and training.parent == train.RUNS and
            c['status'] == 'passed' and c['sources'] == source_hashes() and c['train_config_sha256'] == config_hash and
            digest(checked/'geometry.json') == c['geometry_sha256'], 'Niezgodne wejście E049')
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3',
            'Wymagane Conda attention i jawne przypisanie czterech GPU')
    run = new_run('eval')
    (run/'logs').mkdir()
    for rel in c['sources']:
        dest = run/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    shutil.copy2(checked/'geometry.json', run/'geometry.json')
    write(run/'manifest.json', {'status':'queued', 'utc':utc(), 'training':str(training), 'cpu_check':str(checked),
                                'train_config_sha256':config_hash, 'geometry_sha256':c['geometry_sha256'],
                                'sources':c['sources'], 'benchmark_used':False, 'operator':e29.COMMIT,
                                'python':sys.version, 'torch':torch.__version__, 'cuda':torch.version.cuda})
    with (run/'launch.log').open('w') as log:
        pid = subprocess.Popen([sys.executable,'-B',str(HERE/'evaluate.py'),'supervise','--run-dir',str(run)], cwd=ROOT,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True).pid
    print(f'RUN={run.relative_to(ROOT)} PID={pid}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check','queue','supervise','worker'])
    parser.add_argument('--check', type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == 'check':
        check(args.smoke_run.resolve() if args.smoke_run else None)
    elif args.action == 'queue':
        queue(args.training.resolve(), args.check.resolve())
    elif args.action == 'supervise':
        supervise(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve(), args.gpu)
