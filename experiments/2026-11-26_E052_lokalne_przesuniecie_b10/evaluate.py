"""E052 (OPIS.md): CPU geometria, selekcja train i jednorazowy dev przy b10.

Wejście: geometria/referencje E051, checkpointy E048, train i jego podziały.
Wyjście: zamrożone indeksy/pula/wybór, manifest, logi i audyt w runs/E052/.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT/'runs/E052'
REFERENCE = ROOT/'runs/E051/dev_20261126T213703Z_20c2a1d5'
SOURCE = ROOT/'experiments/2026-11-26_E051_lagodny_tilt/evaluate.py'
spec = importlib.util.spec_from_file_location('_e052_e051', SOURCE)
e51 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e51
spec.loader.exec_module(e51)
e49 = e51.e49
TRAINING = e51.e50.TRAINING
read, write, digest, require, utc = e51.read, e51.write, e51.digest, e51.require, e51.utc
DS, SEEDS, N_DEV = e51.DS, e51.SEEDS, e51.N
CONTROLS = ('full', 'b10_U', 'b10_T', 'b10_L06')
DISPLAY = dict(zip(CONTROLS, ('Pełny', 'UNIFORM', 'TILT', 'Frozen offsets')))
N_SELECT, RNG_SEED, CROP, LIMIT = 8000, 20261126, .65, 900
LIFTS = {'S01': (4, 0), 'S02': (0, 6), 'S03': (2, 3),
         'S04': (8, 0), 'S05': (0, 12), 'S06': (4, 6)}
TARGET = {'cifar100': '79.7500', 'tiny': '59.6125'}


def new_run(prefix):
    path = RUNS/f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    (path/'logs').mkdir(parents=True, exist_ok=False)
    return path


def sources():
    m51, a51, old = (read(REFERENCE/f) for f in ('manifest.json', 'audit.json', 'frozen.json'))
    m48, a48, cfg = (read(TRAINING/f) for f in ('manifest.json', 'audit.json', 'config.json'))
    require(m51['status'] == m48['status'] == 'completed' and m51['exit_codes'] == [0]*4 and
            m51['dev_only'] and a51['status'] == a48['status'] == 'passed' and
            a51['scores'] == 56 and a51['reproduced_controls'] == 32 and
            a48['validated_checkpoints'] == 8, 'Brak poprawnego źródła E048/E051')
    require(m51['operator'] == e49.e29.COMMIT and old['tiny_crop_min'] == CROP and
            m51['frozen_sha256'] == digest(REFERENCE/'frozen.json') and
            m48['config_sha256'] == digest(TRAINING/'config.json'), 'Inny protokół źródłowy')
    require(m51['torch'] == torch.__version__ and m51['torchvision'] == torchvision.__version__
            and m51['cuda'] == torch.version.cuda, 'Inne środowisko niż E051')
    # Tylko kod i wejścia train: nie czytamy ścieżek benchmarkowych z manifestu E051.
    hashes = {p: h for p, h in m51['sources'].items() if p.endswith('.py')}
    require(str(SOURCE.relative_to(ROOT)) in hashes and
            m48['inputs'] == {p: m51['sources'][p] for p in m48['inputs']}, 'Inny kod lub dane')
    hashes.update(m48['inputs'])
    for name in ('config.json', 'audit.json', 'manifest.json'):
        rel = str((TRAINING/name).relative_to(ROOT))
        hashes[rel] = m51['sources'][rel]
    for ds in DS:
        for seed in SEEDS:
            record = old['checkpoints'][ds][str(seed)]
            require(record['config_sha256'] == m48['config_sha256'], 'Inny config w checkpoincie')
            hashes[str((TRAINING/ds/f'seed_{seed}/last.pt').relative_to(ROOT))] = record['sha256']
    for path in (Path(__file__), HERE/'OPIS.md', HERE/'URUCHOMIENIE.md',
                 *(REFERENCE/f for f in ('manifest.json', 'audit.json', 'frozen.json',
                                         *(f'seed_{s}.json' for s in SEEDS)))):
        hashes[str(path.relative_to(ROOT))] = digest(path)
    return old, cfg, hashes


def validate_sources(hashes):
    for rel, expected in hashes.items():
        require(digest(ROOT/rel) == expected, f'Zmienione wejście {rel}')


def budget_ok(cost, full, uniform, tilt):
    p, m = cost['parameters'], cost['mac']
    return (999*uniform['parameters'] <= 1000*p <= 1000*uniform['parameters'] and
            1000*p <= 905*full['parameters'] and p <= tilt['parameters'] and m <= tilt['mac'])


def geometry(old, cfg):
    plans = {ds: {v: old['geometry'][ds][v] for v in CONTROLS} for ds in DS}
    full, tilt = (plans[DS[0]][v]['widths'] for v in ('full', 'b10_T'))
    require(all(plans[ds]['full']['widths'] == full and plans[ds]['b10_T']['widths'] == tilt
                for ds in DS) and list(full) == list(e49.projector.SIZES) and
            [i for i, (n, w) in enumerate(zip(full, tilt, strict=True)) if w < n] == [9, 10, 11] and
            tilt[9:] == [502, 474, 446], 'Inna geometria TILT')

    def costs(ds, w):
        p, m = map(int, e49.projector.costs(w, cfg['datasets'][ds]['classes']))
        return {'widths': w, 'parameters': p, 'mac': m}

    used = {tuple(entry['widths']) for ds in DS for entry in old['geometry'][ds].values()}
    for name, (a, b) in LIFTS.items():
        w = tilt.copy()
        w[10] += a
        w[11] += b
        for step in range(33):
            if all(costs(ds, w)['parameters'] <= plans[ds]['b10_T']['parameters'] for ds in DS):
                break
            require(step < 32, f'{name}: wymagane za duże przesunięcie')
            w[(6, 8)[step % 2]] -= 1
        require(tuple(w) not in used and all(1 <= a <= b for a, b in zip(w, full, strict=True)),
                f'{name}: duplikat lub zła szerokość')
        used.add(tuple(w))
        for ds, profiles in plans.items():
            entry = costs(ds, w)
            require(budget_ok(entry, profiles['full'], profiles['b10_U'], profiles['b10_T']),
                    f'{ds}/{name}: poza pułapami')
            profiles[name] = entry

    for ds, profiles in plans.items():
        cfg_ds = cfg['datasets'][ds]
        torch.manual_seed(42)
        base = e49.train.model_for(ds, cfg).eval().requires_grad_(False)
        g = e49.e29.bn.Geometry(base)
        require(tuple(g.names) == e49.projector.GEOMETRY_NAMES and list(g.sizes) == full,
                'Inna kolejność grup modelu')
        require(e49.train.size_cost(base, cfg_ds['input_size'], cfg_ds['classes']) ==
                (profiles['full']['parameters'], profiles['full']['mac']), 'Zły koszt rzeczywisty pełnego modelu')
        for name, entry in profiles.items():
            w = entry['widths']
            require(costs(ds, w) == entry and g.count(w) == entry['parameters'], 'Niezgodne wzory P/MAC')
            if name in LIFTS:
                print(ds, name, entry, flush=True)
    return plans


def split_plan(old):
    result = {}
    for ds in DS:
        split = e49.train.DATA[ds].load_split()['splits']
        a, dev = (old['indices'][ds][k] for k in ('A', 'dev'))
        sa, sd, train = set(a), set(dev), set(split['train'])
        require(len(a) == len(sa) == 4096 and sa <= train and len(dev) == len(sd) == N_DEV[ds]
                and sd == set(split['dev']) and not train & sd, 'Błędne A4096 albo dev')
        rng = np.random.Generator(np.random.PCG64(RNG_SEED))
        select = sorted(rng.choice(sorted(train-sa), size=N_SELECT, replace=False).tolist())
        ss = set(select)
        require(len(ss) == N_SELECT and ss <= train and not ss & (sa | sd), 'Błędny podzbiór selekcji')
        if ds == 'cifar100':
            require(not (sa | sd | ss) & set(split['reserve']), 'Przecięcie z rezerwą CIFAR-100')
        result[ds] = {'A': a, 'select': select, 'dev': dev}
    return result


def preflight():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
            and os.environ.get('CONDA_DEFAULT_ENV') == 'attention', 'Kontrola CPU w attention z ukrytym GPU')
    torch.set_num_threads(2)
    old, cfg, hashes = sources()
    validate_sources(hashes)
    book = {ds: {str(s): {
        'checkpoint_state_sha256': old['reference'][ds][str(s)]['full_pre']['state_sha256'],
        'a_views_sha256': old['reference'][ds][str(s)]['full']['calibration_views_sha256'],
    } for s in SEEDS} for ds in DS}
    frozen = dict(pool=list(LIFTS), controls=list(CONTROLS), geometry=geometry(old, cfg),
                  indices=split_plan(old), checkpoints=old['checkpoints'], reference_hashes=book,
                  selection_rule='min(mean of four train NLL, then ascending profile ID)',
                  n_select=N_SELECT, rng_seed=RNG_SEED, tiny_crop_min=CROP,
                  target_top1_pct=TARGET, min_gain_pp=1, min_positive_pairs=3, max_loss_pp=3)
    out = new_run('cpu')
    write(out/'frozen.json', frozen)
    write(out/'manifest.json', dict(status='passed', phase='cpu', frozen_utc=utc(),
                                    frozen_sha256=digest(out/'frozen.json'), sources=hashes,
                                    cpu_geometry_checks=20, full_forwards=2, gpu_used=False, scored=False))
    print(f'CPU={out.relative_to(ROOT)} SHA256={digest(out/"frozen.json")}', flush=True)


def tensors(ds, indices, a_ids, crop):
    bn = e49.e29.bn
    if ds == 'cifar100':
        images = bn.Images()  # Wyłącznie oryginalny plik train.
        aug = bn.augment_uint8(images.raw, a_ids)
        a = bn.normalize(aug)
        x, y = images.dev(indices)  # Funkcja przyjmuje dowolną podaną listę indeksów train.
    else:
        e30 = e49.old.e30
        images, labels, _ = e49.old.dt.load_tiny_train()
        ref = e30.RRC
        e30.RRC = torchvision.transforms.RandomResizedCrop(ref.size, scale=(crop, ref.scale[1]),
                                                          ratio=ref.ratio, interpolation=ref.interpolation,
                                                          antialias=ref.antialias)
        try:
            aug = e30.augment(images, a_ids)
        finally:
            e30.RRC = ref
        a = e30.normalize(aug)
        x = e30.normalize(torch.from_numpy(images[indices]).permute(0, 3, 1, 2).float()/255)
        y = torch.tensor(labels[indices], dtype=torch.long)
    require(len(x) == len(y) == len(indices) and len(a) == 4096 and
            torch.isfinite(a).all() and torch.isfinite(x).all(), 'Złe tensory')
    return a, x, y, bn.tensor_hash(aug)


def worker(run, phase, gpu):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1,
            'Złe przypisanie GPU')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = SEEDS[gpu]
    m = read(run/'manifest.json')
    require(digest(run/'frozen.json') == m['frozen_sha256'] and m['phase'] == phase, 'Zmienione wejścia')
    f = read(run/'frozen.json')
    cfg = read(TRAINING/'config.json')
    train_only = phase == 'select'
    chosen = train_rows = None
    if not train_only:
        path = run/'selection.json'
        require(digest(path) == m['selection_sha256'], 'Brak zamrożonego wyboru')
        selection = read(path)
        train_path = run/f'select_{seed}.json'
        require(selection['frozen_sha256'] == m['frozen_sha256'] and
                selection['train_sha256'][str(seed)] == digest(train_path), 'Zmieniona selekcja train')
        chosen = selection['chosen']
        train_rows = {(r['dataset'], r['variant']): r for r in read(train_path)['rows']}
    rows = []
    for ds in DS:
        ref = f['reference_hashes'][ds][str(seed)]
        base = e49.load_model(TRAINING, f['checkpoints'][ds][str(seed)], ds, seed, cfg)
        require(e49.state_hash(base) == ref['checkpoint_state_sha256'], 'Inny stan źródłowy E048')
        a, x, y, views_sha = tensors(ds, f['indices'][ds][phase], f['indices'][ds]['A'], f['tiny_crop_min'])
        require(views_sha == ref['a_views_sha256'], f'{ds}: widoki CMA różne od E051')
        for v in (f['pool'] if train_only else [*CONTROLS, chosen[ds]]):
            plan = f['geometry'][ds][v]
            model = e49.fold(base, v, plan)
            spec_ds = cfg['datasets'][ds]
            p, mac = e49.train.size_cost(model, spec_ds['input_size'], spec_ds['classes'])
            require((p, mac) == (plan['parameters'], plan['mac']), f'{ds}/{v}: inny P/MAC')
            weights = e49.e29.bn.weights_hash(model)
            repair = e49.e29.calibrate(model.cuda(), a, 'CMA')
            require(repair['params_unchanged'] and repair['estimator'] == 'CMA_reset' and
                    repair['samples'] == 4096 and repair['batches'] == 32 and
                    e49.e29.bn.weights_hash(model) == weights, 'Niezgodna CMA lub zmiana wag')
            state = e49.state_hash(model)
            if not train_only and v == chosen[ds]:
                require(state == train_rows[(ds, v)]['state_sha256'], 'Niezgodny wybrany stan po CMA')
            metrics = e49.score(model, x, y, state)
            if train_only:
                metrics = {k: metrics[k] for k in ('n', 'nll')}
            rows.append(dict(dataset=ds, seed=seed, gpu=gpu, variant=v, parameters=p, mac=mac,
                             state_sha256=state, weights_sha256=weights, calibration_views_sha256=views_sha,
                             cma_samples=repair['samples'], cma_batches=repair['batches'], **metrics))
            print(ds, v, f"NLL={metrics['nll']:.8f}" if train_only else
                  f"top-1={metrics['top1_pct']:.4f}%", flush=True)
        require(e49.state_hash(base) == ref['checkpoint_state_sha256'], 'Zmieniony model źródłowy')
        del a, x, y, base, model
    write(run/f'{phase}_{seed}.json', dict(phase=phase, frozen_sha256=m['frozen_sha256'],
                                           selection_sha256=m.get('selection_sha256'), rows=rows))


def collect(run, m, phase, pool):
    f = read(run/'frozen.json')
    rows = {}
    for gpu, s in enumerate(SEEDS):
        bundle = read(run/f'{phase}_{s}.json')
        require(bundle['phase'] == phase and bundle['frozen_sha256'] == m['frozen_sha256'] and
                (phase == 'select' or bundle['selection_sha256'] == m['selection_sha256']), 'Inny etap')
        require([(r['dataset'], r['variant'], r['seed'], r['gpu']) for r in bundle['rows']] ==
                [(ds, v, s, gpu) for ds in DS for v in pool[ds]], 'Brak dokładnego kompletu wierszy')
        for r in bundle['rows']:
            ds, v = r['dataset'], r['variant']
            n = N_SELECT if phase == 'select' else N_DEV[ds]
            plan = f['geometry'][ds][v]
            require(r['n'] == n and math.isfinite(r['nll']) and r['nll'] >= 0 and
                    (r['parameters'], r['mac']) == (plan['parameters'], plan['mac']) and
                    r['calibration_views_sha256'] == f['reference_hashes'][ds][str(s)]['a_views_sha256'] and
                    (r['cma_samples'], r['cma_batches']) == (4096, 32), 'Niepoprawny wiersz pomiaru')
            if phase == 'dev':
                require(0 <= r['correct'] <= n and r['top1_pct'] == 100*r['correct']/n, 'Błędny top-1')
            rows[ds, v, s] = r
    return rows


def train_choice(rows):
    means = {ds: {v: math.fsum(rows[ds, v, s]['nll'] for s in SEEDS)/len(SEEDS) for v in LIFTS} for ds in DS}
    chosen = {ds: min(LIFTS, key=lambda v: (means[ds][v], v)) for ds in DS}
    return means, chosen


def select(run, m):
    rows = collect(run, m, 'select', {ds: LIFTS for ds in DS})
    means, chosen = train_choice(rows)
    write(run/'selection.json', dict(frozen_sha256=m['frozen_sha256'], frozen_utc=utc(), chosen=chosen,
                                     mean_train_nll=means, train_sha256={str(s): digest(run/f'select_{s}.json')
                                                                       for s in SEEDS}))
    print(f'Zamrożony wybór po train: {chosen}', flush=True)


def audit(run, m):
    f = read(run/'frozen.json')
    selection = read(run/'selection.json')
    require(digest(run/'selection.json') == m['selection_sha256'] and
            digest(run/'frozen.json') == m['frozen_sha256'] == selection['frozen_sha256'] and
            [w['phase'] for w in m['waves']] == ['select', 'dev'] and
            all(w['exit_codes'] == [0]*4 for w in m['waves']), 'Niezgodne fale lub wybór')
    require(m['waves'][0]['finished_utc'] <= selection['frozen_utc'] <= m['waves'][1]['started_utc'],
            'Dev przed zamrożeniem wyboru')
    train_rows = collect(run, m, 'select', {ds: LIFTS for ds in DS})
    chosen = selection['chosen']
    require(train_choice(train_rows) == (selection['mean_train_nll'], chosen) and
            all(digest(run/f'select_{s}.json') == selection['train_sha256'][str(s)] for s in SEEDS),
            'Zmieniony wybór albo metryki train')
    pool = {ds: [*CONTROLS, chosen[ds]] for ds in DS}
    rows = collect(run, m, 'dev', pool)
    for s in SEEDS:
        prior = read(REFERENCE/f'seed_{s}.json')
        require(prior['frozen_sha256'] == read(REFERENCE/'manifest.json')['frozen_sha256'], 'Inne wyniki E051')
        keys = ('state_sha256', 'calibration_views_sha256', 'correct', 'n', 'parameters', 'mac')
        old = {(r['dataset'], r['variant']): r for r in prior['rows']}
        for ds in DS:
            require(rows[ds, chosen[ds], s]['state_sha256'] == train_rows[ds, chosen[ds], s]['state_sha256'],
                    'Różne stany wybranego modelu na train i dev')
            for v in CONTROLS:
                require(all(rows[ds, v, s][k] == old[ds, v][k] for k in keys), f'Nie odtworzono E051: {ds}/{v}/{s}')

    results = {}
    notes = ['# E052 — jednorazowy wynik b10 na dev', '', '| Zbiór | Wariant | Top‑1 [%] | Strata [pp] | P | MAC/obraz |',
             '|---|---|---:|---:|---:|---:|']
    for ds in DS:
        n, v = N_DEV[ds], chosen[ds]
        hits = {p: [rows[ds, p, s]['correct'] for s in SEEDS] for p in pool[ds]}
        pct = lambda k: Fraction(100*sum(hits[k]), n*len(SEEDS))
        means = {p: float(pct(p)) for p in pool[ds]}
        loss = {p: float(pct('full')-pct(p)) for p in pool[ds]}
        diffs = {}
        for ref in CONTROLS:
            delta = [100*(a-b)/n for a, b in zip(hits[v], hits[ref], strict=True)]
            diffs[ref] = dict(mean_pp=float(pct(v)-pct(ref)), per_seed_pp=dict(zip(SEEDS, delta)),
                              positive_pairs=sum(x > 0 for x in delta))
        gates = dict(loss_le_3pp=pct('full')-pct(v) <= f['max_loss_pp'],
                     gain_vs_U_ge_1pp=pct(v)-pct('b10_U') >= f['min_gain_pp'],
                     positive_pairs_vs_U_ge_3=diffs['b10_U']['positive_pairs'] >= f['min_positive_pairs'],
                     target_met=pct(v) >= Fraction(f['target_top1_pct'][ds]))
        p = f['geometry'][ds]
        require(budget_ok(p[v], p['full'], p['b10_U'], p['b10_T']), 'Poza pułapami')
        results[ds] = dict(chosen=v, top1_mean_pct=means, loss_to_full_pp=loss, chosen_minus=diffs,
                           gates=gates, passed=all(gates.values()))
        for name in pool[ds]:
            notes.append(f"| {ds} | {DISPLAY.get(name, name)} | {means[name]:.4f} | {loss[name]:.4f} | "
                         f"{p[name]['parameters']} | {p[name]['mac']} |")
    for name, data in (('train_per_seed', train_rows), ('per_seed', rows)):
        with (run/f'{name}.csv').open('w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(next(iter(data.values()))))
            w.writeheader()
            w.writerows(data.values())
    (run/'WYNIKI.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    write(run/'audit.json', dict(status='passed', train_scores=len(train_rows), dev_scores=len(rows),
                                reproduced_E051=32, reproduced_selected_states=8, selection_dev_overlap=0,
                                selection_A4096_overlap=0, training=False, benchmark_used=False,
                                reserve_used=False, imagenetv2_used=False, datasets=results,
                                joint_gate_met=all(d['passed'] for d in results.values())))


def check_gpus():
    gpus = e49.train.require_idle_gpus()
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                         '--format=csv,noheader'], text=True).strip(), 'Zajęta karta GPU')
    return gpus


def wave(run, m, phase, deadline):
    w = dict(phase=phase, started_utc=utc(), workers=[])
    m['phase'] = phase
    m['waves'].append(w)
    write(run/'manifest.json', m)
    procs = []
    try:
        for gpu, seed in enumerate(SEEDS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            cmd = [sys.executable, '-B', str(Path(__file__)), '--worker', phase, str(gpu), '--run', str(run)]
            with (run/'logs'/f'{phase}_{seed}.log').open('w') as log:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            procs.append(p)
            w['workers'].append(dict(pid=p.pid, seed=seed, gpu=gpu))
        write(run/'manifest.json', m)
        while (codes := [p.poll() for p in procs]) != [0]*4:
            require(all(code in (None, 0) for code in codes), f'{phase}: worker zakończył się błędem {codes}')
            require(time.monotonic() < deadline, 'Przekroczony limit E052')
            time.sleep(1)
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()
        w.update(exit_codes=[p.returncode for p in procs], finished_utc=utc())
        write(run/'manifest.json', m)


def execute(cpu):
    cpu = cpu.resolve(strict=True)
    require(cpu.parent == RUNS and cpu.name.startswith('cpu_'), 'Podaj unikalny katalog kontroli CPU E052')
    with (RUNS/'.gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any(RUNS.glob('b10_*')), 'E052 ma już bieg GPU; nie dublujemy')
        c = read(cpu/'manifest.json')
        require(c['status'] == 'passed' and c['phase'] == 'cpu' and not c['scored'] and
                digest(cpu/'frozen.json') == c['frozen_sha256'], 'Brak zamrożonego planu CPU')
        validate_sources(c['sources'])
        f = read(cpu/'frozen.json')
        old = read(REFERENCE/'frozen.json')
        require(f['indices'] == split_plan(old) and f['pool'] == list(LIFTS) and f['controls'] == list(CONTROLS),
                'Zmienione pule lub podziały')
        gpus = check_gpus()
        run = new_run('b10')
        shutil.copyfile(cpu/'frozen.json', run/'frozen.json')
        m = dict(status='running', phase='prepare', started_utc=utc(), pid=os.getpid(), waves=[],
                 cpu_run=str(cpu.relative_to(ROOT)), cpu_manifest_sha256=digest(cpu/'manifest.json'),
                 frozen_sha256=c['frozen_sha256'], gpus=gpus, limit_seconds=LIMIT,
                 torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
                 python=sys.version, numpy=np.__version__, operator=e49.e29.COMMIT,
                 command=f"CUDA_VISIBLE_DEVICES=0,1,2,3 {sys.executable} -B "
                         f"{Path(__file__).relative_to(ROOT)} --execute {cpu.relative_to(ROOT)}")
        write(run/'manifest.json', m)
        print(f'RUN={run.relative_to(ROOT)} PID={os.getpid()}', flush=True)
        start = time.monotonic()
        try:
            wave(run, m, 'select', start+LIMIT)
            select(run, m)
            m['selection_sha256'] = digest(run/'selection.json')
            time.sleep(2)  # Zamknięcie kontekstów CUDA przed drugą falą.
            check_gpus()
            wave(run, m, 'dev', start+LIMIT)
            audit(run, m)
            m.update(status='completed', phase='done', audit_sha256=digest(run/'audit.json'),
                     per_seed_sha256=digest(run/'per_seed.csv'))
        except BaseException:
            m.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            m.update(finished_utc=utc(), wall_seconds=time.monotonic()-start)
            write(run/'manifest.json', m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--execute', type=Path, metavar='KATALOG_CPU')
    action.add_argument('--worker', nargs=2, metavar=('FAZA', 'GPU'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.check:
        preflight()
    elif args.execute:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        execute(args.execute)
    else:
        phase, gpu = args.worker
        require(args.run is not None and phase in ('select', 'dev') and int(gpu) in range(4), 'Złe argumenty workera')
        worker(args.run, phase, int(gpu))
