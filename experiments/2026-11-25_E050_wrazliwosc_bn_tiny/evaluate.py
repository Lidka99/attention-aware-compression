"""E050 (OPIS.md): porównanie dwóch skal cropu dla BN na Tiny dev.

Wejście: last.pt E048, zamrożone geometrie i dev E049, cache train Tiny.
Wyjście: manifest, per_seed.csv, audit.json i raport w runs/E050/.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import os
import secrets
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E49 = ROOT/'experiments/2026-11-24_E049_folding_mocniejszych_baz/evaluate.py'
REFERENCE = ROOT/'runs/E049/eval_20261124T234559Z_b4ca081b'
TRAINING = ROOT/'runs/E048/train_20261124T232822Z_89dfc621'
SEEDS = (42, 43, 44, 45)
PROFILES = ('full', 'b10_U', 'b10_Tproj', 'b10_L06')
VIEWS = {'crop040': .40, 'crop065': .65}
N_DEV = 10000
LIMIT_SECONDS = 900

spec = importlib.util.spec_from_file_location('_e050_e049', E49)
e49 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e49
spec.loader.exec_module(e49)
read, write, digest, utc, require = e49.read, e49.write, e49.digest, e49.utc, e49.require


def provenance():
    old = read(REFERENCE/'manifest.json')
    train = read(TRAINING/'manifest.json')
    audit = read(TRAINING/'audit.json')
    data = e49.old.dt
    require(old['status'] == train['status'] == 'completed' and audit['status'] == 'passed' and
            audit['validated_checkpoints'] == 8 and Path(old['training']) == TRAINING, 'Nieaudytowane źródło')
    require(e49.source_hashes() == old['sources'] and digest(REFERENCE/'frozen.json') == old['frozen_sha256'] and
            digest(TRAINING/'config.json') == train['config_sha256'] == old['train_config_sha256'], 'Zmienione wejście E049')
    require(torch.__version__ == old['torch'] and torchvision.__version__ == train['torchvision'], 'Inny stos PyTorch')
    release = read(REFERENCE/'benchmark_release.json')  # tylko hashe zapisanych stanów przed benchmarkiem
    require(release['passed'] and release['frozen_sha256'] == old['frozen_sha256'], 'Zmieniony audyt dev E049')
    ready = [REFERENCE/f'seed_{s}/ready.json' for s in SEEDS]
    files = {str(p.relative_to(ROOT)): digest(p) for p in (
        HERE/'OPIS.md', HERE/'URUCHOMIENIE.md', Path(__file__), REFERENCE/'manifest.json',
        REFERENCE/'frozen.json', REFERENCE/'benchmark_release.json', TRAINING/'manifest.json',
        TRAINING/'audit.json', TRAINING/'config.json', data.CACHE_TRAIN, data.SPLITS_FILE, *ready)}
    for p in (data.CACHE_TRAIN, data.SPLITS_FILE):
        rel = str(p.relative_to(ROOT))
        require(files[rel] == train['inputs'][rel], f'Zmienione dane: {rel}')
    frozen = read(REFERENCE/'frozen.json')
    ids = frozen['indices']['tiny']
    split = data.load_split()['splits']
    require(len(ids['A']) == len(set(ids['A'])) == 4096 and set(ids['A']) <= set(split['train']) and
            len(ids['dev']) == N_DEV and set(ids['dev']) == set(split['dev']), 'Błędne A4096/dev')
    profiles = {p: frozen['geometry']['tiny']['profiles'][p] for p in PROFILES}
    for cost in ('parameters', 'mac'):
        cap = profiles['b10_U'][cost]
        for profile in PROFILES[1:]:
            value = profiles[profile][cost]
            require(999*cap <= 1000*value <= 1000*cap and 1000*value <= 905*profiles['full'][cost], 'Błędny budżet b10')
    archive = {}
    for seed, path in zip(SEEDS, ready, strict=True):
        require(files[str(path.relative_to(ROOT))] == release['ready_sha256'][str(seed)], 'Zmieniona kontrola dev E049')
        row = read(path)
        require(row['seed'] == seed and not row['benchmark_opened'], 'Inna próba E049')
        archive[str(seed)] = {r['variant']: r for r in row['records'] if r['dataset'] == 'tiny' and
                              r['variant'] in (*PROFILES, 'full_pre')}
        require(set(archive[str(seed)]) == {*PROFILES, 'full_pre'}, 'Niepełna kontrola E049')
    return {'profiles': profiles, 'checkpoints': frozen['checkpoints']['tiny'], 'archive_dev': archive,
            'indices': {k: ids[k] for k in ('A', 'dev')}, 'views': VIEWS}, {**old['sources'], **files}


def tensors(frozen):
    e30 = e49.old.e30
    ids = frozen['indices']
    images, labels, _ = e49.old.dt.load_tiny_train()
    x = e30.normalize(torch.from_numpy(images[ids['dev']]).permute(0, 3, 1, 2).float()/255)
    y = torch.tensor(labels[ids['dev']], dtype=torch.long)
    require(len(x) == len(y) == N_DEV and torch.isfinite(x).all(), 'Błędny dev Tiny')
    reference = e30.RRC
    views = {}
    for key, minimum in frozen['views'].items():
        # Ten sam generator widoków E030, z podmienionym wyłącznie dolnym ograniczeniem skali.
        e30.RRC = T.RandomResizedCrop(reference.size, scale=(minimum, reference.scale[1]),
                                    ratio=reference.ratio, interpolation=reference.interpolation,
                                    antialias=reference.antialias)
        try:
            aug = e30.augment(images, ids['A'])
        finally:
            e30.RRC = reference
        views[key] = (e30.normalize(aug), hashlib.sha256(aug.contiguous().numpy().tobytes()).hexdigest())
        require(torch.isfinite(views[key][0]).all(), 'Niefinitywne widoki A4096')
    require(views['crop040'][1] != views['crop065'][1], 'Identyczne widoki mimo zmiany skali')
    return views, x, y


def worker(run, gpu):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1, 'Błędne GPU')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = SEEDS[gpu]
    manifest = read(run/'manifest.json')
    require(digest(run/'frozen.json') == manifest['frozen_sha256'], 'Zmieniony plan')
    frozen = read(run/'frozen.json')
    cfg_path = TRAINING/'config.json'
    require(digest(cfg_path) == manifest['sources'][str(cfg_path.relative_to(ROOT))], 'Zmieniona konfiguracja E048')
    cfg = read(cfg_path)
    config = cfg['datasets']['tiny']
    require(tuple(config['crop_scale']) == (VIEWS['crop065'], 1.0), 'Inna skala cropu E048')
    archive = frozen['archive_dev'][str(seed)]
    base = e49.load_model(TRAINING, frozen['checkpoints'][str(seed)], 'tiny', seed, cfg)
    base_hash = e49.state_hash(base)
    require(base_hash == archive['full_pre']['state_sha256'], 'Inny stan źródłowy')
    views, x, y = tensors(frozen)
    rows = []
    for profile in PROFILES:
        geometry = frozen['profiles'][profile]
        p, m = geometry['parameters'], geometry['mac']
        model = e49.fold(base, profile, geometry)
        require(e49.train.size_cost(model, config['input_size'], config['classes']) == (p, m), 'Inny koszt modelu')
        weight_hash = e49.e29.bn.weights_hash(model)
        for key, (calibration, views_hash) in views.items():
            candidate = copy.deepcopy(model)
            repair = e49.e29.calibrate(candidate.cuda(), calibration, 'CMA')
            require(repair['params_unchanged'] and repair['samples'] == 4096 and repair['batches'] == 32 and
                    e49.e29.bn.weights_hash(candidate) == weight_hash, 'Niespójna kalibracja CMA')
            state = e49.state_hash(candidate)
            score = e49.score(candidate, x, y, state)
            if key == 'crop040':
                ref = archive[profile]
                require((views_hash, state, score['correct'], score['n'], p, m) ==
                        (ref['calibration_views_sha256'], ref['state_sha256'], ref['dev']['correct'],
                         ref['dev']['n'], ref['parameters'], ref['mac']), f'{seed}/{profile}: nie odtworzono E049')
            rows.append(dict(seed=seed, gpu=gpu, profile=profile, views=key, parameters=p, mac=m,
                             parameters_sha256=weight_hash, state_sha256=state, views_sha256=views_hash, **score))
            print(f"{seed} {profile} {key}: {score['top1_pct']:.4f}%", flush=True)
    require(e49.state_hash(base) == base_hash, 'Zmienione wagi źródłowe')
    write(run/f'seed_{seed}.json', {'rows': rows, 'frozen_sha256': manifest['frozen_sha256']})


def paired(values):
    # Dla każdego seeda n = 10 000: 100 trafień to 1 pp.
    pp = [v*100/N_DEV for v in values]
    return {'mean_pp': statistics.mean(pp), 'sd_pp': statistics.stdev(pp), 'per_seed_pp': pp,
            'positive_pairs': sum(v > 0 for v in values)}


def summarize(run, manifest):
    rows = []
    for gpu, seed in enumerate(SEEDS):
        bundle = read(run/f'seed_{seed}.json')
        require(bundle['frozen_sha256'] == manifest['frozen_sha256'] and
                [(r['seed'], r['gpu'], r['profile'], r['views']) for r in bundle['rows']] ==
                [(seed, gpu, p, v) for p in PROFILES for v in VIEWS] and
                all(r['n'] == N_DEV for r in bundle['rows']), 'Niepełne wyniki dev')
        rows.extend(bundle['rows'])
    require(all(len({r['views_sha256'] for r in rows if r['views'] == v}) == 1 for v in VIEWS), 'Różne widoki między seedami')
    hits = {v: {p: [r['correct'] for r in rows if r['views'] == v and r['profile'] == p]
                for p in PROFILES} for v in VIEWS}
    old, new = hits.values()
    diff = lambda a, b: [x-y for x, y in zip(a, b, strict=True)]
    changes = {p: diff(new[p], old[p]) for p in PROFILES}
    gain = diff(new['b10_L06'], new['b10_U'])
    loss = diff(new['full'], new['b10_L06'])
    n = N_DEV*len(SEEDS)
    gate = (100*sum(gain) >= n and sum(v > 0 for v in gain) >= 3 and 100*sum(loss) <= 3*n and
            sum(v > 0 for v in changes['b10_L06']) >= 3 and sum(changes['full']) >= 0 and sum(changes['b10_U']) >= 0)
    summaries = {k: {'top1_mean_pct': {p: statistics.mean(h[p])*100/N_DEV for p in PROFILES},
                     'L06_minus_U': paired(diff(h['b10_L06'], h['b10_U'])),
                     'L06_minus_Tproj': paired(diff(h['b10_L06'], h['b10_Tproj'])),
                     'full_minus_L06': paired(diff(h['full'], h['b10_L06']))} for k, h in hits.items()}
    audit = {'status': 'passed', 'scores': len(rows), 'reproduced_E049': len(rows)//2, 'dev_only': True,
             'recipes': summaries, 'shift_in_top1_by_profile': {p: paired(d) for p, d in changes.items()},
             'shift_in_L06_minus_U': paired(diff(changes['b10_L06'], changes['b10_U'])),
             'shift_in_full_minus_L06': paired(diff(changes['full'], changes['b10_L06'])),
             'pilot_gate_met': gate}
    with (run/'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    notes = ['# E050 — Tiny dev, skala cropu przy A4096/CMA', '',
             'Odtworzenie kontroli E049: 16/16. Cztery sparowane seedy (42–45).', '',
             '| Profil | Top‑1 crop040 [%] | Top‑1 crop065 [%] | Zmiana [pp] |',
             '|---|---:|---:|---:|']
    for p in PROFILES:
        a, b = (s['top1_mean_pct'][p] for s in summaries.values())
        notes.append(f'| {p} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    for key, s in summaries.items():
        g = s['L06_minus_U']
        notes += ['', f"**{key}:** L06−UNIFORM {g['mean_pp']:+.4f} pp ({g['positive_pairs']}/4), "
                  f"L06−T-proj {s['L06_minus_Tproj']['mean_pp']:+.4f} pp, "
                  f"strata L06 do pełnego {s['full_minus_L06']['mean_pp']:.4f} pp."]
    notes += ['', f'Bramka diagnostyczna: **{"TAK" if gate else "NIE"}**. Dane per seed w `per_seed.csv` i `audit.json`.']
    (run/'WYNIKI.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    write(run/'audit.json', audit)


def run_experiment():
    frozen, sources = provenance()
    gpus = e49.train.require_idle_gpus()
    run = ROOT/'runs/E050'/f'dev_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    (run/'logs').mkdir(parents=True, exist_ok=False)
    write(run/'frozen.json', frozen)
    manifest = dict(status='running', started_utc=utc(), pid=os.getpid(), sources=sources, gpus=gpus,
                    frozen_sha256=digest(run/'frozen.json'), workers=[], dev_only=True,
                    command=f'CUDA_VISIBLE_DEVICES=0,1,2,3 {sys.executable} -B {Path(__file__).relative_to(ROOT)}',
                    operator=e49.e29.COMMIT, python=sys.version, torch=torch.__version__,
                    torchvision=torchvision.__version__, cuda=torch.version.cuda, limit_seconds=LIMIT_SECONDS)
    write(run/'manifest.json', manifest)
    print(f'RUN={run.relative_to(ROOT)} PID={os.getpid()}', flush=True)
    procs = []
    start = time.monotonic()
    try:
        for gpu, seed in enumerate(SEEDS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(gpu), '--run', str(run)]
            with (run/'logs'/f'seed_{seed}.log').open('w') as log:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(p)
            manifest['workers'].append(dict(pid=p.pid, seed=seed, gpu=gpu))
        write(run/'manifest.json', manifest)
        while (codes := [p.poll() for p in procs]) != [0]*4:
            require(all(code in (None, 0) for code in codes), f'Kod wyjścia workera: {codes}')
            require(time.monotonic()-start < LIMIT_SECONDS, 'Przekroczono limit E050')
            time.sleep(1)
        summarize(run, manifest)
        manifest['status'] = 'completed'
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()
        manifest.update(exit_codes=[p.returncode for p in procs], finished_utc=utc(), wall_seconds=time.monotonic()-start)
        write(run/'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=range(4))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.worker is None:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        run_experiment()
    else:
        worker(args.run, args.worker)
