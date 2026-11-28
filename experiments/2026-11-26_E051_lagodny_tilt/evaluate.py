"""E051 (OPIS.md): łagodny TILT przy dopasowanym P na CIFAR-100 i Tiny dev.

Wejście: last.pt E048, audyt E049/E050, archiwalne A4096/dev i geometrie b10.
Wyjście: zamrożony plan, manifest, logi, CSV i audyt pod runs/E051/.
"""
from __future__ import annotations

import argparse
import csv
import os
import secrets
import signal
import statistics as st
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import importlib.util

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E50 = ROOT/'experiments/2026-11-25_E050_wrazliwosc_bn_tiny/evaluate.py'
RECROP = ROOT/'runs/E050/dev_20261125T231044Z_d44c66e4'
spec = importlib.util.spec_from_file_location('_e051_e050', E50)
e50 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e50
spec.loader.exec_module(e50)
e49 = e50.e49
read, write, digest, utc, require = e49.read, e49.write, e49.digest, e49.utc, e49.require
DS = ('cifar100', 'tiny')
SEEDS = e49.SEEDS
VARIANTS = ('full', 'b10_U', 'b10_Tproj', 'b10_L06', 'b10_T', 'b5_U', 'b5_T')
CONTROLS = VARIANTS[:4]
N = {'cifar100': 5000, 'tiny': 10000}
LIMIT = 900


def mild_at_p(geom, cap):
    widths = list(map(int, e49.mild_linear(geom, cap)))
    while 1000*geom.count(widths) < 999*cap:
        trials = []
        for g, n in enumerate(geom.sizes):
            if widths[g] < n:
                cand = widths.copy()
                cand[g] += 1
                p = geom.count(cand)
                if p <= cap:
                    trials.append((p, -g, cand))
        require(bool(trials), 'Nie udało się dopasować P do 0,1%')
        widths = max(trials)[2]
    require(999*cap <= 1000*geom.count(widths) <= 1000*cap, 'Złe dopasowanie P')
    return widths


def plans(cfg, frozen):
    out = {}
    for ds in DS:
        base = e49.train.model_for(ds, cfg)
        g = e49.e29.bn.Geometry(base)
        c = cfg['datasets'][ds]['classes']
        require(tuple(g.names) == e49.projector.GEOMETRY_NAMES and
                list(g.sizes) == list(e49.projector.SIZES), 'Zmieniona geometria grup')
        prior = frozen['geometry'][ds]['profiles']
        p0 = prior['full']['parameters']
        widths = {p: prior[p]['widths'] for p in CONTROLS}
        widths['b10_T'] = mild_at_p(g, prior['b10_U']['parameters'])
        widths['b5_U'] = e49.uniform(g, 95*p0//100)
        widths['b5_T'] = mild_at_p(g, g.count(widths['b5_U']))
        out[ds] = {}
        for p, w in widths.items():
            params, mac = map(int, e49.projector.costs(w, c))
            require(params == g.count(w) and all(1 <= a <= b for a,b in zip(w, g.sizes, strict=True)), 'Błędna geometria')
            if p in CONTROLS:
                require((params,mac) == (prior[p]['parameters'],prior[p]['mac']), 'Zmieniony koszt kontroli')
            elif p.startswith('b10'):
                require(1000*params <= 905*p0 and w not in [widths[k] for k in CONTROLS], 'TILT b10 nie jest odrębny lub za szeroki')
            out[ds][p] = {'widths': w, 'parameters': params, 'mac': mac}
        require(widths['b5_U'] != widths['b5_T'], 'Nierozróżnialne profile b5')
    return out


def plan_and_sources():
    _, sources = e50.provenance()
    train = read(e50.TRAINING/'manifest.json')
    m50, a50 = (read(RECROP/f) for f in ('manifest.json','audit.json'))
    require(m50['status'] == 'completed' and m50['exit_codes'] == [0]*4 and m50['dev_only'] and
            a50['status'] == 'passed' and a50['reproduced_E049'] == 16 and
            digest(RECROP/'frozen.json') == m50['frozen_sha256'], 'Nieaudytowany E050')
    recrop_source = str(E50.relative_to(ROOT))
    require(m50['sources'][recrop_source] == digest(E50), 'Zmieniony kod E050')
    frozen = read(e50.REFERENCE/'frozen.json')
    cfg = read(e50.TRAINING/'config.json')
    new_inputs = [Path(__file__), HERE/'OPIS.md', HERE/'URUCHOMIENIE.md', E50,
                  *(RECROP/f for f in ('manifest.json','audit.json','frozen.json'))]
    for rel,h in train['inputs'].items():
        if 'cifar100' in rel:
            require(digest(ROOT/rel) == h, f'Zmienione dane: {rel}')
            sources[rel] = h
    indices = e49.validated_indices()
    require(frozen['indices'] == indices, 'Zmienione indeksy E049')
    reference = {ds: {} for ds in DS}
    for s in SEEDS:
        prior = read(e50.REFERENCE/f'seed_{s}/ready.json')
        reference['cifar100'][str(s)] = {r['variant']: r for r in prior['records']
                                        if r['dataset'] == 'cifar100' and r['variant'] in (*CONTROLS, 'full_pre')}
        path = RECROP/f'seed_{s}.json'
        new_inputs.append(path)
        bundle = read(path)
        require(bundle['frozen_sha256'] == m50['frozen_sha256'], 'Zmieniony stan E050')
        tiny = {r['profile']: {'state_sha256': r['state_sha256'], 'calibration_views_sha256': r['views_sha256'],
                               'dev': {'correct': r['correct'], 'n': r['n']}}
                for r in bundle['rows'] if r['views'] == 'crop065' and r['seed'] == s}
        tiny['full_pre'], = (r for r in prior['records'] if r['dataset'] == 'tiny' and r['variant'] == 'full_pre')
        reference['tiny'][str(s)] = tiny
        for ds in DS:
            require(set(reference[ds][str(s)]) == {*CONTROLS,'full_pre'}, 'Niepełne kontrole')
    sources.update({str(f.relative_to(ROOT)): digest(f) for f in new_inputs})
    return {'geometry': plans(cfg, frozen), 'reference': reference, 'checkpoints': frozen['checkpoints'],
            'indices': indices, 'tiny_crop_min': .65, 'splits': {ds: 'dev' for ds in DS}}, sources


def work(run, gpu):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1, 'Złe GPU')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = SEEDS[gpu]
    m = read(run/'manifest.json')
    require(digest(run/'frozen.json') == m['frozen_sha256'], 'Zmieniony plan')
    f = read(run/'frozen.json')
    cfg = read(e50.TRAINING/'config.json')
    rows = []
    for ds in DS:
        model = e49.load_model(e50.TRAINING, f['checkpoints'][ds][str(seed)], ds, seed, cfg)
        spec = cfg['datasets'][ds]
        ref = f['reference'][ds][str(seed)]
        original = e49.state_hash(model)
        require(original == ref['full_pre']['state_sha256'], 'Niezgodny checkpoint')
        ids = f['indices'][ds]
        if ds == 'tiny':
            views, x, y = e50.tensors({'indices': ids, 'views': {'crop040': .40, 'crop065': f['tiny_crop_min']}})
            a, views_sha = views['crop065']
            del views
        else:
            a, x, y, views_sha = e49.old.data(ds, ids)
        require(len(y) == len(x) == N[ds], 'Zły dev')
        for profile in VARIANTS:
            plan = f['geometry'][ds][profile]
            folded = e49.fold(model, profile, plan)
            p, mac = e49.train.size_cost(folded, spec['input_size'], spec['classes'])
            require((p,mac) == (plan['parameters'],plan['mac']), 'Błędny koszt rzeczywisty')
            weights = e49.e29.bn.weights_hash(folded)
            bn = e49.e29.calibrate(folded.cuda(), a, 'CMA')
            state = e49.state_hash(folded)
            metrics = e49.score(folded, x, y, state)
            require(bn['params_unchanged'] and bn['samples'] == 4096 and bn['batches'] == 32 and
                    e49.e29.bn.weights_hash(folded) == weights, 'Zmienione wagi lub CMA')
            if profile in CONTROLS:
                row = ref[profile]
                require((views_sha,state,metrics['correct'],metrics['n']) ==
                        (row['calibration_views_sha256'],row['state_sha256'],row['dev']['correct'],row['dev']['n']),
                        f'Nie odtworzono {ds}/{profile}/{seed}')
            rows.append(dict(dataset=ds, seed=seed, gpu=gpu, variant=profile, parameters=p, mac=mac,
                             state_sha256=state, calibration_views_sha256=views_sha, **metrics))
            print(f"{ds} {profile} {metrics['top1_pct']:.4f}%", flush=True)
        require(e49.state_hash(model) == original, 'Zmieniono stan źródłowy')
    write(run/f'seed_{seed}.json', {'frozen_sha256': m['frozen_sha256'], 'rows': rows})


def summarize(run, m):
    rows = []
    for gpu, s in enumerate(SEEDS):
        bundle = read(run/f'seed_{s}.json')
        require(bundle['frozen_sha256'] == m['frozen_sha256'] and
                [(r['seed'],r['gpu'],r['dataset'],r['variant']) for r in bundle['rows']] ==
                [(s,gpu,ds,v) for ds in DS for v in VARIANTS], 'Niepełne wyniki')
        rows.extend(bundle['rows'])
    results = {}
    notes = ['# E051 — łagodny TILT przy dopasowanym P (dev)', '',
             'Cztery sparowane seedy. CIFAR-100: CMA E049; Tiny: crop 0,65 z E050.',
             'Odtworzenie stanów i trafności kontroli: 32/32.', '']
    for ds in DS:
        score = {v: [r for r in rows if r['variant'] == v and r['dataset'] == ds] for v in VARIANTS}
        require(all(len(v) == 4 for v in score.values()), 'Brak par')
        require(len({r['calibration_views_sha256'] for v in score.values() for r in v}) == 1, 'Różne widoki CMA')
        n = N[ds]
        hits = {v: [r['correct'] for r in s] for v,s in score.items()}
        for v, group in score.items():
            require(all(r['n'] == n and 0 <= r['correct'] <= n for r in group) and
                    len({(r['parameters'],r['mac']) for r in group}) == 1, 'Niespójne koszty lub dev')

        def contrasts(a,b):
            pp = [100*(x-y)/n for x,y in zip(hits[a], hits[b], strict=True)]
            return {'mean_pp': st.mean(pp), 'sample_sd_pp': st.stdev(pp), 'per_seed_pp': pp,
                    'positive_pairs': sum(x > 0 for x in pp)}

        loss = {v: contrasts('full', v)['mean_pp'] for v in VARIANTS}
        means = {v: 100*st.mean(a)/n for v,a in hits.items()}
        diff = {f'{a}-{b}': contrasts(a,b) for a,b in (('b10_T','b10_U'),('b10_T','b10_L06'),
                                                       ('b10_T','b10_Tproj'),('b5_T','b5_U'))}
        h = diff['b10_T-b10_U']
        passed = (100*(sum(hits['b10_T'])-sum(hits['b10_U'])) >= 4*n and h['positive_pairs'] >= 3 and
                  100*(sum(hits['full'])-sum(hits['b10_T'])) <= 12*n)
        results[ds] = dict(top1_mean_pct=means, loss_to_full_pp=loss, contrasts=diff, b10_dev_gate_met=passed)
        notes += [f'## {ds}', '', '| Wariant | Top‑1 [%] | Strata [pp] | P | MAC/obraz |',
                  '|---|---:|---:|---:|---:|']
        for v in VARIANTS:
            r = score[v][0]
            notes.append(f"| {v} | {means[v]:.4f} | {loss[v]:.4f} | {r['parameters']:,} | {r['mac']:,} |")
        notes += ['', f"**B10 TILT−UNIFORM: {h['mean_pp']:+.4f} pp ({h['positive_pairs']}/4); "
                  f"strata do pełnego: {loss['b10_T']:.4f} pp; kryterium dev: {'TAK' if passed else 'NIE'}.**", '']
    audit = dict(status='passed', dev_only=True, scores=len(rows), reproduced_controls=32, datasets=results,
                 joint_b10_dev_gate_met=all(d['b10_dev_gate_met'] for d in results.values()))
    with (run/'per_seed.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    notes.append(f"Łączny werdykt b10 na dev: **{'TAK' if audit['joint_b10_dev_gate_met'] else 'NIE'}** (tylko dopasowane P).")
    (run/'WYNIKI.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    write(run/'audit.json', audit)


def run_experiment():
    frozen, sources = plan_and_sources()
    gpus = e49.train.require_idle_gpus()
    run = ROOT/'runs/E051'/f'dev_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    (run/'logs').mkdir(parents=True, exist_ok=False)
    write(run/'frozen.json', frozen)
    m = dict(status='running', pid=os.getpid(), started_utc=utc(), sources=sources,
             frozen_sha256=digest(run/'frozen.json'), dev_only=True, gpus=gpus, workers=[], limit_seconds=LIMIT,
             python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
             torchvision=e50.torchvision.__version__, operator=e49.e29.COMMIT,
             command=f'CUDA_VISIBLE_DEVICES=0,1,2,3 {sys.executable} -B {Path(__file__).relative_to(ROOT)}')
    write(run/'manifest.json', m)
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
            m['workers'].append(dict(pid=p.pid, seed=seed, gpu=gpu))
        write(run/'manifest.json', m)
        while (codes := [p.poll() for p in procs]) != [0]*4:
            require(all(code in (None,0) for code in codes), f'Błąd workera {codes}')
            require(time.monotonic()-start < LIMIT, 'Przekroczono limit E051')
            time.sleep(1)
        summarize(run, m)
        m['status'] = 'completed'
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
        m.update(exit_codes=[p.returncode for p in procs], finished_utc=utc(), wall_seconds=time.monotonic()-start)
        write(run/'manifest.json', m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(4))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.check:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Kontrola CPU wymaga ukrycia GPU')
        shapes = plans(read(e50.TRAINING/'config.json'), read(e50.REFERENCE/'frozen.json'))
        for ds, profiles in shapes.items():
            for p, r in profiles.items():
                print(ds, p, r['parameters'], r['mac'], r['widths'])
    elif args.worker is not None:
        work(args.run, args.worker)
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        run_experiment()
