"""2026-11-21_E043_l06_potwierdzenie_transfer: audyt wejść i geometrii na CPU.
Odtwarza U/T-proj/L06 z zamrożonych planów, sprawdza indeksy i cztery wagi E015.
Zapisuje manifest i frozen.json w runs/E043; bez jakości lub danych obrazowych.
Pełny protokół: OPIS.md.
"""

import argparse
import hashlib
import importlib.util
import itertools
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
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R42 = ROOT / 'runs/E042/20261121T212257Z_207ae03f'
CPU42 = ROOT / 'runs/E042/cpu_20261121T211306Z_957fea21'
E42 = HERE.parent / '2026-11-21_E042_zapas_pod_uniform'
R35 = ROOT / 'runs/E035/20261107T183831Z_8b0b14e0'
E37 = HERE.parent / '2026-11-08_E037_kontrola_mac'
R24 = ROOT / 'runs/E024/20261031T141558Z_971ba86e'
R15 = ROOT / 'runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12'
SPLIT = ROOT / 'data/cifar100/splits.json'
TRAIN = ROOT / 'data/cifar100/cifar-100-python/train'
TEST = TRAIN.with_name('test')
F42_SHA = '2069b0d002ddc892be39bc6b24d66f4d059f1ee143f829d43a19a869c54da7c4'
UPSTREAM_COMMIT = '694691556e8314a835178a741938dd3d4d80c171'
PAIR, DS = 'resnet_cifar100', 'cifar100'
SEEDS, TAGS, PROFILES = (42, 43, 44, 45), ('b50', 'b30'), ('U', 'T-proj', 'L06')
CHECKPOINTS = {s: R15 / f'seed_{s}/last.pt' for s in SEEDS}
ALLOWED = {SPLIT, *CHECKPOINTS.values()}
OPENED, DENIED = set(), []
LIMIT = 600


def rel(path):
    return str(path.relative_to(ROOT))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha256(path.read_bytes())


def read(path, hashes, expected=None):
    data = path.read_bytes()
    actual = sha256(data)
    if expected is not None:
        assert actual == expected, rel(path)
    hashes[rel(path)] = actual
    return json.loads(data)


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    protected = path.is_relative_to(ROOT / 'data') or path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors')
    if protected and path.is_relative_to(ROOT):
        if path not in ALLOWED or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
            DENIED.append(rel(path))
            raise RuntimeError(f'E043 CPU: niedozwolony dostęp do {rel(path)}')
        OPENED.add(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def index_hash(ids):
    return sha256(json.dumps(ids, separators=(',', ':')).encode())


def audit_data(f42, f35, run42, hashes):
    split_hash = f42['hashes'][rel(SPLIT)]
    train_hash = f42['hashes'][rel(TRAIN)]
    parts = read(SPLIT, hashes, split_hash)['splits']
    assert parts == read(R15 / 'splits.json', hashes, split_hash)['splits']
    sizes = {'train': 40000, 'dev': 5000, 'reserve': 5000}
    assert set(parts) == set(sizes)
    sets = {name: set(ids) for name, ids in parts.items()}
    assert all(type(i) is int for ids in parts.values() for i in ids)
    assert all(len(parts[k]) == len(sets[k]) == n for k, n in sizes.items())
    assert set.union(*sets.values()) == set(range(50000))
    assert all(not a & b for a, b in itertools.combinations(sets.values(), 2))

    train = read(R15 / 'manifest.json', hashes)
    data = read(R15 / 'data_manifest.json', hashes, train['data_manifest_sha256'])
    assert train['status'] == 'completed' and train['worker_exit_codes'] == {str(s): 0 for s in SEEDS}
    for meta in (train['data'], data):
        assert meta['reserve_used'] is meta['official_test_used'] is False
        assert meta['split_sha256'] == split_hash
        assert all(meta[f'{k}_count'] == n for k, n in sizes.items())
    assert data['official_test_loaded'] is False
    assert data['train_file_sha256'] == train_hash == f35['provenance'][rel(TRAIN)]

    idx_file = R24 / 'indices.json'
    idx24 = read(idx_file, hashes, f35['provenance'][rel(idx_file)])
    idx42 = f42['indices'][DS]
    meta24 = read(R24 / 'manifest.json', hashes)
    assert meta24['reserve_used'] is meta24['official_test_used'] is False and meta24['status'] == 'completed'
    assert idx42 == f35['indices'][DS] and idx42['source'] == rel(idx_file)
    assert meta24['indices_sha256'] == hashes[rel(idx_file)] == idx42['source_sha256']
    assert idx24['dev'] == idx42['dev'] == parts['dev'] and idx24['A'] == idx42['A']
    calibration = {k: set(idx24[k]) for k in ('A', 'B', 'C')}
    for key, n in (('A', 4096), ('B', 2048), ('C', 2048)):
        assert len(calibration[key]) == len(idx24[key]) == n and calibration[key] <= sets['train']
    assert all(not a & b for a, b in itertools.combinations(calibration.values(), 2))
    assert f42['splits'][DS] == dict(A=4096, dev=5000, reserve_in_A_or_dev=0)

    # Odczyt wyłącznie metadanych E035; numery testu należą do innego pliku.
    test35 = read(R35 / 'test_data.json', hashes)
    audit35 = read(R35 / 'analysis/data_verification.json', hashes)
    run35 = read(R35 / 'manifest.json', hashes)
    assert run35['status'] == 'completed' and run35['phases']['test']['exit_codes'] == [0] * 4
    assert run35['frozen_sha256'] == hashes[rel(R35 / 'frozen.json')]
    assert test35['cifar100_reserve_used'] is audit35['cifar100_reserve_used'] is False
    assert audit35['status'] == 'PASS' and test35['indices'][DS] == list(range(10000))
    assert test35['files'][rel(TEST)] != train_hash

    audit = read(R42 / 'analysis/audit.json', hashes)
    assert run42['status'] == 'completed' and run42['settings']['phase'] == 'dev'
    assert run42['frozen_sha256'] == F42_SHA and audit['manifest_sha256'] == hashes[rel(R42 / 'manifest.json')]
    assert audit['passed'] and 'A_and_dev_no_reserve' in audit['checks']
    ckpts, views = {}, set()
    for seed, path in CHECKPOINTS.items():
        worker = train['workers'][str(seed)]
        s = read(path.with_name('summary.json'), hashes, worker['summary_sha256'])
        assert s['status'] == 'completed' and s['seed'] == seed and s['final_epoch'] == 200
        assert s['selection'] == 'final_epoch_no_best' and s['best_checkpoint'] is None
        assert s['reserve_used'] is s['official_test_used'] is False
        assert s['split_sha256'] == split_hash and s['data_manifest_sha256'] == hashes[rel(R15 / 'data_manifest.json')]
        assert s['train_count'] == sizes['train'] and s['dev_count'] == sizes['dev']
        assert s['normalization_mean'] == data['normalization_mean'] and s['normalization_std'] == data['normalization_std']
        assert s['last_checkpoint'] == path.name
        ck = digest(path)
        assert ck == f42['hashes'][rel(path)] == s['last_checkpoint_sha256'] == worker['last_checkpoint_sha256']
        item = R42 / f'seed_{seed}/{PAIR}/L06.json'
        prior = read(item, hashes, audit['source_hashes'][rel(item)])
        assert prior['phase'] == 'dev' and prior['seed'] == seed and prior['checkpoint_sha256'] == ck
        assert prior['reserve_used'] is prior['official_test_used'] is False
        views.add(prior['views_sha256'])
        hashes[rel(path)] = ck
        ckpts[str(seed)] = dict(path=rel(path), sha256=ck)
    assert len(views) == 1
    return dict(index_space=rel(TRAIN), expected_train_sha256=train_hash,
                split=rel(SPLIT), split_sha256=split_hash, reserve_key='splits.reserve',
                sizes=sizes | {k: len(v) for k, v in calibration.items()},
                reserve_intersection_counts={k: len(sets['reserve'] & v) for k, v in
                                             (sets | calibration).items() if k != 'reserve'},
                index_list_sha256={k: index_hash(v) for k, v in
                                   (parts | {k: idx24[k] for k in calibration}).items()},
                calibration_source=idx42['source'], expected_A_views_sha256=views.pop(),
                normalization=dict(mean=data['normalization_mean'], std=data['normalization_std']),
                used_E035_test=dict(file=rel(TEST), samples=10000, sha256=test35['files'][rel(TEST)]),
                checkpoints=ckpts)


def project_pair(f42, f35, f37, bank, g):
    np = g.np
    assert f42['bank'] == bank and bank['passed']
    assert tuple(bank['names']) == g.NAMES and bank['sizes'] == g.SIZES.tolist()
    old = bank['pairs'][PAIR]
    trial, = [x for x in old['proposals'] if x['profile'] == 'L06']
    archived, = [x for x in old['candidates'] if x['profile'] == 'L06']
    assert trial['accepted'] and trial['result'] == archived
    offsets = np.asarray(trial['offsets_pp'], dtype=np.int64)
    assert offsets.shape == g.SIZES.shape

    def stats(widths):
        p, mac = map(int, g.costs(widths, 100))
        return dict(params=p, mac=mac)

    budgets, solvers = {}, {}
    for tag in TAGS:
        plan, ref = f35['plans'][PAIR][tag], f37['plans'][PAIR][tag]
        for profile in ('U', 'T'):
            assert plan[profile] == ref[profile]
        assert plan['ratio'] == ref['ratio']
        assert stats(plan['U']) == dict(params=plan['params']['U'], mac=ref['U_mac'])
        assert stats(plan['T']) == dict(params=plan['params']['T'], mac=ref['T_mac'])
        u = np.array(plan['U'], dtype=np.int64)
        assert (np.asarray(g.SIZES * (1 - plan['ratio']), dtype=np.int64) == u).all()
        caps = np.array([plan['params']['U'], ref['U_mac']], dtype=np.int64)
        proposal = np.clip(np.rint(u + offsets * g.SIZES / 100), g.LOW, g.HIGH).astype(np.int64)
        solvers[tag] = {}
        widths = {'U': u}
        for name, raw in (('T-proj', plan['T']), ('L06', proposal)):
            widths[name], solvers[tag][name] = g.project(raw, u, caps, 100)
            assert widths[name] is not None, (tag, name, solvers[tag][name])
        rows = {}
        for name, w in widths.items():
            assert (g.LOW <= w).all() and (w <= g.HIGH).all()
            assert g.inside(g.costs(w, 100), caps), (tag, name)
            rows[name] = dict(profile=name, widths=w.tolist(), **stats(w))
        distances = {f'{a} vs {b}': g.rms(widths[a], widths[b]) for a, b in itertools.combinations(PROFILES, 2)}
        assert min(distances.values()) > 0
        if tag == 'b50':
            assert plan == f42['plans'][PAIR] and caps.tolist() == old['caps']
            assert proposal.tolist() == trial['proposal'] and min(distances.values()) >= .025
            assert rows == {r['profile']: {k: v for k, v in r.items() if k != 'family'}
                            for r in old['references'] + [archived]}
        budgets[tag] = dict(UNIFORM_plan=plan, caps=stats(u), L06_proposal=proposal.tolist(),
                            profiles=rows, rms_kn=distances)
    return dict(groups=list(g.NAMES), sizes=g.SIZES.tolist(), offsets_pp=offsets.tolist(),
                offsets_kn=(offsets / 100).tolist(), teacher=stats(g.SIZES), budgets=budgets,
                b50_exact_E042=True), solvers


def verify_models(geometry, e42, manifest, out, start):
    e42.cpu_only()
    rt, e35 = e42.rt, e42.e35
    torch = rt.torch
    assert not torch.cuda.is_available()
    up = e35.e29.UP
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=up, text=True).strip() == UPSTREAM_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=up, text=True).strip()
    manifest['torch'] = str(torch.__version__)
    for seed in SEEDS:
        base = e35.model_source(PAIR, seed)
        geo = e42.verify_group_map(PAIR, base)
        before = e35.state_hash(base)
        assert base.fc.out_features == 100 and not base.training
        teacher = dict(params=sum(p.numel() for p in base.parameters()), mac=rt.e37.mac(base, PAIR))
        assert teacher == geometry['teacher']
        manifest['checks'].append(dict(seed=seed, profile='teacher', **teacher))
        for tag, budget in geometry['budgets'].items():
            for name in PROFILES:
                row = budget['profiles'][name]
                model, actual, diag = e42.build(PAIR, base, row, budget['UNIFORM_plan'], geo)
                assert not model.training and all(p.device.type == 'cpu' for p in model.parameters())
                manifest['checks'].append(dict(seed=seed, budget=tag, profile=name,
                                               widths=[d['k'] for d in diag], **actual))
                del model
                assert time.monotonic() - start < LIMIT, 'Limit CPU 600 s'
        assert e35.state_hash(base) == before
        print(f'CPU seed {seed}: PASS (6 profili + teacher)', flush=True)
        write(out / 'manifest.json', manifest)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Ukryj GPU dla kontroli CPU'
    assert not sys.flags.optimize and sys.dont_write_bytecode, 'Uruchom z -B, bez -O'
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and '/envs/attention/' in sys.executable
    sys.addaudithook(guard)
    start = time.monotonic()
    now = datetime.now(timezone.utc)
    out = ROOT / 'runs/E043' / f'cpu_{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    (out / 'source').mkdir(parents=True)
    sources = [HERE / name for name in ('kontrola_cpu.py', 'OPIS.md', 'URUCHOMIENIE.md')]
    for p in sources:
        shutil.copy2(p, out / 'source' / p.name)
    manifest = dict(status='running', phase='cpu', run=rel(out), started_utc=now.isoformat(),
                    pid=os.getpid(), command=[sys.executable, '-B', *sys.argv],
                    environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CONDA_DEFAULT_ENV',
                                                                 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    python=platform.python_version(), source_hashes={rel(p): digest(p) for p in sources},
                    checks=[], quality_evaluated=False, bn_calibrated=False, training=False,
                    reserve_used=False, official_test_used=False)
    write(out / 'manifest.json', manifest)
    print('RUN', rel(out), flush=True)
    try:
        hashes = {}
        f42 = read(R42 / 'frozen.json', hashes, F42_SHA)
        f35 = read(R35 / 'frozen.json', hashes, f42['hashes'][rel(R35 / 'frozen.json')])
        f37 = read(E37 / 'frozen.json', hashes, f42['hashes'][rel(E37 / 'frozen.json')])
        assert f42['geometry_run'] == rel(CPU42)
        cpu42 = read(CPU42 / 'manifest.json', hashes, f42['hashes'][rel(CPU42 / 'manifest.json')])
        assert cpu42['status'] == 'completed' and cpu42['quality_used'] is False
        bank = read(CPU42 / 'bank.json', hashes, cpu42['bank_sha256'])
        assert cpu42['bank_sha256'] == f42['hashes'][rel(CPU42 / 'bank.json')]
        snapshot = CPU42 / 'geometry_E041.py'
        hashes[rel(snapshot)] = digest(snapshot)
        solver_key = 'experiments/2026-11-15_E041_kontrola_p_mac/geometry.py'
        assert hashes[rel(snapshot)] == cpu42['hashes'][solver_key] == f42['hashes'][solver_key]
        g = load(snapshot, '_e043_geometry_e041')
        manifest.update(numpy=g.np.__version__, scipy=g.scipy.__version__)
        assert g.np.__version__ == cpu42['numpy'] and g.scipy.__version__ == cpu42['scipy']
        geometry, solvers = project_pair(f42, f35, f37, bank, g)
        manifest['projection'] = solvers
        run42 = read(R42 / 'manifest.json', hashes)
        data = audit_data(f42, f35, run42, hashes)
        manifest['data'] = data

        # Kod pomocniczy zgodny z zamrożonym runem; uruchamiamy tylko konstrukcję.
        deps = {p: h for p, h in f42['hashes'].items() if p.endswith('.py')}
        deps.update({p: h for p, h in run42['code_hashes'].items() if p.endswith('.py')})
        for name, expected in deps.items():
            assert digest(ROOT / name) == expected, name
        manifest['checked_dependency_hashes'] = len(deps)
        load(E42 / 'geometria.py', 'geometria')
        e42 = load(E42 / 'experiment.py', '_e043_e042_operator')
        verify_models(geometry, e42, manifest, out, start)
        assert len(manifest['checks']) == 28 and OPENED == ALLOWED and not DENIED and not e42.denied()

        frozen = dict(protocol_sha256=manifest['source_hashes'][rel(HERE / 'OPIS.md')], pair=PAIR,
                      cpu_code_sha256=manifest['source_hashes'][rel(HERE / 'kontrola_cpu.py')],
                      upstream_commit=UPSTREAM_COMMIT, **geometry, data=data, input_sha256=hashes,
                      planned_evaluation=dict(gpu_to_seed=dict(enumerate(SEEDS)), mode='CMA_reset',
                                              batch_per_gpu=128, samples=5000, rows=28,
                                              requires_separate_approval=True),
                      decision=dict(primary_budget='b50', transfer_budget='b30', contrast='L06-U',
                                    metric='top1_CMA', min_gain_pp=1.0, min_sum_count_gain=200,
                                    min_strictly_positive_seeds=3))
        write(out / 'frozen.json', frozen)
        manifest.update(status='passed', frozen_sha256=digest(out / 'frozen.json'))
        print('CPU PASS 24 modeli skompresowanych + 4 teachery', manifest['frozen_sha256'], flush=True)
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - start,
                        files_opened=sorted(map(rel, OPENED)), denied=DENIED)
        write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
