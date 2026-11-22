"""2026-11-22_E044_l06_tiny: geometria L06, U i T-proj oraz audyt metadanych.
Czyta plany E035/E037, zamrożone przesunięcia E043 i checkpointy E019.
Zapisuje status CPU, szerokości, koszty i hashe do runs/E044; zob. OPIS.md.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
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
F43 = ROOT / 'runs/E043/cpu_20261121T230452Z_8b558297/frozen.json'
F43_SHA = 'f84a40c4d4b435495904723aef61664ef11b359de4bee12ecc9160d6d81ded81'
SOLVER = ROOT / 'runs/E042/cpu_20261121T211306Z_957fea21/geometry_E041.py'
R35 = ROOT / 'runs/E035/20261107T183831Z_8b0b14e0'
F37 = HERE.parent / '2026-11-08_E037_kontrola_mac/frozen.json'
RT = HERE.parent / '2026-11-08_E038_operator_alokacja/runtime.py'
R19 = ROOT / 'runs/E019/20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7'
DATA = ROOT / 'data/tiny-imagenet'
SPLIT, WNIDS = DATA / 'splits.json', DATA / 'tiny-imagenet-200/wnids.txt'
TRAIN, VAL = DATA / 'cache_train.npz', DATA / 'cache_val.npz'
PAIR, TAGS, SEEDS = 'resnet_tiny', ('b50', 'b30'), (42, 43, 44, 45)
CLASSES, PROFILES = 200, ('U', 'T-proj', 'L06')
CKPT = {s: R19 / f'seed_{s}/last.pt' for s in SEEDS}
ALLOWED = {SPLIT, WNIDS, *CKPT.values()}
OPENED, DENIED = set(), []


def rel(path):
    return str(path.relative_to(ROOT))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT):
        return
    if path.is_relative_to(ROOT / 'data') or path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors'):
        if path not in ALLOWED or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
            DENIED.append(rel(path))
            raise RuntimeError(f'E044 CPU: niedozwolony odczyt/zapis {path}')
        OPENED.add(path)


def inputs(manifest):
    hashes = manifest['input_sha256']

    def record(path, expected=None):
        blob = path.read_bytes()
        h = sha(blob)
        assert expected is None or h == expected, rel(path)
        hashes[rel(path)] = h
        return blob

    def read(path, expected=None):
        return json.loads(record(path, expected))

    f43 = read(F43, F43_SHA)
    old_hashes = f43['input_sha256']
    f42 = read(R42 / 'frozen.json', old_hashes[rel(R42 / 'frozen.json')])
    run42 = read(R42 / 'manifest.json', old_hashes[rel(R42 / 'manifest.json')])
    f35 = read(R35 / 'frozen.json', old_hashes[rel(R35 / 'frozen.json')])
    f37 = read(F37, old_hashes[rel(F37)])
    record(SOLVER, old_hashes[rel(SOLVER)])
    assert set(f42['bank']['pairs']) == {'resnet_cifar10', 'resnet_cifar100'}
    assert f43['b50_exact_E042'] and f43['pair'] == 'resnet_cifar100'
    for bank in f42['bank']['pairs'].values():
        trial, = [p for p in bank['proposals'] if p['profile'] == 'L06']
        assert trial['accepted'] and trial['offsets_pp'] == f43['offsets_pp']
    assert f43['groups'] == f42['bank']['names'] and f43['sizes'] == f42['bank']['sizes']

    meta = read(R19 / 'data_manifest.json')
    split = read(SPLIT, meta['split_sha256'])
    assert split == read(R19 / 'splits.json', meta['split_sha256'])
    wnids = record(WNIDS).decode().splitlines()
    assert split['classes'] == meta['classes'] == sorted(wnids) and len(set(wnids)) == 200
    train, dev, val = (split['splits'][k] for k in ('train', 'dev', 'reserve'))
    assert (len(train), len(dev), len(val)) == (90000, 10000, 10000)
    st, sd = set(train), set(dev)
    assert len(st) == len(train) and len(sd) == len(dev) and not st & sd and st | sd == set(range(100000))
    assert meta['reserve_role'].startswith('oficjalny val w cache_val.npz') and not meta['official_test_loaded']
    ids = f35['indices']['tiny']
    assert ids['source'] == rel(TRAIN) and ids['source_sha256'] == meta['cache_train_sha256']
    assert ids['split'] == rel(SPLIT) and ids['split_sha256'] == meta['split_sha256']
    assert len(set(ids['A'])) == len(ids['A']) == 4096 and set(ids['A']) <= st and ids['dev'] == dev
    prior_test = read(R35 / 'test_data.json')
    audit35 = read(R35 / 'analysis/data_verification.json')
    assert audit35['status'] == 'PASS' and audit35['tiny_pixels_equal'] == audit35['tiny_labels_equal'] == 10000
    assert prior_test['indices']['tiny'] == val == list(range(10000))
    assert prior_test['files'][rel(VAL)] == audit35['source_sha256'][rel(VAL)] != meta['cache_train_sha256']
    assert TRAIN.is_file() and VAL.is_file()
    views = {read(R35 / f'dev/seed_{s}/{PAIR}/controls.json')['views_sha256'] for s in SEEDS}
    assert len(views) == 1

    checkpoints = {}
    for s, path in CKPT.items():
        sr = read(path.with_name('summary.json'))
        assert sr['seed'] == s and sr['status'] == 'completed' and sr['final_epoch'] == 200
        assert sr['selection'] == 'final_epoch_no_best' and sr['best_checkpoint'] is None
        assert not sr['reserve_used'] and not sr['official_test_used']
        assert sr['data_manifest_sha256'] == hashes[rel(R19 / 'data_manifest.json')]
        assert sr['split_sha256'] == meta['split_sha256'] and sr['train_count'] == len(train)
        h = digest(path)
        assert h == sr['last_checkpoint_sha256'] == f35['provenance'][rel(path)]
        hashes[rel(path)] = h
        checkpoints[str(s)] = dict(path=rel(path), sha256=h)

    deps = {p: h for p, h in (f42['hashes'] | run42['code_hashes']).items() if p.endswith('.py')}
    for p, h in deps.items():
        assert digest(ROOT / p) == h, p
    assert rel(RT) in deps
    manifest['verified_dependency_hashes'] = len(deps)
    data = dict(calibration=dict(cache=rel(TRAIN), sha256=meta['cache_train_sha256'], A=ids['A'],
                                 views_sha256=views.pop(), normalization_mean=meta['normalization_mean'],
                                 normalization_std=meta['normalization_std']),
                evaluation=dict(cache=rel(VAL), sha256=prior_test['files'][rel(VAL)], indices='0..9999',
                                count=10000, previously_used_by='E035', fresh_project_test=False),
                classes=meta['classes'], checkpoints=checkpoints,
                split_sha256=meta['split_sha256'], train_dev_disjoint=True,
                tiny_used_for_selecting_L06=False)
    return f43, f35, f37, data


def geometry(g, f43, f35, f37):
    np = g.np
    assert list(g.NAMES) == f43['groups'] and g.SIZES.tolist() == f43['sizes']
    offsets = np.asarray(f43['offsets_pp'], dtype=np.int64)
    cifar_cost = g.costs

    def tiny_cost(widths, classes):
        # Te same etapy; stem 7x7 zamiast 3x3, po maxpool pole warstw = 1/4 CIFAR.
        w = np.asarray(widths)
        p, m = np.moveaxis(cifar_cost(w, classes), -1, 0)
        heads = classes * w[..., 10]
        conv = m - 27648 * w[..., 0] - heads
        if np.issubdtype(w.dtype, np.integer):
            assert np.all(conv % 4 == 0)
            conv = conv // 4
        else:
            conv = conv / 4
        return np.stack((p + 120 * w[..., 0], conv + 150528 * w[..., 0] + heads), axis=-1)

    def costs(w):
        p, mac = map(int, tiny_cost(w, CLASSES))
        return dict(params=p, mac=mac)

    g.costs = tiny_cost  # bez modyfikacji zamrożonego pliku solvera
    budgets, solvers = {}, {}
    for tag in TAGS:
        plan, ref = f35['plans'][PAIR][tag], f37['plans'][PAIR][tag]
        assert (plan['U'], plan['T'], plan['ratio']) == (ref['U'], ref['T'], ref['ratio'])
        u = np.array(plan['U'], dtype=np.int64)
        assert np.array_equal((g.SIZES * (1 - plan['ratio'])).astype(int), u)
        caps = np.array([plan['params']['U'], ref['U_mac']], dtype=np.int64)
        assert np.array_equal(tiny_cost(u, CLASSES), caps)
        assert costs(plan['T']) == dict(params=plan['params']['T'], mac=ref['T_mac'])
        proposal = np.clip(np.rint(u + offsets * g.SIZES / 100), g.LOW, g.HIGH).astype(np.int64)
        widths, solvers[tag] = {'U': u}, {}
        for profile, raw in (('T-proj', plan['T']), ('L06', proposal)):
            w, diagnostic = g.project(raw, u, caps, CLASSES)
            solvers[tag][profile] = diagnostic
            assert w is not None, (tag, profile, diagnostic)
            widths[profile] = w
        for name, w in widths.items():
            assert np.all(g.LOW <= w) and np.all(w <= g.HIGH) and g.inside(tiny_cost(w, CLASSES), caps), (tag, name)
        distances = {f'{a} vs {b}': g.rms(widths[a], widths[b]) for a, b in itertools.combinations(PROFILES, 2)}
        assert min(distances.values()) > 0
        budgets[tag] = dict(caps=costs(u), uniform_plan=plan, L06_proposal=proposal.tolist(), rms_kn=distances,
                            profiles={k: dict(widths=w.tolist(), **costs(w)) for k, w in widths.items()})
    return dict(groups=f43['groups'], sizes=f43['sizes'], offsets_pp=f43['offsets_pp'], offsets_kn=f43['offsets_kn'],
                teacher=costs(g.SIZES), budgets=budgets), solvers


def verify_models(rt, g, spec, data, checks):
    e35, torch = rt.e35, rt.torch
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    up = e35.e29.UP
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=up, text=True).strip() == e35.e29.COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=up, text=True).strip()
    assert list(e35.e30.MEAN) == data['calibration']['normalization_mean']
    assert list(e35.e30.STD) == data['calibration']['normalization_std']
    base = e35.model_source(PAIR, 42)
    geo = rt.e37.geometry(PAIR, base)
    axes = e35.e29.get_axis_to_perm_ResNet18(override=False)
    upstream = e35.e29.axes2perm_to_perm2axes(axes)
    assert tuple(upstream) == g.NAMES and tuple(geo.names) == g.GEOMETRY_NAMES
    assert list(geo.sizes) == spec['sizes'] and list(upstream.values()) == [geo.members[k] for k in geo.names]
    assert base.conv1.kernel_size == (7, 7) and base.conv1.stride == (2, 2) and base.fc.out_features == 200
    before = e35.state_hash(base)

    def realized(model):
        return dict(params=sum(p.numel() for p in model.parameters()), mac=rt.e37.mac(model, PAIR))

    assert (teacher := realized(base)) == spec['teacher']
    checks.append(dict(seed=42, profile='teacher', **teacher))
    for tag, budget in spec['budgets'].items():
        for profile, row in budget['profiles'].items():
            w = row['widths']
            if profile == 'U':
                model, diag = e35.fold(PAIR, base, budget['uniform_plan'], 'U')
            else:
                model, diag = rt.e37.fold_widths(PAIR, base, w)
            assert [d['k'] for d in diag] == w and [d['n'] for d in diag] == spec['sizes']
            for d, name in zip(diag, geo.names):
                assert d['first_axis'] == geo.members[name][0][0] and len(d['labels']) == d['n']
                assert set(d['labels']) == set(range(d['k']))
            assert (measured := realized(model)) == {k: row[k] for k in ('params', 'mac')}
            assert measured['params'] == geo.count(w) and torch.equal(model.fc.bias, base.fc.bias)
            assert all(p.device.type == 'cpu' and not p.requires_grad and p.grad is None for p in model.parameters())
            checks.append(dict(seed=42, budget=tag, profile=profile, **measured))
            print(tag, profile, measured, flush=True)
    assert e35.state_hash(base) == before and not rt.DENIED and not rt.e37.DENIED


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    assert not sys.flags.optimize and sys.dont_write_bytecode
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert '/envs/attention/' in sys.executable
    sys.addaudithook(guard)
    start = time.monotonic()
    now = datetime.now(timezone.utc)
    run = ROOT / 'runs/E044' / f'cpu_{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    (run / 'source').mkdir(parents=True)
    files = [HERE / n for n in ('kontrola_cpu.py', 'OPIS.md', 'URUCHOMIENIE.md')]
    for p in files:
        shutil.copyfile(p, run / 'source' / p.name)
    m = dict(status='running', started_utc=now.isoformat(), run=rel(run), pid=os.getpid(),
             command=[sys.executable, '-B', *sys.argv], python=sys.version,
             source_sha256={p.name: digest(p) for p in files}, input_sha256={}, checks=[],
             quality_evaluated=False, bn_calibrated=False, training=False, image_file_opened=False)
    write(run / 'manifest.json', m)
    print('RUN', rel(run), flush=True)
    try:
        f43, f35, f37, data = inputs(m)
        g = load(SOLVER, '_e044_e042_solver')
        m.update(numpy=g.np.__version__, scipy=g.scipy.__version__)
        assert (m['numpy'], m['scipy']) == ('2.4.6', '1.17.1')
        spec, m['projection'] = geometry(g, f43, f35, f37)
        rt = load(RT, '_e044_runtime')
        m['torch'] = str(rt.torch.__version__)
        verify_models(rt, g, spec, data, m['checks'])
        assert OPENED == ALLOWED and not DENIED and len(m['checks']) == 7
        frozen = dict(pair=PAIR, upstream_commit=rt.e35.e29.COMMIT, **spec, data=data, input_sha256=m['input_sha256'],
                      protocol_sha256=m['source_sha256']['OPIS.md'], cpu_code_sha256=m['source_sha256']['kontrola_cpu.py'],
                      planned_evaluation=dict(gpu_to_seed=dict(enumerate(SEEDS)), mode='CMA_reset', batch_per_gpu=128,
                                              rows=28, metric='top1', primary_budget='b50', transfer_budget='b30',
                                              contrast='L06-U', minimum_strictly_positive_seeds=3,
                                              bootstrap_draws=10000, bootstrap_seed=20261122))
        write(run / 'frozen.json', frozen)
        m.update(status='passed', frozen_sha256=digest(run / 'frozen.json'))
        print('CPU PASS 6 profili + pełny model; 4/4 hashe wag', m['frozen_sha256'], flush=True)
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        m.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
                 files_opened=sorted(map(rel, OPENED)), denied=DENIED)
        write(run / 'manifest.json', m)


if __name__ == '__main__':
    main()
