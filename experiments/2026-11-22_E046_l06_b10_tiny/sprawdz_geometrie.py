"""2026-11-22_E046_l06_b10_tiny: kontrola geometrii jednego L06 przy b10.
Czyta zamrożone metadane E043/E045 i wagi E019; zapisuje geometrię, manifest
i (po PASS) plan do runs/E046. Bez obrazów i trafności; zob. OPIS.md.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R19 = ROOT / 'runs/E019/20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7'
R42 = ROOT / 'runs/E042/20261121T212257Z_207ae03f'
F43 = ROOT / 'runs/E043/cpu_20261121T230452Z_8b558297/frozen.json'
R45 = ROOT / 'runs/E045/20261122T031350Z_a58b56a3'
F45_SHA = 'a7eaca3b8289a6c80784ef992e7ebe39c284ecfe344d98e80e3d18bd09b8dcfd'
AUDIT45_SHA = 'fcca52a0fbd3020931b34d275f6fb84f278ac1d47f6575a850c5c0d3df20ad47'
MANIFEST45_SHA = 'e0ffa594d74af6d974e7290538f0b25bb38602473245707eb06c563c1afffc67'
SOLVER = ROOT / 'runs/E042/cpu_20261121T211306Z_957fea21/geometry_E041.py'
E37 = HERE.parent / '2026-11-08_E037_kontrola_mac/experiment.py'
UP = ROOT / 'runs/E029/upstream_6946915'
COMMIT = '694691556e8314a835178a741938dd3d4d80c171'
SPLIT = ROOT / 'data/tiny-imagenet/splits.json'
SEEDS, PAIR, CLASSES = (42, 43, 44, 45), 'resnet_tiny', 200
CKPT = {seed: R19 / f'seed_{seed}/last.pt' for seed in SEEDS}


def rel(path):
    return str(path.relative_to(ROOT))


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def verify(path, expected, hashes):
    actual = digest(path)
    assert actual == expected, (rel(path), actual, expected)
    hashes[rel(path)] = actual


def read(path, expected, hashes):
    verify(path, expected, hashes)
    return json.loads(path.read_bytes())


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def input_metadata(hashes):
    f45 = read(R45 / 'frozen.json', F45_SHA, hashes)
    a45 = read(R45 / 'analysis/audit.json', AUDIT45_SHA, hashes)
    r45 = read(R45 / 'manifest.json', MANIFEST45_SHA, hashes)
    assert a45['status'] == 'PASS' and a45['phase'] == 'dev' and a45['four_gpu_exit0']
    assert not a45['val_used'] and a45['frozen_sha256'] == r45['frozen_sha256'] == F45_SHA
    assert r45['status'] == 'completed' and r45['assessments'] == 12
    assert f45['settings'] == dict(seeds=list(SEEDS), batch_per_gpu=128, mode='CMA_reset')
    prior = f45['input_sha256']
    f43 = read(F43, prior[rel(F43)], hashes)
    assert f43['b50_exact_E042'] and f43['pair'] == 'resnet_cifar100'
    verify(SOLVER, prior[rel(SOLVER)], hashes)
    f42 = read(R42 / 'frozen.json', prior[rel(R42 / 'frozen.json')], hashes)
    for path, expected in f42['hashes'].items():
        if path.endswith('.py'):
            verify(ROOT / path, expected, hashes)
    assert rel(E37) in hashes

    cal = f45['calibration']
    assert cal['cache'] == 'data/tiny-imagenet/cache_train.npz'
    split = read(SPLIT, prior[rel(SPLIT)], hashes)['splits']
    train, dev, a = set(split['train']), set(f45['dev']), set(cal['A'])
    assert f45['dev'] == split['dev'] and len(dev) == len(f45['dev']) == 10000
    assert len(train) == 90000 and not train & dev
    assert len(a) == len(cal['A']) == 4096 and a <= train and not a & dev
    assert set(f45['checkpoints']) == set(map(str, SEEDS))
    for seed, path in CKPT.items():
        ckpt = f45['checkpoints'][str(seed)]
        assert ckpt['path'] == rel(path) and ckpt['sha256'] == prior[rel(path)]
        verify(path, ckpt['sha256'], hashes)
    refs = {f'seed_{seed}/metrics.json': r45['output_sha256'][f'seed_{seed}/metrics.json']
            for seed in SEEDS}
    return f43, f45, refs


def project_once(g, f43, f45):
    np = g.np
    n, old_high = g.SIZES, g.HIGH
    assert list(g.NAMES) == f43['groups'] and n.tolist() == f43['sizes']
    assert np.array_equal(g.LOW, n // 10) and np.array_equal(old_high, 9 * n // 10)
    offsets = np.asarray(f43['offsets_pp'], dtype=np.int64)
    cifar_costs = g.costs

    def tiny_costs(widths, classes):
        # Korekta CIFAR -> Tiny64 jak w E044: zmieniony stem 7x7;
        # po maxpool pole map bloków to 1/4 pola CIFAR.
        w = np.asarray(widths)
        p, m = np.moveaxis(cifar_costs(w, classes), -1, 0)
        head = classes * w[..., 10]
        conv = m - 27648 * w[..., 0] - head
        if np.issubdtype(w.dtype, np.integer):
            assert np.all(conv % 4 == 0)
            conv = conv // 4
        else:
            conv = conv / 4
        return np.stack((p + 120 * w[..., 0], conv + 150528 * w[..., 0] + head), axis=-1)

    def array(cost):
        return np.array([cost['params'], cost['mac']], dtype=np.int64)

    def cost(w):
        p, m = map(int, tiny_costs(w, CLASSES))
        return dict(params=p, mac=m)

    teacher, caps = f45['costs']['teacher'], f45['costs']['b10']
    u = np.asarray(f45['plans']['b10']['U'], dtype=np.int64)
    assert cost(n) == teacher and cost(u) == caps
    assert f45['plans']['b10']['params']['U'] == caps['params']
    target = array(caps)
    minimum = (999 * target + 999) // 1000
    old_cost = cost(old_high)
    assert np.all(array(old_cost) < minimum)

    g.costs, g.HIGH = tiny_costs, n  # Tylko ta instancja solvera; bez edycji historii.
    proposal = np.clip(np.rint(u + offsets * n / 100), g.LOW, g.HIGH).astype(np.int64)
    widths, solver = g.project(proposal, u, target, CLASSES)
    record = dict(groups=f43['groups'], sizes=n.tolist(), offsets_pp=offsets.tolist(),
                  low=g.LOW.tolist(), old_high=old_high.tolist(), high=g.HIGH.tolist(),
                  old_max_cost=old_cost, teacher=teacher, caps=caps,
                  minimum=dict(zip(('params', 'mac'), map(int, minimum))),
                  uniform=u.tolist(), proposal=proposal.tolist(), solver=solver, candidate=None)
    if widths is None:
        reason = ('naprawa całkowitoliczbowa nie znalazła punktu w oknie kosztu'
                  if solver['success'] and max(map(abs, solver['residual'])) <= 1e-7
                  else 'projekcja ciągła nie spełniła bramki zbieżności')
        return dict(record, passed=False, stop_reason=reason)

    actual = cost(widths)
    legal = bool(np.issubdtype(widths.dtype, np.integer) and np.all(g.LOW <= widths)
                 and np.all(widths <= g.HIGH))
    distinct = not np.array_equal(widths, u)
    matched = g.inside(array(actual), target)
    candidate = dict(widths=widths.tolist(), **actual, rms_from_uniform=g.rms(widths, u),
                     wider_than_old_limit=[name for name, k, limit in zip(g.NAMES, widths, old_high) if k > limit])
    passed = legal and distinct and matched
    return dict(record, candidate=candidate, passed=passed,
                stop_reason=None if passed else 'naruszona legalność, odrębność lub okno kosztów',
                checks=dict(legal=legal, distinct_from_uniform=distinct, matched_costs=matched))


def verify_models(rt, geo, f45, checks):
    torch, e35 = rt.torch, rt.e35
    assert not torch.cuda.is_available() and e35.e29.COMMIT == COMMIT
    torch.set_num_threads(2)
    assert list(e35.e30.MEAN) == f45['calibration']['normalization_mean']
    assert list(e35.e30.STD) == f45['calibration']['normalization_std']
    u, l06 = geo['uniform'], geo['candidate']['widths']
    for seed in SEEDS:
        assert e35.checkpoint(PAIR, seed) == CKPT[seed]
        base = e35.model_source(PAIR, seed)
        source_state = e35.state_hash(base)
        geom = rt.geometry(PAIR, base)
        axes = e35.e29.axes2perm_to_perm2axes(e35.e29.get_axis_to_perm_ResNet18(override=False))
        assert list(axes) == geo['groups'] and list(geom.sizes) == geo['sizes']
        assert list(axes.values()) == [geom.members[name] for name in geom.names]
        assert base.conv1.kernel_size == (7, 7) and base.conv1.stride == (2, 2)
        assert base.fc.out_features == CLASSES

        def check(model, profile, widths, expected):
            assert not model.training and all(p.device.type == 'cpu' and not p.requires_grad
                                              and p.grad is None for p in model.parameters())
            actual = dict(params=sum(p.numel() for p in model.parameters()), mac=rt.mac(model, PAIR))
            assert actual['params'] == geom.count(widths)
            assert actual == {key: expected[key] for key in actual}, (seed, profile, actual)
            assert torch.equal(model.fc.bias, base.fc.bias)
            checks.append(dict(seed=seed, profile=profile, **actual))

        check(base, 'teacher', geo['sizes'], geo['teacher'])
        for profile, widths, expected in (('U/b10', u, geo['caps']), ('L06/b10', l06, geo['candidate'])):
            model, diag = (e35.fold(PAIR, base, f45['plans']['b10'], 'U') if profile == 'U/b10'
                           else rt.fold_widths(PAIR, base, widths))
            assert [d['n'] for d in diag] == geo['sizes'] and [d['k'] for d in diag] == widths
            for d, group_axes in zip(diag, axes.values()):
                assert d['first_axis'] == group_axes[0][0] and len(d['labels']) == d['n']
                assert set(d['labels']) == set(range(d['k']))
            check(model, profile, widths, expected)
        assert e35.state_hash(base) == source_state


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    assert not sys.flags.optimize and sys.dont_write_bytecode
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert '/envs/attention/' in sys.executable
    started, clock = datetime.now(timezone.utc), time.monotonic()
    run = ROOT / 'runs/E046' / f'cpu_{started:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    source = run / 'source'
    source.mkdir(parents=True)
    sources = ('OPIS.md', 'URUCHOMIENIE.md', Path(__file__).name)
    for name in sources:
        shutil.copyfile(HERE / name, source / name)
    m = dict(run=rel(run), phase='geometry_cpu', status='running', started_utc=started.isoformat(),
             pid=os.getpid(), command=[sys.executable, '-B', *sys.argv], python=sys.version,
             source_sha256={name: digest(source / name) for name in sources}, input_sha256={},
             settings={key: os.environ.get(key) for key in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES',
                       'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
             model_checks=[], gpu_used=False, images_opened=False, quality_evaluated=False,
             training=False, bn_calibrated=False)
    write(run / 'manifest.json', m)
    print('CPU RUN', rel(run), flush=True)
    allowed, opened, denied = {SPLIT, *CKPT.values()}, set(), []

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT / 'data') or path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors'):
            if path not in allowed or args[2] & (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC):
                denied.append(str(path))
                raise RuntimeError(f'E046 CPU: niedozwolony dostęp do {path}')
            opened.add(path)

    sys.addaudithook(guard)
    try:
        f43, f45, refs = input_metadata(m['input_sha256'])
        g = load(SOLVER, '_e046_solver')
        m.update(numpy=g.np.__version__, scipy=g.scipy.__version__)
        assert (m['numpy'], m['scipy']) == ('2.4.6', '1.17.1')
        geometry = project_once(g, f43, f45)
        write(run / 'geometry.json', geometry)
        m['geometry_sha256'] = digest(run / 'geometry.json')
        if not geometry['passed']:
            m.update(status='stopped_geometry', gate_passed=False)
            print('CPU STOP:', geometry['stop_reason'], flush=True)
            return

        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UP, text=True).strip() == COMMIT
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=UP, text=True).strip()
        with (run / 'construction.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            rt = load(E37, '_e046_e037')
            m['torch'] = str(rt.torch.__version__)
            verify_models(rt, geometry, f45, m['model_checks'])
        assert opened == allowed and not denied and len(m['model_checks']) == 12
        frozen = dict(pair=PAIR, budget='b10', profile='L06', upstream_commit=COMMIT,
                      geometry={k: v for k, v in geometry.items() if k != 'solver'},
                      checkpoints=f45['checkpoints'], calibration=f45['calibration'],
                      dev=f45['dev'], reference_run=rel(R45), reference_frozen_sha256=F45_SHA,
                      reference_metrics_sha256=refs,
                      measurement=dict(**f45['settings'], gpu_to_seed=dict(enumerate(SEEDS)),
                                       new_candidate_scores=4, control_replays=8),
                      input_sha256=m['input_sha256'], source_sha256=m['source_sha256'])
        write(run / 'frozen.json', frozen)
        m.update(status='passed', gate_passed=True, frozen_sha256=digest(run / 'frozen.json'))
        print('CPU PASS:', geometry['candidate'], '\nfrozen SHA-256:', m['frozen_sha256'], flush=True)
    except Exception:
        m.update(status='failed', gate_passed=False, traceback=traceback.format_exc())
        raise
    finally:
        m.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - clock,
                 data_and_weights_opened=sorted(map(rel, opened)), data_access_denied=denied)
        write(run / 'manifest.json', m)


if __name__ == '__main__':
    main()
