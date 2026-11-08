"""2026-11-08_E038_operator_alokacja: infrastruktura kontroli, także dla E039.
Wejście: zamrożone źródła E035; wyjście: manifesty, cztery procesy i metryki.
Bez treningu i dostępu do testu; pełny protokół: OPIS.md (oraz OPIS.md E039).
"""
import csv
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

ROOT = Path(__file__).resolve().parents[2]
DENIED = []


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = str(Path(os.fsdecode(args[0])).resolve())
        if any(s in path for s in ('/test_batch', '/cifar-100-python/test',
                                  '/cache_val.npz', '/test_data.json',
                                  '/attention_v2/', '/attentionv3/')):
            DENIED.append(path)
            raise RuntimeError('E038/E039: zabroniony odczyt: ' + path)


sys.addaudithook(guard)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e37 = load(ROOT / 'experiments/2026-11-08_E037_kontrola_mac/experiment.py', '_e038_e37')
e35, np, torch = e37.e35, e37.np, e37.torch
read, write, digest = e35.read, e35.write, e35.digest
R35 = e37.R35
PAIRS, TAGS, MODES, SEEDS = e35.PAIRS, e35.TAGS, e35.MODES, tuple(e35.SEEDS)


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def hashes(here):
    return {rel(p): digest(p) for p in sorted(here.glob('*.py'))} | {
        rel(here / 'OPIS.md'): digest(here / 'OPIS.md')}


def input_freeze(pairs=PAIRS):
    ref = read(R35 / 'frozen.json')
    provenance = dict(ref['provenance'])
    for p in (R35 / 'frozen.json', Path(e35.__file__), Path(e37.__file__), Path(__file__)):
        provenance[rel(p)] = digest(p)
    refs = {}
    for pair in pairs:
        refs[pair] = {}
        for tag in TAGS:
            refs[pair][tag] = {}
            for seed in SEEDS:
                p = R35 / f'dev/seed_{seed}/{pair}/{tag}/summary.json'
                c = R35 / f'dev/seed_{seed}/{pair}/controls.json'
                s, ctrl = read(p), read(c)
                assert s['status'] == 'completed' and s['plan'] == ref['plans'][pair][tag]
                refs[pair][tag][str(seed)] = dict(metrics=s['metrics'], evidence=s['evidence'],
                    controls=ctrl, views_sha256=s['views_sha256'], source=rel(p))
                provenance[rel(p)] = digest(p)
                provenance[rel(c)] = digest(c)
    e35.validate_hashes(provenance)
    return dict(created_utc=utc(), plans={p: ref['plans'][p] for p in pairs},
                references=refs, indices=ref['indices'], hashes=provenance)


def configure(gpu):
    assert torch.cuda.device_count() == 1
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42 + gpu)
    np.random.seed(42 + gpu)


def telemetry():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)


def validate_check(here):
    check = read(here / 'check.json')
    assert check['status'] == 'passed'
    assert check['code_hashes'] == hashes(here)
    assert check['frozen_sha256'] == digest(here / 'frozen.json')
    frozen = read(here / 'frozen.json')
    e35.validate_hashes(frozen['hashes'])
    return frozen


def launch(here, experiment, settings):
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3' and torch.cuda.device_count() == 4
    frozen = validate_check(here)
    info = telemetry()
    assert len(info.strip().splitlines()) == 4, info
    for i, line in enumerate(info.strip().splitlines()):
        row = [s.strip() for s in line.split(',')]
        assert int(row[0]) == i and int(row[3]) < 256 and int(row[4]) == 0, info
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    assert not apps.strip(), apps
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(4)
    run = ROOT / 'runs' / experiment / name
    (run / 'source').mkdir(parents=True)
    for p in here.iterdir():
        if p.suffix in ('.py', '.md', '.json'):
            shutil.copy2(p, run / 'source' / p.name)
    # Snapshot zależności kodowych; wejściowe wagi i dane pozostają pod hashami.
    for path in frozen['hashes']:
        p = ROOT / path
        if p.suffix == '.py':
            dest = run / 'source/dependencies' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.copy2(here / 'frozen.json', run / 'frozen.json')
    manifest = dict(status='running', pid=os.getpid(), run=rel(run), started_utc=utc(),
        limit_seconds=5400, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        python=platform.python_version(), executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu_start=info, settings=settings, code_hashes=hashes(here),
        frozen_sha256=digest(run / 'frozen.json'), workers=[],
        environment={k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')},
        command=[sys.executable, '-B', rel(here / 'experiment.py'), 'run'])
    (run / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    write(run / 'manifest.json', manifest)
    print('RUN', run, 'PID', os.getpid(), flush=True)
    processes, logs = [], []
    start = time.monotonic()
    try:
        for gpu in range(4):
            log = (run / f'worker_{gpu}.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            p = subprocess.Popen([sys.executable, '-B', str(here / 'experiment.py'), 'worker',
                                  '--run', str(run), '--gpu', str(gpu)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p)
            manifest['workers'].append(dict(gpu=gpu, seed=42 + gpu, pid=p.pid))
        write(run / 'manifest.json', manifest)
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in processes]
                monitor.write(json.dumps(dict(utc=utc(), elapsed=time.monotonic() - start,
                    exit_codes=codes, gpu=telemetry())) + '\n')
                monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0] * 4:
                    break
                if time.monotonic() - start > 5400:
                    raise TimeoutError('Limit90min')
                time.sleep(10)
        assert hashes(here) == manifest['code_hashes']
        e35.validate_hashes(frozen['hashes'])
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
                p.kill()
                p.wait()
        raise
    finally:
        for row, p in zip(manifest['workers'], processes):
            row['exit_code'] = p.poll()
        for log in logs:
            log.close()
        manifest.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - start, gpu_end=telemetry())
        write(run / 'manifest.json', manifest)
        print(manifest['status'], rel(run), flush=True)
    return run


def stats(values):
    values = list(map(float, values))
    assert len(values) == 4 and np.isfinite(values).all()
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = 3.182446305284263 * sd / 2
    return dict(mean=mean, sd=sd, per_seed=values, positive=sum(v > 0 for v in values),
                ci95_t3=[mean - half, mean + half])


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def verify_metric(metric, evidence):
    assert 0 <= metric['top1'] <= 1 and np.isfinite(metric['nll']) and metric['nll'] >= 0
    assert evidence['weights_unchanged'] and evidence['no_grad']
    assert evidence['samples'] in (5000, 10000)


def check_reference(metrics, evidence, ref_metrics, ref_evidence):
    for mode in MODES:
        assert abs(metrics[mode]['top1'] - ref_metrics[mode]['top1']) < 1e-10
        assert abs(metrics[mode]['nll'] - ref_metrics[mode]['nll']) <= 1e-5
        assert evidence[mode]['state_sha256'] == ref_evidence[mode]['state_sha256']


def verify_run(run):
    manifest = read(run / 'manifest.json')
    assert manifest['status'] == 'completed'
    assert [w['gpu'] for w in manifest['workers']] == list(range(4))
    assert [w['seed'] for w in manifest['workers']] == list(SEEDS)
    assert [w['exit_code'] for w in manifest['workers']] == [0] * 4
    assert manifest['frozen_sha256'] == digest(run / 'frozen.json')
    e35.validate_hashes(manifest['code_hashes'])
    frozen = read(run / 'frozen.json')
    e35.validate_hashes(frozen['hashes'])
    for seed in SEEDS:
        s = read(run / f'seed_{seed}/status.json')
        assert s['status'] == 'completed' and not s['data_open_attempts']
    return manifest, frozen
