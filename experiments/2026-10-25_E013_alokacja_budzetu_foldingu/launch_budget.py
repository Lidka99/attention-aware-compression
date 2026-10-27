"""E013: launcher przesiewu alokacji budżetu foldingu na czterech GPU.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Uruchamia 33 niezależne warianty (budżet × rodzina alokacji) i zapisuje
unikalny run, manifest, logi oraz zagregowany summary.json.
"""

import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / 'config.json'
E011_TRAIN = ROOT / 'experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/train.py'
MIN_FREE_BYTES = 2 * 1024 ** 3


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def query_gpus():
    command = ['nvidia-smi',
               '--query-gpu=index,name,memory.total,memory.used',
               '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 4:
            continue
        rows.append({
            'index': int(parts[0]),
            'name': parts[1],
            'memory_total_mib': int(float(parts[2])),
            'memory_used_mib': int(float(parts[3])),
        })
    rows.sort(key=lambda row: row['index'])
    return rows


def preflight():
    require(shutil.disk_usage(ROOT).free >= MIN_FREE_BYTES,
            'Za mało wolnego dysku (wymagane 2 GiB)')
    gpus = query_gpus()
    require(len(gpus) >= 4, f'Wymagane 4 GPU, wykryto {len(gpus)}')
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True).strip()
    require(not apps, f'GPU zajęte przez procesy: {apps}')
    for gpu in gpus[:4]:
        require(gpu['memory_total_mib'] - gpu['memory_used_mib'] >= 4096,
                f"GPU {gpu['index']} ma za mało VRAM")
    return gpus[:4]


def build_specs(cfg):
    specs = []
    for budget in cfg['budgets']:
        tag = f'b{int(round(float(budget) * 100)):02d}'
        specs.append({
            'name': f'{tag}_uniform', 'budget': float(budget),
            'family': 'uniform'})
        for tilt in (-0.30, -0.15, 0.15, 0.30):
            label = ('m' if tilt < 0 else 'p') + f'{abs(tilt):.2f}'.replace('.', '')
            specs.append({
                'name': f'{tag}_tilt_{label}', 'budget': float(budget),
                'family': 'tilt', 'tilt': float(tilt)})
        for seed in (101, 102, 103, 104):
            specs.append({
                'name': f'{tag}_random_{seed}', 'budget': float(budget),
                'family': 'random', 'sigma': 0.20, 'random_seed': seed})
        specs.append({
            'name': f'{tag}_mag1', 'budget': float(budget), 'family': 'mag1'})
        specs.append({
            'name': f'{tag}_mag2', 'budget': float(budget), 'family': 'mag2'})
    names = [spec['name'] for spec in specs]
    require(len(names) == len(set(names)), 'Zduplikowane nazwy specyfikacji')
    return specs


def worker_command(run, config, spec_path, gpu):
    return [sys.executable, '-B', str(HERE / 'budget_worker.py'),
            '--run-dir', str(run), '--config', str(config),
            '--spec', str(spec_path), '--gpu', str(gpu)]


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait()


def main():
    require(CONFIG_PATH.is_file(), 'Brak config.json')
    cfg = read_json(CONFIG_PATH)
    require(cfg.get('status') == 'frozen', 'Config nie jest frozen')
    require(Path(cfg['checkpoint']).is_file(), 'Brak checkpointu')
    require(Path(cfg['split']).is_file(), 'Brak splitu')
    require(digest(cfg['checkpoint']) == cfg['checkpoint_sha256'],
            'Hash checkpointu nie zgadza się z configiem')
    require(digest(cfg['split']) == cfg['split_sha256'],
            'Hash splitu nie zgadza się z configiem')
    require(digest(E011_TRAIN) == cfg['e011_train_sha256'],
            'Hash train.py E011 nie zgadza się z configiem')
    require(not subprocess.check_output(
        ['git', 'status', '--porcelain', '--', str(HERE)], cwd=ROOT,
        text=True).strip(), 'Przed biegiem commituj folder E013')
    gpus = preflight()
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'runs/E013' / f'{stamp}_{git_sha[:7]}_{secrets.token_hex(4)}'
    run.mkdir(parents=True)
    source = run / 'source'
    source.mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            shutil.copy2(path, source / path.name)
    shutil.copy2(E011_TRAIN, source / 'e011_train.py')
    specs = build_specs(cfg)
    specs_dir = run / 'specs'
    specs_dir.mkdir()
    spec_records = []
    for spec in specs:
        path = specs_dir / f"{spec['name']}.json"
        write_json_atomic(path, spec)
        spec_records.append({
            'name': spec['name'],
            'path': str(path.relative_to(run)),
            'sha256': digest(path),
        })
    logs = run / 'logs'
    logs.mkdir()
    manifest = {
        'experiment': HERE.name,
        'protocol': 'OPIS.md',
        'run_id': run.name,
        'status': 'running',
        'started_utc': utc_now(),
        'finished_utc': None,
        'git_sha': git_sha,
        'git_dirty': bool(git_status.strip()),
        'config_sha256': digest(source / 'config.json'),
        'source_sha256': {path.name: digest(path)
                          for path in sorted(source.iterdir())},
        'specs': spec_records,
        'gpus': gpus,
        'gpu_assignment': 'one worker per GPU; waves of four',
        'batch_size_per_gpu': int(cfg['batch_size']),
        'bn_seed': int(cfg['bn_seed']),
        'bn_samples': int(cfg['bn_samples']),
        'official_test_used': False,
        'independent_seeds': 1,
        'workers': [],
        'exit_codes': {},
    }
    (run / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    (run / 'nvidia-smi.txt').write_text(subprocess.check_output(
        ['nvidia-smi'], text=True))
    write_json_atomic(run / 'manifest.json', manifest)
    print(f'E013_RUN={run}', flush=True)
    failures = []
    active = []
    try:
        for spec in specs:
            gpu = gpus[len(manifest['workers']) % 4]['index']
            spec_path = specs_dir / f"{spec['name']}.json"
            command = worker_command(run, source / 'config.json', spec_path, gpu)
            log_path = logs / f"{spec['name']}.log"
            log = log_path.open('w')
            environment = dict(os.environ)
            environment.update({
                'CUDA_VISIBLE_DEVICES': str(gpu),
                'PYTHONHASHSEED': '42',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
            })
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            record = {
                'name': spec['name'], 'gpu': gpu, 'pid': process.pid,
                'command': command, 'log': str(log_path.relative_to(run)),
                'status': 'running',
            }
            manifest['workers'].append(record)
            active.append((process, log, record))
            write_json_atomic(run / 'manifest.json', manifest)
            if len(active) == 4 or len(manifest['workers']) == len(specs):
                # Wszystkie cztery startują równolegle; wait zbiera kody.
                for process, log, record in active:
                    code = process.wait()
                    log.close()
                    record['returncode'] = int(code)
                for process, log, record in active:
                    summary_path = (
                        run / 'allocations' / record['name'] / 'summary.json')
                    summary = read_json(summary_path) if summary_path.is_file() else {}
                    ok = record['returncode'] == 0 and summary.get('status') == 'completed'
                    record['status'] = summary.get('status', 'failed')
                    manifest['exit_codes'][record['name']] = (
                        0 if ok else max(1, record['returncode']))
                    if not ok:
                        failures.append((record['name'], summary.get('error')))
                active = []
                write_json_atomic(run / 'manifest.json', manifest)
                print(f'E013_WAVE_DONE={len(manifest["workers"])}', flush=True)
                if failures:
                    raise RuntimeError(f'Workery nie powiodły się: {failures}')
        summaries = {}
        for spec in specs:
            path = run / 'allocations' / spec['name'] / 'summary.json'
            require(path.is_file(), f'Brak summary dla {spec["name"]}')
            summary = read_json(path)
            require(summary.get('status') == 'completed',
                    f'{spec["name"]}: {summary.get("status")}')
            summaries[spec['name']] = summary
        write_json_atomic(run / 'summary.json', summaries)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = repr(error)
        raise
    finally:
        for process, log, record in active:
            if process.poll() is None:
                terminate(process)
            log.close()
            record['status'] = 'terminated'
        manifest['finished_utc'] = utc_now()
        write_json_atomic(run / 'manifest.json', manifest)
        artifacts = []
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.name != 'artifacts.sha256':
                artifacts.append(f'{digest(path)}  {path.relative_to(run)}')
        (run / 'artifacts.sha256').write_text('\n'.join(artifacts) + '\n')
        print(f'E013_STATUS={manifest["status"]}', flush=True)


if __name__ == '__main__':
    main()
