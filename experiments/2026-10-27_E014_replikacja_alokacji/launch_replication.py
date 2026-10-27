"""E014: launcher replikacji alokacji TILT+0.30 na czterech seedach E011.

Eksperyment: 2026-10-27_E014_replikacja_alokacji; pełny opis: OPIS.md.
Uruchamia 48 workerów (4 seedy × 3 budżety × 4 warianty) falami po cztery.
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
CONFIGS = HERE / 'configs'
E011_TRAIN = ROOT / 'experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/train.py'
MIN_FREE_BYTES = 2 * 1024 ** 3
SEEDS = (42, 43, 44, 45)
BUDGETS = (0.30, 0.50, 0.70)
VARIANTS = ('uniform', 'tilt_p030', 'tilt_p015', 'mag2')


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
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
         '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 4:
            rows.append({
                'index': int(parts[0]), 'name': parts[1],
                'memory_total_mib': int(float(parts[2])),
                'memory_used_mib': int(float(parts[3])),
            })
    return sorted(rows, key=lambda row: row['index'])


def preflight():
    require(shutil.disk_usage(ROOT).free >= MIN_FREE_BYTES,
            'Za mało wolnego dysku (2 GiB)')
    gpus = query_gpus()
    require(len(gpus) >= 4, f'Wymagane 4 GPU, jest {len(gpus)}')
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True).strip()
    require(not apps, f'GPU zajęte: {apps}')
    for gpu in gpus[:4]:
        require(gpu['memory_total_mib'] - gpu['memory_used_mib'] >= 4096,
                f"GPU {gpu['index']} ma za mało VRAM")
    return gpus[:4]


def build_specs():
    specs = []
    for budget in BUDGETS:
        tag = f'b{int(round(budget * 100)):02d}'
        specs.append({'name': f'{tag}_uniform', 'budget': budget,
                      'family': 'uniform'})
        specs.append({'name': f'{tag}_tilt_p030', 'budget': budget,
                      'family': 'tilt', 'tilt': 0.30})
        specs.append({'name': f'{tag}_tilt_p015', 'budget': budget,
                      'family': 'tilt', 'tilt': 0.15})
        specs.append({'name': f'{tag}_mag2', 'budget': budget,
                      'family': 'mag2'})
    return specs


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


def run_seed(seed, config_source, seed_run, gpus, overall_manifest):
    config_path = seed_run / 'source' / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_source, config_path)
    specs = build_specs()
    specs_dir = seed_run / 'specs'
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_records = []
    for spec in specs:
        path = specs_dir / f"{spec['name']}.json"
        write_json_atomic(path, spec)
        spec_records.append({'name': spec['name'],
                             'sha256': digest(path),
                             'path': str(path.relative_to(seed_run))})
    logs = seed_run / 'logs'
    logs.mkdir(exist_ok=True)
    manifest = {
        'experiment': HERE.name,
        'seed': int(seed),
        'status': 'running',
        'started_utc': utc_now(),
        'finished_utc': None,
        'config_sha256': digest(config_path),
        'config_source': str(config_source),
        'source_sha256': {path.name: digest(path)
                          for path in sorted((seed_run / 'source').iterdir())},
        'specs': spec_records,
        'gpus': gpus,
        'official_test_used': False,
        'workers': [],
        'exit_codes': {},
    }
    write_json_atomic(seed_run / 'manifest.json', manifest)
    print(f'E014_SEED={seed}_RUN={seed_run}', flush=True)
    active = []
    failures = []
    try:
        for spec in specs:
            gpu = gpus[len(manifest['workers']) % 4]['index']
            command = [sys.executable, '-B', str(HERE / 'worker.py'),
                       '--run-dir', str(seed_run),
                       '--config', str(config_path),
                       '--spec', str(specs_dir / f"{spec['name']}.json"),
                       '--gpu', str(gpu)]
            log_path = logs / f"{spec['name']}.log"
            log = log_path.open('w')
            environment = dict(os.environ)
            environment.update({
                'CUDA_VISIBLE_DEVICES': str(gpu),
                'PYTHONHASHSEED': str(seed),
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
                'command': command, 'log': str(log_path.relative_to(seed_run)),
                'status': 'running',
            }
            manifest['workers'].append(record)
            active.append((process, log, record))
            write_json_atomic(seed_run / 'manifest.json', manifest)
            if len(active) == 4 or len(manifest['workers']) == len(specs):
                for process, log, record in active:
                    code = process.wait()
                    log.close()
                    record['returncode'] = int(code)
                for process, log, record in active:
                    summary_path = (seed_run / 'allocations' / record['name']
                                    / 'summary.json')
                    summary = read_json(summary_path) if summary_path.is_file() else {}
                    ok = (record['returncode'] == 0
                          and summary.get('status') == 'completed')
                    record['status'] = summary.get('status', 'failed')
                    manifest['exit_codes'][record['name']] = (
                        0 if ok else max(1, record['returncode']))
                    if not ok:
                        failures.append((record['name'], summary.get('error')))
                active = []
                write_json_atomic(seed_run / 'manifest.json', manifest)
                print(f'E014_SEED={seed}_WAVE_DONE={len(manifest["workers"])}',
                      flush=True)
                if failures:
                    raise RuntimeError(f'Seed {seed} failures: {failures}')
        summaries = {}
        for spec in specs:
            path = seed_run / 'allocations' / spec['name'] / 'summary.json'
            require(path.is_file(), f'Brak summary {spec["name"]}')
            summary = read_json(path)
            require(summary.get('status') == 'completed',
                    f'{spec["name"]}: {summary.get("status")}')
            summaries[spec['name']] = summary
        write_json_atomic(seed_run / 'summary.json', summaries)
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
        write_json_atomic(seed_run / 'manifest.json', manifest)
        artifacts = []
        for path in sorted(seed_run.rglob('*')):
            if path.is_file() and path.name != 'artifacts.sha256':
                artifacts.append(f'{digest(path)}  {path.relative_to(seed_run)}')
        (seed_run / 'artifacts.sha256').write_text('\n'.join(artifacts) + '\n')
        overall_manifest['seeds'][str(seed)] = {
            'path': str(seed_run.relative_to(seed_run.parents[1])),
            'status': manifest['status'],
            'config_sha256': manifest['config_sha256'],
            'exit_codes': manifest['exit_codes'],
        }


def main():
    require(not subprocess.check_output(
        ['git', 'status', '--porcelain', '--', str(HERE)], cwd=ROOT,
        text=True).strip(), 'Przed biegiem commituj folder E014')
    configs = {seed: CONFIGS / f'seed_{seed}.json' for seed in SEEDS}
    for seed, path in configs.items():
        require(path.is_file(), f'Brak configu {path}')
        cfg = read_json(path)
        require(int(cfg['seed']) == seed, f'Zły seed w {path}')
        require(Path(cfg['checkpoint']).is_file(), f'Brak checkpointu {path}')
        require(Path(cfg['split']).is_file(), f'Brak splitu {path}')
        require(digest(cfg['checkpoint']) == cfg['checkpoint_sha256'],
                f'Zły hash checkpointu {path}')
        require(digest(cfg['split']) == cfg['split_sha256'],
                f'Zły hash splitu {path}')
        require(digest(E011_TRAIN) == cfg['e011_train_sha256'],
                f'Zły hash train.py E011 {path}')
    gpus = preflight()
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = (ROOT / 'runs/E014'
           / f'{stamp}_{git_sha[:7]}_{secrets.token_hex(4)}')
    run.mkdir(parents=True)
    source = run / 'source'
    source.mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            shutil.copy2(path, source / path.name)
    shutil.copy2(E011_TRAIN, source / 'e011_train.py')
    overall = {
        'experiment': HERE.name,
        'protocol': 'OPIS.md',
        'run_id': run.name,
        'status': 'running',
        'started_utc': utc_now(),
        'finished_utc': None,
        'git_sha': git_sha,
        'git_dirty': bool(git_status.strip()),
        'source_sha256': {path.name: digest(path)
                          for path in sorted(source.iterdir())},
        'seeds': {},
        'gpus': gpus,
        'official_test_used': False,
    }
    (run / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    (run / 'nvidia-smi.txt').write_text(subprocess.check_output(
        ['nvidia-smi'], text=True))
    write_json_atomic(run / 'manifest.json', overall)
    print(f'E014_RUN={run}', flush=True)
    try:
        for seed in SEEDS:
            seed_run = run / 'seeds' / f'seed_{seed}'
            seed_run.mkdir(parents=True)
            run_seed(seed, configs[seed], seed_run, gpus, overall)
            write_json_atomic(run / 'manifest.json', overall)
        overall['status'] = 'completed'
    except BaseException as error:
        overall['status'] = 'failed'
        overall['error'] = repr(error)
        raise
    finally:
        overall['finished_utc'] = utc_now()
        write_json_atomic(run / 'manifest.json', overall)
        artifacts = []
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.name != 'artifacts.sha256':
                artifacts.append(f'{digest(path)}  {path.relative_to(run)}')
        (run / 'artifacts.sha256').write_text('\n'.join(artifacts) + '\n')
        print(f'E014_STATUS={overall["status"]}', flush=True)


if __name__ == '__main__':
    main()
