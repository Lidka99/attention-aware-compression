"""E015: launcher czterech niezależnych treningów ResNet18/CIFAR-100.

Eksperyment: 2026-10-27_E015_cifar100_alokacja; pełny opis: OPIS.md.
Jeden proces na GPU (seedy 42–45), wyłącznie CIFAR-100 train i istniejący
podział 40 000 train / 5 000 dev; rezerwa i oficjalny test nietknięte.
Wyjście: unikalny katalog ``runs/E015/<UTC>_<git>_seeds42-45_<token>/``
z manifestem, snapshotem źródeł, kopią splits.json, środowiskiem, logami,
metrykami i summary.json czterech workerów.

Uruchomienie: ``launch_training.py [--smoke|--data-only]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_cifar100 as data  # noqa: E402

ROOT = data.ROOT
RUNS = ROOT / 'runs' / 'E015'
CONFIG_PATH = HERE / 'configs' / 'train.json'
TRAIN_PATH = HERE / 'train_cifar100.py'
REUSED_SOURCES = {
    'e012_folding.py': ROOT / (
        'experiments/2026-10-24_E012_folding_referencja/folding.py'),
    'e012_evaluate.py': ROOT / (
        'experiments/2026-10-24_E012_folding_referencja/evaluate.py'),
    'e014_folding_budget.py': ROOT / (
        'experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py'),
    'e014_pruning.py': ROOT / (
        'experiments/2026-10-27_E014_replikacja_alokacji/pruning.py'),
}
SEEDS = [42, 43, 44, 45]
MIN_FREE_BYTES = 2 * 1024 ** 3
MIN_FREE_MIB = 4096
EXPECTED_CONFIG = {
    'num_classes': 100,
    'epochs': 200,
    'batch_size': 128,
    'gradient_accumulation': 1,
    'effective_global_batch_per_run': 128,
    'split_sha256': data.SPLITS_SHA256,
    'split_seed': data.SPLIT_SEED,
    'train_count': data.EXPECTED_TRAIN,
    'dev_count': data.EXPECTED_DEV,
    'reserve_count': data.EXPECTED_RESERVE,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'nesterov': False,
    'schedule': 'cosine',
    'amp': False,
    'label_smoothing': 0.0,
    'num_workers': 2,
    'official_test_used': False,
    'reserve_used': False,
}
COMMON_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONFAULTHANDLER': '1',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digester(path):
    return data.digest(path)


def read_json(path):
    return data.read_json(path)


def write_json_atomic(path, value):
    data.write_json_atomic(path, value)


def utc_now():
    return data.utc_now()


def check_environment():
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    executable = sys.executable.replace('\\', '/')
    require(env_name == 'attention' or '/envs/attention/' in executable,
            'Uruchom launcher w środowisku Conda attention')


def load_config(path):
    path = Path(path)
    require(path.is_file(), f'Brak frozen config.json: {path}')
    cfg = read_json(path)
    missing = [key for key in EXPECTED_CONFIG if key not in cfg]
    require(not missing, f'Brak pól konfiguracji: {missing}')
    require(cfg.get('status') == 'frozen', 'Konfiguracja nie jest zamrożona')
    seeds = [int(value) for value in cfg['seeds']]
    require(seeds == SEEDS, f'seeds musi być dokładnie {SEEDS}')
    require(cfg.get('gpu_by_seed') == {str(seed): index for index, seed in enumerate(SEEDS)},
            'gpu_by_seed musi przypisywać seedy 42–45 do GPU 0–3')
    for key, expected in EXPECTED_CONFIG.items():
        require(cfg[key] == expected,
                f'{key} musi wynosić {expected!r} (jest {cfg[key]!r})')
    require(tuple(float(value) for value in cfg['normalization_mean'])
            == data.NORMALIZATION_MEAN, 'Zła średnia normalizacji')
    require(tuple(float(value) for value in cfg['normalization_std'])
            == data.NORMALIZATION_STD, 'Złe odchylenie normalizacji')
    return cfg, seeds


def disk_info():
    usage = shutil.disk_usage(ROOT)
    return {
        'path': str(ROOT),
        'total_bytes': usage.total,
        'used_bytes': usage.used,
        'free_bytes': usage.free,
    }


def query_gpus():
    command = [
        'nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f'Nie można odczytać GPU przez nvidia-smi: {error}') from error
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 6:
            continue
        rows.append({
            'index': int(row[0].strip()),
            'uuid': row[1].strip(),
            'name': row[2].strip(),
            'memory_total_mib': int(float(row[3])),
            'memory_used_mib': int(float(row[4])),
            'utilization_gpu_percent': int(float(row[5])),
        })
    rows.sort(key=lambda item: item['index'])
    return rows


def query_compute_apps():
    command = [
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    apps = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 2:
            continue
        used = None
        if len(row) > 2:
            try:
                used = int(float(row[2]))
            except ValueError:
                used = None
        apps.setdefault(row[0].strip(), []).append(
            {'pid': int(row[1]), 'used_memory_mib': used})
    return apps


def preflight_gpus():
    gpus = query_gpus()
    require(len(gpus) >= 4, f'Wymagane 4 GPU, wykryto {len(gpus)}')
    selected = gpus[:4]
    require([gpu['index'] for gpu in selected] == [0, 1, 2, 3],
            'Oczekiwano fizycznych GPU 0–3')
    apps = query_compute_apps()
    for gpu in selected:
        free = gpu['memory_total_mib'] - gpu['memory_used_mib']
        gpu['memory_free_mib'] = free
        require(not apps.get(gpu['uuid']),
                f"GPU {gpu['index']} ma aktywne procesy: {apps.get(gpu['uuid'])}")
        require(free >= MIN_FREE_MIB,
                f"GPU {gpu['index']} ma za mało wolnej VRAM: {free} MiB")
    return selected


def create_run(git_sha):
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for _ in range(100):
        name = f'{stamp}_{git_sha[:7]}_seeds42-45_{secrets.token_hex(4)}'
        run = RUNS / name
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć unikalnego katalogu biegu')


def snapshot_sources(run):
    source = run / 'source'
    source.mkdir()
    hashes = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            target = source / path.name
            shutil.copy2(path, target)
            hashes[path.name] = digester(target)
    for name in ('train.json', 'allocation.json'):
        config_source = HERE / 'configs' / name
        config_target = source / 'configs' / name
        config_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_source, config_target)
        hashes[f'configs/{name}'] = digester(config_target)
    reused = source / 'reused'
    reused.mkdir()
    for name, path in REUSED_SOURCES.items():
        require(path.is_file(), f'Brak używanego źródła: {path}')
        shutil.copy2(path, reused / name)
        hashes[f'reused/{name}'] = digester(reused / name)
    return hashes


def version_of(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_environment(run, gpus):
    import torch
    environment = {
        'python': platform.python_version(),
        'python_full': sys.version,
        'executable': sys.executable,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torchvision': version_of('torchvision'),
        'numpy': version_of('numpy'),
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'driver': subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True).strip(),
        'gpus': gpus,
        'deterministic_algorithms': True,
        'tf32': False,
        'amp': False,
        'env': COMMON_ENV,
    }
    write_json_atomic(run / 'environment.json', environment)
    for name, command in (
        ('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
        ('conda-explicit.txt', ['conda', 'list', '--explicit']),
        ('nvidia-smi.txt', ['nvidia-smi']),
    ):
        try:
            text = subprocess.check_output(command, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            text = f'Nie udało się wykonać {command}: {error}\n'
        (run / name).write_text(text)
    return environment


def worker_command(run, config, seed, gpu_index, epochs, smoke):
    command = [
        sys.executable, '-B', str(TRAIN_PATH), '--run-dir', str(run),
        '--config', str(config), '--seed', str(seed), '--gpu', str(gpu_index),
        '--epochs', str(epochs),
    ]
    if smoke:
        command.append('--smoke')
    return command


def write_command_script(run, commands, gpu_by_seed):
    lines = [
        '# E015: cztery niezależne treningi ResNet18/CIFAR-100.',
        f'# cwd: {ROOT}',
        'export ' + ' '.join(
            f'{key}={shlex.quote(value)}' for key, value in COMMON_ENV.items()),
        '# data_prepare',
        shlex.join(commands['data_prepare']),
    ]
    for seed in SEEDS:
        if f'seed_{seed}' not in commands:
            continue
        lines.append(
            f'# seed_{seed} (CUDA_VISIBLE_DEVICES={gpu_by_seed[seed]})')
        lines.append(
            f'CUDA_VISIBLE_DEVICES={gpu_by_seed[seed]} '
            f'PYTHONHASHSEED={seed} ' + shlex.join(commands[f'seed_{seed}'])
            + ' &')
    lines.append('wait')
    (run / 'command.sh').write_text('\n'.join(lines) + '\n')


def run_data_preparation(run, config):
    log_path = run / 'logs' / 'data_prep.log'
    command = [
        sys.executable, '-B', str(TRAIN_PATH), '--data-only',
        '--run-dir', str(run), '--config', str(config),
    ]
    environment = os.environ.copy()
    environment.update(COMMON_ENV)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['PYTHONHASHSEED'] = '0'
    with log_path.open('w') as log:
        result = subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0,
            f'Przygotowanie danych nie powiodło się: {log_path}')
    split_path = run / 'splits.json'
    data_path = run / 'data_manifest.json'
    require(split_path.is_file() and data_path.is_file(),
            'Brak splits.json lub data_manifest.json po przygotowaniu danych')
    split = read_json(split_path)
    return command, {
        'split_sha256': digester(split_path),
        'data_manifest_sha256': digester(data_path),
        'train_count': len(split['splits']['train']),
        'dev_count': len(split['splits']['dev']),
        'reserve_count': len(split['splits']['reserve']),
        'split_seed': int(split['split_seed']),
        'official_test_used': False,
        'reserve_used': False,
    }


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            process.terminate()
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait()


def check_summaries(run, seeds, epochs):
    summaries = {}
    for seed in seeds:
        directory = run / f'seed_{seed}'
        summary_path = directory / 'summary.json'
        last_path = directory / 'last.pt'
        require(summary_path.is_file(), f'Brak summary.json dla seed {seed}')
        require(last_path.is_file(), f'Brak last.pt dla seed {seed}')
        summary = read_json(summary_path)
        require(summary.get('status') == 'completed',
                f"Seed {seed}: status {summary.get('status')!r}")
        require(int(summary.get('seed')) == seed,
                f'Seed {seed}: niezgodny seed w summary')
        require(int(summary.get('final_epoch')) == epochs,
                f'Seed {seed}: final_epoch != {epochs}')
        require(summary.get('official_test_used') is False,
                f'Seed {seed}: użyto oficjalnego testu')
        require(summary.get('reserve_used') is False,
                f'Seed {seed}: użyto rezerwy')
        summaries[str(seed)] = {
            'summary_sha256': digester(summary_path),
            'last_checkpoint_sha256': digester(last_path),
            'final_metrics': summary.get('final_metrics'),
            'runtime_seconds': summary.get('runtime_seconds'),
            'peak_memory_mb': summary.get('peak_memory_mb'),
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--smoke', action='store_true',
                        help='2 epoki, 1000 train / 200 dev')
    parser.add_argument('--data-only', action='store_true',
                        help='tylko walidacja CIFAR-100 i przygotowanie podziału')
    args = parser.parse_args()

    check_environment()
    cfg, seeds = load_config(CONFIG_PATH)
    if not args.data_only and not args.smoke:
        require(args.epochs == int(cfg['epochs']),
                f"--epochs={args.epochs} różni się od config epochs={cfg['epochs']}")
    effective_epochs = 2 if args.smoke else int(args.epochs)
    disk = disk_info()
    require(disk['free_bytes'] > MIN_FREE_BYTES,
            f"Za mało wolnego dysku: {disk['free_bytes']} B <= {MIN_FREE_BYTES} B")
    data.verify_source_files()
    gpus = [] if args.data_only else preflight_gpus()
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    run = create_run(git_sha)
    source_hashes = snapshot_sources(run)
    config_snapshot = run / 'source' / 'configs' / 'train.json'
    require(config_snapshot.is_file(), 'Brak snapshotu config.json')
    config_sha256 = digester(CONFIG_PATH)
    require(digester(config_snapshot) == config_sha256,
            'Snapshot configu różni się od config.json')
    logs = run / 'logs'
    logs.mkdir()
    if args.data_only:
        gpu_by_seed = {seed: index for index, seed in enumerate(seeds)}
    else:
        gpu_by_seed = {seed: gpu['index'] for seed, gpu in zip(seeds, gpus)}
    commands = {
        'data_prepare': [
            sys.executable, '-B', str(TRAIN_PATH), '--data-only',
            '--run-dir', str(run), '--config', str(config_snapshot),
        ]
    }
    if not args.data_only:
        for seed in seeds:
            commands[f'seed_{seed}'] = worker_command(
                run, config_snapshot, seed, gpu_by_seed[seed],
                effective_epochs, args.smoke)
    write_command_script(run, commands, gpu_by_seed)
    manifest = {
        'experiment': HERE.name,
        'opis': 'OPIS.md',
        'protocol': 'OPIS.md',
        'run_id': run.name,
        'created_utc': utc_now(),
        'started_utc': utc_now(),
        'finished_utc': None,
        'status': 'running',
        'data_only': bool(args.data_only),
        'cwd': str(ROOT),
        'git_sha': git_sha,
        'git_dirty': bool(git_status.strip()),
        'git_note': (
            'Folder E015 może być niecommitowany; źródłem prawdy jest snapshot '
            'source/ i jego hashe, nie czystość git.'),
        'executable': sys.executable,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'versions': {},
        'config_sha256': config_sha256,
        'source_sha256': source_hashes,
        'commands': commands,
        'env': COMMON_ENV,
        'seeds': seeds,
        'epochs_requested': int(args.epochs),
        'effective_epochs': effective_epochs,
        'smoke': bool(args.smoke),
        'batch_size_per_gpu': int(cfg['batch_size']),
        'gradient_accumulation': int(cfg['gradient_accumulation']),
        'effective_global_batch_per_run': int(
            cfg['effective_global_batch_per_run']),
        'independent_runs': len(seeds),
        'aggregation': 'none (independent seeds)',
        'devices': [
            {**gpu, 'seed': seed} for seed, gpu in zip(seeds, gpus)
        ] if gpus else [],
        'gpu_by_seed': {str(seed): index for seed, index in gpu_by_seed.items()},
        'disk': disk,
        'official_test_used': False,
        'reserve_used': False,
        'split_sha256': None,
        'split_counts': None,
        'workers': {},
        'pids': {},
        'worker_exit_codes': {},
        'summaries': {},
        'checkpoint_retention': cfg['checkpoint_retention'],
        'exit_code': None,
    }
    write_json_atomic(run / 'manifest.json', manifest)
    procs = {}
    handles = {}
    exit_codes = {}
    summaries = {}
    failure = None
    status = 'failed'
    print(f'E015_TRAIN_RUN={run}', flush=True)
    try:
        manifest['versions'] = write_environment(run, gpus)
        write_json_atomic(run / 'manifest.json', manifest)
        prep_command, data_info = run_data_preparation(run, config_snapshot)
        require(prep_command == commands['data_prepare'],
                'Niezgodna komenda przygotowania danych')
        manifest.update({
            'split_sha256': data_info['split_sha256'],
            'split_counts': {
                'train': data_info['train_count'],
                'dev': data_info['dev_count'],
                'reserve': data_info['reserve_count'],
            },
            'data_manifest_sha256': data_info['data_manifest_sha256'],
            'data': data_info,
        })
        write_json_atomic(run / 'manifest.json', manifest)
        if args.data_only:
            status = 'completed'
            print(f"E015_DATA_ONLY_RUN={run} "
                  f"SPLIT_SHA256={data_info['split_sha256']}", flush=True)
            return 0

        for seed in seeds:
            gpu_index = gpu_by_seed[seed]
            log_path = logs / f'seed_{seed}.log'
            handles[seed] = log_path.open('w')
            environment = os.environ.copy()
            environment.update(COMMON_ENV)
            environment['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
            environment['PYTHONHASHSEED'] = str(seed)
            process = subprocess.Popen(
                commands[f'seed_{seed}'], cwd=ROOT, env=environment,
                stdout=handles[seed], stderr=subprocess.STDOUT,
                start_new_session=True)
            procs[seed] = process
            manifest['pids'][str(seed)] = process.pid
            manifest['workers'][str(seed)] = {
                'seed': seed,
                'physical_gpu': gpu_index,
                'pid': process.pid,
                'status': 'running',
                'command': commands[f'seed_{seed}'],
                'log': str(log_path.relative_to(run)),
            }
        write_json_atomic(run / 'manifest.json', manifest)

        pending = set(seeds)
        while pending:
            finished = [
                seed for seed in sorted(pending)
                if procs[seed].poll() is not None
            ]
            if not finished:
                time.sleep(2)
                continue
            seed = finished[0]
            pending.remove(seed)
            returncode = procs[seed].returncode
            exit_codes[str(seed)] = returncode
            manifest['workers'][str(seed)]['returncode'] = returncode
            manifest['workers'][str(seed)]['status'] = (
                'completed' if returncode == 0 else 'failed')
            if returncode != 0:
                failure = f'Worker seed {seed} zakończył się kodem {returncode}'
                for other in seeds:
                    if other == seed or procs[other].poll() is not None:
                        continue
                    terminate(procs[other])
                    exit_codes[str(other)] = procs[other].returncode
                    manifest['workers'][str(other)][
                        'returncode'] = procs[other].returncode
                    manifest['workers'][str(other)]['status'] = 'terminated'
                break
            print(f'E015_WORKER_SEED={seed}_EXIT=0', flush=True)

        if failure is None:
            try:
                summaries = check_summaries(run, seeds, effective_epochs)
                for seed in seeds:
                    manifest['workers'][str(seed)].update(summaries[str(seed)])
                status = 'completed'
            except Exception as error:
                failure = str(error)
        if failure is not None:
            status = 'failed'
            manifest['failure'] = failure
            for seed in seeds:
                if str(seed) not in exit_codes:
                    exit_codes[str(seed)] = procs[seed].returncode
        manifest['worker_exit_codes'] = exit_codes
        write_json_atomic(run / 'manifest.json', manifest)
        return 0 if status == 'completed' else 1
    except BaseException as error:
        status = 'failed'
        failure = repr(error)
        (run / 'launcher_error.txt').write_text(traceback.format_exc())
        for seed, process in procs.items():
            terminate(process)
            exit_codes[str(seed)] = process.returncode
            if str(seed) in manifest['workers']:
                manifest['workers'][str(seed)][
                    'returncode'] = process.returncode
                manifest['workers'][str(seed)]['status'] = 'terminated'
        raise
    finally:
        for handle in handles.values():
            handle.close()
        manifest['status'] = status
        manifest['failure'] = failure
        manifest['worker_exit_codes'] = exit_codes
        manifest['summaries'] = summaries
        manifest['finished_utc'] = utc_now()
        manifest['exit_code'] = 0 if status == 'completed' else 1
        write_json_atomic(run / 'manifest.json', manifest)
        files = sorted(
            path for path in run.rglob('*')
            if path.is_file() and path.name != 'artifacts.sha256'
            and path.suffix != '.tmp')
        (run / 'artifacts.sha256').write_text(
            ''.join(
                f'{digester(path)}  {path.relative_to(run)}\n'
                for path in files))
        print(f'E015_STATUS={status} RUN={run}', flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
