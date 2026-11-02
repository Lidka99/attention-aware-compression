"""E016: launcher etapu S1 — diagnostyka wrażliwości pojedynczych grup.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu; pełny opis: OPIS.md
(``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS.md``).

Wejście: zamrożony ``configs/stage1.json`` oraz istniejące checkpointy E011
i E015. Wyjście: unikalny katalog ``runs/E016/s1_<UTC>_<git>_<token>/``
z manifestem, snapshotem źródeł, środowiskiem, shardami, logami i wynikami
98 itemów. Uruchamia cztery niezależne procesy-workerów, po jednym na GPU,
w dwóch falach (CIFAR-10, potem CIFAR-100). Każdy item ma unikalny katalog.
Bez nowego treningu, bez gradientów, bez oficjalnego testu i bez rezerwy.

Uwaga infrastrukturalna: ten launcher jawnie importuje ``os`` i używa go w
``environment_for``/``terminate``. Poprawka E015 (brak ``import os``) jest
udokumentowana w ``experiments/2026-10-27_E015_cifar100_alokacja/
AUDYT_NAPRAWY_LAUNCHERA.md``; tutaj nie powtarzamy tego błędu.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402

ROOT = common.ROOT
RUNS = ROOT / 'runs' / 'E016'
WORKER_PATH = HERE / 'sensitivity_worker.py'
MIN_FREE_BYTES = 3 * 1024 ** 3
MIN_FREE_MIB = 4096
COMMON_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONFAULTHANDLER': '1',
}
E016_SOURCES = (
    'OPIS.md',
    'PLAN.md',
    'configs/stage1.json',
    'e016_common.py',
    'sensitivity_worker.py',
    'launch_sensitivity.py',
    'analyze_e016.py',
    'check_cpu_e016.py',
)


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def utc_now():
    """Zwraca bieżący znacznik czasu UTC."""
    return datetime.now(timezone.utc).isoformat()


def check_environment():
    """Wymaga środowiska Conda attention."""
    executable = sys.executable.replace('\\', '/')
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    require(env_name == 'attention' or '/envs/attention/' in executable,
            'Uruchom launcher w środowisku Conda attention')


def disk_info():
    """Zwraca informacje o wolnym dysku."""
    usage = shutil.disk_usage(ROOT)
    return {
        'path': str(ROOT),
        'total_bytes': usage.total,
        'used_bytes': usage.used,
        'free_bytes': usage.free,
    }


def query_gpus():
    """Odczytuje GPU przez nvidia-smi."""
    command = [
        'nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,memory.used',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f'Nie można odczytać GPU: {error}') from error
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 5:
            continue
        rows.append({
            'index': int(row[0].strip()),
            'uuid': row[1].strip(),
            'name': row[2].strip(),
            'memory_total_mib': int(float(row[3].strip())),
            'memory_used_mib': int(float(row[4].strip())),
        })
    rows.sort(key=lambda item: item['index'])
    return rows


def query_compute_apps():
    """Zwraca aktywne procesy GPU pogrupowane po UUID."""
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
    """Wymaga czterech wolnych GPU 0–3 z co najmniej 4 GiB VRAM."""
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


def create_run(git_sha, smoke):
    """Tworzy unikalny katalog biegu S1 albo smoke."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'smoke' if smoke else 's1'
    for _ in range(100):
        name = f'{prefix}_{stamp}_{git_sha[:7]}_{secrets.token_hex(4)}'
        run = RUNS / name
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć unikalnego katalogu biegu')


def snapshot_sources(run, cfg):
    """Kopiuje źródła E016 i importowane źródła, zwraca ich hashe."""
    source = run / 'source'
    source.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in E016_SOURCES:
        path = HERE / relative
        require(path.is_file(), f'Brak źródła E016: {path}')
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative] = common.digest(target)
    reused = source / 'reused'
    reused.mkdir(parents=True, exist_ok=False)
    for name, path in common.SOURCES.items():
        require(path.is_file(), f'Brak importowanego źródła: {path}')
        target = reused / name
        shutil.copy2(path, target)
        hashes[f'reused/{name}'] = common.digest(target)
    return hashes


def write_environment(run, gpus):
    """Zapisuje wersje środowiska, pip freeze i nvidia-smi."""
    import torch
    import torchvision

    info = {
        'timestamp_utc': utc_now(),
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'conda_default_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'numpy': __import__('numpy').__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'deterministic_algorithms': bool(
            torch.are_deterministic_algorithms_enabled()),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'gpus': gpus,
    }
    common.write_json_atomic(run / 'environment.json', info)
    try:
        freeze = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        freeze = f'pip freeze failed: {error!r}\n'
    (run / 'pip-freeze.txt').write_text(freeze, encoding='utf-8')
    try:
        smi = subprocess.check_output(['nvidia-smi'], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        smi = f'nvidia-smi failed: {error!r}\n'
    (run / 'nvidia-smi.txt').write_text(smi, encoding='utf-8')
    conda = shutil.which('conda')
    if conda:
        try:
            explicit = subprocess.check_output(
                [conda, 'list', '--explicit', '-n', 'attention'],
                text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as error:
            explicit = f'conda list --explicit failed: {error!r}\n'
    else:
        explicit = 'conda not found on PATH\n'
    (run / 'conda-explicit.txt').write_text(explicit, encoding='utf-8')
    return info


def write_shards(run, cfg, smoke):
    """Zapisuje pliki shardów dla workerów."""
    items = common.build_items(cfg, smoke=smoke)
    written = {}
    for dataset in cfg['datasets']:
        name = dataset['name']
        dataset_items = [item for item in items if item['dataset'] == name]
        shards = common.shard_items(dataset_items, int(cfg['gpu_count']))
        for gpu, shard in shards.items():
            path = run / 'shards' / name / f'gpu{gpu}.json'
            common.write_json_atomic(path, shard)
            written[f'{name}/gpu{gpu}'] = {
                'path': str(path.relative_to(run)),
                'sha256': common.digest(path),
                'count': len(shard),
                'names': [item['name'] for item in shard],
            }
    return written


def environment_for(gpu_index):
    """Zwraca środowisko procesu workera; jawnie używa zaimportowanego os."""
    environment = dict(os.environ)
    environment.update(COMMON_ENV)
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    environment['PYTHONHASHSEED'] = '42'
    return environment


def terminate(process):
    """Kończy proces i jego grupę; jawnie używa zaimportowanego os."""
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
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait()


def run_wave(run, dataset, gpus, manifest):
    """Uruchamia cztery niezależne shardy jednego datasetu na GPU 0–3."""
    logs = run / 'logs' / dataset
    logs.mkdir(parents=True, exist_ok=True)
    active = []
    failures = []
    for gpu in gpus:
        gpu_index = int(gpu['index'])
        shard_path = run / 'shards' / dataset / f'gpu{gpu_index}.json'
        require(shard_path.is_file(), f'Brak shardu: {shard_path}')
        command = [
            sys.executable, '-B', str(WORKER_PATH),
            '--run-dir', str(run), '--dataset', dataset,
            '--gpu', str(gpu_index), '--items', str(shard_path),
        ]
        log_path = logs / f'gpu{gpu_index}.log'
        handle = log_path.open('w', encoding='utf-8')
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment_for(gpu_index),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active.append((gpu_index, process, handle, command, log_path))
        manifest['workers'].append({
            'kind': 'sensitivity_shard',
            'dataset': dataset,
            'gpu': gpu_index,
            'pid': process.pid,
            'command': command,
            'log': str(log_path.relative_to(run)),
            'status': 'running',
        })
    common.write_json_atomic(run / 'manifest.json', manifest)
    for gpu_index, process, handle, command, log_path in active:
        returncode = process.wait()
        handle.close()
        record = next(
            item for item in manifest['workers']
            if item['kind'] == 'sensitivity_shard'
            and item['dataset'] == dataset and item['gpu'] == gpu_index)
        record['returncode'] = int(returncode)
        record['status'] = 'completed' if returncode == 0 else 'failed'
        if returncode != 0:
            failures.append((dataset, gpu_index, returncode, str(log_path)))
            for other_gpu, other, other_handle, _, _ in active:
                if other is process:
                    continue
                terminate(other)
                other_handle.close()
    common.write_json_atomic(run / 'manifest.json', manifest)
    if failures:
        raise RuntimeError(f'Błędy workerów S1: {failures}')
    print(f'E016_WAVE_DONE dataset={dataset} gpus=0,1,2,3', flush=True)


def verify_summaries(run, cfg, items, smoke=False):
    """Sprawdza kompletność i poprawność wszystkich summary S1."""
    require(items, 'Brak itemów do weryfikacji')
    controls = 0
    for item in items:
        path = run / 'items' / item['dataset'] / item['name'] / 'summary.json'
        require(path.is_file(), f'Brak summary: {path}')
        summary = common.read_json(path)
        require(summary.get('status') == 'completed',
                f"{item['name']}: status {summary.get('status')}")
        require(summary.get('official_test_used') is False,
                f"{item['name']}: użyto oficjalnego testu")
        require(summary.get('reserve_used') is False,
                f"{item['name']}: użyto rezerwy")
        require(summary['compressed']['post_bn']['count'] == 5000,
                f"{item['name']}: dev count != 5000")
        require(summary['compressed']['pre_bn']['count'] == 5000,
                f"{item['name']}: pre-BN dev count != 5000")
        dataset = common.dataset_by_name(cfg, item['dataset'])
        require(
            abs(summary['baseline']['metrics']['top1']
                - float(dataset['baseline_top1'])) <= 1e-12,
            f"{item['name']}: baseline top1 nie zgadza się")
        if item['control']:
            controls += 1
            require(
                abs(summary['damage']['pre_bn_top1_pp'])
                <= float(cfg['criteria']['control_pre_bn_top1_abs_pp_max']),
                f"{item['name']}: kontrola pre-BN top1 poza tolerancją")
            require(
                summary['predictors']['calib_prediction_agreement_pre_bn']
                >= float(cfg['criteria']['control_pre_bn_agreement_min']),
                f"{item['name']}: kontrola agreement poniżej progu")
            require(
                summary['predictors']['weight_reconstruction_rel_error'] == 0.0,
                f"{item['name']}: kontrola ma niezerowy błąd rekonstrukcji")
    expected_controls = 0 if smoke else 1
    require(controls == expected_controls,
            f'Oczekiwano {expected_controls} kontroli, jest {controls}')
    return {'count': len(items), 'controls': controls}


def write_artifacts(run):
    """Zapisuje listę SHA-256 wszystkich artefaktów biegu."""
    rows = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name != 'artifacts.sha256':
            rows.append(f'{common.digest(path)}  {path.relative_to(run)}')
    (run / 'artifacts.sha256').write_text('\n'.join(rows) + '\n',
                                          encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='8 itemów smoke; bez znaczenia naukowego')
    args = parser.parse_args()

    check_environment()
    cfg = common.load_config()
    common.verify_frozen_sources(cfg)
    for dataset in cfg['datasets']:
        common.verify_dataset_inputs(dataset)
    disk = disk_info()
    require(disk['free_bytes'] >= MIN_FREE_BYTES,
            f"Za mało wolnego dysku: {disk['free_bytes']} B")
    gpus = preflight_gpus()
    git_sha = common.git_head()
    run = create_run(git_sha, smoke=args.smoke)
    source_hashes = snapshot_sources(run, cfg)
    config_target = run / 'config.json'
    shutil.copy2(common.CONFIG_PATH, config_target)
    shard_records = write_shards(run, cfg, smoke=args.smoke)
    items = common.build_items(cfg, smoke=args.smoke)
    records = []
    for dataset in cfg['datasets']:
        name = dataset['name']
        for gpu in range(int(cfg['gpu_count'])):
            for item in common.read_json(
                    run / 'shards' / name / f'gpu{gpu}.json'):
                records.append({
                    'name': item['name'],
                    'dataset': item['dataset'],
                    'group_index': item['group_index'],
                    'group_name': item['group_name'],
                    'level': item['level'],
                    'control': item['control'],
                    'gpu': gpu,
                    'shard': f'shards/{name}/gpu{gpu}.json',
                })
    require(len(records) == len(items), 'Rekordy itemów nie zgadzają się')
    manifest = {
        'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH,
        'protocol': 'OPIS.md',
        'kind': 'stage1_single_group_sensitivity',
        'run_id': run.name,
        'status': 'running',
        'smoke': bool(args.smoke),
        'started_utc': utc_now(),
        'finished_utc': None,
        'git_sha': git_sha,
        'git_dirty': common.git_dirty(),
        'source_sha256': source_hashes,
        'config_sha256': common.digest(config_target),
        'config_source': str(common.CONFIG_PATH.relative_to(ROOT)),
        'datasets': [item['name'] for item in cfg['datasets']],
        'seeds': {item['name']: int(item['seed']) for item in cfg['datasets']},
        'fold_seed': int(cfg['fold_seed']),
        'bn_seed': int(cfg['bn_seed']),
        'calibration_seed': int(cfg['calibration_seed']),
        'calibration_samples': int(cfg['calibration_samples']),
        'levels': [float(value) for value in cfg['levels']],
        'gpu_count': int(cfg['gpu_count']),
        'gpus': gpus,
        'disk': disk,
        'shards': shard_records,
        'items': records,
        'workers': [],
        'summaries': {},
        'official_test_used': False,
        'reserve_used': False,
    }
    common.write_json_atomic(run / 'manifest.json', manifest)
    print(f'E016_RUN={run}{" SMOKE=1" if args.smoke else ""}', flush=True)
    status = 'failed'
    failure = None
    try:
        manifest['environment'] = write_environment(run, gpus)
        common.write_json_atomic(run / 'manifest.json', manifest)
        for dataset in cfg['datasets']:
            name = dataset['name']
            run_wave(run, name, gpus, manifest)
            manifest['summaries'][name] = verify_summaries(
                run, cfg, [item for item in items if item['dataset'] == name],
                smoke=bool(args.smoke))
            common.write_json_atomic(run / 'manifest.json', manifest)
            print(
                f'E016_DATASET_DONE dataset={name} '
                f"items={manifest['summaries'][name]['count']}", flush=True)
        status = 'completed'
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        manifest['status'] = status
        manifest['finished_utc'] = utc_now()
        if failure is not None:
            manifest['error'] = failure
        common.write_json_atomic(run / 'manifest.json', manifest)
        write_artifacts(run)
    print(
        f'E016_RUN_DONE run={run} status={status} '
        f"items={sum(value['count'] for value in manifest['summaries'].values())}"
        f" smoke={bool(args.smoke)}", flush=True)


if __name__ == '__main__':
    main()
