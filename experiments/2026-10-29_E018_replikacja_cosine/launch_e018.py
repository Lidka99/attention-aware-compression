"""E018: launcher 24 zadań replikacji reference vs cosine na czterech GPU.

Eksperyment: 2026-10-29_E018_replikacja_cosine; pełny protokół: OPIS.md.
Wejście: zamrożony ``configs/stage1.json``
(6 checkpointów × 2 miary × 2 budżety).
Wyjście: unikalny ``runs/E018/<UTC>_<git>_<token>/`` z manifestem, snapshotem
źródeł, środowiskiem, specami zadań, logami i summary; fale po cztery zadania,
po jednym na GPU 0–3. Bez treningu wag.
"""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e018_common as common  # noqa: E402

ROOT = common.ROOT
RUNS = ROOT / 'runs' / 'E018'
WORKER = HERE / 'worker.py'
MIN_FREE_BYTES = 2 * 1024 ** 3
MIN_FREE_MIB = 4096
TASK_TIMEOUT_SECONDS = 1200
COMMON_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONFAULTHANDLER': '1',
}


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


def query_gpus():
    """Zwraca listę GPU z nvidia-smi."""
    command = ['nvidia-smi',
               '--query-gpu=index,uuid,name,memory.total,memory.used',
               '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True)
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 5:
            continue
        rows.append({
            'index': int(row[0]), 'uuid': row[1].strip(),
            'name': row[2].strip(),
            'memory_total_mib': int(float(row[3])),
            'memory_used_mib': int(float(row[4])),
        })
    rows.sort(key=lambda item: item['index'])
    return rows


def query_apps():
    """Zwraca PID-y procesów GPU według UUID."""
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
         '--format=csv,noheader,nounits'], text=True)
    apps = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 2:
            continue
        apps.setdefault(row[0].strip(), []).append(int(row[1]))
    return apps


def preflight_gpus():
    """Sprawdza cztery wolne fizyczne GPU 0–3."""
    gpus = query_gpus()
    require(len(gpus) >= 4, f'Wymagane 4 GPU, wykryto {len(gpus)}')
    selected = gpus[:4]
    require([gpu['index'] for gpu in selected] == [0, 1, 2, 3],
            'Oczekiwano fizycznych GPU 0–3')
    apps = query_apps()
    for gpu in selected:
        free = gpu['memory_total_mib'] - gpu['memory_used_mib']
        gpu['memory_free_mib'] = free
        require(not apps.get(gpu['uuid']),
                f"GPU {gpu['index']} ma aktywne procesy")
        require(free >= MIN_FREE_MIB,
                f"GPU {gpu['index']} ma za mało VRAM: {free} MiB")
    return selected


def create_run(git_sha):
    """Tworzy unikalny katalog biegu."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for _ in range(100):
        run = RUNS / f'{stamp}_{git_sha[:7]}_{secrets.token_hex(4)}'
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć katalogu biegu')


def snapshot_sources(run):
    """Kopiuje źródła i zapisuje ich hashe."""
    source = run / 'source'
    source.mkdir()
    hashes = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            target = source / path.name
            shutil.copy2(path, target)
            hashes[path.name] = common.digest(target)
    config_target = source / 'configs' / 'stage1.json'
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(common.CONFIG_PATH, config_target)
    hashes['configs/stage1.json'] = common.digest(config_target)
    reused = source / 'reused'
    reused.mkdir()
    for name, path in common.REUSED_SOURCES.items():
        require(path.is_file(), f'Brak używanego źródła: {path}')
        shutil.copy2(path, reused / name)
        hashes[f'reused/{name}'] = common.digest(reused / name)
    return hashes


def version_of(package):
    """Zwraca wersję pakietu lub None."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_environment(run, gpus):
    """Zapisuje środowisko i wersje."""
    import torch
    environment = {
        'python': platform.python_version(),
        'executable': sys.executable,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torchvision': version_of('torchvision'),
        'numpy': version_of('numpy'),
        'sklearn': version_of('scikit-learn'),
        'hkmeans': version_of('hkmeans'),
        'cuda': torch.version.cuda,
        'gpus': gpus,
        'deterministic_algorithms': True,
        'tf32': False,
        'amp': False,
        'env': COMMON_ENV,
    }
    common.write_json_atomic(run / 'environment.json', environment)
    for name, command in (
            ('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
            ('nvidia-smi.txt', ['nvidia-smi'])):
        try:
            text = subprocess.check_output(command, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            text = f'Nie udało się wykonać {command}: {error}\n'
        (run / name).write_text(text, encoding='utf-8')
    return environment


def write_specs(run, tasks):
    """Zapisuje specy zadań i zwraca ich hashe."""
    records = {}
    for index, task in enumerate(tasks):
        path = run / 'specs' / f'{index:02d}_{task["name"]}.json'
        common.write_json_atomic(path, task)
        records[task['name']] = {
            'path': str(path.relative_to(run)),
            'sha256': common.digest(path),
        }
    return records


def terminate(process):
    """Kończy proces i jego grupę."""
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
            process.kill()
        process.wait()


def worker_environment(gpu_index, seed):
    """Środowisko jednego workera na wskazanym GPU."""
    environment = os.environ.copy()
    environment.update(COMMON_ENV)
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    environment['PYTHONHASHSEED'] = str(seed)
    return environment


ACTIVE = []


def terminate_active():
    """Kończy wszystkie aktywne procesy workerów (best effort)."""
    for process in list(ACTIVE):
        try:
            terminate(process)
        except Exception:
            pass
    ACTIVE.clear()


def run_wave(run, tasks, gpus, config_path, manifest):
    """Uruchamia falę do czterech zadań, po jednym na GPU."""
    active = []
    failures = []
    logs = run / 'logs'
    logs.mkdir(exist_ok=True)
    for gpu, task in zip(gpus, tasks):
        gpu_index = int(gpu['index'])
        spec_path = run / manifest['specs'][task['name']]['path']
        command = [
            sys.executable, '-B', str(WORKER), '--run-dir', str(run),
            '--config', str(config_path), '--task', str(spec_path),
            '--gpu', str(gpu_index),
        ]
        log_path = logs / f"{task['name']}.log"
        handle = log_path.open('w', encoding='utf-8')
        process = subprocess.Popen(
            command, cwd=ROOT, env=worker_environment(
                gpu_index, int(manifest['fold_seed'])),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active.append({'task': task, 'process': process, 'handle': handle,
                       'log': str(log_path.relative_to(run)),
                       'gpu': gpu_index, 'command': command})
        ACTIVE.append(process)
        manifest['workers'].append({
            'task': task['name'], 'dataset': task['dataset'],
            'measure': task['measure'], 'budget': task['budget'],
            'gpu': gpu_index, 'pid': process.pid, 'command': command,
            'log': str(log_path.relative_to(run)), 'status': 'running',
            'started_utc': utc_now(),
        })
    common.write_json_atomic(run / 'manifest.json', manifest)
    deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
    try:
        for item in active:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                returncode = item['process'].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                terminate(item['process'])
                returncode = 124
            item['handle'].close()
            record = next(entry for entry in manifest['workers']
                          if entry['task'] == item['task']['name'])
            record['returncode'] = int(returncode)
            record['finished_utc'] = utc_now()
            summary_path = (run / 'tasks' / item['task']['dataset']
                            / item['task']['measure']
                            / f"b{int(round(float(item['task']['budget']) * 100)):02d}"
                            / 'summary.json')
            summary = common.read_json(summary_path) if summary_path.is_file() else {}
            ok = returncode == 0 and summary.get('status') == 'completed'
            record['status'] = 'completed' if ok else 'failed'
            if not ok:
                failures.append((item['task']['name'], returncode,
                                 summary.get('error')))
                for other in active:
                    if other is item:
                        continue
                    terminate(other['process'])
                    record_other = next(
                        entry for entry in manifest['workers']
                        if entry['task'] == other['task']['name'])
                    record_other['status'] = 'terminated'
                    record_other['finished_utc'] = utc_now()
    except BaseException:
        terminate_active()
        raise
    finally:
        ACTIVE.clear()
    common.write_json_atomic(run / 'manifest.json', manifest)
    if failures:
        raise RuntimeError(f'Błędy zadań E018: {failures}')
    print(f'E018_WAVE_DONE tasks={[task["name"] for task in tasks]}',
          flush=True)


def verify_summaries(run, cfg, manifest):
    """Sprawdza kompletność, równy budżet i brak użycia testu/rezerwy."""
    expected_dev = int(cfg['run']['dev_count'])
    summaries = {}
    for task in common.build_tasks(cfg):
        path = (run / 'tasks' / task['dataset'] / task['measure']
                / f"b{int(round(task['budget'] * 100)):02d}" / 'summary.json')
        require(path.is_file(), f'Brak summary: {path}')
        summary = common.read_json(path)
        require(summary.get('status') == 'completed',
                f"{task['name']}: status {summary.get('status')}")
        require(summary.get('official_test_used') is False,
                f"{task['name']}: użyto testu")
        require(summary.get('reserve_used') is False,
                f"{task['name']}: użyto rezerwy")
        require(summary['compressed']['post_bn']['count'] == expected_dev,
                f"{task['name']}: zły dev count")
        require(len(summary['groups']) == 12,
                f"{task['name']}: oczekiwano 12 grup")
        for record in summary['groups']:
            require(sum(record['cluster_counts']) == record['n_channels'],
                    f"{task['name']}: złe rozmiary klastrów")
            require(len(record['cluster_counts']) == record['n_clusters'],
                    f"{task['name']}: zła liczba klastrów")
        summaries[task['name']] = summary
    require(len(summaries) == 24, 'Oczekiwano 24 summary')
    for dataset in cfg['datasets']:
        for budget in cfg['budgets']:
            values = {
                task['measure']: summaries[task['name']]['params_after']
                for task in common.build_tasks(cfg)
                if task['dataset'] == dataset['tag']
                and task['budget'] == budget
            }
            require(len(set(values.values())) == 1,
                    f"Budżety różnią się między miarami: {dataset['tag']} "
                    f"{budget}: {values}")
    return summaries


def write_artifacts(run):
    """Zapisuje SHA-256 wszystkich artefaktów."""
    rows = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name != 'artifacts.sha256':
            rows.append(f'{common.digest(path)}  {path.relative_to(run)}')
    (run / 'artifacts.sha256').write_text(
        '\n'.join(rows) + '\n', encoding='utf-8')


def main():
    check_environment()
    cfg = common.load_config()
    tasks = common.build_tasks(cfg)
    usage = shutil.disk_usage(ROOT)
    require(usage.free >= MIN_FREE_BYTES,
            f'Za mało wolnego dysku: {usage.free} B')
    gpus = preflight_gpus()
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    run = create_run(git_sha)
    source_hashes = snapshot_sources(run)
    config_path = run / 'source' / 'configs' / 'stage1.json'
    require(common.digest(config_path) == common.digest(common.CONFIG_PATH),
            'Snapshot configu różni się od źródła')
    environment = write_environment(run, gpus)
    specs = write_specs(run, tasks)
    manifest = {
        'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH,
        'kind': 'stage1_cosine_replication',
        'run_id': run.name,
        'status': 'running',
        'started_utc': utc_now(),
        'finished_utc': None,
        'git_sha': git_sha,
        'git_dirty': git_dirty,
        'source_sha256': source_hashes,
        'config_sha256': common.digest(config_path),
        'environment': environment,
        'fold_seed': int(cfg['run']['fold_seed']),
        'bn_seed': int(cfg['run']['bn_seed']),
        'batch_size': int(cfg['run']['batch_size']),
        'gpus': gpus,
        'tasks': [task['name'] for task in tasks],
        'specs': specs,
        'workers': [],
        'summaries': {},
        'official_test_used': False,
        'reserve_used': False,
        'dev_used_for_grouping': False,
    }
    common.write_json_atomic(run / 'manifest.json', manifest)
    print(f'E018_RUN={run}', flush=True)
    status = 'failed'
    failure = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal={signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for start in range(0, len(tasks), 4):
            run_wave(run, tasks[start:start + 4], gpus, config_path,
                     manifest)
        manifest['summaries'] = verify_summaries(run, cfg, manifest)
        status = 'completed'
    except BaseException as error:
        failure = repr(error)
        terminate_active()
        raise
    finally:
        manifest['status'] = status
        manifest['finished_utc'] = utc_now()
        if failure is not None:
            manifest['error'] = failure
        common.write_json_atomic(run / 'manifest.json', manifest)
        write_artifacts(run)
        print(f'E018_STATUS={status} RUN={run}', flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
