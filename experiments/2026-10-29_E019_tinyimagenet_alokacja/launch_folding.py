"""E019: launcher zamrożonego foldingu na czterech checkpointach Tiny ImageNet.

Eksperyment: 2026-10-29_E019_tinyimagenet_alokacja; pełny opis: OPIS.md.
Wejście: ukończony bieg ``launch_training.py`` (``--train-run``). Wyjście:
nowy katalog ``runs/E019/fold_<UTC>_<git>_<token>/`` z 4 configami seedów,
48 specami, logami, summary.json i manifestem. Najpierw cztery procesy
konfiguracyjne liczą baseline na dev (GPU 0–3), potem 48 workerów
(4 seedy × 3 budżety × 4 warianty) falami po cztery na czterech GPU.
Bez fine-tuningu; oficjalny test i rezerwa nietknięte.
"""

from __future__ import annotations

import argparse
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_tiny as data  # noqa: E402
import launch_training as train_launch  # noqa: E402

ROOT = data.ROOT
RUNS = ROOT / 'runs' / 'E019'
ALLOCATION_PATH = HERE / 'configs' / 'allocation.json'
PREPARE_PATH = HERE / 'prepare_folding_config.py'
WORKER_PATH = HERE / 'folding_worker.py'
SEEDS = (42, 43, 44, 45)
MIN_FREE_BYTES = 2 * 1024 ** 3
FOLDING_ONLY_SOURCES = (
    'folding_worker.py', 'prepare_folding_config.py', 'launch_folding.py',
    'check_cpu_e019.py', 'analyze_e019.py',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc_now():
    return data.utc_now()


def read_json(path):
    return data.read_json(path)


def load_allocation():
    allocation = read_json(ALLOCATION_PATH)
    require(allocation.get('status') == 'frozen',
            'Config alokacji nie jest zamrożony')
    require(allocation.get('official_test_used') is False,
            'Alokacja używa oficjalnego testu')
    require(allocation.get('reserve_used') is False,
            'Alokacja używa rezerwy')
    require([float(value) for value in allocation['budgets']]
            == [0.30, 0.50, 0.70], 'Budżety muszą wynosić 30/50/70%')
    variants = allocation['variants']
    require([item['name'] for item in variants]
            == ['uniform', 'tilt_p030', 'tilt_p015', 'mag2'],
            'Nieoczekiwany zestaw wariantów')
    require([item['role'] for item in variants]
            == ['baseline', 'primary', 'secondary', 'comparator'],
            'Nieoczekiwane role wariantów')
    return allocation


def load_training_run(path):
    train_run = Path(path).expanduser().resolve()
    manifest_path = train_run / 'manifest.json'
    require(manifest_path.is_file(), f'Brak manifestu treningu: {manifest_path}')
    manifest = read_json(manifest_path)
    require(manifest.get('experiment') == data.EXPERIMENT,
            'Bieg treningowy nie dotyczy E019')
    require(manifest.get('status') == 'completed',
            'Bieg treningowy E019 nie jest ukończony')
    require(manifest.get('smoke') is False,
            'Folding wymaga pełnego treningu, nie smoke')
    require(manifest.get('data_only') is False,
            'Folding wymaga treningu, nie samego przygotowania danych')
    require(int(manifest.get('effective_epochs', 0)) == 200,
            'Folding wymaga 200 epok treningu')
    require(manifest.get('official_test_used') is False,
            'Bieg treningowy użył oficjalnego testu')
    require(manifest.get('reserve_used') is False,
            'Bieg treningowy użył rezerwy')
    require([int(value) for value in manifest.get('seeds', [])] == list(SEEDS),
            'Bieg treningowy nie zawiera seedów 42–45')
    for seed in SEEDS:
        checkpoint = train_run / f'seed_{seed}' / 'last.pt'
        summary = train_run / f'seed_{seed}' / 'summary.json'
        require(checkpoint.is_file(), f'Brak checkpointu seeda {seed}')
        require(summary.is_file(), f'Brak summary seeda {seed}')
        require(read_json(summary).get('status') == 'completed',
                f'Trening seeda {seed} nie jest ukończony')
        entry = manifest.get('summaries', {}).get(str(seed), {})
        require(entry.get('last_checkpoint_sha256') == data.digest(checkpoint),
                f'Hash checkpointu seeda {seed} nie zgadza się z manifestem')
    return train_run, manifest


def write_json_atomic(path, value):
    data.write_json_atomic(path, value)


def verify_source_consistency(training_manifest, training_run):
    """Weryfikuje wejścia treningu i rejestruje pliki foldingu dodane później.

    Trening wystartował przed zapisaniem kodu foldingu, więc te pliki nie są
    w snapshocie treningu. Ich hashe są zapisywane w manifeście foldingu;
    żaden plik wejściowy treningu nie może się zmienić.
    """
    recorded = training_manifest.get('source_sha256') or {}
    folding_only = set(FOLDING_ONLY_SOURCES)
    revisions = {}
    verified = 0
    for name in recorded:
        if name.startswith('reused/'):
            reused_name = name.split('/', 1)[1]
            require(reused_name in train_launch.REUSED_SOURCES,
                    f'Nieznane źródło wielokrotnego użytku: {name}')
            path = train_launch.REUSED_SOURCES[reused_name]
        else:
            path = HERE / name
        require(path.is_file(), f'Brak źródła {name}')
        value = data.digest(path)
        if recorded[name] == value:
            verified += 1
            continue
        require(name in folding_only,
                f'Kod {name} zmienił się po treningu; folding przerwany')
        revisions[name] = {
            'training_sha256': recorded[name],
            'current_sha256': value,
            'reason': 'Plik foldingu, nie wejście treningu.',
        }
    added = {}
    for name in FOLDING_ONLY_SOURCES:
        path = HERE / name
        require(path.is_file(), f'Brak pliku foldingu: {path}')
        value = data.digest(path)
        if name not in recorded:
            added[name] = value
    return {
        'verified_training_sources': verified,
        'folding_revisions': revisions,
        'added_after_training': added,
        'note': ('Pliki foldingu powstały lub były poprawiane po starcie '
                 'treningu; ich hashe są rejestrowane tutaj. Wejścia treningu '
                 '(dane, model, config) bez zmian.'),
    }


def create_run(git_sha):
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for _ in range(100):
        name = f'fold_{stamp}_{git_sha[:7]}_{secrets.token_hex(4)}'
        run = RUNS / name
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć unikalnego katalogu foldingu')


def build_specs(allocation):
    specs = []
    for budget in allocation['budgets']:
        tag = f'b{int(round(float(budget) * 100)):02d}'
        for variant in allocation['variants']:
            spec = {
                'name': f"{tag}_{variant['name']}",
                'budget': float(budget),
                'family': variant['family'],
                'role': variant['role'],
            }
            if 'tilt' in variant:
                spec['tilt'] = float(variant['tilt'])
            specs.append(spec)
    require(len(specs) == 12, 'Oczekiwano 12 speców na seed')
    return specs


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


def environment_for(gpu_index, seed):
    environment = dict(os.environ)
    environment.update(train_launch.COMMON_ENV)
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    environment['PYTHONHASHSEED'] = str(seed)
    return environment


def run_prepare_configs(fold_run, train_run, gpus, manifest):
    logs = fold_run / 'logs' / 'prepare'
    logs.mkdir(parents=True, exist_ok=True)
    active = []
    for seed, gpu in zip(SEEDS, gpus):
        command = [
            sys.executable, '-B', str(PREPARE_PATH),
            '--train-run', str(train_run), '--fold-run', str(fold_run),
            '--seed', str(seed), '--gpu', str(gpu['index']),
        ]
        log_path = logs / f'seed_{seed}.log'
        handle = log_path.open('w')
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment_for(gpu['index'], seed),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active.append((seed, gpu['index'], process, handle, command, log_path))
        manifest['workers'].append({
            'kind': 'prepare_config',
            'seed': seed,
            'gpu': gpu['index'],
            'pid': process.pid,
            'command': command,
            'log': str(log_path.relative_to(fold_run)),
            'status': 'running',
        })
    write_json_atomic(fold_run / 'manifest.json', manifest)
    failures = []
    for seed, gpu_index, process, handle, command, log_path in active:
        returncode = process.wait()
        handle.close()
        record = next(
            item for item in manifest['workers']
            if item['kind'] == 'prepare_config' and item['seed'] == seed)
        record['returncode'] = int(returncode)
        record['status'] = 'completed' if returncode == 0 else 'failed'
        if returncode != 0:
            failures.append((seed, returncode, log_path))
    if failures:
        raise RuntimeError(f'Przygotowanie configów nie powiodło się: {failures}')
    config_hashes = {}
    for seed in SEEDS:
        config_path = fold_run / 'seeds' / f'seed_{seed}' / 'source' / 'config.json'
        require(config_path.is_file(), f'Brak configu seeda {seed}')
        cfg = read_json(config_path)
        require(int(cfg['seed']) == seed, 'Niezgodny seed w configu')
        require(data.digest(Path(cfg['checkpoint'])) == cfg['checkpoint_sha256'],
                'Zły hash checkpointu w configu')
        require(data.digest(Path(cfg['split'])) == cfg['split_sha256'],
                'Zły hash splitu w configu')
        config_hashes[str(seed)] = data.digest(config_path)
    manifest['config_sha256'] = config_hashes
    write_json_atomic(fold_run / 'manifest.json', manifest)


def write_specs(fold_run, specs):
    specs_dir = fold_run / 'specs'
    specs_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        path = specs_dir / f"{spec['name']}.json"
        write_json_atomic(path, spec)
        records.append({
            'name': spec['name'],
            'path': str(path.relative_to(fold_run)),
            'sha256': data.digest(path),
            'budget': spec['budget'],
            'family': spec['family'],
            'role': spec['role'],
        })
    return records


def worker_tasks(fold_run, gpus):
    tasks = []
    for seed in SEEDS:
        config_path = fold_run / 'seeds' / f'seed_{seed}' / 'source' / 'config.json'
        for name in sorted(
                path.stem for path in (fold_run / 'specs').glob('*.json')):
            spec_path = fold_run / 'specs' / f'{name}.json'
            tasks.append({
                'seed': seed,
                'name': name,
                'config': config_path,
                'spec': spec_path,
            })
    require(len(tasks) == 48, 'Oczekiwano 48 zadań foldingu')
    return tasks


def run_workers(fold_run, tasks, gpus, manifest):
    logs = fold_run / 'logs'
    failures = []
    wave_size = 4
    for start in range(0, len(tasks), wave_size):
        wave = tasks[start:start + wave_size]
        active = []
        for offset, task in enumerate(wave):
            gpu_index = gpus[(start + offset) % 4]['index']
            command = [
                sys.executable, '-B', str(WORKER_PATH),
                '--run-dir', str(fold_run), '--config', str(task['config']),
                '--spec', str(task['spec']), '--gpu', str(gpu_index),
            ]
            log_dir = logs / f"seed_{task['seed']}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{task['name']}.log"
            handle = log_path.open('w')
            process = subprocess.Popen(
                command, cwd=ROOT,
                env=environment_for(gpu_index, task['seed']),
                stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            active.append((task, gpu_index, process, handle, command, log_path))
            manifest['workers'].append({
                'kind': 'allocation',
                'seed': task['seed'],
                'name': task['name'],
                'gpu': gpu_index,
                'pid': process.pid,
                'command': command,
                'log': str(log_path.relative_to(fold_run)),
                'status': 'running',
            })
        write_json_atomic(fold_run / 'manifest.json', manifest)
        for task, gpu_index, process, handle, command, log_path in active:
            returncode = process.wait()
            handle.close()
            summary_path = (
                fold_run / 'allocations' / f"seed_{task['seed']}"
                / task['name'] / 'summary.json')
            summary = read_json(summary_path) if summary_path.is_file() else {}
            ok = (returncode == 0 and summary.get('status') == 'completed')
            record = next(
                item for item in manifest['workers']
                if item['kind'] == 'allocation' and item['seed'] == task['seed']
                and item['name'] == task['name'])
            record['returncode'] = int(returncode)
            record['status'] = (
                'completed' if ok else summary.get('status', 'failed'))
            if ok:
                manifest['summaries'][f"seed_{task['seed']}/{task['name']}"] = {
                    'summary_sha256': data.digest(summary_path),
                    'post_bn_top1': summary['compressed']['post_bn']['top1'],
                    'post_bn_nll': summary['compressed']['post_bn']['nll'],
                    'params': summary['compressed']['params'],
                }
            else:
                failures.append((
                    task['seed'], task['name'], int(returncode),
                    summary.get('error'), str(log_path)))
                for other_task, other_gpu, other_process, other_handle, _, _ in active:
                    if other_process is process:
                        continue
                    terminate(other_process)
        write_json_atomic(fold_run / 'manifest.json', manifest)
        print(f'E019_FOLD_WAVE_DONE={min(start + wave_size, len(tasks))}/48',
              flush=True)
        if failures:
            raise RuntimeError(f'Błędy workerów foldingu: {failures}')


def verify_summaries(fold_run, allocation):
    count = 0
    summaries = {}
    for seed in SEEDS:
        for spec_path in sorted((fold_run / 'specs').glob('*.json')):
            name = spec_path.stem
            summary_path = (
                fold_run / 'allocations' / f'seed_{seed}' / name / 'summary.json')
            require(summary_path.is_file(), f'Brak summary {seed}/{name}')
            summary = read_json(summary_path)
            require(summary.get('status') == 'completed',
                    f'{seed}/{name}: status {summary.get("status")}')
            require(summary.get('official_test_used') is False,
                    f'{seed}/{name}: użyto oficjalnego testu')
            require(summary.get('reserve_used') is False,
                    f'{seed}/{name}: użyto rezerwy')
            require(summary['compressed']['post_bn']['count'] == data.EXPECTED_DEV,
                    f'{seed}/{name}: dev count != {data.EXPECTED_DEV}')
            summaries[(seed, name)] = summary
            count += 1
    require(count == 48, f'Oczekiwano 48 summary, jest {count}')

    variants = [item['name'] for item in allocation['variants']]
    budgets = [int(round(float(value) * 100))
               for value in allocation['budgets']]
    for seed in SEEDS:
        for budget in budgets:
            uniform = summaries[(seed, f'b{budget}_uniform')]
            reference = int(uniform['target_params'])
            nominal = int(uniform['nominal_target_params'])
            require(reference == int(uniform['compressed']['params']),
                    f'{seed}/{budget}%: target UNIFORM != jego parametrom')
            require(reference == int(uniform['uniform_reference_params']),
                    f'{seed}/{budget}%: zły uniform_reference_params')
            require(nominal == int(round(
                (1.0 - budget / 100.0)
                * int(uniform['compressed']['params_before']))),
                f'{seed}/{budget}%: zły nominalny cel')
            for variant in variants:
                name = f'b{budget}_{variant}'
                summary = summaries[(seed, name)]
                params = int(summary['compressed']['params'])
                target = int(summary['target_params'])
                require(target == reference,
                        f'{seed}/{name}: target != faktyczny UNIFORM')
                require(int(summary['uniform_reference_params']) == reference,
                        f'{seed}/{name}: zły wspólny target')
                require(int(summary['nominal_target_params']) == nominal,
                        f'{seed}/{name}: zły nominalny cel')
                require(params <= target,
                        f'{seed}/{name}: przekroczono wspólny target')
                tolerance = float(summary.get('budget_tolerance_percent', 1.0))
                delta = 100.0 * (params - target) / target
                require(-tolerance - 1e-9 <= delta <= 1e-9,
                        f'{seed}/{name}: odchylenie {delta:.4f}% poza '
                        f'tolerancją {tolerance:.1f}%')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', type=Path, required=True,
                        help='ukończony katalog runs/E019/... z treningiem')
    args = parser.parse_args()

    train_launch.check_environment()
    allocation = load_allocation()
    train_run, training_manifest = load_training_run(args.train_run)
    source_revisions = verify_source_consistency(training_manifest, train_run)
    disk = train_launch.disk_info()
    require(disk['free_bytes'] > MIN_FREE_BYTES,
            f"Za mało wolnego dysku: {disk['free_bytes']} B")
    gpus = train_launch.preflight_gpus()
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    fold_run = create_run(git_sha)
    source_hashes = train_launch.snapshot_sources(fold_run)
    logs = fold_run / 'logs'
    logs.mkdir()
    specs = build_specs(allocation)
    spec_records = write_specs(fold_run, specs)
    manifest = {
        'experiment': data.EXPERIMENT,
        'opis': data.OPIS_PATH,
        'protocol': 'OPIS.md',
        'kind': 'folding',
        'run_id': fold_run.name,
        'status': 'running',
        'started_utc': utc_now(),
        'finished_utc': None,
        'git_sha': git_sha,
        'git_dirty': bool(git_status.strip()),
        'source_sha256': source_hashes,
        'source_revisions': source_revisions,
        'training_run': str(train_run),
        'training_manifest_sha256': data.digest(train_run / 'manifest.json'),
        'training_git_sha': training_manifest.get('git_sha'),
        'seeds': list(SEEDS),
        'gpus': gpus,
        'specs': spec_records,
        'config_sha256': {},
        'workers': [],
        'summaries': {},
        'official_test_used': False,
        'reserve_used': False,
    }
    write_json_atomic(fold_run / 'manifest.json', manifest)
    print(f'E019_FOLD_RUN={fold_run}', flush=True)
    status = 'failed'
    failure = None
    try:
        manifest['versions'] = train_launch.write_environment(fold_run, gpus)
        write_json_atomic(fold_run / 'manifest.json', manifest)
        run_prepare_configs(fold_run, train_run, gpus, manifest)
        tasks = worker_tasks(fold_run, gpus)
        run_workers(fold_run, tasks, gpus, manifest)
        count = verify_summaries(fold_run, allocation)
        summary = {
            'experiment': data.EXPERIMENT,
            'opis': data.OPIS_PATH,
            'status': 'completed',
            'fold_run': fold_run.name,
            'training_run': str(train_run),
            'seeds': list(SEEDS),
            'budgets': allocation['budgets'],
            'variants': [item['name'] for item in allocation['variants']],
            'summaries_checked': count,
            'official_test_used': False,
            'reserve_used': False,
            'finished_utc': utc_now(),
        }
        write_json_atomic(fold_run / 'summary.json', summary)
        status = 'completed'
    except BaseException as error:
        status = 'failed'
        failure = repr(error)
        raise
    finally:
        manifest['status'] = status
        manifest['failure'] = failure
        manifest['finished_utc'] = utc_now()
        write_json_atomic(fold_run / 'manifest.json', manifest)
        artifacts = []
        for path in sorted(fold_run.rglob('*')):
            if path.is_file() and path.name != 'artifacts.sha256':
                artifacts.append(
                    f'{data.digest(path)}  {path.relative_to(fold_run)}')
        (fold_run / 'artifacts.sha256').write_text(
            '\n'.join(artifacts) + '\n')
        print(f'E019_FOLD_STATUS={status}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
