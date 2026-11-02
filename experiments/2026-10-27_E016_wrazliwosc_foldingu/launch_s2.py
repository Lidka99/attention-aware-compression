"""E016 S2: launcher automatycznej alokacji 50% na czterech GPU.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu, etap S2; pełny protokół:
``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS_S2.md``.

Launcher najpierw wyznacza i zapisuje **zamrożone** alokacje UNIFORM,
TILT+0.30, TILT+0.15 i AUTO50 wyłącznie z checkpointów i primary predyktora
S1 (bez oceny dev), a dopiero potem uruchamia osiem workerów: cztery warianty
na CIFAR-10 i cztery na CIFAR-100, po jednym na GPU 0–3. Zapisuje snapshoty,
hashe, środowisko, przypisanie procesów i artefakty. Bez treningu,
bez gradientów, bez oficjalnego testu i bez rezerwy.

Uwaga infrastrukturalna: jawny ``import os``; nie powtarzamy błędu E015
(``NameError: os``). Nie ma tu kontroli liczby kontroli z etapu S1.
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
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402
import s2_common as s2  # noqa: E402

ROOT = common.ROOT
RUNS = ROOT / 'runs' / 'E016'
WORKER_PATH = HERE / 's2_worker.py'
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
    'OPIS_S2.md',
    'configs/stage2.json',
    's2_common.py',
    's2_worker.py',
    'launch_s2.py',
    'check_cpu_s2.py',
    'analyze_s2.py',
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
    """Tworzy unikalny katalog biegu S2 albo smoke."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 's2_smoke' if smoke else 's2'
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
    """Kopiuje źródła S2, importowane źródła i artefakty S1; zwraca hashe."""
    source = run / 'source'
    source.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in E016_SOURCES:
        path = HERE / relative
        require(path.is_file(), f'Brak źródła S2: {path}')
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
    s1_analysis = ROOT / cfg['s1']['analysis_dir']
    s1_copy = source / 's1_analysis'
    s1_copy.mkdir(parents=True, exist_ok=False)
    for name in ('analysis.json', 'group_level.csv', 'analysis.md'):
        path = s1_analysis / name
        require(path.is_file(), f'Brak artefaktu S1: {path}')
        target = s1_copy / name
        shutil.copy2(path, target)
        hashes[f's1_analysis/{name}'] = common.digest(target)
    return hashes


def write_environment(run, gpus):
    """Zapisuje wersje środowiska, pip freeze, conda list i nvidia-smi."""
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


def freeze_allocations(run, cfg, predictors):
    """Wyznacza alokacje bez dev i zapisuje je przed startem workerów."""
    frozen = {
        'experiment': cfg['experiment'],
        'stage': cfg['stage'],
        'opis': cfg['opis'],
        'created_utc': None,
        'primary_predictor': cfg['primary_predictor'],
        'dev_used_for_allocation': False,
        'uses_test': False,
        'uses_reserve': False,
        'datasets': {},
    }
    spec_records = {}
    audit = {
        'created_utc': None,
        'primary_predictor': cfg['primary_predictor'],
        'dev_used_for_allocation': False,
        'uses_test': False,
        'uses_reserve': False,
        'datasets': {},
    }
    for dataset in cfg['datasets']:
        name = dataset['name']
        print(f'E016_S2_ALLOC_START dataset={name}', flush=True)
        bundle = common.load_dataset_bundle(dataset, verify_hashes=True)
        source_cpu = bundle['source_cpu']
        variants = s2.resolve_dataset(source_cpu, cfg, predictors, name)
        baseline_params = int(sum(parameter.numel()
                                  for parameter in source_cpu.parameters()))
        uniform = next(item for item in variants if item['name'] == 'uniform')
        target = int(uniform['params'])
        for allocation in variants:
            allocation['uniform_reference_params'] = target
            allocation['nominal_target_params'] = int(round(
                0.5 * baseline_params))
            allocation['baseline_params'] = baseline_params
        frozen['datasets'][name] = {
            'baseline_params': baseline_params,
            'uniform_reference_params': target,
            'variants': {item['name']: item for item in variants},
        }
        audit['datasets'][name] = {
            'baseline_params': baseline_params,
            'nominal_target_params': int(round(0.5 * baseline_params)),
            'uniform_reference_params': target,
            'group_names': predictors[name]['group_names'],
            'levels': predictors[name]['levels'],
            'recon_matrix': predictors[name]['matrix'].tolist(),
            'level_mean': predictors[name]['level_mean'].tolist(),
            'level_sd': predictors[name]['level_sd'].tolist(),
            'level_z': predictors[name]['level_z'].tolist(),
            'score': predictors[name]['score'].tolist(),
            'z': predictors[name]['z'].tolist(),
            'fixed_level_z': predictors[name]['fixed_level_z'].tolist(),
            'variants': {
                item['name']: {
                    'params': int(item['params']),
                    'target_params': int(item['target_params']),
                    'delta_params': int(item['delta_params']),
                    'delta_percent': float(item['delta_percent']),
                    'ratios': item['ratios'],
                    'target_ratios': item.get('target_ratios'),
                    'widths': item.get('widths'),
                    'beta': item.get('beta'),
                    'base': item.get('base'),
                    'repair_steps': item.get('repair_steps'),
                    'repair_applied': item.get('repair_applied'),
                    'gap_before_repair_params': item.get(
                        'gap_before_repair_params'),
                    'gap_before_repair_percent': item.get(
                        'gap_before_repair_percent'),
                    'base_fallback_no_feasible': item.get(
                        'base_fallback_no_feasible'),
                    'base_feasible_count': item.get('base_feasible_count'),
                }
                for item in variants
            },
        }
        spec_dir = run / 'specs' / name
        spec_dir.mkdir(parents=True, exist_ok=False)
        spec_records[name] = {}
        for allocation in variants:
            payload = s2.spec_payload(
                allocation, name, int(dataset['seed']), cfg)
            path = spec_dir / f"{allocation['name']}.json"
            common.write_json_atomic(path, payload)
            spec_records[name][allocation['name']] = {
                'path': str(path.relative_to(run)),
                'sha256': common.digest(path),
                'params': int(allocation['params']),
                'target_params': int(allocation['target_params']),
                'delta_params': int(allocation['delta_params']),
                'delta_percent': float(allocation['delta_percent']),
                'dev_used_for_allocation': False,
            }
        del bundle
        print(
            f'E016_S2_ALLOC_DONE dataset={name} target={target} '
            f"variants={[item['name'] for item in variants]}", flush=True)
    frozen['created_utc'] = utc_now()
    audit['created_utc'] = frozen['created_utc']
    common.write_json_atomic(run / 'allocations_frozen.json', frozen)
    common.write_json_atomic(run / 'allocation_audit.json', audit)
    return frozen, audit, spec_records


def run_wave(run, cfg, dataset, gpus, manifest, spec_records, smoke):
    """Uruchamia cztery warianty jednego datasetu na GPU 0–3."""
    logs = run / 'logs' / dataset
    logs.mkdir(parents=True, exist_ok=True)
    active = []
    failures = []
    for gpu, variant in zip(gpus, cfg['variants']):
        variant_name = variant['name']
        gpu_index = int(gpu['index'])
        spec_path = run / spec_records[dataset][variant_name]['path']
        require(spec_path.is_file(), f'Brak spec: {spec_path}')
        command = [
            sys.executable, '-B', str(WORKER_PATH),
            '--run-dir', str(run),
            '--config', str(run / 'config.json'),
            '--spec', str(spec_path),
            '--gpu', str(gpu_index),
        ]
        if smoke:
            command.append('--smoke')
        log_path = logs / f'{variant_name}.log'
        handle = log_path.open('w', encoding='utf-8')
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment_for(gpu_index),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record = {
            'kind': 's2_allocation',
            'dataset': dataset,
            'variant': variant_name,
            'gpu': gpu_index,
            'pid': process.pid,
            'command': command,
            'log': str(log_path.relative_to(run)),
            'spec_sha256': spec_records[dataset][variant_name]['sha256'],
            'status': 'running',
            'started_utc': utc_now(),
        }
        manifest['workers'].append(record)
        active.append((gpu_index, variant_name, process, handle, record))
    manifest['workers_launched_utc'] = utc_now()
    common.write_json_atomic(run / 'manifest.json', manifest)
    for gpu_index, variant_name, process, handle, record in active:
        returncode = process.wait()
        handle.close()
        record['returncode'] = int(returncode)
        record['finished_utc'] = utc_now()
        summary_path = run / 'allocations' / dataset / variant_name / 'summary.json'
        summary = common.read_json(summary_path) if summary_path.is_file() else {}
        ok = returncode == 0 and summary.get('status') == 'completed'
        record['status'] = 'completed' if ok else 'failed'
        if not ok:
            failures.append((dataset, variant_name, returncode,
                             summary.get('error'), str(summary_path)))
            for other_gpu, other_name, other, other_handle, other_record in active:
                if other is process:
                    continue
                terminate(other)
                other_handle.close()
                other_record['status'] = 'terminated'
                other_record['finished_utc'] = utc_now()
    common.write_json_atomic(run / 'manifest.json', manifest)
    if failures:
        raise RuntimeError(f'Błędy workerów S2: {failures}')
    print(f'E016_S2_WAVE_DONE dataset={dataset} gpus=0,1,2,3', flush=True)


def verify_summaries(run, cfg, spec_records, smoke, dataset_name=None):
    """Sprawdza kompletność, target, BN, tolerancję i brak testu/rezerwy."""
    expected_dev = s2.SMOKE_DEV if smoke else 5000
    report = {}
    for dataset in cfg['datasets']:
        name = dataset['name']
        if dataset_name is not None and name != dataset_name:
            continue
        summaries = {}
        bn_hashes = set()
        for variant in cfg['variants']:
            variant_name = variant['name']
            path = run / 'allocations' / name / variant_name / 'summary.json'
            require(path.is_file(), f'Brak summary: {path}')
            summary = common.read_json(path)
            require(summary.get('status') == 'completed',
                    f'{name}/{variant_name}: status {summary.get("status")}')
            require(summary.get('official_test_used') is False,
                    f'{name}/{variant_name}: użyto testu')
            require(summary.get('reserve_used') is False,
                    f'{name}/{variant_name}: użyto rezerwy')
            require(summary.get('dev_used_for_allocation') is False,
                    f'{name}/{variant_name}: dev użyty do alokacji')
            require(summary.get('allocation_frozen_before_evaluation') is True,
                    f'{name}/{variant_name}: alokacja niezamrożona')
            require(summary.get('allocation_dev_used') is False,
                    f'{name}/{variant_name}: alokacja użyła dev')
            require(summary.get('spec_sha256')
                    == spec_records[name][variant_name]['sha256'],
                    f'{name}/{variant_name}: zły hash spec')
            allocation = summary['allocation']
            require(allocation['name'] == variant_name,
                    f'{name}/{variant_name}: zła nazwa alokacji')
            require(int(summary['compressed']['params'])
                    == int(allocation['params']),
                    f'{name}/{variant_name}: params != spec')
            require(int(summary['target_params'])
                    == int(allocation['target_params']),
                    f'{name}/{variant_name}: target != spec')
            require(int(summary['compressed']['params'])
                    <= int(summary['target_params']),
                    f'{name}/{variant_name}: przekroczony target')
            require(-0.5 <= float(summary['delta_percent']) <= 0.0,
                    f'{name}/{variant_name}: poza tolerancją 0.5%')
            require(summary['compressed']['post_bn']['count'] == expected_dev,
                    f'{name}/{variant_name}: zły dev count')
            require(summary['compressed']['pre_bn']['count'] == expected_dev,
                    f'{name}/{variant_name}: zły pre-BN dev count')
            if not smoke:
                require(abs(summary['baseline']['metrics']['top1']
                            - float(dataset['baseline_top1'])) <= 1e-12,
                        f'{name}/{variant_name}: baseline top1')
                require(abs(summary['baseline']['metrics']['nll']
                            - float(dataset['baseline_nll'])) <= 1e-5,
                        f'{name}/{variant_name}: baseline NLL')
            bn = summary['compressed']['bn']
            require(bn['params_unchanged'] is True,
                    f'{name}/{variant_name}: BN zmienił parametry')
            require(bn['params_have_grad'] is False,
                    f'{name}/{variant_name}: gradienty w BN')
            expected_bn = s2.SMOKE_TRAIN if smoke else int(dataset['bn_samples'])
            require(int(bn['samples']) == expected_bn,
                    f'{name}/{variant_name}: zła liczba próbek BN')
            bn_hashes.add(str(bn['first_batch_sha256']))
            logits_path = run / 'allocations' / name / variant_name / 'logits.npz'
            require(logits_path.is_file(), f'Brak logits: {logits_path}')
            require(common.digest(logits_path) == summary['logits_sha256'],
                    f'{name}/{variant_name}: zły hash logits')
            diagnostics_path = (run / 'allocations' / name / variant_name
                                / 'diagnostics.json')
            require(diagnostics_path.is_file(),
                    f'Brak diagnostics: {diagnostics_path}')
            require(common.digest(diagnostics_path)
                    == summary['diagnostics_sha256'],
                    f'{name}/{variant_name}: zły hash diagnostics')
            summaries[variant_name] = summary
        require(len(bn_hashes) == 1,
                f'{name}: warianty nie użyły identycznego pierwszego batcha BN')
        uniform = summaries['uniform']
        target = int(uniform['compressed']['params'])
        for variant_name, summary in summaries.items():
            require(int(summary['target_params']) == target,
                    f'{name}/{variant_name}: target != UNIFORM')
            require(int(summary['uniform_reference_params']) == target,
                    f'{name}/{variant_name}: uniform_reference != UNIFORM')
        report[name] = {
            'variants': list(summaries),
            'uniform_params': target,
            'bn_first_batch_sha256': next(iter(bn_hashes)),
            'dev_count': expected_dev,
            'bn_samples': s2.SMOKE_TRAIN if smoke else int(dataset['bn_samples']),
        }
        print(f'E016_S2_VERIFY_OK dataset={name} target={target}', flush=True)
    return report


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
                        help='8 wariantów na małym BN/dev; bez znaczenia naukowego')
    args = parser.parse_args()

    check_environment()
    cfg = s2.load_config()
    s2.verify_sources(cfg)
    for dataset in cfg['datasets']:
        common.verify_dataset_inputs(dataset)
    disk = disk_info()
    require(disk['free_bytes'] >= MIN_FREE_BYTES,
            f"Za mało wolnego dysku: {disk['free_bytes']} B")
    gpus = preflight_gpus()
    git_sha = common.git_head()
    run = create_run(git_sha, smoke=bool(args.smoke))
    source_hashes = snapshot_sources(run, cfg)
    shutil.copy2(s2.CONFIG_PATH, run / 'config.json')
    environment = write_environment(run, gpus)
    manifest = {
        'experiment': cfg['experiment'],
        'opis': cfg['opis'],
        'protocol': 'OPIS_S2.md',
        'kind': 'stage2_global_allocation_50',
        'run_id': run.name,
        'status': 'running',
        'smoke': bool(args.smoke),
        'started_utc': utc_now(),
        'finished_utc': None,
        'git_sha': git_sha,
        'git_dirty': common.git_dirty(),
        'source_sha256': source_hashes,
        'config_sha256': common.digest(run / 'config.json'),
        'config_source': str(s2.CONFIG_PATH.relative_to(ROOT)),
        'datasets': [item['name'] for item in cfg['datasets']],
        'seeds': {item['name']: int(item['seed']) for item in cfg['datasets']},
        'fold_seed': int(cfg['run']['fold_seed']),
        'bn_seed': int(cfg['run']['bn_seed']),
        'batch_size': int(cfg['run']['batch_size']),
        'gpu_count': int(cfg['run']['gpu_count']),
        'gpus': gpus,
        'disk': disk,
        'environment': environment,
        'allocations_frozen_utc': None,
        'workers_launched_utc': None,
        'specs': {},
        'workers': [],
        'summaries': {},
        'official_test_used': False,
        'reserve_used': False,
        'dev_used_for_allocation': False,
    }
    common.write_json_atomic(run / 'manifest.json', manifest)
    print(f'E016_S2_RUN={run}{" SMOKE=1" if args.smoke else ""}', flush=True)
    status = 'failed'
    failure = None
    try:
        predictors = s2.load_predictors(cfg)
        frozen, audit, spec_records = freeze_allocations(
            run, cfg, predictors)
        manifest['allocations_frozen_utc'] = frozen['created_utc']
        manifest['specs'] = spec_records
        common.write_json_atomic(run / 'manifest.json', manifest)
        require(manifest['allocations_frozen_utc'] is not None,
                'Brak znacznika zamrożenia alokacji')
        for dataset in cfg['datasets']:
            name = dataset['name']
            run_wave(run, cfg, name, gpus, manifest, spec_records,
                     bool(args.smoke))
            manifest['summaries'].update(verify_summaries(
                run, cfg, spec_records, bool(args.smoke), dataset_name=name))
            common.write_json_atomic(run / 'manifest.json', manifest)
        require(set(manifest['summaries']) == set(manifest['datasets']),
                'Nie zweryfikowano wszystkich datasetów')
        require(manifest['workers_launched_utc'] is not None,
                'Brak znacznika startu workerów')
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
        f'E016_S2_RUN_DONE run={run} status={status} '
        f"datasets={len(manifest['summaries'])} smoke={bool(args.smoke)}",
        flush=True)


if __name__ == '__main__':
    main()
