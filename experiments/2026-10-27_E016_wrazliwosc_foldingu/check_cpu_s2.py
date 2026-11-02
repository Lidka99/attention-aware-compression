"""E016 S2: CPU preflight bez GPU i bez oceny dev.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu, etap S2; pełny protokół:
``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS_S2.md``.

Sprawdza środowisko, hashe wejść i artefaktów S1, dane train/dev, brak
użycia testu/rezerwy, formułę AUTO wyłącznie z primary predyktora S1,
rzeczywiste liczby parametrów po foldingu dla UNIFORM/TILT/AUTO, target
UNIFORM, tolerancję 0,5%, kolejność zamrożenia alokacji przed startem
workerów oraz brak importów attention_v2/attention_v3. Zapisuje raport w
``runs/E016/s2_cpu_preflight_<UTC>_<token>/``. Nie uruchamia GPU ani
ewaluacji dev.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402
import s2_common as s2  # noqa: E402

ROOT = common.ROOT
RUNS = ROOT / 'runs' / 'E016'


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def utc_now():
    """Zwraca bieżący znacznik czasu UTC."""
    return datetime.now(timezone.utc).isoformat()


def make_report_dir():
    """Tworzy unikalny katalog raportu CPU preflight."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for _ in range(100):
        path = RUNS / f's2_cpu_preflight_{stamp}_{secrets.token_hex(4)}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć katalogu preflight')


def check_environment():
    """Sprawdza środowisko Conda attention i wersje bibliotek."""
    executable = sys.executable.replace('\\', '/')
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    require(env_name == 'attention' or '/envs/attention/' in executable,
            'Uruchom preflight w środowisku Conda attention')
    import torch
    import torchvision

    return {
        'timestamp_utc': utc_now(),
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'conda_default_env': env_name,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_cuda_available': bool(torch.cuda.is_available()),
    }


def check_new_sources():
    """Sprawdza obecność plików S2 i brak importów attention_v2/v3."""
    files = (
        'OPIS_S2.md',
        'configs/stage2.json',
        's2_common.py',
        's2_worker.py',
        'launch_s2.py',
        'check_cpu_s2.py',
        'analyze_s2.py',
    )
    hashes = {}
    for name in files:
        path = HERE / name
        require(path.is_file(), f'Brak pliku S2: {path}')
        hashes[name] = common.digest(path)
        if path.suffix == '.py' and name != 'check_cpu_s2.py':
            text = path.read_text(encoding='utf-8')
            bad_v2 = 'attention' + '_v2'
            bad_v3 = 'attention' + '_v3'
            require(f'import {bad_v2}' not in text
                    and f'import {bad_v3}' not in text
                    and f'from {bad_v2}' not in text
                    and f'from {bad_v3}' not in text,
                    f'{name}: niedozwolony import attention_v2/v3')
    launcher = (HERE / 'launch_s2.py').read_text(encoding='utf-8')
    require('import os' in launcher,
            'launcher S2 musi jawnie importować os')
    require('expected_controls' not in launcher,
            'launcher S2 nie może powielać expected_controls z S1')
    require('allocations_frozen_utc' in launcher
            and 'workers_launched_utc' in launcher,
            'launcher musi zapisywać kolejność zamrożenia i startu')
    require(launcher.index('freeze_allocations')
            < launcher.index('run_wave'),
            'alokacje muszą być zamrożone przed run_wave')
    e015 = ROOT / 'experiments/2026-10-27_E015_cifar100_alokacja/launch_folding.py'
    require(e015.is_file(), 'Brak launchera E015 do audytu')
    e015_text = e015.read_text(encoding='utf-8')
    require('import os' in e015_text,
            'Poprawka E015 import os nie jest obecna')
    require('LAUNCHER_REVISION' in e015_text,
            'Brak LAUNCHER_REVISION w E015')
    return hashes


def check_data(cfg):
    """Sprawdza dane, splity i brak użycia testu/rezerwy."""
    reports = {}
    for dataset in cfg['datasets']:
        common.verify_dataset_inputs(dataset)
        bundle = common.load_dataset_bundle(dataset, verify_hashes=True)
        train = bundle['train_indices']
        dev = bundle['dev_indices']
        require(len(train) == int(dataset['train_count']),
                f"{dataset['name']}: zły train count")
        require(len(dev) == int(dataset['dev_count']),
                f"{dataset['name']}: zły dev count")
        require(len(set(train.tolist())) == len(train),
                f"{dataset['name']}: duplikaty train")
        require(len(set(dev.tolist())) == len(dev),
                f"{dataset['name']}: duplikaty dev")
        require(not set(train.tolist()) & set(dev.tolist()),
                f"{dataset['name']}: train/dev nachodzą")
        split = bundle['split']
        require(split.get('official_test_used') is False,
                f"{dataset['name']}: split używa testu")
        reserve_count = 0
        if dataset['name'] == 'cifar100':
            reserve = split['splits']['reserve']
            reserve_count = len(reserve)
            require(reserve_count == 5000,
                    'CIFAR-100 rezerwa musi mieć 5000')
            require(not set(train.tolist()) & set(reserve),
                    'CIFAR-100 train/rezerwa nachodzą')
            require(not set(dev.tolist()) & set(reserve),
                    'CIFAR-100 dev/rezerwa nachodzą')
        reports[dataset['name']] = {
            'train_count': int(len(train)),
            'dev_count': int(len(dev)),
            'reserve_count': int(reserve_count),
            'official_test_used': False,
            'reserve_used': False,
            'checkpoint_sha256': common.digest(Path(dataset['checkpoint'])),
            'split_sha256': common.digest(Path(dataset['split'])),
            'baseline_top1': float(dataset['baseline_top1']),
        }
        del bundle
    return reports


def check_predictors(cfg):
    """Sprawdza, że scores AUTO pochodzą tylko z primary predyktora S1."""
    predictors = s2.load_predictors(cfg)
    text = (HERE / 's2_common.py').read_text(encoding='utf-8')
    for forbidden in ('damage_post_bn_pp', 'damage_pre_bn_pp',
                      'post_bn_top1', 'dev_top1', 'baseline_top1'):
        require(forbidden not in text,
                f's2_common.py nie może używać {forbidden} do alokacji')
    preview = {}
    for dataset, item in predictors.items():
        require(len(item['z']) == 12, f'{dataset}: zła długość z')
        require(np.isfinite(item['z']).all(), f'{dataset}: niefinitywne z')
        require(abs(float(np.mean(item['z']))) <= 1e-12,
                f'{dataset}: z nie jest wycentrowane')
        require(abs(float(np.std(item['z'], ddof=0)) - 1.0) <= 1e-9,
                f'{dataset}: z nie ma jednostkowej wariancji')
        preview[dataset] = {
            'group_names': item['group_names'],
            'levels': item['levels'],
            'score': [float(value) for value in item['score']],
            'z': [float(value) for value in item['z']],
            'fixed_level_z': [float(value) for value in item['fixed_level_z']],
            'recon_matrix': item['matrix'].tolist(),
        }
    return predictors, preview


def check_allocations(cfg, predictors):
    """Wyznacza i weryfikuje cztery alokacje dla obu datasetów na CPU."""
    result = {}
    for dataset in cfg['datasets']:
        name = dataset['name']
        bundle = common.load_dataset_bundle(dataset, verify_hashes=True)
        model = bundle['source_cpu']
        variants = s2.resolve_dataset(model, cfg, predictors, name)
        require([item['name'] for item in variants]
                == ['uniform', 'tilt_p030', 'tilt_p015', 'auto50'],
                f'{name}: zła kolejność wariantów')
        uniform = variants[0]
        target = int(uniform['params'])
        records = {}
        for allocation in variants:
            require(allocation.get('dev_used_for_allocation') is False,
                    f"{name}/{allocation['name']}: dev w alokacji")
            require(int(allocation['params']) <= target,
                    f"{name}/{allocation['name']}: przekroczony target")
            require(int(allocation['delta_params']) <= 0,
                    f"{name}/{allocation['name']}: delta_params > 0")
            require(abs(float(allocation['delta_percent'])) <= 0.5,
                    f"{name}/{allocation['name']}: poza tolerancją")
            target_ratios = allocation.get(
                'target_ratios', allocation['ratios'])
            require(len(allocation['ratios']) == 12
                    and len(target_ratios) == 12,
                    f"{name}/{allocation['name']}: złe ratios")
            require(all(float(0.10 <= value <= 0.90)
                        for value in target_ratios),
                    f"{name}/{allocation['name']}: target ratio poza [0.1,0.9]")
            require(all(float(0.10 - 0.02 <= value <= 0.90 + 0.02)
                        for value in allocation['ratios']),
                    f"{name}/{allocation['name']}: realized ratio poza tolerancją")
            exact = s2.verify_allocation(model, allocation, cfg)
            require(exact == int(allocation['params']),
                    f"{name}/{allocation['name']}: model != spec")
            records[allocation['name']] = {
                'params': int(allocation['params']),
                'target_params': int(allocation['target_params']),
                'delta_params': int(allocation['delta_params']),
                'delta_percent': float(allocation['delta_percent']),
                'ratios': allocation['ratios'],
                'target_ratios': allocation.get('target_ratios'),
                'widths': allocation.get('widths'),
                'beta': allocation.get('beta'),
                'base': allocation.get('base'),
                'repair_steps': allocation.get('repair_steps'),
                'repair_applied': allocation.get('repair_applied'),
                'gap_before_repair_params': allocation.get(
                    'gap_before_repair_params'),
                'gap_before_repair_percent': allocation.get(
                    'gap_before_repair_percent'),
                'base_fallback_no_feasible': allocation.get(
                    'base_fallback_no_feasible'),
                'base_feasible_count': allocation.get('base_feasible_count'),
                'base_grid_count': allocation.get('base_grid_count'),
                'coarse_grid_trace_len': len(
                    allocation.get('coarse_grid_trace', [])),
                'fine_grid_trace_len': len(
                    allocation.get('fine_grid_trace', [])),
            }
        auto = variants[3]
        require(int(auto['params']) <= int(uniform['params']),
                f'{name}: AUTO przekracza UNIFORM')
        require(auto.get('dev_used_for_allocation') is False,
                f'{name}: AUTO używa dev')
        require(abs(float(auto.get('beta', -1.0)) - 0.30) <= 1e-12,
                f'{name}: AUTO musi mieć beta=0.30')
        require(int(auto.get('base_grid_count', 0)) == 96,
                f'{name}: AUTO ma złą siatkę bazy')
        require(len(auto.get('coarse_grid_trace', [])) == 96,
                f'{name}: AUTO ma niepełny coarse_grid_trace')
        require(len(auto.get('fine_grid_trace', [])) == 21,
                f'{name}: AUTO ma niepełny fine_grid_trace')
        require(auto.get('base') is not None,
                f'{name}: AUTO bez bazy')
        require(float(auto.get('gap_before_repair_percent', 999.0)) <= 0.5
                or auto.get('repair_applied') is True,
                f'{name}: AUTO luka >0.5% bez repair')
        require(auto.get('z') is not None
                and len(auto['z']) == 12,
                f'{name}: AUTO bez z')
        result[name] = {
            'baseline_params': int(sum(parameter.numel()
                                       for parameter in model.parameters())),
            'nominal_target_params': int(auto['nominal_target_params']),
            'uniform_reference_params': target,
            'variants': records,
            'auto_vs_uniform_params': int(auto['params']) - target,
            'auto_vs_uniform_percent': float(auto['delta_percent']),
        }
        del bundle
        print(
            f"E016_S2_PREFLIGHT_ALLOC dataset={name} "
            f"uniform={target} auto={records['auto50']['params']} "
            f"beta={records['auto50']['beta']}", flush=True)
    return result


def main():
    report_dir = make_report_dir()
    print(f'E016_S2_PREFLIGHT_DIR={report_dir}', flush=True)
    environment = check_environment()
    cfg = s2.load_config()
    sources = s2.verify_sources(cfg)
    new_sources = check_new_sources()
    data = check_data(cfg)
    predictors, predictor_preview = check_predictors(cfg)
    allocations = check_allocations(cfg, predictors)
    report = {
        'experiment': cfg['experiment'],
        'stage': cfg['stage'],
        'protocol': cfg['opis'],
        'created_utc': utc_now(),
        'environment': environment,
        'sources_reused': sources,
        'sources_new': new_sources,
        'config_sha256': common.digest(s2.CONFIG_PATH),
        's1': cfg['s1'],
        'data': data,
        'predictors': predictor_preview,
        'allocations': allocations,
        'guards': {
            'dev_used_for_allocation': False,
            'uses_test': False,
            'uses_reserve': False,
            'actual_model_parameter_count': True,
            'target_from_actual_uniform': True,
            'auto_not_above_uniform': all(
                item['auto_vs_uniform_params'] <= 0
                for item in allocations.values()),
            'launcher_imports_os': True,
            'no_expected_controls': True,
            'e015_launcher_fix_present': True,
            'no_attention_v2_v3_imports': True,
        },
    }
    common.write_json_atomic(report_dir / 'preflight.json', report)
    lines = [
        '# E016 S2 — CPU preflight',
        '',
        f'Katalog: `{report_dir.relative_to(ROOT)}`',
        f'Data UTC: {report["created_utc"]}',
        '',
        '## Środowisko',
        '',
        f'- Python: {environment["python"]}',
        f'- Torch: {environment["torch"]}',
        f'- GPU widoczne w preflight: {environment["torch_cuda_available"]}',
        '',
        '## Targety i alokacje',
        '',
        '| dataset | N | nominal | UNIFORM | AUTO | beta | AUTO - UNIFORM |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, item in allocations.items():
        auto = item['variants']['auto50']
        lines.append(
            f"| {name} | {item['baseline_params']} | "
            f"{item['nominal_target_params']} | "
            f"{item['uniform_reference_params']} | "
            f"{auto['params']} | {auto['beta']} | "
            f"{item['auto_vs_uniform_params']} |")
    lines += [
        '',
        '## Bramki',
        '',
        '- dev nieużywany do alokacji: **TAK**',
        '- test/rezerwa nietknięte: **TAK**',
        '- rzeczywisty model do liczenia parametrów: **TAK**',
        '- AUTO nie przekracza UNIFORM: **TAK**',
        '- launcher jawnie importuje `os`: **TAK**',
        '- brak `expected_controls` z S1: **TAK**',
        '- poprawka E015 (`import os`, `LAUNCHER_REVISION`): **TAK**',
        '- brak importów attention_v2/v3: **TAK**',
    ]
    (report_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('E016_S2_CPU_PREFLIGHT_OK', flush=True)


if __name__ == '__main__':
    main()
