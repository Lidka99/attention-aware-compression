"""E016 S2: wspólne helpery automatycznej alokacji 50%.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu, etap S2; pełny protokół:
``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS_S2.md``.

Wejście: zamrożony ``configs/stage2.json``, analiza S1 (primary predictor
``weight_reconstruction_rel_error``), checkpointy E011/E015 seed42 oraz
operator E012/E014 importowany bez zmian. Wyjście: czyste funkcje do budowy
alokacji UNIFORM/TILT/AUTO, liczenia rzeczywistych parametrów zrealizowanego
modelu, zapisu specy i audytu. Moduł nie używa dev accuracy do alokacji.
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402

ROOT = common.ROOT
CONFIG_PATH = HERE / 'configs' / 'stage2.json'
LEVELS = (0.25, 0.50, 0.75, 0.90)
VARIANTS = ('uniform', 'tilt_p030', 'tilt_p015', 'auto50')
DATASETS = ('cifar10', 'cifar100')
SMOKE_TRAIN = 2048
SMOKE_DEV = 1000


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def load_config(path=CONFIG_PATH):
    """Wczytuje i waliduje zamrożony protokół S2."""
    cfg = common.read_json(path)
    require(cfg.get('status') == 'frozen', 'Config S2 musi być frozen')
    require(cfg.get('protocol_version') == '2.1',
            'Nieoczekiwana wersja protokołu S2')
    require(cfg.get('stage') == 'S2_global_allocation_50',
            'Zły etap S2')
    require(cfg.get('primary_predictor')
            == 'weight_reconstruction_rel_error',
            'Zły primary predictor')
    require(cfg['allocation']['uses_dev_accuracy'] is False,
            'AUTO nie może używać dev accuracy')
    require(cfg['allocation']['uses_dev_nll'] is False,
            'AUTO nie może używać dev NLL')
    require(cfg['allocation']['uses_test'] is False,
            'AUTO nie może używać testu')
    require(cfg['allocation']['uses_reserve'] is False,
            'AUTO nie może używać rezerwy')
    require(cfg['allocation']['selection_from_dev'] is False,
            'AUTO nie może selekcjonować na dev')
    require(cfg['allocation']['penalty_from_dev'] is False,
            'AUTO nie może brać kary z dev')
    require(cfg['allocation']['level_choice_from_dev'] is False,
            'AUTO nie może wybierać poziomu na dev')
    require(float(cfg['budget']['nominal_reduction']) == 0.5,
            'S2 dotyczy tylko budżetu 50%')
    require(float(cfg['budget']['max_over_target_params']) == 0.0,
            'AUTO nie może przekraczać targetu UNIFORM')
    require([item['name'] for item in cfg['datasets']] == list(DATASETS),
            'Kolejność datasetów musi być cifar10, cifar100')
    require([item['name'] for item in cfg['variants']] == list(VARIANTS),
            'Kolejność wariantów musi być uniform, tilt_p030, tilt_p015, auto50')
    require(abs(float(cfg['allocation']['beta_fixed']) - 0.30) <= 1e-12,
            'AUTO musi mieć zamrożone beta=0.30')
    grid = cfg['allocation']['base_grid']
    values = base_grid(grid)
    require(len(values) == int(grid['count']) == 96,
            'Siatka bazy musi mieć 96 punktów')
    fine = cfg['allocation']['local_refinement']
    require(float(fine['step']) == 0.001
            and float(fine['radius']) == 0.01
            and int(fine['count']) == 21,
            'Zła siatka lokalnego zawężenia')
    for dataset in cfg['datasets']:
        require(int(dataset['seed']) == 42, 'S2 używa tylko seeda 42')
        require(int(dataset['dev_count']) == 5000, 'Dev musi mieć 5000')
        require(int(dataset['bn_samples']) == int(dataset['train_count']),
                'BN musi obejmować pełny train')
    return cfg


def verify_sources(cfg):
    """Sprawdza hashe importowanych źródeł i artefaktów S1."""
    actual = {name: common.digest(path)
              for name, path in common.SOURCES.items()}
    expected = dict(cfg['source_sha256'])
    e016_key = expected.pop('E016_e016_common.py')
    require(actual == expected,
            f'Hash źródeł nie zgadza się z configiem: {actual} != {expected}')
    require(common.digest(common.__file__) == e016_key,
            'Hash e016_common.py nie zgadza się z configiem')
    s1 = cfg['s1']
    run_dir = ROOT / s1['run_dir']
    analysis_dir = ROOT / s1['analysis_dir']
    require(run_dir.is_dir(), f'Brak katalogu S1: {run_dir}')
    require(analysis_dir.is_dir(), f'Brak analizy S1: {analysis_dir}')
    checks = {
        run_dir / 'manifest.json': s1['run_manifest_sha256'],
        run_dir / 'artifacts.sha256': s1['run_artifacts_sha256'],
        analysis_dir / 'analysis.json': s1['analysis_json_sha256'],
        analysis_dir / 'group_level.csv': s1['group_level_csv_sha256'],
        analysis_dir / 'analysis.md': s1['analysis_md_sha256'],
    }
    for path, expected_hash in checks.items():
        require(path.is_file(), f'Brak artefaktu S1: {path}')
        require(common.digest(path) == expected_hash,
                f'Hash artefaktu S1 nie zgadza się: {path}')
    return actual


def base_grid(grid):
    """Zwraca deterministyczną siatkę bazy a z zamrożonego configu."""
    start = float(grid['start'])
    stop = float(grid['stop'])
    step = float(grid['step'])
    require(start == 0.0 and stop == 0.95 and step == 0.01,
            'Nieoczekiwana siatka bazy')
    count = int(round((stop - start) / step)) + 1
    values = [round(start + index * step, 10) for index in range(count)]
    require(values[0] == 0.0 and values[-1] == 0.95,
            'Zła siatka bazy')
    return values


def refined_bases(center, fine):
    """Zwraca lokalną siatkę wokół wybranej bazy."""
    step = float(fine['step'])
    radius = float(fine['radius'])
    count = int(fine['count'])
    start = max(0.0, float(center) - radius)
    values = [round(start + index * step, 10) for index in range(count)]
    values = [value for value in values if value <= 0.95 + 1e-12]
    require(values, 'Pusta lokalna siatka bazy')
    return values


def _load_analysis(cfg):
    s1 = cfg['s1']
    return common.read_json(ROOT / s1['analysis_dir'] / 'analysis.json')


def _load_group_csv(cfg):
    path = ROOT / cfg['s1']['analysis_dir'] / 'group_level.csv'
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    require(len(rows) == 96, f'Oczekiwano 96 wierszy S1, jest {len(rows)}')
    return rows


def load_predictors(cfg):
    """Buduje scores AUTO wyłącznie z primary predyktora S1.

    Zwraca dict dataset -> {
      group_names, matrix (12x4), level_mean, level_sd, level_z,
      score, z, fixed_level_z, fixed_level_rho_descriptive
    }.
    """
    analysis = _load_analysis(cfg)
    rows = _load_group_csv(cfg)
    names = common.group_names()
    result = {}
    for dataset in DATASETS:
        table = analysis['datasets'][dataset]['group_table']
        require(len(table) == 12, f'{dataset}: oczekiwano 12 grup')
        table = sorted(table, key=lambda item: int(item['group_index']))
        require([item['group_name'] for item in table] == names,
                f'{dataset}: zła kolejność grup')
        matrix = np.full((12, 4), np.nan, dtype=np.float64)
        for index, item in enumerate(table):
            require(int(item['group_index']) == index,
                    f'{dataset}: zły group_index')
            for level_index, level in enumerate(LEVELS):
                key = f'{level:.2f}'.rstrip('0').rstrip('.')
                value = item['levels'].get(key)
                require(value is not None,
                        f'{dataset}: brak poziomu {key}')
                matrix[index, level_index] = float(
                    value['weight_reconstruction_rel_error'])
        require(np.isfinite(matrix).all(),
                f'{dataset}: niefinitywne recon')
        require(((matrix >= 0.0) & (matrix <= 1.0)).all(),
                f'{dataset}: recon poza [0,1]')

        csv_matrix = np.full((12, 4), np.nan, dtype=np.float64)
        for row in rows:
            if row['dataset'] != dataset:
                continue
            group_index = int(row['group_index'])
            level = float(row['level'])
            level_index = min(
                range(4), key=lambda index: abs(level - LEVELS[index]))
            require(abs(level - LEVELS[level_index]) <= 1e-12,
                    f'{dataset}: nieznany poziom w CSV: {level}')
            csv_matrix[group_index, level_index] = float(
                row['weight_reconstruction_rel_error'])
        require(np.isfinite(csv_matrix).all(),
                f'{dataset}: CSV niekompletne')
        require(np.allclose(matrix, csv_matrix, atol=1e-9, rtol=0.0),
                f'{dataset}: analysis.json i CSV niezgodne')

        level_mean = matrix.mean(axis=0)
        level_sd = matrix.std(axis=0, ddof=0)
        require((level_sd > 0.0).all(),
                f'{dataset}: zerowe sd w poziomie')
        level_z = (matrix - level_mean) / level_sd
        score = level_z.mean(axis=1)
        score_mean = score.mean()
        score_sd = score.std(ddof=0)
        require(score_sd > 0.0, f'{dataset}: zerowe sd score')
        z = (score - score_mean) / score_sd

        fixed_index = LEVELS.index(0.50)
        fixed = matrix[:, fixed_index]
        fixed_z = (fixed - fixed.mean()) / fixed.std(ddof=0)
        require(np.isfinite(z).all() and np.isfinite(fixed_z).all(),
                f'{dataset}: niefinitywne z')
        result[dataset] = {
            'group_names': names,
            'levels': list(LEVELS),
            'matrix': matrix,
            'level_mean': level_mean,
            'level_sd': level_sd,
            'level_z': level_z,
            'score': score,
            'z': z,
            'fixed_level_z': fixed_z,
        }
    return result


def _parameter_count_exact(model, ratios):
    """Liczba parametrów zrealizowanego modelu po foldingu (bez sumowania)."""
    fb = common.folding_budget()
    values = [float(value) for value in ratios]
    require(len(values) == 12, 'Oczekiwano 12 ratios')
    count = fb.parameter_count_for_ratios(model, values)
    return int(count)


def _ratios_for_base(z, base, beta):
    return [float(min(max(float(base) - float(beta) * float(value), 0.10), 0.90))
            for value in z]


def _allocation_dict(name, family, ratios, params, target, extra):
    fb = common.folding_budget()
    names = fb.group_names()
    require(len(ratios) == len(names) == 12, 'Zła liczba grup')
    delta = int(params) - int(target)
    return {
        'name': str(name),
        'family': str(family),
        'group_order': list(names),
        'ratios': [float(value) for value in ratios],
        'params': int(params),
        'target_params': int(target),
        'delta_params': int(delta),
        'delta_percent': 100.0 * delta / float(target),
        'max_under_target_percent': 0.5,
        'dev_used_for_allocation': False,
        **extra,
    }


def allocation_uniform(model, cfg):
    """UNIFORM pod nominalny cel; zwraca alokację i faktyczny target."""
    fb = common.folding_budget()
    N = int(sum(parameter.numel() for parameter in model.parameters()))
    nominal = int(round((1.0 - float(cfg['budget']['nominal_reduction'])) * N))
    solved = fb.solve_base_for_budget(model, 'uniform', nominal)
    exact = _parameter_count_exact(model, solved['ratios'])
    require(exact == int(solved['params']),
            f'UNIFORM: solver {solved["params"]} != model {exact}')
    require(exact <= nominal, 'UNIFORM przekroczył nominalny cel')
    require(exact > 0, 'UNIFORM nie ma parametrów')
    allocation = _allocation_dict(
        'uniform', 'uniform', solved['ratios'], exact, exact, {
            'nominal_target_params': nominal,
            'baseline_params': N,
            'target_ratios': [float(value) for value in solved['ratios']],
            'base': float(solved['base']),
            'solver_params': int(solved['params']),
            'uniform_reference_params': exact,
            'repair_steps': 0,
            'widths': [int(value) for value in solved['widths']],
            'allocation_method': 'solve_base_for_budget(uniform, nominal)',
        })
    return allocation


def allocation_tilt(model, cfg, tilt, target):
    """TILT pod wspólny target UNIFORM; solver + repair + weryfikacja modelu."""
    fb = common.folding_budget()
    N = int(sum(parameter.numel() for parameter in model.parameters()))
    solved = fb.solve_base_for_budget(
        model, 'tilt', int(target), tilt=float(tilt))
    repaired = fb.repair_budget(
        model, solved['ratios'], int(target),
        min_ratio=float(cfg['budget']['min_group_reduction']),
        max_ratio=float(cfg['budget']['max_group_reduction']),
        max_steps=500)
    exact = _parameter_count_exact(model, repaired['ratios'])
    require(exact == int(repaired['params']),
            f'TILT: repair {repaired["params"]} != model {exact}')
    require(exact <= int(target), 'TILT przekroczył target UNIFORM')
    allocation = _allocation_dict(
        f'tilt_p{int(round(float(tilt) * 100)):03d}',
        'tilt', repaired['ratios'], exact, int(target), {
            'tilt': float(tilt),
            'baseline_params': N,
            'target_ratios': [float(value) for value in solved['ratios']],
            'nominal_target_params': int(round(
                (1.0 - float(cfg['budget']['nominal_reduction'])) * N)),
            'base': float(solved['base']),
            'solver_params': int(solved['params']),
            'solver_delta_percent': float(solved['delta_percent']),
            'repair_steps': int(repaired['repair_steps']),
            'widths': [int(value) for value in repaired['widths']],
            'uniform_reference_params': int(target),
            'allocation_method': 'solve_base_for_budget(tilt) + repair_budget',
        })
    return allocation


def allocation_auto(model, cfg, predictors, target):
    """AUTO50: primary predictor S1, stała beta=0.30, baza do targetu."""
    fb = common.folding_budget()
    N = int(sum(parameter.numel() for parameter in model.parameters()))
    z = [float(value) for value in predictors['z']]
    beta = float(cfg['allocation']['beta_fixed'])
    coarse = base_grid(cfg['allocation']['base_grid'])
    fine = cfg['allocation']['local_refinement']
    target = int(target)

    def evaluate(base):
        ratios = _ratios_for_base(z, base, beta)
        params = _parameter_count_exact(model, ratios)
        return {
            'base': float(base),
            'params': int(params),
            'feasible': bool(params <= target),
        }

    coarse_trace = [evaluate(base) for base in coarse]
    coarse_feasible = [item for item in coarse_trace if item['feasible']]
    fallback = not coarse_feasible
    if coarse_feasible:
        chosen = max(
            coarse_feasible,
            key=lambda item: (int(item['params']), -float(item['base'])))
        fine_values = refined_bases(chosen['base'], fine)
        fine_trace = [evaluate(base) for base in fine_values]
        fine_feasible = [item for item in fine_trace if item['feasible']]
        if fine_feasible:
            chosen = max(
                fine_feasible,
                key=lambda item: (int(item['params']), -float(item['base'])))
    else:
        chosen = evaluate(0.95)
        fine_trace = [chosen]
        fine_feasible = []

    ratios = _ratios_for_base(z, chosen['base'], beta)
    gap_before = target - int(chosen['params'])
    gap_percent = 100.0 * gap_before / target
    repair_threshold = float(cfg['budget']['max_under_target_percent'])
    repair_applied = bool(gap_percent > repair_threshold)
    if repair_applied:
        repaired = fb.repair_budget(
            model, ratios, target,
            min_ratio=float(cfg['budget']['min_group_reduction']),
            max_ratio=float(cfg['budget']['max_group_reduction']),
            max_steps=500)
        final_ratios = [float(value) for value in repaired['ratios']]
        repair_steps = int(repaired['repair_steps'])
        repair_params = int(repaired['params'])
        widths = [int(value) for value in repaired['widths']]
    else:
        final_ratios = [float(value) for value in ratios]
        repair_steps = 0
        repair_params = int(chosen['params'])
        widths = None
    exact = _parameter_count_exact(model, final_ratios)
    if repair_applied:
        require(exact == repair_params,
                f'AUTO: repair {repair_params} != model {exact}')
    else:
        require(exact == int(chosen['params']),
                f'AUTO: base {chosen["params"]} != model {exact}')
    require(exact <= target, 'AUTO przekroczył target UNIFORM')
    require(100.0 * (target - exact) / target <= repair_threshold,
            'AUTO poza tolerancją 0.5% pod targetem')
    allocation = _allocation_dict(
        'auto50', 'auto', final_ratios, exact, target, {
            'baseline_params': N,
            'target_ratios': [float(value) for value in ratios],
            'nominal_target_params': int(round(
                (1.0 - float(cfg['budget']['nominal_reduction'])) * N)),
            'beta': beta,
            'base': float(chosen['base']),
            'beta_params_before_repair': int(chosen['params']),
            'gap_before_repair_params': int(gap_before),
            'gap_before_repair_percent': float(gap_percent),
            'repair_threshold_percent': float(repair_threshold),
            'repair_applied': bool(repair_applied),
            'repair_steps': int(repair_steps),
            'base_fallback_no_feasible': bool(fallback),
            'base_grid_start': 0.0,
            'base_grid_stop': 0.95,
            'base_grid_step': 0.01,
            'base_grid_count': len(coarse),
            'base_feasible_count': len(coarse_feasible),
            'fine_grid_count': len(fine_trace),
            'fine_feasible_count': len(fine_feasible),
            'widths': widths,
            'uniform_reference_params': target,
            'score': [float(value) for value in predictors['score']],
            'z': z,
            'allocation_method': (
                'S1 primary predictor: per-level z, mean, group z; '
                'r=clip(a-0.30*z,0.10,0.90); exact base grid; local '
                'refinement; repair only if gap>0.5%'),
            'coarse_grid_trace': coarse_trace,
            'fine_grid_trace': fine_trace,
        })
    return allocation

def resolve_dataset(model, cfg, predictors, dataset_name):
    """Zwraca cztery zamrożone alokacje dla jednego datasetu."""
    require(dataset_name in DATASETS, f'Nieznany dataset: {dataset_name}')
    uniform = allocation_uniform(model, cfg)
    target = int(uniform['params'])
    tilt030 = allocation_tilt(model, cfg, 0.30, target)
    tilt015 = allocation_tilt(model, cfg, 0.15, target)
    auto = allocation_auto(model, cfg, predictors[dataset_name], target)
    variants = [uniform, tilt030, tilt015, auto]
    require([item['name'] for item in variants] == list(VARIANTS),
            'Zła kolejność wariantów')
    for item in variants:
        require(int(item['params']) <= target,
                f"{item['name']}: przekroczony target")
        require(int(item['delta_params']) <= 0,
                f"{item['name']}: dodatni delta_params")
        require(abs(float(item['delta_percent'])) <= 0.5,
                f"{item['name']}: poza tolerancją 0.5%")
    return variants


def verify_allocation(model, allocation, cfg):
    """Weryfikuje spec alokacji przez rzeczywisty model i target."""
    require(allocation.get('dev_used_for_allocation') is False,
            'Alokacja nie może używać dev')
    ratios = [float(value) for value in allocation['ratios']]
    target_ratios = [float(value) for value in allocation.get(
        'target_ratios', ratios)]
    require(len(ratios) == 12 and len(target_ratios) == 12,
            'Zła długość ratios')
    require(all(math.isfinite(value) and 0.10 <= value <= 0.90
                for value in target_ratios),
            'Target ratio poza [0.10,0.90]')
    require(all(math.isfinite(value) and 0.10 - 0.02 <= value <= 0.90 + 0.02
                for value in ratios),
            'Realized ratio poza [0.10,0.90] + dyskretyzacja 0.02')
    exact = _parameter_count_exact(model, ratios)
    require(exact == int(allocation['params']),
            f"Model {exact} != spec {allocation['params']}")
    require(exact <= int(allocation['target_params']),
            'Alokacja przekracza target UNIFORM')
    require(abs(100.0 * (exact - int(allocation['target_params']))
                / int(allocation['target_params'])) <= 0.5,
            'Alokacja poza tolerancją 0.5%')
    return exact


def spec_payload(allocation, dataset_name, seed, cfg):
    """Buduje pełny, samowystarczalny plik spec dla workera."""
    return {
        'experiment': cfg['experiment'],
        'stage': cfg['stage'],
        'opis': cfg['opis'],
        'dataset': dataset_name,
        'seed': int(seed),
        'variant': allocation['name'],
        'family': allocation['family'],
        'allocation': allocation,
        'dev_used_for_allocation': False,
        'uses_test': False,
        'uses_reserve': False,
        'fold_seed': int(cfg['run']['fold_seed']),
        'bn_seed': int(cfg['run']['bn_seed']),
        'batch_size': int(cfg['run']['batch_size']),
        'num_workers': int(cfg['run']['num_workers']),
    }


def file_digest(path):
    """Skrót SHA-256 pliku."""
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()
