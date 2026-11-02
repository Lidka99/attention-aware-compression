"""E019: worker zamrożonego porównania alokacji na Tiny ImageNet (bez fine-tuningu).

Eksperyment: 2026-10-29_E019_tinyimagenet_alokacja; pełny opis: OPIS.md.
Wejście: checkpoint E019 dla jednego seeda, split train/dev, frozen config
i spec alokacji (budżet + rodzina). Wyjście w
``<fold-run>/allocations/seed_<seed>/<spec>/``: ``summary.json``,
``logits.npz`` i ``diagnostics.json``. Pomiar przed/po korekcie BN na dev;
BN na pełnym train 40 000 bez gradientów; oficjalny test i rezerwa nietknięte.

Operator foldingu i alokacji jest importowany bez zmian z E012/E014:
``folding.py``, ``folding_budget.py`` i ``pruning.py``. Moduł nie modyfikuje
wcześniejszych eksperymentów.

Uczciwe porównanie budżetów: dla każdego nominalnego budżetu najpierw
rozwiązywany jest UNIFORM, a jego faktyczna liczba parametrów staje się
wspólnym targetem TILT+0.30, TILT+0.15 i MAG2. Summary zapisuje cel nominalny,
target UNIFORM, parametry faktyczne oraz redukcję nominalną i faktyczną.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_tiny as data  # noqa: E402

ROOT = data.ROOT
E012_EVALUATE = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/evaluate.py')
E014_DIR = ROOT / 'experiments' / '2026-10-27_E014_replikacja_alokacji'
E014_FOLDING_BUDGET = E014_DIR / 'folding_budget.py'
E014_PRUNING = E014_DIR / 'pruning.py'
TRAIN_PATH = HERE / 'train_tiny.py'
DATA_PATH = HERE / 'data_tiny.py'
REQUIRED_CONFIG = (
    'checkpoint', 'checkpoint_sha256', 'split', 'split_sha256', 'seed',
    'budgets', 'batch_size', 'bn_seed', 'bn_samples', 'num_workers',
    'fold_seed', 'baseline_top1', 'baseline_nll', 'e019_train_sha256',
    'e019_data_sha256',
)
REQUIRED_SPEC = ('name', 'budget', 'family')
BASELINE_NLL_TOLERANCE = 1e-5
BUDGET_TOLERANCE_PERCENT = 1.0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f'Nie można wczytać modułu {name} z {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E012 = load_module(E012_EVALUATE, '_e015_folding_e012_evaluate')
FOLDING_BUDGET = load_module(
    E014_FOLDING_BUDGET, '_e015_e014_folding_budget')
PRUNING = load_module(E014_PRUNING, '_e015_e014_pruning')


def load_config(path):
    path = Path(path)
    require(path.is_file(), f'Brak configu: {path}')
    cfg = data.read_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in cfg]
    require(not missing, f'Brak pól configu: {missing}')
    require(cfg.get('status') == 'frozen', 'Config musi być frozen')
    for key in ('checkpoint', 'split'):
        value = Path(str(cfg[key])).expanduser().resolve()
        require(value.is_absolute(), f'{key} musi być absolutny')
        cfg[key] = str(value)
    for key in ('checkpoint_sha256', 'split_sha256', 'e019_train_sha256',
                'e019_data_sha256'):
        value = str(cfg[key]).strip().lower()
        require(len(value) == 64
                and all(character in '0123456789abcdef' for character in value),
                f'{key} musi być sha256')
        cfg[key] = value
    for key in ('seed', 'batch_size', 'bn_seed', 'bn_samples', 'num_workers',
                'fold_seed'):
        cfg[key] = int(cfg[key])
    cfg['baseline_top1'] = float(cfg['baseline_top1'])
    cfg['baseline_nll'] = float(cfg['baseline_nll'])
    cfg['budgets'] = [float(value) for value in cfg['budgets']]
    require(cfg['batch_size'] > 0 and cfg['bn_samples'] > 0
            and cfg['num_workers'] >= 0, 'Złe parametry przebiegu')
    require(cfg['official_test_used'] is False, 'Config używa oficjalnego testu')
    require(cfg['reserve_used'] is False, 'Config używa rezerwy')
    return cfg


def load_spec(path):
    spec = data.read_json(path)
    missing = [key for key in REQUIRED_SPEC if key not in spec]
    require(not missing, f'Brak pól spec: {missing}')
    require(str(spec['family']) in ('uniform', 'tilt', 'random', 'mag1', 'mag2'),
            f"Nieznana rodzina: {spec['family']}")
    spec['budget'] = float(spec['budget'])
    return spec


def logit_metrics(candidate, baseline):
    candidate = torch.as_tensor(candidate, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    require(candidate.shape == baseline.shape, 'Niezgodne kształty logitów')
    diff = candidate - baseline
    logp_candidate = F.log_softmax(candidate, dim=1)
    logp_baseline = F.log_softmax(baseline, dim=1)
    return {
        'prediction_agreement': float(
            (candidate.argmax(1) == baseline.argmax(1)).float().mean()),
        'logit_mae': float(diff.abs().mean()),
        'logit_rmse': float(diff.pow(2).mean().sqrt()),
        'logit_max_abs': float(diff.abs().max()),
        'kl_baseline_candidate': float(F.kl_div(
            logp_candidate, logp_baseline.exp(), reduction='batchmean')),
        'kl_candidate_baseline': float(F.kl_div(
            logp_baseline, logp_candidate.exp(), reduction='batchmean')),
    }


def solve_mag_ratio(model, p, target_params):
    cache = {}

    def evaluate(ratio):
        ratio = float(min(max(ratio, 0.0), 0.95))
        key = round(ratio, 10)
        if key not in cache:
            folded, diagnostics = PRUNING.prune(model, ratio, p)
            cache[key] = int(diagnostics['nparams_after'])
        return cache[key]

    low, high = 0.0, 0.95
    for _ in range(28):
        middle = 0.5 * (low + high)
        if evaluate(middle) <= target_params:
            high = middle
        else:
            low = middle
    candidates = [0.0, 0.95, low, high]
    for ratio in (low, high):
        for delta in (1e-4, 2e-4, 5e-4):
            candidates.append(min(max(ratio + delta, 0.0), 0.95))
    feasible = [(evaluate(value), value) for value in candidates]
    below = [item for item in feasible if item[0] <= target_params]
    if below:
        chosen = max(below, key=lambda item: (item[0], -item[1]))
    else:
        chosen = min(feasible, key=lambda item: abs(item[0] - target_params))
    return chosen[1], chosen[0]


def resolve_allocation(model, spec, target_params):
    family = str(spec['family'])
    if family in ('mag1', 'mag2'):
        p = 1 if family == 'mag1' else 2
        ratio, params = solve_mag_ratio(model, p, target_params)
        return {
            'kind': 'magnitude',
            'family': family,
            'p': int(p),
            'ratio': float(ratio),
            'params': int(params),
            'target_params': int(target_params),
            'delta_params': int(params - target_params),
            'delta_percent': 100.0 * (params - target_params) / target_params,
        }
    solved = FOLDING_BUDGET.solve_base_for_budget(
        model,
        family,
        target_params,
        tilt=float(spec.get('tilt', 0.0)),
        sigma=float(spec.get('sigma', 0.0)),
        random_seed=int(spec.get('random_seed', 0)),
    )
    if family != 'uniform':
        repaired = FOLDING_BUDGET.repair_budget(
            model, solved['ratios'], target_params)
        repaired['solver_params'] = int(solved['params'])
        repaired['solver_delta_percent'] = float(solved['delta_percent'])
        repaired['base'] = float(solved['base'])
        repaired['family'] = family
        repaired['kind'] = 'folding'
        repaired['name'] = spec['name']
        return repaired
    solved['kind'] = 'folding'
    solved['name'] = spec['name']
    solved['repair_steps'] = 0
    return solved


def solve_uniform_reference(model, nominal_target_params):
    """Liczba parametrów UNIFORM dla nominalnego budżetu jest wspólnym celem.

    UNIFORM jest rozwiązywany jako pierwszy i to jego faktyczna liczba
    parametrów (nie nominalny cel) staje się targetem dla TILT i MAG2.
    Dzięki temu porównanie nie przyznaje TILT ani MAG2 dodatkowych parametrów.
    """
    nominal = int(nominal_target_params)
    solved = FOLDING_BUDGET.solve_base_for_budget(model, 'uniform', nominal)
    params = int(solved['params'])
    require(params <= nominal, 'UNIFORM przekroczył nominalny cel budżetu')
    return solved, params


def resolve_for_spec(model, spec, nominal_target_params):
    """Zwraca (alokacja, wspólny target) dla jednego specu i budżetu.

    ``nominal_target_params`` to ``round((1-budget) * baseline_params)``.
    Wszystkie warianty danego budżetu dostają target równy faktycznej liczbie
    parametrów UNIFORM. Solver UNIFORM nadal capuje się nominalnym celem.
    """
    nominal = int(nominal_target_params)
    uniform, reference = solve_uniform_reference(model, nominal)
    if str(spec['family']) == 'uniform':
        resolved = dict(uniform)
        resolved.update({
            'kind': 'folding',
            'family': 'uniform',
            'name': spec['name'],
            'repair_steps': 0,
            'solver_delta_percent': float(uniform['delta_percent']),
        })
    else:
        resolved = resolve_allocation(model, spec, reference)
    params = int(resolved['params'])
    resolved['nominal_target_params'] = nominal
    resolved['uniform_reference_params'] = int(reference)
    resolved['target_params'] = int(reference)
    resolved['delta_params'] = int(params - reference)
    resolved['delta_percent'] = 100.0 * (params - reference) / reference
    resolved['nominal_delta_percent'] = 100.0 * (params - nominal) / nominal
    return resolved, int(reference)


def apply_allocation(source_cpu, spec, resolved, cfg):
    if resolved['kind'] == 'magnitude':
        folded, diagnostics = PRUNING.prune(
            source_cpu, float(resolved['ratio']), int(resolved['p']))
        return folded, diagnostics, 'magnitude'
    folded, diagnostics = FOLDING_BUDGET.fold_with_ratios(
        source_cpu, resolved['ratios'], seed=int(cfg['fold_seed']),
        deterministic=False, preserve_fc_bias=True)
    return folded, diagnostics, 'folding'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()
    cfg = load_config(config_path)
    spec = load_spec(spec_path)
    require(spec['name'].replace('/', '_') not in ('', '.', '..'),
            'Zła nazwa spec')
    output_dir = run_dir / 'allocations' / f"seed_{cfg['seed']}" / spec['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = data.utc_now()
    started = time.monotonic()
    state = {
        'experiment': data.EXPERIMENT,
        'opis': data.OPIS_PATH,
        'status': 'running',
        'spec': spec,
        'seed': int(cfg['seed']),
        'gpu': int(args.gpu),
        'batch_size_per_gpu': int(cfg['batch_size']),
        'gradient_accumulation': 1,
        'effective_global_batch_per_run': int(cfg['batch_size']),
        'aggregation': 'none (independent allocations)',
        'started_utc': started_utc,
        'official_test_used': False,
        'reserve_used': False,
    }
    try:
        manifest = data.read_json(run_dir / 'manifest.json')
        require(manifest.get('status') == 'running',
                'Manifest foldingu nie jest aktywny')
        require(manifest.get('config_sha256', {}).get(str(cfg['seed']))
                == data.digest(config_path),
                'Config biegu nie zgadza się z manifestem')
        require(data.digest(TRAIN_PATH) == cfg['e019_train_sha256'],
                'Hash train_tiny.py nie zgadza się z configiem')
        require(data.digest(DATA_PATH) == cfg['e019_data_sha256'],
                'Hash data_tiny.py nie zgadza się z configiem')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Worker wymaga dokładnie jednego widocznego GPU')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        require(visible is not None and visible.strip() == str(args.gpu),
                'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        data.seed_everything(int(cfg['seed']))
        torch.cuda.reset_peak_memory_stats(device)

        checkpoint_path = Path(cfg['checkpoint'])
        split_path = Path(cfg['split'])
        require(data.digest(checkpoint_path) == cfg['checkpoint_sha256'],
                'Hash checkpointu nie zgadza się')
        require(data.digest(split_path) == cfg['split_sha256'],
                'Hash splitu nie zgadza się')
        images, labels, classes = data.load_tiny_train(data.DATA_ROOT)
        split = data.load_split(split_path)
        data.validate_split(split, labels, data.SPLIT_SEED)
        train_indices = np.asarray(split['splits']['train'], dtype=np.int64)
        dev_indices = np.asarray(split['splits']['dev'], dtype=np.int64)
        require(len(train_indices) == int(cfg['bn_samples']),
                'bn_samples musi obejmować pełny train')
        require(len(dev_indices) == data.EXPECTED_DEV,
                'Dev nie ma 5 000 próbek')
        require(len(classes) == data.NUM_CLASSES, 'Zła liczba klas')

        payload = torch.load(checkpoint_path, map_location='cpu',
                             weights_only=True)
        require(isinstance(payload, dict)
                and isinstance(payload.get('state_dict'), dict),
                'Checkpoint bez state_dict')
        source_cpu = data.build_model()
        source_cpu.load_state_dict(payload['state_dict'], strict=True)
        source_cpu.eval()
        require(data.count_parameters(source_cpu) == data.EXPECTED_PARAMETERS,
                'Zła liczba parametrów checkpointu')
        baseline_params = E012.count_parameters(source_cpu)
        baseline_macs = E012.compute_macs(source_cpu)
        dev_loader = E012.make_eval_loader(
            data, images, labels, dev_indices, cfg, int(cfg['seed']))

        baseline_model = copy.deepcopy(source_cpu).to(device)
        baseline_metrics, baseline_logits, baseline_labels, baseline_idx = (
            E012.evaluate_model(
                baseline_model, dev_loader, device, collect=True))
        del baseline_model
        require(abs(baseline_metrics['top1'] - float(cfg['baseline_top1']))
                <= 1e-12, 'Baseline top1 nie zgadza się')
        require(abs(baseline_metrics['nll'] - float(cfg['baseline_nll']))
                <= BASELINE_NLL_TOLERANCE, 'Baseline NLL nie zgadza się')

        nominal_target_params = int(round(
            (1.0 - float(spec['budget'])) * baseline_params))
        resolved, reference_params = resolve_for_spec(
            source_cpu, spec, nominal_target_params)
        target_params = int(resolved['target_params'])
        require(target_params == reference_params,
                'Wspólny target nie zgadza się z liczbą parametrów UNIFORM')
        require(target_params <= nominal_target_params,
                'Target UNIFORM przekracza nominalny cel budżetu')
        actual_params = int(resolved['params'])
        require(actual_params <= target_params,
                'Alokacja przekracza wspólny target parametrów')
        require(actual_params <= nominal_target_params,
                'Alokacja przekracza nominalny budżet parametrów')
        require(-BUDGET_TOLERANCE_PERCENT <= resolved['delta_percent'] <= 0.0,
                f"Alokacja odbiega od wspólnego targetu o "
                f"{resolved['delta_percent']:.3f}% (dozwolone "
                f"{BUDGET_TOLERANCE_PERCENT:.1f}% poniżej, nigdy powyżej)")
        folded_cpu, diagnostics, kind = apply_allocation(
            source_cpu, spec, resolved, cfg)
        folded_params = E012.count_parameters(folded_cpu)
        require(folded_params == actual_params,
                'Liczba parametrów po kompresji nie zgadza się z solverem')
        require(FOLDING_BUDGET.load_reference_folding().state_dict_all_finite(
            folded_cpu), 'Niefinitywne parametry po kompresji')
        folded_macs = E012.compute_macs(folded_cpu)

        model_gpu = copy.deepcopy(folded_cpu).to(device)
        pre_metrics, pre_logits, pre_labels, pre_idx = E012.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        pre_vs_base = logit_metrics(pre_logits, baseline_logits)

        data.seed_everything(int(cfg['bn_seed']))
        bn_loader = E012.make_bn_loader(
            data, images, labels, train_indices, cfg)
        bn_stats = E012.adapt_bn(
            model_gpu, bn_loader, device, len(train_indices),
            int(cfg['bn_seed']))
        del bn_loader
        post_metrics, post_logits, post_labels, post_idx = E012.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        post_vs_base = logit_metrics(post_logits, baseline_logits)
        require(np.array_equal(pre_idx, baseline_idx)
                and np.array_equal(post_idx, baseline_idx),
                'Niezgodna kolejność indeksów dev')
        require(np.array_equal(pre_labels, baseline_labels)
                and np.array_equal(post_labels, baseline_labels),
                'Niezgodne etykiety dev')

        npz_path = output_dir / 'logits.npz'
        np.savez_compressed(
            npz_path, baseline=baseline_logits, pre_bn=pre_logits,
            post_bn=post_logits, labels=baseline_labels, indices=baseline_idx)
        diag_path = output_dir / 'diagnostics.json'
        data.write_json_atomic(diag_path, diagnostics)

        state.update({
            'status': 'completed',
            'kind': kind,
            'allocation': resolved,
            'nominal_target_params': int(nominal_target_params),
            'uniform_reference_params': int(reference_params),
            'target_params': int(target_params),
            'nominal_target_reduction_fraction': (
                (baseline_params - nominal_target_params) / baseline_params),
            'target_reduction_fraction': (
                (baseline_params - target_params) / baseline_params),
            'budget_tolerance_percent': BUDGET_TOLERANCE_PERCENT,
            'baseline': {
                'params': int(baseline_params),
                'macs': baseline_macs,
                'metrics': baseline_metrics,
            },
            'compressed': {
                'params': int(folded_params),
                'params_before': int(baseline_params),
                'param_reduction': int(baseline_params - folded_params),
                'param_reduction_fraction': (
                    (baseline_params - folded_params) / baseline_params),
                'macs': folded_macs,
                'pre_bn': pre_metrics,
                'post_bn': post_metrics,
                'pre_bn_vs_baseline': pre_vs_base,
                'post_bn_vs_baseline': post_vs_base,
                'bn': bn_stats,
            },
            'diagnostics_file': diag_path.name,
            'logits_file': npz_path.name,
            'logits_sha256': data.digest(npz_path),
            'finished_utc': data.utc_now(),
            'runtime_seconds': time.monotonic() - started,
            'peak_gpu_allocated_mb': (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            'peak_gpu_reserved_mb': (
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
            'environment': {
                'python': platform.python_version(),
                'torch': torch.__version__,
                'torchvision': __import__('torchvision').__version__,
                'numpy': np.__version__,
                'cuda': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'cudnn': torch.backends.cudnn.version(),
                'cublas_workspace_config': os.environ.get(
                    'CUBLAS_WORKSPACE_CONFIG'),
                'deterministic_algorithms': bool(
                    torch.are_deterministic_algorithms_enabled()),
            },
            'classes': classes,
            'reused_sources': {
                'e012_evaluate': str(E012_EVALUATE.relative_to(ROOT)),
                'e014_folding_budget': str(E014_FOLDING_BUDGET.relative_to(ROOT)),
                'e014_pruning': str(E014_PRUNING.relative_to(ROOT)),
            },
        })
        data.write_json_atomic(output_dir / 'summary.json', state)
        print(f"E019_ALLOC=seed{cfg['seed']}/{spec['name']}_STATUS=completed",
              flush=True)
    except BaseException as error:
        state.update({
            'status': 'failed',
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'finished_utc': data.utc_now(),
            'runtime_seconds': time.monotonic() - started,
        })
        data.write_json_atomic(output_dir / 'summary.json', state)
        raise


if __name__ == '__main__':
    main()
