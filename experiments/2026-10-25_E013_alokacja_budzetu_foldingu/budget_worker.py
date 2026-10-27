"""E013: worker jednej alokacji budżetu foldingu (jeden wariant = jeden GPU).

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Wejście: E011 seed42, E012 operator, spec alokacji i frozen config.json.
Wyjście: run/allocations/<name>/summary.json, logits.npz i diagnostics.json;
pomiar przed/po korekcie BN, bez gradientów i bez fine-tuningu.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
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
ROOT = HERE.parents[1]
E012_EVAL = ROOT / 'experiments/2026-10-24_E012_folding_referencja/evaluate.py'
E011_TRAIN = ROOT / 'experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/train.py'
EXPERIMENT = '2026-10-25_E013_alokacja_budzetu_foldingu'
OPIS_PATH = 'experiments/2026-10-25_E013_alokacja_budzetu_foldingu/OPIS.md'
REQUIRED_CONFIG = (
    'checkpoint', 'checkpoint_sha256', 'split', 'split_sha256', 'seed',
    'budgets', 'batch_size', 'bn_seed', 'bn_samples', 'num_workers',
    'baseline_top1', 'baseline_nll', 'e011_train_sha256',
)
REQUIRED_SPEC = ('name', 'budget', 'family')

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import folding_budget  # noqa: E402
import pruning  # noqa: E402


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


E012 = load_module(E012_EVAL, '_e013_e012_evaluate')


def utc_now():
    return __import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat()


def digest(path):
    return E012.digest(path)


def read_json(path):
    return E012.read_json(path)


def write_json_atomic(path, value):
    E012.write_json_atomic(path, value)


def load_config(path):
    path = Path(path)
    require(path.is_file(), f'Brak configu: {path}')
    cfg = read_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in cfg]
    require(not missing, f'Brak pól configu: {missing}')
    if 'status' in cfg:
        require(cfg['status'] == 'frozen', 'Config musi być frozen')
    for key in ('checkpoint', 'split'):
        value = Path(str(cfg[key])).expanduser().resolve()
        require(value.is_absolute(), f'{key} musi być absolutny')
        cfg[key] = str(value)
    for key in ('checkpoint_sha256', 'split_sha256', 'e011_train_sha256'):
        value = str(cfg[key]).strip().lower()
        require(len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                f'{key} musi być sha256')
        cfg[key] = value
    for key in ('seed', 'batch_size', 'bn_seed', 'bn_samples', 'num_workers'):
        cfg[key] = int(cfg[key])
    cfg['baseline_top1'] = float(cfg['baseline_top1'])
    cfg['baseline_nll'] = float(cfg['baseline_nll'])
    cfg['budgets'] = [float(value) for value in cfg['budgets']]
    require(cfg['batch_size'] > 0 and cfg['bn_samples'] > 0
            and cfg['num_workers'] >= 0, 'Złe parametry przebiegu')
    return cfg


def load_spec(path):
    spec = read_json(path)
    missing = [key for key in REQUIRED_SPEC if key not in spec]
    require(not missing, f'Brak pól spec: {missing}')
    require(str(spec['family']) in ('uniform', 'tilt', 'random', 'mag1', 'mag2'),
            f"Nieznana rodzina: {spec['family']}")
    spec['budget'] = float(spec['budget'])
    return spec


def parameter_digest(model):
    return E012.parameter_digest(model)


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
            folded, diagnostics = pruning.prune(model, ratio, p)
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
    chosen = max(below, key=lambda item: (item[0], -item[1])) if below else min(
        feasible, key=lambda item: abs(item[0] - target_params))
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
    solved = folding_budget.solve_base_for_budget(
        model,
        family,
        target_params,
        tilt=float(spec.get('tilt', 0.0)),
        sigma=float(spec.get('sigma', 0.0)),
        random_seed=int(spec.get('random_seed', 0)),
    )
    solved['kind'] = 'folding'
    solved['name'] = spec['name']
    return solved


def apply_allocation(source_cpu, spec, resolved):
    if resolved['kind'] == 'magnitude':
        folded, diagnostics = pruning.prune(
            source_cpu, float(resolved['ratio']), int(resolved['p']))
        return folded, diagnostics, 'magnitude'
    folded, diagnostics = folding_budget.fold_with_ratios(
        source_cpu, resolved['ratios'], seed=42, deterministic=False,
        preserve_fc_bias=True)
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
    output_dir = run_dir / 'allocations' / spec['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    started = time.monotonic()
    state = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'status': 'running',
        'spec': spec,
        'gpu': int(args.gpu),
        'started_utc': started_utc,
        'official_test_used': False,
    }
    try:
        require(digest(config_path) == read_json(
            run_dir / 'manifest.json')['config_sha256'],
            'Config biegu nie zgadza się z manifestem')
        require(digest(E011_TRAIN) == cfg['e011_train_sha256'],
                'Hash train.py E011 nie zgadza się z configiem')
        e011 = E012.load_e011_module()
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Worker wymaga dokładnie jednego widocznego GPU')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        require(visible is not None and visible.strip() == str(args.gpu),
                'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        e011.seed_everything(int(cfg['seed']))
        torch.cuda.reset_peak_memory_stats(device)

        checkpoint_path = Path(cfg['checkpoint'])
        split_path = Path(cfg['split'])
        require(digest(checkpoint_path) == cfg['checkpoint_sha256'],
                'Hash checkpointu nie zgadza się')
        require(digest(split_path) == cfg['split_sha256'],
                'Hash splitu nie zgadza się')
        images, labels, classes = e011.load_cifar10_train(e011.DATA_ROOT)
        split = read_json(split_path)
        e011.validate_split(split, labels, int(split.get('seed')))
        train_indices = np.asarray(split['train'], dtype=np.int64)
        dev_indices = np.asarray(split['dev'], dtype=np.int64)
        require(len(train_indices) == int(cfg['bn_samples']),
                'bn_samples musi obejmować pełny train')
        payload = torch.load(checkpoint_path, map_location='cpu',
                             weights_only=True)
        require(isinstance(payload, dict)
                and isinstance(payload.get('state_dict'), dict),
                'Checkpoint bez state_dict')
        source_cpu = e011.build_model()
        source_cpu.load_state_dict(payload['state_dict'], strict=True)
        source_cpu.eval()
        require(E012.count_parameters(source_cpu) == e011.EXPECTED_PARAMETERS,
                'Zła liczba parametrów checkpointu')
        baseline_params = E012.count_parameters(source_cpu)
        baseline_macs = E012.compute_macs(source_cpu)
        dev_loader = E012.make_eval_loader(
            e011, images, labels, dev_indices, cfg, int(cfg['seed']))

        baseline_model = copy.deepcopy(source_cpu).to(device)
        baseline_metrics, baseline_logits, baseline_labels, baseline_idx = (
            E012.evaluate_model(baseline_model, dev_loader, device, collect=True))
        del baseline_model
        require(abs(baseline_metrics['top1'] - float(cfg['baseline_top1'])) <= 1e-12,
                'Baseline top1 nie zgadza się')
        require(abs(baseline_metrics['nll'] - float(cfg['baseline_nll'])) <= 1e-5,
                'Baseline NLL nie zgadza się')

        target_params = int(round(
            (1.0 - float(spec['budget'])) * baseline_params))
        resolved = resolve_allocation(source_cpu, spec, target_params)
        actual_params = int(resolved['params'])
        require(actual_params <= target_params,
                'Alokacja przekracza budżet parametrów')
        require(abs(resolved['delta_percent']) <= 0.5,
                f"Alokacja odbiega od budżetu o {resolved['delta_percent']:.3f}%")
        folded_cpu, diagnostics, kind = apply_allocation(
            source_cpu, spec, resolved)
        folded_params = E012.count_parameters(folded_cpu)
        require(folded_params == actual_params,
                'Liczba parametrów po foldingu nie zgadza się z solverem')
        require(folding_budget.load_reference_folding().state_dict_all_finite(
            folded_cpu), 'Niefinitywne parametry po kompresji')
        folded_macs = E012.compute_macs(folded_cpu)

        model_gpu = copy.deepcopy(folded_cpu).to(device)
        pre_metrics, pre_logits, pre_labels, pre_idx = E012.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        pre_vs_base = logit_metrics(pre_logits, baseline_logits)

        e011.seed_everything(int(cfg['bn_seed']))
        bn_loader = E012.make_bn_loader(
            e011, images, labels, train_indices, cfg)
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
        write_json_atomic(diag_path, diagnostics)

        state.update({
            'status': 'completed',
            'allocation': resolved,
            'target_params': int(target_params),
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
            'logits_sha256': digest(npz_path),
            'finished_utc': utc_now(),
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
        })
        write_json_atomic(output_dir / 'summary.json', state)
        print(f"E013_ALLOC={spec['name']}_STATUS=completed", flush=True)
    except BaseException as error:
        state.update({
            'status': 'failed',
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'finished_utc': utc_now(),
            'runtime_seconds': time.monotonic() - started,
        })
        write_json_atomic(output_dir / 'summary.json', state)
        raise


if __name__ == '__main__':
    main()
