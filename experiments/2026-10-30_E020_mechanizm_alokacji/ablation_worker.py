"""E020: worker ablacji ANTI-TILT (odwrócony kierunek alokacji).

Eksperyment: 2026-10-30_E020_mechanizm_alokacji; pełny opis: OPIS.md.
Wejście: checkpoint E011/E015/E019, split, budżet i seed; wspólny target
parametrów = faktyczna liczba parametrów UNIFORM z istniejącego runu.
Wyjście w ``<run>/allocations/<dataset>/seed_<seed>/b<budget>_antitilt_p030/``:
``summary.json``, ``diagnostics.json`` i ``logits.npz``. Operator, BN i
fold_seed są identyczne jak w E014/E015/E019; bez treningu wag.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e020_common as common  # noqa: E402


def reference_summary_path(root, dataset, seed, budget):
    """Zwraca ścieżkę do istniejącego summary UNIFORM."""
    cfg = common.load_ablation_config()
    run = root / cfg['reference_fold_runs'][dataset]
    tag = f"b{int(round(float(budget) * 100))}_uniform"
    if dataset == 'cifar10':
        return run / 'seeds' / f'seed_{seed}' / 'allocations' / tag / 'summary.json'
    return run / 'allocations' / f'seed_{seed}' / tag / 'summary.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    cfg = common.read_json(args.config)
    common.require(cfg.get('status') == 'frozen', 'Config musi być frozen')
    items = common.read_json(args.items)
    common.require(isinstance(items, list) and items, 'Pusta lista itemów')
    common.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                   'Worker wymaga dokładnie jednego widocznego GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    common.require(visible is not None and visible.strip() == str(args.gpu),
                   'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    root = common.ROOT
    inventory = common.load_inventory()
    budget = common.folding_budget()
    e012 = common.e012_evaluate()

    data_cache = {}
    failures = []
    for item in items:
        dataset = str(item['dataset'])
        seed = int(item['seed'])
        budget_value = float(item['budget'])
        name = str(item['name'])
        output_dir = (
            run_dir / 'allocations' / dataset / f'seed_{seed}' / name)
        output_dir.mkdir(parents=True, exist_ok=True)
        started_utc = common.utc_now()
        started = time.monotonic()
        state = {
            'experiment': common.EXPERIMENT,
            'opis': common.OPIS_PATH,
            'status': 'running',
            'dataset': dataset,
            'seed': seed,
            'budget': budget_value,
            'variant': cfg['variant'],
            'gpu': int(args.gpu),
            'batch_size_per_gpu': int(cfg['batch_size']),
            'effective_global_batch_per_run': int(cfg['batch_size']),
            'official_test_used': False,
            'reserve_used': False,
            'started_utc': started_utc,
        }
        try:
            if dataset not in data_cache:
                data_cache[dataset] = common.load_dataset(dataset)
            module, images, labels, classes, split = data_cache[dataset]
            source_cpu, entry = common.load_checkpoint(dataset, seed, inventory)
            baseline_params = common.count_parameters(source_cpu)
            common.require(baseline_params == int(
                inventory['datasets'][dataset]['params']),
                'Niezgodna liczba parametrów baseline')

            dev_loader, _, train_indices = common.make_loaders(
                module, images, labels, split, cfg['batch_size'],
                cfg['num_workers'], seed, cfg['bn_seed'])
            common.require(len(train_indices) == int(
                inventory['datasets'][dataset]['train_count']),
                'BN musi obejmować pełny train')

            baseline_model = copy.deepcopy(source_cpu).to(device)
            baseline_metrics, baseline_logits, baseline_labels, baseline_idx = (
                common.evaluate(baseline_model, dev_loader, device))
            del baseline_model
            common.require(
                abs(baseline_metrics['top1'] - float(entry['baseline_top1']))
                <= 1e-12,
                'Baseline top1 na dev nie zgadza się z inwentarzem')
            common.require(
                abs(baseline_metrics['nll'] - float(entry['baseline_nll']))
                <= 1e-5,
                'Baseline NLL na dev nie zgadza się z inwentarzem')

            reference_path = reference_summary_path(
                root, dataset, seed, budget_value)
            common.require(reference_path.is_file(),
                           f'Brak referencji UNIFORM: {reference_path}')
            reference = common.read_json(reference_path)
            reference_params = int(reference['allocation']['params'])
            common.require(reference['status'] == 'completed',
                           'Referencja UNIFORM nie jest completed')
            if 'seed' in reference:
                common.require(int(reference['seed']) == seed,
                               'Referencja UNIFORM ma inny seed')
            common.require(
                abs(float(reference['spec']['budget']) - budget_value) <= 1e-12,
                'Referencja UNIFORM ma inny budżet')

            nominal_target = int(round((1.0 - budget_value) * baseline_params))
            uniform = budget.solve_base_for_budget(
                source_cpu, 'uniform', nominal_target)
            common.require(int(uniform['params']) == reference_params,
                           'Solver UNIFORM nie odtwarza referencji '
                           f"({uniform['params']} vs {reference_params})")

            solved = budget.solve_base_for_budget(
                source_cpu, 'tilt', reference_params,
                tilt=float(cfg['variant']['tilt']))
            repaired = budget.repair_budget(
                source_cpu, solved['ratios'], reference_params,
                min_ratio=float(cfg['min_ratio']),
                max_ratio=float(cfg['max_ratio']))
            repaired['solver_params'] = int(solved['params'])
            repaired['solver_delta_percent'] = float(solved['delta_percent'])
            repaired['base'] = float(solved['base'])
            repaired['family'] = 'tilt'
            repaired['kind'] = 'folding'
            repaired['tilt'] = float(cfg['variant']['tilt'])
            repaired['name'] = name
            actual_params = int(repaired['params'])
            common.require(actual_params <= reference_params,
                           'ANTI-TILT przekracza wspólny target')
            delta_percent = 100.0 * (
                actual_params - reference_params) / reference_params
            common.require(
                -float(cfg['budget_tolerance_percent']) <= delta_percent <= 0.0,
                f'ANTI-TILT odbiega od targetu o {delta_percent:.3f}%')
            repaired['target_params'] = reference_params
            repaired['uniform_reference_params'] = reference_params
            repaired['delta_params'] = actual_params - reference_params
            repaired['delta_percent'] = delta_percent
            repaired['nominal_target_params'] = nominal_target
            repaired['nominal_delta_percent'] = 100.0 * (
                actual_params - nominal_target) / nominal_target

            folded_cpu, diagnostics = common.fold_with_ratios(
                source_cpu, repaired['ratios'], int(cfg['fold_seed']))
            folded_params = common.count_parameters(folded_cpu)
            common.require(folded_params == actual_params,
                           'Liczba parametrów po foldingu nie zgadza się '
                           'z solverem')

            model_gpu = copy.deepcopy(folded_cpu).to(device)
            pre_metrics, pre_logits, pre_labels, pre_idx = common.evaluate(
                model_gpu, dev_loader, device)
            pre_vs_base = common.logit_metrics(pre_logits, baseline_logits)

            module.seed_everything(int(cfg['bn_seed']))
            bn_loader = e012.make_bn_loader(
                module, images, labels, train_indices,
                {'batch_size': int(cfg['batch_size']),
                 'num_workers': int(cfg['num_workers']),
                 'bn_seed': int(cfg['bn_seed'])})
            common.adapt_bn(
                model_gpu, bn_loader, device, len(train_indices),
                int(cfg['bn_seed']))
            post_metrics, post_logits, post_labels, post_idx = (
                common.evaluate(model_gpu, dev_loader, device))
            post_vs_base = common.logit_metrics(post_logits, baseline_logits)
            common.require(
                (pre_idx == baseline_idx).all()
                and (post_idx == baseline_idx).all(),
                'Niezgodna kolejność indeksów dev')
            common.require(
                (pre_labels == baseline_labels).all()
                and (post_labels == baseline_labels).all(),
                'Niezgodne etykiety dev')

            npz_path = output_dir / 'logits.npz'
            np.savez_compressed(
                npz_path, baseline=baseline_logits, pre_bn=pre_logits,
                post_bn=post_logits, labels=baseline_labels,
                indices=baseline_idx)
            common.write_json_atomic(output_dir / 'diagnostics.json', diagnostics)

            state.update({
                'status': 'completed',
                'allocation': repaired,
                'baseline': {
                    'params': baseline_params,
                    'metrics': baseline_metrics,
                    'checkpoint_sha256': entry['checkpoint_sha256'],
                },
                'compressed': {
                    'params': folded_params,
                    'params_before': baseline_params,
                    'param_reduction': baseline_params - folded_params,
                    'param_reduction_fraction': 1.0 - (
                        folded_params / baseline_params),
                    'pre_bn': pre_metrics,
                    'post_bn': post_metrics,
                    'pre_bn_vs_baseline': pre_vs_base,
                    'post_bn_vs_baseline': post_vs_base,
                    'bn': {
                        'samples': int(len(train_indices)),
                        'seed': int(cfg['bn_seed']),
                    },
                },
                'damage': {
                    'pre_bn_top1_pp': 100.0 * (
                        baseline_metrics['top1'] - pre_metrics['top1']),
                    'post_bn_top1_pp': 100.0 * (
                        baseline_metrics['top1'] - post_metrics['top1']),
                    'bn_recovery_top1_pp': 100.0 * (
                        post_metrics['top1'] - pre_metrics['top1']),
                    'pre_bn_nll_delta': float(
                        pre_metrics['nll'] - baseline_metrics['nll']),
                    'post_bn_nll_delta': float(
                        post_metrics['nll'] - baseline_metrics['nll']),
                },
                'logits_file': npz_path.name,
                'logits_sha256': common.digest(npz_path),
                'diagnostics_file': 'diagnostics.json',
                'runtime_seconds': time.monotonic() - started,
                'finished_utc': common.utc_now(),
                'peak_gpu_allocated_mb': (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
                'environment': common.environment(),
            })
            common.write_json_atomic(output_dir / 'summary.json', state)
        except Exception as error:
            state.update({
                'status': 'failed',
                'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
                'finished_utc': common.utc_now(),
                'runtime_seconds': time.monotonic() - started,
            })
            common.write_json_atomic(output_dir / 'summary.json', state)
            failures.append(name)
    if failures:
        raise SystemExit(f'Nieudane itemy: {failures}')


if __name__ == '__main__':
    main()
