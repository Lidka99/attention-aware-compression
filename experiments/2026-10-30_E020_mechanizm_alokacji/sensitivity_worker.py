"""E020: worker pomiaru wrażliwości pojedynczych grup na Tiny ImageNet.

Eksperyment: 2026-10-30_E020_mechanizm_alokacji; pełny opis: OPIS.md.
Wejście: checkpoint E019 seed42, split Tiny ImageNet, zamrożony config
i lista itemów (12 grup x poziomy 25/50/75/90% + kontrola ratio 0).
Wyjście w ``<run>/items/<item>/``: ``summary.json``, ``diagnostics.json``
i ``logits.npz`` (baseline/pre-BN/post-BN). Bez treningu wag; BN na pełnym
train; oficjalny val/rezerwa nietknięte.
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

    common.require(
        common.digest(cfg['checkpoint']) == cfg['checkpoint_sha256'],
        'Hash checkpointu nie zgadza się z configiem')
    common.require(
        common.digest(cfg['split']) == cfg['split_sha256'],
        'Hash splitu nie zgadza się z configiem')
    dataset = cfg['dataset']
    module, images, labels, classes, split = common.load_dataset(
        dataset, cfg['split'])
    source_cpu, entry = common.load_checkpoint(dataset, int(cfg['seed']))
    common.require(
        abs(float(entry['baseline_top1']) - float(cfg['baseline_top1'])) <= 1e-12,
        'Baseline top1 nie zgadza się z inwentarzem')
    common.require(
        abs(float(entry['baseline_nll']) - float(cfg['baseline_nll'])) <= 1e-5,
        'Baseline NLL nie zgadza się z inwentarzem')

    dev_loader, bn_loader, train_indices = common.make_loaders(
        module, images, labels, split, cfg['batch_size'], cfg['num_workers'],
        cfg['seed'], cfg['bn_seed'])
    common.require(len(train_indices) == int(cfg['train_count']),
                   'BN musi obejmować pełny train')

    baseline_model = copy.deepcopy(source_cpu).to(device)
    baseline_metrics, baseline_logits, baseline_labels, baseline_indices = (
        common.evaluate(baseline_model, dev_loader, device))
    del baseline_model
    common.require(
        abs(baseline_metrics['top1'] - float(cfg['baseline_top1'])) <= 1e-12,
        'Baseline top1 na dev nie zgadza się z configiem')
    common.require(
        abs(baseline_metrics['nll'] - float(cfg['baseline_nll'])) <= 1e-5,
        'Baseline NLL na dev nie zgadza się z configiem')

    e012 = common.e012_evaluate()
    del bn_loader
    failures = []
    for item in items:
        name = str(item['name'])
        output_dir = run_dir / 'items' / name
        output_dir.mkdir(parents=True, exist_ok=True)
        started_utc = common.utc_now()
        started = time.monotonic()
        state = {
            'experiment': common.EXPERIMENT,
            'opis': common.OPIS_PATH,
            'status': 'running',
            'item': item,
            'dataset': dataset,
            'seed': int(cfg['seed']),
            'gpu': int(args.gpu),
            'batch_size_per_gpu': int(cfg['batch_size']),
            'effective_global_batch_per_run': int(cfg['batch_size']),
            'official_test_used': False,
            'reserve_used': False,
            'started_utc': started_utc,
        }
        try:
            group_index = item.get('group_index')
            level = float(item.get('level', 0.0))
            ratios = common.ratios_for_single_group(group_index, level)
            folded_cpu, diagnostics = common.fold_with_ratios(
                source_cpu, ratios, int(cfg['fold_seed']))
            folded_params = common.count_parameters(folded_cpu)
            common.require(
                folded_params <= common.count_parameters(source_cpu),
                'Folding zwiększył liczbę parametrów')

            model_gpu = copy.deepcopy(folded_cpu).to(device)
            pre_metrics, pre_logits, pre_labels, pre_indices = common.evaluate(
                model_gpu, dev_loader, device)
            pre_vs_base = common.logit_metrics(pre_logits, baseline_logits)

            data_seed = int(cfg['bn_seed'])
            module.seed_everything(data_seed)
            bn_loader = e012.make_bn_loader(
                module, images, labels, train_indices,
                {'batch_size': int(cfg['batch_size']),
                 'num_workers': int(cfg['num_workers']),
                 'bn_seed': data_seed})
            bn_stats = common.adapt_bn(
                model_gpu, bn_loader, device, len(train_indices), data_seed)
            del bn_loader
            post_metrics, post_logits, post_labels, post_indices = (
                common.evaluate(model_gpu, dev_loader, device))
            post_vs_base = common.logit_metrics(post_logits, baseline_logits)
            common.require(
                np.array_equal(pre_indices, baseline_indices)
                and np.array_equal(post_indices, baseline_indices),
                'Niezgodna kolejność indeksów dev')
            common.require(
                np.array_equal(pre_labels, baseline_labels)
                and np.array_equal(post_labels, baseline_labels),
                'Niezgodne etykiety dev')

            npz_path = output_dir / 'logits.npz'
            np.savez_compressed(
                npz_path, baseline=baseline_logits, pre_bn=pre_logits,
                post_bn=post_logits, labels=baseline_labels,
                indices=baseline_indices)
            common.write_json_atomic(output_dir / 'diagnostics.json', diagnostics)

            depth_index = None if group_index is None else int(group_index)
            depth_norm = None if depth_index is None else depth_index / 11.0
            predictors = {
                'depth_index': depth_index,
                'depth_norm': depth_norm,
                'weight_reconstruction_rel_error':
                    common.weight_reconstruction_error(
                        source_cpu, group_index, diagnostics),
                'calib_logit_rmse_pre_bn': None,
                'calib_logit_rmse_post_bn': None,
                'calib_prediction_agreement_pre_bn': None,
                'calib_prediction_agreement_post_bn': None,
            }
            calib_seed = int(cfg['calibration_seed'])
            calib_samples = int(cfg['calibration_samples'])
            generator = np.random.default_rng(calib_seed)
            calib_index = generator.choice(
                train_indices, size=calib_samples, replace=False)
            calib_index.sort()
            calib_loader = e012.make_eval_loader(
                module, images, labels, calib_index,
                {'batch_size': int(cfg['batch_size']),
                 'num_workers': int(cfg['num_workers'])}, calib_seed)
            calib_base_metrics, calib_base_logits, _, _ = common.evaluate(
                copy.deepcopy(source_cpu).to(device), calib_loader, device)
            calib_pre_metrics, calib_pre_logits, _, _ = common.evaluate(
                copy.deepcopy(folded_cpu).to(device), calib_loader, device)
            calib_pre = common.logit_metrics(
                calib_pre_logits, calib_base_logits)
            post_model = copy.deepcopy(model_gpu)
            calib_post_metrics, calib_post_logits, _, _ = common.evaluate(
                post_model, calib_loader, device)
            calib_post = common.logit_metrics(
                calib_post_logits, calib_base_logits)
            predictors.update({
                'calib_logit_rmse_pre_bn': float(calib_pre['logit_rmse']),
                'calib_logit_rmse_post_bn': float(calib_post['logit_rmse']),
                'calib_prediction_agreement_pre_bn':
                    float(calib_pre['prediction_agreement']),
                'calib_prediction_agreement_post_bn':
                    float(calib_post['prediction_agreement']),
            })

            damage = {
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
                'bn_recovery_nll_delta': float(
                    pre_metrics['nll'] - post_metrics['nll']),
            }
            state.update({
                'status': 'completed',
                'baseline': {
                    'params': common.count_parameters(source_cpu),
                    'metrics': baseline_metrics,
                    'checkpoint_sha256': entry['checkpoint_sha256'],
                },
                'fold': {
                    'ratios': ratios,
                    'level': level,
                    'group_index': depth_index,
                    'group_name': item.get('group_name'),
                    'params_after': folded_params,
                    'params_before': common.count_parameters(source_cpu),
                    'param_reduction':
                        common.count_parameters(source_cpu) - folded_params,
                    'param_reduction_fraction': 1.0 - (
                        folded_params / common.count_parameters(source_cpu)),
                },
                'compressed': {
                    'pre_bn': pre_metrics,
                    'post_bn': post_metrics,
                    'pre_bn_vs_baseline': pre_vs_base,
                    'post_bn_vs_baseline': post_vs_base,
                    'bn': bn_stats,
                },
                'damage': damage,
                'predictors': predictors,
                'diagnostics_file': 'diagnostics.json',
                'logits_file': npz_path.name,
                'logits_sha256': common.digest(npz_path),
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
