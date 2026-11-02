"""E016: worker jednego sharda diagnostyki wrażliwości foldingu.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu; pełny opis: OPIS.md
(``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS.md``).

Wejście: katalog biegu S1, nazwa datasetu, numer GPU i plik z listą itemów
(shard). Wyjście: dla każdego itemu unikalny katalog
``items/<dataset>/<item>/`` z ``summary.json``, ``diagnostics.json`` i
``logits.npz``. Jeden proces = jedno GPU = jeden rozłączny shard. Bez
treningu, bez gradientów, bez fine-tuningu; tylko train/dev, bez oficjalnego
testu i bez rezerwy.
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402


def environment():
    """Zwraca wersje środowiska i ustawienia determinizmu."""
    import torchvision

    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu_name': torch.cuda.get_device_name(0),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': bool(
            torch.are_deterministic_algorithms_enabled()),
        'conda_default_env': os.environ.get('CONDA_DEFAULT_ENV'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--items', type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    items_path = args.items.expanduser().resolve()
    config_path = run_dir / 'config.json'
    manifest_path = run_dir / 'manifest.json'
    common.require(config_path.is_file(), f'Brak configu biegu: {config_path}')
    common.require(manifest_path.is_file(), f'Brak manifestu: {manifest_path}')
    cfg = common.load_config(config_path)
    manifest = common.read_json(manifest_path)
    common.require(manifest.get('status') == 'running',
                   'Manifest biegu nie jest aktywny')
    common.require(manifest.get('config_sha256') == common.digest(config_path),
                   'Config biegu nie zgadza się z manifestem')
    common.require(manifest.get('official_test_used') is False,
                   'Bieg nie może używać oficjalnego testu')
    common.require(manifest.get('reserve_used') is False,
                   'Bieg nie może używać rezerwy')
    common.require(args.dataset in [item['name'] for item in cfg['datasets']],
                   f'Nieznany dataset: {args.dataset}')
    common.require(0 <= args.gpu < int(cfg['gpu_count']), 'Zły numer GPU')
    common.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                   'Worker wymaga dokładnie jednego widocznego GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    common.require(visible is not None and visible.strip() == str(args.gpu),
                   'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')
    common.verify_frozen_sources(cfg)
    items = common.read_json(items_path)
    common.require(isinstance(items, list) and items, 'Pusty shard itemów')
    by_name = {item['name']: item for item in manifest.get('items', [])}
    for item in items:
        common.require(item['name'] in by_name,
                       f"Item {item['name']} nie jest w manifeście")
        common.require(by_name[item['name']]['gpu'] == args.gpu,
                       f"Item {item['name']} przypisano do innego GPU")
        common.require(by_name[item['name']]['dataset'] == args.dataset,
                       f"Item {item['name']} należy do innego datasetu")

    dataset = common.dataset_by_name(cfg, args.dataset)
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    bundle = common.load_dataset_bundle(dataset, verify_hashes=True)
    module = bundle['module']
    module.seed_everything(int(dataset['seed']))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    started_utc = common.utc_now()
    pid = os.getpid()

    source_cpu = bundle['source_cpu']
    baseline_params = int(sum(value.numel() for value in source_cpu.parameters()))
    dev_loader = common.make_eval_loader(
        module, bundle['images'], bundle['labels'], bundle['dev_indices'],
        cfg, int(dataset['seed']))
    calibration = common.calibration_indices(
        bundle['train_indices'], int(cfg['calibration_seed']),
        int(cfg['calibration_samples']))
    calibration_sha = common.indices_digest(calibration)
    calibration_loader = common.make_eval_loader(
        module, bundle['images'], bundle['labels'], calibration, cfg,
        int(cfg['calibration_seed']))

    baseline_model = copy.deepcopy(source_cpu).to(device)
    baseline_metrics, baseline_logits, baseline_labels, baseline_indices = (
        common.e012_evaluate().evaluate_model(
            baseline_model, dev_loader, device, collect=True))
    baseline_calibration_logits = common.collect_logits(
        baseline_model, calibration_loader, device)
    del baseline_model
    torch.cuda.empty_cache()
    common.require(
        abs(baseline_metrics['top1'] - float(dataset['baseline_top1'])) <= 1e-12,
        'Baseline top1 nie zgadza się z configiem')
    common.require(
        abs(baseline_metrics['nll'] - float(dataset['baseline_nll'])) <= 1e-5,
        'Baseline NLL nie zgadza się z configiem')

    print(
        f'E016_WORKER_START dataset={args.dataset} gpu={args.gpu} '
        f'pid={pid} items={len(items)}', flush=True)
    completed = 0
    for item in items:
        item_started = time.monotonic()
        output_dir = run_dir / 'items' / args.dataset / item['name']
        common.require(not output_dir.exists(),
                       f'Katalog itemu już istnieje: {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=False)
        state = {
            'experiment': common.EXPERIMENT,
            'opis': common.OPIS_PATH,
            'status': 'running',
            'run_id': run_dir.name,
            'dataset': args.dataset,
            'seed': int(dataset['seed']),
            'item': item,
            'gpu': int(args.gpu),
            'pid': pid,
            'items_shard': str(items_path),
            'started_utc': common.utc_now(),
            'official_test_used': False,
            'reserve_used': False,
        }
        try:
            folded_cpu, diagnostics = common.fold_item(
                source_cpu, item, int(cfg['fold_seed']))
            folded_params = int(
                sum(value.numel() for value in folded_cpu.parameters()))
            reference = common.folding_budget().load_reference_folding()
            common.require(
                reference.state_dict_all_finite(folded_cpu),
                'Niefinitywne parametry po foldingu')
            common.require(folded_params <= baseline_params,
                           'Folding zwiększył liczbę parametrów')
            model_gpu = copy.deepcopy(folded_cpu).to(device)
            pre_metrics, pre_logits, pre_labels, pre_indices = (
                common.e012_evaluate().evaluate_model(
                    model_gpu, dev_loader, device, collect=True))
            pre_calibration_logits = common.collect_logits(
                model_gpu, calibration_loader, device)

            module.seed_everything(int(cfg['bn_seed']))
            bn_loader = common.make_bn_loader(
                module, bundle['images'], bundle['labels'],
                bundle['train_indices'], cfg)
            bn_stats = common.e012_evaluate().adapt_bn(
                model_gpu, bn_loader, device, len(bundle['train_indices']),
                int(cfg['bn_seed']))
            del bn_loader
            post_metrics, post_logits, post_labels, post_indices = (
                common.e012_evaluate().evaluate_model(
                    model_gpu, dev_loader, device, collect=True))
            post_calibration_logits = common.collect_logits(
                model_gpu, calibration_loader, device)
            common.require(
                np.array_equal(pre_indices, baseline_indices)
                and np.array_equal(post_indices, baseline_indices),
                'Niezgodna kolejność indeksów dev')
            common.require(
                np.array_equal(pre_labels, baseline_labels)
                and np.array_equal(post_labels, baseline_labels),
                'Niezgodne etykiety dev')

            group_entry = None
            group_width = None
            group_n_channels = None
            if item['group_index'] is not None:
                group_entry = next(
                    value for value in diagnostics['groups']
                    if value['name'] == item['group_name'])
                group_width = int(group_entry['width'])
                group_n_channels = int(group_entry['n_channels'])
            reconstruction = common.weight_reconstruction_error(
                source_cpu, item['group_index'], diagnostics)
            depth_index = (
                None if item['group_index'] is None
                else int(item['group_index']))
            depth_norm = (
                None if depth_index is None else depth_index / 11.0)

            logits_path = output_dir / 'logits.npz'
            np.savez_compressed(
                logits_path, baseline=baseline_logits, pre_bn=pre_logits,
                post_bn=post_logits, labels=baseline_labels,
                indices=baseline_indices)
            diagnostics_path = output_dir / 'diagnostics.json'
            common.write_json_atomic(diagnostics_path, diagnostics)

            pre_vs_baseline = common.logit_change(pre_logits, baseline_logits)
            post_vs_baseline = common.logit_change(post_logits, baseline_logits)
            calibration_pre = common.logit_change(
                pre_calibration_logits, baseline_calibration_logits)
            calibration_post = common.logit_change(
                post_calibration_logits, baseline_calibration_logits)
            state.update({
                'status': 'completed',
                'config_sha256': common.digest(config_path),
                'checkpoint_sha256': dataset['checkpoint_sha256'],
                'split_sha256': dataset['split_sha256'],
                'calibration': {
                    'samples': int(len(calibration)),
                    'seed': int(cfg['calibration_seed']),
                    'indices_sha256': calibration_sha,
                    'labels_used': False,
                    'transform': cfg['calibration_transform'],
                },
                'fold': {
                    'fold_seed': int(cfg['fold_seed']),
                    'group_index': item['group_index'],
                    'group_name': item['group_name'],
                    'group_stage': common.stage_of_group(item['group_name']),
                    'level': float(item['level']),
                    'control': bool(item['control']),
                    'ratios': common.ratios_for_item(item),
                    'group_n_channels': group_n_channels,
                    'group_width': group_width,
                    'params_before': baseline_params,
                    'params_after': folded_params,
                    'param_reduction': baseline_params - folded_params,
                    'param_reduction_fraction': (
                        (baseline_params - folded_params) / baseline_params),
                },
                'baseline': {
                    'metrics': baseline_metrics,
                    'params': baseline_params,
                },
                'compressed': {
                    'pre_bn': pre_metrics,
                    'post_bn': post_metrics,
                    'pre_bn_vs_baseline': pre_vs_baseline,
                    'post_bn_vs_baseline': post_vs_baseline,
                    'bn': bn_stats,
                },
                'damage': {
                    'pre_bn_top1_pp': 100.0 * (
                        baseline_metrics['top1'] - pre_metrics['top1']),
                    'post_bn_top1_pp': 100.0 * (
                        baseline_metrics['top1'] - post_metrics['top1']),
                    'bn_recovery_top1_pp': 100.0 * (
                        post_metrics['top1'] - pre_metrics['top1']),
                    'pre_bn_nll_delta': (
                        pre_metrics['nll'] - baseline_metrics['nll']),
                    'post_bn_nll_delta': (
                        post_metrics['nll'] - baseline_metrics['nll']),
                    'bn_recovery_nll_delta': (
                        post_metrics['nll'] - pre_metrics['nll']),
                },
                'predictors': {
                    'depth_index': depth_index,
                    'depth_norm': depth_norm,
                    'weight_reconstruction_rel_error': float(reconstruction),
                    'calib_logit_rmse_pre_bn': float(
                        calibration_pre['logit_rmse']),
                    'calib_logit_mae_pre_bn': float(
                        calibration_pre['logit_mae']),
                    'calib_prediction_agreement_pre_bn': float(
                        calibration_pre['prediction_agreement']),
                    'calib_kl_baseline_candidate_pre_bn': float(
                        calibration_pre['kl_baseline_candidate']),
                    'calib_logit_rmse_post_bn': float(
                        calibration_post['logit_rmse']),
                    'calib_logit_mae_post_bn': float(
                        calibration_post['logit_mae']),
                    'calib_prediction_agreement_post_bn': float(
                        calibration_post['prediction_agreement']),
                    'calib_kl_baseline_candidate_post_bn': float(
                        calibration_post['kl_baseline_candidate']),
                },
                'diagnostics_file': diagnostics_path.name,
                'logits_file': logits_path.name,
                'logits_sha256': common.digest(logits_path),
                'runtime_seconds': time.monotonic() - item_started,
                'finished_utc': common.utc_now(),
                'peak_gpu_allocated_mb': (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
                'peak_gpu_reserved_mb': (
                    torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
                'environment': environment(),
            })
            common.write_json_atomic(output_dir / 'summary.json', state)
            del model_gpu, folded_cpu
            torch.cuda.empty_cache()
            completed += 1
            print(
                f"E016_ITEM_DONE dataset={args.dataset} gpu={args.gpu} "
                f"name={item['name']} status=completed", flush=True)
        except BaseException as error:
            state.update({
                'status': 'failed',
                'error': repr(error),
                'traceback': traceback.format_exc(),
                'runtime_seconds': time.monotonic() - item_started,
                'finished_utc': common.utc_now(),
            })
            common.write_json_atomic(output_dir / 'summary.json', state)
            raise
    print(
        f'E016_WORKER_DONE dataset={args.dataset} gpu={args.gpu} '
        f'pid={pid} completed={completed}/{len(items)} '
        f'wall_seconds={time.monotonic() - started:.1f} '
        f'started_utc={started_utc}', flush=True)


if __name__ == '__main__':
    main()
