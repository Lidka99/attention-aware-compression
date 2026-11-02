"""E016 S2: worker jednej zamrożonej alokacji 50% (bez fine-tuningu).

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu, etap S2; pełny protokół:
``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS_S2.md``.

Wejście: katalog biegu S2, zamrożony config, jeden plik spec alokacji
(dataset + wariant) i fizyczny numer GPU. Wyjście:
``allocations/<dataset>/<variant>/`` z ``summary.json``, ``diagnostics.json``
i ``logits.npz``. Jeden proces = jedno GPU = jeden wariant. Bez treningu,
bez gradientów, bez oficjalnego testu i bez rezerwy. Alokacja jest tylko
odczytywana; worker nie zmienia jej i nie używa dev do żadnej decyzji.
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
import s2_common as s2  # noqa: E402


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


def logit_metrics(candidate, baseline):
    """Metryki zmiany logitów względem baseline (bez etykiet w decyzji)."""
    import torch.nn.functional as functional

    candidate = torch.as_tensor(candidate, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    common.require(candidate.shape == baseline.shape,
                   'Niezgodne kształty logitów')
    diff = candidate - baseline
    logp_candidate = functional.log_softmax(candidate, dim=1)
    logp_baseline = functional.log_softmax(baseline, dim=1)
    return {
        'prediction_agreement': float(
            (candidate.argmax(1) == baseline.argmax(1)).float().mean()),
        'logit_mae': float(diff.abs().mean()),
        'logit_rmse': float(diff.pow(2).mean().sqrt()),
        'logit_max_abs': float(diff.abs().max()),
        'kl_baseline_candidate': float(functional.kl_div(
            logp_candidate, logp_baseline.exp(), reduction='batchmean')),
        'kl_candidate_baseline': float(functional.kl_div(
            logp_baseline, logp_candidate.exp(), reduction='batchmean')),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true',
                        help='mały podzbiór BN/dev; bez znaczenia naukowego')
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()
    cfg = s2.load_config(config_path)
    s2.verify_sources(cfg)
    common.require(config_path.is_file(), f'Brak configu: {config_path}')
    common.require(spec_path.is_file(), f'Brak spec: {spec_path}')
    manifest = common.read_json(run_dir / 'manifest.json')
    common.require(manifest.get('status') == 'running',
                   'Manifest biegu nie jest aktywny')
    common.require(manifest.get('config_sha256')
                   == common.digest(config_path),
                   'Config biegu nie zgadza się z manifestem')
    common.require(manifest.get('official_test_used') is False,
                   'Bieg nie może używać oficjalnego testu')
    common.require(manifest.get('reserve_used') is False,
                   'Bieg nie może używać rezerwy')
    common.require(manifest.get('allocations_frozen_utc') is not None,
                   'Alokacje muszą być zamrożone przed startem workera')
    spec = common.read_json(spec_path)
    spec_sha = common.digest(spec_path)
    require = common.require
    require(spec_sha == manifest['specs'][spec['dataset']][spec['variant']]['sha256'],
            'Spec alokacji nie zgadza się z manifestem')
    require(spec.get('dev_used_for_allocation') is False,
            'Spec alokacji nie może używać dev')
    require(spec.get('uses_test') is False
            and spec.get('uses_reserve') is False,
            'Spec alokacji nie może używać testu/rezerwy')
    require(spec['dataset'] in s2.DATASETS, f"Nieznany dataset: {spec['dataset']}")
    require(spec['variant'] in s2.VARIANTS,
            f"Nieznany wariant: {spec['variant']}")
    require(0 <= int(args.gpu) < int(cfg['run']['gpu_count']),
            'Zły numer GPU')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Worker wymaga dokładnie jednego widocznego GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(visible is not None and visible.strip() == str(args.gpu),
            'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')

    dataset = common.dataset_by_name(cfg, spec['dataset'])
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    common.require(common.folding_budget().load_reference_folding().hkmeans_available(),
                   'Brak hkmeans/sklearn do foldingu')
    bundle = common.load_dataset_bundle(dataset, verify_hashes=True)
    module = bundle['module']
    module.seed_everything(int(dataset['seed']))
    torch.cuda.reset_peak_memory_stats(device)

    source_cpu = bundle['source_cpu']
    baseline_params = int(sum(parameter.numel()
                              for parameter in source_cpu.parameters()))
    require(baseline_params == int(dataset['expected_parameters']),
            'Zła liczba parametrów checkpointu')
    allocation = spec['allocation']
    target = int(allocation['target_params'])
    require(int(allocation['params']) <= target,
            'Alokacja przekracza target już w spec')
    exact = s2.verify_allocation(source_cpu, allocation, cfg)
    require(exact == int(allocation['params']),
            'Spec alokacji nie zgadza się z rzeczywistym modelem')

    runtime_cfg = {
        'batch_size': int(cfg['run']['batch_size']),
        'num_workers': int(cfg['run']['num_workers']),
        'bn_seed': int(cfg['run']['bn_seed']),
    }
    train_indices = bundle['train_indices']
    dev_indices = bundle['dev_indices']
    if args.smoke:
        train_indices = train_indices[:int(s2.SMOKE_TRAIN)]
        dev_indices = dev_indices[:int(s2.SMOKE_DEV)]
    dev_loader = common.make_eval_loader(
        module, bundle['images'], bundle['labels'], dev_indices,
        runtime_cfg, int(dataset['seed']))

    started = time.monotonic()
    started_utc = common.utc_now()
    output_dir = run_dir / 'allocations' / spec['dataset'] / spec['variant']
    require(not output_dir.exists(), f'Katalog wyniku już istnieje: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    state = {
        'experiment': cfg['experiment'],
        'opis': cfg['opis'],
        'stage': cfg['stage'],
        'status': 'running',
        'run_id': run_dir.name,
        'dataset': spec['dataset'],
        'variant': spec['variant'],
        'family': spec['family'],
        'seed': int(dataset['seed']),
        'gpu': int(args.gpu),
        'pid': os.getpid(),
        'spec_file': str(spec_path),
        'spec_sha256': spec_sha,
        'allocation': allocation,
        'started_utc': started_utc,
        'official_test_used': False,
        'reserve_used': False,
        'dev_used_for_allocation': False,
        'smoke': bool(args.smoke),
        'batch_size_per_gpu': int(cfg['run']['batch_size']),
        'gradient_accumulation': 1,
        'effective_global_batch_per_run': int(cfg['run']['batch_size']),
        'aggregation': 'none (independent allocations)',
    }

    try:
        e012 = common.e012_evaluate()
        baseline_model = copy.deepcopy(source_cpu).to(device)
        baseline_metrics, baseline_logits, baseline_labels, baseline_idx = (
            e012.evaluate_model(baseline_model, dev_loader, device,
                                collect=True))
        del baseline_model
        torch.cuda.empty_cache()
        if not args.smoke:
            require(abs(baseline_metrics['top1']
                        - float(dataset['baseline_top1'])) <= 1e-12,
                    'Baseline top1 nie zgadza się z configiem')
            require(abs(baseline_metrics['nll']
                        - float(dataset['baseline_nll'])) <= 1e-5,
                    'Baseline NLL nie zgadza się z configiem')
        else:
            require(0.0 <= baseline_metrics['top1'] <= 1.0
                    and np.isfinite(baseline_metrics['nll']),
                    'Baseline smoke nie jest poprawny')
        baseline_macs = e012.compute_macs(source_cpu)

        fold_started = time.monotonic()
        folded_cpu, diagnostics = common.folding_budget().fold_with_ratios(
            source_cpu, allocation['ratios'],
            seed=int(cfg['run']['fold_seed']), deterministic=False,
            preserve_fc_bias=True)
        fold_seconds = time.monotonic() - fold_started
        folded_params = e012.count_parameters(folded_cpu)
        require(folded_params == int(allocation['params']),
                f'Fold {folded_params} != spec {allocation["params"]}')
        require(folded_params <= target,
                'Fold przekroczył target UNIFORM')
        require(common.folding_budget().load_reference_folding()
                .state_dict_all_finite(folded_cpu),
                'Niefinitywne parametry po foldingu')
        folded_macs = e012.compute_macs(folded_cpu)
        diagnostics['requested_ratios'] = allocation['ratios']
        diagnostics['allocation_name'] = allocation['name']
        diagnostics['fold_seconds'] = fold_seconds
        common.write_json_atomic(output_dir / 'diagnostics.json', diagnostics)

        model_gpu = copy.deepcopy(folded_cpu).to(device)
        pre_started = time.monotonic()
        pre_metrics, pre_logits, pre_labels, pre_idx = e012.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        pre_seconds = time.monotonic() - pre_started
        pre_vs_base = logit_metrics(pre_logits, baseline_logits)

        module.seed_everything(int(cfg['run']['bn_seed']))
        bn_loader = common.make_bn_loader(
            module, bundle['images'], bundle['labels'], train_indices,
            runtime_cfg)
        bn_started = time.monotonic()
        bn_stats = e012.adapt_bn(
            model_gpu, bn_loader, device, len(train_indices),
            int(cfg['run']['bn_seed']))
        bn_seconds = time.monotonic() - bn_started
        del bn_loader

        post_started = time.monotonic()
        post_metrics, post_logits, post_labels, post_idx = e012.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        post_seconds = time.monotonic() - post_started
        post_vs_base = logit_metrics(post_logits, baseline_logits)
        require(np.array_equal(pre_idx, baseline_idx)
                and np.array_equal(post_idx, baseline_idx),
                'Niezgodna kolejność indeksów dev')
        require(np.array_equal(pre_labels, baseline_labels)
                and np.array_equal(post_labels, baseline_labels),
                'Niezgodne etykiety dev')
        expected_bn = int(s2.SMOKE_TRAIN) if args.smoke else int(dataset['bn_samples'])
        require(int(bn_stats['samples']) == expected_bn,
                'BN nie użył oczekiwanej liczby próbek')
        require(bn_stats['params_unchanged'] is True,
                'Parametry zmieniły się podczas BN')
        require(bn_stats['params_have_grad'] is False,
                'Pojawiły się gradienty podczas BN')
        expected_dev = int(s2.SMOKE_DEV) if args.smoke else 5000
        require(pre_metrics['count'] == expected_dev
                and post_metrics['count'] == expected_dev,
                'Dev nie ma oczekiwanej liczby próbek')

        npz_path = output_dir / 'logits.npz'
        np.savez_compressed(
            npz_path, baseline=baseline_logits, pre_bn=pre_logits,
            post_bn=post_logits, labels=baseline_labels, indices=baseline_idx)
        logits_sha = common.digest(npz_path)
        param_reduction = baseline_params - folded_params
        state.update({
            'status': 'completed',
            'completed_utc': common.utc_now(),
            'baseline': {
                'params': int(baseline_params),
                'macs': baseline_macs,
                'metrics': baseline_metrics,
            },
            'compressed': {
                'params': int(folded_params),
                'params_before': int(baseline_params),
                'param_reduction': int(param_reduction),
                'param_reduction_fraction': (
                    param_reduction / baseline_params),
                'actual_reduction_vs_baseline': (
                    1.0 - folded_params / baseline_params),
                'target_reduction_fraction': (
                    1.0 - target / baseline_params),
                'macs': folded_macs,
                'pre_bn': pre_metrics,
                'post_bn': post_metrics,
                'pre_bn_vs_baseline': pre_vs_base,
                'post_bn_vs_baseline': post_vs_base,
                'bn': bn_stats,
            },
            'target_params': int(target),
            'delta_params': int(folded_params - target),
            'delta_percent': float(
                100.0 * (folded_params - target) / target),
            'max_under_target_percent': float(
                cfg['budget']['max_under_target_percent']),
            'nominal_target_params': int(
                allocation.get('nominal_target_params', target)),
            'uniform_reference_params': int(
                allocation.get('uniform_reference_params', target)),
            'allocation_frozen_before_evaluation': True,
            'allocation_dev_used': False,
            'calibration_cost': {
                'kind': 'full_train_bn_adaptation',
                'samples': int(bn_stats['samples']),
                'batches': int(bn_stats['batches']),
                'full_train': not bool(args.smoke),
                'batch_size': int(bn_stats['batch_size']),
                'seconds': float(bn_seconds),
                'gradients': False,
                'labels_used_in_optimization': False,
                'same_for_all_variants': True,
            },
            'timing': {
                'fold_seconds': float(fold_seconds),
                'pre_bn_seconds': float(pre_seconds),
                'bn_seconds': float(bn_seconds),
                'post_bn_seconds': float(post_seconds),
                'total_seconds': time.monotonic() - started,
            },
            'logits_file': npz_path.name,
            'logits_sha256': logits_sha,
            'diagnostics_file': 'diagnostics.json',
            'diagnostics_sha256': common.digest(
                output_dir / 'diagnostics.json'),
            'peak_gpu_allocated_mb': (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            'peak_gpu_reserved_mb': (
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
            'environment': environment(),
            'reused_sources': {
                'e012_evaluate': str(
                    common.E012_EVALUATE.relative_to(common.ROOT)),
                'e014_folding_budget': str(
                    common.E014_FOLDING_BUDGET.relative_to(common.ROOT)),
                'e016_common': 'experiments/2026-10-27_E016_wrazliwosc_foldingu/e016_common.py',
            },
        })
        common.write_json_atomic(output_dir / 'summary.json', state)
        print(f"E016_S2={spec['dataset']}/{spec['variant']}_STATUS=completed",
              flush=True)
    except BaseException as error:
        state.update({
            'status': 'failed',
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'finished_utc': common.utc_now(),
            'runtime_seconds': time.monotonic() - started,
        })
        common.write_json_atomic(output_dir / 'summary.json', state)
        raise


if __name__ == '__main__':
    main()
