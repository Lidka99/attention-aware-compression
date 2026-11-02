"""E021: jedno zadanie benchmarku metryk grupowania.

Eksperyment: 2026-10-31_E021_metryki_grupowania; pełny protokół: OPIS.md.
Wejście: checkpoint E011/E015/E019, zamrożony config i spec zadania.
Wyjście: tasks/<tag>/<wariant>/bXX/summary.json i diagnostics.json;
pomiar na dev przed/po BN, bez treningu wag, bez testu i rezerwy.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e021_common as common  # noqa: E402
import grouping_e021 as grouping  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def logit_metrics(candidate, baseline):
    candidate = torch.as_tensor(candidate, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    diff = candidate - baseline
    logp_candidate = F.log_softmax(candidate, dim=1)
    logp_baseline = F.log_softmax(baseline, dim=1)
    return {
        'prediction_agreement': float(
            (candidate.argmax(1) == baseline.argmax(1)).float().mean()),
        'logit_mae': float(diff.abs().mean()),
        'logit_rmse': float(diff.pow(2).mean().sqrt()),
        'kl_baseline_candidate': float(F.kl_div(
            logp_candidate, logp_baseline.exp(), reduction='batchmean')),
    }


def fold_with_metric(source_cpu, ratios, task):
    folding = common.folding_module()
    names = folding.group_order()
    ratio_by_name = dict(zip(names, [float(v) for v in ratios]))
    model = copy.deepcopy(source_cpu).to(torch.device('cpu'))
    context = {
        'dataset': task['dataset'],
        'budget': f"{float(task['budget']):.2f}",
    }
    groups = []
    for group in folding.resnet18_groups():
        params = folding.collect_group_params(model, group)
        n_channels = folding.group_n_channels(params, group)
        ratio = ratio_by_name[group.name]
        k = max(int(n_channels * (1.0 - ratio)), 1)
        labels, record = grouping.labels_for(
            params, group, task['kind'], k, context)
        merge = folding.build_merge_matrix(labels, k, n_channels)
        merged = folding.merge_group_params(params, group, merge)
        folding._rebuild_group(
            model, group, merged, labels, k, True, 42)
        groups.append(record)
    return model, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    cfg = common.load_config(args.config)
    task = common.read_json(args.task)
    entry = common.dataset_entry(cfg, task['dataset'])
    output_dir = (run_dir / 'tasks' / task['dataset'] / task['variant']
                  / f"b{int(round(float(task['budget']) * 100)):02d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        'experiment': common.EXPERIMENT, 'opis': common.OPIS_PATH,
        'status': 'running', 'task': task, 'gpu': int(args.gpu),
        'started_utc': common.utc_now(), 'official_test_used': False,
        'reserve_used': False,
    }
    common.write_json_atomic(output_dir / 'summary.json', state)
    try:
        manifest = common.read_json(run_dir / 'manifest.json')
        require(manifest.get('status') == 'running',
                'Manifest biegu nie jest aktywny')
        require(manifest['config_sha256'] == common.digest(args.config),
                'Config biegu nie zgadza się z manifestem')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Worker wymaga jednego widocznego GPU')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        require(visible is not None and visible.strip() == str(args.gpu),
                'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)

        bundle = common.load_bundle(entry)
        data = bundle['data']
        ev = common.evaluate_module()
        folding = common.folding_module()
        data.seed_everything(42)
        dev_loader = ev.make_eval_loader(
            data, bundle['images'], bundle['labels'], bundle['dev_indices'],
            cfg['run'], 42)
        baseline_model = copy.deepcopy(bundle['model']).to(device)
        baseline_metrics, baseline_logits, baseline_labels, baseline_idx = (
            ev.evaluate_model(baseline_model, dev_loader, device, collect=True))
        del baseline_model
        require(abs(baseline_metrics['top1']
                    - float(entry['baseline_top1'])) <= 1e-12,
                'Baseline top1 nie zgadza się')
        require(abs(baseline_metrics['nll']
                    - float(entry['baseline_nll'])) <= 1e-5,
                'Baseline NLL nie zgadza się')

        nominal = int(round((1.0 - float(task['budget'])) * bundle['params']))
        solved = common.folding_budget.solve_base_for_budget(
            bundle['model'], 'uniform', nominal)
        require(int(solved['params']) <= nominal,
                'UNIFORM przekroczył nominalny cel')
        ratios = [float(value) for value in solved['ratios']]
        expected = common.folding_budget.parameter_count_for_ratios(
            bundle['model'], ratios)
        folded_cpu, groups = fold_with_metric(
            bundle['model'], ratios, task)
        require(folding.state_dict_all_finite(folded_cpu),
                'Niefinitywne parametry po foldingu')
        params = folding.count_parameters(folded_cpu)
        require(params == int(expected),
                f'Parametry {params} != oczekiwane {expected}')
        require(params <= nominal, 'Przekroczono nominalny budżet')

        model_gpu = copy.deepcopy(folded_cpu).to(device)
        pre_metrics, pre_logits, pre_labels, pre_idx = ev.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        pre_vs_base = logit_metrics(pre_logits, baseline_logits)
        data.seed_everything(int(cfg['run']['bn_seed']))
        bn_loader = ev.make_bn_loader(
            data, bundle['images'], bundle['labels'],
            bundle['train_indices'], cfg['run'])
        bn_stats = ev.adapt_bn(
            model_gpu, bn_loader, device, len(bundle['train_indices']),
            int(cfg['run']['bn_seed']))
        del bn_loader
        post_metrics, post_logits, post_labels, post_idx = ev.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        post_vs_base = logit_metrics(post_logits, baseline_logits)
        require(np.array_equal(pre_idx, baseline_idx)
                and np.array_equal(post_idx, baseline_idx),
                'Niezgodna kolejność indeksów dev')
        require(np.array_equal(pre_labels, baseline_labels)
                and np.array_equal(post_labels, baseline_labels),
                'Niezgodne etykiety dev')

        common.write_json_atomic(output_dir / 'diagnostics.json', {
            'experiment': common.EXPERIMENT, 'opis': common.OPIS_PATH,
            'task': task, 'groups': groups, 'ratios': ratios,
            'params_before': int(bundle['params']),
            'params_after': int(params),
        })
        state.update({
            'status': 'completed', 'finished_utc': common.utc_now(),
            'runtime_seconds': time.monotonic() - started,
            'params_before': int(bundle['params']),
            'params_after': int(params),
            'nominal_target_params': int(nominal),
            'uniform_params': int(solved['params']),
            'reduction_fraction': float(1.0 - params / bundle['params']),
            'baseline': {'metrics': baseline_metrics},
            'compressed': {
                'pre_bn': pre_metrics, 'post_bn': post_metrics,
                'bn': bn_stats,
                'pre_bn_vs_baseline': pre_vs_base,
                'post_bn_vs_baseline': post_vs_base,
            },
            'groups': groups,
            'diagnostics_sha256': common.digest(
                output_dir / 'diagnostics.json'),
        })
        common.write_json_atomic(output_dir / 'summary.json', state)
        print(f"E021_TASK_DONE {task['name']} "
              f"top1={post_metrics['top1']:.5f}", flush=True)
    except BaseException as error:
        state.update({
            'status': 'failed', 'error': repr(error),
            'traceback': traceback.format_exc(),
            'finished_utc': common.utc_now(),
        })
        common.write_json_atomic(output_dir / 'summary.json', state)
        raise


if __name__ == '__main__':
    main()
