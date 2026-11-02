"""E018: jedno zadanie replikacji reference vs cosine.

Eksperyment: 2026-10-29_E018_replikacja_cosine; pełny protokół: OPIS.md.
Wejście: checkpoint seed43–45, zamrożony config i spec zadania.
Wyjście: ``tasks/<tag>/<miara>/bXX/summary.json``, logits.npz i
diagnostics.json; pomiar na dev przed/po BN, bez treningu wag.
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

import e018_common as common  # noqa: E402

E017 = common.E017
if str(E017) not in sys.path:
    sys.path.insert(0, str(E017))

import e017_common as base  # noqa: E402

E017_WORKER = base.load_module(E017 / 'worker.py', '_e018_e017_worker')


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


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
    task['preserve_fc_bias'] = bool(cfg['run']['preserve_fc_bias'])
    task['fold_seed'] = int(cfg['run']['fold_seed'])
    output_dir = (run_dir / 'tasks' / task['dataset'] / task['measure']
                  / f"b{int(round(float(task['budget']) * 100)):02d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH,
        'status': 'running',
        'task': task,
        'gpu': int(args.gpu),
        'started_utc': common.utc_now(),
        'official_test_used': False,
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
        torch.cuda.reset_peak_memory_stats(device)

        bundle = common.load_bundle(entry)
        data = bundle['data']
        fn = common.folding_module()
        ev = common.evaluate_module()
        data.seed_everything(int(cfg['run']['fold_seed']))
        dev_loader = ev.make_eval_loader(
            data, bundle['images'], bundle['labels'], bundle['dev_indices'],
            cfg['run'], int(cfg['run']['fold_seed']))
        baseline_model = copy.deepcopy(bundle['model']).to(device)
        baseline_metrics, baseline_logits, baseline_labels, baseline_idx = (
            ev.evaluate_model(baseline_model, dev_loader, device, collect=True))
        del baseline_model
        require(abs(baseline_metrics['top1']
                    - float(entry['baseline_top1'])) <= 1e-12,
                'Baseline top1 nie zgadza się z configiem')
        require(abs(baseline_metrics['nll']
                    - float(entry['baseline_nll'])) <= 1e-5,
                'Baseline NLL nie zgadza się z configiem')

        folded_cpu, groups = E017_WORKER.fold_with_measure(
            bundle['model'], task, fn)
        require(fn.state_dict_all_finite(folded_cpu),
                'Niefinitywne parametry po foldingu')
        folded_params = fn.count_parameters(folded_cpu)

        model_gpu = copy.deepcopy(folded_cpu).to(device)
        pre_metrics, pre_logits, pre_labels, pre_idx = ev.evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        pre_vs_base = E017_WORKER.logit_metrics(pre_logits, baseline_logits)

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
        post_vs_base = E017_WORKER.logit_metrics(post_logits, baseline_logits)
        require(np.array_equal(pre_idx, baseline_idx)
                and np.array_equal(post_idx, baseline_idx),
                'Niezgodna kolejność indeksów dev')
        require(np.array_equal(pre_labels, baseline_labels)
                and np.array_equal(post_labels, baseline_labels),
                'Niezgodne etykiety dev')

        np.savez_compressed(
            output_dir / 'logits.npz', baseline=baseline_logits,
            pre_bn=pre_logits, post_bn=post_logits,
            labels=baseline_labels, indices=baseline_idx)
        diagnostics = {
            'experiment': common.EXPERIMENT,
            'opis': common.OPIS_PATH,
            'task': task,
            'groups': groups,
            'params_before': int(bundle['params']),
            'params_after': int(folded_params),
        }
        common.write_json_atomic(
            output_dir / 'diagnostics.json', diagnostics)
        state.update({
            'status': 'completed',
            'finished_utc': common.utc_now(),
            'runtime_seconds': time.monotonic() - started,
            'params_before': int(bundle['params']),
            'params_after': int(folded_params),
            'reduction_fraction': float(
                1.0 - folded_params / bundle['params']),
            'baseline': {'metrics': baseline_metrics},
            'compressed': {
                'pre_bn': pre_metrics,
                'post_bn': post_metrics,
                'pre_bn_vs_baseline': pre_vs_base,
                'post_bn_vs_baseline': post_vs_base,
                'bn': bn_stats,
            },
            'groups': groups,
            'diagnostics_sha256': common.digest(
                output_dir / 'diagnostics.json'),
            'logits_sha256': common.digest(output_dir / 'logits.npz'),
        })
        common.write_json_atomic(output_dir / 'summary.json', state)
        print(f"E018_TASK_DONE {task['name']} "
              f"top1={post_metrics['top1']:.5f}", flush=True)
    except BaseException as error:
        state.update({
            'status': 'failed',
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'finished_utc': common.utc_now(),
        })
        common.write_json_atomic(output_dir / 'summary.json', state)
        raise


if __name__ == '__main__':
    main()
