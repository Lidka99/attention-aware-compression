"""E012: ocena referencyjnego foldingu ResNet18/CIFAR-10 bez treningu.

Eksperyment: 2026-10-24_E012_folding_referencja; pełny opis: OPIS.md.
Wejście: zamrożony config (checkpoint i split E011 z hashami), jeden budżet
ratio oraz fizyczny GPU. Wyjście w <run>/ratio_XX/: summary.json,
fold_diagnostics.json, baseline.npz, <wariant>_{pre,post}_bn.npz oraz jeden
model <wariant>_post_bn.pt. Używany jest wyłącznie train/dev z E011; test
CIFAR-10 nie jest pobierany, wypakowywany ani oceniany. Brak optymalizatora
i gradientów; BN adaptowany jest na pełnym train (jedna epoka, bez drop).
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

EXPERIMENT = '2026-10-24_E012_folding_referencja'
OPIS_PATH = 'experiments/2026-10-24_E012_folding_referencja/OPIS.md'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E011_TRAIN = ROOT / (
    'experiments/2026-10-22_E011_cifar10_resnet18_checkpoint/train.py')
E011_MODULE = 'e012_e011_train'
EXPECTED_BASELINE_TOP1 = 0.9528
BASELINE_NLL_TOLERANCE = 1e-5
REQUIRED_CONFIG = (
    'checkpoint', 'checkpoint_sha256', 'split', 'split_sha256', 'seed',
    'ratios', 'batch_size', 'bn_seed', 'bn_samples', 'num_workers',
    'baseline_nll',
)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import folding


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_e011_module():
    require(E011_TRAIN.is_file(), f'Brak train.py E011: {E011_TRAIN}')
    spec = importlib.util.spec_from_file_location(E011_MODULE, E011_TRAIN)
    require(spec is not None and spec.loader is not None,
            f'Nie można załadować modułu E011: {E011_TRAIN}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[E011_MODULE] = module
    spec.loader.exec_module(module)
    return module


def load_config(path):
    path = Path(path)
    require(path.is_file(), f'Brak pliku konfiguracji: {path}')
    cfg = read_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in cfg]
    require(not missing, f'Brak pól konfiguracji: {missing}')

    for key in ('checkpoint', 'split'):
        value = Path(str(cfg[key])).expanduser()
        require(value.is_absolute(), f'{key} musi być ścieżką absolutną')
        cfg[key] = str(value.resolve())

    for key in ('checkpoint_sha256', 'split_sha256'):
        value = str(cfg[key]).strip().lower()
        require(len(value) == 64
                and all(character in '0123456789abcdef' for character in value),
                f'{key} musi być hashem sha256')
        cfg[key] = value

    for key in ('seed', 'batch_size', 'bn_seed', 'bn_samples', 'num_workers'):
        cfg[key] = int(cfg[key])
    cfg['baseline_nll'] = float(cfg['baseline_nll'])
    cfg['baseline_top1'] = float(cfg.get('baseline_top1', EXPECTED_BASELINE_TOP1))
    cfg['ratios'] = [float(value) for value in cfg['ratios']]

    require(cfg['batch_size'] > 0, 'batch_size musi być dodatni')
    require(cfg['bn_samples'] > 0, 'bn_samples musi być dodatnie')
    require(cfg['num_workers'] >= 0, 'num_workers musi być nieujemne')
    require(math.isfinite(cfg['baseline_nll']), 'baseline_nll musi być skończone')
    require(abs(cfg['baseline_top1'] - EXPECTED_BASELINE_TOP1) <= 1e-9,
            f'baseline_top1 musi wynosić {EXPECTED_BASELINE_TOP1}')
    require(cfg['ratios'], 'ratios nie może być puste')
    require(all(math.isfinite(value) and 0.0 <= value <= 1.0
                for value in cfg['ratios']),
            'Każde ratio musi być skończone i należeć do [0, 1]')
    require(len(set(cfg['ratios'])) == len(cfg['ratios']),
            'ratios zawiera duplikaty')
    return cfg


def ratio_dirname(ratio):
    if not math.isfinite(ratio):
        return 'ratio_invalid'
    return f'ratio_{int(round(float(ratio) * 100)):02d}'


def _digest_named_tensors(named_tensors):
    hasher = hashlib.sha256()
    for name, tensor in named_tensors:
        hasher.update(name.encode('utf-8'))
        hasher.update(str(tensor.dtype).encode('utf-8'))
        hasher.update(str(tuple(tensor.shape)).encode('utf-8'))
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def parameter_digest(model):
    return _digest_named_tensors(model.named_parameters())


def buffer_digest(model):
    return _digest_named_tensors(model.named_buffers())


def buffer_snapshot(model):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.named_buffers()}


def changed_buffer_names(before, after):
    require(set(before) == set(after), 'Zmienił się zestaw buforów modelu')
    return [name for name in sorted(before)
            if not torch.equal(before[name], after[name])]


def state_dict_shapes(model):
    return {name: [int(value) for value in tensor.shape]
            for name, tensor in model.state_dict().items()}


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def compute_macs(model):
    device = next(model.parameters()).device
    dummy = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device)
    totals = {'conv': 0, 'linear': 0}

    def conv_hook(module, inputs, output):
        kernel = int(module.kernel_size[0] * module.kernel_size[1])
        totals['conv'] += int(
            output.shape[0] * output.shape[1] * output.shape[2]
            * output.shape[3] * (module.in_channels // module.groups) * kernel)

    def linear_hook(module, inputs, output):
        totals['linear'] += int(
            output.shape[0] * module.in_features * module.out_features)

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    total = int(totals['conv'] + totals['linear'])
    return {
        'conv_macs': int(totals['conv']),
        'linear_macs': int(totals['linear']),
        'total_macs': total,
        'flops': int(2 * total),
        'input_shape': [1, 3, 32, 32],
        'excludes': ['batchnorm', 'relu', 'pooling', 'residual_add'],
    }


def save_npz_atomic(path, logits, labels, indices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, logits=logits, labels=labels, indices=indices)
    os.replace(temporary, path)
    return {
        'file': path.name,
        'sha256': digest(path),
        'logits_shape': [int(value) for value in logits.shape],
        'count': int(len(labels)),
    }


def save_model_atomic(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state_dict': {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        'metadata': metadata,
    }
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        'file': path.name,
        'sha256': digest(path),
    }


def evaluate_model(model, loader, device, collect=False):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    logits_parts = []
    label_parts = []
    index_parts = []
    indices = np.asarray(loader.dataset.indices, dtype=np.int64)
    cursor = 0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            require(logits.dtype == torch.float32,
                    f'Logity nie są float32: {logits.dtype}')
            require(bool(torch.isfinite(logits).all()), 'Niefinitywne logity')
            batch = int(targets.numel())
            loss_sum += float(F.cross_entropy(logits, targets, reduction='sum'))
            correct += int((logits.argmax(1) == targets).sum())
            count += batch
            if collect:
                logits_parts.append(logits.detach().cpu().to(torch.float32))
                label_parts.append(targets.detach().cpu().to(torch.int64))
                index_parts.append(
                    torch.from_numpy(indices[cursor:cursor + batch].copy()))
            cursor += batch

    require(count > 0, 'Pusty zbiór ewaluacyjny')
    metrics = {
        'nll': loss_sum / count,
        'top1': correct / count,
        'correct': int(correct),
        'count': int(count),
    }
    if not collect:
        return metrics, None, None, None
    require(cursor == len(indices), 'Niezgodna liczba indeksów ewaluacji')
    return (
        metrics,
        torch.cat(logits_parts, dim=0).numpy(),
        torch.cat(label_parts, dim=0).numpy(),
        torch.cat(index_parts, dim=0).numpy(),
    )


def make_eval_loader(e011, images, labels, indices, cfg, seed):
    dataset = e011.CifarTrainImages(
        images, labels, indices, e011.dev_transform())
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(cfg['batch_size']),
        shuffle=False,
        num_workers=int(cfg['num_workers']),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=e011.seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def make_bn_loader(e011, images, labels, train_indices, cfg):
    dataset = e011.CifarTrainImages(
        images, labels, train_indices, e011.train_transform())
    generator = torch.Generator().manual_seed(int(cfg['bn_seed']))
    return DataLoader(
        dataset,
        batch_size=int(cfg['batch_size']),
        shuffle=True,
        num_workers=int(cfg['num_workers']),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=e011.seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def adapt_bn(model, loader, device, expected_samples, seed):
    before_params = parameter_digest(model)
    before_buffers = buffer_snapshot(model)
    samples = 0
    batches = 0
    first_batch_sha256 = None

    model.train()
    with torch.no_grad():
        for images, targets in loader:
            if first_batch_sha256 is None:
                first_batch_sha256 = hashlib.sha256(
                    images.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
            images = images.to(device, non_blocking=True)
            logits = model(images)
            require(bool(torch.isfinite(logits).all()),
                    'Niefinitywne logity podczas adaptacji BN')
            samples += int(images.shape[0])
            batches += 1

    require(samples == int(expected_samples),
            f'Adaptacja BN zużyła {samples} próbek, oczekiwano {expected_samples}')
    require(batches > 0, 'Brak batchy podczas adaptacji BN')
    after_params = parameter_digest(model)
    require(before_params == after_params,
            'Parametry zmieniły się podczas adaptacji BN')
    after_buffers = buffer_snapshot(model)
    changed = changed_buffer_names(before_buffers, after_buffers)
    require(all(parameter.grad is None for parameter in model.parameters()),
            'Pojawiły się gradienty podczas adaptacji BN')
    model.eval()

    return {
        'seed': int(seed),
        'samples': int(samples),
        'expected_samples': int(expected_samples),
        'batches': int(batches),
        'batch_size': int(loader.batch_size),
        'num_workers': int(loader.num_workers),
        'persistent_workers': bool(loader.persistent_workers),
        'drop_last': bool(loader.drop_last),
        'shuffle': True,
        'augmentation': 'E011 train_transform',
        'first_batch_sha256': first_batch_sha256,
        'param_digest_before': before_params,
        'param_digest_after': after_params,
        'params_unchanged': True,
        'buffer_digest_before': _digest_named_tensors(before_buffers.items()),
        'buffer_digest_after': buffer_digest(model),
        'buffers_changed_count': int(len(changed)),
        'buffers_changed_names': changed,
        'params_have_grad': False,
    }


def fc_bias_summary(model, original_bias, policy):
    bias = model.fc.bias
    require(bias is not None, 'Model nie ma biasu fc')
    difference = bias.detach().cpu() - original_bias.detach().cpu()
    return {
        'policy': policy,
        'equal_original': bool(
            torch.equal(bias.detach().cpu(), original_bias.detach().cpu())),
        'norm': float(bias.detach().cpu().norm()),
        'original_norm': float(original_bias.detach().cpu().norm()),
        'max_abs_diff_from_original': float(difference.abs().max()),
    }


def verify_saved_model(path, reference_model, architecture_model, seed):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict),
            'Zapisany model nie ma state_dict')
    saved_state = payload['state_dict']
    reference_state = reference_model.state_dict()
    require(set(saved_state) == set(reference_state),
            'Niezgodny zestaw kluczy state_dict po zapisie')
    for name, tensor in reference_state.items():
        require(tuple(saved_state[name].shape) == tuple(tensor.shape),
                f'Niezgodny kształt po zapisie: {name}')

    reloaded = copy.deepcopy(architecture_model)
    reloaded.load_state_dict(saved_state, strict=True)
    reloaded.eval()
    reference_model.eval()
    for name, tensor in reference_state.items():
        require(torch.equal(tensor.detach().cpu(),
                            reloaded.state_dict()[name].detach().cpu()),
                f'Stan po reload nie jest identyczny: {name}')

    generator = torch.Generator().manual_seed(int(seed))
    synthetic = torch.randn((8, 3, 32, 32), generator=generator,
                            dtype=torch.float32)
    with torch.inference_mode():
        expected = reference_model(synthetic)
        actual = reloaded(synthetic)
    require(bool(torch.isfinite(expected).all())
            and bool(torch.isfinite(actual).all()),
            'Niefinitywne logity syntetyczne po reload')
    require(torch.equal(expected, actual),
            'Logity syntetyczne po reload nie są bitowo równe')
    return {
        'strict_load': True,
        'state_dict_equal': True,
        'synthetic_seed': int(seed),
        'synthetic_shape': [8, 3, 32, 32],
        'synthetic_logits_bit_equal': True,
        'synthetic_predictions_equal': bool(
            torch.equal(expected.argmax(1), actual.argmax(1))),
    }


def _environment(args):
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': __import__('torchvision').__version__,
        'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'platform': platform.platform(),
        'physical_gpu': int(args.gpu),
        'local_gpu': 0,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': torch.cuda.get_device_name(0),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': bool(
            torch.are_deterministic_algorithms_enabled()),
        'cpu_threads': int(torch.get_num_threads()),
    }


def run_worker(args, cfg, e011, ratio, run_dir, ratio_dir, started_utc,
               started_monotonic):
    seed = int(cfg['seed'])
    checkpoint_path = Path(cfg['checkpoint'])
    split_path = Path(cfg['split'])
    require(checkpoint_path.is_file(), f'Brak checkpointu: {checkpoint_path}')
    require(split_path.is_file(), f'Brak split: {split_path}')
    checkpoint_sha256 = digest(checkpoint_path)
    split_sha256 = digest(split_path)
    require(checkpoint_sha256 == cfg['checkpoint_sha256'],
            'Hash checkpointu nie zgadza się z configiem')
    require(split_sha256 == cfg['split_sha256'],
            'Hash split nie zgadza się z configiem')
    config_path = Path(args.config).resolve()
    config_sha256 = digest(config_path)
    data_root = Path(cfg.get('data_root', e011.DATA_ROOT)).expanduser().resolve()
    require(data_root == Path(e011.DATA_ROOT).resolve(),
            f'data_root musi wskazywać dane E011: {e011.DATA_ROOT}')

    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(visible is not None and visible.strip() == str(args.gpu),
            'CUDA_VISIBLE_DEVICES musi wskazywać fizyczny indeks z --gpu')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Worker wymaga dokładnie jednego widocznego GPU')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    e011.seed_everything(seed)
    torch.cuda.reset_peak_memory_stats(device)

    images, labels, classes = e011.load_cifar10_train(data_root)
    split = read_json(split_path)
    require('train' in split and 'dev' in split,
            'Split musi zawierać train i dev')
    split_seed = split.get('seed', cfg.get('split_seed'))
    require(split_seed is not None, 'Brak seeda split')
    if 'split_seed' in cfg:
        require(int(cfg['split_seed']) == int(split_seed),
                'split_seed w configu nie zgadza się ze split')
    e011.validate_split(split, labels, int(split_seed))
    train_indices = np.asarray(split['train'], dtype=np.int64)
    dev_indices = np.asarray(split['dev'], dtype=np.int64)
    require(len(train_indices) == int(cfg['bn_samples']),
            'bn_samples musi obejmować pełny train (bez podpróbkowania)')
    require(len(dev_indices) == e011.DEV_PER_CLASS * 10,
            'Dev nie ma oczekiwanej liczby próbek')

    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict),
            'Checkpoint nie zawiera state_dict')
    source_cpu = e011.build_model()
    source_cpu.load_state_dict(payload['state_dict'], strict=True)
    source_cpu.eval()
    require(count_parameters(source_cpu) == e011.EXPECTED_PARAMETERS,
            'Niezgodna liczba parametrów checkpointu')
    checkpoint_metadata = payload.get('metadata', {})
    if isinstance(checkpoint_metadata, dict) and 'seed' in checkpoint_metadata:
        require(int(checkpoint_metadata['seed']) == seed,
                'Seed checkpointu nie zgadza się z configiem')
    original_fc_bias = source_cpu.fc.bias.detach().cpu().clone()

    baseline_params = count_parameters(source_cpu)
    baseline_macs = compute_macs(source_cpu)
    dev_loader = make_eval_loader(
        e011, images, labels, dev_indices, cfg, seed)

    baseline_started = time.monotonic()
    baseline_model = copy.deepcopy(source_cpu).to(device)
    baseline_metrics, baseline_logits, baseline_labels, baseline_indices = (
        evaluate_model(baseline_model, dev_loader, device, collect=True))
    baseline_seconds = time.monotonic() - baseline_started
    del baseline_model

    expected_top1 = float(cfg['baseline_top1'])
    expected_correct = int(round(expected_top1 * baseline_metrics['count']))
    require(baseline_metrics['correct'] == expected_correct,
            f"Baseline top1 nie wynosi {expected_top1}: "
            f"{baseline_metrics['correct']}/{baseline_metrics['count']}")
    require(abs(baseline_metrics['top1'] - expected_top1) <= 1e-12,
            f"Baseline top1 nie wynosi {expected_top1}")
    baseline_nll_diff = abs(
        baseline_metrics['nll'] - float(cfg['baseline_nll']))
    require(baseline_nll_diff <= BASELINE_NLL_TOLERANCE,
            'Baseline NLL różni się od configu o więcej niż '
            f'{BASELINE_NLL_TOLERANCE}: {baseline_nll_diff}')
    baseline_npz = save_npz_atomic(
        ratio_dir / 'baseline.npz',
        baseline_logits, baseline_labels, baseline_indices)
    baseline_record = dict(baseline_metrics)
    baseline_record.update({
        'expected_top1': expected_top1,
        'expected_nll': float(cfg['baseline_nll']),
        'nll_abs_diff': baseline_nll_diff,
        'npz': baseline_npz,
    })

    fold_started = time.monotonic()
    if ratio > 0.0:
        require(folding.hkmeans_available(),
                'Brak hkmeans/sklearn potrzebnych do foldingu ratio > 0')
    folded_cpu, fold_diagnostics = folding.fold(
        source_cpu, ratio, seed=seed, preserve_fc_bias=False)
    fold_seconds = time.monotonic() - fold_started
    require(bool(fold_diagnostics['all_params_finite']),
            'Niefinitywne parametry po foldingu')
    folded_params = count_parameters(folded_cpu)
    folded_macs = compute_macs(folded_cpu)
    require(folded_params > 0, 'Model po foldingu nie ma parametrów')
    if ratio == 0.0:
        require(bool(fold_diagnostics['bypassed']),
                'ratio 0 musi być kontrolnym bypassem')
        require(folded_params == baseline_params,
                'Bypass zmienił liczbę parametrów')
    else:
        require(not fold_diagnostics['bypassed'], 'ratio > 0 nie może być bypassem')
        require(folded_params <= baseline_params,
                'Folding zwiększył liczbę parametrów')

    variant_names = ['ref_behavior'] if ratio == 0.0 else [
        'ref_behavior', 'preserve_bias']
    group_policy = {
        'folding_runs': 1,
        'shared_across_variants': True,
        'labels_source': 'folding.py fold()',
        'fc_bias_grouped': False,
        'preserve_bias_recluster': False,
        'variants': variant_names,
    }
    fold_record = dict(fold_diagnostics)
    fold_record.update({
        'requested_ratio': float(ratio),
        'fold_seconds': fold_seconds,
        'group_policy': group_policy,
        'nparams_baseline': baseline_params,
        'macs_baseline': baseline_macs,
        'macs_folded': folded_macs,
    })
    fold_path = ratio_dir / 'fold_diagnostics.json'
    write_json_atomic(fold_path, fold_record)
    fold_file = {
        'file': fold_path.name,
        'sha256': digest(fold_path),
    }

    variants = {}
    for variant in variant_names:
        variant_started = time.monotonic()
        pre_cpu = copy.deepcopy(folded_cpu)
        if variant == 'preserve_bias':
            require(original_fc_bias.shape == pre_cpu.fc.bias.shape,
                    'Niezgodny kształt biasu fc')
            pre_cpu.fc.bias.data.copy_(original_fc_bias)
            folded_state = folded_cpu.state_dict()
            pre_state = pre_cpu.state_dict()
            changed_keys = [
                name for name in folded_state
                if not torch.equal(folded_state[name], pre_state[name])
            ]
            require(changed_keys == ['fc.bias'],
                    f'Wariant preserve_bias zmienił więcej niż fc.bias: '
                    f'{changed_keys}')
            state_change = {
                'only_fc_bias_changed': True,
                'changed_keys': changed_keys,
            }
            fc_policy = 'copied_original_bias'
        elif ratio == 0.0:
            state_change = {
                'only_fc_bias_changed': False,
                'changed_keys': [],
            }
            fc_policy = 'bypass_unchanged'
        else:
            state_change = {
                'only_fc_bias_changed': False,
                'changed_keys': [],
            }
            fc_policy = 'regenerated_official_style'

        fc_bias_info = fc_bias_summary(
            pre_cpu, original_fc_bias, fc_policy)
        params_before = count_parameters(pre_cpu)
        macs_before = compute_macs(pre_cpu)
        require(macs_before['total_macs'] == folded_macs['total_macs'],
                'MACs wariantu różnią się od modelu po foldingu')

        model_gpu = copy.deepcopy(pre_cpu).to(device)
        pre_started = time.monotonic()
        pre_metrics, pre_logits, pre_labels, pre_indices = evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        pre_seconds = time.monotonic() - pre_started
        pre_npz = save_npz_atomic(
            ratio_dir / f'{variant}_pre_bn.npz',
            pre_logits, pre_labels, pre_indices)
        pre_record = dict(pre_metrics)
        pre_record['npz'] = pre_npz

        e011.seed_everything(int(cfg['bn_seed']))
        bn_loader = make_bn_loader(
            e011, images, labels, train_indices, cfg)
        bn_started = time.monotonic()
        bn_stats = adapt_bn(
            model_gpu, bn_loader, device, len(train_indices),
            int(cfg['bn_seed']))
        bn_seconds = time.monotonic() - bn_started
        del bn_loader

        post_started = time.monotonic()
        post_metrics, post_logits, post_labels, post_indices = evaluate_model(
            model_gpu, dev_loader, device, collect=True)
        post_seconds = time.monotonic() - post_started
        post_npz = save_npz_atomic(
            ratio_dir / f'{variant}_post_bn.npz',
            post_logits, post_labels, post_indices)
        post_record = dict(post_metrics)
        post_record['npz'] = post_npz

        post_cpu = copy.deepcopy(model_gpu).cpu()
        post_cpu.eval()
        del model_gpu
        params_after = count_parameters(post_cpu)
        require(params_after == params_before,
                'Adaptacja BN zmieniła liczbę parametrów')
        require(folding.state_dict_all_finite(post_cpu),
                'Niefinitywny stan modelu po adaptacji BN')
        macs_after = compute_macs(post_cpu)
        require(macs_after['total_macs'] == macs_before['total_macs'],
                'Adaptacja BN zmieniła MACs')
        variant_fold_record = copy.deepcopy(fold_record)
        variant_fold_record['fc_bias'] = fc_bias_info
        variant_fold_record['variant_fc_bias_policy'] = fc_policy
        variant_fold_record['folding_bypassed'] = bool(
            fold_diagnostics['bypassed'])

        model_metadata = {
            'experiment': EXPERIMENT,
            'opis': OPIS_PATH,
            'stage': 'post_bn',
            'ratio': float(ratio),
            'variant': variant,
            'architecture': 'resnet18_cifar10_conv1_3x3_maxpool_identity_fc10',
            'folded': bool(ratio > 0.0),
            'folding_bypassed': bool(fold_diagnostics['bypassed']),
            'fc_bias_policy': fc_policy,
            'seed': seed,
            'bn_seed': int(cfg['bn_seed']),
            'bn_samples': int(cfg['bn_samples']),
            'batch_size': int(cfg['batch_size']),
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': checkpoint_sha256,
            'split': str(split_path),
            'split_sha256': split_sha256,
            'config_sha256': config_sha256,
            'parameter_count': int(params_after),
            'state_dict_shapes': state_dict_shapes(post_cpu),
            'macs': macs_after,
            'fc_bias': fc_bias_info,
            'folding_diagnostics': variant_fold_record,
            'checkpoint_metadata': checkpoint_metadata,
            'official_test_used': False,
        }
        model_path = ratio_dir / f'{variant}_post_bn.pt'
        model_file = save_model_atomic(model_path, post_cpu, model_metadata)
        serialization = verify_saved_model(
            model_path, post_cpu, pre_cpu, seed)
        serialization['sha256'] = model_file['sha256']

        variants[variant] = {
            'fc_bias_policy': fc_policy,
            'fc_bias': fc_bias_info,
            'folding_bypassed': bool(fold_diagnostics['bypassed']),
            'state_vs_folded': state_change,
            'params_before': int(params_before),
            'params_after': int(params_after),
            'macs_before': macs_before,
            'macs_after': macs_after,
            'pre_bn': pre_record,
            'bn': bn_stats,
            'post_bn': post_record,
            'post_bn_model': {
                'file': model_file['file'],
                'sha256': model_file['sha256'],
                'metadata': {
                    'stage': model_metadata['stage'],
                    'ratio': model_metadata['ratio'],
                    'variant': model_metadata['variant'],
                    'architecture': model_metadata['architecture'],
                    'folded': model_metadata['folded'],
                    'fc_bias_policy': model_metadata['fc_bias_policy'],
                    'parameter_count': model_metadata['parameter_count'],
                },
            },
            'serialization': serialization,
            'timing': {
                'pre_bn_seconds': pre_seconds,
                'bn_seconds': bn_seconds,
                'post_bn_seconds': post_seconds,
                'total_seconds': time.monotonic() - variant_started,
            },
        }

    if len(variant_names) > 1:
        first_batch = [
            variants[name]['bn']['first_batch_sha256']
            for name in variant_names
        ]
        require(len(set(first_batch)) == 1,
                'Warianty nie użyły identycznego loadera/augmentacji BN')
        digests = [
            variants[name]['bn']['param_digest_before']
            for name in variant_names
        ]
        require(len(set(digests)) == 2,
                'Warianty powinny różnić się wyłącznie biasem fc')

    finished_utc = utc_now()
    total_seconds = time.monotonic() - started_monotonic
    runtime = {
        'started_utc': started_utc,
        'finished_utc': finished_utc,
        'total_seconds': total_seconds,
        'baseline_seconds': baseline_seconds,
        'fold_seconds': fold_seconds,
        'variants': {
            name: {
                'pre_bn_seconds': variants[name]['timing']['pre_bn_seconds'],
                'bn_seconds': variants[name]['timing']['bn_seconds'],
                'post_bn_seconds': variants[name]['timing']['post_bn_seconds'],
                'total_seconds': variants[name]['timing']['total_seconds'],
                'bn_samples': variants[name]['bn']['samples'],
            }
            for name in variant_names
        },
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        'peak_gpu_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        'max_rss_mb_main': resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        'max_rss_unit': 'MiB (Linux ru_maxrss / 1024)',
        'cpu_threads': int(torch.get_num_threads()),
        'device': str(device),
    }

    param_reduction = baseline_params - folded_params
    macs_reduction = (
        baseline_macs['total_macs'] - folded_macs['total_macs'])
    summary = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'status': 'completed',
        'scope': 'reference_folding_without_finetuning',
        'official_test_used': False,
        'run_dir': str(run_dir),
        'ratio_dir': str(ratio_dir),
        'ratio': float(ratio),
        'ratio_percent': int(round(float(ratio) * 100)),
        'ratios_config': [float(value) for value in cfg['ratios']],
        'config': {
            'path': str(config_path),
            'sha256': config_sha256,
            'values': cfg,
        },
        'input': {
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': checkpoint_sha256,
            'checkpoint_sha256_expected': cfg['checkpoint_sha256'],
            'checkpoint_metadata': checkpoint_metadata,
            'split': str(split_path),
            'split_sha256': split_sha256,
            'split_sha256_expected': cfg['split_sha256'],
            'split_seed': int(split_seed),
            'train_count': int(len(train_indices)),
            'dev_count': int(len(dev_indices)),
            'classes': classes,
            'data_root': str(data_root),
            'image_shape': [int(value) for value in images.shape[1:]],
            'seed': seed,
            'bn_seed': int(cfg['bn_seed']),
            'bn_samples': int(cfg['bn_samples']),
            'batch_size': int(cfg['batch_size']),
            'num_workers': int(cfg['num_workers']),
            'baseline_nll_expected': float(cfg['baseline_nll']),
            'baseline_top1_expected': float(cfg['baseline_top1']),
        },
        'baseline': baseline_record,
        'fold': {
            'ratio': float(ratio),
            'bypassed': bool(fold_diagnostics['bypassed']),
            'params_before': int(baseline_params),
            'params_after': int(folded_params),
            'param_reduction': int(param_reduction),
            'param_reduction_fraction': (
                param_reduction / baseline_params if baseline_params else 0.0),
            'macs_before': baseline_macs,
            'macs_after': folded_macs,
            'macs_reduction': int(macs_reduction),
            'macs_reduction_fraction': (
                macs_reduction / baseline_macs['total_macs']
                if baseline_macs['total_macs'] else 0.0),
            'diagnostics': fold_file,
            'group_policy': group_policy,
        },
        'params': {
            'before': int(baseline_params),
            'after': int(folded_params),
            'reduction': int(param_reduction),
            'reduction_fraction': (
                param_reduction / baseline_params if baseline_params else 0.0),
        },
        'macs': {
            'before': baseline_macs,
            'after': folded_macs,
            'reduction': int(macs_reduction),
            'reduction_fraction': (
                macs_reduction / baseline_macs['total_macs']
                if baseline_macs['total_macs'] else 0.0),
        },
        'variants': variants,
        'runtime': runtime,
        'environment': _environment(args),
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--gpu', type=int, required=True,
                        help='fizyczny indeks GPU z CUDA_VISIBLE_DEVICES')
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    ratio = float(args.ratio)
    ratio_dir = run_dir / ratio_dirname(ratio)
    started_utc = utc_now()
    started_monotonic = time.monotonic()

    try:
        cfg = load_config(args.config.expanduser().resolve())
        require(any(abs(ratio - value) <= 1e-12 for value in cfg['ratios']),
                f'Ratio {ratio} nie występuje w configu: {cfg["ratios"]}')
        require(0.0 <= ratio <= 1.0, 'ratio musi należeć do [0, 1]')
        ratio_dir.mkdir(parents=True, exist_ok=True)
        e011 = load_e011_module()
        summary = run_worker(
            args, cfg, e011, ratio, run_dir, ratio_dir,
            started_utc, started_monotonic)
        write_json_atomic(ratio_dir / 'summary.json', summary)
        failure_path = ratio_dir / 'failure.json'
        if failure_path.exists():
            failure_path.unlink()
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        failure = {
            'experiment': EXPERIMENT,
            'opis': OPIS_PATH,
            'status': 'failed',
            'scope': 'reference_folding_without_finetuning',
            'official_test_used': False,
            'run_dir': str(run_dir),
            'ratio_dir': str(ratio_dir),
            'ratio': ratio,
            'config': str(args.config),
            'error': repr(error),
            'traceback': traceback.format_exc(),
            'started_utc': started_utc,
            'finished_utc': utc_now(),
            'runtime_seconds': time.monotonic() - started_monotonic,
        }
        try:
            ratio_dir.mkdir(parents=True, exist_ok=True)
            summary_path = ratio_dir / 'summary.json'
            if summary_path.exists():
                summary_path.unlink()
            write_json_atomic(ratio_dir / 'failure.json', failure)
        except BaseException:
            pass
        raise


if __name__ == '__main__':
    main()
