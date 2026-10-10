"""E003: kontrole CPU operatora foldingu na modelu losowym, bez GPU i danych.

Eksperyment: 2026-10-09_E003_kompresja_10proc_parametrow.
Cel: dowiezc kontrolowany dowod techniczny, ze folding.py buduje poprawny graf
o 3721424 parametrach, zachowuje niezmieniony model bazowy, odtwarza sciezke
MBConv przy zduplikowanych kanalach, ratuje proporcjonalny podsystem SE/proj,
przezywa serializacje (materialize + strict load bez refoldingu) i odrzuca
powtorzenie foldingu oraz nachodzace pary.
Wejscia: folding.py, PLAN_STRUKTURY.json, losowe wagi efficientnet_b0 (CPU).
Wyjscia: nowy plik JSON dowodow (odmawia nadpisania). Seed 20261010, jeden
watek CPU, bez checkpointu, bez realnych obrazow, bez GPU.
Dokumenty: OPIS.md oraz PROTOKOL_v1.md w tym samym folderze eksperymentu.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import MBConv

import folding

HERE = Path(__file__).resolve().parent
SEED = 20261010
PLAN_SHA256 = 'fdfa35242201b9544afa47838a83f1683aa35482f9164555910b90ed225cab74'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def random_budget_groups(rows):
    groups = {}
    for row in rows:
        pairs = []
        for index in range(int(row['remove_count'])):
            pairs.append({'c': 2 * index, 'd': 2 * index + 1, 'alpha': 0.5, 'beta': 0.75})
        groups[row['block']] = pairs
    return groups


def check_full_budget(base, rows, expected_compressed):
    groups = random_budget_groups(rows)
    folded = folding.fold_model(base, rows, groups)
    require(folding.count_parameters(folded) == expected_compressed,
            'Full-budget folded parameter count mismatch.')
    delta_total = sum(int(row['remove_count']) * folding.per_removed_channel(
        base.get_submodule(row['block'])) for row in rows)
    require(folding.BASE_PARAMETERS - delta_total == expected_compressed,
            'Per-channel cost formula does not reproduce the planned budget.')
    folded.eval()
    max_abs = 0.0
    with torch.inference_mode():
        for _ in range(100):
            output = folded(torch.randn(1, 3, 96, 96))
            require(output.shape == (1, 100), 'Unexpected classifier output shape.')
            require(bool(torch.isfinite(output).all()), 'Non-finite forward output.')
            max_abs = max(max_abs, float(output.abs().max()))
    attrs_ok = True
    for row in rows:
        path = row['block']
        src = base.get_submodule(path)
        dst = folded.get_submodule(path)
        src_conv, src_dw, src_proj = src.block[0][0], src.block[1][0], src.block[3][0]
        dst_conv, dst_dw, dst_proj = dst.block[0][0], dst.block[1][0], dst.block[3][0]
        for a, b in ((src_conv, dst_conv), (src_dw, dst_dw), (src_proj, dst_proj)):
            require(a.stride == b.stride and a.padding == b.padding and a.kernel_size == b.kernel_size,
                    f'Conv geometry changed in {path}.')
            require((a.bias is None) == (b.bias is None), f'Conv bias flag changed in {path}.')
            require(a.weight.dtype == b.weight.dtype and a.weight.device == b.weight.device,
                    f'Conv dtype/device changed in {path}.')
        for a, b in ((src.block[0][1], dst.block[0][1]), (src.block[1][1], dst.block[1][1]),
                     (src.block[3][1], dst.block[3][1])):
            require(type(a) is type(b), f'BN class changed in {path}.')
            require(a.eps == b.eps and a.momentum == b.momentum
                    and a.track_running_stats == b.track_running_stats,
                    f'BN hyperparameters changed in {path}.')
        require(dst.block[1][0].groups == dst.block[1][0].out_channels,
                f'Depthwise groups not updated in {path}.')
        attrs_ok = attrs_ok and True
    return {'compressed_parameters': folding.count_parameters(folded),
            'removed_parameters': folding.BASE_PARAMETERS - folding.count_parameters(folded),
            'forward_passes': 100, 'forward_max_abs': max_abs,
            'attributes_preserved': attrs_ok}, folded, groups


def duplicate_channel(module, src, dst):
    for name in ('weight', 'bias', 'running_mean', 'running_var'):
        tensor = getattr(module, name, None)
        if tensor is not None:
            tensor[dst] = tensor[src].clone()


def check_duplicate_block(base):
    block = copy.deepcopy(base.get_submodule('features.4.1')).double().eval()
    c, d = 4, 418
    with torch.no_grad():
        expansion, depthwise, se, _ = block.block
        duplicate_channel(expansion[0], d, c)
        duplicate_channel(expansion[1], d, c)
        duplicate_channel(depthwise[0], d, c)
        duplicate_channel(depthwise[1], d, c)
        se.fc2.weight[c] = se.fc2.weight[d].clone()
        se.fc2.bias[c] = se.fc2.bias[d].clone()
    x = torch.randn(2, 80, 6, 6, dtype=torch.float64)
    with torch.no_grad():
        before = block(x).clone()
    keep = [i for i in range(expansion[0].out_channels) if i != d]
    folding._transform_block(block, keep, [(c, d, 1.0, 1.0)])
    with torch.no_grad():
        after = block(x)
    error = float((before - after).abs().max())
    require(torch.allclose(before, after, atol=1e-10, rtol=1e-10),
            'Duplicated-path full MBConv forward was not preserved.')
    return {'c': c, 'd': d, 'input_shape': [2, 80, 6, 6], 'max_abs_error': error}


def check_proportional_subsystem(base):
    block = copy.deepcopy(base.get_submodule('features.4.1')).double().eval()
    c, d = 4, 418
    se, proj = block.block[2], block.block[3]
    with torch.no_grad():
        se.fc2.weight.zero_()
        se.fc2.bias.zero_()
        se.fc2.bias[c] = math.log(0.25 / 0.75)
        se.fc2.bias[d] = 0.0
    channels = se.fc1.in_channels
    h = torch.randn(2, channels, 6, 6, dtype=torch.float64)
    h[:, d] = 2.0 * h[:, c]
    with torch.no_grad():
        g = se.avgpool(h)
        z = se(h)
        before = proj(z).clone()
    alpha = float((g[:, c] * g[:, d]).sum() / (g[:, c] * g[:, c]).sum())
    beta = float((z[:, c] * z[:, d]).sum() / (z[:, c] * z[:, c]).sum())
    require(abs(alpha - 2.0) < 1e-12 and abs(beta - 4.0) < 1e-12,
            'Proportional subsystem coefficients are not alpha=2, beta=4.')
    keep = [i for i in range(channels) if i != d]
    folding._transform_block(block, keep, [(c, d, alpha, beta)])
    with torch.no_grad():
        after = proj(se(h[:, keep]))
    error = float((before - after).abs().max())
    require(torch.allclose(before, after, atol=1e-10, rtol=1e-10),
            'Proportional SE/projection subsystem was not preserved.')
    return {'c': c, 'd': d, 'alpha': alpha, 'beta': beta, 'max_abs_error': error}


def check_serialization(base, rows, folded):
    materialized = folding.materialize_model(base, rows)
    require(folding.count_parameters(materialized) == folding.count_parameters(folded),
            'Materialized graph parameter count mismatch.')
    require(model_digest(materialized) != model_digest(folded),
            'Materialized graph should differ before weight loading.')
    compact_state = folded.state_dict()
    materialized.load_state_dict(compact_state, strict=True)
    require(model_digest(materialized) == model_digest(folded),
            'Strict load did not reproduce the folded state dict.')
    x = torch.randn(2, 3, 96, 96)
    folded.eval()
    materialized.eval()
    with torch.inference_mode():
        a = folded(x)
        b = materialized(x)
    require(torch.equal(a, b), 'Materialized forward differs after strict load.')
    return {'strict_load': True, 'state_dict_hash_matches': True,
            'prediction_identical': bool(torch.equal(a.argmax(1), b.argmax(1)))}


def check_partial_row(base):
    """Obsluga czesciowego planu: jeden blok, jedna para, szerokosc 479."""
    path = 'features.4.1'
    c, d = 4, 418
    rows = [{'block': path, 'new_channels': 479, 'remove_count': 1}]
    groups = {path: [{'c': c, 'd': d, 'alpha': 0.25, 'beta': 0.5}]}
    delta = folding.per_removed_channel(base.get_submodule(path))
    folded = folding.fold_model(base, rows, groups)
    require(folding.count_parameters(folded) == folding.BASE_PARAMETERS - delta,
            'Partial-row fold has wrong parameter count.')
    folded.eval()
    with torch.inference_mode():
        output = folded(torch.randn(1, 3, 96, 96))
    require(output.shape == (1, 100) and bool(torch.isfinite(output).all()),
            'Partial-row forward is not finite.')
    materialized = folding.materialize_model(base, rows)
    materialized.load_state_dict(folded.state_dict(), strict=True)
    require(model_digest(materialized) == model_digest(folded),
            'Partial-row materialize/strict-load mismatch.')
    return {'block': path, 'base_width': 480, 'new_channels': 479,
            'parameters': folding.count_parameters(folded)}


def check_rejections(base, rows, folded, groups):
    rejected = {}
    try:
        folding.fold_model(folded, rows, groups)
    except RuntimeError:
        rejected['double_fold'] = True
    require(rejected.get('double_fold'), 'Double folding was not rejected.')
    flagged = copy.deepcopy(base)
    flagged.compression_applied = True
    try:
        folding.fold_model(flagged, rows, groups)
    except RuntimeError:
        rejected['already_compressed_flag'] = True
    require(rejected.get('already_compressed_flag'), 'compression_applied input was not rejected.')
    bad_rows = [{'block': 'features.2.0', 'new_channels': 94}]
    bad_groups = {'features.2.0': [
        {'c': 0, 'd': 4, 'alpha': 0.5, 'beta': 0.5},
        {'c': 2, 'd': 4, 'alpha': 0.5, 'beta': 0.5}]}
    try:
        folding.fold_model(base, bad_rows, bad_groups)
    except RuntimeError:
        rejected['overlapping_pairs'] = True
    require(rejected.get('overlapping_pairs'), 'Overlapping pairs were not rejected.')
    short_rows = [{'block': 'features.2.0', 'new_channels': 94}]
    short_groups = {'features.2.0': [{'c': 0, 'd': 4, 'alpha': 0.5, 'beta': 0.5}]}
    try:
        folding.fold_model(base, short_rows, short_groups)
    except RuntimeError:
        rejected['pair_count_mismatch'] = True
    require(rejected.get('pair_count_mismatch'), 'Pair-count mismatch was not rejected.')
    return rejected


def check_syncbn(base):
    sync = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(base))
    block = sync.get_submodule('features.2.0')
    folding._transform_block(block, [i for i in range(96) if i != 1], [(0, 1, 0.5, 0.75)])
    classes = {type(module).__name__ for module in block.modules()
               if 'BatchNorm' in type(module).__name__}
    require(classes == {'SyncBatchNorm'}, 'SyncBatchNorm class was not preserved.')
    expansion, depthwise, se, _ = block.block
    require(expansion[0].out_channels == 95 and depthwise[0].groups == 95
            and se.fc1.in_channels == 95 and se.fc2.out_channels == 95
            and expansion[1].num_features == 95 and depthwise[1].num_features == 95,
            'SyncBatchNorm width update failed.')
    return {'classes': sorted(classes), 'channels': 95}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite CPU folding evidence.')

    torch.set_num_threads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    require(digest(HERE / 'PLAN_STRUKTURY.json') == PLAN_SHA256,
            'Structure plan hash differs from the frozen E003 plan.')
    plan = json.loads((HERE / 'PLAN_STRUKTURY.json').read_text())
    rows = plan['blocks']
    require(plan['base_parameters'] == folding.BASE_PARAMETERS
            and plan['expected_compressed_parameters'] == folding.EXPECTED_COMPRESSED_PARAMETERS,
            'Plan budget does not match folding.py constants.')

    base = efficientnet_b0(weights=None, num_classes=100)
    base_digest_before = model_digest(base)
    require(folding.count_parameters(base) == folding.BASE_PARAMETERS,
            'Fresh model is not the 4135648-parameter E003 base.')

    budget, folded, groups = check_full_budget(base, rows, plan['expected_compressed_parameters'])
    duplicate = check_duplicate_block(base)
    proportional = check_proportional_subsystem(base)
    serialization = check_serialization(base, rows, folded)
    partial_row = check_partial_row(base)
    rejections = check_rejections(base, rows, folded, groups)
    syncbn = check_syncbn(base)

    require(model_digest(base) == base_digest_before,
            'Base model state changed during the CPU checks.')
    require(folding.count_parameters(base) == folding.BASE_PARAMETERS,
            'Base model parameter count changed.')

    evidence = {
        'status': 'passed',
        'experiment': '2026-10-09_E003_kompresja_10proc_parametrow',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'CPU',
        'seed': SEED,
        'torch_num_threads': 1,
        'real_images_used': False,
        'checkpoint_used': False,
        'gpu_used': False,
        'checks': [
            'full_budget_3721424_parameters', 'forward_100_finite', 'base_state_unchanged',
            'base_conv_bn_attributes_preserved', 'duplicated_path_block_forward',
            'proportional_subsystem_alpha2_beta4', 'materialize_strict_load_no_refold',
            'partial_row_width_479',
            'double_fold_rejected', 'already_compressed_rejected',
            'overlapping_pairs_rejected', 'pair_count_mismatch_rejected',
            'sync_batchnorm_preserved',
        ],
        'base_parameters': folding.BASE_PARAMETERS,
        'expected_compressed_parameters': plan['expected_compressed_parameters'],
        'base_state_unchanged': True,
        'full_budget': budget,
        'duplicate_block': duplicate,
        'proportional_subsystem': proportional,
        'serialization': serialization,
        'partial_row': partial_row,
        'rejections': rejections,
        'sync_batchnorm': syncbn,
        'versions': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torchvision': __import__('torchvision').__version__,
            'numpy': np.__version__,
        },
        'plan_sha256': PLAN_SHA256,
        'source_sha256': {
            'folding.py': digest(HERE / 'folding.py'),
            'check_folding_cpu.py': digest(HERE / 'check_folding_cpu.py'),
        },
    }
    with args.output.open('x') as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(f"CPU_FOLDING_CHECKS=passed; base={folding.BASE_PARAMETERS}; "
          f"compressed={budget['compressed_parameters']}; "
          f"duplicate_error={duplicate['max_abs_error']:.3e}; "
          f"subsystem_error={proportional['max_abs_error']:.3e}")


if __name__ == '__main__':
    main()
