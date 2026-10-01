"""2026-10-01_E002_obserwacja_se: kontrole miar i hooków na sztucznych danych.
Wejścia: kod E002, config.json, małe tablice i model z losowymi wagami na CPU.
Wyjście: nowy JSON dowodów kontroli; nie używa CIFAR/checkpointu ani GPU.
Protokół: PROTOKOL_v1.md; szkic: OPIS.md. To test techniczny, nie wynik hipotezy.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0

from collect import SECapture
from common import HERE, digest, model_digest, read_json, require, write_json
from pair_metrics import compute_pairs, select_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite CPU test evidence')
    cfg = read_json(HERE / 'config.json')
    draft = read_json(HERE / 'config.draft.json')
    require(all(cfg[k] == v for k, v in draft.items() if k not in ('status', 'protocol_version')),
            'Frozen parameters differ from reviewed draft')
    rng = np.random.default_rng(90817)
    h = rng.normal(size=(7, 5, 2, 3)).astype(np.float32)
    a = rng.uniform(.05, .95, size=(7, 5)).astype(np.float32)
    z = a[:, :, None, None] * h
    p, channels, diagnostics = compute_pairs(h, z, a, cfg)
    for index, (c, d) in enumerate(zip(p['c'], p['d'])):
        u, v = h[:, c].astype(np.float64).ravel(), h[:, d].astype(np.float64).ravel()
        q, r = z[:, c].astype(np.float64).ravel(), z[:, d].astype(np.float64).ravel()
        b = a[:, c].astype(np.float64) - a[:, d].astype(np.float64)
        l = np.log(a[:, c].astype(np.float64)) - np.log(a[:, d].astype(np.float64))
        direct = {'s_h': u @ v / (np.linalg.norm(u) * np.linalg.norm(v)),
                  's_z': q @ r / (np.linalg.norm(q) * np.linalg.norm(r)),
                  'd_h': np.linalg.norm(u-v) / (np.linalg.norm(u)+np.linalg.norm(v)),
                  'd_z': np.linalg.norm(q-r) / (np.linalg.norm(q)+np.linalg.norm(r)),
                  'b_mean': b.mean(), 'b_rms': np.sqrt((b*b).mean()), 'b_std': b.std(ddof=0),
                  'l_mean': l.mean(), 'l_std': l.std(ddof=0)}
        direct['delta_s'] = direct['s_z'] - direct['s_h']
        require(all(np.isclose(p[k][index], v, atol=1e-12, rtol=1e-12) for k, v in direct.items()),
                'Gram-based metrics differ from direct formulas')
    h[:, 0] = 0; z[:, 0] = 0
    zero, _, diag_zero = compute_pairs(h, z, a, cfg)
    require(np.isnan(zero['s_h'][:4]).all() and not np.isin(select_pairs(zero, 100), np.arange(4)).any()
            and diag_zero['norm_zero_channels_h'] == 1, 'Zero channel policy failed')
    prototype_h = np.asarray([[[[1, 2]], [[2, 4]]], [[[1, 2]], [[2, 4]]]], dtype=np.float64)
    prototype_a = np.asarray([[.9, .1], [.1, .9]])
    prototype_z = prototype_a[:, :, None, None] * prototype_h
    proto, _, _ = compute_pairs(prototype_h, prototype_z, prototype_a, cfg)
    require(np.isclose(proto['s_h'][0], 1) and np.isclose(proto['s_z'][0], .2195121951219512),
            'Variable rescaling example failed')
    const_a = np.asarray([[.4, .8], [.1, .2]])
    const, _, _ = compute_pairs(prototype_h, const_a[:, :, None, None] * prototype_h, const_a, cfg)
    require(const['l_std'][0] < 1e-7 and np.isclose(const['s_z'][0], 1)
            and np.isclose(const['b_std'][0], .15), 'Constant ratio control failed')
    ties = {'s_h': np.asarray([.7, .7, np.nan]), 'c': np.asarray([2, 0, 1]), 'd': np.asarray([3, 1, 2])}
    require(select_pairs(ties, 100).tolist() == [1, 0], 'Tie order failed')
    torch.set_num_threads(1)
    torch.manual_seed(90817)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(efficientnet_b0(weights=None, num_classes=100))
    model.eval()
    x = torch.randn(2, 3, 96, 96)
    before = model_digest(model)
    with torch.inference_mode():
        plain = model(x)
        capture = SECapture(model.get_submodule(cfg['se_module']))
        try:
            hooked = model(x)
            hh, aa, zz, error = capture.checked(2, cfg)
            require(torch.equal(plain, hooked), 'Hooks changed model output')
            capture.counts['h'] = 2
            rejected = False
            try:
                capture.checked(2, cfg)
            except RuntimeError:
                rejected = True
            require(rejected, 'Duplicate hook invocation was not rejected')
        finally:
            capture.close()
    require(before == model_digest(model), 'CPU hooks changed model state')
    write_json(args.output, {
        'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'CPU', 'real_images_used': False, 'checkpoint_used': False,
        'checks': ['direct_pair_formulas', 'zero_channel_policy', 'variable_scaling',
                   'constant_ratio', 'selection_ties', 'hook_shapes_identity',
                   'hook_output_unchanged', 'duplicate_hook_rejected', 'model_state_unchanged'],
        'prototype_cosine_before': float(proto['s_h'][0]),
        'prototype_cosine_after': float(proto['s_z'][0]),
        'hook_max_identity_error': error,
        'config_sha256': digest(HERE / 'config.json'),
        'source_sha256': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}})
    print(f'CPU_CHECKS=passed; prototype_cosine_after={proto["s_z"][0]}; hook_error={error}')


if __name__ == '__main__':
    main()
