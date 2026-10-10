"""E004: kontrole kodu przed treningiem, na CPU, bez GPU i bez obrazów CIFAR.
Eksperyment: 2026-10-10_E004_wspolne_dostrajanie; PROTOKOL_v1.md i OPIS.md.
Wejścia: kod E004, frozen config, modele E003 (odczyt i hash, bez zmiany).
Wyjście: nowy TESTY_CPU_v1.json z hashami kodu/configu i listą kontroli.
Sprawdza harmonogram LR, wczytywanie modeli, forward/backward na sztucznych danych.
"""
import argparse
import ast
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from support_e004 import HERE, ROOT, E003, E002, digest, read_json, write_json, require, code_hashes

sys.path.insert(0, str(E003))
sys.path.insert(0, str(E002))
from common import model_digest
from support import base_model
from folding import materialize_model, count_parameters
from train_finetune import lr_value, build_variant_model
from summarize import paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite evidence')
    for path in sorted(HERE.glob('*.py')):
        ast.parse(path.read_text())
    cfg = read_json(HERE / 'config.json')
    draft = read_json(HERE / 'config.draft.json')
    require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.0', 'Not frozen')
    require(all(cfg[k] == v for k, v in draft.items() if k not in ('status', 'protocol_version')),
            'Frozen parameters differ from reviewed draft')
    tune = cfg['fine_tune']
    require(lr_value(0, tune) == 0.01 and lr_value(59, tune) == 0.01
            and lr_value(60, tune) == 0.001 and lr_value(89, tune) == 0.001
            and lr_value(90, tune) == 0.0001 and lr_value(119, tune) == 0.0001,
            'LR schedule mismatch')
    require(tune['epochs'] == 120 and tune['batch_per_gpu'] * cfg['world_size'] == 256,
            'Budget mismatch')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    base = base_model(cfg)
    checks = []
    with torch.inference_mode():
        for variant in tune['variants_sequential_order']:
            model = build_variant_model(cfg, base, variant)
            expected = cfg['base_parameters'] if variant == 'BASE' else cfg['compressed_parameters']
            require(count_parameters(model) == expected, f'{variant}: parameter mismatch')
            x = torch.randn(2, 3, cfg['image_size'], cfg['image_size'])
            logits = model.eval()(x)
            require(logits.shape == (2, 100) and bool(torch.isfinite(logits).all()),
                    f'{variant}: forward invalid')
            stored = torch.load(ROOT / cfg['source_model_files'][variant]['path'],
                                map_location='cpu', weights_only=True)
            require(model_digest(model) == stored['model_sha256'],
                    f'{variant}: digest differs from E003 record')
            checks.append(f'load_{variant}_strict_params_forward_digest')
    smoke = build_variant_model(cfg, base, 'POST')
    smoke.train()
    optimizer = torch.optim.SGD(smoke.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    before = model_digest(smoke)
    x = torch.randn(4, 3, cfg['image_size'], cfg['image_size'])
    y = torch.randint(0, 100, (4,))
    loss = nn.functional.cross_entropy(smoke(x), y)
    require(bool(torch.isfinite(loss)), 'Training smoke loss not finite')
    loss.backward()
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in smoke.parameters()),
            'Non-finite gradient in smoke')
    optimizer.step()
    require(model_digest(smoke) != before, 'Optimizer step did not change weights')
    checks.append('training_smoke_backward_step')
    fresh_base = base_model(cfg)
    require(model_digest(base) == model_digest(fresh_base), 'Base model mutated by checks')
    checks.append('base_state_unchanged')
    labels = np.repeat(np.arange(100), 50)
    flags = np.ones(5000, dtype=bool)
    same = paired_bootstrap(flags, flags, labels, 50, cfg['bootstrap_seed'])
    require(same['delta_correct'] == 0 and same['ci95_correct'] == [0, 0], 'Bootstrap mismatch')
    checks.append('bootstrap_identity')
    write_json(args.output, {
        'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'CPU', 'real_images_used': False, 'training_performed': False,
        'seed': cfg['seed'], 'checks': checks,
        'config_sha256': digest(HERE / 'config.json'),
        'source_sha256': code_hashes(),
        'versions': {'python': __import__('platform').python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__}})
    print('E004_CPU_CHECKS=passed; code and model loading verified on synthetic CPU data; '
          'actual fine-tuning runs later on 4 GPUs')


if __name__ == '__main__':
    main()
