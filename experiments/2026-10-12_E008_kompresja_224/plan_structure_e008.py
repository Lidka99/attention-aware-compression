"""E008: wyznacza wspolny budzet redukcji parametrow calej sieci ze struktury.
Eksperyment: 2026-10-12_E008_kompresja_224; protokol: OPIS.md.
Wejscia: zainstalowany torchvision EfficientNet-B0, weights=None, 200 klas.
Wyjscie: nowy PLAN_STRUKTURY_<P>proc.json (P = --reduction-percent, np. 10, 30).
CPU, bez obrazow, checkpointow i forwardu; plan zalezy tylko od architektury.
To zapis planu, nie wykonanie kompresji wytrenowanego modelu.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import torch
import torchvision
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import MBConv

EXPERIMENT = '2026-10-12_E008_kompresja_224'
EXPECTED_BASE_PARAMETERS = 4263748


def plan(model, reduction_percent):
    base_params = sum(p.numel() for p in model.parameters())
    if base_params != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(
            f'Base model has {base_params} parameters, expected {EXPECTED_BASE_PARAMETERS}.')
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, MBConv) or len(module.block) != 4:
            continue
        expansion, depthwise, se, projection = module.block
        conv, dw, proj = expansion[0], depthwise[0], projection[0]
        channels = conv.out_channels
        if channels % 8 or dw.groups != channels:
            raise RuntimeError('Unsupported expansion structure')
        bn_params = sum(int(getattr(bn, attr) is not None)
                        for bn in (expansion[1], depthwise[1]) for attr in ('weight', 'bias'))
        delta = (conv.weight[0].numel() + int(conv.bias is not None)
                 + dw.weight[0].numel() + int(dw.bias is not None) + bn_params
                 + se.fc1.weight[:, 0].numel() + se.fc2.weight[0].numel()
                 + int(se.fc2.bias is not None) + proj.weight[:, 0].numel())
        rows.append({'block': name, 'in_channels': conv.in_channels, 'out_channels': proj.out_channels,
                     'expanded_channels': channels, 'squeeze_channels': se.fc1.out_channels,
                     'kernel': list(dw.kernel_size), 'stride': list(dw.stride),
                     'parameters_per_removed_channel': int(delta), 'remove_count': 0})
    target = (base_params * int(reduction_percent) + 99) // 100
    saved = 0
    steps = []
    while saved < target:
        eligible = [row for row in rows if row['remove_count'] + 8 <= row['expanded_channels'] // 2]
        if not eligible:
            raise RuntimeError('Target cannot be met with disjoint-pair budget')
        chosen = min(eligible, key=lambda row: (
            Fraction(row['remove_count'] + 8, row['expanded_channels']), row['block']))
        chosen['remove_count'] += 8
        saved += 8 * chosen['parameters_per_removed_channel']
        steps.append({'block': chosen['block'], 'remove_count': chosen['remove_count'], 'saved_total': saved})
    for row in rows:
        row['new_channels'] = row['expanded_channels'] - row['remove_count']
        row['saved_parameters'] = row['remove_count'] * row['parameters_per_removed_channel']
    return {'base_parameters': base_params, 'reduction_percent': int(reduction_percent),
            'target_removed_parameters': target,
            'expected_removed_parameters': saved, 'expected_compressed_parameters': base_params - saved,
            'expected_reduction_percent': 100 * saved / base_params,
            'removed_channels': sum(row['remove_count'] for row in rows), 'blocks': rows,
            'allocation_steps': len(steps), 'last_step': steps[-1],
            'allocation_uses_activations_or_quality': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reduction-percent', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.reduction_percent <= 50:
        raise RuntimeError('reduction-percent must be in [1, 50]')
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite existing structure plan')
    torch.set_num_threads(1)
    model = efficientnet_b0(weights=None, num_classes=200)
    result = plan(model, args.reduction_percent)
    result.update(experiment=EXPERIMENT,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  status='structure_plan_only', trained_model_compressed=False, forward_performed=False,
                  device='CPU', python=platform.python_version(), torch=torch.__version__,
                  torchvision=torchvision.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(f"STRUCTURE_PLAN[{args.reduction_percent}%]: {result['base_parameters']} -> "
          f"{result['expected_compressed_parameters']} parameters "
          f"({result['expected_reduction_percent']:.4f}%); "
          f"{result['removed_channels']} removed channels in {len(result['blocks'])} blocks")
    print('Plan architecture only; trained model has not been compressed or evaluated.')


if __name__ == '__main__':
    main()
