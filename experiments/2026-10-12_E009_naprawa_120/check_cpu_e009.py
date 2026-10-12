"""E009: kontrole CPU przed zamrozeniem (pochodzenie wejsc, integralnosc modeli).
Eksperyment: 2026-10-12_E009_naprawa_120; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: config.draft.json oraz katalog etapu 1 biegu E008 (tylko odczyt).
Wyjscia: TESTY_CPU_v1.json. Bez GPU i bez treningu; jeden maly forward kontrolny.
"""
import json

import torch

from support_e009 import (HERE, digest, read_json, write_json, require, model_digest,
                          base_model, e008_run)
from folding_e008 import materialize_model, count_parameters


def main():
    cfg = read_json(HERE / 'config.draft.json')
    stage1 = e008_run(cfg) / 'stage1'
    for name, expected in cfg['input']['sha256'].items():
        require((stage1 / name).exists(), f'Missing input: {name}')
        require(digest(stage1 / name) == expected, f'Input changed: {name}')
    metrics = read_json(stage1 / 'stage1_metrics.json')
    require(metrics['baseline_reproduced'], 'Stage 1 baseline not reproduced')
    model_files = metrics['model_files']
    cpu_base = base_model(cfg)
    nums = {'base_parameters': count_parameters(cpu_base)}
    for name in cfg['repair']['variants']:
        key = 'BASE' if name == 'BASE' else f'30_{name}'
        payload = torch.load(stage1 / 'models' / f'{key}.pt', map_location='cpu',
                             weights_only=True)
        require(payload['variant'] == key, f'Variant mismatch: {key}')
        if name == 'BASE':
            require(payload['compression_applied'] is False, 'BASE flagged compressed')
            model = cpu_base
        else:
            require(payload['compression_applied'] is True, f'{key} not flagged compressed')
            model = materialize_model(cpu_base, payload['structure_rows'])
            model.load_state_dict(payload['state_dict'], strict=True)
        require(count_parameters(model) == payload['parameters'], f'Parameter count: {key}')
        require(model_digest(model) == model_files[key]['model_sha256'],
                f'Digest mismatch vs stage1 metrics: {key}')
        require(model_digest(model) == payload['model_sha256'], f'Payload digest: {key}')
        nums[f'{key}_parameters'] = payload['parameters']
        if key == '30_PRE':
            model.eval()
            with torch.no_grad():
                output = model(torch.randn(2, 3, cfg['image_size'], cfg['image_size']))
            require(tuple(output.shape) == (2, cfg['num_classes'])
                    and bool(torch.isfinite(output).all()), 'Forward check failed')
    files = read_json(stage1 / 'model_files.json')
    require(files['BASE']['model_sha256'] == model_files['BASE']['model_sha256'],
            'model_files cross-check')
    write_json(HERE / 'TESTY_CPU_v1.json', {'experiment': cfg['experiment'],
               'status': 'passed', 'numbers': nums})
    print('E009_CPU_CHECKS=passed')
    print(json.dumps(nums, sort_keys=True))


if __name__ == '__main__':
    main()
