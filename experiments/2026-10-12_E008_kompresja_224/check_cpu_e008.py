"""E008: kontrole CPU przed zamrozeniem (plany, folding, selekcja, dane, baseline).
Eksperyment: 2026-10-12_E008_kompresja_224; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: config.draft.json, plany struktury, baseline E007, cache/splits Tiny ImageNet.
Wyjscia: TESTY_CPU_v1.json z wynikami; bez GPU, bez treningu i bez oceny dev.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from support_e008 import (HERE, ROOT, DATA, digest, read_json, write_json, require,
                          load_data, base_model)
from folding_e008 import fold_model, materialize_model, count_parameters, BASE_PARAMETERS
from selection import choose_groups, selection_settings


def random_groups(rows, rng):
    groups = {}
    for row in rows:
        channels = row['expanded_channels']
        remove = row['remove_count']
        order = rng.permutation(channels)
        chosen = order[:2 * remove]
        pairs = []
        for i in range(remove):
            c, d = int(chosen[2 * i]), int(chosen[2 * i + 1])
            if c > d:
                c, d = d, c
            pairs.append({'c': c, 'd': d,
                          'alpha': float(rng.uniform(-0.5, 0.5)),
                          'beta': float(rng.uniform(-0.5, 0.5))})
        groups[row['block']] = pairs
    return groups


def synthetic_moments(rows, rng):
    moments = {}
    for row in rows:
        channels = row['expanded_channels']
        base = rng.standard_normal((channels, 8))
        matrix = base @ base.T + np.eye(channels) * 0.1
        moments[row['block']] = {key: matrix.copy() for key in ('h', 'z', 'g')}
    return moments


def main():
    cfg = read_json(HERE / 'config.draft.json')
    nums = {}
    plans = {b: read_json(HERE / cfg['plans'][b]['file']) for b in cfg['budgets']}
    for budget, plan in plans.items():
        require(plan['base_parameters'] == BASE_PARAMETERS, f'{budget}: base parameters')
        require(plan['expected_compressed_parameters'] == cfg['plans'][budget]['compressed_parameters'],
                f'{budget}: plan parameters vs config')
        for row in plan['blocks']:
            require(row['new_channels'] % 8 == 0 and 0 < row['new_channels'] <= row['expanded_channels'],
                    f'{budget}: block width')
            require(row['remove_count'] * row['parameters_per_removed_channel'] == row['saved_parameters'],
                    f'{budget}: saved parameters formula')
            require(row['expanded_channels'] - row['new_channels'] == row['remove_count'],
                    f'{budget}: removed channel count')
        nums[f'plan_{budget}_compressed'] = plan['expected_compressed_parameters']
        nums[f'plan_{budget}_removed_channels'] = plan['removed_channels']

    rng = np.random.default_rng(20261015)
    model = base_model(cfg)
    require(count_parameters(model) == BASE_PARAMETERS, 'Base parameter count')
    base_digest = hashlib.sha256(
        b''.join(t.detach().numpy().tobytes() for _, t in sorted(model.state_dict().items()))
    ).hexdigest()
    x = torch.randn(2, 3, cfg['image_size'], cfg['image_size'], dtype=torch.float32)

    for budget in cfg['budgets']:
        rows = plans[budget]['blocks']
        groups = random_groups(rows, rng)
        folded = fold_model(model, rows, groups)
        require(count_parameters(folded) == cfg['plans'][budget]['compressed_parameters'],
                f'{budget}: folded parameter count')
        folded.eval()
        with torch.no_grad():
            output = folded(x)
        require(tuple(output.shape) == (2, cfg['num_classes']) and bool(torch.isfinite(output).all()),
                f'{budget}: folded forward')
        materialized = materialize_model(model, rows)
        materialized.load_state_dict(folded.state_dict(), strict=True)
        require(count_parameters(materialized) == cfg['plans'][budget]['compressed_parameters'],
                f'{budget}: materialized parameter count')
        folded_bytes = b''.join(t.detach().numpy().tobytes()
                                for _, t in sorted(folded.state_dict().items()))
        materialized_bytes = b''.join(t.detach().numpy().tobytes()
                                      for _, t in sorted(materialized.state_dict().items()))
        require(folded_bytes == materialized_bytes, f'{budget}: strict reload mismatch')
        nums[f'folded_{budget}_parameters'] = count_parameters(folded)
        del folded, materialized

    model_bytes = b''.join(t.detach().numpy().tobytes()
                           for _, t in sorted(model.state_dict().items()))
    require(hashlib.sha256(model_bytes).hexdigest() == base_digest, 'Base altered by folding')
    nums['base_parameters'] = count_parameters(model)

    for budget in cfg['budgets']:
        rows = plans[budget]['blocks']
        moments = synthetic_moments(rows, rng)
        groups = choose_groups(moments, rows, cfg)
        for mode in cfg['modes']:
            total = 0
            for row in rows:
                entries = groups[mode][row['block']]
                require(len(entries) == row['remove_count'], f'{budget}/{mode}: pair count')
                used = set()
                for entry in entries:
                    c, d = int(entry['c']), int(entry['d'])
                    require(0 <= c < d < row['expanded_channels'], f'{budget}/{mode}: pair range')
                    require(c not in used and d not in used, f'{budget}/{mode}: pair overlap')
                    require(np.isfinite(entry['alpha']) and np.isfinite(entry['beta']),
                            f'{budget}/{mode}: coefficient')
                    used.add(c)
                    used.add(d)
                total += len(entries)
            nums[f'selection_{budget}_{mode}_pairs'] = total
        settings = selection_settings(cfg)
        nums[f'selection_{budget}_settings_seed'] = settings.get(
            'random_selection_seed', cfg['random_selection_seed'])

    images, labels, records = load_data(cfg)
    require(len(records['A']) == 1000 and len(records['B']) == 1000
            and len(records['dev']) == 10000, 'Calibration/dev sizes')
    a_ids = {r['id'] for r in records['A']}
    b_ids = {r['id'] for r in records['B']}
    require(not (a_ids & b_ids), 'A/B overlap')
    require(set(r['target'] for r in records['A']) == set(range(200)), 'A lacks classes')
    nums['calibration_A'] = len(records['A'])
    nums['calibration_B'] = len(records['B'])
    nums['dev'] = len(records['dev'])
    nums['cache_train_sha256'] = digest(DATA / 'cache_train.npz')
    nums['cache_val_sha256'] = digest(DATA / 'cache_val.npz')
    nums['splits_sha256'] = digest(DATA / 'splits.json')
    require(nums['cache_train_sha256'] == cfg['data']['cache_train_sha256'], 'cache_train hash')
    require(nums['cache_val_sha256'] == cfg['data']['cache_val_sha256'], 'cache_val hash')
    require(nums['splits_sha256'] == cfg['data']['splits_sha256'], 'splits hash')

    write_json(HERE / 'TESTY_CPU_v1.json', {'experiment': cfg['experiment'],
               'status': 'passed', 'numbers': nums})
    print('E008_CPU_CHECKS=passed')
    print(json.dumps(nums, sort_keys=True))


if __name__ == '__main__':
    main()
