"""2026-10-01_E002_obserwacja_se: kontrola miar szablonu na sztucznych mapach.
Wejścia: frozen config i małe syntetyczne tablice, bez realnych aktywacji.
Wyjście: nowy JSON dowodów kontroli wraz z hashami kodu i zależności.
Protokół: OPIS.md obok; kontekst: ../../OPIS.md. Wyłącznie CPU.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from analyze_template import HERE, digest, read_json, write_json, require, code_hashes
from template_metrics import decompose_templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite test evidence')
    cfg = read_json(HERE / 'config.json')
    pair_ids = np.asarray([[0, 1]])
    template = np.asarray([-10., 10.])
    perturb = np.asarray([[1., -1.], [-1., 1.]])
    maps = np.stack([template + perturb, template - perturb], axis=1)[:, :, None, :]
    saved = maps.copy()
    channels, pairs, diagnostics = decompose_templates(maps, pair_ids, cfg)
    require(np.array_equal(saved, maps), 'Input mutation')
    require(np.isclose(pairs['s_raw'][0], 99/101) and np.isclose(pairs['s_dynamic'][0], -1)
            and np.isclose(pairs['s_interaction'][0], -1)
            and np.allclose(channels['frac_template'], 100/101), 'Shared template control failed')
    fixed = np.broadcast_to(template, (4, 2, 1, 2)).copy()
    ch, pa, _ = decompose_templates(fixed, pair_ids, cfg)
    require(np.isnan(pa['s_dynamic'][0]) and np.isnan(pa['s_interaction'][0])
            and np.allclose(ch['frac_template'], 1), 'Zero residual policy failed')
    image_only = np.empty((3, 2, 2, 2))
    image_only[:, 0] = np.asarray([1., 2., 3.])[:, None, None]
    image_only[:, 1] = np.asarray([3., 2., 1.])[:, None, None]
    ch, pa, _ = decompose_templates(image_only, pair_ids, cfg)
    require(np.isnan(pa['s_interaction'][0]) and np.isclose(pa['s_dynamic'][0], -1)
            and np.allclose(ch['frac_interaction'], 0), 'Image-only control failed')
    ch, pa, _ = decompose_templates(np.zeros((3, 2, 2, 2)), pair_ids, cfg)
    require(np.isnan(ch['frac_template']).all() and np.isnan(pa['s_raw']).all(), 'Zero channel policy failed')
    rng = np.random.default_rng(cfg['test_seed'])
    random_maps = rng.normal(size=(7, 4, 2, 3)) + rng.normal(size=(1, 4, 2, 3))
    all_pairs = np.stack(np.triu_indices(4, 1), axis=1)
    ch, pa, diag = decompose_templates(random_maps, all_pairs, cfg)
    total = (random_maps**2).sum(axis=(0, 2, 3))
    mu = random_maps.mean(axis=(0, 2, 3)); t = random_maps.mean(axis=0)
    image = random_maps.mean(axis=(2, 3)) - mu
    spatial = t - mu[:, None, None]
    dynamic = random_maps - t
    interaction = dynamic - image[:, :, None, None]
    direct = {'frac_const': 7*6*mu**2/total, 'frac_image': 6*(image**2).sum(axis=0)/total,
              'frac_spatial_template': 7*(spatial**2).sum(axis=(1, 2))/total,
              'frac_interaction': (interaction**2).sum(axis=(0, 2, 3))/total,
              'frac_template': 7*(t**2).sum(axis=(1, 2))/total,
              'frac_dynamic': (dynamic**2).sum(axis=(0, 2, 3))/total}
    require(all(np.allclose(ch[key], value, atol=1e-12, rtol=1e-12) for key, value in direct.items()),
            'Direct energy formula mismatch')
    for index, (c, d) in enumerate(all_pairs):
        for key, data in (('s_raw', random_maps), ('s_dynamic', dynamic), ('s_interaction', interaction)):
            u, v = data[:, c].ravel(), data[:, d].ravel()
            require(np.isclose(pa[key][index], u@v/(np.linalg.norm(u)*np.linalg.norm(v)),
                               atol=1e-12, rtol=1e-12), 'Direct cosine mismatch')
        u, v = t[c].ravel(), t[d].ravel()
        require(np.isclose(pa['s_template'][index], u@v/(np.linalg.norm(u)*np.linalg.norm(v)),
                           atol=1e-12, rtol=1e-12), 'Template cosine factor-N mismatch')
    near = np.stack([template + 1e-8*perturb, template - 1e-8*perturb], axis=1)[:, :, None, :]
    near_ch, _, _ = decompose_templates(near, pair_ids, cfg)
    require(near_ch['near_zero_dynamic'].all(), 'Near-zero residual flag failed')
    write_json(args.output, {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'device': 'CPU', 'real_activations_used': False, 'test_seed': cfg['test_seed'],
               'checks': ['shared_template_opposite_changes', 'zero_residual', 'image_only', 'zero_channels',
                          'direct_energy_formulas', 'direct_cosines', 'factor_N', 'near_zero_flag', 'input_unchanged'],
               'prototype_raw_cosine': float(pairs['s_raw'][0]),
               'prototype_dynamic_cosine': float(pairs['s_dynamic'][0]),
               'random_max_energy_relative_error': diag['max_energy_relative_error'],
               'config_sha256': digest(HERE / 'config.json'), 'source_sha256': code_hashes()})
    print('TEMPLATE_CPU_CHECKS=passed; no real activations, no model, no GPU')


if __name__ == '__main__':
    main()
