"""2026-10-01_E002_obserwacja_se: testuje rozkład tła na sztucznych mapach.
Wejścia: kod/config tej analizy i małe syntetyczne mapy; bez realnych tensorów.
Wyjścia: nowy TESTY_CPU.json z hashami kodu i konfiguracji.
Protokół: OPIS.md obok; kontekst: ../../OPIS.md. Wyłącznie CPU, bez modelu.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze import HERE, digest, read_json, require, write_json
from metrics import decompose_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite test evidence')
    cfg = read_json(HERE / 'config.json')
    pair_ids = np.asarray([[0, 1]])
    pattern = np.asarray([-1., 1., -1., 1.])
    maps = np.stack([10 + pattern, 10 - pattern])[None, :, None, :].repeat(3, axis=0)
    saved = maps.copy()
    channels, pairs, diagnostics = decompose_maps(maps, pair_ids, cfg)
    require(np.array_equal(maps, saved), 'Input mutated')
    require(np.isclose(pairs['s_raw'][0], 99 / 101) and np.isclose(pairs['s_global'][0], -1)
            and np.isclose(pairs['s_spatial'][0], -1), 'Shared background control failed')
    require(np.allclose(channels['frac_const'], 100/101)
            and np.allclose(channels['frac_spatial'], 1/101)
            and np.allclose(channels['frac_image'], 0), 'Energy fraction control failed')
    constant_maps = np.empty((3, 2, 2, 2))
    constant_maps[:, 0] = np.asarray([1., 2., 3.])[:, None, None]
    constant_maps[:, 1] = np.asarray([3., 2., 1.])[:, None, None]
    _, const_pairs, diag = decompose_maps(constant_maps, pair_ids, cfg)
    require(np.isnan(const_pairs['s_spatial'][0]) and np.isclose(const_pairs['s_global'][0], -1)
            and diag['undefined_spatial_pairs'] == 1, 'Per-image centering policy failed')
    zeros, zero_pairs, zero_diag = decompose_maps(np.zeros((3, 2, 2, 2)), pair_ids, cfg)
    require(np.isnan(zeros['frac_const']).all() and np.isnan(zero_pairs['s_raw']).all()
            and zero_diag['zero_raw_channels'] == 2, 'Zero channel policy failed')
    rng = np.random.default_rng(cfg['test_seed'])
    random_maps = rng.normal(size=(9, 4, 3, 2)) + rng.normal(size=(1, 4, 1, 1))
    random_pairs = np.stack(np.triu_indices(4, 1), axis=1)
    channel, pair, diag_random = decompose_maps(random_maps, random_pairs, cfg)
    for index, (c, d) in enumerate(random_pairs):
        uc, ud = random_maps[:, c].ravel(), random_maps[:, d].ravel()
        gc, gd = uc - uc.mean(), ud - ud.mean()
        sc = (random_maps[:, c] - random_maps[:, c].mean(axis=(1, 2), keepdims=True)).ravel()
        sd = (random_maps[:, d] - random_maps[:, d].mean(axis=(1, 2), keepdims=True)).ravel()
        for name, v, w in (('s_raw', uc, ud), ('s_global', gc, gd), ('s_spatial', sc, sd)):
            require(np.isclose(pair[name][index], np.dot(v, w)/(np.linalg.norm(v)*np.linalg.norm(w)),
                               atol=1e-12, rtol=1e-12), 'Direct cosine comparison failed')
    require(np.allclose(channel['frac_const'] + channel['frac_image'] + channel['frac_spatial'], 1,
                        atol=1e-12), 'Orthogonal energy sum failed')
    near = np.stack([1 + 1e-8 * pattern, 1 - 1e-8 * pattern])[None, :, None, :]
    near_channels, _, _ = decompose_maps(near, pair_ids, cfg)
    require(near_channels['near_zero_global'].all(), 'Near-constant flag failed')
    write_json(args.output, {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'device': 'CPU', 'real_activations_used': False, 'test_seed': cfg['test_seed'],
               'checks': ['shared_background_opposite_patterns', 'orthogonal_energy', 'direct_cosines',
                          'constant_spatial_maps', 'zero_channels', 'near_constant_flags', 'input_unchanged'],
               'prototype_raw_cosine': float(pairs['s_raw'][0]),
               'prototype_centered_cosine': float(pairs['s_global'][0]),
               'random_max_energy_relative_error': diag_random['max_energy_relative_error'],
               'config_sha256': digest(HERE / 'config.json'),
               'source_sha256': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}})
    print('CONSTANT_CPU_CHECKS=passed; prototype raw=0.980198 centered=-1; no real data used')


if __name__ == '__main__':
    main()
