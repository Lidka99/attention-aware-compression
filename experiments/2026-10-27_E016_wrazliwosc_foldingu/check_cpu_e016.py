"""E016: preflight CPU etapu S1 bez użycia GPU.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu; pełny opis: OPIS.md
(``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS.md``).

Sprawdza bez GPU: config, hashe źródeł i checkpointów, splity, 98 itemów,
shardy, determinizm kalibracji, arytmetykę predyktorów, kontrolę ratio 0
oraz jeden rzeczywisty folding pojedynczej grupy (hkmeans) na CPU. Nie
uruchamia treningu ani nie używa dev do strojenia. Kod wyjścia 0 oznacza
sukces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402


def check_forbidden_imports():
    """Sprawdza, że pliki E016 nie importują attention_v2/attention_v3."""
    for path in sorted(HERE.glob('*.py')):
        text = path.read_text(encoding='utf-8')
        for forbidden in ('attention_v2', 'attentionv3', 'attention_v3'):
            common.require(
                f'import {forbidden}' not in text
                and f'from {forbidden}' not in text,
                f'{path.name} importuje zabroniony moduł {forbidden}')
    print('OK: brak importów attention_v2/attention_v3')


def check_items(cfg):
    """Sprawdza liczbę, unikalność i shardy itemów S1 oraz smoke."""
    items = common.build_items(cfg, smoke=False)
    smoke = common.build_items(cfg, smoke=True)
    common.require(len(items) == 98, 'Pełny S1 musi mieć 98 itemów')
    common.require(len(smoke) == 8, 'Smoke musi mieć 8 itemów')
    common.require(len({item['name'] for item in items}) == 98,
                   'Nazwy itemów S1 nie są unikalne')
    for dataset in cfg['datasets']:
        subset = [item for item in items if item['dataset'] == dataset['name']]
        common.require(len(subset) == 49,
                       f"{dataset['name']}: oczekiwano 49 itemów")
        shards = common.shard_items(subset, int(cfg['gpu_count']))
        all_names = [item['name'] for shard in shards.values() for item in shard]
        common.require(len(all_names) == 49, 'Shardy nie są kompletne')
        common.require(len(set(all_names)) == 49, 'Shardy się nakładają')
        common.require(sorted(len(shard) for shard in shards.values())
                       in ([12, 12, 12, 13],),
                       'Nierówny podział shardów')
    print('OK: 98 itemów S1, 8 smoke, 4 rozłączne shardy na dataset')


def check_calibration(bundle, cfg):
    """Sprawdza determinizm i stałość kalibracji bez etykiet."""
    first = common.calibration_indices(
        bundle['train_indices'], int(cfg['calibration_seed']),
        int(cfg['calibration_samples']))
    second = common.calibration_indices(
        bundle['train_indices'], int(cfg['calibration_seed']),
        int(cfg['calibration_samples']))
    common.require(np.array_equal(first, second), 'Kalibracja nie jest stała')
    common.require(len(first) == int(cfg['calibration_samples']),
                   'Zła liczba próbek kalibracji')
    common.require(len(set(first.tolist())) == len(first),
                   'Duplikaty w kalibracji')
    common.require(np.all(np.diff(first) > 0), 'Kalibracja nie jest posortowana')
    train_set = set(bundle['train_indices'].tolist())
    common.require(set(first.tolist()) <= train_set,
                   'Kalibracja zawiera indeksy poza train')
    digest = common.indices_digest(first)
    common.require(len(digest) == 64, 'Zły hash kalibracji')
    print(f'OK: kalibracja {len(first)} próbek, sha256={digest[:16]}...')
    return first


def check_predictor_math(bundle):
    """Sprawdza logit_change i błąd rekonstrukcji na danych kontrolnych."""
    baseline = np.zeros((4, 5), dtype=np.float32)
    same = common.logit_change(baseline, baseline)
    common.require(same['logit_rmse'] == 0.0, 'Identyczne logity mają RMSE != 0')
    common.require(same['prediction_agreement'] == 1.0,
                   'Identyczne logity mają agreement != 1')
    shifted = baseline.copy()
    shifted[:, 1] += 1.0
    changed = common.logit_change(shifted, baseline)
    common.require(changed['logit_rmse'] > 0.0, 'Zmiana logitów nie jest wykryta')
    common.require(changed['prediction_agreement'] < 1.0,
                   'Zmiana predykcji nie jest wykryta')
    control_item = {
        'name': 'control', 'dataset': 'cifar10', 'group_index': None,
        'group_name': None, 'level': 0.0, 'control': True,
    }
    folded, diagnostics = common.fold_item(bundle['source_cpu'], control_item)
    error = common.weight_reconstruction_error(
        bundle['source_cpu'], None, diagnostics)
    common.require(error == 0.0, 'Kontrola ma niezerowy błąd rekonstrukcji')
    del folded
    print('OK: predyktory i kontrola ratio 0')


def check_identity_fold(bundle):
    """Sprawdza, że folding z ratios 0 nie zmienia wartości state_dict."""
    control_item = {
        'name': 'control', 'dataset': 'cifar10', 'group_index': None,
        'group_name': None, 'level': 0.0, 'control': True,
    }
    folded, _diagnostics = common.fold_item(bundle['source_cpu'], control_item)
    reference = bundle['source_cpu'].state_dict()
    candidate = folded.state_dict()
    common.require(reference.keys() == candidate.keys(),
                   'Kontrola zmieniła zestaw kluczy state_dict')
    for key in reference:
        if key.endswith('num_batches_tracked'):
            continue
        common.require(torch.equal(reference[key], candidate[key]),
                       f'Kontrola zmieniła tensor {key}')
    print('OK: kontrola ratio 0 jest tożsamościowa na CPU '
          '(poza licznikiem num_batches_tracked)')


def check_single_group_fold(bundle, cfg, names):
    """Sprawdza rzeczywisty folding jednej grupy z hkmeans na CPU."""
    item = {
        'name': 'probe', 'dataset': 'cifar10', 'group_index': 1,
        'group_name': names[1], 'level': 0.25, 'control': False,
    }
    folded, diagnostics = common.fold_item(
        bundle['source_cpu'], item, int(cfg['fold_seed']))
    reference = common.folding_budget().load_reference_folding()
    common.require(reference.state_dict_all_finite(folded),
                   'Folding pojedynczej grupy dał niefinitywne wartości')
    before = int(sum(value.numel()
                     for value in bundle['source_cpu'].parameters()))
    after = int(sum(value.numel() for value in folded.parameters()))
    common.require(after < before, 'Folding pojedynczej grupy nie zmniejszył modelu')
    entry = next(value for value in diagnostics['groups']
                 if value['name'] == names[1])
    common.require(int(entry['n_clusters']) == 48,
                   f"Oczekiwano 48 kanałów po 25% dla {names[1]}, "
                   f"jest {entry['n_clusters']}")
    error = common.weight_reconstruction_error(
        bundle['source_cpu'], 1, diagnostics)
    common.require(0.0 < error < 1.0,
                   f'Błąd rekonstrukcji poza (0,1): {error}')
    print(
        f'OK: folding grupy {names[1]} 25%: params {before}->{after}, '
        f'recon={error:.4f}')


def check_e015_launcher_fix():
    """Sprawdza, że udokumentowana poprawka launchera E015 jest obecna."""
    path = common.ROOT / (
        'experiments/2026-10-27_E015_cifar100_alokacja/launch_folding.py')
    common.require(path.is_file(), 'Brak launchera E015')
    text = path.read_text(encoding='utf-8')
    common.require('\nimport os\n' in text,
                   'Launcher E015 nie ma import os (poprawka niezastosowana)')
    common.require('LAUNCHER_REVISION' in text,
                   'Launcher E015 nie ma jawnej rewizji')
    print('OK: poprawka launchera E015 (import os) obecna')


def main():
    print('E016 CPU PREFLIGHT')
    common.require('attention' in sys.executable
                   or __import__('os').environ.get('CONDA_DEFAULT_ENV')
                   == 'attention',
                   'Uruchom preflight w środowisku Conda attention')
    cfg = common.load_config()
    sources = common.verify_frozen_sources(cfg)
    print(f'OK: źródła zamrożone ({len(sources)} plików)')
    check_forbidden_imports()
    check_items(cfg)
    check_e015_launcher_fix()
    names = common.group_names()

    for dataset in cfg['datasets']:
        print(f"--- dataset {dataset['name']} ---")
        common.verify_dataset_inputs(dataset)
        bundle = common.load_dataset_bundle(dataset, verify_hashes=True)
        parameters = int(sum(value.numel()
                             for value in bundle['source_cpu'].parameters()))
        common.require(parameters == int(dataset['expected_parameters']),
                       f"Zła liczba parametrów {dataset['name']}")
        print(
            f"OK: {dataset['name']} params={parameters}, "
            f"train={len(bundle['train_indices'])}, "
            f"dev={len(bundle['dev_indices'])}")
        check_calibration(bundle, cfg)
        check_predictor_math(bundle)
        check_identity_fold(bundle)
        check_single_group_fold(bundle, cfg, names)
        del bundle
    print('E016_CPU_PREFLIGHT_OK')


if __name__ == '__main__':
    main()
