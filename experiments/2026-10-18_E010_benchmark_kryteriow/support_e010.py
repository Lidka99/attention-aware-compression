"""E010: wspolne wejscia, adaptery dwoch datasetow i kontrole (bez GPU).

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: frozen config, baseline E007 (Tiny ImageNet 224) i E001 (CIFAR-100 96),
modele etapu 1 E008/E003, plany struktur, momenty Grama, dane i recipe E009.
Wyjscia: helpery dla stats/criteria/merge/repair/summarize: weryfikacja hashy,
adaptery danych i transformacji, modele bazowe, loadery i sciezki.
Bez GPU i bez treningu; stale funkcje deterministyczne.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E008_DIR = ROOT / 'experiments/2026-10-12_E008_kompresja_224'
E003_DIR = ROOT / 'experiments/2026-10-09_E003_kompresja_10proc_parametrow'
E002_DIR = ROOT / 'experiments/2026-10-01_E002_obserwacja_se'
for path in (E008_DIR, E003_DIR, E002_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from support_e008 import (digest, read_json, write_json, require, model_digest,
                          seed_all, seed_worker, load_data as load_data_tiny,
                          base_model as base_model_tiny, loader as loader_tiny,
                          eval_transform as eval_transform_tiny)
import support as support_e003  # noqa: E402
from folding_e010 import count_parameters, fold_model, materialize_model  # noqa: E402

DATASET_KEYS = ('tiny', 'cifar100')


def load_config(path):
    return read_json(Path(path))


def dataset_cfg(cfg, key):
    require(key in cfg['datasets'], f'Unknown dataset: {key}')
    return cfg['datasets'][key]


def plan_spec(ds, ratio):
    require(ratio in ds['plans'], f'Missing plan for ratio {ratio}')
    return ds['plans'][ratio]


def plan_rows(ds, ratio):
    spec = plan_spec(ds, ratio)
    plan = read_json(ROOT / spec['file'])
    require(digest(ROOT / spec['file']) == spec['sha256'],
            f"Plan changed: {spec['file']}")
    require(int(plan['base_parameters']) == int(spec['base_parameters']),
            'Plan base parameter count mismatch')
    require(int(plan['expected_compressed_parameters']) == int(spec['compressed_parameters']),
            'Plan compressed parameter count mismatch')
    require(int(plan['removed_channels']) == int(spec['removed_channels']),
            'Plan removed channel count mismatch')
    return plan['blocks']


def block_map(rows):
    return {f'b{i}': row['block'] for i, row in enumerate(rows)}


def load_moments(ds):
    path = ROOT / ds['moments']['file']
    require(digest(path) == ds['moments']['sha256'], f'Moments changed: {path}')
    return np.load(path, allow_pickle=False)


def load_candidate_manifest(ds):
    path = ROOT / ds['candidate_manifest']['file']
    require(digest(path) == ds['candidate_manifest']['sha256'], f'Manifest changed: {path}')
    return read_json(path)


def load_stats(cfg, key, run_dir):
    path = Path(run_dir) / 'stats' / f'{key}.npz'
    require(path.exists(), f'Missing stats file: {path}')
    meta = read_json(path.with_suffix('.json'))
    archive = np.load(path, allow_pickle=False)
    data = {name: archive[name] for name in archive.files}
    data['settings'] = meta['settings']
    data['meta'] = meta
    return data


def base_model(ds):
    if ds['key'] == 'tiny':
        shim = {'baseline': ds['baseline'], 'num_classes': int(ds['num_classes'])}
        model = base_model_tiny(shim)
    else:
        shim = {'checkpoint': ds['baseline']['file'],
                'checkpoint_sha256': ds['baseline']['sha256'],
                'training_git_sha': ds['baseline']['git_sha']}
        model = support_e003.base_model(shim)
    require(count_parameters(model) == int(ds['baseline']['parameters']),
            'Base model parameter count mismatch')
    if ds['key'] == 'tiny':
        require(model_digest(model) == ds['baseline']['model_sha256'],
                'Base model digest mismatch')
    return model


def load_dataset(ds):
    """Zwraca obrazy, etykiety, rekordy A/B/dev i indeksy train."""
    if ds['key'] == 'tiny':
        shim = {'data': ds['data'], 'image_size': int(ds['image_size']),
                'workers': int(ds['workers']), 'seed': int(ds['seed'])}
        images, labels, records = load_data_tiny(shim)
    else:
        shim = {'calibration_samples': ds['calibration']['samples_file']}
        images, labels, _samples, records = support_e003.load_data(shim)
    splits = read_json(ROOT / ds['data']['splits_file'])['splits']
    train_indices = np.asarray(splits['train'])
    require(len(train_indices) == int(ds['data']['train_count']), 'Train split size mismatch')
    require(len(records['dev']) == int(ds['data']['dev_count']), 'Dev split size mismatch')
    return images, labels, records, train_indices


def train_transform(ds):
    size = int(ds['image_size'])
    if ds['key'] == 'tiny':
        return T.Compose([T.RandomResizedCrop(size, scale=tuple(ds['rrc_scale']),
                                              ratio=tuple(ds['rrc_ratio']),
                                              interpolation=T.InterpolationMode.BICUBIC,
                                              antialias=True),
                          T.ColorJitter(**ds['color_jitter']),
                          T.RandomHorizontalFlip(),
                          T.ToTensor(),
                          T.Normalize(ds['normalization_mean'], ds['normalization_std'])])
    return T.Compose([T.RandomCrop(32, padding=4),
                      T.RandomHorizontalFlip(),
                      T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR,
                               antialias=True),
                      T.ToTensor(),
                      T.Normalize(ds['normalization_mean'], ds['normalization_std'])])


def eval_transform(ds):
    size = int(ds['image_size'])
    if ds['key'] == 'tiny':
        shim = {'image_size': size, 'normalization_mean': ds['normalization_mean'],
                'normalization_std': ds['normalization_std']}
        return eval_transform_tiny(shim)
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR,
                               antialias=True),
                      T.ToTensor(),
                      T.Normalize(ds['normalization_mean'], ds['normalization_std'])])


class ImagesWithIds(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index]), index


def loader(images, labels, records, ds, rank, batch_size):
    return DataLoader(ImagesWithIds(images, labels, [r['id'] for r in records[rank::4]],
                                    eval_transform(ds)),
                      batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=int(ds['workers']), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(int(ds['seed']) + rank + 100000),
                      pin_memory=True, in_order=True)


def dependency_hashes(cfg):
    out = {}
    for relative, expected in cfg['dependencies'].items():
        require(digest(ROOT / relative) == expected, f'Dependency changed: {relative}')
        out[relative] = expected
    return out


def verify_inputs(cfg):
    checks = {'datasets': {}, 'dependencies': dependency_hashes(cfg)}
    recipe = cfg['recipe']['config']
    require(digest(ROOT / recipe['file']) == recipe['sha256'], 'E009 recipe config changed')
    checks['recipe_config_sha256'] = recipe['sha256']
    for key in DATASET_KEYS:
        ds = dataset_cfg(cfg, key)
        for name, spec in (('baseline', ds['baseline']),
                           ('moments', ds['moments']),
                           ('candidate_manifest', ds['candidate_manifest'])):
            require(digest(ROOT / spec['file']) == spec['sha256'],
                    f'{key}: {name} changed')
        for ratio, plan_spec_item in ds['plans'].items():
            require(digest(ROOT / plan_spec_item['file']) == plan_spec_item['sha256'],
                    f'{key}: plan {ratio} changed')
        for mode, model in ds['models'].items():
            require(digest(ROOT / model['file']) == model['sha256'],
                    f'{key}: model {mode} changed')
        for name, spec in ds['data'].items():
            if isinstance(spec, dict) and 'file' in spec:
                require(digest(ROOT / spec['file']) == spec['sha256'],
                        f'{key}: data {name} changed')
        if key == 'tiny':
            run = ROOT / ds['run']['file']
            manifest = read_json(run / 'manifest.json')
            require(manifest['status'] == 'completed' and manifest.get('exit_code') == 0,
                    'E008 run not completed')
            require(manifest['git_sha'] == ds['run']['git_sha'], 'E008 git sha mismatch')
            for name, expected in ds['run']['sha256'].items():
                require(digest(run / 'stage1' / name) == expected,
                        f'E008 stage1 input changed: {name}')
        else:
            run = ROOT / ds['run']['file']
            manifest = read_json(run / 'manifest.json')
            require(manifest['status'] == 'completed' and manifest.get('exit_code') == 0,
                    'E003 run not completed')
            require(manifest['git_sha'] == ds['run']['git_sha'], 'E003 git sha mismatch')
            for name, expected in ds['run']['sha256'].items():
                require(digest(run / name) == expected, f'E003 input changed: {name}')
            samples = ds['calibration']
            require(digest(ROOT / samples['samples_file']) == samples['samples_sha256'],
                    'Calibration samples changed')
        checks['datasets'][key] = {
            'baseline': ds['baseline']['sha256'],
            'plans': {ratio: spec['sha256'] for ratio, spec in ds['plans'].items()},
            'moments': ds['moments']['sha256'],
            'candidate_manifest': ds['candidate_manifest']['sha256'],
            'models': {mode: model['sha256'] for mode, model in ds['models'].items()},
            'run': ds['run']['file']}
    return checks
