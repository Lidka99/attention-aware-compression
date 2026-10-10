"""E003: wspólne wejścia i kontrole badania kompresji całej sieci.
Eksperyment 2026-10-09_E003_kompresja_10proc_parametrow; protokół PROTOKOL_v1.md i OPIS.md.
Wejścia: frozen config, własne artefakty E001/E002 i plik train CIFAR.
Wyjścia: sprawdzone modele/listy/loader i metadane; bez pliku testowego.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E002 = ROOT / 'experiments/2026-10-01_E002_obserwacja_se'
sys.path.insert(0, str(E002))
from common import (SOURCE, DATA, SelectedImages, digest, object_digest, read_json,
                    write_json, require, inspect_inputs, training_container, model_digest,
                    environment, check_versions)


def code_hashes():
    return {'local': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
            'dependencies': {str((E002 / 'common.py').relative_to(ROOT)): digest(E002 / 'common.py')}}


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def verify_all_inputs(cfg):
    integrity = inspect_inputs(cfg)
    require(digest(HERE / cfg['structure_plan']) == cfg['structure_plan_sha256'], 'Structure plan changed')
    require(digest(ROOT / cfg['calibration_samples']) == cfg['sample_manifest_sha256'], 'Samples changed')
    plan = read_json(HERE / cfg['structure_plan'])
    require(plan['base_parameters'] == cfg['base_parameters']
            and plan['expected_compressed_parameters'] == cfg['expected_compressed_parameters'], 'Plan count mismatch')
    integrity.update(structure_plan_sha256=cfg['structure_plan_sha256'],
                     sample_manifest_sha256=cfg['sample_manifest_sha256'])
    return integrity


def load_data(cfg):
    images, labels = training_container()
    samples = read_json(ROOT / cfg['calibration_samples'])
    splits = read_json(DATA / 'splits.json')['splits']
    train_ids = set(splits['train'])
    require(len(samples['records']) == 1000 and len(set(r['id'] for r in samples['records'])) == 1000,
            'Sample uniqueness mismatch')
    require(not set(samples['halves']['A']) & set(samples['halves']['B']), 'A/B overlap')
    for row in samples['records']:
        require(row['id'] in train_ids and row['target'] == int(labels[row['id']]), 'Calibration membership/label mismatch')
    for half in ('A', 'B'):
        ids = samples['halves'][half]
        require(len(ids) == 500 and np.bincount(labels[ids], minlength=100).tolist() == [5]*100,
                'Calibration class counts mismatch')
    dev_ids = splits['dev']
    require(len(dev_ids) == len(set(dev_ids)) == 5000
            and np.bincount(labels[dev_ids], minlength=100).tolist() == [50]*100, 'Dev coverage mismatch')
    records = {half: [r for r in samples['records'] if r['half'] == half] for half in ('A', 'B')}
    records['dev'] = [{'id': i, 'target': int(labels[i])} for i in dev_ids]
    return images, labels, samples, records


def loader(images, labels, records, cfg, rank):
    return DataLoader(SelectedImages(images, labels, records[rank::4], cfg),
                      batch_size=cfg['batch_per_gpu'], shuffle=False, drop_last=False,
                      num_workers=cfg['workers'], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000),
                      pin_memory=True, in_order=True)


def base_model(cfg):
    require(digest(ROOT / cfg['checkpoint']) == cfg['checkpoint_sha256'], 'Checkpoint hash mismatch')
    state = torch.load(ROOT / cfg['checkpoint'], map_location='cpu', weights_only=False)
    require(state['git_sha'] == cfg['training_git_sha'] and state['epoch'] + 1 == 177
            and state['config_sha256'] == object_digest(read_json(SOURCE / 'config.resolved.json'))
            and state['data_sha256'] == object_digest(read_json(SOURCE / 'data_manifest.json')),
            'Checkpoint metadata mismatch')
    base = nn.SyncBatchNorm.convert_sync_batchnorm(efficientnet_b0(weights=None, num_classes=100))
    base.load_state_dict(state['model'], strict=True)
    base.eval()
    require(model_digest(base) == '0ce0a309b9108d0d4b0db6af3ebd207c7c91b7107009df469ed2c2f08917f4ea',
            'Base model differs from verified checkpoint')
    return base
