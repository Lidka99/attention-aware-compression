"""2026-10-01_E002_obserwacja_se: wspólne kontrole i odczyt wejść.
Wejścia: konfiguracja, manifesty E001, lokalny kontener treningowy CIFAR.
Wyjścia: zweryfikowane metadane i wybrane obrazy, bez pliku testowego.
Protokół: PROTOKOL_v1.md; historyczny szkic: OPIS.md obok tego pliku.
"""
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg'
DATA = ROOT / 'data/cifar100'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False) + '\n')


def git_output(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def check_versions(cfg):
    actual = {'python': platform.python_version(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'numpy': np.__version__}
    require(actual == cfg['expected_versions'], f'Version mismatch: {actual}')


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def inspect_inputs(cfg):
    check_versions(cfg)
    manifest = read_json(SOURCE / 'manifest.json')
    training_cfg = read_json(SOURCE / 'config.resolved.json')
    data_manifest = read_json(SOURCE / 'data_manifest.json')
    splits = read_json(DATA / 'splits.json')
    hashes = {}
    for line in (SOURCE / 'artifacts.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require(Path(name).name == name, 'Invalid source artifact path')
        actual = digest(SOURCE / name)
        require(actual == expected, f'Source artifact changed: {name}')
        hashes[str((SOURCE / name).relative_to(ROOT))] = actual
    hashes[str((SOURCE / 'artifacts.sha256').relative_to(ROOT))] = digest(SOURCE / 'artifacts.sha256')
    require(manifest['git_sha'] == cfg['training_git_sha'] and manifest['status'] == 'completed'
            and not manifest['dirty'] and manifest['exit_code'] == 0 and not manifest['smoke'],
            'Unexpected E001 provenance')
    require(object_digest(training_cfg) == manifest['config_sha256'], 'E001 configuration mismatch')
    require(digest(ROOT / cfg['checkpoint']) == cfg['checkpoint_sha256'], 'Checkpoint mismatch')
    require(data_manifest == {'splits_sha256': digest(DATA / 'splits.json'), **splits},
            'Data manifest mismatch')
    require(set(splits['files_sha256']) == {'cifar-100-python/train', 'cifar-100-python/meta'},
            'Unexpected source data files')
    for name, expected in splits['files_sha256'].items():
        require(digest(DATA / name) == expected, f'Data file mismatch: {name}')
        hashes[str((DATA / name).relative_to(ROOT))] = expected
    hashes[str((DATA / 'splits.json').relative_to(ROOT))] = digest(DATA / 'splits.json')
    flat = splits['splits']['train'] + splits['splits']['dev'] + splits['splits']['reserve']
    require(len(flat) == len(set(flat)) == 50000 and sorted(flat) == list(range(50000)),
            'Overlapping/invalid source splits')
    require([len(splits['splits'][k]) for k in ('train', 'dev', 'reserve')] == [40000, 5000, 5000],
            'Unexpected source split sizes')
    training_sources = {}
    for name, expected in manifest['source_sha256'].items():
        original = subprocess.check_output(['git', 'show',
            f"{cfg['training_git_sha']}:experiments/2026-09-26_E001_efficientnet_b0_od_zera/{name}"], cwd=ROOT)
        require(hashlib.sha256(original).hexdigest() == expected, 'Training source commit mismatch')
        training_sources[name] = expected
    return {'files_sha256': hashes, 'training_sources': training_sources,
            'config_object_sha256': object_digest(training_cfg),
            'data_object_sha256': object_digest(data_manifest), 'verified': True}


def training_container():
    # Trusted CIFAR pickle, verified by SHA-256 before use. No test-file access.
    with (DATA / 'cifar-100-python/train').open('rb') as f:
        raw = pickle.load(f, encoding='latin1')
    images = raw['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(raw['fine_labels'], dtype=np.int64)
    require(images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
            and labels.shape == (50000,) and np.all((labels >= 0) & (labels < 100)),
            'Invalid CIFAR training container')
    return images, labels


def sample_manifest(cfg, labels):
    train_ids = np.asarray(sorted(read_json(DATA / 'splits.json')['splits']['train']))
    generator = np.random.default_rng(cfg['sample_seed'])
    halves = {'A': [], 'B': []}
    for cls in range(100):
        eligible = train_ids[labels[train_ids] == cls]
        require(len(eligible) == 400, 'Unexpected train class size')
        chosen = generator.permutation(eligible)
        halves['A'].extend(int(x) for x in chosen[:cfg['selection_per_class']])
        halves['B'].extend(int(x) for x in chosen[cfg['selection_per_class']:
                                                 cfg['samples_per_class']])
    for half in halves:
        halves[half].sort()
    records = sorted([{'id': i, 'target': int(labels[i]), 'half': half}
                      for half, ids in halves.items() for i in ids], key=lambda r: r['id'])
    require(len(records) == len(set(r['id'] for r in records)) == cfg['sample_count'],
            'Sample duplication')
    return {'experiment': cfg['experiment'], 'sample_seed': cfg['sample_seed'],
            'source': 'E001.train', 'halves': halves, 'records': records}


def verify_samples(cfg, samples, labels):
    require(samples == sample_manifest(cfg, labels), 'Stored samples differ from selection algorithm')
    for half, count in (('A', cfg['selection_count']), ('B', cfg['check_count'])):
        ids = samples['halves'][half]
        require(len(ids) == count and np.bincount(labels[ids], minlength=100).tolist() == [5] * 100,
                'Sample class/count mismatch')


def transform(cfg):
    return T.Compose([T.Resize((cfg['image_size'], cfg['image_size']),
        interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.ToTensor(),
        T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])


class SelectedImages(Dataset):
    def __init__(self, images, labels, records, cfg):
        self.images, self.labels, self.records = images, labels, records
        self.transform = transform(cfg)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample_id = self.records[index]['id']
        return self.transform(Image.fromarray(self.images[sample_id])), int(self.labels[sample_id]), sample_id


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': [torch.cuda.get_device_name(i) for i in range(4)],
            'gpu_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(4)],
            'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                               '--format=csv,noheader'], text=True).strip(),
            'deterministic_algorithms': True, 'tf32': False,
            'env': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
                                                 'PYTHONHASHSEED', 'OMP_NUM_THREADS')}}
