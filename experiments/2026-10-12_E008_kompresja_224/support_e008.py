"""E008: wspolne wejscia i kontrole etapu pomiaru szkody (kompresja 10%/30%).
Eksperyment: 2026-10-12_E008_kompresja_224; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: frozen config, plany struktury, baseline E007 (Tiny ImageNet 224),
cache_train.npz / cache_val.npz / splits.json w data/tiny-imagenet.
Wyjscia: hashe, dane A/B/dev, model bazowy, loadery i kontrole dla study_224.py.
Selekcja par: import selection.py z E003 (hash przypiety) - ta sama semantyka.
"""
import hashlib
import json
import random
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
DATA = ROOT / 'data/tiny-imagenet'
E003 = ROOT / 'experiments/2026-10-09_E003_kompresja_10proc_parametrow'
if str(E003) not in sys.path:
    sys.path.insert(0, str(E003))

from folding_e008 import BASE_PARAMETERS, count_parameters  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32)).tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def code_hashes():
    return {'local': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
            'dependencies': {str((E003 / 'selection.py').relative_to(ROOT)): digest(E003 / 'selection.py')}}


def verify_inputs(cfg):
    run = ROOT / cfg['baseline']['run']
    manifest = read_json(run / 'manifest.json')
    require(manifest['status'] == 'completed' and manifest['exit_code'] == 0,
            'Baseline run not completed')
    require(digest(run / 'last.pt') == cfg['baseline']['sha256'], 'Baseline checkpoint hash mismatch')
    require(manifest['source_data']['splits_sha256'] == cfg['data']['splits_sha256'],
            'Splits hash mismatch vs baseline manifest')
    require(digest(DATA / 'splits.json') == cfg['data']['splits_sha256'], 'Splits file changed')
    require(digest(DATA / 'cache_train.npz') == cfg['data']['cache_train_sha256'], 'Cache train changed')
    require(digest(DATA / 'cache_val.npz') == cfg['data']['cache_val_sha256'], 'Cache val changed')
    plans = {}
    for name, spec in cfg['plans'].items():
        plan = read_json(HERE / spec['file'])
        require(digest(HERE / spec['file']) == spec['sha256'], f'Plan {name} changed')
        require(plan['reduction_percent'] == int(name), f'Plan {name} reduction mismatch')
        require(plan['expected_compressed_parameters'] == spec['compressed_parameters'],
                f'Plan {name} parameter count mismatch')
        plans[name] = {'file': spec['file'], 'sha256': spec['sha256'],
                       'compressed_parameters': plan['expected_compressed_parameters'],
                       'removed_channels': plan['removed_channels']}
    selection_hash = digest(E003 / 'selection.py')
    require(selection_hash == cfg['selection_py_sha256'], 'selection.py changed')
    return {'baseline_run': run.name, 'baseline_sha256': cfg['baseline']['sha256'],
            'splits_sha256': cfg['data']['splits_sha256'],
            'cache_train_sha256': cfg['data']['cache_train_sha256'],
            'cache_val_sha256': cfg['data']['cache_val_sha256'],
            'plans': plans, 'selection_sha256': selection_hash,
            'code_hashes': code_hashes()}


def base_model(cfg):
    run = ROOT / cfg['baseline']['run']
    require(digest(run / 'last.pt') == cfg['baseline']['sha256'], 'Baseline hash mismatch')
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    require(state['epoch'] == cfg['baseline']['epoch'] and state['smoke'] is False,
            'Baseline checkpoint metadata mismatch')
    model = nn.SyncBatchNorm.convert_sync_batchnorm(
        efficientnet_b0(weights=None, num_classes=cfg['num_classes']))
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    require(count_parameters(model) == BASE_PARAMETERS, 'Base model parameter count mismatch')
    return model


def sample_indices(labels, train_idx, per_class, seed):
    chosen = []
    for cls in range(200):
        members = train_idx[labels[train_idx] == cls]
        generator = np.random.default_rng(seed + cls)
        picked = generator.permutation(members)
        chosen.extend(int(x) for x in picked[:2 * per_class])
    chosen.sort()
    return chosen


class EvalImages(Dataset):
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


def eval_transform(cfg):
    size = cfg['image_size']
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC,
                               antialias=True),
                      T.ToTensor(),
                      T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])


def load_data(cfg):
    cache = np.load(DATA / 'cache_train.npz', allow_pickle=False)
    images, labels = cache['images'], cache['labels'].astype(np.int64)
    require(images.shape == (100000, 64, 64, 3) and labels.shape == (100000,),
            'Train cache shape mismatch')
    splits = read_json(DATA / 'splits.json')['splits']
    train_idx = np.asarray(splits['train'])
    dev_idx = np.asarray(splits['dev'])
    picked = sample_indices(labels, train_idx, cfg['data']['a_per_class'], cfg['data']['sample_seed'])
    require(len(picked) == 400 * cfg['data']['a_per_class'], 'Calibration sample count mismatch')
    a_idx = np.asarray(picked[0::2])
    b_idx = np.asarray(picked[1::2])
    require(len(a_idx) == len(b_idx) == 200 * cfg['data']['a_per_class'], 'A/B size mismatch')
    expected_counts = [cfg['data']['a_per_class']] * 200
    require(np.bincount(labels[a_idx], minlength=200).tolist() == expected_counts
            and np.bincount(labels[b_idx], minlength=200).tolist() == expected_counts,
            'A/B class counts mismatch')
    require(len(dev_idx) == len(set(dev_idx.tolist())) == cfg['data']['dev_count'],
            'Dev coverage mismatch')
    require(np.bincount(labels[dev_idx], minlength=200).tolist() == [50] * 200,
            'Dev class counts mismatch')
    records = {'A': [{'id': int(i), 'target': int(labels[int(i)])} for i in a_idx],
               'B': [{'id': int(i), 'target': int(labels[int(i)])} for i in b_idx],
               'dev': [{'id': int(i), 'target': int(labels[int(i)])} for i in dev_idx]}
    return images, labels, records


def loader(images, labels, records, cfg, rank, batch_size):
    return DataLoader(EvalImages(images, labels, [r['id'] for r in records[rank::4]],
                                 eval_transform(cfg)),
                      batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg['workers'], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000),
                      pin_memory=True, in_order=True)
