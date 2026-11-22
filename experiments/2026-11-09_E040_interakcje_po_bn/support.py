"""E040: kontrolowane wejścia, operator upstream i CMA; patrz OPIS.md.
Wejście: własne checkpointy/train i jawne indeksy; wyjście: modele RAM/metryki.
Bez treningu, zapisu wag oraz dostępu do testu/rezerwy.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
import types
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import design as d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UP = ROOT / 'runs/E029/upstream_6946915'
UP_COMMIT = '694691556e8314a835178a741938dd3d4d80c171'
DENIED = []
OPENED_DATA = []
TRAIN_FILES = {
    'cifar10': [ROOT / f'data/cifar10/cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)],
    'cifar100': [ROOT / 'data/cifar100/cifar-100-python/train'],
}
CP_ROOT = {
    'cifar10': ROOT / 'runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012',
    'cifar100': ROOT / 'runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12',
}
SPLITS = {'cifar10': CP_ROOT['cifar10'] / 'splits.json', 'cifar100': ROOT / 'data/cifar100/splits.json'}
SPLIT_SHA = {
    'cifar10': 'd364d4501b20e99d63a7fa680480c5e17a76d4a5f977ff7a0eb75a9c5b50ff34',
    'cifar100': 'ecb8804e41fbc4bee7c61fd0585bf1c7dcfc94ac8472333fc283d51dc6460e86',
}
CP_SHA = {
    'cifar10': {
        42: 'f74d18ef8906b296c7dcaa9184faead6aec22611d2129f6652c4a149f0813596',
        43: '91423a840233b843d05caa9ed76f2c274139198dcc9e07c64cc66e2f964abab7',
        44: 'c8ff0efbe3fe39c18ecf14b400d1f4b7e471a856edf43896398600354b69186d',
        45: '850985a6941c329a2bb9b4dbf06672e2397b05c9f18849e5fb8204b17409e840',
    },
    'cifar100': {
        42: 'ca312df29a70f88597449f7f540a48f97b90f96226163ae0e7182204fa25b386',
        43: 'e528182ef1a261ed471b50c909d4841d728d4eba35a7c081398ea6cef9cbba5e',
        44: 'bd6414908abbdae600ab2476091f1702fffd3979d3195ffe6625c6275499c16a',
        45: 'dfd7fde8c46c7239b4556f40a72c5b8ee92770f0e46d291e8fbec939544ddc5f',
    },
}
EXPECTED_TOP1 = {'cifar10': {42: .9528, 43: .956, 44: .9546, 45: .955},
                 'cifar100': {42: .7494, 43: .751, 44: .7504, 45: .7586}}
MEAN_STD = {'cifar10': ((.4914, .4822, .4465), (.2023, .1994, .2010)),
            'cifar100': ((.507421, .486865, .441380), (.267468, .256482, .275979))}


def _guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    name = str(p)
    prohibited = any(x in name for x in (
        '/test_batch', '/cifar-100-python/test', '/cache_val.npz', '/test_data.json',
        '/attention_v2/', '/attentionv3/'))
    if '/data/' in name and name.startswith(str(ROOT) + '/') and p.is_file():
        allowed = {str(v) for vals in TRAIN_FILES.values() for v in vals} | {str(SPLITS['cifar100'])}
        if name not in allowed:
            prohibited = True
        elif name not in OPENED_DATA:
            OPENED_DATA.append(name)
    if (str(ROOT / 'runs/E040') in name or str(HERE) in name) and p.suffix in ('.pt', '.pth', '.safetensors'):
        prohibited = True
    if prohibited:
        DENIED.append(name)
        raise RuntimeError('E040: zabroniony dostęp: ' + name)


sys.addaudithook(_guard)


def utc():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def write_arrays(path, arrays):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(p)


def array_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_hash(model, parameters_only=False):
    h = hashlib.sha256()
    source = model.named_parameters() if parameters_only else model.state_dict().items()
    for name, tensor in source:
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def checkpoint(dataset, seed):
    return CP_ROOT[dataset] / f'seed_{seed}/last.pt'


def upstream():
    if 'open_clip' not in sys.modules and importlib.util.find_spec('open_clip') is None:
        mod = types.ModuleType('open_clip')

        def unavailable(*args, **kwargs):
            raise RuntimeError('E040: CLIP poza zakresem')

        mod.create_model_and_transforms = unavailable
        sys.modules['open_clip'] = mod
    if str(UP) not in sys.path:
        sys.path.insert(0, str(UP))
    from models.resnet import ResNet18, get_axis_to_perm_ResNet18
    from compression.fold import ResNet18_ModelFolding
    from utils.weight_clustering import axes2perm_to_perm2axes
    axes = get_axis_to_perm_ResNet18(override=False)
    groups = axes2perm_to_perm2axes(axes)
    assert tuple(groups) == d.UP_NAMES
    return ResNet18, ResNet18_ModelFolding, axes, groups


def source_model(dataset, seed):
    path = checkpoint(dataset, seed)
    assert digest(path) == CP_SHA[dataset][seed]
    ResNet18, _, _, _ = upstream()
    model = ResNet18(num_classes=10 if dataset == 'cifar10' else 100)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == d.cost(d.SIZES, dataset)[0]
    return model


def fold(base, widths):
    _, Folding, _, groups = upstream()
    assignment = dict(zip(d.NAMES, widths))
    torch.manual_seed(42)
    np.random.seed(42)
    op = Folding(copy.deepcopy(base), compression_ratio=.5)
    original = op.compress_function
    diagnostics = []

    def wrapped(axes, params):
        group = tuple(groups)[len(diagnostics)]
        n = int(params[axes[0][0]].shape[0])
        k = int(assignment[group])
        op.keep_ratio = (k + .25) / n
        compressed, meta = original(axes, params)
        labels = meta['cluster_labels'].detach().cpu()
        assert torch.equal(torch.unique(labels), torch.arange(k))
        diagnostics.append(dict(group=group, n=n, k=k, labels=labels.tolist()))
        return compressed, meta

    op.compress_function = wrapped
    with redirect_stdout(io.StringIO()):
        model = op.apply().eval().requires_grad_(False)
    model.fc.bias.data.copy_(base.fc.bias)
    # Nieobliczeniowy cache upstream utrzymuje stare moduły; forward go nie używa.
    if hasattr(model, '_module_dict'):
        delattr(model, '_module_dict')
    assert len(diagnostics) == 12
    assert {r['group']: r['k'] for r in diagnostics} == assignment
    return model, diagnostics


def prune(base, widths):
    _, _, axis_map, groups = upstream()
    assignment = dict(zip(d.NAMES, widths))
    selected, diag = {}, []
    for name, axes in groups.items():
        anchors = [path for path, axis in axes if axis == 0 and path.endswith('.weight')
                   and isinstance(base.get_submodule(path.rsplit('.', 1)[0]), nn.Conv2d)]
        assert anchors
        anchor, k = anchors[0], int(assignment[name])
        scores = base.state_dict()[anchor].detach().flatten(1).norm(p=2, dim=1)
        ids = torch.argsort(scores, descending=True, stable=True)[:k]
        selected[name] = ids
        diag.append(dict(group=name, anchor=anchor, n=len(scores), k=k, keep=ids.tolist()))
    model = copy.deepcopy(base)
    for name, tensor in base.state_dict().items():
        value = tensor.detach().clone()
        for axis, group in enumerate(axis_map.get(name, ())):
            if group is not None:
                value = value.index_select(axis, selected[group])
        path, field = name.rsplit('.', 1)
        module = model.get_submodule(path)
        if field in module._parameters:
            module._parameters[field] = nn.Parameter(value.contiguous(), requires_grad=False)
        else:
            module._buffers[field] = value.contiguous()
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            assert layer.groups == 1
            layer.out_channels, layer.in_channels = layer.weight.shape[:2]
        elif isinstance(layer, nn.BatchNorm2d):
            layer.num_features = layer.weight.numel()
            layer.num_batches_tracked.zero_()
        elif isinstance(layer, nn.Linear):
            layer.out_features, layer.in_features = layer.weight.shape
    return model.eval().requires_grad_(False), diag


def validate_shapes(base, model, widths, dataset):
    _, _, axes, _ = upstream()
    assignment = dict(zip(d.NAMES, widths))
    for name, old in base.state_dict().items():
        expected = tuple(assignment.get(g, n) for n, g in zip(old.shape, axes.get(name, (None,) * old.ndim)))
        assert tuple(model.state_dict()[name].shape) == expected, (name, expected)
    assert sum(p.numel() for p in model.parameters()) == d.cost(widths, dataset)[0]
    assert torch.equal(base.fc.bias, model.fc.bias)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())


@torch.inference_mode()
def predict(model, x, expected_mac=None):
    model.eval()
    handles, observed = [], []
    if expected_mac is not None:
        names = {id(layer): name for name, layer in model.named_modules()}
        def hook(layer, inputs, output):
            if isinstance(layer, nn.Conv2d):
                value = output[0].numel() * layer.weight[0].numel()
            else:
                value = output[0].numel() * inputs[0].shape[-1]
            observed.append(dict(name=names[id(layer)], mac=int(value),
                                  weight_shape=list(layer.weight.shape), output_shape=list(output.shape[1:])))
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(layer.register_forward_hook(hook))
    device = next(model.parameters()).device
    values = []
    for i, batch in enumerate(x.split(128)):
        z = model(batch.to(device))
        assert torch.isfinite(z).all()
        values.append(z.detach().cpu())
        if i == 0 and handles:
            for h in handles:
                h.remove()
            actual = sum(row['mac'] for row in observed)
            assert actual == expected_mac, (actual, expected_mac)
            model._e040_geometry = dict(mac=actual, layers=observed)
    return torch.cat(values)


@torch.inference_mode()
def calibrate(model, a):
    before = state_hash(model, True)
    model.eval()
    names = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
            names.append(name)
    assert len(a) == 4096
    device = next(model.parameters()).device
    for batch in a.split(128):
        model(batch.to(device))
    model.eval()
    assert before == state_hash(model, True)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    evidence = []
    for name in names:
        layer = model.get_submodule(name)
        assert int(layer.num_batches_tracked) == 32
        assert torch.isfinite(layer.running_mean).all() and torch.isfinite(layer.running_var).all()
        evidence.append(dict(name=name, batches=32, mean_sha256=array_hash(layer.running_mean),
                             var_sha256=array_hash(layer.running_var)))
    return dict(samples=4096, batches=32, parameters_sha256=before, layers=evidence)


def split_indices(dataset):
    assert digest(SPLITS[dataset]) == SPLIT_SHA[dataset]
    source = read(SPLITS[dataset])
    split = source if dataset == 'cifar10' else source['splits']
    assert len(split['train']) == (45000 if dataset == 'cifar10' else 40000)
    assert len(split['dev']) == 5000
    perm = np.random.default_rng(20261102 if dataset == 'cifar10' else 20261031).permutation(split['train'])
    parts = {'A': perm[:4096].tolist(), 'B': perm[4096:6144].tolist(),
             'C': perm[6144:8192].tolist(), 'dev': split['dev']}
    reserve = set(split.get('reserve', []))
    seen = set()
    for name, ids in parts.items():
        assert len(ids) == len(set(ids)) and not seen.intersection(ids)
        assert not reserve.intersection(ids)
        seen.update(ids)
    return parts, sorted(reserve)


class Images:
    """Kontroluje indeksowanie: tylko A/B przed barierą, C i dev odroczone."""
    def __init__(self, dataset, indices, phase='B'):
        self.dataset, self.indices, self.phase = dataset, indices, phase
        raw, labels = [], []
        for path in TRAIN_FILES[dataset]:
            with path.open('rb') as f:
                entry = pickle.load(f, encoding='latin1')
            raw.append(np.asarray(entry['data'], dtype=np.uint8))
            labels.extend(entry['labels'] if dataset == 'cifar10' else entry['fine_labels'])
        self.raw = np.concatenate(raw)
        self._labels = labels
        assert self.raw.shape == (50000, 3072)
        expected, self.reserve = split_indices(dataset)
        assert expected == indices
        self.access = []

    def inputs(self, part, augment=False):
        allowed = {'B': {'A', 'B'}, 'C': {'A', 'B', 'C'}, 'dev': {'A', 'B', 'C', 'dev'}}[self.phase]
        assert part in allowed and (not augment or part == 'A'), (self.phase, part)
        ids = self.indices[part]
        assert not set(ids).intersection(self.reserve)
        if augment:
            a = np.empty((len(ids), 3, 32, 32), dtype=np.uint8)
            for j, index in enumerate(ids):
                rng = np.random.default_rng(20261101 + int(index))
                y, x = rng.integers(0, 9, size=2)
                image = np.pad(self.raw[index].reshape(3, 32, 32), ((0, 0), (4, 4), (4, 4)))
                crop = image[:, y:y + 32, x:x + 32]
                if rng.random() < .5:
                    crop = crop[:, :, ::-1]
                a[j] = crop
        else:
            a = self.raw[ids].reshape(-1, 3, 32, 32).copy()
        mean, std = MEAN_STD[self.dataset]
        x = torch.from_numpy(a).to(torch.float32) / 255
        x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        self.access.append(dict(phase=self.phase, part=part, count=len(ids), augmented=augment,
                                indices_sha256=array_hash(np.asarray(ids, dtype=np.int64)),
                                pixels_sha256=array_hash(a), tensor_sha256=array_hash(x)))
        return x

    def labels_dev(self):
        assert self.phase == 'dev'
        self.access.append(dict(phase='dev', part='dev_labels', count=5000))
        return torch.tensor([self._labels[i] for i in self.indices['dev']], dtype=torch.long)


def prediction_array(logits):
    return logits.argmax(1).numpy().astype(np.uint8)


def metric(logits, labels):
    pred = prediction_array(logits)
    ce = F.cross_entropy(logits, labels, reduction='none').double().numpy()
    return dict(top1=float(np.mean(pred == labels.numpy())), correct=int(np.count_nonzero(pred == labels.numpy())),
                nll=float(np.mean(ce)), n=len(pred)), pred, ce


def code_hashes():
    files = sorted(HERE.glob('*.py')) + [HERE / 'OPIS.md', HERE / 'UZUPELNIENIE_WYKONAWCZE.md']
    return {relative(p): digest(p) for p in files}


def validate_hashes(mapping):
    for name, expected in mapping.items():
        assert digest(ROOT / name) == expected, name


def configure(gpu=None):
    assert 'envs/attention' in sys.executable
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if gpu is None:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
        assert torch.cuda.device_count() == 1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.cuda.set_device(0)
    torch.manual_seed(42 if gpu is None else 42 + gpu)
    np.random.seed(42 if gpu is None else 42 + gpu)
