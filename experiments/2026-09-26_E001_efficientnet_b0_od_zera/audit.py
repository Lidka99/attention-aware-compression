"""2026-09-26_E001_efficientnet_b0_od_zera: audyt best.pt na czterech GPU.
Wejścia: lokalne artefakty E001 i treningowy kontener CIFAR-100; tylko dev.
Wyjścia: integralność, predykcje, metryki i logi ranków w nowym biegu.
Protokół: AUDYT_OPIS_v1.md; kontekst: OPIS.md. Bez treningu i modyfikacji wag.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image
import PIL
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parents[2]
TRAIN_COMMIT = '7cc3f46667990ff45686f7499c5dfd00d6d57649'
BEST_HASH = 'f2ac9c3fddd140c0d45f2c244c78b432cbdd164b46dff32d00be956202ec5be8'
E001 = 'experiments/2026-09-26_E001_efficientnet_b0_od_zera'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


class DevImages(Dataset):
    """Only selected dev images are exposed to inference; no test-file access."""
    def __init__(self, data, targets, ids, transform):
        self.data = data
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image = Image.fromarray(self.data[sample_id])
        return self.transform(image), self.targets[sample_id], sample_id


def verify_inputs(source, data_root, run):
    hashes = {}
    for line in (source / 'artifacts.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require(Path(name).name == name, 'Unexpected artifact path')
        actual = digest(source / name)
        require(actual == expected, f'Artifact hash mismatch: {name}')
        hashes[name] = actual
    required = {'best.pt', 'last.pt', 'config.resolved.json', 'data_manifest.json',
                'manifest.json', 'metrics.json', 'metrics.jsonl',
                'best_dev_predictions.json', 'environment.json', 'pip-freeze.txt'}
    require(required <= hashes.keys(), 'Incomplete E001 checksum manifest')
    hashes['artifacts.sha256'] = digest(source / 'artifacts.sha256')
    require(hashes['best.pt'] == BEST_HASH, 'Unexpected checkpoint')
    cfg = read_json(source / 'config.resolved.json')
    manifest = read_json(source / 'manifest.json')
    require(manifest['git_sha'] == TRAIN_COMMIT and not manifest['dirty']
            and manifest['status'] == 'completed' and manifest['exit_code'] == 0
            and not manifest['smoke'] and manifest['resume'] is None,
            'Unexpected training provenance')
    require(object_digest(cfg) == manifest['config_sha256'], 'Config object hash mismatch')
    require(cfg['model'] == 'efficientnet_b0' and cfg['weights'] is None
            and cfg['dataset'] == 'CIFAR100' and cfg['num_classes'] == 100
            and cfg['world_size'] == 4 and cfg['batch_per_gpu'] == 32
            and cfg['workers'] == 2 and cfg['image_size'] == 96
            and cfg['seed'] == 42 and cfg['epochs'] == 200
            and cfg['normalization_mean'] == cfg['normalization_std'] == [0.5] * 3,
            'Unexpected model/evaluation configuration')
    source_hashes = {}
    for name, expected in manifest['source_sha256'].items():
        raw = subprocess.check_output(['git', 'show', f'{TRAIN_COMMIT}:{E001}/{name}'], cwd=ROOT)
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected, f'Training source mismatch: {name}')
        require(digest(ROOT / E001 / name) == expected, f'Current E001 source differs: {name}')
        source_hashes[name] = actual
    splits_path = data_root / 'splits.json'
    splits = read_json(splits_path)
    data_manifest = read_json(source / 'data_manifest.json')
    require(data_manifest == {'splits_sha256': digest(splits_path), **splits},
            'Split manifest mismatch')
    require(splits['split_seed'] == cfg['split_seed'], 'Split seed mismatch')
    require(set(splits['files_sha256']) == {'cifar-100-python/train', 'cifar-100-python/meta'},
            'Unexpected data files; test must not be accessed')
    data_hashes = {'splits.json': digest(splits_path)}
    for name, expected in splits['files_sha256'].items():
        actual = digest(data_root / name)
        require(actual == expected, f'Data hash mismatch: {name}')
        data_hashes[name] = actual
    ids = splits['splits']
    require(set(ids) == {'train', 'dev', 'reserve'}, 'Unexpected split names')
    flat = ids['train'] + ids['dev'] + ids['reserve']
    require(len(flat) == len(set(flat)) == 50000 and sorted(flat) == list(range(50000)),
            'Split overlap or invalid coverage')
    require([len(ids[k]) for k in ('train', 'dev', 'reserve')] == [40000, 5000, 5000],
            'Unexpected split sizes')
    require(ids['dev'] == sorted(ids['dev']), 'Dev order mismatch')
    write_json(run / 'integrity.json', {'status': 'passed', 'artifacts': hashes,
               'data_files': data_hashes, 'training_source': source_hashes,
               'config_object_sha256': object_digest(cfg),
               'data_object_sha256': object_digest(data_manifest)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == 4 and torch.cuda.device_count() == 4,
            'Exactly four GPU processes required')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=device)
    run, source = args.run_dir.resolve(), args.source.resolve()
    try:
        if rank == 0:
            verify_inputs(source, args.data_root, run)
        dist.barrier()
        cfg = read_json(source / 'config.resolved.json')
        data_manifest = read_json(source / 'data_manifest.json')
        dev_ids = data_manifest['splits']['dev']
        # Trusted, hash-verified local CIFAR pickle; never instantiate CIFAR100(test).
        with (args.data_root / 'cifar-100-python/train').open('rb') as f:
            raw = pickle.load(f, encoding='latin1')
        data = raw['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        targets = np.asarray(raw['fine_labels'], dtype=np.int64)
        require(data.shape == (50000, 32, 32, 3) and data.dtype == np.uint8,
                'Invalid training container')
        require(np.bincount(targets[dev_ids], minlength=100).tolist() == [50] * 100,
                'Dev class balance mismatch')
        transform = T.Compose([
            T.Resize((cfg['image_size'], cfg['image_size']),
                     interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(), T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])
        local_ids = dev_ids[rank::4]
        loader = DataLoader(DevImages(data, targets, local_ids, transform),
                            batch_size=cfg['batch_per_gpu'], shuffle=False,
                            num_workers=cfg['workers'], worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000),
                            pin_memory=True, drop_last=False)
        random.seed(cfg['seed'] + rank)
        np.random.seed(cfg['seed'] + rank)
        torch.manual_seed(cfg['seed'] + rank)
        torch.cuda.manual_seed_all(cfg['seed'] + rank)
        # Only the locally generated and hash-verified E001 checkpoint is accepted.
        require(digest(source / 'best.pt') == BEST_HASH, 'Checkpoint changed after integrity check')
        state = torch.load(source / 'best.pt', map_location='cpu', weights_only=False)
        require(state['git_sha'] == TRAIN_COMMIT
                and state['config_sha256'] == object_digest(cfg)
                and state['data_sha256'] == object_digest(data_manifest)
                and state['epoch'] + 1 == 177 and state['best_dev_top1'] == 3342 / 5000,
                'Checkpoint metadata mismatch')
        require(state['environment'] == read_json(source / 'environment.json'),
                'Checkpoint environment provenance mismatch')
        require(state['initial_sha256'] == read_json(source / 'metrics.json')['initial_model_sha256'],
                'Initialization provenance mismatch')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            efficientnet_b0(weights=None, num_classes=100)).to(device)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval()
        before = model_digest(model)
        torch.save(rng_state(device), run / f'rng_before_rank{rank}.pt')
        print(f'rank={rank} local_rank={local_rank} GPU={torch.cuda.get_device_name(local_rank)} '
              f'samples={len(local_ids)} checkpoint_sha256={BEST_HASH} model_sha256={before}', flush=True)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        predictions = []
        with torch.inference_mode():
            for images, labels, sample_ids in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                require(bool(torch.isfinite(logits).all()), 'Non-finite logits')
                loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
                require(bool(torch.isfinite(loss)), 'Non-finite loss')
                pred = logits.argmax(1)
                totals += torch.stack([loss.double(), (pred == labels).sum().double(),
                                       torch.tensor(labels.numel(), device=device).double()])
                predictions.extend([int(i), int(y), int(p)] for i, y, p in
                                   zip(sample_ids.tolist(), labels.tolist(), pred.tolist()))
        after = model_digest(model)
        require(before == after, 'Weights or buffers changed during inference')
        require([p[0] for p in predictions] == local_ids and len(predictions) == 1250,
                'Rank coverage mismatch')
        torch.save(rng_state(device), run / f'rng_after_rank{rank}.pt')
        rank_result = {'rank': rank, 'local_rank': local_rank,
                       'gpu': torch.cuda.get_device_name(local_rank),
                       'count': int(totals[2]), 'correct': int(totals[1]),
                       'nll_sum': float(totals[0]), 'model_before_sha256': before,
                       'model_after_sha256': after,
                       'peak_memory_bytes': torch.cuda.max_memory_allocated(device)}
        write_json(run / f'rank{rank}.json', rank_result)
        all_predictions, ranks = [None] * 4, [None] * 4
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(ranks, rank_result)
        dist.all_reduce(totals)
        if rank == 0:
            require(len({r['model_before_sha256'] for r in ranks}) == 1, 'Different GPU replicas')
            rows = sorted(sum(all_predictions, []))
            require([p[0] for p in rows] == dev_ids and len(set(p[0] for p in rows)) == 5000,
                    'Global sample coverage mismatch')
            require([p[1] for p in rows] == targets[dev_ids].tolist(), 'Target mismatch')
            reference = read_json(source / 'best_dev_predictions.json')
            require(len(reference) == 5000 and [p[:2] for p in reference] == [p[:2] for p in rows],
                    'Reference sample/label mismatch')
            mismatches = [{'id': a[0], 'target': a[1], 'reference': b[2], 'audit': a[2]}
                          for a, b in zip(rows, reference) if a[2] != b[2]]
            metrics = [json.loads(line) for line in (source / 'metrics.jsonl').read_text().splitlines()]
            require([m['epoch'] for m in metrics] == list(range(1, 201)), 'Incomplete epoch log')
            best = max(metrics, key=lambda m: m['dev_top1'])
            require(best['epoch'] == 177 and best['dev_correct'] == 3342
                    and best['dev_count'] == 5000, 'Reference epoch mismatch')
            correct, count = int(totals[1]), int(totals[2])
            require(correct == sum(a[1] == a[2] for a in rows) and count == len(rows),
                    'Aggregated statistics mismatch')
            nll = float(totals[0] / totals[2])
            delta = abs(nll - best['dev_nll'])
            passed = correct == 3342 and count == 5000 and not mismatches and delta <= 1e-6
            result = {'status': 'passed' if passed else 'mismatch', 'checkpoint_epoch': 177,
                      'correct': correct, 'count': count, 'top1': correct / count, 'nll': nll,
                      'reference_nll': best['dev_nll'], 'nll_abs_difference': delta,
                      'nll_tolerance': 1e-6, 'prediction_mismatches': len(mismatches),
                      'unique_ids': len(set(a[0] for a in rows)), 'gpu_results': ranks,
                      'checkpoint_sha256': BEST_HASH, 'model_sha256': before,
                      'official_test_used': False, 'reserve_evaluated': False,
                      'full_training_reproduced': False}
            write_json(run / 'dev_predictions.json', rows)
            write_json(run / 'prediction_mismatches.json', mismatches)
            write_json(run / 'result.json', result)
            print(json.dumps(result), flush=True)
            require(digest(source / 'best.pt') == BEST_HASH, 'Source checkpoint changed')
            require(passed, 'Audit comparison failed; see result.json')
        dist.barrier()
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
