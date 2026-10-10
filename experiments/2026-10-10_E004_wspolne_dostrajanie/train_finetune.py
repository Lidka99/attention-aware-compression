"""E004: wspólne dostrajanie BASE/PRE/POST/RANDOM na dokładnie 4 GPU (DDP).
Eksperyment: 2026-10-10_E004_wspolne_dostrajanie; protokół: PROTOKOL_v1.md i OPIS.md.
Wejścia: modele E003 (hashowane), train/dev CIFAR-100 z E001, frozen config.json.
Wyjścia: nowy runs/E004 z metrykami, predykcjami dev i wagami po dostrajaniu.
Każdy wariant: 120 epok SGD (momentum 0,9, wd 5e-4, LR 1e-2/1e-3/1e-4), AMP,
batch 64/GPU; bez zmiany selekcji/szerokości, bez testu i rezerwy.
"""
import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from PIL import Image

from support_e004 import HERE, ROOT, E003, E002, digest, read_json, write_json, require

sys.path.insert(0, str(E003))
sys.path.insert(0, str(E002))
from common import (DATA, training_container, model_digest, environment, check_versions,
                    object_digest)
from support import base_model
from folding import materialize_model, count_parameters


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


def lr_value(epoch, tune):
    """LR dla epoki wg opublikowanego harmonogramu multi-step (bez warmupu)."""
    value = tune['lr_initial']
    for step in tune['lr_steps']:
        if epoch >= step['epoch']:
            value = step['lr']
    return value


def train_transform(cfg):
    return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.Resize((cfg['image_size'], cfg['image_size']),
                               interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                      T.ToTensor(),
                      T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])


def dev_transform(cfg):
    return T.Compose([T.Resize((cfg['image_size'], cfg['image_size']),
                               interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                      T.ToTensor(),
                      T.Normalize(cfg['normalization_mean'], cfg['normalization_std'])])


class CifarImages(Dataset):
    """Wybrane obrazy train/dev z kontenera CIFAR; transform zależny od trybu."""

    def __init__(self, images, labels, records, transform):
        self.images = images
        self.labels = labels
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample_id = self.records[index]['id']
        image = Image.fromarray(self.images[sample_id])
        return self.transform(image), int(self.labels[sample_id]), sample_id


def verify_inputs(cfg, run):
    require(digest(DATA / 'splits.json') == cfg['splits_sha256'], 'splits.json changed')
    require(digest(DATA / 'cifar-100-python/train') == cfg['train_file_sha256'],
            'CIFAR train container changed')
    manifest = read_json(ROOT / cfg['source_run'] / 'manifest.json')
    require(manifest['status'] == 'completed' and manifest['exit_code'] == 0
            and manifest['experiment'] == cfg['source_experiment'], 'E003 run provenance mismatch')
    for variant, meta in cfg['source_model_files'].items():
        require(digest(ROOT / meta['path']) == meta['sha256'],
                f'Source model changed: {variant}')
    return {'e003_run': cfg['source_run'], 'e003_manifest_sha256': digest(ROOT / cfg['source_run'] / 'manifest.json'),
            'source_model_sha256': {v: m['sha256'] for v, m in cfg['source_model_files'].items()},
            'splits_sha256': cfg['splits_sha256'], 'train_file_sha256': cfg['train_file_sha256'],
            'experiment': cfg['experiment']}


def build_variant_model(cfg, base, variant):
    if variant == 'BASE':
        model = copy.deepcopy(base)
    else:
        rows = read_json(E003 / 'PLAN_STRUKTURY.json')['blocks']
        model = materialize_model(base, rows)
    state = torch.load(ROOT / cfg['source_model_files'][variant]['path'],
                       map_location='cpu', weights_only=True)
    require(state['variant'] == variant
            and state['compression_applied'] == (variant != 'BASE'), 'Artifact metadata mismatch')
    model.load_state_dict(state['state_dict'], strict=True)
    expected = cfg['base_parameters'] if variant == 'BASE' else cfg['compressed_parameters']
    require(count_parameters(model) == expected, 'Physical parameter count mismatch')
    require(model_digest(model) == state['model_sha256'], 'Loaded weights differ from E003 record')
    return model


def evaluate(model, dev_loader, cfg, device, collect=False):
    model.eval()
    val = torch.zeros(3, dtype=torch.float64, device=device)
    collected = {'ids': [], 'targets': [], 'preds': [], 'logits': []}
    with torch.inference_mode():
        for x, y, ids in dev_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            require(bool(torch.isfinite(logits).all()), 'Non-finite dev logits')
            val += torch.stack([nn.functional.cross_entropy(logits, y, reduction='sum').double(),
                                (logits.argmax(1) == y).sum().double(),
                                torch.tensor(y.numel(), device=device).double()])
            if collect:
                collected['ids'].append(ids.numpy().copy())
                collected['targets'].append(y.cpu().numpy().copy())
                collected['preds'].append(logits.argmax(1).cpu().numpy().copy())
                collected['logits'].append(logits.cpu().numpy().astype(np.float32))
    dist.all_reduce(val)
    if collect:
        collected = {k: np.concatenate(v) for k, v in collected.items()}
    return val, collected


def finetune_variant(run, cfg, variant, base, images, labels, train_recs, dev_recs,
                     device, rank, local_rank):
    tune = cfg['fine_tune']
    seed_all(cfg['seed'])
    model = build_variant_model(cfg, base, variant).to(device)
    seed_all(cfg['seed'] + rank)
    ddp = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=tune['lr_initial'],
                                momentum=tune['momentum'], weight_decay=tune['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    dataset = CifarImages(images, labels, train_recs, train_transform(cfg))
    sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=True,
                                 seed=cfg['seed'], drop_last=True)
    dev_loader = DataLoader(CifarImages(images, labels, dev_recs[rank::4], dev_transform(cfg)),
                            batch_size=cfg['eval_batch_per_gpu'], shuffle=False, drop_last=False,
                            num_workers=cfg['workers'], worker_init_fn=seed_worker, pin_memory=True,
                            generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000))
    rows = []
    for epoch in range(tune['epochs']):
        tick = time.monotonic()
        sampler.set_epoch(epoch)
        lr = lr_value(epoch, tune)
        for group in optimizer.param_groups:
            group['lr'] = lr
        loader = DataLoader(dataset, batch_size=cfg['batch_per_gpu'], sampler=sampler,
                            num_workers=cfg['workers'], drop_last=True, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(cfg['seed'] + rank + 10000 * epoch),
                            pin_memory=True)
        ddp.train()
        stats = torch.zeros(3, dtype=torch.float64, device=device)
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                logits = ddp(x)
                loss = nn.functional.cross_entropy(logits, y)
            require(bool(torch.isfinite(loss)), 'Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stats += torch.stack([loss.detach().double() * y.numel(),
                                  (logits.argmax(1) == y).sum().double(),
                                  torch.tensor(y.numel(), device=device).double()])
        dist.all_reduce(stats)
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
        final = epoch == tune['epochs'] - 1
        val, collected = evaluate(model, dev_loader, cfg, device, collect=final)
        require(int(val[2]) == len(dev_recs), 'Dev count mismatch')
        if final:
            gathered = [None] * 4
            dist.all_gather_object(gathered, collected)
            if rank == 0:
                merged = {k: np.concatenate([g[k] for g in gathered]) for k in gathered[0]}
                order = np.argsort(merged['ids'])
                merged = {k: v[order] for k, v in merged.items()}
                expected_ids = [r['id'] for r in dev_recs]
                require(merged['ids'].tolist() == expected_ids
                        and len(set(merged['ids'].tolist())) == cfg['data']['dev_count'], 'Dev coverage mismatch')
                require(merged['targets'].tolist() == [r['target'] for r in dev_recs],
                        'Dev target mismatch')
                np.savez_compressed(run / f'dev_final_{variant}.npz', **merged)
        row = {'variant': variant, 'epoch': epoch + 1, 'lr': lr,
               'train_nll': float(stats[0] / stats[2]), 'train_correct': int(stats[1]),
               'train_count': int(stats[2]), 'dev_nll': float(val[0] / val[2]),
               'dev_correct': int(val[1]), 'dev_count': int(val[2]),
               'dev_top1': float(val[1] / val[2]), 'elapsed_seconds': time.monotonic() - tick}
        rows.append(row)
        if rank == 0:
            with (run / f'metrics_{variant}.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        dist.barrier()
    replica_hashes = [None] * 4
    dist.all_gather_object(replica_hashes, model_digest(model))
    require(len(set(replica_hashes)) == 1, 'Model replicas differ after training')
    torch.save(rng_state(device), run / f'rng_after_{variant}_rank{rank}.pt')
    if rank == 0:
        (run / 'models').mkdir(exist_ok=True)
        model.eval()
        path = run / 'models' / f'{variant}_ft.pt'
        torch.save({'variant': variant, 'epoch': tune['epochs'],
                    'state_dict': model.state_dict(), 'parameters': count_parameters(model),
                    'model_sha256': model_digest(model),
                    'start_model_file_sha256': cfg['source_model_files'][variant]['sha256'],
                    'protocol_version': cfg['protocol_version']}, path)
        best = max(rows, key=lambda r: r['dev_top1'])
        write_json(run / f'variant_{variant}.json', {
            'variant': variant, 'parameters': count_parameters(model),
            'final_model_sha256': model_digest(model), 'file_sha256': digest(path),
            'file_bytes': path.stat().st_size, 'epochs': tune['epochs'],
            'final_row': rows[-1], 'best_dev_top1': best['dev_top1'],
            'best_dev_epoch': best['epoch'], 'replica_hashes': replica_hashes,
            'source_model_file_sha256': cfg['source_model_files'][variant]['sha256']})
    del optimizer, scaler, dataset, sampler, dev_loader, ddp
    dist.barrier()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == 4 and torch.cuda.device_count() == 4,
            'Exactly 4 GPUs/processes required')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=60), device_id=device)
    try:
        cfg = read_json(HERE / 'config.json')
        check_versions(cfg)
        require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.0', 'Unfrozen config')
        if rank == 0:
            write_json(run / 'integrity.json', verify_inputs(cfg, run))
        dist.barrier()
        images, labels = training_container()
        splits = read_json(DATA / 'splits.json')['splits']
        require(len(splits['train']) == cfg['data']['train_count']
                and len(splits['dev']) == cfg['data']['dev_count'], 'Split sizes changed')
        require(np.bincount(labels[splits['dev']], minlength=100).tolist() == [50] * 100,
                'Dev class balance mismatch')
        train_recs = [{'id': int(i), 'target': int(labels[i])} for i in splits['train']]
        dev_recs = [{'id': int(i), 'target': int(labels[i])} for i in splits['dev']]
        torch.save(rng_state(device), run / f'rng_before_rank{rank}.pt')
        for variant in cfg['fine_tune']['variants_sequential_order']:
            print(f'rank={rank} GPU={torch.cuda.get_device_name(local_rank)} variant={variant} '
                  f'epochs={cfg["fine_tune"]["epochs"]} batch_per_gpu={cfg["batch_per_gpu"]}', flush=True)
            base = base_model(cfg)
            start = time.monotonic()
            finetune_variant(run, cfg, variant, base, images, labels, train_recs, dev_recs,
                             device, rank, local_rank)
            if rank == 0:
                print(f'VARIANT_DONE={variant} seconds={time.monotonic() - start:.1f}', flush=True)
            del base
            torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            print('E004_FINETUNE=completed', flush=True)
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
