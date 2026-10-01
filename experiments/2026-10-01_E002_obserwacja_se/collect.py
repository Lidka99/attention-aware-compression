"""2026-10-01_E002_obserwacja_se: odczytuje h/a/z jednego SE na 4 GPU.
Wejścia: config.json, SAMPLES.json, INPUTS.json, niezmieniony best.pt E001.
Wyjścia: rank*.npz, kontrole i RNG w nowym katalogu runs/E002.
Protokół: PROTOKOL_v1.md; szkic: OPIS.md. Bez treningu i aktualizacji wag.
"""
import argparse
import os
import random
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from common import (HERE, ROOT, SOURCE, SelectedImages, digest, inspect_inputs, model_digest,
                    object_digest, read_json, require, training_container, verify_samples, write_json)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


class SECapture:
    def __init__(self, module):
        self.values = {}
        self.counts = {}
        self.handles = [module.register_forward_pre_hook(self.pre),
                        module.scale_activation.register_forward_hook(self.scale),
                        module.register_forward_hook(self.post)]

    def record(self, name, value):
        self.counts[name] = self.counts.get(name, 0) + 1
        self.values[name] = value.detach().clone()

    def pre(self, _, inputs):
        self.record('h', inputs[0])

    def scale(self, _, inputs, output):
        self.record('a', output)

    def post(self, _, inputs, output):
        self.record('z', output)

    def reset(self):
        self.values.clear()
        self.counts.clear()

    def checked(self, batch, cfg):
        require(self.counts == {'h': 1, 'a': 1, 'z': 1}, 'Hook invocation mismatch')
        h, a, z = (self.values[k] for k in ('h', 'a', 'z'))
        shape = (batch, cfg['expanded_channels'], *cfg['map_size'])
        require(tuple(h.shape) == tuple(z.shape) == shape
                and tuple(a.shape) == (batch, cfg['expanded_channels'], 1, 1), 'Hook shape mismatch')
        require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in (h, a, z)),
                'Invalid capture precision or non-finite activations')
        require(bool(((a >= 0) & (a <= 1)).all()), 'SE weights out of range')
        require(torch.allclose(z, a * h, atol=cfg['hook_identity_atol'],
                               rtol=cfg['hook_identity_rtol']), 'SE identity violated')
        return h, a, z, float((z - a * h).abs().max())

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == 4 and torch.cuda.device_count() == 4,
            'Exactly four visible GPUs/processes required')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=device)
    run = args.run_dir.resolve()
    capture = None
    try:
        cfg = read_json(HERE / 'config.json')
        require(cfg['status'] == 'frozen' and cfg['world_size'] == 4, 'Unfrozen configuration')
        if rank == 0:
            require(inspect_inputs(cfg) == {k: v for k, v in read_json(HERE / 'INPUTS.json').items()
                                            if k != 'samples_sha256'}, 'Frozen input manifest changed')
            require(digest(HERE / 'SAMPLES.json') == read_json(HERE / 'INPUTS.json')['samples_sha256'],
                    'Sample manifest changed')
            write_json(run / 'integrity.json', read_json(HERE / 'INPUTS.json'))
        dist.barrier()
        images, labels = training_container()
        samples = read_json(HERE / 'SAMPLES.json')
        verify_samples(cfg, samples, labels)
        records = samples['records'][rank::4]
        require(len(records) == 250, 'Unexpected rank sample count')
        random.seed(cfg['seed'] + rank)
        np.random.seed(cfg['seed'] + rank)
        torch.manual_seed(cfg['seed'] + rank)
        torch.cuda.manual_seed_all(cfg['seed'] + rank)
        loader = DataLoader(SelectedImages(images, labels, records, cfg), batch_size=cfg['batch_per_gpu'],
                            shuffle=False, drop_last=False, num_workers=cfg['workers'],
                            worker_init_fn=seed_worker, pin_memory=True,
                            generator=torch.Generator().manual_seed(cfg['seed'] + rank + 100000))
        checkpoint = ROOT / cfg['checkpoint']
        require(digest(checkpoint) == cfg['checkpoint_sha256'], 'Checkpoint changed')
        # Trusted, locally trained checkpoint; full pickle only after hash verification.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(state['git_sha'] == cfg['training_git_sha'] and state['epoch'] + 1 == cfg['checkpoint_epoch']
                and state['config_sha256'] == read_json(HERE / 'INPUTS.json')['config_object_sha256']
                and state['data_sha256'] == read_json(HERE / 'INPUTS.json')['data_object_sha256']
                and state['environment'] == read_json(SOURCE / 'environment.json'),
                'Checkpoint metadata mismatch')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            efficientnet_b0(weights=None, num_classes=cfg['num_classes'])).to(device)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval()
        se = model.get_submodule(cfg['se_module'])
        require(se.fc1.in_channels == 480 and se.fc1.out_channels == 20
                and se.fc2.out_channels == 480 and isinstance(se.scale_activation, nn.Sigmoid),
                'Unexpected SE structure')
        before = model_digest(model)
        capture = SECapture(se)
        torch.save(rng_state(device), run / f'rng_before_rank{rank}.pt')
        print(f'rank={rank} GPU={torch.cuda.get_device_name(local_rank)} samples=250 '
              f'SE={cfg["se_module"]} model_sha256={before}', flush=True)
        output = {key: [] for key in ('ids', 'targets', 'h', 'a', 'z')}
        batches, max_error = 0, 0.0
        with torch.inference_mode():
            for x, y, ids in loader:
                capture.reset()
                logits = model(x.to(device, non_blocking=True))
                require(bool(torch.isfinite(logits).all()), 'Non-finite model output')
                h, a, z, error = capture.checked(len(ids), cfg)
                max_error = max(max_error, error)
                for key, value in (('ids', ids), ('targets', y), ('h', h), ('a', a[:, :, 0, 0]), ('z', z)):
                    output[key].append(value.cpu().numpy().copy())
                batches += 1
        output = {key: np.concatenate(values) for key, values in output.items()}
        require(output['ids'].tolist() == [r['id'] for r in records]
                and output['targets'].tolist() == [r['target'] for r in records], 'Rank coverage mismatch')
        after = model_digest(model)
        require(after == before, 'Parameters/buffers changed')
        torch.save(rng_state(device), run / f'rng_after_rank{rank}.pt')
        np.savez(run / f'rank{rank}.npz', **output)
        result = {'rank': rank, 'local_rank': local_rank, 'gpu': torch.cuda.get_device_name(local_rank),
                  'count': len(records), 'batches': batches, 'last_batch_size': len(records) % 32,
                  'model_before_sha256': before, 'model_after_sha256': after,
                  'max_identity_abs_error': max_error, 'se_module': cfg['se_module'],
                  'hooks_per_batch': {'h': 1, 'a': 1, 'z': 1},
                  'peak_memory_bytes': torch.cuda.max_memory_allocated(device)}
        write_json(run / f'rank{rank}.json', result)
        ranks = [None] * 4
        dist.all_gather_object(ranks, result)
        if rank == 0:
            require(len({r['model_before_sha256'] for r in ranks}) == 1
                    and sum(r['count'] for r in ranks) == 1000, 'GPU replica/coverage mismatch')
            require(digest(checkpoint) == cfg['checkpoint_sha256'], 'Checkpoint file changed')
            write_json(run / 'collection.json', {'status': 'passed', 'ranks': ranks,
                       'count': 1000, 'model_sha256': before, 'weights_changed': False,
                       'test_used': False, 'reserve_used': False, 'dev_used': False})
        dist.barrier()
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        if capture is not None:
            capture.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
