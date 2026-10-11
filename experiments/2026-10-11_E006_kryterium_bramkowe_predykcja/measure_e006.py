"""E006: pomiar predykcyjny kryterium z bramek SE na 4 GPU (bez kompresji).
Eksperyment: 2026-10-11_E006_kryterium_bramkowe_predykcja; protokół: OPIS.md v0.2.
Wejścia: baseline E005 last.pt (hash z configu), train cache/splits Tiny ImageNet.
Wyjścia: nowy runs/E006 z aktywacjami per-rank, szkodami par, sygnałami, result.json.
Rola: czy sygnał bramkowy przewiduje szkodę scalenia ponad s_post. Zero dev/rezerwy.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e006_metrics import (all_pair_cosines, sample_disjoint_pairs, fit_alpha_beta,
                          pair_signals, fold_block_for_pair, spearman, partial_spearman,
                          bootstrap_spearman_diff)

ROOT = HERE.parents[1]
DATA = ROOT / 'data/tiny-imagenet'
SECONDARY = ('s_pre', 'g_corr', 'g_lstd', 'g_flip', 'g_amp')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False) + '\n')


class EvalImages(Dataset):
    def __init__(self, images, labels, indices, size):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return self.transform(Image.fromarray(self.images[index])), index


class Capture:
    """Odczyt hookami: h (wejście SE), a (scale_activation), z (wyjście SE);
    opcjonalnie block_in (wejście MBConv). Bez modyfikacji forwardu."""

    def __init__(self, se, mbconv, need_block_input):
        self.values = {}
        self.counts = {}
        self.handles = [
            se.register_forward_pre_hook(lambda m, x: self._record('h', x[0])),
            se.scale_activation.register_forward_hook(lambda m, x, y: self._record('a', y)),
            se.register_forward_hook(lambda m, x, y: self._record('z', y))]
        if need_block_input:
            self.handles.append(mbconv.register_forward_pre_hook(
                lambda m, x: self._record('block_in', x[0])))

    def _record(self, key, value):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.values[key] = value.detach().clone()

    def reset(self):
        self.values.clear()
        self.counts.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def sample_indices(labels, train_idx, per_class, seed):
    chosen = []
    for cls in range(200):
        members = train_idx[labels[train_idx] == cls]
        generator = np.random.default_rng(seed + cls)
        picked = generator.permutation(members)
        chosen.extend(int(x) for x in picked[:2 * per_class])
    chosen.sort()
    return chosen


def capture_split(model, se, mbconv, loader, device, capture_block_input):
    cap = Capture(se, mbconv, capture_block_input)
    out = {'h': [], 'a': [], 'z': [], 'block_in': [], 'pos': []}
    try:
        with torch.inference_mode():
            for x, positions in loader:
                cap.reset()
                model(x.to(device))
                require(cap.counts.get('h') == 1 and cap.counts.get('a') == 1
                        and cap.counts.get('z') == 1, 'Hook count mismatch')
                require(not capture_block_input or cap.counts.get('block_in') == 1,
                        'block_in hook mismatch')
                out['h'].append(cap.values['h'].cpu().numpy().astype(np.float32))
                out['a'].append(cap.values['a'][:, :, 0, 0].cpu().numpy().astype(np.float32))
                out['z'].append(cap.values['z'].cpu().numpy().astype(np.float32))
                if capture_block_input:
                    out['block_in'].append(cap.values['block_in'].cpu().numpy().astype(np.float32))
                out['pos'].append(positions.numpy())
    finally:
        cap.close()
    merged = {k: np.concatenate(v) for k, v in out.items() if v}
    order = np.argsort(merged['pos'])
    return {k: v[order] for k, v in merged.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
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
    dist.init_process_group('nccl', timeout=timedelta(minutes=10), device_id=device)
    run = args.run_dir.resolve()
    require(run.is_dir() and not (run / 'result.json').exists(), 'Fresh run directory required')
    try:
        cfg = json.loads(args.config.read_text())
        require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.1', 'Unfrozen config')
        baseline = ROOT / cfg['baseline']['path']
        if rank == 0:
            require(digest(baseline) == cfg['baseline']['sha256'], 'Baseline hash mismatch')
            require(digest(DATA / 'splits.json') == cfg['splits_sha256'], 'Splits hash mismatch')
            write_json(run / 'config.resolved.json', cfg)
        dist.barrier()
        splits = json.loads((DATA / 'splits.json').read_text())
        train_idx = np.asarray(splits['splits']['train'])
        cache = np.load(DATA / 'cache_train.npz', allow_pickle=False)
        images, labels = cache['images'], cache['labels'].astype(np.int64)
        picked = sample_indices(labels, train_idx, cfg['a_per_class'], cfg['sample_seed'])
        require(len(picked) == 400 * cfg['a_per_class'], 'Unexpected sample count')
        a_idx = np.asarray(picked[0::2])
        b_idx = np.asarray(picked[1::2])
        require(len(a_idx) == len(b_idx) == 200 * cfg['a_per_class'], 'A/B size mismatch')
        require(a_idx.tolist() == sorted(a_idx.tolist()) and b_idx.tolist() == sorted(b_idx.tolist()),
                'A/B not sorted')
        loader_a = DataLoader(EvalImages(images, labels, a_idx[rank::4], cfg['image_size']),
                              batch_size=cfg['eval_batch_per_gpu'], shuffle=False, num_workers=2,
                              pin_memory=True)
        loader_b = DataLoader(EvalImages(images, labels, b_idx[rank::4], cfg['image_size']),
                              batch_size=cfg['eval_batch_per_gpu'], shuffle=False, num_workers=2,
                              pin_memory=True)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            efficientnet_b0(weights=None, num_classes=cfg['baseline']['classes']))
        state = torch.load(baseline, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        mbconv_path = cfg['block']
        mbconv = model.get_submodule(mbconv_path)
        se = mbconv.block[2]
        mbconv_cpu = copy.deepcopy(mbconv)
        se_cpu = copy.deepcopy(se)
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        print(f'rank={rank} baseline_loaded block={mbconv_path}', flush=True)
        with torch.no_grad():
            a_arrays = capture_split(model, se, mbconv, loader_a, device, False)
            b_arrays = capture_split(model, se, mbconv, loader_b, device, True)
        require(a_arrays['h'].shape[1:] == (480, 6, 6) and a_arrays['a'].shape[1] == 480
                and b_arrays['block_in'].shape[1:] == (80, 6, 6), 'Unexpected capture shapes')
        np.savez_compressed(run / f'a_rank{rank}.npz', **a_arrays)
        np.savez_compressed(run / f'b_rank{rank}.npz', **b_arrays)
        dist.barrier()
        if rank == 0:
            merged_a = {k: np.concatenate([np.load(run / f'a_rank{r}.npz')[k] for r in range(4)])
                        for k in ('h', 'a', 'z', 'pos')}
            order = np.argsort(merged_a['pos'])
            merged_a = {k: v[order] for k, v in merged_a.items()}
            require(merged_a['pos'].tolist() == a_idx.tolist(), 'A merge coverage')
            merged_b = {k: np.concatenate([np.load(run / f'b_rank{r}.npz')[k] for r in range(4)])
                        for k in ('block_in', 'pos')}
            order = np.argsort(merged_b['pos'])
            merged_b = {k: v[order] for k, v in merged_b.items()}
            require(merged_b['pos'].tolist() == b_idx.tolist(), 'B merge coverage')
            np.savez_compressed(run / 'a_merged.npz',
                                **{k: merged_a[k] for k in ('h', 'a', 'z')})
            np.savez_compressed(run / 'b_merged.npz', block_in=merged_b['block_in'])
            ci, di, s_post_all = all_pair_cosines(merged_a['z'])
            pairs = sample_disjoint_pairs(ci, di, s_post_all, bins=cfg['pairs']['bins'],
                                          per_bin=cfg['pairs']['per_bin'], seed=cfg['pairs']['seed'],
                                          binning=cfg['pairs']['binning'])
            require(pairs.shape == (cfg['pairs']['count'], 2), 'Pair count mismatch')
            h_a, z_a, a_a = merged_a['h'], merged_a['z'], merged_a['a']
            g_a = h_a.mean(axis=(2, 3))
            fc1 = se_cpu.fc1.weight.detach().double().numpy()[:, :, 0, 0]
            rows = []
            for c, d in pairs:
                c, d = int(c), int(d)
                alpha, beta = fit_alpha_beta(h_a, g_a, z_a, c, d)
                signals = pair_signals(h_a, z_a, a_a, g_a, fc1, c, d)
                require(np.isclose(signals['alpha'], alpha) and np.isclose(signals['beta'], beta),
                        'Alpha/beta mismatch between functions')
                row = {'c': c, 'd': d}
                for key, value in signals.items():
                    row[key] = value if isinstance(value, (int, float)) else float(value)
                rows.append(row)
            write_json(run / 'pairs.json', rows)
        dist.barrier()
        rows = json.loads((run / 'pairs.json').read_text())
        b_inputs = np.load(run / 'b_merged.npz')['block_in']
        x_all = torch.tensor(b_inputs, device=device)
        require(x_all.shape == (len(b_idx), 80, 6, 6), 'B inputs shape mismatch')
        my_rows = rows[rank::4]
        damages = []
        with torch.inference_mode():
            branch_t = mbconv.block(x_all).clone()
            out_t = branch_t + x_all if mbconv.use_res_connect else branch_t
            for row in my_rows:
                folded = fold_block_for_pair(mbconv_cpu, row['c'], row['d'],
                                             row['alpha'], row['beta']).to(device)
                folded.eval()
                branch_s = folded.block(x_all)
                out_s = branch_s + x_all if folded.use_res_connect else branch_s
                branch_num = float(((branch_s.double() - branch_t.double()) ** 2).sum())
                branch_den = float((branch_t.double() ** 2).sum())
                block_num = float(((out_s.double() - out_t.double()) ** 2).sum())
                block_den = float((out_t.double() ** 2).sum())
                damages.append({'c': row['c'], 'd': row['d'],
                                'branch_relative_mse': branch_num / branch_den,
                                'block_relative_mse': block_num / block_den})
                del folded
        write_json(run / f'damage_rank{rank}.json', damages)
        dist.barrier()
        if rank == 0:
            collected = {}
            for r in range(4):
                for entry in json.loads((run / f'damage_rank{r}.json').read_text()):
                    collected[(entry['c'], entry['d'])] = entry
            require(len(collected) == len(rows), 'Damage coverage mismatch')
            for row in rows:
                row['damage'] = collected[(row['c'], row['d'])]
            damage_values = np.asarray([row['damage'][cfg['primary_damage']] for row in rows],
                                       dtype=np.float64)
            require(np.isfinite(damage_values).all() and (damage_values > 0).all(),
                    'Invalid damage values')
            s_post = np.asarray([row['s_post'] for row in rows])
            g_pert = np.asarray([row['g_pert'] for row in rows])
            require(np.isfinite(g_pert).all(), 'Non-finite g_pert')
            result = {'experiment': cfg['experiment'], 'run_id': run.name, 'status': 'completed',
                      'pairs': len(rows), 'config_sha256': digest(args.config),
                      'baseline_sha256': cfg['baseline']['sha256'],
                      'primary_damage': cfg['primary_damage'],
                      'positive_control_spearman_s_post': float(spearman(s_post, damage_values)),
                      'primary_g_pert_spearman': float(spearman(g_pert, damage_values))}
            result['primary_comparison'] = bootstrap_spearman_diff(
                g_pert, s_post, damage_values, reps=cfg['bootstrap']['reps'],
                seed=cfg['bootstrap']['seed'])
            result['primary_passed'] = bool(result['primary_comparison']['ci95_low'] > 0)
            result['partial_spearman_g_pert_given_s_post'] = float(
                partial_spearman(g_pert, damage_values, s_post))
            result['secondary'] = {key: {'spearman': float(spearman(
                np.asarray([row[key] for row in rows]), damage_values))} for key in SECONDARY}
            result['secondary']['g_lstd']['value_spearman'] = float(spearman(
                np.asarray([row['g_lstd'] for row in rows]), damage_values))
            result['channels_used'] = int(len(np.unique(
                [row['c'] for row in rows] + [row['d'] for row in rows])))
            result['official_test_used'] = False
            result['reserve_used'] = False
            result['dev_used'] = False
            write_json(run / 'result.json', result)
            keys = ['c', 'd', 'alpha', 'beta', 's_post', 's_pre', 'g_pert', 'g_corr',
                    'g_lstd', 'g_flip', 'g_amp', 'branch_relative_mse', 'block_relative_mse']
            with (run / 'pairs.csv').open('w') as stream:
                stream.write(','.join(keys) + '\n')
                for row in rows:
                    stream.write(','.join(str(row.get(k, row['damage'].get(k))) for k in keys) + '\n')
            print('E006_RESULT=' + json.dumps({k: result[k] for k in (
                'positive_control_spearman_s_post', 'primary_g_pert_spearman',
                'primary_comparison', 'primary_passed')}), flush=True)
            print(f'E006 completed pairs={len(rows)} channels_used={result["channels_used"]}',
                  flush=True)
    except Exception:
        (run / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
