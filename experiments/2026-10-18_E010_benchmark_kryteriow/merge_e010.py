"""E010: scalanie nowych kryteriow i zero-shot dev na 4 GPU (bez barrierow).

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md
(wersja 1.1). Wejscia: config frozen, model bazowy, plany per ratio, grupy
z criteria_e010.py, modele etapu 1 E008/E003 (reuzyte).
Wyjscia: merge/models/<dataset>_<ratio>_<kryterium>.pt, merge/dev/*.npz,
merge/zero_metrics.json. Kazdy rank ocenia swoja czesc dev niezaleznie i
zapisuje plik *_rank<r>.npz; rank0 po zakonczeniu czeka na pliki rang i scala
je bez NCCL. Bramki: parametry, pokrycie dev, odtworzenie top-1/NLL modeli
reuzytych i BASE (top-1 dokladnie, |dNLL| <= 1e-5).
"""
import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from support_e010 import (dataset_cfg, plan_rows, plan_spec, base_model, load_dataset,
                          loader, materialize_model, fold_model, count_parameters,
                          model_digest, digest, read_json, write_json, require,
                          load_config, DATASET_KEYS)


def evaluate(model, data_loader, records, rank, device):
    model.eval()
    stats = [0.0, 0, 0]
    logits, ids_seen, targets_seen = [], [], []
    with torch.inference_mode():
        for x, y, ids in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            require(output.dtype == torch.float32 and bool(torch.isfinite(output).all()),
                    'Non-finite dev logits')
            stats[0] += float(nn.functional.cross_entropy(output, y, reduction='sum').double())
            stats[1] += int((output.argmax(1) == y).sum())
            stats[2] += len(ids)
            logits.append(output.cpu().numpy().copy())
            ids_seen.extend(ids.tolist())
            targets_seen.extend(y.tolist())
    require(ids_seen == [r['id'] for r in records[rank::4]]
            and targets_seen == [r['target'] for r in records[rank::4]],
            'Dev rank coverage mismatch')
    return stats, {'ids': np.asarray(ids_seen), 'targets': np.asarray(targets_seen),
                   'logits': np.concatenate(logits)}


def log(path, text):
    with Path(path).open('a') as stream:
        stream.write(time.strftime('%H:%M:%S ') + text + '\n')
        stream.flush()


def wait_for(paths, timeout=1800, interval=0.5):
    started = time.time()
    while True:
        if all(path.exists() for path in paths):
            return
        if time.time() - started > timeout:
            raise RuntimeError(f'Timeout waiting for rank files: {paths}')
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = args.run_dir.resolve()
    (run / 'merge' / 'models').mkdir(parents=True, exist_ok=True)
    (run / 'merge' / 'dev').mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
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
    progress = run / 'merge' / f'progress_rank{rank}.log'
    log(progress, f'start rank={rank} local_rank={local_rank}')
    cell_names = {}
    dev_records = {}
    try:
        for key in DATASET_KEYS:
            ds = dataset_cfg(cfg, key)
            images, labels, records, _train = load_dataset(ds)
            dev_records[key] = records['dev']
            base = base_model(ds)
            dev_batch = int(ds['repair']['eval_batch_per_gpu'])
            cell_names[key] = []
            for ratio, spec in ds['criteria']['ratios'].items():
                rows = plan_rows(ds, ratio)
                plan = plan_spec(ds, ratio)
                for criterion in spec['criteria']:
                    name = f'{key}_{ratio}_{criterion}'
                    cell_names[key].append(name)
                    log(progress, f'build {name}')
                    if criterion in ds['criteria']['reused']:
                        entry = ds['models'][f'{ratio}_{criterion}']
                        payload = torch.load(Path(entry['file']), map_location='cpu',
                                             weights_only=True)
                        model = materialize_model(base, rows, int(plan['base_parameters']))
                        model.load_state_dict(payload['state_dict'], strict=True)
                        groups_sha = ds['candidate_manifest']['sha256']
                    else:
                        groups = read_json(run / 'groups' / f'{key}_{ratio}_{criterion}.json')
                        model = fold_model(base, rows, groups['groups'],
                                           int(plan['base_parameters']))
                        groups_sha = digest(run / 'groups' / f'{key}_{ratio}_{criterion}.json')
                    require(count_parameters(model) == int(plan['compressed_parameters']),
                            f'{name}: parameter budget mismatch')
                    if rank == 0:
                        torch.save({'variant': name, 'dataset': key, 'ratio': ratio,
                                    'criterion': criterion, 'compression_applied': True,
                                    'structure_rows': rows, 'groups_sha256': groups_sha,
                                    'parameters': count_parameters(model),
                                    'model_sha256': model_digest(model),
                                    'state_dict': model.state_dict()},
                                   run / 'merge' / 'models' / f'{name}.pt')
                    model = model.to(device).eval()
                    log(progress, f'eval {name}')
                    _stats, preds = evaluate(
                        model, loader(images, labels, records['dev'], ds, rank, dev_batch),
                        records['dev'], rank, device)
                    np.savez_compressed(run / 'merge' / 'dev' / f'{name}_rank{rank}.npz', **preds)
                    log(progress, f'done {name}')
                    del model
                    torch.cuda.empty_cache()
            name = f'{key}_BASE'
            cell_names[key].append(name)
            log(progress, f'eval {name}')
            base = base.to(device).eval()
            _stats, preds = evaluate(
                base, loader(images, labels, records['dev'], ds, rank, dev_batch),
                records['dev'], rank, device)
            np.savez_compressed(run / 'merge' / 'dev' / f'{name}_rank{rank}.npz', **preds)
            log(progress, f'done {name}')
            del base
            torch.cuda.empty_cache()
    except Exception:
        (run / 'merge' / f'error_rank{rank}.txt').write_text(traceback.format_exc())
        log(progress, 'ERROR')
        raise
    if rank != 0:
        log(progress, 'exit (non-zero rank)')
        return
    try:
        zero_metrics = {}
        for key in DATASET_KEYS:
            ds = dataset_cfg(cfg, key)
            if key == 'tiny':
                reference = read_json(Path(ds['run']['file']) / 'stage1' / 'stage1_metrics.json')['modes']
            else:
                reference = read_json(Path(ds['run']['file']) / 'raw_metrics.json')['modes']
            for name in cell_names[key]:
                paths = [run / 'merge' / 'dev' / f'{name}_rank{r}.npz' for r in range(4)]
                log(progress, f'wait merge {name}')
                wait_for(paths)
                merged = []
                for path in paths:
                    with np.load(path, allow_pickle=False) as archive:
                        merged.append({field: archive[field].copy() for field in archive.files})
                joined = {field: np.concatenate([item[field] for item in merged])
                          for field in merged[0]}
                order = np.argsort(joined['ids'])
                joined = {field: value[order] for field, value in joined.items()}
                require(joined['ids'].tolist() == [r['id'] for r in dev_records[key]]
                        and len(np.unique(joined['ids'])) == int(ds['data']['dev_count']),
                        f'Global dev coverage mismatch for {name}')
                np.savez_compressed(run / 'merge' / 'dev' / f'{name}.npz', **joined)
                targets = joined['targets']
                logits = joined['logits']
                correct = logits.argmax(axis=1) == targets
                maximum = logits.max(axis=1, keepdims=True)
                lse = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
                nll = float(np.mean(lse - logits[np.arange(len(targets)), targets]))
                total = len(targets)
                row = {'dataset': key, 'name': name, 'correct': int(correct.sum()),
                       'count': int(total), 'nll': nll, 'top1': float(correct.mean())}
                criterion = name.split('_', 2)[2]
                ratio = name.split('_')[1]
                if name.endswith('_BASE'):
                    ref = reference['BASE']
                elif criterion in ds['criteria']['reused']:
                    ref = reference[f'{ratio}_{criterion}'] if key == 'tiny' else reference[criterion]
                else:
                    ref = None
                if ref is not None:
                    require(int(correct.sum()) == int(ref['correct']),
                            f'{name}: top-1 does not reproduce reference')
                    require(abs(nll - float(ref['nll'])) <= 1e-5,
                            f'{name}: NLL does not reproduce reference')
                    row['reference_top1'] = float(ref['top1'])
                zero_metrics[name] = row
                print(f'E010_ZERO={name} top1={row["top1"]:.4f} nll={row["nll"]:.4f}', flush=True)
                log(progress, f'merged {name}')
        write_json(run / 'merge' / 'zero_metrics.json', zero_metrics)
        print('E010_ZERO_METRICS=' + str(len(zero_metrics)), flush=True)
        log(progress, 'exit (rank0 merged all)')
    except Exception:
        (run / 'merge' / 'error_rank0.txt').write_text(traceback.format_exc())
        log(progress, 'ERROR rank0 merge')
        os._exit(1)


if __name__ == '__main__':
    main()
