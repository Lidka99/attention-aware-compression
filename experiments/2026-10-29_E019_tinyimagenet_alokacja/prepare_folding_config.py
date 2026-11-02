"""E019: przygotowanie zamrożonego configu foldingu dla jednego seeda.

Eksperyment: 2026-10-29_E019_tinyimagenet_alokacja; pełny opis: OPIS.md.
Wejście: ukończony bieg treningowy E019 (``--train-run``), jego checkpoint
``seed_<seed>/last.pt`` i ``splits.json``. Wyjście: nowy
``<fold-run>/seeds/seed_<seed>/source/config.json`` z hashami wejść,
baseline top-1/NLL na dev oraz zamrożonymi parametrami alokacji. Skrypt
ocenia tylko dev; oficjalny test i rezerwa nie są ładowane.

Uruchamiany przez ``launch_folding.py`` (cztery seedy równolegle na GPU 0–3).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_tiny as data  # noqa: E402

ROOT = data.ROOT
E012_EVALUATE = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/evaluate.py')
TRAIN_PATH = HERE / 'train_tiny.py'
DATA_PATH = HERE / 'data_tiny.py'
ALLOCATION_PATH = HERE / 'configs' / 'allocation.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f'Nie można wczytać modułu {name} z {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E012 = load_module(E012_EVALUATE, '_e015_prepare_e012_evaluate')


def load_training_manifest(train_run):
    manifest_path = Path(train_run) / 'manifest.json'
    require(manifest_path.is_file(), f'Brak manifestu treningu: {manifest_path}')
    manifest = data.read_json(manifest_path)
    require(manifest.get('experiment') == data.EXPERIMENT,
            'Bieg treningowy nie dotyczy E019')
    require(manifest.get('status') == 'completed',
            'Bieg treningowy E019 nie jest ukończony')
    require(manifest.get('smoke') is False,
            'Folding wymaga pełnego treningu, nie smoke')
    require(manifest.get('data_only') is False,
            'Folding wymaga treningu, nie samego przygotowania danych')
    require(int(manifest.get('effective_epochs', 0)) == 200,
            'Folding wymaga 200 epok treningu')
    require(manifest.get('official_test_used') is False,
            'Bieg treningowy użył oficjalnego testu')
    require(manifest.get('reserve_used') is False,
            'Bieg treningowy użył rezerwy')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', type=Path, required=True)
    parser.add_argument('--fold-run', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    train_run = args.train_run.expanduser().resolve()
    fold_run = args.fold_run.expanduser().resolve()
    seed = int(args.seed)
    require(seed in (42, 43, 44, 45), 'Seed musi należeć do 42–45')
    manifest = load_training_manifest(train_run)
    require(seed in [int(value) for value in manifest.get('seeds', [])],
            'Seed nie występuje w manifeście treningu')
    require(manifest.get('gpu_by_seed', {}).get(str(seed)) == int(args.gpu),
            'Przypisanie seeda do GPU nie zgadza się z manifestem')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(visible is not None and visible.strip() == str(args.gpu),
            'CUDA_VISIBLE_DEVICES musi wskazywać --gpu')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Konfigurator wymaga dokładnie jednego widocznego GPU')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)

    checkpoint_path = train_run / f'seed_{seed}' / 'last.pt'
    split_path = train_run / 'splits.json'
    summary_path = train_run / f'seed_{seed}' / 'summary.json'
    require(checkpoint_path.is_file(), f'Brak checkpointu: {checkpoint_path}')
    require(split_path.is_file(), f'Brak splitu: {split_path}')
    require(summary_path.is_file(), f'Brak summary: {summary_path}')
    training_summary = data.read_json(summary_path)
    require(training_summary.get('status') == 'completed',
            'Trening seeda nie jest ukończony')
    require(int(training_summary.get('seed')) == seed,
            'Niezgodny seed w summary treningu')

    images, labels, _ = data.load_tiny_train(data.DATA_ROOT)
    split = data.load_split(split_path)
    data.validate_split(split, labels, data.SPLIT_SEED)
    train_indices = np.asarray(split['splits']['train'], dtype=np.int64)
    dev_indices = np.asarray(split['splits']['dev'], dtype=np.int64)
    require(len(train_indices) == data.EXPECTED_TRAIN,
            'Train nie ma 40 000 próbek')
    require(len(dev_indices) == data.EXPECTED_DEV,
            'Dev nie ma 5 000 próbek')

    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(isinstance(payload, dict)
            and isinstance(payload.get('state_dict'), dict),
            'Checkpoint nie zawiera state_dict')
    model = data.build_model()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    require(data.count_parameters(model) == data.EXPECTED_PARAMETERS,
            'Zła liczba parametrów checkpointu')
    metadata = payload.get('metadata', {})
    if isinstance(metadata, dict) and 'seed' in metadata:
        require(int(metadata['seed']) == seed,
                'Seed checkpointu nie zgadza się z seedem')

    data.seed_everything(seed)
    allocation = data.read_json(ALLOCATION_PATH)
    require(allocation.get('status') == 'frozen',
            'Config alokacji nie jest zamrożony')
    cfg = {
        'batch_size': int(allocation['batch_size']),
        'num_workers': int(allocation['num_workers']),
    }
    dev_loader = E012.make_eval_loader(
        data, images, labels, dev_indices, cfg, seed)
    model_gpu = model.to(device)
    metrics, _, _, _ = E012.evaluate_model(
        model_gpu, dev_loader, device, collect=False)

    final = training_summary.get('final_metrics') or {}
    require(abs(metrics['top1'] - float(final['dev_top1'])) <= 1e-12,
            'Baseline top1 nie zgadza się z summary treningu')
    require(abs(metrics['nll'] - float(final['dev_loss'])) <= 1e-5,
            'Baseline NLL nie zgadza się z summary treningu')

    config = {
        'status': 'frozen',
        'protocol_version': '0.1',
        'experiment': data.EXPERIMENT,
        'seed': seed,
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': data.digest(checkpoint_path),
        'split': str(split_path),
        'split_sha256': data.digest(split_path),
        'training_run': str(train_run),
        'training_summary_sha256': data.digest(summary_path),
        'budgets': [float(value) for value in allocation['budgets']],
        'batch_size': int(allocation['batch_size']),
        'bn_seed': int(allocation['bn_seed']),
        'bn_samples': int(allocation['bn_samples']),
        'num_workers': int(allocation['num_workers']),
        'fold_seed': int(allocation['fold_seed']),
        'baseline_top1': float(metrics['top1']),
        'baseline_nll': float(metrics['nll']),
        'e019_train_sha256': data.digest(TRAIN_PATH),
        'e019_data_sha256': data.digest(DATA_PATH),
        'official_test_used': False,
        'reserve_used': False,
    }
    output_dir = fold_run / 'seeds' / f'seed_{seed}' / 'source'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'config.json'
    data.write_json_atomic(output_path, config)
    print(data.json.dumps({
        'seed': seed,
        'config': str(output_path),
        'config_sha256': data.digest(output_path),
        'baseline_top1': config['baseline_top1'],
        'baseline_nll': config['baseline_nll'],
        'physical_gpu': int(args.gpu),
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
