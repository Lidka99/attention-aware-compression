"""E005: przygotowuje listy i pamięciowy cache Tiny ImageNet (tylko CPU).
Eksperyment: 2026-10-10_E005_tiny_imagenet_od_zera; protokół: OPIS.md.
Wejścia: rozpakowane tiny-imagenet-200 (train/, val/, wnids.txt, val_annotations.txt).
Wyjścia (nowe, w katalogu danych): cache_train.npz, cache_val.npz, splits.json.
Podział: 450/50 na klasę train/dev z oficjalnego train; oficjalny val jako rezerwa.
Odmawia nadpisania istniejących plików; nie trenuje i nie używa GPU.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

EXPERIMENT = '2026-10-10_E005_tiny_imagenet_od_zera'
SPLIT_SEED = 20261010


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_image(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    source = root / 'tiny-imagenet-200'
    outputs = [root / 'splits.json', root / 'cache_train.npz', root / 'cache_val.npz']
    if any(p.exists() for p in outputs):
        raise RuntimeError('Outputs already exist; refusing to overwrite')
    wnids = (source / 'wnids.txt').read_text().split()
    if len(wnids) != 200:
        raise RuntimeError('Expected 200 wnids')
    classes = sorted(wnids)
    class_index = {wnid: i for i, wnid in enumerate(classes)}
    rng = np.random.default_rng(SPLIT_SEED)
    all_train_wnids = []
    train_paths = []
    for wnid in classes:
        folder = source / 'train' / wnid / 'images'
        files = sorted(folder.glob('*.JPEG'))
        if len(files) != 500:
            raise RuntimeError(f'{wnid}: expected 500 train images, got {len(files)}')
        order = rng.permutation(500)
        for index in order:
            train_paths.append(files[index])
            all_train_wnids.append(wnid)
    val_annotations = {}
    for line in (source / 'val' / 'val_annotations.txt').read_text().splitlines():
        name, wnid = line.split('\t')[:2]
        val_annotations[name] = wnid
    val_paths = sorted((source / 'val' / 'images').glob('*.JPEG'))
    if len(val_paths) != 10000:
        raise RuntimeError('Expected 10000 val images')
    reserve_ids = [val_annotations[p.name] for p in val_paths]
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        train_images = np.stack(list(pool.map(load_image, train_paths)))
        val_images = np.stack(list(pool.map(load_image, val_paths)))
    train_labels = np.asarray([class_index[w] for w in all_train_wnids], dtype=np.int16)
    reserve_labels = np.asarray([class_index[w] for w in reserve_ids], dtype=np.int16)
    if train_images.shape != (100000, 64, 64, 3) or val_images.shape != (10000, 64, 64, 3):
        raise RuntimeError('Unexpected image shapes')
    np.savez(root / 'cache_train.npz', images=train_images, labels=train_labels)
    np.savez(root / 'cache_val.npz', images=val_images, labels=reserve_labels)
    # Jawny podział na indeksy: pierwsze 450 pozycji każdej klasy to train,
    # kolejne 50 to dev; kolejność klas jak w classes.
    train_idx, dev_idx = [], []
    position = 0
    for wnid in classes:
        train_idx.extend(range(position, position + 450))
        dev_idx.extend(range(position + 450, position + 500))
        position += 500
    splits = {
        'dataset': 'TinyImageNet', 'experiment': EXPERIMENT, 'split_seed': SPLIT_SEED,
        'classes': classes,
        'splits': {'train': train_idx, 'dev': dev_idx, 'reserve': list(range(10000))},
        'counts': {'train': len(train_idx), 'dev': len(dev_idx), 'reserve': 10000},
        'reserve_note': 'Oficjalny val (etykiety); rola w kompresji nieprzypisana.',
        'source_zip_sha256': digest(root / 'tiny-imagenet-200.zip'),
        'official_test_used': False,
        'train_images_sha256': hashlib.sha256(train_images.tobytes()).hexdigest(),
        'val_images_sha256': hashlib.sha256(val_images.tobytes()).hexdigest(),
        'prepared_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (root / 'splits.json').write_text(json.dumps(splits, indent=2, sort_keys=True) + '\n')
    label_balance = np.bincount(train_labels.reshape(200, 500)[:, :450].ravel(), minlength=200)
    print(f'cache: train{list(train_images.shape)} val{list(val_images.shape)}; '
          f'split train={len(train_idx)} dev={len(dev_idx)} reserve=10000; '
          f'train labels per class min={label_balance.min()} max={label_balance.max()}; '
          f'elapsed={time.monotonic() - start:.1f}s')


if __name__ == '__main__':
    main()
