"""E001: pobiera CIFAR-100 i utrwala rozłączne listy ID trening/dev/rezerwa.
Wejście: config.json i katalog danych. Wyjście: splits.json z hashami źródeł.
Nie ocenia oficjalnego testu. Pełny protokół: OPIS.md w tym folderze.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR100


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    data = CIFAR100(args.data_root, train=True, download=True)
    rng = np.random.default_rng(config['split_seed'])
    labels = np.asarray(data.targets)
    splits = {'train': [], 'dev': [], 'reserve': []}
    for cls in range(100):
        ids = rng.permutation(np.flatnonzero(labels == cls))
        assert len(ids) == 500
        for name, subset in zip(splits, (ids[:400], ids[400:450], ids[450:])):
            splits[name].extend(int(x) for x in subset)
    for ids in splits.values():
        ids.sort()
    assert len(set(sum(splits.values(), []))) == 50000
    files = ['cifar-100-python/train', 'cifar-100-python/meta']
    manifest = {
        'dataset': 'CIFAR100', 'source': CIFAR100.url,
        'split_seed': config['split_seed'], 'splits': splits,
        'classes': data.classes,
        'files_sha256': {name: digest(args.data_root / name) for name in files},
        'official_test_used': False,
        'reserve_role': 'Untouched; role in later compression studies not yet assigned.'
    }
    output = args.data_root / 'splits.json'
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if output.exists() and output.read_text() != encoded:
        raise RuntimeError('Existing split differs; refusing to overwrite it.')
    output.write_text(encoded)
    print(f'{output}: train=40000, dev=5000, reserve=5000; SHA256={digest(output)}')


if __name__ == '__main__':
    main()
