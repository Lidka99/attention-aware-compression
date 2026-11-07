"""E035: audyt pochodzenia Tiny val i archiwum CIFAR10 po ocenie; OPIS.md.
Wejście: istniejące dane/artefakty; wyjście analysis/data_verification.json.
Nie zmienia metryk, checkpointów, danych ani zamrożonego protokołu.
"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from audit import digest, read, ROOT

def main(run):
    meta=read(run/'test_data.json');tiny=ROOT/'data/tiny-imagenet';src=tiny/'tiny-imagenet-200'
    split=read(tiny/'splits.json');classes=sorted((src/'wnids.txt').read_text().split())
    assert classes==split['classes'];indices=split['splits']['reserve']
    assert sorted(indices)==list(range(10000)) and meta['indices']['tiny']==indices
    labels={line.split('\t')[0]:line.split('\t')[1] for line in (src/'val/val_annotations.txt').read_text().splitlines()}
    files=sorted((src/'val/images').glob('*.JPEG'));assert len(files)==10000
    source_hashes={}
    with np.load(tiny/'cache_val.npz',allow_pickle=False) as cache:
        images=cache['images'];y=cache['labels'];assert images.shape==(10000,64,64,3)
        for i,p in enumerate(files):
            assert int(y[i])==classes.index(labels[p.name])
            with Image.open(p) as image:assert np.array_equal(images[i],np.asarray(image.convert('RGB')))
            source_hashes[str(p.relative_to(ROOT))]=digest(p)
    for p in (src/'wnids.txt',src/'val/val_annotations.txt',tiny/'tiny-imagenet-200.zip',tiny/'splits.json',tiny/'cache_val.npz'):
        source_hashes[str(p.relative_to(ROOT))]=digest(p)
    archive=ROOT/meta['archive'];assert digest(archive)==meta['archive_sha256']
    frozen=read(run/'frozen.json');train=ROOT/'runs/E034/train_20261105T014209Z_21b44f52'
    for seed in range(42,46):
        for ds in ('cifar10','cifar100'):
            summary=train/f'seed_{seed}/{ds}/summary.json';checkpoint=summary.parent/'last.pt'
            assert digest(checkpoint)==read(summary)['checkpoint_sha256']==frozen['provenance'][str(checkpoint.relative_to(ROOT))]
            source_hashes[str(summary.relative_to(ROOT))]=digest(summary)
    with tarfile.open(archive,'r:gz') as tar:
        with tar.extractfile('cifar-10-batches-py/test_batch') as stream:
            assert hashlib.sha256(stream.read()).hexdigest()==meta['files']['data/cifar10/cifar-10-batches-py/test_batch']
    result=dict(status='PASS',tiny_pixels_equal=10000,tiny_labels_equal=10000,
        tiny_classes_equal=True,tiny_cache_index_space='official val; separate from cache_train',
        cifar10_archive_test_equal=True,cifar100_reserve_used=False,preact_historical_checkpoint_hashes_equal=8,
        code_sha256=digest(Path(__file__)),source_sha256=source_hashes)
    (run/'analysis/data_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DATA AUDIT PASS: Tiny 10000/10000 pixels+labels match official val; CIFAR10 archive/test equal')
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
