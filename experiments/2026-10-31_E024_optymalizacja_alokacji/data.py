"""E024: niezależne wejścia CIFAR-100 i podział train na A/B/C; OPIS.md.
Wyłącznie istniejące train/split/checkpointy; bez importu kodu eksperymentów.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TRAIN=ROOT/'data/cifar100/cifar-100-python/train'
SPLIT=ROOT/'data/cifar100/splits.json'
CHECKPOINT_ROOT=ROOT/'runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12'
CHECKPOINT_HASHES={42:'ca312df29a70f88597449f7f540a48f97b90f96226163ae0e7182204fa25b386',
43:'e528182ef1a261ed471b50c909d4841d728d4eba35a7c081398ea6cef9cbba5e',
44:'bd6414908abbdae600ab2476091f1702fffd3979d3195ffe6625c6275499c16a',
45:'dfd7fde8c46c7239b4556f40a72c5b8ee92770f0e46d291e8fbec939544ddc5f'}
BASELINE={42:.7494,43:.751,44:.7504,45:.7586}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def split_indices():
    assert digest(SPLIT)=='ecb8804e41fbc4bee7c61fd0585bf1c7dcfc94ac8472333fc283d51dc6460e86'
    split=json.loads(SPLIT.read_text())['splits']
    assert [len(split[k]) for k in ('train','dev','reserve')]==[40000,5000,5000]
    assert len(set(split['train'])|set(split['dev'])|set(split['reserve']))==50000
    p=np.random.default_rng(20261031).permutation(split['train'])
    parts={'A':p[:4096].tolist(),'B':p[4096:6144].tolist(),'C':p[6144:8192].tolist(),'dev':split['dev']}
    seen=set()
    for indices in parts.values():
        assert not seen.intersection(indices) and not set(indices).intersection(split['reserve'])
        seen.update(indices)
    return parts


class Images:
    def __init__(self):
        assert digest(TRAIN)=='735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b'
        with TRAIN.open('rb') as f:
            raw=pickle.load(f,encoding='latin1')
        self.raw=raw['data']
        self.labels=np.asarray(raw['fine_labels'])

    def inputs(self,indices):
        x=torch.tensor(self.raw[indices].reshape(-1,3,32,32),dtype=torch.float32)/255
        mean=torch.tensor([.507421,.486865,.441380]).view(1,3,1,1)
        std=torch.tensor([.267468,.256482,.275979]).view(1,3,1,1)
        return (x-mean)/std

    def dev(self,indices):
        return self.inputs(indices),torch.tensor(self.labels[indices],dtype=torch.long)
