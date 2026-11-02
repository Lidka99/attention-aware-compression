"""E028: niezależne wejście CIFAR-10 i podział train A/B/C; OPIS.md.
Wejście: lokalne batche CIFAR-10, split E011, checkpointy E011.
Wyjście: surowe obrazy, znormalizowane tensory, deterministyczna augmentacja.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'data/cifar10/cifar-10-batches-py'
SPLIT=ROOT/'runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012/splits.json'
CKPT=ROOT/'runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012'
SPLIT_SHA256='d364d4501b20e99d63a7fa680480c5e17a76d4a5f977ff7a0eb75a9c5b50ff34'
HASHES={42:'f74d18ef8906b296c7dcaa9184faead6aec22611d2129f6652c4a149f0813596',
        43:'91423a840233b843d05caa9ed76f2c274139198dcc9e07c64cc66e2f964abab7',
        44:'c8ff0efbe3fe39c18ecf14b400d1f4b7e471a856edf43896398600354b69186d',
        45:'850985a6941c329a2bb9b4dbf06672e2397b05c9f18849e5fb8204b17409e840'}
BASELINE={42:.9528,43:.956,44:.9546,45:.955}
MEAN=(.4914,.4822,.4465);STD=(.2023,.1994,.2010)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def architecture():
    net=resnet18(weights=None,num_classes=10)
    net.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    net.maxpool=nn.Identity()
    return net.eval().requires_grad_(False)


def source_model(seed):
    path=CKPT/f'seed_{seed}/last.pt'
    assert digest(path)==HASHES[seed]
    model=architecture()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    return model


class Images:
    def __init__(self):
        assert digest(SPLIT)==SPLIT_SHA256
        raw=[];labels=[]
        for index in range(1,6):
            with (DATA/f'data_batch_{index}').open('rb') as f:
                entry=pickle.load(f,encoding='latin1')
            raw.append(np.asarray(entry['data'],dtype=np.uint8))
            labels.append(np.asarray(entry['labels'],dtype=np.int64))
        self.raw=np.concatenate(raw);self.labels=np.concatenate(labels)
        assert self.raw.shape==(50000,3072) and len(self.labels)==50000
        split=json.loads(SPLIT.read_text())
        assert split['official_test_used'] is False
        assert len(split['train'])==45000 and len(split['dev'])==5000
        assert not set(split['train']).intersection(split['dev'])
        self.split=split

    def inputs(self,indices):
        x=torch.tensor(self.raw[indices].reshape(-1,3,32,32),dtype=torch.float32)/255
        return (x-torch.tensor(MEAN).view(1,3,1,1))/torch.tensor(STD).view(1,3,1,1)

    def dev(self,indices):
        return self.inputs(indices),torch.tensor(self.labels[indices],dtype=torch.long)


def indices(images):
    perm=np.random.default_rng(20261102).permutation(images.split['train'])
    parts={'A':perm[:4096].tolist(),'B':perm[4096:6144].tolist(),'C':perm[6144:8192].tolist(),
           'dev':images.split['dev']}
    seen=set()
    for key,value in parts.items():
        assert not seen.intersection(value)
        seen.update(value)
    return parts


def augment_uint8(raw,ids):
    result=np.empty((len(ids),3,32,32),dtype=np.uint8)
    for j,index in enumerate(ids):
        rng=np.random.default_rng(20261101+int(index))
        y,x=rng.integers(0,9,size=2)
        image=raw[index].reshape(3,32,32)
        padded=np.pad(image,((0,0),(4,4),(4,4)),constant_values=0)
        crop=padded[:,y:y+32,x:x+32]
        if rng.random()<.5:
            crop=crop[:,:,::-1]
        result[j]=crop
    return torch.from_numpy(result)


def normalize(batch):
    x=batch.to(dtype=torch.float32)/255
    return (x-torch.tensor(MEAN).view(1,3,1,1))/torch.tensor(STD).view(1,3,1,1)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
