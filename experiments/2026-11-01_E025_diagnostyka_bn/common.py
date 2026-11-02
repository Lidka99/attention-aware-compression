"""E025: dane kontrolowane i procedury BN, OPIS.md.
Wejście: zweryfikowany samodzielny operator E024 i lokalne train/profile;
wyjście: widoki kalibracyjne, metryki i diagnostyka bez uczenia wag.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOOLKIT=ROOT/'experiments/2026-10-31_E024_optymalizacja_alokacji'
SOURCE=ROOT/'runs/E024/20261031T141558Z_971ba86e'
sys.path.insert(0,str(TOOLKIT))
from data import Images,CHECKPOINT_ROOT,CHECKPOINT_HASHES,SPLIT,digest,write
from folding import architecture,fold,Geometry

ESTIMATORS=('CMA_reset','EMA_reset','EMA_retain')
MODELS=('teacher','UNIFORM','TILT','SEARCH')
REUSED='A_plain_CMA_reset'


def read(path):
    return json.loads(Path(path).read_text())


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def weights_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cells():
    return [dict(name=f'{size}_{view}_{est}',size=size,view=view,estimator=est)
            for size in ('A','L') for view in ('plain','aug') for est in ESTIMATORS]


def inputs_manifest():
    original=read(SOURCE/'manifest.json')
    assert original['status']=='completed' and original['exit_codes']==[0]*4
    for name,expected in original['source_hashes'].items():
        assert digest(TOOLKIT/name)==expected
    files=[SOURCE/'manifest.json',SOURCE/'indices.json',SPLIT]
    assert digest(SOURCE/'indices.json')==original['indices_sha256']
    for seed in range(42,46):
        folder=SOURCE/f'seed_{seed}'
        summary=read(folder/'summary.json')
        assert digest(folder/'selection.json')==summary['selection_sha256']
        files.extend([folder/'summary.json',folder/'selection.json'])
        files.extend(sorted(folder.glob('*_diagnostics.json')))
        ckpt=CHECKPOINT_ROOT/f'seed_{seed}/last.pt'
        assert digest(ckpt)==CHECKPOINT_HASHES[seed]
        files.append(ckpt)
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def indices():
    old=read(SOURCE/'indices.json')
    split=read(SPLIT)['splits']
    perm=np.random.default_rng(20261031).permutation(split['train']).tolist()
    blocked=set(old['B'])|set(old['C'])
    large=[i for i in perm if i not in blocked]
    assert large[:4096]==old['A'] and len(large)==35904
    for key in ('B','C','dev'):
        assert not set(large).intersection(old[key])
    assert not set(large).intersection(split['reserve'])
    return dict(old,L=large)


def augment_uint8(raw,ids):
    """Crop/flip w przestrzeni pikseli; RNG per indeks, bez globalnego RNG."""
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
    return (x-torch.tensor([.507421,.486865,.441380]).view(1,3,1,1))/torch.tensor([.267468,.256482,.275979]).view(1,3,1,1)


class Views:
    def __init__(self,images,ids):
        self.plain=torch.from_numpy(images.raw[ids].reshape(-1,3,32,32).copy())
        self.aug=augment_uint8(images.raw,ids)
        self.hashes={name:tensor_hash(getattr(self,name)) for name in ('plain','aug')}

    def batches(self,view,n):
        for batch in getattr(self,view)[:n].split(128):
            yield normalize(batch)


@torch.inference_mode()
def calibrate(model,batches,estimator,device='cuda'):
    assert estimator in ESTIMATORS
    weight_before=weights_hash(model)
    source_stats={name:(m.running_mean.clone(),m.running_var.clone(),int(m.num_batches_tracked))
                  for name,m in model.named_modules() if isinstance(m,nn.BatchNorm2d)}
    model.eval()
    starts={}
    for name,m in model.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            assert m.track_running_stats
            if estimator.endswith('reset'):
                m.reset_running_stats()
                assert (m.running_mean==0).all() and (m.running_var==1).all() and int(m.num_batches_tracked)==0
            starts[name]=int(m.num_batches_tracked)
            m.momentum=None if estimator=='CMA_reset' else .1
            m.train()
    assert all(not m.training for m in model.modules() if not isinstance(m,nn.BatchNorm2d))
    sizes=[]
    for batch in batches:
        model(batch.to(device));sizes.append(len(batch))
    model.eval()
    assert weights_hash(model)==weight_before
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    layers=[]
    for name,m in model.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            before_mean,before_var,old_count=source_stats[name]
            assert int(m.num_batches_tracked)==starts[name]+len(sizes)
            assert torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all()
            layers.append(dict(name=name,counter_source=old_count,counter_start=starts[name],counter_end=int(m.num_batches_tracked),
                               mean_rmse=float((m.running_mean-before_mean).square().mean().sqrt()),
                               var_rmse=float((m.running_var-before_var).square().mean().sqrt()),
                               var_before_mean=float(before_var.mean()),var_after_mean=float(m.running_var.mean()),
                               mean_before_sha256=tensor_hash(before_mean),mean_after_sha256=tensor_hash(m.running_mean),
                               var_before_sha256=tensor_hash(before_var),var_after_sha256=tensor_hash(m.running_var)))
    return dict(estimator=estimator,samples=sum(sizes),batches=len(sizes),last_batch=sizes[-1],params_unchanged=True,
                parameters_sha256=weight_before,initial_weight_ema=None if estimator=='CMA_reset' else .9**len(sizes),layers=layers)


@torch.inference_mode()
def predict(model,x):
    model.eval()
    return torch.cat([model(batch.to('cuda')).cpu() for batch in x.split(128)])


def source_model(seed):
    path=CHECKPOINT_ROOT/f'seed_{seed}/last.pt'
    assert digest(path)==CHECKPOINT_HASHES[seed]
    model=architecture()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    return model
