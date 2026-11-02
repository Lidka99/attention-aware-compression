"""E030: Tiny64, niezmieniony operator upstream na naszym modelu; OPIS.md.
Wejście: checkpointy E019, cache_train/split Tiny, klon E029.
Wyjście: model/diagnostyki i dane do oceny bez treningu.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E029=ROOT/'experiments/2026-11-01_E029_oficjalny_operator'
_spec=importlib.util.spec_from_file_location('_e030_e29',E029/'common.py')
e29=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=e29;_spec.loader.exec_module(e29)
bn,legacy=e29.bn,e29.legacy
write,digest,read=e29.write,e29.digest,e29.read

_dspec=importlib.util.spec_from_file_location('_e030_tiny',ROOT/'experiments/2026-10-29_E019_tinyimagenet_alokacja/data_tiny.py')
dt=importlib.util.module_from_spec(_dspec);sys.modules['_e030_tiny']=dt;_dspec.loader.exec_module(dt)

CKPT=ROOT/'runs/E019/20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7'
HASHES={42:'c62680c1ab43c3b80b0add9843898a387bed0aaecaa11a04d24a01d85ae607f5',
        43:'ed6cfbc5eefe765c1bbd38753a0eb1a11793720705a8f930acac832fcf30448f',
        44:'4598da30fb82ac453506fdb87472eaff43b060db14fead9b953f2cb006e5e707',
        45:'39af5fc4a0be696655868e30193d1ce658c03aa7c1bc43a8b2d877242fcdc940'}
BASELINE={42:.5108,43:.506,44:.5065,45:.5095}
MEAN=(.48030321444761837,.4482903024765458,.3977083882277699)
STD=(.2763871312320068,.2688358266538272,.28161664632282396)
AUG_SEED=20261103
RRC=T.RandomResizedCrop(64,scale=(.4,1.0))
FLIP=T.RandomHorizontalFlip()
_=T.Normalize(MEAN,STD)


def source(seed):
    path=CKPT/f'seed_{seed}/last.pt'
    assert digest(path)==HASHES[seed]
    model=dt.build_model().eval().requires_grad_(False)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    return model


def indices(images):
    split=dt.load_split()
    train=np.asarray(split['splits']['train'],dtype=np.int64)
    dev=np.asarray(split['splits']['dev'],dtype=np.int64)
    reserve=np.asarray(split['splits']['reserve'],dtype=np.int64)
    a=np.random.default_rng(AUG_SEED).permutation(train)[:4096].tolist()
    # Indeksy rezerwy dotyczą osobnego cache walidacyjnego, nie train.
    assert set(a).issubset(set(train.tolist()))
    assert not set(a).intersection(dev.tolist())
    assert not set(train.tolist()).intersection(dev.tolist())
    assert len(reserve)==10000
    return dict(A=a,dev=dev.tolist())


def augment(raw,ids):
    out=torch.empty((len(ids),3,64,64),dtype=torch.float32)
    for j,index in enumerate(ids):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(AUG_SEED+int(index))
            image=torch.from_numpy(raw[index]).permute(2,0,1).float()/255
            out[j]=FLIP(RRC(image))
    return out


def normalize(x):
    return (x-torch.tensor(MEAN).view(1,3,1,1))/torch.tensor(STD).view(1,3,1,1)


def probe_token(raw,ids):
    import hashlib
    h=hashlib.sha256()
    tensor=augment(raw,ids)
    h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def provenance():
    files=[E029/'common.py',E029/'OPIS.md',E029/'WYNIKI.md',ROOT/'runs/E029/20261102T005941Z_064da5c2/manifest.json',
           ROOT/'experiments/2026-10-24_E012_folding_referencja/folding.py',Path(legacy.__file__),Path(bn.__file__),Path(dt.__file__),
           dt.CACHE_TRAIN,dt.SPLITS_FILE,ROOT/'runs/E019/fold_20261030T220139Z_cb9ffaf_51bc405f/allocations/seed_42/b50_tilt_p030/summary.json']
    result={str(p.relative_to(ROOT)):digest(p) for p in files}
    for seed in range(42,46):
        path=CKPT/f'seed_{seed}/last.pt';assert digest(path)==HASHES[seed]
        result[str(path.relative_to(ROOT))]=HASHES[seed]
    return result


VARIANTS=('UP_RAW_U','UP_FIXED_U','LEGACY_U','LOCAL_U','UP_FIXED_T','LEGACY_T','LOCAL_T')


def construct(seed):
    import copy
    tv=source(seed)
    geo,u,t,ratio=e29.profiles(tv)
    raw,rawdiag=e29.official(tv,ratio)
    assert sum(p.numel() for p in raw.parameters())==geo.count(u)
    fixed=copy.deepcopy(raw);fixed.fc.bias.data.copy_(tv.fc.bias)
    upt,tdiag=e29.official(tv,ratio,t);upt.fc.bias.data.copy_(tv.fc.bias)
    models={'UP_RAW_U':(raw,rawdiag),'UP_FIXED_U':(fixed,rawdiag),'UP_FIXED_T':(upt,tdiag)}
    for tag,widths in [('U',u),('T',t)]:
        torch.manual_seed(42);np.random.seed(42)
        ratios=[1-w/n for w,n in zip(widths,geo.sizes)]
        old,diag=legacy.fold_with_ratios(tv,ratios,seed=42,deterministic=False,preserve_fc_bias=True)
        old.eval().requires_grad_(False)
        models['LEGACY_'+tag]=(old,[dict(labels=g['cluster_labels'],n=g['n_channels'],k=g['n_clusters']) for g in diag['groups']])
        torch.manual_seed(42);np.random.seed(42)
        modern,diag=bn.fold(tv,widths)
        models['LOCAL_'+tag]=(modern,[dict(labels=g['labels'],n=g['original'],k=g['width']) for g in diag])
    for name,(model,_) in models.items():
        assert sum(p.numel() for p in model.parameters())==geo.count(t if name.endswith('_T') else u)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        if name!='UP_RAW_U':
            assert torch.equal(model.fc.bias,tv.fc.bias)
    return tv,models,dict(uniform=u,tilt=t,target=geo.count(u),tilt_params=geo.count(t),ratio=ratio)
