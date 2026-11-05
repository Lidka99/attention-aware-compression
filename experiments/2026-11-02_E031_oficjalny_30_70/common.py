"""E031: oficjalny operator przy 30/70% kompresji, bez treningu; OPIS.md.
Wejście: checkpointy E011/E015/E019, dane i klon upstream E029.
Wyjście: plany profili, provenance i adaptery operatorów.
"""
import copy
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E029=ROOT/'experiments/2026-11-01_E029_oficjalny_operator'
E030=ROOT/'experiments/2026-11-02_E030_oficjalny_tiny'
UP=ROOT/'runs/E029/upstream_6946915'
COMMIT='694691556e8314a835178a741938dd3d4d80c171'
BUDGETS=(0.30,0.70)
TAGS={0.30:'b30',0.70:'b70'}
DATASETS=('cifar10','cifar100','tiny')
VARIANTS=('UP_RAW_U','UP_FIXED_U','LEGACY_U','LOCAL_U','UP_FIXED_T','LEGACY_T','LOCAL_T')
MODES=('PRE','CMA','UP_REPAIR')

CIFAR10_RUN=ROOT/'runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012'
CIFAR100_RUN=ROOT/'runs/E015/20261027T190911Z_cb9ffaf_seeds42-45_82c98b12'
TINY_RUN=ROOT/'runs/E019/20261030T000950Z_cb9ffaf_seeds42-45_4ef97df7'
E028_INDICES=ROOT/'runs/E028/20261101T154700Z_086b311e/indices.json'
E024_RUN=ROOT/'runs/E024/20261031T141558Z_971ba86e'
E024_INDICES=E024_RUN/'indices.json'
E029_RUN=ROOT/'runs/E029/20261102T005941Z_064da5c2'
E030_RUN=ROOT/'runs/E030/20261102T013740Z_609b7737'
HIST={
    'cifar10': ROOT/'runs/E014/20261027T094707Z_267904d_6a0e93eb/seeds/seed_42/allocations',
    'cifar100': ROOT/'runs/E015/fold_20261027T210324Z_cb9ffaf_36a01249/allocations/seed_42',
    'tiny': ROOT/'runs/E019/fold_20261030T220139Z_cb9ffaf_51bc405f/allocations/seed_42',
}


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


e30=load(E030/'common.py','_e031_e30')
e29=e30.e29
bn,legacy,d10,dt=e29.bn,e29.legacy,e29.d10,e30.dt
digest,write,read=e29.digest,e29.write,e29.read
BASELINE={'cifar10':d10.BASELINE,
          'cifar100':{42:.7494,43:.751,44:.7504,45:.7586},
          'tiny':e30.BASELINE}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def provenance():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()==COMMIT
    assert subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True).strip()==''
    files=list(UP.rglob('*.py'))
    files += [Path(bn.__file__),Path(d10.__file__),E029/'common.py',E029/'OPIS.md',E029/'WYNIKI.md',
              E030/'common.py',E030/'OPIS.md',E030/'WYNIKI.md',
              ROOT/'experiments/2026-10-24_E012_folding_referencja/folding.py',
              ROOT/'experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py',
              ROOT/'experiments/2026-10-31_E024_optymalizacja_alokacji/folding.py',
              ROOT/'experiments/2026-10-31_E024_optymalizacja_alokacji/data.py',
              ROOT/'experiments/2026-10-29_E019_tinyimagenet_alokacja/data_tiny.py',
              E028_INDICES,E024_INDICES,E029_RUN/'manifest.json',E030_RUN/'manifest.json']
    for ds in DATASETS:
        for tag in ('b30','b70'):
            for variant in ('uniform','tilt_p030'):
                files.append(HIST[ds]/f'{tag}_{variant}/summary.json')
    result={rel(p):digest(p) for p in files}
    for ds in ('cifar10','cifar100','tiny'):
        for seed in range(42,46):
            if ds=='cifar10':
                path=CIFAR10_RUN/f'seed_{seed}/last.pt';expected=d10.HASHES[seed]
            elif ds=='cifar100':
                path=CIFAR100_RUN/f'seed_{seed}/last.pt';expected=bn.CHECKPOINT_HASHES[seed]
            else:
                path=TINY_RUN/f'seed_{seed}/last.pt';expected=e30.HASHES[seed]
            assert digest(path)==expected
            result[rel(path)]=expected
    for path in [d10.SPLIT,bn.SPLIT,dt.CACHE_TRAIN,dt.SPLITS_FILE,
                 ROOT/'data/cifar100/cifar-100-python/train']+[d10.DATA/f'data_batch_{i}' for i in range(1,6)]:
        result[rel(path)]=digest(path)
    return result


def indices_all():
    parts={}
    raw=read(E028_INDICES)
    assert len(raw['A'])==4096 and len(raw['dev'])==5000
    assert not set(raw['A']).intersection(raw['dev'])
    assert not set(raw['A']).intersection(raw['B']+raw['C'])
    parts['cifar10']=dict(A=raw['A'],dev=raw['dev'],source=rel(E028_INDICES),
                          source_sha256=digest(E028_INDICES))
    raw=read(E024_INDICES)
    split=read(bn.SPLIT)['splits']
    assert len(raw['A'])==4096 and len(raw['dev'])==5000
    assert not set(raw['A']).intersection(raw['dev'])
    assert not set(raw['A']).intersection(raw['B']+raw['C'])
    assert not set(raw['A']+raw['dev']).intersection(split['reserve'])
    parts['cifar100']=dict(A=raw['A'],dev=raw['dev'],source=rel(E024_INDICES),
                           source_sha256=digest(E024_INDICES))
    split=dt.load_split()
    train=np.asarray(split['splits']['train'],dtype=np.int64)
    dev=np.asarray(split['splits']['dev'],dtype=np.int64)
    reserve=np.asarray(split['splits']['reserve'],dtype=np.int64)
    a=np.random.default_rng(e30.AUG_SEED).permutation(train)[:4096].tolist()
    assert set(a).issubset(set(train.tolist()))
    assert not set(a).intersection(dev.tolist())
    assert len(dev)==10000 and len(reserve)==10000
    assert not set(train.tolist()).intersection(dev.tolist())
    parts['tiny']=dict(A=a,dev=dev.tolist(),source=rel(dt.CACHE_TRAIN),
                       source_sha256=digest(dt.CACHE_TRAIN),split=rel(dt.SPLITS_FILE),
                       split_sha256=digest(dt.SPLITS_FILE))
    return parts


def load_source(ds,seed):
    if ds=='cifar10':
        tv=d10.source_model(seed)
        up=e29.ResNet18(num_classes=10)
        up.load_state_dict(tv.state_dict(),strict=True)
    elif ds=='cifar100':
        tv=bn.source_model(seed)
        up=e29.ResNet18(num_classes=100)
        up.load_state_dict(tv.state_dict(),strict=True)
    elif ds=='tiny':
        tv=e30.source(seed);up=None
    else:
        raise ValueError(ds)
    tv.eval().requires_grad_(False)
    if up is not None:
        up.eval().requires_grad_(False)
    return tv,up


def data(ds,ids):
    if ds in ('cifar10','cifar100'):
        images=d10.Images() if ds=='cifar10' else bn.Images()
        aug=bn.augment_uint8(images.raw,ids['A'])
        a=(d10.normalize if ds=='cifar10' else bn.normalize)(aug)
        dx,dy=images.dev(ids['dev'])
        views=bn.tensor_hash(aug)
    elif ds=='tiny':
        images,labels,_=dt.load_tiny_train(dt.DATA_ROOT)
        aug=e30.augment(images,ids['A'])
        a=e30.normalize(aug)
        dx=e30.normalize(torch.from_numpy(images[ids['dev']]).permute(0,3,1,2).float()/255)
        dy=torch.tensor(labels[ids['dev']],dtype=torch.long)
        views=hashlib.sha256(aug.contiguous().numpy().tobytes()).hexdigest()
    else:
        raise ValueError(ds)
    assert len(a)==4096 and len(dx)==len(dy)
    assert torch.isfinite(a).all() and torch.isfinite(dx).all()
    return a,dx,dy,views


def profiles(model,compression):
    assert compression in BUDGETS
    geo=bn.Geometry(model)
    retained=1-compression
    nominal=round(retained*geo.count(geo.sizes))
    u=geo.profile(nominal,0)
    target=geo.count(u)
    t=geo.profile(target,.30)
    assert .999*target<=geo.count(t)<=target
    lo=max(w/n for w,n in zip(u,geo.sizes))
    hi=min((w+1)/n for w,n in zip(u,geo.sizes))
    keep=(lo+hi)/2
    assert [int(n*keep) for n in geo.sizes]==u
    return geo,u,t,1-keep,nominal


def construct(ds,seed,compression,tv=None,up=None):
    if tv is None:
        tv,up=load_source(ds,seed)
    geo,u,t,ratio,nominal=profiles(tv,compression)
    base=up if up is not None else tv
    raw,rawdiag=e29.official(base,ratio)
    assert sum(p.numel() for p in raw.parameters())==geo.count(u)
    fixed=copy.deepcopy(raw);fixed.fc.bias.data.copy_(tv.fc.bias)
    upt,tdiag=e29.official(base,ratio,t);upt.fc.bias.data.copy_(tv.fc.bias)
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
    plan=dict(compression=compression,retained=1-compression,
              nominal=nominal,uniform=u,tilt=t,target=geo.count(u),
              tilt_params=geo.count(t),ratio=ratio)
    return models,plan


def teacher_controls(ds,seed,tv,up,a,dx,dy):
    model=copy.deepcopy(tv).to('cuda')
    logits=bn.predict(model,dx);del model
    def metrics(z):
        return dict(top1=float((z.argmax(1)==dy).double().mean()),
                    nll=float(F.cross_entropy(z,dy)))
    expected=BASELINE[ds][seed]
    assert abs(metrics(logits)['top1']-expected)<1e-12
    architecture_error=None
    if up is not None:
        native=copy.deepcopy(up).to('cuda')
        native_logits=bn.predict(native,dx);del native
        architecture_error=float((logits-native_logits).abs().max())
        assert architecture_error<=1e-4
    controls={'PRE':metrics(logits)}
    for mode in ('CMA','UP_REPAIR'):
        model=copy.deepcopy(tv).to('cuda');e29.calibrate(model,a,mode)
        controls[mode]=metrics(bn.predict(model,dx));del model
    return controls,architecture_error
