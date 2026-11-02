"""E029: adaptery do niezmienionego upstream i naszych operatorów; OPIS.md.
Wejście: istniejące checkpointy, przypięty commit; wyjście modele/diagnostyki.
"""
import copy
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
UP=ROOT/'runs/E029/upstream_6946915'
COMMIT='694691556e8314a835178a741938dd3d4d80c171'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


bn=load(ROOT/'experiments/2026-11-01_E025_diagnostyka_bn/common.py','_e029_bn')
d10=load(ROOT/'experiments/2026-11-01_E028_transfer_cifar10/data10.py','_e029_data10')
legacy=load(ROOT/'experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py','_e029_legacy')
digest,write,read=bn.digest,bn.write,bn.read

# Repo importuje CLIP także w module CNN. Stub nie zmienia żadnego kodu CNN.
if importlib.util.find_spec('open_clip') is None:
    optional=types.ModuleType('open_clip')
    def unused(*args,**kwargs):
        raise RuntimeError('E029: CLIP poza zakresem; nie wolno go wywoływać')
    optional.create_model_and_transforms=unused
    sys.modules['open_clip']=optional
sys.path.insert(0,str(UP))
from models.resnet import ResNet18,get_axis_to_perm_ResNet18
from compression.fold import ResNet18_ModelFolding
from utils.tune_utils import repair_bn
from utils.weight_clustering import axes2perm_to_perm2axes

VARIANTS=('UP_RAW_U','UP_FIXED_U','LEGACY_U','LOCAL_U','UP_FIXED_T','LEGACY_T','LOCAL_T')


def provenance():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()==COMMIT
    assert subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True).strip()==''
    files=[p for p in UP.rglob('*.py')]
    files += [bn.TOOLKIT/'folding.py',bn.TOOLKIT/'data.py',Path(bn.__file__),Path(d10.__file__),Path(legacy.__file__),
              ROOT/'experiments/2026-10-24_E012_folding_referencja/folding.py',bn.SOURCE/'indices.json',
              ROOT/'runs/E028/20261101T154700Z_086b311e/indices.json']
    result={str(p.relative_to(ROOT)):digest(p) for p in files}
    for ds in ('cifar10','cifar100'):
        for seed in range(42,46):
            path=(d10.CKPT if ds=='cifar10' else bn.CHECKPOINT_ROOT)/f'seed_{seed}/last.pt'
            expected=(d10.HASHES if ds=='cifar10' else bn.CHECKPOINT_HASHES)[seed]
            assert digest(path)==expected
            result[str(path.relative_to(ROOT))]=expected
    for p in [d10.SPLIT,bn.SPLIT]+[d10.DATA/f'data_batch_{i}' for i in range(1,6)]+[ROOT/'data/cifar100/cifar-100-python/train']:
        result[str(p.relative_to(ROOT))]=digest(p)
    return result


def source(ds,seed):
    tv=d10.source_model(seed) if ds=='cifar10' else bn.source_model(seed)
    upstream=ResNet18(num_classes=10 if ds=='cifar10' else 100)
    upstream.load_state_dict(tv.state_dict(),strict=True)
    upstream.eval().requires_grad_(False)
    return tv,upstream


def data(ds):
    images=d10.Images() if ds=='cifar10' else bn.Images()
    path=ROOT/'runs/E028/20261101T154700Z_086b311e/indices.json' if ds=='cifar10' else bn.SOURCE/'indices.json'
    ids=read(path)
    assert len(ids['A'])==4096 and len(ids['dev'])==5000
    assert not set(ids['A']).intersection(ids['dev'])
    assert not set(ids['A']).intersection(ids['B']+ids['C'])
    if ds=='cifar100':
        assert not set(ids['A']+ids['dev']).intersection(read(bn.SPLIT)['splits']['reserve'])
    aug=bn.augment_uint8(images.raw,ids['A'])
    a=(d10.normalize if ds=='cifar10' else bn.normalize)(aug)
    dx,dy=images.dev(ids['dev'])
    return a,dx,dy,bn.tensor_hash(aug)


def profiles(model):
    geo=bn.Geometry(model)
    u=geo.profile(round(.5*geo.count(geo.sizes)),0)
    target=geo.count(u)
    t=geo.profile(target,.30)
    assert .999*target<=geo.count(t)<=target
    lo=max(w/n for w,n in zip(u,geo.sizes))
    hi=min((w+1)/n for w,n in zip(u,geo.sizes))
    keep=(lo+hi)/2
    assert [int(n*keep) for n in geo.sizes]==u
    return geo,u,t,1-keep


def official(model,ratio,widths=None):
    """Pristine apply/compress; opcjonalnie wyłącznie keep_ratio per grupa."""
    torch.manual_seed(42);np.random.seed(42)
    work=copy.deepcopy(model)
    op=ResNet18_ModelFolding(work,compression_ratio=ratio)
    original=op.compress_function
    diagnostics=[]
    def observed(axes,params):
        n=int(params[axes[0][0]].shape[0])
        if widths is not None:
            op.keep_ratio=(int(widths[len(diagnostics)])+.25)/n
        k=max(int(n*op.keep_ratio),1)
        compressed,meta=original(axes,params)
        labels=meta['cluster_labels'].cpu()
        assert torch.equal(torch.unique(labels),torch.arange(k)), 'Puste klastry upstream'
        diagnostics.append(dict(n=n,k=k,labels=labels.tolist(),first_axis=axes[0][0]))
        return compressed,meta
    op.compress_function=observed
    result=op.apply().eval().requires_grad_(False)
    assert len(diagnostics)==12
    assert all(torch.isfinite(v).all() for v in result.state_dict().values())
    return result,diagnostics


def construct(ds,seed):
    tv,up=source(ds,seed)
    geo,u,t,ratio=profiles(tv)
    raw,rawdiag=official(up,ratio)
    assert sum(p.numel() for p in raw.parameters())==geo.count(u)
    fixed=copy.deepcopy(raw)
    fixed.fc.bias.data.copy_(up.fc.bias)
    upstream_t,tdiag=official(up,ratio,t)
    upstream_t.fc.bias.data.copy_(up.fc.bias)
    result={'UP_RAW_U':(raw,rawdiag),'UP_FIXED_U':(fixed,rawdiag),'UP_FIXED_T':(upstream_t,tdiag)}
    for tag,widths in [('U',u),('T',t)]:
        torch.manual_seed(42);np.random.seed(42)
        ratios=[1-w/n for w,n in zip(widths,geo.sizes)]
        old,diag=legacy.fold_with_ratios(tv,ratios,seed=42,deterministic=False,preserve_fc_bias=True)
        old.eval().requires_grad_(False)
        result['LEGACY_'+tag]=(old,[dict(labels=g['cluster_labels'],n=g['n_channels'],k=g['n_clusters']) for g in diag['groups']])
        torch.manual_seed(42);np.random.seed(42)
        modern,diag=bn.fold(tv,widths)
        result['LOCAL_'+tag]=(modern,[dict(labels=g['labels'],n=g['original'],k=g['width']) for g in diag])
    for name,(model,_) in result.items():
        assert sum(p.numel() for p in model.parameters())==geo.count(t if name.endswith('_T') else u)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        if name!='UP_RAW_U':
            assert torch.equal(model.fc.bias,tv.fc.bias)
    return tv,up,result,dict(uniform=u,tilt=t,target=geo.count(u),ratio=ratio,tilt_params=geo.count(t))


def calibrate(model,a,mode):
    if mode=='CMA':
        return bn.calibrate(model,a.split(128),'CMA_reset')
    assert mode=='UP_REPAIR'
    before=bn.weights_hash(model)
    for layer in model.modules():
        if isinstance(layer,torch.nn.BatchNorm2d):
            layer.momentum=.1
    loader=[(batch,torch.zeros(len(batch),dtype=torch.long)) for batch in a.split(128)]
    repair_bn(model,loader,device='cuda')
    model.eval()
    assert bn.weights_hash(model)==before
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    return dict(params_unchanged=True,samples=len(a),batches=len(loader),parameters_sha256=before)
