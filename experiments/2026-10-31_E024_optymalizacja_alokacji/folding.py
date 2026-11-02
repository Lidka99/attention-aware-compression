"""E024: samodzielny folding ResNet18 i rachunek budżetu; patrz OPIS.md.
Wejście: model CPU i12 szerokości; wyjście: nowy model i diagnostyka.
Bez zależności od kodu wcześniejszych eksperymentów.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


def architecture():
    net=resnet18(weights=None,num_classes=100)
    net.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    net.maxpool=nn.Identity()
    return net.eval().requires_grad_(False)


class Geometry:
    def __init__(self,model):
        self.axes={}
        def conv(path,gin,gout):
            self.axes[path+'.weight']=(gout,gin,None,None)
        def bn(path,g):
            for suffix in ('weight','bias','running_mean','running_var'):
                self.axes[path+'.'+suffix]=(g,)
        conv('conv1',None,'stem')
        bn('bn1','stem')
        previous='stem'
        for stage in range(1,5):
            output='stem' if stage==1 else f'stage{stage}_out'
            for block in range(2):
                path=f'layer{stage}.{block}'
                hidden=f'{path}_hidden'
                incoming=previous if block==0 else output
                conv(path+'.conv1',incoming,hidden)
                bn(path+'.bn1',hidden)
                conv(path+'.conv2',hidden,output)
                bn(path+'.bn2',output)
                if stage>1 and block==0:
                    conv(path+'.downsample.0',incoming,output)
                    bn(path+'.downsample.1',output)
            previous=output
        self.axes['fc.weight']=(None,previous)
        self.names=list(dict.fromkeys(g for axis in self.axes.values() for g in axis if g is not None))
        self.members={g:[] for g in self.names}
        state=model.state_dict()
        for name,axis in self.axes.items():
            for dim,g in enumerate(axis):
                if g is not None:
                    self.members[g].append((name,dim))
        self.sizes=[]
        for g in self.names:
            lengths={state[name].shape[dim] for name,dim in self.members[g]}
            assert len(lengths)==1
            self.sizes.append(lengths.pop())
        self.shapes={name:tuple(p.shape) for name,p in model.named_parameters()}
        assert len(self.names)==12

    def count(self,widths):
        assignment=dict(zip(self.names,map(int,widths)))
        total=0
        for name,shape in self.shapes.items():
            axis=self.axes.get(name,(None,)*len(shape))
            total+=math.prod(assignment[g] if g is not None else n for g,n in zip(axis,shape))
        return total

    def fill(self,widths,target):
        w=list(map(int,widths))
        maxima=[math.floor(.9*n) for n in self.sizes]
        while True:
            possibilities=[]
            for i in range(12):
                if w[i]>=maxima[i]:
                    continue
                v=w.copy(); v[i]+=1
                params=self.count(v)
                if params<=target:
                    possibilities.append((params,-i,v))
            if not possibilities:
                return w
            w=max(possibilities,key=lambda p:(p[0],p[1]))[2]

    def profile(self,target,tilt):
        def widths(base):
            ratios=np.clip(base+tilt*np.linspace(-1,1,12),.1,.9)
            return [int(n*(1-r)) for n,r in zip(self.sizes,ratios)]
        lo,hi=-abs(tilt),1+abs(tilt)
        assert self.count(widths(lo))>=target>=self.count(widths(hi))
        for _ in range(60):
            mid=(lo+hi)/2
            if self.count(widths(mid))>target:
                lo=mid
            else:
                hi=mid
        result=widths(hi)
        return self.fill(result,target) if tilt else result

    def proposals(self,current,target,round_index):
        rng=np.random.default_rng(20261031+round_index)
        pairs=[(i,j) for i in range(12) for j in range(12) if i!=j]
        rng.shuffle(pairs)
        unique=set()
        for donor,receiver in pairs:
            v=list(current)
            v[donor]-=max(1,round(.05*self.sizes[donor]))
            if v[donor]<math.ceil(.1*self.sizes[donor]):
                continue
            old_receiver=v[receiver]
            while v[receiver]<math.floor(.9*self.sizes[receiver]):
                trial=v.copy();trial[receiver]+=1
                if self.count(trial)>target:
                    break
                v=trial
            if v[receiver]==old_receiver:
                continue
            v=self.fill(v,target)
            key=tuple(v)
            if key not in unique and key!=tuple(current) and .999*target<=self.count(v)<=target:
                unique.add(key)
                yield v


def reduce_axis(tensor,axis,labels,k,average):
    moved=tensor.movedim(axis,0)
    result=torch.zeros((k,*moved.shape[1:]),dtype=tensor.dtype)
    result.index_add_(0,labels,moved)
    if average:
        counts=torch.bincount(labels,minlength=k).to(tensor.dtype)
        result/=counts.reshape(k,*([1]*(result.ndim-1)))
    return result.movedim(0,axis).contiguous()


def fold(source,widths,identity_labels=False):
    from sklearn.decomposition import PCA
    from hkmeans import HKMeans
    geo=Geometry(source)
    assert len(widths)==12
    state={name:value.detach().cpu().clone() for name,value in source.state_dict().items()}
    bn_names={name for name,module in source.named_modules() if isinstance(module,nn.BatchNorm2d)}
    diagnostics=[]
    for g,n,k in zip(geo.names,geo.sizes,widths):
        k=int(k)
        assert 1<=k<=n
        if k==n:
            diagnostics.append(dict(group=g,original=n,width=k,bypass=True))
            continue
        members=geo.members[g]
        if identity_labels:
            labels=torch.arange(n)%k
        else:
            features=torch.cat([state[name].movedim(axis,0).reshape(n,-1)
                                for name,axis in members if 'running_' not in name],dim=1).numpy()
            assert np.isfinite(features).all()
            x=PCA(n_components=min(features.shape),svd_solver='full').fit_transform(features)
            cluster=HKMeans(n_clusters=k,random_state=42,n_init=1,n_jobs=1,max_iter=10,verbose=False)
            labels=torch.tensor(cluster.fit(x).labels_,dtype=torch.long)
        assert torch.equal(torch.unique(labels),torch.arange(k)), 'Puste lub nieciągłe klastry'
        for name,axis in members:
            module_name=name.rsplit('.',1)[0]
            # Parametry BN zawsze średnia; wyjście Conv suma, wejście średnia.
            state[name]=reduce_axis(state[name],axis,labels,k,average=axis==1 or module_name in bn_names)
        diagnostics.append(dict(group=g,original=n,width=k,labels=labels.tolist()))
    result=copy.deepcopy(source)
    for name,tensor in state.items():
        path,field=name.rsplit('.',1)
        module=result.get_submodule(path)
        if field in module._parameters:
            module._parameters[field]=nn.Parameter(tensor,requires_grad=False)
        else:
            module._buffers[field]=tensor
    for module in result.modules():
        if isinstance(module,nn.Conv2d):
            module.out_channels,module.in_channels=module.weight.shape[:2]
        elif isinstance(module,nn.BatchNorm2d):
            module.num_features=module.weight.numel()
        elif isinstance(module,nn.Linear):
            module.out_features,module.in_features=module.weight.shape
    assert sum(p.numel() for p in result.parameters())==geo.count(widths)
    return result.eval(),diagnostics
