"""E028: kontrole CPU transferu reguły na CIFAR-10; OPIS.md.
Wyjście check.json przed GPU. Bez ocen naukowych dev.
"""
import copy
import torch
from support import *


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(42)
    inputs=sources()
    images=Images()
    parts=indices(images)
    assert {k:len(v) for k,v in parts.items()}=={'A':4096,'B':2048,'C':2048,'dev':5000}
    seen=set()
    for value in parts.values():
        assert not seen.intersection(value);seen.update(value)
    source=source_model(42)
    assert sum(p.numel() for p in source.parameters())==11173962
    geometry=bn.Geometry(source)
    assert geometry.sizes==[64,64,64,128,128,128,256,256,256,512,512,512]
    uniform=geometry.profile(round(.5*geometry.count(geometry.sizes)),0)
    target=geometry.count(uniform)
    tilt=geometry.profile(target,.30)
    assert target==geometry.count(uniform) and geometry.count(tilt)<=target
    x=torch.randn(2,3,32,32)
    with torch.inference_mode():
        raw=source(x)
        identity,diagnostics=bn.fold(source,geometry.sizes)
        assert torch.equal(raw,identity(x))
        assert sum(p.numel() for p in identity.parameters())==geometry.count(geometry.sizes)
        model,d=bn.fold(source,uniform)
        y=model(x)
    assert y.shape==(2,10) and torch.isfinite(y).all()
    assert torch.equal(model.fc.bias,source.fc.bias)
    assert sum(p.numel() for p in model.parameters())==target
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in source.state_dict().items())
    aug=augment_uint8(images.raw,parts['A'])
    assert torch.equal(aug[:2],augment_uint8(images.raw,parts['A'][:2]))
    assert tensor_hash(aug)==tensor_hash(aug.clone())
    assert torch.equal(normalize(torch.from_numpy(images.raw[parts['A'][:3]].reshape(-1,3,32,32).copy())),
                       images.inputs(parts['A'][:3]))
    teacher=torch.tensor([[3.,1.,0.],[0.,2.,5.]])
    assert measure_values(teacher,teacher)['disagreement']==0
    assert measure_values(teacher,torch.tensor([[0.,3.,1.],[5.,2.,0.]]))['disagreement']==1
    # Reguła wyboru: malejąca atrapa disagreement musi dać50 unikalnych ocen.
    trace_seen=[]
    def fake(widths):
        trace_seen.append(tuple(widths))
        return dict(kl=float(len(trace_seen)),disagreement=1.0/(1+len(trace_seen)),teacher_ce=float(len(trace_seen)))
    chosen,path,trace=optimize(geometry,target,uniform,tilt,fake,'disagreement')
    assert len(trace)==50 and len({tuple(r['widths']) for r in trace})==50
    assert chosen['disagreement']<=min(r['disagreement'] for r in trace)+1e-12
    result=dict(status='passed',dataset='CIFAR-10',params_teacher=11173962,uniform_params=target,
                tilt_params=geometry.count(tilt),augmented_views_sha256=tensor_hash(aug),
                proposal_replay=len(trace),inputs=inputs,
                source_hashes={p.name:digest(p) for p in HERE.glob('*.py')},
                protocol_sha256=digest(HERE/'OPIS.md'))
    write(HERE/'check.json',result)
    print({k:v for k,v in result.items() if k not in ('inputs','source_hashes')})

if __name__=='__main__':
    main()
