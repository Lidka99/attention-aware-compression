"""E027: kontrole CPU trzech miar i izolacji dev; OPIS.md.
Wyjście check.json przed uruchomieniem GPU. Bez ocen naukowych modeli.
"""
import ast
import copy
import torch
from torch.nn import functional as F
from support import *


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    inputs=sources()
    # Formuły: zgodność, kierunek KL, teacher_ce.
    teacher=torch.tensor([[3.,1.,0.],[0.,2.,5.]])
    same=teacher.clone()
    values=measure_values(teacher,same)
    assert abs(values['kl'])<1e-6 and values['disagreement']==0
    assert abs(values['teacher_ce']-F.cross_entropy(same,teacher.argmax(1)).item())<1e-12
    other=torch.tensor([[0.,3.,1.],[5.,2.,0.]])
    trained=measure_values(teacher,other)
    assert trained['disagreement']==1.0
    p=F.softmax(teacher,dim=1);q=F.softmax(other,dim=1)
    manual=float((p*(p.log()-q.log())).sum()/2)
    assert abs(trained['kl']-manual)<1e-6
    # KL jest nieujemne i nie myli kierunku przy identycznych logitach.
    assert measure_values(teacher,teacher+0.5*torch.randn_like(teacher))['kl']>0
    # Deterministyczne propozycje i limit; pomiar atrapą, bez modeli.
    geometry=bn.Geometry(bn.architecture())
    ids=bn.read(bn.SOURCE/'indices.json')
    images=bn.Images()
    aug=bn.augment_uint8(images.raw,ids['A'])
    view=bn.read(e26.REF/'seed_42/view_hashes.json')['A']['aug']
    assert bn.tensor_hash(aug)==view
    uniform=geometry.profile(round(.5*geometry.count(geometry.sizes)),0)
    tilt=geometry.profile(geometry.count(uniform),.30)
    seen=[]
    def fake(widths):
        seen.append(tuple(widths))
        return dict(kl=float(len(seen)),disagreement=1.0/(1+len(seen)),teacher_ce=float(len(seen))+0.25)
    chosen,trajectory,trace=optimize(geometry,geometry.count(uniform),uniform,tilt,fake)
    assert len(trace)==50 and chosen['disagreement']<=min(r['disagreement'] for r in trace)+1e-12
    assert len({tuple(r['widths']) for r in trace})==50
    # Kontrola braku etykiet/dev w module wyboru.
    tree=ast.parse((HERE/'support.py').read_text())
    names={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
    assert 'dev' not in names and 'labels' not in names
    result=dict(status='passed',objectives=OBJECTIVES,kl_direction=True,disagreement_identity=True,
                teacher_ce_formula=True,proposal_count=len(trace),augmented_views_sha256=view,
                inputs=inputs,source_hashes={p.name:digest(p) for p in HERE.glob('*.py')},
                protocol_sha256=digest(HERE/'OPIS.md'))
    write(HERE/'check.json',result)
    print({k:v for k,v in result.items() if k not in ('inputs','source_hashes')})

if __name__=='__main__':
    main()
