"""E026: replay wyszukiwania E024 i kontrola wejść bez GPU; OPIS.md.
Wyjście check.json. Nie wykonuje nowych ocen modeli ani nie dobiera celu.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from support import *


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    inputs=sources()
    geometry=bn.Geometry(bn.architecture())
    replay=[]
    for seed in range(42,46):
        old=bn.read(bn.SOURCE/f'seed_{seed}/trace.json')
        selection=bn.read(bn.SOURCE/f'seed_{seed}/selection.json')
        seen=[]
        def measure(widths):
            row=old[len(seen)]
            assert widths==row['widths']
            seen.append(widths)
            return dict(kl_B=row['kl_B'])
        chosen,path,trace=optimize(geometry,selection['target'],selection['uniform'],selection['tilt'],measure)
        assert len(seen)==len(old) and chosen['widths']==selection['widths'] and path==selection['trajectory']
        assert all(r['origin']==s['origin'] and r['kl_B']==s['kl_B'] for r,s in zip(trace,old))
        replay.append(dict(seed=seed,count=len(seen),same_order=True,same_selection=True))
    ids=bn.indices()
    images=bn.Images()
    aug=bn.augment_uint8(images.raw,ids['A'])
    expected=bn.read(REF/'seed_42/view_hashes.json')['A']['aug']
    assert bn.tensor_hash(aug)==expected
    for seed in range(42,46):
        assert bn.read(REF/f'seed_{seed}/view_hashes.json')['A']['aug']==expected
    # Wyłącznie wejścia BN zmienione; estymator jest CMA jak w E024.
    model=nn.Sequential(nn.BatchNorm2d(1)).eval().requires_grad_(False)
    original=copy.deepcopy(model)
    batches=[torch.tensor([0.,2.]).reshape(2,1,1,1),torch.tensor([2.,4.]).reshape(2,1,1,1)]
    bn.calibrate(model,batches,'CMA_reset','cpu')
    assert torch.allclose(model[0].running_mean,torch.tensor([2.]))
    assert torch.allclose(model[0].running_var,torch.tensor([2.]))
    assert bn.weights_hash(model)==bn.weights_hash(original)
    p=torch.tensor([[.7,.2,.1]],dtype=torch.float64)
    q=torch.tensor([[.5,.3,.2]],dtype=torch.float64)
    assert torch.allclose(F.kl_div(q.log(),p,reduction='batchmean'),(p*(p.log()-q.log())).sum())
    result=dict(status='passed',replay=replay,augmented_views_sha256=expected,kl_direction=True,cma_formula=True,
                inputs=inputs,source_hashes={p.name:digest(p) for p in HERE.glob('*.py')},protocol_sha256=digest(HERE/'OPIS.md'))
    write(HERE/'check.json',result)
    print({k:v for k,v in result.items() if k not in ('inputs','source_hashes')})

if __name__=='__main__':
    main()
