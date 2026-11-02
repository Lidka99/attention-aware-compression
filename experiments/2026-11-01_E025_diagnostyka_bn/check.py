"""E025: kontrole CPU formuł BN, augmentacji i izolacji danych; OPIS.md.
Wyjście check.json przed oceną dev. Nie wykonuje pomiarów naukowych.
"""
import copy
import numpy as np
import torch
from torch import nn
from common import *


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    provenance=inputs_manifest()
    ids=indices()
    images=Images()
    views=Views(images,ids['A'][:5])
    assert torch.equal(normalize(views.plain),images.inputs(ids['A'][:5]))
    reverse=augment_uint8(images.raw,ids['A'][:5][::-1])
    assert torch.equal(views.aug,reverse.flip(0))
    assert torch.equal(views.aug[:2],augment_uint8(images.raw,ids['A'][:2]))
    zeros=np.zeros((1,3072),dtype=np.uint8)
    assert (augment_uint8(zeros,[0])==0).all()
    batches=[torch.tensor([0.,2.]).reshape(2,1,1,1),torch.tensor([2.,4.]).reshape(2,1,1,1)]
    base=nn.Sequential(nn.BatchNorm2d(1)).eval().requires_grad_(False)
    base[0].running_mean.fill_(10);base[0].running_var.fill_(4);base[0].num_batches_tracked.fill_(7)
    expected={'CMA_reset':(2.,2.,2),'EMA_reset':(.39,1.19,2),'EMA_retain':(8.49,3.62,9)}
    for est,(mean,var,count) in expected.items():
        model=copy.deepcopy(base)
        meta=calibrate(model,batches,est,'cpu')
        assert torch.allclose(model[0].running_mean,torch.tensor([mean]),atol=1e-5)
        assert torch.allclose(model[0].running_var,torch.tensor([var]),atol=1e-5)
        assert int(model[0].num_batches_tracked)==count and meta['params_unchanged']
    assert float(base[0].running_mean)==10 and float(base[0].running_var)==4
    # CMA reset dokładnie taki jak w E024, na identycznych batchach.
    left=copy.deepcopy(base);right=copy.deepcopy(base)
    calibrate(left,batches,'CMA_reset','cpu')
    with torch.inference_mode():
        right.eval();right[0].reset_running_stats();right[0].momentum=None;right[0].train()
        for batch in batches:
            right(batch)
        right.eval()
    assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())
    result=dict(status='passed',analytic_bn=True,source_unchanged=True,augmentation_before_normalization=True,
                augmentation_index_deterministic=True,plain_identical=True,split_disjoint=True,cma_e024_identical=True,
                new_cases=176,reused_cases=16,protocol_sha256=digest(HERE/'OPIS.md'),inputs=provenance,
                source_hashes={p.name:digest(p) for p in HERE.glob('*.py')})
    write(HERE/'check.json',result)
    print({k:v for k,v in result.items() if k not in ('inputs','source_hashes')})

if __name__=='__main__':
    main()
