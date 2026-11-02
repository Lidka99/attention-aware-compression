"""E024: kontrole CPU operatora/budżetu/rozdzielenia danych; OPIS.md.
Wejście: istniejący seed42 i split; wyjście: check.json bez wyników dev.
"""
import ast
import torch
from data import HERE,CHECKPOINT_ROOT,CHECKPOINT_HASHES,digest,split_indices,write
from folding import Geometry,architecture,fold,reduce_axis


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source=architecture()
    path=CHECKPOINT_ROOT/'seed_42/last.pt'
    assert digest(path)==CHECKPOINT_HASHES[42]
    source.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    before={k:v.clone() for k,v in source.state_dict().items()}
    g=Geometry(source)
    assert g.sizes==[64,64,64,128,128,128,256,256,256,512,512,512]
    assert g.count(g.sizes)==11220132
    unchanged,_=fold(source,g.sizes)
    x=torch.randn(2,3,32,32)
    with torch.inference_mode():
        assert torch.equal(source(x),unchanged(x))
    tensor=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    labels=torch.tensor([0,0,1])
    assert torch.equal(reduce_axis(tensor,0,labels,2,False),torch.tensor([[4.,6.],[5.,6.]]))
    assert torch.equal(reduce_axis(tensor,0,labels,2,True),torch.tensor([[2.,3.],[5.,6.]]))
    uniform=g.profile(round(.5*g.count(g.sizes)),0)
    target=g.count(uniform)
    tilt=g.profile(target,.30)
    for widths in (uniform,tilt,[max(1,n//3) for n in g.sizes]):
        model,_=fold(source,widths,identity_labels=True)
        assert sum(p.numel() for p in model.parameters())==g.count(widths)
        with torch.inference_mode():
            y=model(x)
        assert y.shape==(2,100) and torch.isfinite(y).all()
        assert torch.equal(model.fc.bias,source.fc.bias)
    # Rzeczywiste PCA/HKMeans jednej grupy, bez danych i oceny naukowej.
    widths=g.sizes.copy();widths[0]=48
    model,_=fold(source,widths)
    with torch.inference_mode():
        assert torch.isfinite(model(x)).all()
    assert all(torch.equal(before[k],v) for k,v in source.state_dict().items())
    proposals=list(g.proposals(tilt,target,0))
    assert len(proposals)>=24 and len({tuple(w) for w in proposals})==len(proposals)
    assert all(.999*target<=g.count(w)<=target for w in proposals)
    parts=split_indices()
    # Kontrola braku importów eksperymentów lub manipulacji sys.path.
    for p in HERE.glob('*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert all('e020' not in name and 'e014' not in name for name in names)
        assert 'sys.path.insert' not in p.read_text() or p.name=='check.py'
    result=dict(status='passed',identity_exact=True,source_unchanged=True,fold_parameter_counts=True,
                residual_forward=True,merge_formula=True,real_clustering=True,split_disjoint=True,
                uniform=uniform,tilt=tilt,target=target,tilt_params=g.count(tilt),proposal_count=len(proposals),
                split_counts={k:len(v) for k,v in parts.items()},protocol_sha256=digest(HERE/'OPIS.md'),
                source_hashes={p.name:digest(p) for p in HERE.glob('*.py')})
    write(HERE/'check.json',result)
    print(result)

if __name__=='__main__':
    main()
