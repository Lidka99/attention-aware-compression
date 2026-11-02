"""E029: CPU strict-load, zgodność architektur/osi i plan budżetu; OPIS.md.
Bez ocen dev ani nowego treningu. Wyjście check.json przed GPU.
"""
from common import *


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(42)
    sources=provenance();plans={};errors={}
    x=torch.randn(3,3,32,32)
    for ds in ('cifar10','cifar100'):
        tv,up=source(ds,42)
        assert set(tv.state_dict())==set(up.state_dict())
        assert all(torch.equal(v,up.state_dict()[k]) for k,v in tv.state_dict().items())
        with torch.inference_mode():
            diff=float((tv(x)-up(x)).abs().max())
        assert diff<=1e-4
        geo,u,t,ratio=profiles(tv)
        orig_axes=get_axis_to_perm_ResNet18(override=False)
        groups=axes2perm_to_perm2axes(orig_axes)
        assert len(groups)==12
        for (_,pairs),name in zip(groups.items(),geo.names):
            assert pairs==geo.members[name]
        old_groups=legacy.load_reference_folding().resnet18_groups()
        for official_axes,old in zip(groups.values(),old_groups):
            assert official_axes==[(axis.full_name,axis.axis) for axis in old.axes]
        plans[ds]=dict(uniform=u,tilt=t,ratio=ratio,target=geo.count(u),tilt_params=geo.count(t))
        errors[ds]=diff
    # Native operator na prawdziwych wagach, bez obrazów/oceny naukowej.
    tv,up=source('cifar10',42)
    geo,u,_,ratio=profiles(tv)
    compressed,diag=official(up,ratio)
    assert sum(p.numel() for p in compressed.parameters())==geo.count(u)
    assert len(diag)==12 and all(len(set(g['labels']))==g['k'] for g in diag)
    with torch.inference_mode():
        output=compressed(x)
    assert output.shape==(3,10) and torch.isfinite(output).all()
    # Konstruktor natywny zmienia FC bias: kontrola zachowania kodu, nie poprawka.
    bias_changed=not torch.equal(compressed.fc.bias,tv.fc.bias)
    assert bias_changed
    # Wszystkie ścieżki adaptera, w tym TILT, przed uruchomieniem workerów.
    _,_,models,plan=construct('cifar100',42)
    with torch.inference_mode():
        for name,(model,_) in models.items():
            y=model(x)
            assert y.shape==(3,100) and torch.isfinite(y).all()
    assert Path(ResNet18_ModelFolding.apply.__code__.co_filename).resolve()==UP/'compression/fold.py'
    write(HERE/'check.json',dict(status='passed',upstream_commit=COMMIT,plans=plans,architecture_logit_error=errors,
          strict_load=True,axes_identical=True,native_forward=True,all_variants_forward=True,native_random_bias=bias_changed,
          inputs=sources,protocol_sha256=digest(HERE/'OPIS.md'),source_hashes={p.name:digest(p) for p in HERE.glob('*.py')}))
    print('PASS',plans,'architecture errors',errors,'native bias changed',bias_changed,flush=True)

if __name__=='__main__':
    main()
