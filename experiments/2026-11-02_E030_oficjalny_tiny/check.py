"""E030: CPU kontrole Tiny, operatora upstream i wariantów; OPIS.md.
Wyjście check.json przed GPU. Bez ocen naukowych dev.
"""
from common import *
from models.resnet import get_axis_to_perm_ResNet18
from utils.weight_clustering import axes2perm_to_perm2axes


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(42)
    inputs=provenance()
    images,labels,classes=dt.load_tiny_train(dt.DATA_ROOT)
    assert images.shape==(100000,64,64,3) and len(classes)==200
    parts=indices(images)
    assert len(parts['A'])==4096 and len(parts['dev'])==10000
    assert not set(parts['A']).intersection(parts['dev'])
    split=dt.load_split()
    assert set(parts['A']).issubset(set(split['splits']['train']))
    assert set(parts['dev'])==set(split['splits']['dev'])
    tv=source(42)
    assert sum(p.numel() for p in tv.parameters())==11279112
    geo,u,t,ratio=e29.profiles(tv)
    groups=axes2perm_to_perm2axes(get_axis_to_perm_ResNet18(override=False))
    assert len(groups)==12
    for pairs,name in zip(groups.values(),geo.names):
        assert pairs==geo.members[name]
    raw,diag=e29.official(tv,ratio)
    assert sum(p.numel() for p in raw.parameters())==geo.count(u) and len(diag)==12
    assert not torch.equal(raw.fc.bias,tv.fc.bias)
    _,models,plan=construct(42)
    x=torch.randn(2,3,64,64)
    with torch.inference_mode():
        for name,(model,_) in models.items():
            y=model(x)
            assert y.shape==(2,200) and torch.isfinite(y).all()
    assert plan['target']==geo.count(u) and plan['tilt_params']<=plan['target']
    small=probe_token(images,parts['A'][:8])
    assert small==probe_token(images,parts['A'][:8])
    assert small!=probe_token(images,parts['A'][:8][::-1])
    probe=probe_token(images,parts['A'])
    plain=normalize(torch.from_numpy(images[parts['A'][:2]].reshape(-1,3,64,64).copy()).float()/255)
    assert plain.shape==(2,3,64,64) and torch.isfinite(plain).all()
    write(HERE/'check.json',dict(status='passed',upstream_commit='694691556e8314a835178a741938dd3d4d80c171',
          plans=plan,strict_load=True,axes_identical=True,all_variants_forward=True,native_random_bias=True,
          augmented_probe=probe,inputs=inputs,protocol_sha256=digest(HERE/'OPIS.md'),
          source_hashes={p.name:digest(p) for p in HERE.glob('*.py')}))
    print('PASS',plan,'probe',probe[:16],flush=True)

if __name__=='__main__':
    main()
