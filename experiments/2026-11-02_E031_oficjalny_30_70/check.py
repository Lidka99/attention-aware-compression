"""E031: CPU kontrole profili 30/70, operatora upstream i wariantów; OPIS.md.
Wyjście check.json przed GPU; bez ocen naukowych dev.
"""
from common import *
from models.resnet import get_axis_to_perm_ResNet18
from utils.weight_clustering import axes2perm_to_perm2axes


def clusters_ok(diag):
    for group in diag:
        labels=group['labels']
        k=group.get('k',group.get('width'))
        assert len(set(labels))==k


def historical(ds,tag,plan):
    result={}
    for variant in ('uniform','tilt_p030'):
        item=read(HIST[ds]/f'{tag}_{variant}/summary.json')['allocation']
        result[variant]=dict(widths=item['widths'],params=item['params'],
                             target_params=item['target_params'])
    assert result['uniform']['widths']==plan['uniform'],(ds,tag)
    diffs=[abs(a-b) for a,b in zip(result['tilt_p030']['widths'],plan['tilt'])]
    result['tilt_width_max_abs_diff']=max(diffs)
    result['tilt_widths_equal']=not any(diffs)
    return result


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(42)
    inputs=provenance()
    parts=indices_all()
    for ds in DATASETS:
        assert len(parts[ds]['A'])==4096
        assert not set(parts[ds]['A']).intersection(parts[ds]['dev'])
    split=dt.load_split()
    assert set(parts['tiny']['dev'])==set(split['splits']['dev'])
    assert set(parts['tiny']['A']).issubset(set(split['splits']['train']))
    assert not set(parts['tiny']['A']).intersection(split['splits']['dev'])
    assert len(split['splits']['reserve'])==10000

    plans={};architecture_errors={};historical_rows={};sizes={};bias_changed={}
    for ds in DATASETS:
        tv,up=load_source(ds,42)
        sizes[ds]=sum(p.numel() for p in tv.parameters())
        geo=bn.Geometry(tv)
        groups=axes2perm_to_perm2axes(get_axis_to_perm_ResNet18(override=False))
        assert len(groups)==12
        for pairs,name in zip(groups.values(),geo.names):
            assert pairs==geo.members[name]
        x=torch.randn(3,3,32,32) if ds!='tiny' else torch.randn(3,3,64,64)
        with torch.inference_mode():
            y=tv(x)
        assert y.shape[0]==3 and torch.isfinite(y).all()
        if up is not None:
            with torch.inference_mode():
                diff=float((tv(x)-up(x)).abs().max())
            assert diff<=1e-4
            architecture_errors[ds]=diff
        for seed in range(43,46):
            other,_=load_source(ds,seed)
            assert sum(p.numel() for p in other.parameters())==sizes[ds]
        plans[ds]={}
        for budget in BUDGETS:
            tag=TAGS[budget]
            models,plan=construct(ds,42,budget,tv,up)
            assert plan['nominal']==round((1-budget)*geo.count(geo.sizes))
            assert plan['tilt_params']<=plan['target']
            assert plan['tilt_params']>=.999*plan['target']
            assert len(plan['uniform'])==12 and len(plan['tilt'])==12
            for name,(model,diag) in models.items():
                assert sum(p.numel() for p in model.parameters())==(
                    plan['tilt_params'] if name.endswith('_T') else plan['target'])
                clusters_ok(diag)
                with torch.inference_mode():
                    z=model(x)
                assert z.shape[0]==3 and torch.isfinite(z).all()
            bias_changed[ds]=not torch.equal(models['UP_RAW_U'][0].fc.bias,tv.fc.bias)
            assert bias_changed[ds]
            historical_rows[f'{ds}/{tag}']=historical(ds,tag,plan)
            plans[ds][tag]=plan
    images,_,_=dt.load_tiny_train(dt.DATA_ROOT)
    small=e30.probe_token(images,parts['tiny']['A'][:8])
    assert small==e30.probe_token(images,parts['tiny']['A'][:8])
    assert small!=e30.probe_token(images,parts['tiny']['A'][:8][::-1])
    write(HERE/'check.json',dict(
        status='passed',upstream_commit=COMMIT,plans=plans,param_counts=sizes,
        architecture_logit_error=architecture_errors,historical_profile_comparison=historical_rows,
        strict_load=True,axes_identical=True,native_forward=True,all_variants_forward=True,
        native_random_bias=bias_changed,augmented_probe=small,inputs=inputs,
        protocol_sha256=digest(HERE/'OPIS.md'),
        source_hashes={p.name:digest(p) for p in HERE.glob('*.py')}))
    print('PASS',plans,'architecture errors',architecture_errors,
          'historical tilt equal',
          {k:v['tilt_widths_equal'] for k,v in historical_rows.items()},flush=True)


if __name__=='__main__':
    main()
