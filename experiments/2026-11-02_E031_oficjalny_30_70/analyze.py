"""E031: analiza audytu 30/70; OPIS.md.
Wejście zakończony run; wyjście analysis JSON/CSV/Markdown, bez GPU.
"""
import argparse
import csv
from datetime import datetime,timezone
from common import *


def stats(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),
                positive=sum(v>0 for v in values),n=len(values))


def table(rows):
    keys=list(rows[0])
    def fmt(v):
        return f'{v:.5f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
                     ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path,required=True)
    run=ap.parse_args().run_dir.resolve();manifest=read(run/'manifest.json')
    assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
    assert manifest['protocol_sha256']==digest(HERE/'OPIS.md') and manifest['inputs']==provenance()
    assert manifest['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    check=read(HERE/'check.json')
    assert check['status']=='passed' and check['protocol_sha256']==manifest['protocol_sha256']
    items={};hashes={};rows=[];comparisons=[];controls=[]
    for ds in DATASETS:
        for budget in BUDGETS:
            tag=TAGS[budget]
            for seed in range(42,46):
                p=run/f'seed_{seed}'/ds/tag/'summary.json';s=read(p)
                hashes[str(p.relative_to(run))]=digest(p)
                assert s['status']=='completed' and s['seed']==seed and s['gpu']==seed-42
                assert s['dataset']==ds and s['tag']==tag and s['budget']==budget
                assert not s['official_test_used'] and not s['reserve_used']
                assert s['controls_sha256']==digest(run/f'seed_{seed}'/f'controls_{ds}.json')
                items[(ds,tag,seed)]=s
                for name in VARIANTS:
                    expected=s['plan']['tilt_params'] if name.endswith('_T') else s['plan']['target']
                    assert s['params'][name]==expected
                    for mode,m in s['metrics'][name].items():
                        if mode!='PRE':
                            assert s['calibrations'][name][mode]['params_unchanged']
                        rows.append(dict(dataset=ds,tag=tag,budget=budget,seed=seed,variant=name,
                                         mode=mode,top1=100*m['top1'],nll=m['nll'],
                                         params=s['params'][name]))
                for name,compare in s['comparisons'].items():
                    for mode,l in compare['logits'].items():
                        comparisons.append(dict(dataset=ds,tag=tag,budget=budget,seed=seed,
                                                variant=name,reference=compare['reference'],mode=mode,
                                                top1_delta_pp=l['top1_delta_pp'],logit_max_abs=l['max_abs'],
                                                logit_rmse=l['rmse'],agreement=l['agreement'],
                                                param_max_abs=compare['parameters']['max_abs'],
                                                param_relative_l2=compare['parameters']['relative_l2'],
                                                min_ari=min(compare['cluster_ari']),
                                                labels_exact=sum(compare['labels_exact'])))
    for ds in DATASETS:
        for seed in range(42,46):
            p=run/f'seed_{seed}'/f'controls_{ds}.json';c=read(p)
            hashes[str(p.relative_to(run))]=digest(p)
            assert c['status']=='completed' and c['dataset']==ds and c['seed']==seed
            for mode,m in c['controls'].items():
                controls.append(dict(dataset=ds,seed=seed,mode=mode,top1=100*m['top1'],nll=m['nll']))
    assert len(rows)==504 and len(comparisons)==360 and len(controls)==36
    averages=[];gates=[];allocation=[];bias=[];bn_delta=[]
    for ds in DATASETS:
        for budget in BUDGETS:
            tag=TAGS[budget]
            for variant in VARIANTS:
                means={mode:stats([r['top1'] for r in rows if r['dataset']==ds and r['tag']==tag
                                   and r['variant']==variant and r['mode']==mode]) for mode in MODES}
                averages.append(dict(dataset=ds,tag=tag,budget=budget,variant=variant,
                                     pre=means['PRE']['mean'],cma=means['CMA']['mean'],
                                     cma_sd=means['CMA']['sd'],upstream_repair=means['UP_REPAIR']['mean'],
                                     upstream_sd=means['UP_REPAIR']['sd']))
            for variant in ('LEGACY_U','LOCAL_U','LEGACY_T','LOCAL_T'):
                sub=[r for r in comparisons if r['dataset']==ds and r['tag']==tag
                     and r['variant']==variant and r['mode']=='CMA']
                vals=[abs(r['top1_delta_pp']) for r in sub]
                gates.append(dict(dataset=ds,tag=tag,variant=variant,mean_abs_pp=float(np.mean(vals)),
                                  max_abs_pp=max(vals),
                                  quality_gate=np.mean(vals)<=.5 and max(vals)<=1,
                                  min_cluster_ari=min(r['min_ari'] for r in sub),
                                  max_parameter_difference=max(r['param_max_abs'] for r in sub),
                                  max_logit_difference=max(r['logit_max_abs'] for r in sub)))
            for mode in MODES:
                def delta(a,b):
                    return [100*(items[(ds,tag,seed)]['metrics'][a][mode]['top1']
                                 -items[(ds,tag,seed)]['metrics'][b][mode]['top1'])
                            for seed in range(42,46)]
                for family in ('UP_FIXED','LEGACY','LOCAL'):
                    st=stats(delta(family+'_T',family+'_U'))
                    allocation.append(dict(dataset=ds,tag=tag,budget=budget,operator=family,mode=mode,
                                           mean=st['mean'],sd=st['sd'],positive=st['positive'],
                                           direction_gate=st['mean']>0 and st['positive']>=3))
                bias.append(dict(dataset=ds,tag=tag,budget=budget,mode=mode,
                                 **stats(delta('UP_FIXED_U','UP_RAW_U'))))
            for variant in VARIANTS:
                diffs=[100*(items[(ds,tag,seed)]['metrics'][variant]['UP_REPAIR']['top1']
                            -items[(ds,tag,seed)]['metrics'][variant]['CMA']['top1'])
                       for seed in range(42,46)]
                bn_delta.append(dict(dataset=ds,tag=tag,variant=variant,**stats(diffs)))
    teacher=[]
    for ds in DATASETS:
        for mode in MODES:
            st=stats([r['top1'] for r in controls if r['dataset']==ds and r['mode']==mode])
            teacher.append(dict(dataset=ds,mode=mode,mean=st['mean'],sd=st['sd']))
    upstream=[a for a in allocation if a['operator']=='UP_FIXED' and a['mode'] in ('CMA','UP_REPAIR')]
    quality_fail=[g for g in gates if not g['quality_gate']]
    allocation_fail=[a for a in upstream if not a['direction_gate']]
    criteria=dict(operator_agreement_all=not quality_fail,
                  operator_agreement_gates=dict(passed=len(gates)-len(quality_fail),total=len(gates)),
                  allocation_upstream_all=not allocation_fail,
                  allocation_upstream_gates=dict(passed=len(upstream)-len(allocation_fail),total=len(upstream)),
                  allocation_upstream_CMA_all=all(a['direction_gate'] for a in upstream if a['mode']=='CMA'),
                  allocation_upstream_UP_REPAIR_all=all(a['direction_gate'] for a in upstream if a['mode']=='UP_REPAIR'),
                  quality_failures=quality_fail,allocation_failures=allocation_fail)
    result=dict(averages=averages,quality_gates=gates,allocation=allocation,bias=bias,
                bn_procedure_delta=bn_delta,teacher=teacher,criteria=criteria,
                plans=read(run/'plans.json'),
                historical_profile_comparison=check['historical_profile_comparison'],
                source_hashes=hashes,wall_seconds=manifest['wall_seconds'],
                upstream_commit=COMMIT,
                limits=dict(rows=len(rows),comparisons=len(comparisons),controls=len(controls)))
    out=run/('analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    write(out/'analysis.json',result)
    for name,rs in [('metrics',rows),('comparisons',comparisons),('controls',controls),
                    ('allocation',allocation),('quality_gates',gates)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    text='\n\n'.join([
        '# E031 — tabele',
        table(gates),
        '## TILT−UNIFORM (wszystkie operatory)',
        table(allocation),
        '## Różnica procedury BN: UP_REPAIR−CMA',
        table(bn_delta),
        '## Zachowany−losowy bias FC',
        table(bias),
        '## Teacher',
        table(teacher),
        '## Kryteria',
        table([dict(name=k,value=str(v)) for k,v in criteria.items() if not isinstance(v,(dict,list))]),
        f"Czas: {manifest['wall_seconds']:.2f}s."])
    (out/'TABLES.md').write_text(text+'\n')
    print(text)
    print('CRITERIA',criteria)
    print('ANALYSIS='+str(out))


if __name__=='__main__':
    main()
