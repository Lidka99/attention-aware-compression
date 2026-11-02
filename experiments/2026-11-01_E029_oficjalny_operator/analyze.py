"""E029: audyt zgodności operatorów i alokacji na naszych wagach; OPIS.md.
Wejście zakończony run; wyjście analysis JSON/CSV/Markdown, bez GPU.
"""
import argparse
import csv
from datetime import datetime,timezone
from common import *


def stats(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),positive=sum(v>0 for v in values))


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
    items={};hashes={};rows=[];comparisons=[];controls=[]
    for ds in ('cifar10','cifar100'):
        for seed in range(42,46):
            p=run/f'seed_{seed}'/ds/'summary.json';s=read(p);hashes[str(p.relative_to(run))]=digest(p)
            assert s['status']=='completed' and s['gpu']==seed-42
            assert not s['official_test_used'] and not s['reserve_used']
            assert s['architecture_max_abs']<=1e-4
            items[(ds,seed)]=s
            for name in VARIANTS:
                expected=s['plan']['tilt_params'] if name.endswith('_T') else s['plan']['target']
                assert s['params'][name]==expected
                for mode,m in s['metrics'][name].items():
                    if mode!='PRE':
                        assert s['calibrations'][name][mode]['params_unchanged']
                    rows.append(dict(dataset=ds,seed=seed,variant=name,mode=mode,top1=100*m['top1'],nll=m['nll'],params=s['params'][name]))
            for mode,m in s['controls'].items():
                controls.append(dict(dataset=ds,seed=seed,mode=mode,top1=100*m['top1'],nll=m['nll']))
            for name,compare in s['comparisons'].items():
                for mode,l in compare['logits'].items():
                    comparisons.append(dict(dataset=ds,seed=seed,variant=name,reference=compare['reference'],mode=mode,
                                            top1_delta_pp=l['top1_delta_pp'],logit_max_abs=l['max_abs'],logit_rmse=l['rmse'],
                                            agreement=l['agreement'],param_max_abs=compare['parameters']['max_abs'],
                                            param_relative_l2=compare['parameters']['relative_l2'],min_ari=min(compare['cluster_ari']),
                                            labels_exact=sum(compare['labels_exact'])))
    assert len(rows)==168 and len(controls)==24
    averages=[];gates=[];allocation=[];bias=[]
    for ds in ('cifar10','cifar100'):
        for variant in VARIANTS:
            means={mode:stats([r['top1'] for r in rows if r['dataset']==ds and r['variant']==variant and r['mode']==mode]) for mode in ('PRE','CMA','UP_REPAIR')}
            averages.append(dict(dataset=ds,variant=variant,pre=means['PRE']['mean'],cma=means['CMA']['mean'],cma_sd=means['CMA']['sd'],
                                 upstream_repair=means['UP_REPAIR']['mean'],upstream_sd=means['UP_REPAIR']['sd']))
        for variant in ('LEGACY_U','LOCAL_U','LEGACY_T','LOCAL_T'):
            sub=[r for r in comparisons if r['dataset']==ds and r['variant']==variant and r['mode']=='CMA']
            vals=[abs(r['top1_delta_pp']) for r in sub]
            gates.append(dict(dataset=ds,variant=variant,mean_abs_pp=float(np.mean(vals)),max_abs_pp=max(vals),
                              quality_gate=np.mean(vals)<=.5 and max(vals)<=1,
                              min_cluster_ari=min(r['min_ari'] for r in sub),max_parameter_difference=max(r['param_max_abs'] for r in sub),
                              max_logit_difference=max(r['logit_max_abs'] for r in sub)))
        for mode in ('PRE','CMA','UP_REPAIR'):
            def delta(a,b):
                return [100*(items[(ds,seed)]['metrics'][a][mode]['top1']-items[(ds,seed)]['metrics'][b][mode]['top1']) for seed in range(42,46)]
            for family in ('UP_FIXED','LEGACY','LOCAL'):
                st=stats(delta(family+'_T',family+'_U'))
                allocation.append(dict(dataset=ds,operator=family,mode=mode,**st,direction_gate=st['mean']>0 and st['positive']>=3))
            bias.append(dict(dataset=ds,mode=mode,**stats(delta('UP_FIXED_U','UP_RAW_U'))))
    result=dict(averages=averages,quality_gates=gates,allocation=allocation,bias=bias,source_hashes=hashes,
                wall_seconds=manifest['wall_seconds'],upstream_commit=COMMIT)
    out=run/('analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    write(out/'analysis.json',result)
    for name,rs in [('metrics',rows),('comparisons',comparisons),('controls',controls)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    text='\n\n'.join(['# E029 — tabele',table(averages),'## Bramki zgodności jakości (CMA)',table(gates),
                       '## TILT−UNIFORM',table(allocation),'## Zachowany−losowy bias FC',table(bias),f"Czas: {manifest['wall_seconds']:.2f}s."])
    (out/'TABLES.md').write_text(text+'\n');print(text);print('ANALYSIS='+str(out))

if __name__=='__main__':
    main()
