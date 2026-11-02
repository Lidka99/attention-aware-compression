"""E025: analiza czynnikowa sparowana po seedach; OPIS.md.
Wejście: run176 nowych+16 reuse; wyjście: nowe analysis JSON/CSV/Markdown.
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
        return f'{v:.3f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
                     ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    run=args.run_dir.resolve()
    manifest=read(run/'manifest.json')
    assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
    assert manifest['protocol_sha256']==digest(HERE/'OPIS.md')
    assert manifest['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    assert manifest['inputs']==inputs_manifest()
    assert manifest['indices_sha256']==digest(run/'indices.json')
    rows=[];index={};source_hashes={};baselines={};view_hashes=[]
    for seed in range(42,46):
        folder=run/f'seed_{seed}'
        state=read(folder/'status.json')
        assert state['status']=='completed' and state['gpu']==seed-42 and state['new_completed']==44 and state['reused']==4
        view_hashes.append(read(folder/'view_hashes.json'))
        for model in MODELS:
            baselines[(seed,model)]=read(folder/f'{model}_baseline.json')['metrics']
        paths=sorted((folder/'items').glob('*.json'))
        assert len(paths)==48
        parameter_hashes={model:set() for model in MODELS}
        for p in paths:
            s=read(p)
            assert s['status']=='completed' and not s['official_test_used'] and not s['reserve_used']
            cell=s['cell']
            assert s['reused']==(cell['name']==REUSED)
            if not s['reused']:
                bn=s['bn'];n=4096 if cell['size']=='A' else 35904
                assert bn['samples']==n and bn['params_unchanged']
                assert bn['batches']==(32 if n==4096 else 281)
                assert bn['last_batch']==(128 if n==4096 else 64)
                for layer in bn['layers']:
                    assert layer['counter_end']-layer['counter_start']==bn['batches']
                parameter_hashes[s['model']].add(bn['parameters_sha256'])
            else:
                assert digest(s['source'])==s['source_sha256']
            row=dict(seed=seed,model=s['model'],cell=cell['name'],size=cell['size'],view=cell['view'],estimator=cell['estimator'],
                     top1=100*s['metrics']['top1'],nll=s['metrics']['nll'],kl_C=s['kl_C'],reused=s['reused'],params=s['params'],seconds=s['runtime_seconds'])
            key=(seed,s['model'],cell['name'])
            assert key not in index
            index[key]=row;rows.append(row)
            source_hashes[str(p.relative_to(run))]=digest(p)
        assert all(len(v)==1 for v in parameter_hashes.values())
    assert len(rows)==192 and sum(r['reused'] for r in rows)==16
    assert all(v==view_hashes[0] for v in view_hashes)
    summary=[];details={};restored=[]
    for cell in cells():
        name=cell['name'];data={}
        for model in MODELS:
            rs=[index[(seed,model,name)] for seed in range(42,46)]
            data[model]={metric:stats([r[metric] for r in rs]) for metric in ('top1','nll')}
            data[model]['kl_C']=None if any(r['kl_C'] is None for r in rs) else stats([r['kl_C'] for r in rs])
            data[model]['gain_vs_E024']=stats([r['top1']-index[(r['seed'],model,REUSED)]['top1'] for r in rs])
        losses=[100*baselines[(seed,'teacher')]['top1']-index[(seed,'teacher',name)]['top1'] for seed in range(42,46)]
        restore=float(np.mean(losses))<=1 and max(losses)<=1.5
        if restore:
            restored.append(name)
        pairs={other:stats([index[(seed,'SEARCH',name)]['top1']-index[(seed,other,name)]['top1'] for seed in range(42,46)])
               for other in ('UNIFORM','TILT')}
        details[name]=dict(models=data,teacher_losses=losses,teacher_restored=restore,search_differences=pairs)
        summary.append(dict(cell=name,teacher=data['teacher']['top1']['mean'],teacher_sd=data['teacher']['top1']['sd'],
                            teacher_loss=float(np.mean(losses)),restored=restore,UNIFORM=data['UNIFORM']['top1']['mean'],
                            TILT=data['TILT']['top1']['mean'],SEARCH=data['SEARCH']['top1']['mean'],search_minus_tilt=pairs['TILT']['mean']))
    effects=[]
    def contrast(model,label,first,second):
        values=[index[(seed,model,second)]['top1']-index[(seed,model,first)]['top1'] for seed in range(42,46)]
        s=stats(values)
        effects.append(dict(model=model,contrast=label,from_cell=first,to_cell=second,mean=s['mean'],sd=s['sd'],positive=s['positive'],
                            practical=s['mean']>=1 and s['positive']>=3))
    for model in MODELS:
        for view in ('plain','aug'):
            for est in ESTIMATORS:
                contrast(model,'size',f'A_{view}_{est}',f'L_{view}_{est}')
        for size in ('A','L'):
            for est in ESTIMATORS:
                contrast(model,'augmentation',f'{size}_plain_{est}',f'{size}_aug_{est}')
            for view in ('plain','aug'):
                contrast(model,'estimator',f'{size}_{view}_CMA_reset',f'{size}_{view}_EMA_reset')
                contrast(model,'initialization',f'{size}_{view}_EMA_reset',f'{size}_{view}_EMA_retain')
    primary=[r for r in effects if r['model']=='teacher' and r['from_cell']==REUSED]
    reset=[r for r in effects if r['model']=='teacher' and r['contrast']=='initialization']
    result=dict(summary=summary,details=details,effects=effects,primary=primary,restored_cells=restored,source_hashes=source_hashes,
                integrity=dict(new=176,reused=16,total=192,view_hashes_identical=True,exit_codes=manifest['exit_codes']),
                wall_seconds=manifest['wall_seconds'],protocol_sha256=manifest['protocol_sha256'])
    out=run/('analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    write(out/'analysis.json',result)
    for name,rs in [('metrics',rows),('summary',summary),('effects',effects)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    text='\n\n'.join(['# E025 — tabele',table(summary),'## Primary teacher',table(primary),'## Reset teacher',table(reset),
                       '## Komórki odtwarzające teacher',str(restored),f"Czas: {manifest['wall_seconds']:.2f}s;176 nowych+16 reused."])
    (out/'TABLES.md').write_text(text+'\n')
    print(text)
    print('ANALYSIS='+str(out))

if __name__=='__main__':
    main()
