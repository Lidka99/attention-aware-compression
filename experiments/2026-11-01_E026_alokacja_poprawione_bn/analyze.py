"""E026: analiza czterech seedów z zamrożonymi kryteriami; OPIS.md.
Wejście zakończony run; wyjście analysis JSON/CSV/Markdown, bez GPU.
"""
import argparse
import csv
from datetime import datetime,timezone
import numpy as np
from support import *


def stats(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),positive=sum(x>0 for x in values))


def table(rows):
    keys=list(rows[0])
    def fmt(v):
        return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
                     ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    run=ap.parse_args().run_dir.resolve()
    manifest=bn.read(run/'manifest.json')
    assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
    assert manifest['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    assert manifest['inputs']==sources() and manifest['protocol_sha256']==digest(HERE/'OPIS.md')
    assert digest(run/'indices.json')==manifest['indices_sha256']
    rows=[];pairs=[];profiles=[];provenance={}
    for seed in range(42,46):
        folder=run/f'seed_{seed}'
        s=bn.read(folder/'summary.json');selection=bn.read(folder/'selection.json');trace=bn.read(folder/'trace.json')
        assert s['status']=='completed' and s['gpu']==seed-42
        assert len(trace)==s['evaluations']<=50
        assert len({tuple(r['widths']) for r in trace})==len(trace)
        assert digest(folder/'selection.json')==s['selection_sha256']
        assert datetime.fromisoformat(selection['selected_utc'])<=datetime.fromisoformat(s['final_evaluation_started_utc'])
        assert selection['kl_B']-min(r['kl_B'] for r in trace)<=1e-6
        assert all(r['params_unchanged'] and r['calibration_samples']==4096 for r in trace)
        for p in (folder/'summary.json',folder/'selection.json',folder/'trace.json'):
            provenance[str(p.relative_to(run))]=digest(p)
        for name,f in s['finals'].items():
            assert .999*selection['target']<=f['params']<=selection['target']
            if f['reused']:
                assert digest(ROOT/f['source'])==f['source_sha256']
                ref=bn.read(ROOT/f['source'])
                assert f['post_bn']==ref['metrics'] and f['kl_C']==ref['kl_C']
            else:
                assert f['bn']['params_unchanged'] and f['bn']['samples']==4096
            b_score=f.get('kl_B')
            if name in ('UNIFORM','TILT'):
                b_score=next(r['kl_B'] for r in trace if r['origin']==name)
            rows.append(dict(seed=seed,method=name,post=100*f['post_bn']['top1'],nll=f['post_bn']['nll'],
                             kl_B=b_score,kl_C=f['kl_C'],params=f['params'],reused=f['reused']))
            profiles.append(dict(seed=seed,method=name,widths=f['widths']))
        f=s['finals'];new=f['NEW_SEARCH']
        pairs.append(dict(seed=seed,new_tilt_pp=100*(new['post_bn']['top1']-f['TILT']['post_bn']['top1']),
                          new_uniform_pp=100*(new['post_bn']['top1']-f['UNIFORM']['post_bn']['top1']),
                          new_old_pp=100*(new['post_bn']['top1']-f['OLD_SEARCH']['post_bn']['top1']),
                          kl_C_gain_vs_tilt=f['TILT']['kl_C']-new['kl_C'],kl_C_gain_vs_old=f['OLD_SEARCH']['kl_C']-new['kl_C'],
                          same_as_old=new['widths']==f['OLD_SEARCH']['widths'],evaluations=s['evaluations']))
    differences={k:stats([r[k] for r in pairs]) for k in ('new_tilt_pp','new_uniform_pp','new_old_pp','kl_C_gain_vs_tilt','kl_C_gain_vs_old')}
    averages={method:{k:stats([r[k] for r in rows if r['method']==method]) for k in ('post','nll','kl_C')}
              for method in ('UNIFORM','TILT','OLD_SEARCH','NEW_SEARCH')}
    gates=dict(primary=differences['new_tilt_pp']['mean']>=1-1e-10 and differences['new_tilt_pp']['positive']>=3,
               beats_uniform=differences['new_uniform_pp']['mean']>0,holdout=differences['kl_C_gain_vs_tilt']['positive']>=3)
    result=dict(averages=averages,differences=differences,gates=gates,success=all(gates.values()),profiles=profiles,
                changed_profiles=sum(not r['same_as_old'] for r in pairs),total_evaluations=sum(r['evaluations'] for r in pairs),
                wall_seconds=manifest['wall_seconds'],source_hashes=provenance,protocol_sha256=manifest['protocol_sha256'])
    assert result['total_evaluations']==manifest['total_evaluations']
    out=run/('analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False);write(out/'analysis.json',result)
    for name,rs in [('metrics',rows),('pairs',pairs)]:
        with (out/f'{name}.csv').open('w') as file:
            w=csv.DictWriter(file,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    aggregate=[dict(method=method,top1=values['post']['mean'],sd=values['post']['sd'],nll=values['nll']['mean'],kl_C=values['kl_C']['mean'])
               for method,values in averages.items()]
    text='\n\n'.join(['# E026 — tabele',table(rows),'## Pary',table(pairs),'## Średnie',table(aggregate),
                       '## Kryteria',str(gates),'## Różnice',str(differences),
                       f"Zmienione profile: {result['changed_profiles']}/4; oceny: {result['total_evaluations']}; czas: {result['wall_seconds']:.2f}s."])
    (out/'TABLES.md').write_text(text+'\n');print(text);print('ANALYSIS='+str(out))

if __name__=='__main__':
    main()
