"""E027: analiza selektorów i pułapu puli; OPIS.md.
Wejście zakończony run; wyjście analysis JSON/CSV/Markdown, bez GPU.
"""
import argparse
import csv
from datetime import datetime,timezone
import numpy as np
from scipy.stats import spearmanr
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
    rows=[];pairs=[];selectors=[];provenance={};alignment=[]
    for seed in range(42,46):
        folder=run/f'seed_{seed}'
        s=bn.read(folder/'summary.json');selection=bn.read(folder/'selection.json')
        trace=bn.read(folder/'trace.json');dev=bn.read(folder/'dev_diagnostics.json')
        assert s['status']=='completed' and s['gpu']==seed-42
        assert len(trace)==s['evaluations']<=50 and len(dev)==len(trace)
        assert len({tuple(r['widths']) for r in trace})==len(trace)
        assert digest(folder/'selection.json')==s['selection_sha256']
        assert datetime.fromisoformat(selection['selected_utc'])<=datetime.fromisoformat(s['final_evaluation_started_utc'])
        for r,d in zip(trace,dev):
            assert r['index']==d['index'] and r['widths']==d['widths']
            for key in OBJECTIVES:
                assert abs(r[key]-d[key])<1e-12
        for p in (folder/'summary.json',folder/'selection.json',folder/'trace.json',folder/'dev_diagnostics.json'):
            provenance[str(p.relative_to(run))]=digest(p)
        best_by={key:min(trace,key=lambda r:(r[key],r['index'])) for key in OBJECTIVES}
        assert selection['index']==best_by['disagreement']['index']
        assert selection['widths']==best_by['disagreement']['widths']
        assert all(r['params_unchanged'] and r['calibration_samples']==4096 for r in trace)
        pool_max=max(d['dev']['top1'] for d in dev)
        for key in OBJECTIVES:
            chosen_dev=next(d['dev'] for d in dev if d['index']==best_by[key]['index'])
            selectors.append(dict(seed=seed,objective=key,selected_index=best_by[key]['index'],
                                  selected_dev_top1=100*chosen_dev['top1'],regret_pp=100*(pool_max-chosen_dev['top1'])))
        for key in OBJECTIVES:
            rho=float(spearmanr([r[key] for r in trace],[d['dev']['top1'] for d in dev]).statistic)
            alignment.append(dict(seed=seed,objective=key,rho_vs_dev_top1=rho,alignment=-rho))
        f=s['finals']
        for name,m in f.items():
            assert .999*selection['target']<=m['params']<=selection['target']
            if m['reused']:
                assert digest(ROOT/m['source'])==m['source_sha256']
            else:
                assert m['bn']['params_unchanged'] and m['bn']['samples']==4096
            rows.append(dict(seed=seed,method=name,post=100*m['post_bn']['top1'],nll=m['post_bn']['nll'],
                             kl_C=m['kl_C'],params=m['params'],reused=m['reused']))
        new=f['NEW_SEARCH']
        pairs.append(dict(seed=seed,new_tilt_pp=100*(new['post_bn']['top1']-f['TILT']['post_bn']['top1']),
                          new_uniform_pp=100*(new['post_bn']['top1']-f['UNIFORM']['post_bn']['top1']),
                          new_old_pp=100*(new['post_bn']['top1']-f['OLD_SEARCH']['post_bn']['top1']),
                          new_e026_pp=100*(new['post_bn']['top1']-f['E026_SEARCH']['post_bn']['top1']),
                          oracle_tilt_pp=100*(pool_max-f['TILT']['post_bn']['top1']),
                          oracle_uniform_pp=100*(pool_max-f['UNIFORM']['post_bn']['top1']),
                          pool_max=100*pool_max,evaluations=s['evaluations']))
    average={method:{k:stats([r[k] for r in rows if r['method']==method]) for k in ('post','nll','kl_C')}
             for method in ('UNIFORM','TILT','OLD_SEARCH','E026_SEARCH','NEW_SEARCH')}
    differences={k:stats([r[k] for r in pairs]) for k in ('new_tilt_pp','new_uniform_pp','new_old_pp','new_e026_pp',
                                                          'oracle_tilt_pp','oracle_uniform_pp')}
    align_mean={key:float(np.mean([r['alignment'] for r in alignment if r['objective']==key])) for key in OBJECTIVES}
    gates=dict(primary=differences['new_tilt_pp']['mean']>=1-1e-10 and differences['new_tilt_pp']['positive']>=3,
               objective_gate=align_mean['disagreement']>align_mean['kl'] and align_mean['disagreement']>align_mean['teacher_ce'],
               space_gate=differences['oracle_tilt_pp']['mean']>=1-1e-10,
               beats_uniform=differences['new_uniform_pp']['mean']>0)
    result=dict(averages=average,differences=differences,alignment=alignment,alignment_mean=align_mean,
                selectors=selectors,gates=gates,total_evaluations=sum(r['evaluations'] for r in pairs),
                wall_seconds=manifest['wall_seconds'],source_hashes=provenance,protocol_sha256=manifest['protocol_sha256'])
    assert result['total_evaluations']==manifest['total_evaluations']
    out=run/('analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False);write(out/'analysis.json',result)
    for name,rs in [('metrics',rows),('pairs',pairs),('selectors',selectors),('alignment',alignment)]:
        with (out/f'{name}.csv').open('w') as file:
            w=csv.DictWriter(file,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    aggregate=[dict(method=method,top1=values['post']['mean'],sd=values['post']['sd'],nll=values['nll']['mean'],kl_C=values['kl_C']['mean'])
               for method,values in average.items()]
    text='\n\n'.join(['# E027 — tabele',table(rows),'## Pary',table(pairs),'## Selektory',table(selectors),
                       '## Zgodność miar z dev',table(alignment),'## Średnie',table(aggregate),
                       '## Kryteria',str(gates),'## Różnice',str(differences),
                       f"Oceny: {result['total_evaluations']}; czas: {result['wall_seconds']:.2f}s."])
    (out/'TABLES.md').write_text(text+'\n');print(text);print('ANALYSIS='+str(out))

if __name__=='__main__':
    main()
