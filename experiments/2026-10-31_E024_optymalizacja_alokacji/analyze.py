"""E024: analiza czterech seedów bez GPU; OPIS.md.
Wejście: zakończony run; wyjście: analysis.json/CSV/Markdown z kryteriami.
"""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from data import HERE,digest,write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    run=args.run_dir.resolve()
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
    assert digest(run/'indices.json')==manifest['indices_sha256']
    assert digest(HERE/'OPIS.md')==manifest['protocol_sha256']
    assert manifest['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    rows=[];pairs=[];provenance={};profiles=[]
    for seed in range(42,46):
        folder=run/f'seed_{seed}'
        s=json.loads((folder/'summary.json').read_text())
        selection=json.loads((folder/'selection.json').read_text())
        trace=json.loads((folder/'trace.json').read_text())
        assert s['status']=='completed' and s['gpu']==seed-42
        assert digest(folder/'selection.json')==s['selection_sha256']
        assert datetime.fromisoformat(selection['selected_utc'])<=datetime.fromisoformat(s['final_evaluation_started_utc'])
        assert len(trace)==s['evaluations']<=50
        assert len({tuple(r['widths']) for r in trace})==len(trace)
        assert selection['kl_B']-min(r['kl_B'] for r in trace)<=1e-6
        for p in (folder/'summary.json',folder/'selection.json',folder/'trace.json'):
            provenance[str(p.relative_to(run))]=digest(p)
        for method,m in s['finals'].items():
            assert .999*selection['target']<=m['params']<=selection['target']
            assert m['params_unchanged']
            rows.append(dict(seed=seed,method=method,baseline=100*s['controls']['teacher']['top1'],
                             baseline_bn=100*s['controls']['teacher_bn']['top1'],pre=100*m['pre_bn']['top1'],
                             post=100*m['post_bn']['top1'],nll=m['post_bn']['nll'],kl_B=m['kl_B'],kl_C=m['kl_C'],
                             params=m['params'],evaluations=s['evaluations'],seconds=s['runtime_seconds']))
            profiles.append(dict(seed=seed,method=method,widths=m['widths']))
        f=s['finals']
        pairs.append(dict(seed=seed,search_tilt_pp=100*(f['SEARCH']['post_bn']['top1']-f['TILT']['post_bn']['top1']),
                          search_uniform_pp=100*(f['SEARCH']['post_bn']['top1']-f['UNIFORM']['post_bn']['top1']),
                          kl_C_improvement=f['TILT']['kl_C']-f['SEARCH']['kl_C'],evaluations=s['evaluations']))
    stats={}
    for method in ('UNIFORM','TILT','SEARCH'):
        rs=[r for r in rows if r['method']==method]
        stats[method]={key:dict(mean=float(np.mean([r[key] for r in rs])),sd=float(np.std([r[key] for r in rs],ddof=1)))
                       for key in ('post','pre','nll','kl_B','kl_C')}
    diffs={k:dict(mean=float(np.mean([r[k] for r in pairs])),sd=float(np.std([r[k] for r in pairs],ddof=1)),
                 positive=sum(r[k]>0 for r in pairs)) for k in ('search_tilt_pp','search_uniform_pp','kl_C_improvement')}
    gates=dict(primary=diffs['search_tilt_pp']['mean']>=1 and diffs['search_tilt_pp']['positive']>=3,
               beats_uniform=diffs['search_uniform_pp']['mean']>0,holdout=diffs['kl_C_improvement']['positive']>=3)
    result=dict(stats=stats,differences=diffs,gates=gates,success=all(gates.values()),profiles=profiles,
                source_hashes=provenance,wall_seconds=manifest['wall_seconds'],total_evaluations=sum(p['evaluations'] for p in pairs))
    out=run/'analysis'
    out.mkdir(exist_ok=False)
    write(out/'analysis.json',result)
    for name,rs in [('metrics',rows),('pairs',pairs)]:
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rs[0]));writer.writeheader();writer.writerows(rs)
    def table(rs):
        keys=list(rs[0])
        def fmt(v):
            return f'{v:.4f}' if isinstance(v,float) else str(v)
        return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
                         ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rs])
    text='# E024 — tabele\n\n'+table(rows)+'\n\n'+table(pairs)+'\n\n'+json.dumps(result,indent=2)
    (out/'TABLES.md').write_text(text+'\n')
    print(text)


if __name__=='__main__':
    main()
