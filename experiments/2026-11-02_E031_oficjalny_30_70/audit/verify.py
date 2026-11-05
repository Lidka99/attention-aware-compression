"""E031: niezależny audyt JSON, reuse i kryteriów; wejście run, wyjście verification.json.
Protokół: ../OPIS.md. Bez GPU i ocen modeli.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

run=Path(sys.argv[1]).resolve()
def read(p):
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(run/'manifest.json')
assert m['status']=='completed' and m['exit_codes']==[0]*4
old=Path(m['reuse_from']).resolve()
for p,h in m['reused_hashes'].items():
    assert sha(old/p)==sha(run/p)==h
assert sha(run/'indices.json')==m['indices_sha256']
analysis=read(run/'analysis_20261104T213425Z/analysis.json')
rows=[]
for ds in ('cifar10','cifar100','tiny'):
    for tag in ('b30','b70'):
        items=[read(run/f'seed_{s}'/ds/tag/'summary.json') for s in range(42,46)]
        for item in items:
            assert item['status']=='completed'
            assert item['plan']==read(run/'plans.json')[ds][tag]
            assert not item['official_test_used'] and not item['reserve_used']
            assert .999*item['plan']['target']<=item['plan']['tilt_params']<=item['plan']['target']
            for name,n in item['params'].items():
                assert n==item['plan']['tilt_params' if name.endswith('_T') else 'target']
        for mode in ('CMA','UP_REPAIR'):
            u=[100*x['metrics']['UP_FIXED_U'][mode]['top1'] for x in items]
            t=[100*x['metrics']['UP_FIXED_T'][mode]['top1'] for x in items]
            delta=[b-a for a,b in zip(u,t)]
            found=next(x for x in analysis['allocation'] if x['dataset']==ds and x['tag']==tag and x['mode']==mode and x['operator']=='UP_FIXED')
            assert abs(statistics.mean(delta)-found['mean'])<1e-10
            assert sum(x>0 for x in delta)==found['positive']
            rows.append(dict(dataset=ds,tag=tag,mode=mode,uniform=statistics.mean(u),tilt=statistics.mean(t),delta=statistics.mean(delta),sd=statistics.stdev(delta),per_seed=delta,positive=sum(x>0 for x in delta)))
        for variant in ('LEGACY_U','LEGACY_T','LOCAL_U','LOCAL_T'):
            reference='UP_FIXED_'+variant[-1]
            diffs=[abs(100*(x['metrics'][variant]['CMA']['top1']-x['metrics'][reference]['CMA']['top1'])) for x in items]
            assert statistics.mean(diffs)<=.5 and max(diffs)<=1
reused_cells=sum(p.endswith('/summary.json') for p in m['reused_hashes'])
size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
assert size<536870912
result=dict(status='passed',rows=rows,reused_cells=reused_cells,new_cells=24-reused_cells,
            reused_evaluations=21*reused_cells,new_evaluations=21*(24-reused_cells),
            artifacts_bytes=size,wall_seconds=m['wall_seconds'],
            reused_hashes_verified=len(m['reused_hashes']),
            allocation_passed=sum(x['delta']>0 and x['positive']>=3 for x in rows))
(run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
