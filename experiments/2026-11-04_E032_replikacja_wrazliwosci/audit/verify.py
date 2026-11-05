"""E032: niezależne przeliczenie rang, H1/H2 i tabel z surowych JSON.
Wejście run; wyjście verification.json i TABLES.md; protokół ../OPIS.md.
"""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

run=Path(sys.argv[1]).resolve()
def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ranks(values):
    return [1+sum(y<x for y in values)+(sum(y==x for y in values)-1)/2 for x in values]
def rho(a,b): return st.correlation(ranks(a),ranks(b))

manifest=read(run/'manifest.json')
assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
analysis=read(run/'analysis_20261105T000640Z/analysis.json')
raw={};group_rows=[];checks=[];aggregates=[]
for ds in ('cifar100','cifar10'):
    for seed in range(42,46):
        folder=run/f'seed_{seed}'/ds
        control=read(folder/'controls.json')
        assert control['baseline_agrees'] and len(control['controls'])==3
        for mode in ('PRE','CMA','UP_REPAIR'):
            damages=[]
            for i in range(12):
                item=read(folder/f'group_{i:02d}.json')
                assert item['dataset']==ds and item['seed']==seed and item['gpu']==seed-42
                assert item['views_sha256']==control['views_sha256']
                if mode!='PRE':
                    calibration=item['calibrations'][mode]
                    assert calibration['params_unchanged'] and calibration['samples']==4096 and calibration['batches']==32
                damages.append(100*(control['controls'][mode]['top1']-item['metrics'][mode]['top1']))
            raw[ds,seed,mode]=damages
            r=rho(list(range(12)),damages)
            gap=st.mean(damages[9:])-st.mean(damages[:3])
            old=next(x for x in analysis['profiles'] if x['dataset']==ds and x['seed']==seed and x['mode']==mode)
            assert abs(old['rho']-r)<1e-12 and abs(old['late_minus_early']-gap)<1e-12
            checks.append(dict(dataset=ds,seed=seed,mode=mode,rho=r,gap=gap,passed=r>=.5 and gap>0))
    for mode in ('PRE','CMA','UP_REPAIR'):
        early=st.mean(x for seed in range(42,46) for x in raw[ds,seed,mode][:3])
        late=st.mean(x for seed in range(42,46) for x in raw[ds,seed,mode][9:])
        mean=st.mean(x for seed in range(42,46) for x in raw[ds,seed,mode])
        aggregates.append(dict(dataset=ds,mode=mode,early=early,late=late,mean=mean,gap=late-early))
    for i in range(12):
        name=read(run/'seed_42'/ds/f'group_{i:02d}.json')['group']
        vals=[raw[ds,s,'CMA'][i] for s in range(42,46)]
        group_rows.append(dict(dataset=ds,index=i,group=name,mean=st.mean(vals),sd=st.stdev(vals)))
h1=sum(x['passed'] for x in checks if x['dataset']=='cifar100' and x['mode']=='CMA')
contrasts=[next(x['rho'] for x in checks if x['dataset']=='cifar100' and x['seed']==s and x['mode']=='CMA')-
           next(x['rho'] for x in checks if x['dataset']=='cifar10' and x['seed']==s and x['mode']=='CMA') for s in range(42,46)]
assert h1==analysis['H1_count']==4
assert (st.mean(contrasts)>=.3 and sum(x>0 for x in contrasts)>=3)==analysis['H2']
assert len(list(run.glob('seed_*/*/group_*.json')))==96
assert len(list(run.glob('seed_*/*/controls.json')))==8
for path,h in analysis['source_hashes'].items(): assert digest(run/path)==h
result=dict(status='passed',H1_count=h1,H2_mean_contrast=st.mean(contrasts),aggregates=aggregates,groups=group_rows,
            verifier_sha256=digest(Path(__file__)),analysis_sha256=digest(run/'analysis_20261105T000640Z/analysis.json'))
(run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# E032 — szkoda pojedynczych grup po CMA [pp]','',
       '| Zbiór | Grupa | Średnia | SD po4 seedach |','|---|---|---:|---:|']
for row in group_rows: lines.append(f"| {row['dataset']} | {row['group']} | {row['mean']:.3f} | {row['sd']:.3f} |")
(run/'analysis_20261105T000640Z/TABLES.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
