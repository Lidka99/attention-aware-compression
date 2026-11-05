"""E033: niezależne przeliczenie kontrastów, bramek i integralności; OPIS.md."""
import hashlib,json,sys,statistics as st
from pathlib import Path
run=Path(sys.argv[1]).resolve()
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(run/'manifest.json');assert m['status']=='completed' and m['exit_codes']==[0]*4
analysis=read(run/'analysis_20261105T011101Z/analysis.json')
assert analysis['rows']==432 and analysis['controls']==36
summaries={};rows=0;quality=[];quality_cells=[]
for ds in ('cifar10','cifar100','tiny'):
    for tag in ('b30','b70'):
        for seed in range(42,46):
            s=read(run/f'seed_{seed}'/ds/tag/'summary.json');summaries[(ds,tag,seed)]=s
            assert s['status']=='completed' and s['seed']==seed and s['dataset']==ds and s['tag']==tag
            assert set(s['metrics'])=={'UP_U','UP_T','UP_EP','UP_LC','LEGACY_EP','LEGACY_LC'}
            for arm in s['metrics']:
                assert len(s['metrics'][arm])==3
                rows+=3
                assert s['params'][arm]==s['plan']['params'][arm.rsplit('_',1)[1]] or arm.startswith('LEGACY')
            for profile in ('EP','LC'):
                for mode in ('CMA','UP_REPAIR'):
                    d=abs(100*(s['metrics']['LEGACY_'+profile][mode]['top1']-s['metrics']['UP_'+profile][mode]['top1']))
                    quality.append(d<=1.0);quality_cells.append((ds,tag,profile,mode,seed,d))
assert rows==432
assert all(quality)
quality_agg=[]
for ds in ('cifar10','cifar100','tiny'):
    for tag in ('b30','b70'):
        for profile in ('EP','LC'):
            for mode in ('CMA','UP_REPAIR'):
                vals=[d for x,y,p_,m_,s_,d in quality_cells if (x,y,p_,m_)==(ds,tag,profile,mode)]
                quality_agg.append(st.mean(vals)<=.5 and max(vals)<=1)
assert all(quality_agg) and len(quality_agg)==24
for ds in ('cifar10','cifar100','tiny'):
    for tag in ('b30','b70'):
        for mode in ('PRE','CMA','UP_REPAIR'):
            for a,b,label in (('UP_EP','UP_T','EP-T'),('UP_LC','UP_T','LC-T'),('UP_EP','UP_U','EP-U'),('UP_LC','UP_U','LC-U')):
                d=[100*(summaries[(ds,tag,s)]['metrics'][a][mode]['top1']-summaries[(ds,tag,s)]['metrics'][b][mode]['top1']) for s in range(42,46)]
                old=next(x for x in analysis['mechanism'] if x['dataset']==ds and x['tag']==tag and x['mode']==mode and x['contrast']==label)
                assert abs(st.mean(d)-old['mean'])<1e-12 and sum(x>0 for x in d)==old['positive']
primary=next(x for x in analysis['mechanism'] if x['dataset']=='cifar100' and x['tag']=='b70' and x['mode']=='CMA' and x['contrast']=='EP-T')
assert analysis['mechanism_gate']==(primary['mean']>0 and primary['positive']>=3)
size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file());assert size<536870912 and m['wall_seconds']<=5400
for p,h in analysis['source_hashes'].items(): assert sha(run/p)==h
result=dict(status='passed',rows=rows,controls=36,quality_gates=len(quality_agg),seed_checks=len(quality),quality_all=all(quality_agg),
            primary=primary,mechanism_gate=analysis['mechanism_gate'],secondary_gate=analysis['secondary_gate'],
            artifacts_bytes=size,wall_seconds=m['wall_seconds'],verifier_sha256=sha(Path(__file__)),
            analysis_sha256=sha(run/'analysis_20261105T011101Z/analysis.json'))
(run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
