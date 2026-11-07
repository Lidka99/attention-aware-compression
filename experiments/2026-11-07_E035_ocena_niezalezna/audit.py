"""E035: oddzielny audyt artefaktów bez GPU; wejście run, wyjście verification.json.
Sprawdza hashe, siatkę, dev gate, parametry/BN i analizę; protokół OPIS.md.
"""
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main(run):
    m=read(run/'manifest.json');f=read(run/'frozen.json');a=read(run/'analysis/analysis.json')
    assert m['status']=='completed' and m['wall_seconds']<=5400
    assert m['assignments']=={'0':42,'1':43,'2':44,'3':45}
    assert all(m['phases'][p]['exit_codes']==[0]*4 for p in ('dev','test'))
    assert len(set(sum([m['phases'][p]['pids'] for p in ('dev','test')],[])))==8
    assert digest(run/'frozen.json')==m['frozen_sha256']
    assert digest(run/'source/OPIS.md')==m['protocol_sha256']
    committed=subprocess.check_output(['git','show','f42c0b8:experiments/2026-11-07_E035_ocena_niezalezna/OPIS.md'],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==m['protocol_sha256']
    for mapping in (f['provenance'],m['code_hashes'],read(run/'test_data.json')['files']):
        for p,h in mapping.items():assert digest(ROOT/p)==h,p
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'runs/E029/upstream_6946915',text=True).strip()==''
    gate=read(run/'dev_gate.json');assert gate['passed'] and not gate['errors']
    assert (gate['evaluations'],gate['controls'])==(360,60)
    assert gate['utc']<read(run/'test_data.json')['utc']<=m['phases']['test']['started_utc']
    hashes={};counts={};max_top=0.;max_nll=0.
    for phase in ('dev','test'):
        n=nc=0
        for pair,plans in f['plans'].items():
            for seed in range(42,46):
                out=run/phase/f'seed_{seed}'/pair;c=read(out/'controls.json')
                hashes[str((out/'controls.json').relative_to(run))]=digest(out/'controls.json')
                for mode,metric in c['metrics'].items():
                    nc+=1;assert 0<=metric['top1']<=1 and math.isfinite(metric['nll'])
                    ev=c['evidence'][mode];assert ev['weights_unchanged'] and ev['no_grad']
                    assert ev['samples']==(10000 if phase=='test' or pair.endswith('tiny') else 5000)
                    if phase=='test':
                        dev=read(run/f'dev/seed_{seed}/{pair}/controls.json')
                        assert ev['state_sha256']==dev['evidence'][mode]['state_sha256']
                for tag,plan in plans.items():
                    p=out/tag/'summary.json';s=read(p);hashes[str(p.relative_to(run))]=digest(p)
                    assert s['status']=='completed' and s['seed']==seed and s['gpu']==seed-42
                    assert s['pair']==pair and s['tag']==tag and s['phase']==phase and s['plan']==plan
                    assert s['bias_preserved'] and s['controls_sha256']==digest(out/'controls.json')
                    assert s['views_sha256']==c['views_sha256']
                    ref=f['references'][pair][tag][str(seed)]
                    if ref['views_sha256']:assert s['views_sha256']==ref['views_sha256']
                    for profile in ('U','T'):
                        assert [d['k'] for d in s['clusters'][profile]]==plan[profile]
                        for d in s['clusters'][profile]:assert sorted(set(d['labels']))==list(range(d['k'])) and len(d['labels'])==d['n']
                        for mode in ('PRE','CMA','UP_REPAIR'):
                            n+=1;val=s['metrics'][profile][mode];ev=s['evidence'][profile][mode]
                            assert 0<=val['top1']<=1 and math.isfinite(val['nll'])
                            assert ev['weights_unchanged'] and ev['no_grad']
                            if mode!='PRE':assert ev['calibration']['params_unchanged']
                            if phase=='dev':
                                for key,tol in [('top1',1e-10),('nll',1e-5)]:
                                    delta=abs(val[key]-ref['metrics'][profile][mode][key]);assert delta<=tol
                                    if key=='top1':max_top=max(max_top,delta)
                                    else:max_nll=max(max_nll,delta)
                            else:
                                dev=read(run/f'dev/seed_{seed}/{pair}/{tag}/summary.json')
                                assert ev['state_sha256']==dev['evidence'][profile][mode]['state_sha256']
                                assert s['clusters'][profile]==dev['clusters'][profile]
                            rows=[r for r in a['rows'] if (r['phase'],r['pair'],r['seed'],r['tag'],r['profile'],r['mode'])==(phase,pair,seed,tag,profile,mode)]
                            assert len(rows)==1 and rows[0]['top1']==val['top1'] and rows[0]['nll']==val['nll']
        assert (n,nc)==(360,60);counts[phase]=dict(evaluations=n,controls=nc)
    assert len(a['rows'])==720 and len(a['controls'])==120 and len(a['contrasts'])==90
    for r in a['contrasts']:
        vals=[]
        for seed in range(42,46):
            s=read(run/f"{r['phase']}/seed_{seed}/{r['pair']}/{r['tag']}/summary.json")
            vals.append(100*(s['metrics']['T'][r['mode']]['top1']-s['metrics']['U'][r['mode']]['top1']))
        assert vals==r['per_seed'] and abs(statistics.mean(vals)-r['mean'])<1e-10
        assert abs(statistics.stdev(vals)-r['sd'])<1e-10
        assert r['direction']==(statistics.mean(vals)>0 and sum(v>0 for v in vals)>=3)
    result=dict(status='PASS',counts=counts,dev_max_top1=max_top,dev_max_nll=max_nll,
                identical_dev_test_model_states=True,artifacts_sha256=hashes,
                analysis_sha256=digest(run/'analysis/analysis.json'),protocol_commit='f42c0b8')
    (run/'analysis/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT PASS',counts,'identical model states dev/test')
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
