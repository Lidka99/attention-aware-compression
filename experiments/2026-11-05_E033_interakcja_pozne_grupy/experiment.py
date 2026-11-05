"""E033: interakcja wczesne/środkowe vs późne grupy przy profilach E031; OPIS.md.
Wejście: checkpointy/dane E031; wyjście unikalny run JSON/CSV/MD, bez treningu.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E31=ROOT/'experiments/2026-11-02_E031_oficjalny_30_70'
spec=importlib.util.spec_from_file_location('_e033_e31',E31/'common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
REFERENCE=ROOT/'runs/E031/20261104T212951Z_6c3d7356'
PROFILES=('U','T','EP','LC')
ARMS=('UP_U','UP_T','UP_EP','UP_LC','LEGACY_EP','LEGACY_LC')
MODES=('PRE','CMA','UP_REPAIR')
write,read,digest=c.write,c.read,c.digest

def utc():
    return datetime.now(timezone.utc).isoformat()

def inputs():
    result=c.provenance()
    for p in [E31/'common.py',E31/'check.json',REFERENCE/'manifest.json',REFERENCE/'indices.json',
              ROOT/'experiments/2026-11-04_E032_replikacja_wrazliwosci/WYNIKI.md']:
        result[str(p.relative_to(ROOT))]=digest(p)
    return result

def source_hashes():
    return {p.name:digest(p) for p in HERE.glob('*.py')}

def hybrid(geo,target,uniform,fixed,tilt):
    sizes=geo.sizes
    def widths(base):
        w=[None]*12
        for i in fixed:w[i]=int(uniform[i])
        for i in tilt:
            p=2*i/11-1
            r=min(max(base+.30*p,.1),.9)
            w[i]=int(sizes[i]*(1-r))
        return w
    lo,hi=-.30,1.30
    assert geo.count(widths(lo))>=target>=geo.count(widths(hi))
    for _ in range(60):
        mid=(lo+hi)/2
        if geo.count(widths(mid))>target: lo=mid
        else: hi=mid
    w=widths(hi)
    maxima=[math.floor(.9*n) for n in sizes]
    while True:
        best=None
        for i in tilt:
            if w[i]>=maxima[i]: continue
            v=w.copy();v[i]+=1
            p=geo.count(v)
            if p<=target and (best is None or (p,-i)>(best[0],best[1])):
                best=(p,i,v)
        if best is None: break
        w=best[2]
    return w

def plans():
    base=read(E31/'check.json')['plans'];result={}
    for ds in c.DATASETS:
        result[ds]={}
        for budget,tag in ((.3,'b30'),(.7,'b70')):
            tv,_=c.load_source(ds,42);geo=c.bn.Geometry(tv)
            u=base[ds][tag]['uniform'];target=base[ds][tag]['target']
            ep=hybrid(geo,target,u,{9,10,11},{0,1,2,3,4,5,6,7,8})
            lc=hybrid(geo,target,u,set(range(9)),{9,10,11})
            assert ep[9:]==u[9:] and lc[:9]==u[:9]
            for w in (ep,lc):
                assert .999*target<=geo.count(w)<=target
            result[ds][tag]=dict(U=u,T=base[ds][tag]['tilt'],EP=ep,LC=lc,
                                 target=target,params={k:geo.count(v) for k,v in
                                 dict(U=u,T=base[ds][tag]['tilt'],EP=ep,LC=lc).items()},
                                 ratio=base[ds][tag]['ratio'],budget=budget)
    return result

def check():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(42)
    frozen=plans()
    for ds in c.DATASETS:
        tv,up=c.load_source(ds,42);geo=c.bn.Geometry(tv)
        x=torch.randn(2,3,32,32) if ds!='tiny' else torch.randn(2,3,64,64)
        for tag in ('b30','b70'):
            p=frozen[ds][tag]
            for name in PROFILES:
                w=p[name]
                assert sum(1 for x in w if x is None)==0 and len(w)==12
                assert all(1<=x<=geo.sizes[i] for i,x in enumerate(w))
                model,diag=c.e29.official(up if up is not None else tv,p['ratio'],w)
                model.fc.bias.data.copy_(tv.fc.bias)
                assert sum(q.numel() for q in model.parameters())==p['params'][name]
                assert len(diag)==12 and all(len(set(g['labels']))==g['k'] for g in diag)
                with torch.inference_mode():
                    y=model(x)
                assert torch.isfinite(y).all()
                if name in ('EP','LC'):
                    ratios=[1-v/geo.sizes[i] for i,v in enumerate(w)]
                    old,diag=c.legacy.fold_with_ratios(tv,ratios,seed=42,deterministic=False,preserve_fc_bias=True)
                    assert sum(q.numel() for q in old.parameters())==p['params'][name]
                    assert torch.equal(old.fc.bias,tv.fc.bias)
                    with torch.inference_mode():
                        assert torch.isfinite(old(x)).all()
    write(HERE/'check.json',dict(status='passed',utc=utc(),plans=frozen,inputs=inputs(),
          source_hashes=source_hashes(),protocol_sha256=digest(HERE/'OPIS.md')))
    print('CPU PASS',json.dumps(frozen,indent=1),flush=True)

def launch():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    assert len(info.strip().splitlines())==4
    for i,line in enumerate(info.strip().splitlines()):
        r=[x.strip() for x in line.split(',')]
        assert int(r[0])==i and int(r[2])<256 and int(r[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    frozen=read(HERE/'check.json')
    assert frozen['status']=='passed' and frozen['inputs']==inputs()
    assert frozen['source_hashes']==source_hashes() and frozen['protocol_sha256']==digest(HERE/'OPIS.md')
    run=ROOT/'runs/E033'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True,exist_ok=False);(run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.is_file(): shutil.copyfile(p,run/'source'/p.name)
    shutil.copyfile(REFERENCE/'indices.json',run/'indices.json')
    write(run/'plans.json',frozen['plans'])
    m=dict(status='running',started_utc=utc(),launcher_pid=os.getpid(),assignments={str(i):42+i for i in range(4)},
           inputs=frozen['inputs'],source_hashes=source_hashes(),protocol_sha256=frozen['protocol_sha256'],
           indices_sha256=digest(run/'indices.json'),upstream_commit=c.COMMIT,gpus=info,
           python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
           official_test_used=False,reserve_used=False,training=False,batch_size=128,
           profiles=list(PROFILES),arms=list(ARMS),limits=dict(wall_seconds=5400,artifacts_bytes=536870912))
    write(run/'manifest.json',m)
    (run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('RUN='+str(run),flush=True)
    processes=[];start=time.monotonic()
    try:
        for gpu in range(4):
            log=(run/'logs'/f'gpu{gpu}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'experiment.py'),'worker','--run-dir',str(run),'--gpu',str(gpu)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            processes.append((p,log))
        m['worker_pids']=[p.pid for p,_ in processes];write(run/'manifest.json',m)
        codes=[p.wait(timeout=max(.1,5400-(time.monotonic()-start))) for p,_ in processes]
        m['exit_codes']=codes;assert codes==[0]*4
        assert all(read(run/f'seed_{s}/status.json')['status']=='completed' for s in range(42,46))
        m['status']='completed'
    except BaseException as error:
        m.update(status='failed',error=repr(error));raise
    finally:
        for p,log in processes:
            if p.poll() is None:
                p.kill();p.wait()
            log.close()
        m.update(finished_utc=utc(),wall_seconds=time.monotonic()-start)
        write(run/'manifest.json',m)
    print(m['status'],m['exit_codes'],flush=True)

def evaluate(base,a,x,y):
    logits={};metrics={};cal={}
    for mode in MODES:
        model=copy.deepcopy(base).to('cuda')
        if mode!='PRE': cal[mode]=c.e29.calibrate(model,a,mode)
        z=c.bn.predict(model,x);assert torch.isfinite(z).all()
        metrics[mode]=dict(top1=float((z.argmax(1)==y).double().mean()),
                           nll=float(torch.nn.functional.cross_entropy(z,y)))
        logits[mode]=z;del model
    return metrics,logits,cal

def tensor_diffs(a,b):
    assert set(a)==set(b)
    max_abs=0.;diff_sq=0.;ref_sq=0.
    for name in a:
        if name.endswith('num_batches_tracked'):
            continue
        assert a[name].shape==b[name].shape
        x=a[name].detach().double();y=b[name].detach().double()
        max_abs=max(max_abs,float((x-y).abs().max()))
        diff_sq+=float((x-y).square().sum());ref_sq+=float(y.square().sum())
    return dict(max_abs=max_abs,relative_l2=(diff_sq/max(ref_sq,1e-30))**.5)


def worker(run,gpu):
    seed=42+gpu
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    out=run/f'seed_{seed}';out.mkdir()
    status=dict(status='running',seed=seed,gpu=gpu,pid=os.getpid(),started_utc=utc(),completed=[])
    write(out/'status.json',status)
    try:
        for ds in c.DATASETS:
            a,x,y,views=c.data(ds,read(run/'indices.json')[ds])
            tv,up=c.load_source(ds,seed)
            sizes=c.bn.Geometry(tv).sizes
            controls,arch=c.teacher_controls(ds,seed,tv,up,a,x,y)
            ref=read(REFERENCE/f'seed_{seed}'/f'controls_{ds}.json')
            assert views==ref['views_sha256']
            assert all(abs(controls[m]['top1']-ref['controls'][m]['top1'])<1e-12 for m in MODES)
            write(out/f'controls_{ds}.json',dict(controls=controls,views_sha256=views,
                                                 architecture_max_abs=arch))
            for tag in ('b30','b70'):
                plan=read(run/'plans.json')[ds][tag]
                (out/ds/tag).mkdir(parents=True,exist_ok=True)
                metrics={};logits={};params={};clusters={};models={}
                for profile in PROFILES:
                    w=plan[profile]
                    base,diag=c.e29.official(up if up is not None else tv,plan['ratio'],w)
                    base.fc.bias.data.copy_(tv.fc.bias)
                    arm='UP_'+profile
                    assert sum(q.numel() for q in base.parameters())==plan['params'][profile]
                    metrics[arm],logits[arm],_=evaluate(base,a,x,y)
                    params[arm]=plan['params'][profile];clusters[arm]=diag;models[arm]=base
                for profile in ('EP','LC'):
                    w=plan[profile];ratios=[1-v/n for v,n in zip(w,sizes)]
                    old,diag=c.legacy.fold_with_ratios(tv,ratios,seed=42,
                                                       deterministic=False,preserve_fc_bias=True)
                    old.eval().requires_grad_(False)
                    arm='LEGACY_'+profile
                    assert sum(q.numel() for q in old.parameters())==plan['params'][profile]
                    metrics[arm],logits[arm],_=evaluate(old,a,x,y)
                    params[arm]=plan['params'][profile];clusters[arm]=diag['groups'];models[arm]=old
                comparisons={}
                for profile in ('EP','LC'):
                    left='LEGACY_'+profile;right='UP_'+profile
                    la=[g.get('cluster_labels',g.get('labels')) for g in clusters[left]]
                    ra=[g.get('labels',g.get('cluster_labels')) for g in clusters[right]]
                    compare=dict(reference=right,
                        parameters=tensor_diffs(dict(models[left].named_parameters()),
                                                dict(models[right].named_parameters())),
                        state=tensor_diffs(models[left].state_dict(),models[right].state_dict()),
                        cluster_ari=[float(adjusted_rand_score(l,r)) for l,r in zip(la,ra)],
                        labels_exact=[l==r for l,r in zip(la,ra)],logits={})
                    for mode in MODES:
                        xx=logits[left][mode];yy=logits[right][mode]
                        compare['logits'][mode]=dict(
                            max_abs=float((xx-yy).abs().max()),
                            rmse=float((xx-yy).square().mean().sqrt()),
                            agreement=float((xx.argmax(1)==yy.argmax(1)).double().mean()),
                            top1_delta_pp=100*(metrics[left][mode]['top1']-metrics[right][mode]['top1']))
                    comparisons[left]=compare
                summary=dict(status='completed',dataset=ds,tag=tag,budget=plan['budget'],
                             seed=seed,gpu=gpu,plan=plan,metrics=metrics,params=params,
                             comparisons=comparisons,views_sha256=views,
                             controls_sha256=digest(out/f'controls_{ds}.json'),
                             official_test_used=False,reserve_used=False)
                write(out/ds/tag/'summary.json',summary)
                status['completed'].append(f'{ds}/{tag}');write(out/'status.json',status)
                print(seed,ds,tag,{k:{m:round(v['top1']*100,2) for m,v in metrics[k].items()}
                                   for k in metrics},flush=True)
                del models,logits,metrics,clusters,comparisons
        status.update(status='completed',finished_utc=utc())
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        write(out/'status.json',status)

def analyze(run):
    m=read(run/'manifest.json')
    assert m['status']=='completed' and m['exit_codes']==[0]*4
    assert m['inputs']==inputs() and m['source_hashes']==source_hashes()
    assert m['protocol_sha256']==digest(HERE/'OPIS.md') and m['indices_sha256']==digest(run/'indices.json')
    rows=[];summary={};hashes={}
    for ds in c.DATASETS:
        for tag in ('b30','b70'):
            for seed in range(42,46):
                p=run/f'seed_{seed}'/ds/tag/'summary.json';s=read(p);hashes[str(p.relative_to(run))]=digest(p)
                assert s['status']=='completed' and s['seed']==seed and s['dataset']==ds and s['tag']==tag
                assert not s['official_test_used'] and not s['reserve_used']
                assert s['controls_sha256']==digest(run/f'seed_{seed}'/f'controls_{ds}.json')
                summary[(ds,tag,seed)]=s
                for arm in ARMS:
                    assert arm in s['metrics'] and arm in s['params']
                    for mode in MODES:
                        rows.append(dict(dataset=ds,tag=tag,seed=seed,arm=arm,mode=mode,
                                         top1=100*s['metrics'][arm][mode]['top1'],nll=s['metrics'][arm][mode]['nll']))
    assert len(rows)==432
    averages=[];mechanism=[];quality=[]
    for ds in c.DATASETS:
        for tag in ('b30','b70'):
            for arm in ARMS:
                for mode in MODES:
                    vals=[r['top1'] for r in rows if r['dataset']==ds and r['tag']==tag and r['arm']==arm and r['mode']==mode]
                    averages.append(dict(dataset=ds,tag=tag,arm=arm,mode=mode,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1))))
            for mode in MODES:
                def delta(a,b):
                    return [100*(summary[(ds,tag,s)]['metrics'][a][mode]['top1']-summary[(ds,tag,s)]['metrics'][b][mode]['top1']) for s in range(42,46)]
                for a,b,label in [('UP_EP','UP_T','EP-T'),('UP_LC','UP_T','LC-T'),
                                  ('UP_EP','UP_U','EP-U'),('UP_LC','UP_U','LC-U')]:
                    v=delta(a,b);mechanism.append(dict(dataset=ds,tag=tag,mode=mode,contrast=label,
                        mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),positive=sum(x>0 for x in v),per_seed=v))
            for profile in ('EP','LC'):
                for mode in ('CMA','UP_REPAIR'):
                    a='LEGACY_'+profile;b='UP_'+profile
                    d=[abs(100*(summary[(ds,tag,s)]['metrics'][a][mode]['top1']-summary[(ds,tag,s)]['metrics'][b][mode]['top1'])) for s in range(42,46)]
                    quality.append(dict(dataset=ds,tag=tag,profile=profile,mode=mode,
                        mean_abs_pp=float(np.mean(d)),max_abs_pp=max(d),gate=float(np.mean(d))<=.5 and max(d)<=1))
    primary=[x for x in mechanism if x['dataset']=='cifar100' and x['tag']=='b70' and x['mode']=='CMA']
    ep=next(x for x in primary if x['contrast']=='EP-T');lc=next(x for x in primary if x['contrast']=='LC-T')
    secondary=next(x for x in mechanism if x['dataset']=='cifar100' and x['tag']=='b70' and x['mode']=='UP_REPAIR' and x['contrast']=='EP-T')
    mechanism_gate=ep['mean']>0 and ep['positive']>=3
    secondary_gate=secondary['mean']>0 and secondary['positive']>=3
    quality_gate=all(x['gate'] for x in quality)
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file());assert size<536870912
    out=run/f'analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir()
    result=dict(mechanism_gate=mechanism_gate,secondary_gate=secondary_gate,
                lc_exploratory=dict(mean=lc['mean'],positive=lc['positive'],per_seed=lc['per_seed']),
                quality_gate=quality_gate,mechanism=mechanism,quality=quality,
                averages=averages,plans=read(run/'plans.json'),wall_seconds=m['wall_seconds'],
                artifacts_bytes=size,source_hashes=hashes,rows=len(rows),controls=36)
    write(out/'analysis.json',result)
    for name,data in [('metrics',rows),('mechanism',mechanism),('quality',quality)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(result,indent=1));print('ANALYSIS='+str(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['check','run','worker','analyze'])
    ap.add_argument('--run-dir',type=Path);ap.add_argument('--gpu',type=int)
    args=ap.parse_args()
    if args.action=='check': check()
    elif args.action=='run': launch()
    elif args.action=='worker': worker(args.run_dir,args.gpu)
    else: analyze(args.run_dir)
