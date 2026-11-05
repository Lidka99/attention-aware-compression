"""E032: kontrola, launcher4GPU, pomiary wrażliwości i analiza; patrz OPIS.md.
Wejście: istniejące checkpointy/dane E031; wyjście unikalne runy JSON/CSV/MD.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E31=ROOT/'experiments/2026-11-02_E031_oficjalny_30_70'
spec=importlib.util.spec_from_file_location('_e032_e31',E31/'common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
REFERENCE=ROOT/'runs/E031/20261104T212951Z_6c3d7356'
DS=('cifar100','cifar10')
MODES=('PRE','CMA','UP_REPAIR')
write,read,digest=c.write,c.read,c.digest

def utc():
    return datetime.now(timezone.utc).isoformat()

def inputs():
    result=c.provenance()
    for p in [E31/'common.py',REFERENCE/'manifest.json',REFERENCE/'indices.json']:
        result[str(p.relative_to(ROOT))]=digest(p)
    for ds in DS:
        for seed in range(42,46):
            for p in [REFERENCE/f'seed_{seed}'/f'controls_{ds}.json',REFERENCE/f'seed_{seed}'/ds/'b70/summary.json']:
                result[str(p.relative_to(ROOT))]=digest(p)
    return result

def source_hashes():
    return {p.name:digest(p) for p in HERE.glob('*.py')}

def fold(tv,index):
    ratios=[0.]*12;ratios[index]=.75
    torch.manual_seed(42);np.random.seed(42)
    model,diag=c.legacy.fold_with_ratios(tv,ratios,seed=42,deterministic=False,preserve_fc_bias=True)
    model.eval().requires_grad_(False)
    geo=c.bn.Geometry(tv)
    expected=list(geo.sizes);expected[index]//=4
    assert diag['widths']==expected
    assert sum(p.numel() for p in model.parameters())==geo.count(expected)
    for i,g in enumerate(diag['groups']):
        assert len(set(g['cluster_labels']))==expected[i]
        if i!=index:
            assert g['cluster_labels']==list(range(geo.sizes[i]))
    assert torch.equal(model.fc.bias,tv.fc.bias)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    return model,diag

def check():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    provenance=inputs()
    for ds in DS:
        tv,_=c.load_source(ds,42)
        identity,_=c.legacy.fold_with_ratios(tv,[0.]*12,seed=42,deterministic=False,preserve_fc_bias=True)
        assert all(torch.equal(v,identity.state_dict()[k]) for k,v in tv.state_dict().items() if not k.endswith('num_batches_tracked'))
        with torch.inference_mode():
            probe=torch.randn(2,3,32,32)
            identity.eval()
            assert torch.equal(tv(probe),identity(probe))
        for i in (0,11):
            model,_=fold(tv,i)
            with torch.inference_mode():
                z=model(torch.zeros(2,3,32,32))
            assert torch.isfinite(z).all()
    write(HERE/'check.json',dict(status='passed',utc=utc(),inputs=provenance,
          source_hashes=source_hashes(),protocol_sha256=digest(HERE/'OPIS.md')))
    print('CPU PASS',flush=True)

def launch():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    assert len(info.strip().splitlines())==4
    for i,line in enumerate(info.strip().splitlines()):
        r=[x.strip() for x in line.split(',')]
        assert int(r[0])==i and int(r[2])<256 and int(r[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    checked=read(HERE/'check.json')
    assert checked['status']=='passed' and checked['inputs']==inputs()
    assert checked['source_hashes']==source_hashes() and checked['protocol_sha256']==digest(HERE/'OPIS.md')
    run=ROOT/'runs/E032'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True,exist_ok=False);(run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.is_file():
            shutil.copyfile(p,run/'source'/p.name)
    shutil.copyfile(REFERENCE/'indices.json',run/'indices.json')
    m=dict(status='running',started_utc=utc(),launcher_pid=os.getpid(),assignments={str(i):42+i for i in range(4)},
           inputs=checked['inputs'],source_hashes=source_hashes(),protocol_sha256=checked['protocol_sha256'],
           indices_sha256=digest(run/'indices.json'),upstream_commit=c.COMMIT,gpus=info,
           python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
           official_test_used=False,reserve_used=False,training=False,batch_size=128)
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
        m['exit_codes']=codes
        assert codes==[0]*4
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
        for ds in DS:
            a,x,y,views=c.data(ds,read(run/'indices.json')[ds])
            tv,_=c.load_source(ds,seed)
            controls,_=c.teacher_controls(ds,seed,tv,None,a,x,y)
            ref=read(REFERENCE/f'seed_{seed}'/f'controls_{ds}.json')
            assert views==ref['views_sha256']
            assert all(abs(controls[mode]['top1']-ref['controls'][mode]['top1'])<1e-12 for mode in MODES)
            write(out/ds/'controls.json',dict(controls=controls,views_sha256=views,baseline_agrees=True))
            for index,name in enumerate(c.legacy.group_names()):
                base,diag=fold(tv,index)
                metrics={};bn_checks={}
                for mode in MODES:
                    model=copy.deepcopy(base).to('cuda')
                    if mode!='PRE':
                        bn_checks[mode]=c.e29.calibrate(model,a,mode)
                    z=c.bn.predict(model,x)
                    assert torch.isfinite(z).all()
                    metrics[mode]=dict(top1=float((z.argmax(1)==y).double().mean()),nll=float(torch.nn.functional.cross_entropy(z,y)))
                    del model
                result=dict(status='completed',seed=seed,gpu=gpu,dataset=ds,index=index,group=name,
                            metrics=metrics,calibrations=bn_checks,diagnostics=diag,
                            views_sha256=views,official_test_used=False,reserve_used=False)
                write(out/ds/f'group_{index:02d}.json',result)
                status['completed'].append(f'{ds}/{index}');write(out/'status.json',status)
                print(seed,ds,index,metrics,flush=True)
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
    rows=[];profiles=[];hashes={};arrays={}
    for ds in DS:
        for seed in range(42,46):
            controls=read(run/f'seed_{seed}'/ds/'controls.json')['controls']
            assert set(controls)==set(MODES)
            items=[read(run/f'seed_{seed}'/ds/f'group_{i:02d}.json') for i in range(12)]
            for mode in MODES:
                damage=[100*(controls[mode]['top1']-s['metrics'][mode]['top1']) for s in items]
                arrays[(ds,seed,mode)]=damage
                rho=float(spearmanr(range(12),damage).statistic)
                gap=float(np.mean(damage[9:])-np.mean(damage[:3]))
                profiles.append(dict(dataset=ds,seed=seed,mode=mode,rho=rho,late_minus_early=gap,
                    mean_damage=float(np.mean(damage)),early=float(np.mean(damage[:3])),late=float(np.mean(damage[9:])),
                    h1_pass=rho>=.5 and gap>0))
                for i,s in enumerate(items):
                    assert s['status']=='completed' and s['index']==i and s['seed']==seed
                    assert not s['official_test_used'] and not s['reserve_used']
                    if mode!='PRE':
                        assert s['calibrations'][mode]['params_unchanged']
                    rows.append(dict(dataset=ds,seed=seed,index=i,group=s['group'],mode=mode,
                        top1=100*s['metrics'][mode]['top1'],teacher=100*controls[mode]['top1'],damage=damage[i],
                        parameters=s['diagnostics']['params']))
    assert len(rows)==288
    primary=[p for p in profiles if p['dataset']=='cifar100' and p['mode']=='CMA']
    contrast=[next(p['rho'] for p in profiles if p['dataset']=='cifar100' and p['seed']==s and p['mode']=='CMA')-
              next(p['rho'] for p in profiles if p['dataset']=='cifar10' and p['seed']==s and p['mode']=='CMA') for s in range(42,46)]
    h1=sum(p['h1_pass'] for p in primary)>=3
    h2=float(np.mean(contrast))>=.3 and sum(x>0 for x in contrast)>=3
    relationships=[];stability=[]
    for ds in DS:
        for mode in ('CMA','UP_REPAIR'):
            gaps=[next(p['late_minus_early'] for p in profiles if p['dataset']==ds and p['seed']==s and p['mode']==mode) for s in range(42,46)]
            gain=[]
            for seed in range(42,46):
                r=read(REFERENCE/f'seed_{seed}'/ds/'b70/summary.json')['metrics']
                gain.append(100*(r['UP_FIXED_T'][mode]['top1']-r['UP_FIXED_U'][mode]['top1']))
            relationships.append(dict(dataset=ds,mode=mode,gaps=gaps,gains=gain,rho=float(spearmanr(gaps,gain).statistic)))
            for s in range(42,46):
                for t in range(s+1,46):
                    stability.append(dict(dataset=ds,mode=mode,seed_a=s,seed_b=t,rho=float(spearmanr(arrays[(ds,s,mode)],arrays[(ds,t,mode)]).statistic)))
    for p in run.glob('seed_*/*/*.json'):
        hashes[str(p.relative_to(run))]=digest(p)
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert size<536870912 and m['wall_seconds']<=5400
    out=run/f'analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir()
    result=dict(H1=h1,H1_count=sum(p['h1_pass'] for p in primary),H2=h2,contrast=contrast,
                profiles=profiles,relationships=relationships,stability=stability,source_hashes=hashes,
                wall_seconds=m['wall_seconds'],artifacts_bytes=size,rows=288,controls=24)
    write(out/'analysis.json',result)
    for name,data in [('groups',rows),('profiles',profiles),('stability',stability)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(result,indent=2));print('ANALYSIS='+str(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['check','run','worker','analyze'])
    ap.add_argument('--run-dir',type=Path);ap.add_argument('--gpu',type=int)
    args=ap.parse_args()
    if args.action=='check': check()
    elif args.action=='run': launch()
    elif args.action=='worker': worker(args.run_dir,args.gpu)
    else: analyze(args.run_dir)
