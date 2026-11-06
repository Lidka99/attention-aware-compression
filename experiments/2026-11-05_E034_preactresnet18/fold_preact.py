"""E034: folding/audyt PreActResNet18 na istniejących checkpointach; OPIS.md.
Wejście: run treningowy E034; wyjście plany, oceny dev i analysis, bez treningu.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
UP=ROOT/'runs/E029/upstream_6946915'
COMMIT='694691556e8314a835178a741938dd3d4d80c171'
DATASETS=('cifar10','cifar100')
SEEDS=(42,43,44,45)
BUDGETS=(.30,.50,.70)
TAGS={.30:'b30',.50:'b50',.70:'b70'}
PROFILES=('U','T')
MODES=('PRE','CMA','UP_REPAIR')

if 'open_clip' not in sys.modules and importlib.util.find_spec('open_clip') is None:
    import types
    optional=types.ModuleType('open_clip')
    def unused(*args,**kwargs): raise RuntimeError('E034: CLIP poza zakresem')
    optional.create_model_and_transforms=unused
    sys.modules['open_clip']=optional
sys.path.insert(0,str(UP))
from models.preact_resnet import PreActResNet18,get_axis_to_perm_PreActResNet18
from compression.fold import PreActResNet18_ModelFolding
from utils.weight_clustering import axes2perm_to_perm2axes
from utils.tune_utils import repair_bn


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


d10=load(ROOT/'experiments/2026-11-01_E028_transfer_cifar10/data10.py','_e034f_d10')
d100=load(ROOT/'experiments/2026-10-31_E024_optymalizacja_alokacji/data.py','_e034f_d100')
bn=load(ROOT/'experiments/2026-11-01_E025_diagnostyka_bn/common.py','_e034f_bn')
write=lambda p,v: Path(p).write_text(json.dumps(v,indent=2)+'\n')
read=lambda p: json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def source_hashes(): return {p.name:digest(p) for p in HERE.glob('*.py')}


def inputs():
    files=[d10.SPLIT,d100.SPLIT,ROOT/'data/cifar100/cifar-100-python/train']+[d10.DATA/f'data_batch_{i}' for i in range(1,6)]
    result={str(p.relative_to(ROOT)):digest(p) for p in files}
    result['runs/E029/upstream_6946915/models/preact_resnet.py']=digest(UP/'models/preact_resnet.py')
    result['runs/E029/upstream_6946915/compression/fold.py']=digest(UP/'compression/fold.py')
    result['runs/E029/upstream_6946915/utils/tune_utils.py']=digest(UP/'utils/tune_utils.py')
    return result


class Geometry:
    def __init__(self,model):
        groups=axes2perm_to_perm2axes(get_axis_to_perm_PreActResNet18(override=False))
        self.names=list(groups);self.axes=groups;self.axis_group={}
        state=model.state_dict();self.sizes=[];self.shapes={n:tuple(t.shape) for n,t in model.named_parameters()}
        for name,axes in groups.items():
            lengths=[]
            for pname,axis in axes:
                self.axis_group[(pname,axis)]=name
                lengths.append(state[pname].shape[axis])
            assert len(set(lengths))==1,(name,lengths)
            self.sizes.append(int(lengths[0]))
        assert len(self.names)==13
    def count(self,widths):
        assignment=dict(zip(self.names,map(int,widths)));total=0
        for name,shape in self.shapes.items():
            total+=math.prod(assignment[self.axis_group[(name,d)]] if (name,d) in self.axis_group else n
                             for d,n in enumerate(shape))
        return int(total)
    def fill(self,widths,target):
        w=list(map(int,widths));maxima=[math.floor(.9*n) for n in self.sizes]
        while True:
            best=None
            for i in range(len(w)):
                if w[i]>=maxima[i]: continue
                v=w.copy();v[i]+=1;p=self.count(v)
                if p<=target and (best is None or (p,-i)>(best[0],best[1])): best=(p,i,v)
            if best is None:return w
            w=best[2]
    def profile(self,target,tilt):
        def widths(base):
            ratios=np.clip(base+tilt*np.linspace(-1,1,13),.1,.9)
            return [int(n*(1-r)) for n,r in zip(self.sizes,ratios)]
        lo,hi=-abs(tilt),1+abs(tilt)
        assert self.count(widths(lo))>=target>=self.count(widths(hi))
        for _ in range(60):
            mid=(lo+hi)/2
            if self.count(widths(mid))>target:lo=mid
            else:hi=mid
        result=widths(hi)
        return self.fill(result,target) if tilt else result


def profiles(model,compression):
    geo=Geometry(model);target=round((1-compression)*geo.count(geo.sizes))
    u=geo.profile(target,0);shared=geo.count(u);t=geo.profile(shared,.30)
    assert .999*shared<=geo.count(t)<=shared
    return geo,u,t,1-min(w/n for w,n in zip(u,geo.sizes))


def official(model,ratio,widths=None):
    torch.manual_seed(42);np.random.seed(42)
    work=copy.deepcopy(model)
    op=PreActResNet18_ModelFolding(work,compression_ratio=ratio)
    original=op.compress_function;diag=[]
    def observed(axes,params):
        n=int(params[axes[0][0]].shape[axes[0][1]])
        if widths is not None:op.keep_ratio=(int(widths[len(diag)])+.25)/n
        k=max(int(n*op.keep_ratio),1)
        compressed,meta=original(axes,params)
        labels=meta['cluster_labels'].cpu()
        assert torch.equal(torch.unique(labels),torch.arange(k)),'Puste klastry PreAct'
        diag.append(dict(n=n,k=k,labels=labels.tolist(),first_axis=axes[0][0]))
        return compressed,meta
    op.compress_function=observed
    result=op.apply().eval().requires_grad_(False)
    assert len(diag)==13
    if hasattr(model,'linear') and hasattr(model.linear,'bias'):
        result.linear.bias=torch.nn.Parameter(model.linear.bias.detach().clone(),requires_grad=False)
    return result,diag


def cuda_model(model):
    model=model.to('cuda')
    model.normalize.mu=torch.zeros(1,3,1,1,device='cuda')
    model.normalize.std=torch.ones(1,3,1,1,device='cuda')
    return model


def source_model(train_run,ds,seed):
    path=train_run/f'seed_{seed}'/ds/'last.pt';assert path.is_file()
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model=PreActResNet18(n_cls=10 if ds=='cifar10' else 100,cuda=False)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval().requires_grad_(False),path


def data(ds,ids):
    if ds=='cifar10':
        images=d10.Images();aug=bn.augment_uint8(images.raw,ids['A']);a=d10.normalize(aug);dx,dy=images.dev(ids['dev'])
    else:
        images=d100.Images();aug=bn.augment_uint8(images.raw,ids['A']);a=bn.normalize(aug);dx,dy=images.dev(ids['dev'])
    return a,dx,dy,bn.tensor_hash(aug)


def calibrate(model,a,mode):
    if mode=='CMA': return bn.calibrate(model,a.split(128),'CMA_reset')
    assert mode=='UP_REPAIR'
    before=bn.weights_hash(model)
    for layer in model.modules():
        if isinstance(layer,torch.nn.BatchNorm2d): layer.momentum=.1
    loader=[(batch,torch.zeros(len(batch),dtype=torch.long)) for batch in a.split(128)]
    repair_bn(model,loader,device='cuda');model.eval()
    assert bn.weights_hash(model)==before
    return dict(params_unchanged=True,samples=len(a),batches=len(loader))


def plans(train_run):
    result={}
    for ds in DATASETS:
        tv,_=source_model(train_run,ds,42);result[ds]={}
        for b in BUDGETS:
            geo,u,t,ratio=profiles(tv,b);tag=TAGS[b]
            result[ds][tag]=dict(U=u,T=t,target=geo.count(u),tilt_params=geo.count(t),
                                 ratio=ratio,compression=b,params={k:geo.count(v) for k,v in dict(U=u,T=t).items()})
    return result


def check():
    assert not torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(42)
    model=PreActResNet18(n_cls=10,cuda=False).eval()
    geo,u,t,ratio=profiles(model,.50)
    assert len(geo.names)==13 and sum(p.numel() for p in model.parameters())==geo.count(geo.sizes)
    folded,diag=official(model,ratio,u)
    assert len(diag)==13 and sum(p.numel() for p in folded.parameters())==geo.count(u)
    with torch.inference_mode(): assert torch.isfinite(folded(torch.randn(2,3,32,32))).all()
    write(HERE/'check_fold.json',dict(status='passed',utc=utc(),inputs=inputs(),
          source_hashes=source_hashes(),protocol_sha256=digest(HERE/'OPIS.md'),
          groups=geo.names,sizes=geo.sizes,params=sum(p.numel() for p in model.parameters()),
          profile_u=u,profile_t=t,folded_params=geo.count(u),commit=COMMIT))
    print('CPU PASS',geo.names,geo.sizes,flush=True)


def fold_worker(fold_run,train_run,gpu):
    seed=42+gpu
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    out=fold_run/f'seed_{seed}';out.mkdir()
    status=dict(status='running',seed=seed,gpu=gpu,pid=os.getpid(),started_utc=utc(),completed=[])
    write(out/'status.json',status)
    try:
        for ds in DATASETS:
            a,x,y,views=data(ds,read(fold_run/'indices.json')[ds])
            tv,_=source_model(train_run,ds,seed);teacher=cuda_model(copy.deepcopy(tv))
            controls={'PRE':dict(top1=float((bn.predict(teacher,x).argmax(1)==y).double().mean()),
                                 nll=float(F.cross_entropy(bn.predict(teacher,x),y)))}
            del teacher
            for mode in ('CMA','UP_REPAIR'):
                model=cuda_model(copy.deepcopy(tv));calibrate(model,a,mode)
                z=bn.predict(model,x);controls[mode]=dict(top1=float((z.argmax(1)==y).double().mean()),
                                                          nll=float(F.cross_entropy(z,y)));del model
            write(out/f'controls_{ds}.json',dict(controls=controls,views_sha256=views))
            for tag in ('b30','b50','b70'):
                plan=read(fold_run/'plans.json')[ds][tag]
                (out/ds/tag).mkdir(parents=True,exist_ok=True)
                metrics={};params={};diags={}
                for profile in PROFILES:
                    folded,diag=official(tv,plan['ratio'],plan[profile])
                    assert sum(p.numel() for p in folded.parameters())==plan['params'][profile]
                    params['UP_'+profile]=plan['params'][profile];diags['UP_'+profile]=diag
                    metrics['UP_'+profile]={}
                    for mode in MODES:
                        model=cuda_model(copy.deepcopy(folded))
                        if mode!='PRE': calibrate(model,a,mode)
                        z=bn.predict(model,x);metrics['UP_'+profile][mode]=dict(
                            top1=float((z.argmax(1)==y).double().mean()),
                            nll=float(F.cross_entropy(z,y)));del model
                    del folded
                summary=dict(status='completed',dataset=ds,tag=tag,seed=seed,gpu=gpu,plan=plan,
                             metrics=metrics,params=params,cluster_diag=diags,views_sha256=views,
                             controls_sha256=digest(out/f'controls_{ds}.json'),
                             official_test_used=False,reserve_used=False)
                write(out/ds/tag/'summary.json',summary)
                status['completed'].append(f'{ds}/{tag}');write(out/'status.json',status)
                print(seed,ds,tag,{k:{m:round(v['top1']*100,2) for m,v in metrics[k].items()} for k in metrics},flush=True)
        status.update(status='completed',finished_utc=utc())
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        write(out/'status.json',status)


def launch(train_run):
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    assert len(info.strip().splitlines())==4
    for i,line in enumerate(info.strip().splitlines()):
        r=[x.strip() for x in line.split(',')]
        assert int(r[0])==i and int(r[2])<256 and int(r[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    checked=read(HERE/'check_fold.json')
    assert checked['status']=='passed' and checked['inputs']==inputs()
    assert checked['source_hashes']==source_hashes() and checked['protocol_sha256']==digest(HERE/'OPIS.md')
    train_run=train_run.resolve();manifest=read(train_run/'manifest.json')
    assert manifest['status']=='completed' and not manifest['smoke']
    for seed in SEEDS:
        for ds in DATASETS:
            p=train_run/f'seed_{seed}'/ds/'last.pt';assert p.is_file()
    run=ROOT/'runs/E034'/f'fold_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True,exist_ok=False);(run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.is_file(): shutil.copyfile(p,run/'source'/p.name)
    shutil.copyfile(ROOT/'runs/E031/20261104T212951Z_6c3d7356/indices.json',run/'indices.json')
    write(run/'plans.json',plans(train_run))
    m=dict(status='running',started_utc=utc(),launcher_pid=os.getpid(),assignments={str(i):42+i for i in range(4)},
           inputs=checked['inputs'],source_hashes=source_hashes(),protocol_sha256=checked['protocol_sha256'],
           train_run=str(train_run),train_manifest_sha256=digest(train_run/'manifest.json'),
           indices_sha256=digest(run/'indices.json'),upstream_commit=COMMIT,gpus=info,
           python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
           official_test_used=False,reserve_used=False,limits=dict(wall_seconds=5400,artifacts_bytes=536870912))
    write(run/'manifest.json',m)
    (run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('RUN='+str(run),flush=True)
    processes=[];start=time.monotonic()
    try:
        for gpu in range(4):
            log=(run/'logs'/f'gpu{gpu}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'fold_preact.py'),'worker','--fold-run',str(run),
                                '--train-run',str(train_run),'--gpu',str(gpu)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            processes.append((p,log))
        m['worker_pids']=[p.pid for p,_ in processes];write(run/'manifest.json',m)
        codes=[p.wait(timeout=max(.1,5400-(time.monotonic()-start))) for p,_ in processes]
        m['exit_codes']=codes;assert codes==[0]*4
        assert all(read(run/f'seed_{s}/status.json')['status']=='completed' for s in SEEDS)
        m['status']='completed'
    except BaseException as error:
        m.update(status='failed',error=repr(error));raise
    finally:
        for p,log in processes:
            if p.poll() is None:p.kill();p.wait()
            log.close()
        m.update(finished_utc=utc(),wall_seconds=time.monotonic()-start)
        write(run/'manifest.json',m)
    print(m['status'],m['exit_codes'],flush=True)


def analyze(fold_run):
    m=read(fold_run/'manifest.json');assert m['status']=='completed' and m['exit_codes']==[0]*4
    train_run=Path(m['train_run']);assert m['inputs']==inputs() and m['source_hashes']==source_hashes()
    assert m['protocol_sha256']==digest(HERE/'OPIS.md') and m['indices_sha256']==digest(fold_run/'indices.json')
    rows=[];summaries={};controls=[];hashes={}
    for ds in DATASETS:
        for tag in ('b30','b50','b70'):
            for seed in SEEDS:
                p=fold_run/f'seed_{seed}'/ds/tag/'summary.json';s=read(p);hashes[str(p.relative_to(fold_run))]=digest(p)
                assert s['status']=='completed' and s['seed']==seed and s['dataset']==ds and s['tag']==tag
                summaries[(ds,tag,seed)]=s
                for profile in ('U','T'):
                    arm='UP_'+profile;assert s['params'][arm]==s['plan']['params'][profile]
                    for mode in MODES:
                        rows.append(dict(dataset=ds,tag=tag,seed=seed,profile=profile,mode=mode,
                                         top1=100*s['metrics'][arm][mode]['top1'],nll=s['metrics'][arm][mode]['nll']))
        for seed in SEEDS:
            c=read(fold_run/f'seed_{seed}'/f'controls_{ds}.json')
            for mode,val in c['controls'].items():controls.append(dict(dataset=ds,seed=seed,mode=mode,top1=100*val['top1'],nll=val['nll']))
    assert len(rows)==144 and len(controls)==24
    averages=[];allocation=[];gates=[]
    for ds in DATASETS:
        for tag in ('b30','b50','b70'):
            for profile in PROFILES:
                for mode in MODES:
                    vals=[r['top1'] for r in rows if r['dataset']==ds and r['tag']==tag and r['profile']==profile and r['mode']==mode]
                    averages.append(dict(dataset=ds,tag=tag,profile=profile,mode=mode,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1))))
            for mode in MODES:
                d=[100*(summaries[(ds,tag,s)]['metrics']['UP_T'][mode]['top1']-summaries[(ds,tag,s)]['metrics']['UP_U'][mode]['top1']) for s in SEEDS]
                st=dict(mean=float(np.mean(d)),sd=float(np.std(d,ddof=1)),positive=sum(x>0 for x in d),per_seed=d)
                gate=(st['mean']>=2.0 and st['positive']>=3) if tag in ('b50','b70') else (st['mean']>0 and st['positive']>=3)
                allocation.append(dict(dataset=ds,tag=tag,mode=mode,**st,gate=gate))
    primary=[a for a in allocation if a['mode']=='CMA']
    result=dict(allocation=allocation,averages=averages,controls=controls,
                primary_pass=all(a['gate'] for a in primary),primary=primary,
                criteria=dict(operator='PreActResNet18 upstream,13grup',training='E034',
                              budgets=list(BUDGETS),modes=list(MODES)),
                source_hashes=hashes,wall_seconds=m['wall_seconds'],rows=len(rows),controls_n=len(controls))
    out=fold_run/f'analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir()
    write(out/'analysis.json',result)
    print(json.dumps(result,indent=1));print('ANALYSIS='+str(out))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['check','fold','worker','analyze'])
    ap.add_argument('--train-run',type=Path);ap.add_argument('--fold-run',type=Path);ap.add_argument('--gpu',type=int)
    args=ap.parse_args()
    if args.action=='check':check()
    elif args.action=='fold':launch(args.train_run)
    elif args.action=='worker':fold_worker(args.fold_run,args.train_run,args.gpu)
    else:analyze(args.fold_run)
