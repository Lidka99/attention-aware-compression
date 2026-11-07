"""E035: zamrożone źródła -> CPU/dev gate -> niezależna ocena4GPU; OPIS.md.
Wejście: checkpointy/plany E029–31/E034; wyjście provenance, metryki i analiza.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
e31=load(ROOT/'experiments/2026-11-02_E031_oficjalny_30_70/common.py','_e035_e31')
e34=load(ROOT/'experiments/2026-11-05_E034_preactresnet18/fold_preact.py','_e035_e34')
e29=e31.e29;e30=e31.e30;bn=e31.bn
read=e31.read;digest=e31.digest
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def utc():return datetime.now(timezone.utc).isoformat()
def rel(p):return str(Path(p).relative_to(ROOT))
R31=ROOT/'runs/E031/20261104T212951Z_6c3d7356'
R29=e31.E029_RUN;R30=e31.E030_RUN
R34=ROOT/'runs/E034/fold_20261106T003625Z_9ec4f2ea'
TRAIN34=ROOT/'runs/E034/train_20261105T014209Z_21b44f52'
PAIRS=('resnet_cifar10','resnet_cifar100','resnet_tiny','preact_cifar10','preact_cifar100')
TAGS=('b30','b50','b70');MODES=('PRE','CMA','UP_REPAIR');SEEDS=range(42,46)
PROTOCOL_COMMIT='f42c0b8'

def source_paths(pair,tag,seed):
    arch,ds=pair.split('_')
    if arch=='preact':return R34/f'seed_{seed}/{ds}/{tag}/summary.json',R34/f'seed_{seed}/controls_{ds}.json'
    if tag!='b50':return R31/f'seed_{seed}/{ds}/{tag}/summary.json',R31/f'seed_{seed}/controls_{ds}.json'
    p=R30/f'seed_{seed}/summary.json' if ds=='tiny' else R29/f'seed_{seed}/{ds}/summary.json'
    return p,p

def checkpoint(pair,seed):
    arch,ds=pair.split('_')
    if arch=='preact':return TRAIN34/f'seed_{seed}/{ds}/last.pt'
    return {'cifar10':e31.CIFAR10_RUN,'cifar100':e31.CIFAR100_RUN,'tiny':e31.TINY_RUN}[ds]/f'seed_{seed}/last.pt'

def model_source(pair,seed):
    arch,ds=pair.split('_')
    if arch=='preact':return e34.source_model(TRAIN34,ds,seed)[0]
    tv,up=e31.load_source(ds,seed)
    return tv if up is None else up

def frozen_inputs():
    provenance=e31.provenance()
    plans={};refs={}
    for pair in PAIRS:
        plans[pair]={};refs[pair]={}
        for seed in SEEDS:
            p=checkpoint(pair,seed);provenance[rel(p)]=digest(p)
        for tag in TAGS:
            refs[pair][tag]={}
            for seed in SEEDS:
                p,c=source_paths(pair,tag,seed);s=read(p);ctrl=read(c)
                assert s['status']=='completed'
                provenance[rel(p)]=digest(p);provenance[rel(c)]=digest(c)
                plan=s['plan'];pre=pair.startswith('preact')
                canonical=dict(U=plan['U'] if pre else plan['uniform'],T=plan['T'] if pre else plan['tilt'],
                               ratio=plan['ratio'],params=dict(U=plan['target'],T=plan['tilt_params']))
                if tag in plans[pair]:assert plans[pair][tag]==canonical
                plans[pair][tag]=canonical
                refs[pair][tag][str(seed)]=dict(source=rel(p),controls_source=rel(c),
                    metrics={k:s['metrics'][('UP_' if pre else 'UP_FIXED_')+k] for k in ('U','T')},
                    controls=ctrl['controls'],views_sha256=s.get('views_sha256',ctrl.get('views_sha256')))
    indices=read(R31/'indices.json')
    for run in (R29,R30,R31,R34):
        for name in ('manifest.json','plans.json','indices.json'):
            p=run/name
            if run==R29 and name=='indices.json':continue  # E029 korzysta z E028/E024 bez kopii.
            provenance[rel(p)]=digest(p)
    assert read(R34/'indices.json')==indices
    for ds in ('cifar10','cifar100'):
        ids=read(e31.E028_INDICES if ds=='cifar10' else e31.E024_INDICES)
        assert ids['A']==indices[ds]['A'] and ids['dev']==indices[ds]['dev']
    ids=read(R30/'indices.json')
    assert ids['A']==indices['tiny']['A'] and ids['dev']==indices['tiny']['dev']
    # Hash pełnej lokalnej bazy kodu helperów, bez zależności spoza repo.
    for p in (ROOT/'experiments').rglob('*.py'):
        if HERE not in p.parents:provenance[rel(p)]=digest(p)
    return dict(plans=plans,references=refs,indices=indices,provenance=provenance)

def code_hashes():return {rel(p):digest(p) for p in HERE.glob('*.py')}
def validate_hashes(mapping):
    for p,h in mapping.items():assert digest(ROOT/p)==h,p

def fold(pair,base,plan,profile):
    if pair.startswith('preact'):model,diag=e34.official(base,plan['ratio'],plan[profile])
    else:
        # U zachowuje oryginalny globalny keep_ratio; T adapter per grupa.
        model,diag=e29.official(base,plan['ratio'],None if profile=='U' else plan['T'])
        model.fc.bias.data.copy_(base.fc.bias)
    model.eval().requires_grad_(False)
    assert [d['k'] for d in diag]==plan[profile]
    assert sum(p.numel() for p in model.parameters())==plan['params'][profile]
    bias='linear' if pair.startswith('preact') else 'fc'
    assert torch.equal(getattr(model,bias).bias,getattr(base,bias).bias)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return model,diag

def check():
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    frozen=frozen_inputs();checks=[]
    for pair in PAIRS:
        base=model_source(pair,42)
        for tag in TAGS:
            plan=frozen['plans'][pair][tag]
            geo=e34.Geometry(base) if pair.startswith('preact') else bn.Geometry(base)
            for profile in ('U','T'):assert geo.count(plan[profile])==plan['params'][profile]
            assert .999*plan['params']['U']<=plan['params']['T']<=plan['params']['U']
        for profile in ('U','T'):
            model,diag=fold(pair,base,frozen['plans'][pair]['b50'],profile)
            size=64 if pair.endswith('tiny') else 32
            with torch.inference_mode():z=model(torch.zeros(2,3,size,size))
            assert torch.isfinite(z).all()
            checks.append(dict(pair=pair,profile=profile,groups=len(diag),params=sum(p.numel() for p in model.parameters())))
    write(HERE/'frozen.json',frozen)
    write(HERE/'check.json',dict(status='passed',utc=utc(),checks=checks,code_hashes=code_hashes(),
          frozen_sha256=digest(HERE/'frozen.json'),protocol_sha256=digest(HERE/'OPIS.md')))
    print('CPU PASS',len(checks),flush=True)

def test_paths():
    return dict(cifar10=e31.d10.DATA/'test_batch',cifar100=ROOT/'data/cifar100/cifar-100-python/test',
                tiny=e31.dt.DATA_ROOT/'cache_val.npz')

def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def prepare_test(run):
    assert read(run/'dev_gate.json')['passed']
    paths=test_paths();p=paths['cifar10'];archive=None;downloaded=False
    if not p.exists():
        candidates=list((ROOT/'data').rglob('cifar-10-python.tar.gz'))
        archive=candidates[0] if candidates else ROOT/'data/cifar10/cifar-10-python.tar.gz'
        if not archive.exists():
            url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            urllib.request.urlretrieve(url,archive);downloaded=True
        assert md5(archive)=='c58f30108f718f92721af3b95e74349a'
        with tarfile.open(archive,'r:gz') as tar:
            member=tar.getmember('cifar-10-batches-py/test_batch')
            assert member.isfile()
            with tar.extractfile(member) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
    assert md5(p)=='40351d587109b95175f43aff81a1287e'
    assert md5(paths['cifar100'])=='f0ef6b0ae62326f3e7ffdfab6717acfc'
    reserve=read(e31.dt.SPLITS_FILE)['splits']['reserve'];assert sorted(reserve)==list(range(10000))
    data=dict(utc=utc(),files={rel(p):digest(p) for p in paths.values()},downloaded=downloaded,
        cifar10_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        archive=rel(archive) if archive else None,archive_sha256=digest(archive) if archive else None,
        archive_md5=md5(archive) if archive else None,cifar100_reserve_used=False,
        tiny_prior_access='E005/E007 shape checks, no model evaluation',
        indices={ds:reserve if ds=='tiny' else list(range(10000)) for ds in paths})
    write(run/'test_data.json',data)
    print('TEST DATA PREPARED',flush=True)

def independent_data(ds,run):
    meta=read(run/'test_data.json');p=test_paths()[ds]
    assert digest(p)==meta['files'][rel(p)]
    ids=meta['indices'][ds]
    if ds=='tiny':
        with np.load(p,allow_pickle=False) as cache:
            raw=cache['images'];labels=cache['labels']
        assert raw.shape==(10000,64,64,3)
        x=e30.normalize(torch.from_numpy(raw[ids]).permute(0,3,1,2).float()/255)
        y=torch.tensor(labels[ids],dtype=torch.long)
    else:
        with p.open('rb') as f:payload=pickle.load(f,encoding='latin1')
        raw=np.asarray(payload['data'],dtype=np.uint8);assert raw.shape==(10000,3072)
        normalize=e31.d10.normalize if ds=='cifar10' else bn.normalize
        x=normalize(torch.from_numpy(raw[ids].reshape(-1,3,32,32)))
        y=torch.tensor(np.asarray(payload['labels' if ds=='cifar10' else 'fine_labels'])[ids],dtype=torch.long)
    classes={'cifar10':10,'cifar100':100,'tiny':200}[ds]
    assert len(y)==10000 and torch.equal(torch.bincount(y),torch.full((classes,),10000//classes))
    return x,y

def evaluate(pair,base,a,x,y,mode):
    model=copy.deepcopy(base)
    model=e34.cuda_model(model) if pair.startswith('preact') else model.to('cuda')
    before=bn.weights_hash(model)
    calibration=None
    if mode!='PRE':calibration=(e34.calibrate if pair.startswith('preact') else e29.calibrate)(model,a,mode)
    z=bn.predict(model,x)
    assert torch.isfinite(z).all()
    assert bn.weights_hash(model)==before
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    metrics=dict(top1=float((z.argmax(1)==y).double().mean()),nll=float(F.cross_entropy(z,y)))
    evidence=dict(weights_sha256=before,weights_unchanged=True,no_grad=True,calibration=calibration,
                  state_sha256=state_hash(model),samples=len(y))
    del model
    return metrics,evidence

def state_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def worker(run,phase,gpu):
    seed=42+gpu;assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(seed);np.random.seed(seed)
    frozen=read(run/'frozen.json');out=run/phase/f'seed_{seed}';out.mkdir(parents=True)
    status=dict(status='running',pid=os.getpid(),seed=seed,gpu=gpu,utc=utc(),completed=[])
    write(out/'status.json',status)
    try:
        if phase=='test':assert read(run/'dev_gate.json')['passed']
        for pair in PAIRS:
            arch,ds=pair.split('_');a,dx,dy,views=e31.data(ds,frozen['indices'][ds])
            x,y=(dx,dy) if phase=='dev' else independent_data(ds,run)
            base=model_source(pair,seed);controls={};control_evidence={}
            for mode in MODES:controls[mode],control_evidence[mode]=evaluate(pair,base,a,x,y,mode)
            write(out/pair/'controls.json',dict(metrics=controls,evidence=control_evidence,views_sha256=views))
            for tag in TAGS:
                plan=frozen['plans'][pair][tag];metrics={};evidence={};clusters={}
                for profile in ('U','T'):
                    folded,diag=fold(pair,base,plan,profile);metrics[profile]={};evidence[profile]={};clusters[profile]=diag
                    for mode in MODES:
                        metrics[profile][mode],evidence[profile][mode]=evaluate(pair,folded,a,x,y,mode)
                    del folded
                write(out/pair/tag/'summary.json',dict(status='completed',pair=pair,tag=tag,seed=seed,gpu=gpu,
                    phase=phase,plan=plan,metrics=metrics,evidence=evidence,clusters=clusters,views_sha256=views,
                    checkpoint_sha256=digest(checkpoint(pair,seed)),bias_preserved=True,
                    controls_sha256=digest(out/pair/'controls.json')))
                status['completed'].append(pair+'/'+tag);write(out/'status.json',status)
                print(phase,seed,pair,tag,{k:metrics[k]['CMA']['top1'] for k in ('U','T')},flush=True)
            del a,dx,dy,x,y,base
        status.update(status='completed',finished_utc=utc())
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:write(out/'status.json',status)

def dev_gate(run):
    frozen=read(run/'frozen.json');errors=[];maximum=dict(top1=0.,nll=0.);n=0;nc=0
    for pair in PAIRS:
        for seed in SEEDS:
            ctrl=read(run/f'dev/seed_{seed}/{pair}/controls.json')
            for tag in TAGS:
                s=read(run/f'dev/seed_{seed}/{pair}/{tag}/summary.json');ref=frozen['references'][pair][tag][str(seed)]
                assert s['plan']==frozen['plans'][pair][tag]
                if ref['views_sha256']:assert s['views_sha256']==ref['views_sha256']
                for profile in ('U','T'):
                    for mode in MODES:
                        n+=1
                        for key,tol in [('top1',1e-10),('nll',1e-5)]:
                            diff=abs(s['metrics'][profile][mode][key]-ref['metrics'][profile][mode][key])
                            maximum[key]=max(maximum[key],diff)
                            if diff>tol:errors.append([pair,tag,seed,profile,mode,key,diff])
                # Kontrole porównane ze wszystkimi właściwymi źródłami (bez potrójnego zliczania).
                for mode in MODES:
                    for key,tol in [('top1',1e-10),('nll',1e-5)]:
                        diff=abs(ctrl['metrics'][mode][key]-ref['controls'][mode][key]);maximum[key]=max(maximum[key],diff)
                        if diff>tol:errors.append([pair,tag,seed,'teacher',mode,key,diff])
            nc+=3
    assert (n,nc)==(360,60)
    gate=dict(passed=not errors,utc=utc(),errors=errors,maximum=maximum,evaluations=n,controls=nc)
    write(run/'dev_gate.json',gate);assert gate['passed'],errors[:10]
    print('DEV GATE PASS',maximum,flush=True)

def gpu_info():
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    assert len(info.strip().splitlines())==4
    for i,line in enumerate(info.strip().splitlines()):
        r=[v.strip() for v in line.split(',')];assert int(r[0])==i and int(r[2])<256 and int(r[3])==0,info
    return info

def launch():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    info=gpu_info();checked=read(HERE/'check.json')
    assert checked['status']=='passed' and checked['code_hashes']==code_hashes()
    assert checked['protocol_sha256']==digest(HERE/'OPIS.md') and checked['frozen_sha256']==digest(HERE/'frozen.json')
    frozen=read(HERE/'frozen.json');validate_hashes(frozen['provenance'])
    run=ROOT/'runs/E035'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True);(run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.is_file():shutil.copyfile(p,run/'source'/p.name)
    shutil.copyfile(HERE/'frozen.json',run/'frozen.json')
    m=dict(status='running',utc=utc(),launcher_pid=os.getpid(),gpus=info,assignments={str(i):42+i for i in range(4)},
        code_hashes=code_hashes(),protocol_sha256=digest(HERE/'OPIS.md'),frozen_sha256=digest(run/'frozen.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),protocol_commit=PROTOCOL_COMMIT,
        python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,limit_seconds=5400,phases={},
        environment={k:os.environ.get(k) for k in ('CONDA_DEFAULT_ENV','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')})
    write(run/'manifest.json',m)
    (run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('RUN='+str(run),flush=True);start=time.monotonic();processes=[]
    try:
        for phase in ('dev','test'):
            if phase=='test':
                dev_gate(run);prepare_test(run);gpu_info()
            processes=[];phase_start=time.monotonic()
            for gpu in range(4):
                log=(run/'logs'/f'{phase}_gpu{gpu}.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                p=subprocess.Popen([sys.executable,'-B',str(HERE/'experiment.py'),'worker','--run',str(run),'--phase',phase,'--gpu',str(gpu)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                processes.append((p,log))
            m['phases'][phase]=dict(pids=[p.pid for p,_ in processes],started_utc=utc());write(run/'manifest.json',m)
            codes=[p.wait(timeout=max(.1,5400-(time.monotonic()-start))) for p,_ in processes]
            m['phases'][phase].update(exit_codes=codes,seconds=time.monotonic()-phase_start,finished_utc=utc())
            write(run/'manifest.json',m);assert codes==[0]*4,codes
            for p,log in processes:log.close()
            for seed in SEEDS:assert read(run/f'{phase}/seed_{seed}/status.json')['status']=='completed'
        validate_hashes(frozen['provenance']);validate_hashes(read(run/'test_data.json')['files'])
        m['status']='completed'
    except BaseException:
        m.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        for p,log in processes:
            if p.poll() is None:p.kill();p.wait()
            log.close()
        m.update(finished_utc=utc(),wall_seconds=time.monotonic()-start);write(run/'manifest.json',m)
    print('COMPLETED',run,flush=True)

def stats(values):return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),per_seed=values)

def analyze(run):
    m=read(run/'manifest.json');assert m['status']=='completed'
    rows=[];controls=[];contrasts=[]
    for phase in ('dev','test'):
        for pair in PAIRS:
            ctrl={s:read(run/f'{phase}/seed_{s}/{pair}/controls.json')['metrics'] for s in SEEDS}
            for seed in SEEDS:
                for mode in MODES:controls.append(dict(phase=phase,pair=pair,seed=seed,mode=mode,**ctrl[seed][mode]))
            for tag in TAGS:
                data={s:read(run/f'{phase}/seed_{s}/{pair}/{tag}/summary.json')['metrics'] for s in SEEDS}
                for seed in SEEDS:
                    for profile in ('U','T'):
                        for mode in MODES:
                            val=data[seed][profile][mode]
                            rows.append(dict(phase=phase,pair=pair,tag=tag,seed=seed,profile=profile,mode=mode,**val,
                                teacher_loss_pp=100*(ctrl[seed][mode]['top1']-val['top1'])))
                for mode in MODES:
                    diffs=[100*(data[s]['T'][mode]['top1']-data[s]['U'][mode]['top1']) for s in SEEDS]
                    nll=[data[s]['T'][mode]['nll']-data[s]['U'][mode]['nll'] for s in SEEDS]
                    st=stats(diffs);positive=sum(d>0 for d in diffs)
                    contrasts.append(dict(phase=phase,pair=pair,tag=tag,mode=mode,**st,positive=positive,
                        direction=st['mean']>0 and positive>=3,magnitude=(st['mean']>=2-1e-12 and positive>=3) if tag!='b30' else None,
                        nll_delta=stats(nll),absolute={k:stats([100*data[s][k][mode]['top1'] for s in SEEDS]) for k in ('U','T')},
                        teacher=stats([100*ctrl[s][mode]['top1'] for s in SEEDS]),
                        teacher_loss={k:stats([100*(ctrl[s][mode]['top1']-data[s][k][mode]['top1']) for s in SEEDS]) for k in ('U','T')}))
    assert len(rows)==720 and len(controls)==120
    primary=[r for r in contrasts if r['phase']=='test' and r['mode']=='CMA']
    result=dict(utc=utc(),rows=rows,controls=controls,contrasts=contrasts,primary_pass=all(r['direction'] for r in primary),
        primary_pass_n=sum(r['direction'] for r in primary),magnitude_pass_n=sum(bool(r['magnitude']) for r in primary if r['tag']!='b30'))
    write(run/'analysis/analysis.json',result)
    import csv
    for name,values in [('per_seed',rows),('controls',controls)]:
        with (run/f'analysis/{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','controls','contrasts')},indent=2))
    for r in primary:print(r['pair'],r['tag'],r['mean'],r['sd'],r['positive'],r['direction'],r['magnitude'])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['check','run','worker','analyze'])
    ap.add_argument('--run',type=Path);ap.add_argument('--phase',choices=['dev','test']);ap.add_argument('--gpu',type=int)
    args=ap.parse_args()
    if args.action=='check':check()
    elif args.action=='run':launch()
    elif args.action=='worker':worker(args.run,args.phase,args.gpu)
    else:analyze(args.run)
