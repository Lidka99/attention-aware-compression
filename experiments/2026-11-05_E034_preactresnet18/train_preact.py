"""E034: trening PreActResNet18 od zera na CIFAR-10/100, 4 seedy; OPIS.md.
Wejście: istniejące dane/podziały E011/E015; wyjście checkpointy i summary.
"""
import argparse
import hashlib
import importlib.util
import json
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
UP=ROOT/'runs/E029/upstream_6946915'
COMMIT='694691556e8314a835178a741938dd3d4d80c171'
DATASETS=('cifar10','cifar100')
SEEDS=(42,43,44,45)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


d10=load(ROOT/'experiments/2026-11-01_E028_transfer_cifar10/data10.py','_e034_d10')
d100=load(ROOT/'experiments/2026-10-31_E024_optymalizacja_alokacji/data.py','_e034_d100')
sys.path.insert(0,str(UP))
from models.preact_resnet import PreActResNet18

write=lambda p,v: (Path(p).write_text(json.dumps(v,indent=2)+'\n'))
read=lambda p: json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {p.name:digest(p) for p in HERE.glob('*.py')}


def inputs():
    files=[d10.SPLIT,d100.SPLIT,ROOT/'data/cifar100/cifar-100-python/train']+[d10.DATA/f'data_batch_{i}' for i in range(1,6)]
    result={str(p.relative_to(ROOT)):digest(p) for p in files}
    result['runs/E029/upstream_6946915/models/preact_resnet.py']=digest(UP/'models/preact_resnet.py')
    return result


def dataset(ds):
    if ds=='cifar10':
        images=d10.Images();raw=images.raw.reshape(-1,3,32,32);labels=images.labels
        split=images.split;mean,std=d10.MEAN,d10.STD
        split_hash=d10.SPLIT_SHA256
    else:
        assert digest(d100.SPLIT)=='ecb8804e41fbc4bee7c61fd0585bf1c7dcfc94ac8472333fc283d51dc6460e86'
        images=d100.Images();raw=images.raw.reshape(-1,3,32,32);labels=images.labels
        split=json.loads(d100.SPLIT.read_text())['splits'];mean,std=(.507421,.486865,.441380),(.267468,.256482,.275979)
        split_hash=digest(d100.SPLIT)
    train=np.asarray(split['train'],dtype=np.int64);dev=np.asarray(split['dev'],dtype=np.int64)
    assert len(train) in (45000,40000) and len(dev)==5000
    assert not set(train.tolist()).intersection(dev.tolist())
    assert raw.shape==(50000,3,32,32) and len(labels)==50000
    return raw,labels,train,dev,mean,std,split_hash


def subset(indices,labels,per_class,seed):
    rng=np.random.default_rng(seed);chosen=[]
    for label in range(int(labels.max())+1):
        pool=indices[labels[indices]==label]
        chosen.extend(int(x) for x in rng.permutation(pool)[:per_class])
    return np.asarray(sorted(chosen),dtype=np.int64)


class Cifar(Dataset):
    def __init__(self,raw,labels,indices,mean,std,train):
        self.raw=raw;self.labels=labels;self.indices=np.asarray(indices,dtype=np.int64)
        ops=[T.RandomCrop(32,padding=4),T.RandomHorizontalFlip()] if train else []
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize(mean,std)])
    def __len__(self): return len(self.indices)
    def __getitem__(self,position):
        index=int(self.indices[position])
        image=Image.fromarray(np.transpose(self.raw[index],(1,2,0)))
        return self.transform(image),int(self.labels[index])


def make_loader(data,batch_size,workers,shuffle,seed):
    generator=torch.Generator();generator.manual_seed(seed)
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=True,generator=generator,
                      worker_init_fn=lambda wid: np.random.seed((seed+wid)%2**32))


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();correct=0;total=0;loss=0.
    for x,y in loader:
        x=x.to(device);y=y.to(device);z=model(x)
        loss+=float(F.cross_entropy(z,y,reduction='sum'))
        correct+=int((z.argmax(1)==y).sum());total+=len(y)
    return dict(top1=correct/total,nll=loss/total,correct=correct,count=total)


def build(num_classes,device):
    model=PreActResNet18(n_cls=num_classes,cuda=True)
    model=model.to(device)
    return model


def save_checkpoint(path,model,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=metadata),temporary)
    os.replace(temporary,path)


def train_worker(run,gpu,smoke=False):
    seed=42+gpu
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda',0)
    out=run/f'seed_{seed}';out.mkdir()
    status=dict(status='running',seed=seed,gpu=gpu,pid=os.getpid(),started_utc=utc(),completed=[])
    write(out/'status.json',status)
    config=read(run/'config.json')
    epochs=2 if smoke else int(config['epochs'])
    try:
        for ds in DATASETS:
            raw,labels,train_idx,dev_idx,mean,std,split_hash=dataset(ds)
            if smoke:
                train_idx=subset(train_idx,labels,100,seed)
                dev_idx=subset(dev_idx,labels,20,seed)
            train_data=Cifar(raw,labels,train_idx,mean,std,True)
            dev_data=Cifar(raw,labels,dev_idx,mean,std,False)
            train_loader=make_loader(train_data,int(config['batch_size']),int(config['num_workers']),True,seed)
            dev_loader=make_loader(dev_data,int(config['batch_size']),int(config['num_workers']),False,seed)
            model=build(10 if ds=='cifar10' else 100,device)
            parameters=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.SGD(model.parameters(),lr=float(config['lr']),
                                      momentum=float(config['momentum']),
                                      weight_decay=float(config['weight_decay']),nesterov=False)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0.)
            target=out/ds;target.mkdir()
            metrics_path=target/'metrics.jsonl';start=time.monotonic()
            for epoch in range(1,epochs+1):
                model.train();train_correct=0;train_total=0;train_loss=0.
                for x,y in train_loader:
                    x=x.to(device);y=y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    z=model(x);loss=F.cross_entropy(z,y)
                    loss.backward();optimizer.step()
                    train_loss+=float(loss.detach())*len(y);train_correct+=int((z.argmax(1)==y).sum());train_total+=len(y)
                dev=evaluate(model,dev_loader,device)
                metadata=dict(seed=seed,dataset=ds,epoch=epoch,epochs=epochs,
                              experiment='2026-11-05_E034_preactresnet18',architecture='PreActResNet18',
                              split_sha256=split_hash,config_sha256=digest(run/'config.json'),
                              parameters=parameters,smoke=smoke)
                save_checkpoint(target/'last.pt',model,metadata)
                row=dict(epoch=epoch,epochs=epochs,seed=seed,dataset=ds,
                         train_loss=train_loss/train_total,train_top1=train_correct/train_total,
                         dev_top1=dev['top1'],dev_nll=dev['nll'],lr=optimizer.param_groups[0]['lr'],
                         elapsed_seconds=time.monotonic()-start,checkpoint_sha256=digest(target/'last.pt'),
                         official_test_used=False,reserve_used=False)
                with metrics_path.open('a') as f: f.write(json.dumps(row)+'\n')
                print(seed,ds,epoch,row,flush=True)
                scheduler.step()
            summary=dict(status='completed',seed=seed,gpu=gpu,dataset=ds,epochs=epochs,smoke=smoke,
                         architecture='PreActResNet18',parameters=parameters,
                         final_metrics=dict(dev_top1=dev['top1'],dev_loss=dev['nll'],
                                            train_top1=train_correct/train_total,train_loss=train_loss/train_total,
                                            epoch=epochs,elapsed_seconds=time.monotonic()-start),
                         checkpoint=str(target/'last.pt'),checkpoint_sha256=digest(target/'last.pt'),
                         split_sha256=split_hash,config_sha256=digest(run/'config.json'),
                         official_test_used=False,reserve_used=False)
            write(target/'summary.json',summary)
            status['completed'].append(ds);write(out/'status.json',status)
        status.update(status='completed',finished_utc=utc())
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        write(out/'status.json',status)


def check():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(42)
    for ds in DATASETS:
        raw,labels,train_idx,dev_idx,mean,std,split_hash=dataset(ds)
        assert len(train_idx)>0 and len(dev_idx)==5000
        num_classes=10 if ds=='cifar10' else 100
        model=PreActResNet18(n_cls=num_classes,cuda=False).eval()
        x=torch.randn(2,3,32,32)
        with torch.inference_mode(): z=model(x)
        assert z.shape==(2,num_classes) and torch.isfinite(z).all()
        clone=PreActResNet18(n_cls=num_classes,cuda=False)
        clone.load_state_dict(model.state_dict(),strict=True)
        assert sum(p.numel() for p in model.parameters())>0
    write(HERE/'check_train.json',dict(status='passed',utc=utc(),inputs=inputs(),
          source_hashes=source_hashes(),protocol_sha256=digest(HERE/'OPIS.md'),
          architecture='PreActResNet18',commit=COMMIT,datasets=list(DATASETS),
          params={ds:sum(p.numel() for p in PreActResNet18(n_cls=10 if ds=='cifar10' else 100,cuda=False).parameters()) for ds in DATASETS}))
    print('CPU PASS',read(HERE/'check_train.json')['params'],flush=True)


def launch(smoke=False):
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    assert len(info.strip().splitlines())==4
    for i,line in enumerate(info.strip().splitlines()):
        r=[x.strip() for x in line.split(',')]
        assert int(r[0])==i and int(r[2])<256 and int(r[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    checked=read(HERE/'check_train.json')
    assert checked['status']=='passed' and checked['inputs']==inputs()
    assert checked['source_hashes']==source_hashes() and checked['protocol_sha256']==digest(HERE/'OPIS.md')
    prefix='smoke' if smoke else 'train'
    run=ROOT/'runs/E034'/f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True,exist_ok=False);(run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.is_file(): shutil.copyfile(p,run/'source'/p.name)
    config=dict(epochs=200,seeds=list(SEEDS),datasets=list(DATASETS),batch_size=128,
                lr=0.1,momentum=.9,weight_decay=5e-4,num_workers=2,smoke=smoke,
                architecture='PreActResNet18',commit=COMMIT,official_test_used=False,reserve_used=False)
    write(run/'config.json',config)
    m=dict(status='running',started_utc=utc(),launcher_pid=os.getpid(),assignments={str(i):42+i for i in range(4)},
           inputs=checked['inputs'],source_hashes=source_hashes(),protocol_sha256=checked['protocol_sha256'],
           config_sha256=digest(run/'config.json'),config=config,gpus=info,
           python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
           smoke=smoke,official_test_used=False,reserve_used=False,
           limits=dict(wall_seconds=600 if smoke else 18000,artifacts_bytes=536870912))
    write(run/'manifest.json',m)
    (run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('RUN='+str(run),flush=True)
    processes=[];start=time.monotonic();limit=600 if smoke else 18000
    try:
        for gpu in range(4):
            log=(run/'logs'/f'gpu{gpu}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cmd=[sys.executable,'-B',str(HERE/'train_preact.py'),'train-worker','--run-dir',str(run),'--gpu',str(gpu)]
            if smoke: cmd.append('--smoke')
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            processes.append((p,log))
        m['worker_pids']=[p.pid for p,_ in processes];write(run/'manifest.json',m)
        codes=[p.wait(timeout=max(.1,limit-(time.monotonic()-start))) for p,_ in processes]
        m['exit_codes']=codes;assert codes==[0]*4
        assert all(read(run/f'seed_{s}/status.json')['status']=='completed' for s in SEEDS)
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


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['check','train','train-worker'])
    ap.add_argument('--run-dir',type=Path);ap.add_argument('--gpu',type=int);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.action=='check': check()
    elif args.action=='train': launch(args.smoke)
    else: train_worker(args.run_dir,args.gpu,args.smoke)
