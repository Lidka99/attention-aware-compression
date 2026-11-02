"""E023: solver łagodnych budżetów i 48 ocen istniejących wag, OPIS.md.
Wejście: inventory E020, train/dev E011/E015/E019; wyjście runs/E023.
Bez treningu; cztery GPU po jednym seedzie, profile przygotowane na CPU.
"""
import argparse
import copy
import json
import os
import secrets
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/2026-10-30_E020_mechanizm_alokacji'))
import e020_common as c
c.EXPERIMENT=HERE.name
c.OPIS_PATH=str((HERE/'OPIS.md').relative_to(ROOT))


def new_run(kind):
    p=ROOT/'runs/E023'/f'{kind}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    p.mkdir(parents=True,exist_ok=False)
    (p/'logs').mkdir()
    return p


def solve(model,target,tilt):
    fb=c.folding_budget()
    def evaluate(base):
        ratios=np.clip(base+tilt*np.linspace(-1,1,12),0,.9).tolist()
        widths=fb._widths_from_ratios(model,ratios)
        return dict(base=base,ratios=ratios,widths=widths,params=fb._parameter_count_from_widths(model,widths))
    low,high=-abs(tilt),.9+abs(tilt)
    assert evaluate(low)['params']>=target and evaluate(high)['params']<=target
    for _ in range(60):
        mid=(low+high)/2
        if evaluate(mid)['params']>target:
            low=mid
        else:
            high=mid
    candidates=[evaluate(x) for x in (high,np.nextafter(high,np.inf),low)]
    result=max((x for x in candidates if x['params']<=target),key=lambda x:x['params'])
    # Rzeczywiste ratios z szerokości, unikają błędu zaokrągleń float.
    result['raw_ratios']=result['ratios']
    result['ratios']=[1-w/n for w,n in zip(result['widths'],fb._n_channels_by_group(model))]
    assert fb.parameter_count_for_ratios(model,result['ratios'])==result['params']
    return result


def prepare():
    assert not torch.cuda.is_available(), 'prepare wymaga CUDA_VISIBLE_DEVICES pustego'
    inv=c.load_inventory()
    plans={}
    inputs={}
    for ds,meta in inv['datasets'].items():
        assert c.digest(meta['split'])==meta['split_sha256']
        inputs[meta['split']]=meta['split_sha256']
        model,_=c.load_checkpoint(ds,42,inv)
        fb=c.folding_budget()
        n=c.count_parameters(model)
        for seed,entry in meta['seeds'].items():
            assert c.digest(entry['checkpoint'])==entry['checkpoint_sha256']
            inputs[entry['checkpoint']]=entry['checkpoint_sha256']
        plans[ds]={}
        for budget in (10,20):
            nominal=round(n*(1-budget/100))
            u=solve(model,nominal,0)
            assert -.5<=100*(u['params']/nominal-1)<=0
            t=solve(model,u['params'],.30)
            repaired=fb.repair_budget(model,t['ratios'],u['params'],min_ratio=0,max_ratio=.9)
            assert -.1<=100*(repaired['params']/u['params']-1)<=0
            assert fb.parameter_count_for_ratios(model,repaired['ratios'])==repaired['params']
            repaired.update(base=t['base'],solver_params=t['params'],solver_ratios=t['ratios'],
                            max_ratio_repair_change=max(abs(x-y) for x,y in zip(t['ratios'],repaired['ratios'])))
            plans[ds][str(budget)]={'uniform':u,'tilt0_p030':repaired,'nominal_target':nominal,'original_params':n}
            print(ds,budget,'U',u['params'],'T',repaired['params'],'T base',t['base'],'nietknięte',sum(r==0 for r in repaired['ratios']),flush=True)
    p=new_run('prepare')
    c.write_json_atomic(p/'plans.json',plans)
    c.write_json_atomic(p/'manifest.json',dict(status='completed',kind='CPU_geometry_only',protocol_sha256=c.digest(HERE/'OPIS.md'),
                        code_sha256=c.digest(Path(__file__)),inventory_sha256=c.digest(c.INVENTORY_PATH),inputs=inputs))
    print(f'PREPARED={p}',flush=True)


def run(prepared):
    info=c.preflight(4,min_disk_gib=1)
    pm=c.read_json(prepared/'manifest.json')
    assert pm['code_sha256']==c.digest(Path(__file__)) and pm['protocol_sha256']==c.digest(HERE/'OPIS.md')
    assert pm['inventory_sha256']==c.digest(c.INVENTORY_PATH)
    plans=c.read_json(prepared/'plans.json')
    p=new_run('fold')
    cfg=dict(status='frozen',gpu_count=4,batch_size=128,num_workers=2,bn_seed=20261024,fold_seed=42,plans=plans)
    c.write_json_atomic(p/'config.json',cfg)
    items=[dict(name=f'{ds}_seed{seed}_b{budget}_{v}',dataset=ds,seed=seed,budget=budget,variant=v)
           for ds in plans for budget in (10,20) for v in ('uniform','tilt0_p030') for seed in range(42,46)]
    assignments=c.shard_items(items,4)
    assert all({i['seed'] for i in shard}=={42+gpu} for gpu,shard in assignments.items())
    m=c.write_manifest(p,cfg,items,{str(k):[i['name'] for i in v] for k,v in assignments.items()})
    m.update(protocol_sha256=c.digest(HERE/'OPIS.md'),code_sha256=c.digest(Path(__file__)),prepared=str(prepared),
             prepared_manifest_sha256=c.digest(prepared/'manifest.json'),plans_sha256=c.digest(prepared/'plans.json'),launcher_pid=os.getpid())
    c.write_json_atomic(p/'manifest.json',m)
    (p/'source').mkdir()
    for src in (HERE/'OPIS.md',Path(__file__),c.INVENTORY_PATH):
        (p/'source'/src.name).write_bytes(src.read_bytes())
    c.write_run_environment(p,info)
    print(f'RUN={p}',flush=True)
    codes=c.launch_shards(p,HERE/'worker.py',p/'config.json',items,4,5400)
    summaries=list((p/'items').glob('*/summary.json'))
    completed=sum(c.read_json(x)['status']=='completed' for x in summaries)
    ok=completed==48 and all(v==0 for v in codes.values())
    c.finish_manifest(p,'completed' if ok else 'failed',dict(completed=completed,total=48,exit_codes=codes))
    print(f'completed={completed}/48; exit={codes}',flush=True)
    if not ok:
        raise SystemExit(1)


def worker():
    ap=argparse.ArgumentParser()
    for a in ('run-dir','config','items'):
        ap.add_argument('--'+a,type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args()
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    cfg=c.read_json(args.config)
    items=c.read_json(args.items)
    cache={}
    for item in items:
        output=args.run_dir/'items'/item['name']
        output.mkdir(parents=True,exist_ok=False)
        s=dict(item=item,status='running',gpu=args.gpu,started_utc=c.utc_now(),official_test_used=False,reserve_used=False)
        started=time.monotonic()
        try:
            ds,seed=item['dataset'],item['seed']
            if ds not in cache:
                # Workery przetwarzają zbiory kolejno; nie kumulują cache danych.
                cache.clear()
                module,images,labels,classes,split=c.load_dataset(ds)
                source,entry=c.load_checkpoint(ds,seed)
                dev,bn,indices=c.make_loaders(module,images,labels,split,128,2,seed,cfg['bn_seed'])
                bm=copy.deepcopy(source).to(device)
                base,_,_,_=c.evaluate(bm,dev,device)
                del bm,bn
                assert abs(base['top1']-entry['baseline_top1'])<1e-12
                assert abs(base['nll']-entry['baseline_nll'])<1e-5
                cache[ds]=(module,images,labels,source,entry,dev,indices,base)
            module,images,labels,source,entry,dev,indices,base=cache[ds]
            plan=cfg['plans'][ds][str(item['budget'])]
            allocation=plan[item['variant']]
            folded,diag=c.fold_with_ratios(source,allocation['ratios'],cfg['fold_seed'])
            assert c.count_parameters(folded)==allocation['params']
            model=folded.to(device)
            pre,_,_,_=c.evaluate(model,dev,device)
            module.seed_everything(cfg['bn_seed'])
            bn=c.e012_evaluate().make_bn_loader(module,images,labels,indices,cfg)
            bn_stats=c.adapt_bn(model,bn,device,len(indices),cfg['bn_seed'])
            assert bn_stats['params_unchanged'] and not bn_stats['params_have_grad']
            post,_,_,_=c.evaluate(model,dev,device)
            c.write_json_atomic(output/'diagnostics.json',diag)
            s.update(status='completed',baseline=base,pre_bn=pre,post_bn=post,bn=bn_stats,allocation=allocation,
                     original_params=plan['original_params'],uniform_target=plan['uniform']['params'],
                     checkpoint_sha256=entry['checkpoint_sha256'],environment=c.environment(),
                     batch_size_per_gpu=128,effective_batch_per_run=128)
            del model,folded,bn
        except Exception as exc:
            s.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            raise
        finally:
            s.update(finished_utc=c.utc_now(),runtime_seconds=time.monotonic()-started)
            c.write_json_atomic(output/'summary.json',s)
            print(item['name'],s['status'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--prepared',type=Path)
    args=ap.parse_args()
    if args.command=='prepare':
        prepare()
    else:
        assert args.prepared is not None
        run(args.prepared.resolve())
