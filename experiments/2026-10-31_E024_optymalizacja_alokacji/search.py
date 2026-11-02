"""E024: wyszukiwanie szerokości pełnej sieci wg KL na train; OPIS.md.
Wejście: seed, rozłączne A/B/C, checkpoint; wyjście: trace, wybór i finalne
metryki. Dev oceniany dopiero po zapisaniu wybranego profilu.
"""
import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data import HERE,Images,CHECKPOINT_ROOT,CHECKPOINT_HASHES,BASELINE,digest,write
from folding import Geometry,architecture,fold


def utc():
    return datetime.now(timezone.utc).isoformat()


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def predict(model,x):
    model.eval()
    return torch.cat([model(batch.to('cuda')).cpu() for batch in x.split(128)])


@torch.inference_mode()
def calibrate(model,x):
    before=parameter_digest(model)
    model.eval()
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum=None
            module.train()
    for batch in x.split(128):
        model(batch.to('cuda'))
    model.eval()
    assert before==parameter_digest(model)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def divergence(teacher,candidate):
    return float(F.kl_div(F.log_softmax(candidate,dim=1),F.softmax(teacher,dim=1),reduction='batchmean'))


def worker(args):
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu) and torch.cuda.device_count()==1
    assert args.seed==42+args.gpu
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    folder=args.run_dir/f'seed_{args.seed}'
    folder.mkdir(exist_ok=False)
    started=time.monotonic()
    state=dict(status='running',seed=args.seed,gpu=args.gpu,started_utc=utc(),official_test_used=False,reserve_used=False)
    write(folder/'status.json',state)
    try:
        checkpoint=CHECKPOINT_ROOT/f'seed_{args.seed}/last.pt'
        assert digest(checkpoint)==CHECKPOINT_HASHES[args.seed]
        source=architecture()
        source.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
        frozen_source=parameter_digest(source)
        geo=Geometry(source)
        parts=json.loads((args.run_dir/'indices.json').read_text())
        images=Images()
        # Etykiety nie są przekazywane do search ani kalibracji.
        a,b=images.inputs(parts['A']),images.inputs(parts['B'])
        teacher=copy.deepcopy(source).to('cuda')
        teacher_b=predict(teacher,b)
        uniform=geo.profile(round(.5*geo.count(geo.sizes)),0)
        target=geo.count(uniform)
        tilt=geo.profile(target,.30)
        cache={}
        trace=[]
        def evaluate(widths,origin):
            key=tuple(widths)
            if key in cache:
                return cache[key]
            assert .999*target<=geo.count(widths)<=target
            begin=time.monotonic()
            model,_=fold(source,widths)
            model.to('cuda')
            calibrate(model,a)
            kl=divergence(teacher_b,predict(model,b))
            assert np.isfinite(kl)
            entry=dict(index=len(trace),origin=origin,widths=widths,params=geo.count(widths),kl_B=kl,seconds=time.monotonic()-begin)
            cache[key]=entry
            trace.append(entry)
            write(folder/'trace.json',trace)
            print(args.seed,entry['index'],origin,'KL',kl,flush=True)
            del model
            return entry
        u=evaluate(uniform,'UNIFORM')
        t=evaluate(tilt,'TILT')
        incumbent=min((u,t),key=lambda r:r['kl_B'])
        trajectory=[incumbent['index']]
        for round_index in range(2):
            candidates=[]
            for widths in geo.proposals(incumbent['widths'],target,round_index):
                if tuple(widths) in cache:
                    continue
                candidates.append(evaluate(widths,f'round{round_index+1}'))
                if len(candidates)==24:
                    break
            best=min([incumbent]+candidates,key=lambda r:r['kl_B'])
            if incumbent['kl_B']-best['kl_B']<=1e-6:
                break
            incumbent=best
            trajectory.append(incumbent['index'])
        assert len(trace)<=50
        assert parameter_digest(source)==frozen_source
        selection=dict(widths=incumbent['widths'],kl_B=incumbent['kl_B'],index=incumbent['index'],trajectory=trajectory,
                       uniform=uniform,tilt=tilt,target=target,groups=geo.names,selected_utc=utc(),evaluations=len(trace))
        write(folder/'selection.json',selection)
        selection_sha=digest(folder/'selection.json')
        # C i dev pojawiają się dopiero po zamrożeniu wyboru.
        final_started=utc()
        cx=images.inputs(parts['C'])
        dx,dy=images.dev(parts['dev'])
        teacher_c=predict(teacher,cx)
        teacher_dev=predict(teacher,dx)
        def metrics(z):
            return dict(top1=float((z.argmax(1)==dy).double().mean()),nll=float(F.cross_entropy(z,dy)))
        baseline=metrics(teacher_dev)
        assert abs(baseline['top1']-BASELINE[args.seed])<1e-12
        baseline_bn=copy.deepcopy(source).to('cuda')
        calibrate(baseline_bn,a)
        controls={'teacher':baseline,'teacher_bn':metrics(predict(baseline_bn,dx))}
        del baseline_bn
        finals={}
        final_cache={}
        for name,widths in [('UNIFORM',uniform),('TILT',tilt),('SEARCH',incumbent['widths'])]:
            key=tuple(widths)
            if key not in final_cache:
                model,diag=fold(source,widths)
                model.to('cuda')
                pre=metrics(predict(model,dx))
                calibrate(model,a)
                kl_b=divergence(teacher_b,predict(model,b))
                assert abs(kl_b-cache[key]['kl_B'])<1e-5
                final_cache[key]=dict(widths=widths,params=geo.count(widths),pre_bn=pre,post_bn=metrics(predict(model,dx)),
                                      kl_B=kl_b,kl_C=divergence(teacher_c,predict(model,cx)),params_unchanged=True)
                write(folder/f'{name}_diagnostics.json',diag)
                del model
            finals[name]=final_cache[key]
        assert parameter_digest(source)==frozen_source
        state.update(status='completed',finished_utc=utc(),runtime_seconds=time.monotonic()-started,controls=controls,finals=finals,
                     selection_sha256=selection_sha,final_evaluation_started_utc=final_started,evaluations=len(trace),
                     checkpoint_sha256=CHECKPOINT_HASHES[args.seed],torch=torch.__version__,gpu_name=torch.cuda.get_device_name(0),
                     cuda=torch.version.cuda,batch_size_per_gpu=128,effective_batch_per_run=128)
        write(folder/'summary.json',state)
    except Exception as e:
        state.update(status='failed',error=str(e),traceback=traceback.format_exc(),finished_utc=utc())
        raise
    finally:
        write(folder/'status.json',state)


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True)
    worker(p.parse_args())
