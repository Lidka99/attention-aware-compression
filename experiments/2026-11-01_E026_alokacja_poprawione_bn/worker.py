"""E026: wyszukiwanie z augmentowanym BN, bez zmiany celu KL; OPIS.md.
Wejście: checkpoint/indeksy; wyjście: trace/selection/finals, reuse E025.
"""
import argparse
import copy
import os
import time
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
from support import *


def utc():
    return datetime.now(timezone.utc).isoformat()


def kl(teacher,candidate):
    return float(F.kl_div(F.log_softmax(candidate,dim=1),F.softmax(teacher,dim=1),reduction='batchmean'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args()
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    seed=42+args.gpu
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=args.run_dir/f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    started=time.monotonic()
    status=dict(status='running',seed=seed,gpu=args.gpu,started_utc=utc(),official_test_used=False,reserve_used=False)
    write(folder/'status.json',status)
    try:
        source=bn.source_model(seed)
        source_hash=bn.weights_hash(source)
        geometry=bn.Geometry(source)
        ids=bn.read(args.run_dir/'indices.json')
        images=bn.Images()
        aug=bn.augment_uint8(images.raw,ids['A'])
        assert bn.tensor_hash(aug)==bn.read(args.run_dir/'manifest.json')['augmented_views_sha256']
        # B bez augmentacji i bez etykiet. C/dev nie są dostępne dla optimize.
        a=bn.normalize(aug);b=images.inputs(ids['B'])
        teacher=copy.deepcopy(source).to('cuda')
        teacher_b=bn.predict(teacher,b)
        uniform=geometry.profile(round(.5*geometry.count(geometry.sizes)),0)
        target=geometry.count(uniform)
        tilt=geometry.profile(target,.30)
        def measure(widths):
            begin=time.monotonic()
            model,_=bn.fold(source,widths)
            model.to('cuda')
            meta=bn.calibrate(model,a.split(128),'CMA_reset')
            assert meta['samples']==4096 and meta['batches']==32 and meta['params_unchanged']
            score=kl(teacher_b,bn.predict(model,b))
            assert np.isfinite(score)
            del model
            return dict(kl_B=score,seconds=time.monotonic()-begin,params_unchanged=True,calibration_samples=4096)
        def progress(trace):
            write(folder/'trace.json',trace)
            print(seed,trace[-1]['index'],trace[-1]['origin'],trace[-1]['kl_B'],flush=True)
        chosen,trajectory,trace=optimize(geometry,target,uniform,tilt,measure,progress)
        assert bn.weights_hash(source)==source_hash
        selection=dict(widths=chosen['widths'],index=chosen['index'],kl_B=chosen['kl_B'],trajectory=trajectory,
                       uniform=uniform,tilt=tilt,target=target,groups=geometry.names,selected_utc=utc(),evaluations=len(trace))
        write(folder/'selection.json',selection)
        selection_hash=digest(folder/'selection.json')
        # Dopiero tutaj odczyt historycznych metryk porównawczych oraz C/dev.
        final_started=utc()
        old=bn.read(bn.SOURCE/f'seed_{seed}/summary.json')
        refs={};reference_paths={}
        for name,oldname in [('UNIFORM','UNIFORM'),('TILT','TILT'),('OLD_SEARCH','SEARCH')]:
            p=REF/f'seed_{seed}/items/{oldname}_A_aug_CMA_reset.json'
            r=bn.read(p)
            assert r['status']=='completed' and r['bn']['params_unchanged']
            refs[name]=dict(widths=r['widths'],params=r['params'],post_bn=r['metrics'],kl_C=r['kl_C'],
                            pre_bn=old['finals'][oldname]['pre_bn'],reused=True,source=str(p.relative_to(ROOT)),source_sha256=digest(p))
            reference_paths[name]=p
        assert refs['UNIFORM']['widths']==uniform and refs['TILT']['widths']==tilt
        match=next((name for name,r in refs.items() if r['widths']==chosen['widths']),None)
        if match:
            result=dict(refs[match],kl_B=chosen['kl_B'],reused_from=match)
        else:
            cx=images.inputs(ids['C'])
            dx,dy=images.dev(ids['dev'])
            teacher_c=bn.predict(teacher,cx)
            def metrics(logits):
                return dict(top1=float((logits.argmax(1)==dy).double().mean()),nll=float(F.cross_entropy(logits,dy)))
            model,diag=bn.fold(source,chosen['widths'])
            model.to('cuda')
            pre=metrics(bn.predict(model,dx))
            meta=bn.calibrate(model,a.split(128),'CMA_reset')
            kl_b=kl(teacher_b,bn.predict(model,b))
            assert abs(kl_b-chosen['kl_B'])<1e-5
            result=dict(widths=chosen['widths'],params=geometry.count(chosen['widths']),pre_bn=pre,
                        post_bn=metrics(bn.predict(model,dx)),kl_B=kl_b,kl_C=kl(teacher_c,bn.predict(model,cx)),
                        bn=meta,reused=False)
            write(folder/'selected_diagnostics.json',diag)
            del model
        teacher_ref=REF/f'seed_{seed}/items/teacher_A_aug_CMA_reset.json'
        controls=dict(teacher=old['controls']['teacher'],teacher_bn=bn.read(teacher_ref)['metrics'],
                      source=str(teacher_ref.relative_to(ROOT)),source_sha256=digest(teacher_ref))
        assert bn.weights_hash(source)==source_hash
        status.update(status='completed',finished_utc=utc(),runtime_seconds=time.monotonic()-started,
                      controls=controls,finals=dict(refs,NEW_SEARCH=result),evaluations=len(trace),
                      selection_sha256=selection_hash,final_evaluation_started_utc=final_started,
                      torch=torch.__version__,cuda=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),
                      batch_size_per_gpu=128,effective_batch_per_run=128)
        write(folder/'summary.json',status)
    except Exception as e:
        status.update(status='failed',error=str(e),traceback=traceback.format_exc(),finished_utc=utc())
        raise
    finally:
        write(folder/'status.json',status)

if __name__=='__main__':
    main()
