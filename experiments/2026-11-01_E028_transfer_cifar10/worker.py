"""E028: transfer reguły E027 na CIFAR-10; OPIS.md.
Wejście: checkpoint/indeksy; wyjście trace/selekcja/finalne metryki.
Wybór wyłącznie po disagreement na B; dev dopiero po hashu selekcji.
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
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status=dict(status='running',seed=seed,gpu=args.gpu,started_utc=utc(),official_test_used=False,reserve_used=False)
    write(folder/'status.json',status)
    try:
        source=source_model(seed)
        source_hash=bn.weights_hash(source)
        geometry=bn.Geometry(source)
        ids=bn.read(args.run_dir/'indices.json')
        images=Images()
        aug=augment_uint8(images.raw,ids['A'])
        assert tensor_hash(aug)==bn.read(args.run_dir/'manifest.json')['augmented_views_sha256']
        a=normalize(aug)
        b=images.inputs(ids['B'])
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
            values=measure_values(teacher_b,bn.predict(model,b))
            del model
            return dict(values,seconds=time.monotonic()-begin,params_unchanged=True,calibration_samples=4096)
        def progress(trace):
            write(folder/'trace.json',trace)
            print(seed,trace[-1]['index'],trace[-1]['origin'],trace[-1]['disagreement'],flush=True)
        chosen,trajectory,trace=optimize(geometry,target,uniform,tilt,measure,'disagreement',progress)
        assert bn.weights_hash(source)==source_hash
        selection=dict(widths=chosen['widths'],index=chosen['index'],disagreement=chosen['disagreement'],
                       kl=chosen['kl'],teacher_ce=chosen['teacher_ce'],trajectory=trajectory,uniform=uniform,tilt=tilt,
                       target=target,groups=geometry.names,selected_utc=utc(),evaluations=len(trace),
                       selection_rule='min disagreement on B')
        write(folder/'selection.json',selection)
        selection_hash=digest(folder/'selection.json')
        final_started=utc()
        dx,dy=images.dev(ids['dev'])
        def dev_metrics(logits):
            return dict(top1=float((logits.argmax(1)==dy).double().mean()),
                        nll=float(F.cross_entropy(logits,dy)))
        diagnostics=[];dev_cache={}
        for entry in trace:
            key=tuple(entry['widths'])
            if key not in dev_cache:
                model,_=bn.fold(source,key)
                model.to('cuda')
                bn.calibrate(model,a.split(128),'CMA_reset')
                dev_cache[key]=dev_metrics(bn.predict(model,dx))
                del model
            diagnostics.append(dict(index=entry['index'],origin=entry['origin'],widths=entry['widths'],
                                    disagreement=entry['disagreement'],kl=entry['kl'],teacher_ce=entry['teacher_ce'],
                                    dev=dev_cache[key]))
        write(folder/'dev_diagnostics.json',diagnostics)
        raw_dev=dev_metrics(bn.predict(teacher,dx))
        assert abs(raw_dev['top1']-BASELINE[seed])<1e-12
        teacher_bn=copy.deepcopy(source).to('cuda')
        bn.calibrate(teacher_bn,a.split(128),'CMA_reset')
        teacher_bn_dev=dev_metrics(bn.predict(teacher_bn,dx))
        del teacher_bn
        cx=images.inputs(ids['C'])
        teacher_c=bn.predict(teacher,cx)
        def c_kl(widths):
            model,_=bn.fold(source,widths)
            model.to('cuda')
            bn.calibrate(model,a.split(128),'CMA_reset')
            score=measure_values(teacher_c,bn.predict(model,cx))['kl']
            del model
            return score
        finals={}
        for name,widths in (('UNIFORM',uniform),('TILT',tilt),('NEW_SEARCH',chosen['widths'])):
            dev=next(d['dev'] for d in diagnostics if d['widths']==widths)
            finals[name]=dict(widths=widths,params=geometry.count(widths),post_bn=dev,
                              kl_C=c_kl(widths) if name!='NEW_SEARCH' else None)
        if finals['NEW_SEARCH']['kl_C'] is None:
            finals['NEW_SEARCH']['kl_C']=c_kl(chosen['widths'])
        controls=dict(teacher=raw_dev,teacher_bn=teacher_bn_dev)
        assert bn.weights_hash(source)==source_hash
        status.update(status='completed',finished_utc=utc(),runtime_seconds=time.monotonic()-started,
                      controls=controls,finals=finals,evaluations=len(trace),selection_sha256=selection_hash,
                      final_evaluation_started_utc=final_started,torch=torch.__version__,cuda=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(0),batch_size_per_gpu=128,effective_batch_per_run=128)
        write(folder/'summary.json',status)
    except Exception as e:
        status.update(status='failed',error=str(e),traceback=traceback.format_exc(),finished_utc=utc())
        raise
    finally:
        write(folder/'status.json',status)

if __name__=='__main__':
    main()
