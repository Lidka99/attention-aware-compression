"""E027: search po disagreement, diagnostyka dev po wyborze; OPIS.md.
Wejście: checkpoint/indeksy; wyjście trace z trzema miarami, selection,
dev każdego kandydata tylko post hoc, referencje E025/E026.
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
        source=bn.source_model(seed)
        source_hash=bn.weights_hash(source)
        geometry=bn.Geometry(source)
        ids=bn.read(args.run_dir/'indices.json')
        images=bn.Images()
        aug=bn.augment_uint8(images.raw,ids['A'])
        assert bn.tensor_hash(aug)==bn.read(args.run_dir/'manifest.json')['augmented_views_sha256']
        a=bn.normalize(aug)
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
        # Dev dopiero teraz, dla każdego odwiedzonego kandydata; nie wpływa na wybór.
        final_started=utc()
        dx,dy=images.dev(ids['dev'])
        def dev_metrics(logits):
            return dict(top1=float((logits.argmax(1)==dy).double().mean()),
                        nll=float(F.cross_entropy(logits,dy)))
        diagnostics=[]
        dev_cache={}
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
        refs={};reference_paths={}
        old=bn.read(bn.SOURCE/f'seed_{seed}/summary.json')
        new26=bn.read(E026_RUN/f'seed_{seed}/summary.json')
        for name,oldname in [('UNIFORM','UNIFORM'),('TILT','TILT'),('OLD_SEARCH','SEARCH')]:
            p=e26.REF/f'seed_{seed}/items/{oldname}_A_aug_CMA_reset.json'
            r=bn.read(p)
            refs[name]=dict(widths=r['widths'],params=r['params'],post_bn=r['metrics'],kl_C=r['kl_C'],
                            pre_bn=old['finals'][oldname]['pre_bn'],reused=True,
                            source=str(p.relative_to(ROOT)),source_sha256=digest(p))
            reference_paths[name]=p
        k26=new26['finals']['NEW_SEARCH']
        refs['E026_SEARCH']=dict(widths=k26['widths'],params=k26['params'],post_bn=k26['post_bn'],kl_C=k26['kl_C'],
                                 pre_bn=k26['pre_bn'],reused=True,
                                 source=str((E026_RUN/f'seed_{seed}/summary.json').relative_to(ROOT)),
                                 source_sha256=digest(E026_RUN/f'seed_{seed}/summary.json'))
        assert refs['UNIFORM']['widths']==uniform and refs['TILT']['widths']==tilt
        match=next((name for name,r in refs.items() if r['widths']==chosen['widths']),None)
        if match:
            result=dict(refs[match],reused_from=match)
        else:
            cx=images.inputs(ids['C'])
            teacher_c=bn.predict(teacher,cx)
            model,diag=bn.fold(source,chosen['widths'])
            model.to('cuda')
            pre=dev_metrics(bn.predict(model,dx))
            meta=bn.calibrate(model,a.split(128),'CMA_reset')
            kl_b=measure_values(teacher_b,bn.predict(model,b))['kl']
            result=dict(widths=chosen['widths'],params=geometry.count(chosen['widths']),pre_bn=pre,
                        post_bn=dev_metrics(bn.predict(model,dx)),kl_B=kl_b,
                        kl_C=measure_values(teacher_c,bn.predict(model,cx))['kl'],bn=meta,reused=False)
            write(folder/'selected_diagnostics.json',diag)
            del model
        teacher_ref=e26.REF/f'seed_{seed}/items/teacher_A_aug_CMA_reset.json'
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
