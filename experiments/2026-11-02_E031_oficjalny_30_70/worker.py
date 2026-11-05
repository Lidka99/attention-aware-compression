"""E031: jeden seed/GPU, trzy zbiory, dwa budżety,7 ramion×3 stany; OPIS.md.
Wyjście: metryki, zgodność logitów/wag/partycji, diagnostyki i status.
"""
import argparse
import copy
import os
import time
import traceback
from datetime import datetime,timezone
from sklearn.metrics import adjusted_rand_score
from torch.nn import functional as F
from common import *


def utc():
    return datetime.now(timezone.utc).isoformat()


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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args();seed=42+args.gpu
    assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    out=args.run_dir/f'seed_{seed}';out.mkdir(exist_ok=True)
    parts=read(args.run_dir/'indices.json');plans=read(args.run_dir/'plans.json')
    start=time.monotonic()
    status=dict(status='running',seed=seed,gpu=args.gpu,started_utc=utc(),completed=[])
    write(out/'status.json',status)
    try:
        for ds in DATASETS:
            done=[f'{ds}/{TAGS[b]}' for b in BUDGETS if (out/ds/TAGS[b]/'summary.json').exists()]
            if len(done)==len(BUDGETS):
                status['completed'].extend(done)
                write(out/'status.json',status)
                continue
            a,dx,dy,views_hash=data(ds,parts[ds])
            tv,up=load_source(ds,seed)
            controls_path=out/f'controls_{ds}.json'
            if controls_path.exists():
                stored=read(controls_path)
                assert stored['views_sha256']==views_hash
                architecture_error=stored['architecture_max_abs']
            else:
                controls,architecture_error=teacher_controls(ds,seed,tv,up,a,dx,dy)
                write(controls_path,dict(status='completed',dataset=ds,seed=seed,views_sha256=views_hash,
                                         architecture_max_abs=architecture_error,controls=controls))
            for budget in BUDGETS:
                tag=TAGS[budget]
                if (out/ds/tag/'summary.json').exists():
                    status['completed'].append(f'{ds}/{tag}')
                    continue
                models,plan=construct(ds,seed,budget,tv,up)
                assert plan==plans[ds][tag],(ds,tag)
                (out/ds/tag).mkdir(parents=True,exist_ok=True)
                metrics_by={};logits_by={};calibrations={}
                for name in VARIANTS:
                    base,diag=models[name]
                    write(out/ds/tag/f'{name}_clusters.json',diag)
                    logits_by[name]={};metrics_by[name]={};calibrations[name]={}
                    for mode in MODES:
                        model=copy.deepcopy(base).to('cuda')
                        if mode!='PRE':
                            calibrations[name][mode]=e29.calibrate(model,a,mode)
                        logits=bn.predict(model,dx)
                        metrics_by[name][mode]=dict(
                            top1=float((logits.argmax(1)==dy).double().mean()),
                            nll=float(F.cross_entropy(logits,dy)))
                        logits_by[name][mode]=logits
                        del model
                    print(seed,ds,tag,name,metrics_by[name],flush=True)
                pairs={}
                for name,reference in [('UP_RAW_U','UP_FIXED_U'),('LEGACY_U','UP_FIXED_U'),
                                       ('LOCAL_U','UP_FIXED_U'),('LEGACY_T','UP_FIXED_T'),
                                       ('LOCAL_T','UP_FIXED_T')]:
                    left,ld=models[name];right,rd=models[reference]
                    compare=dict(reference=reference,
                                 parameters=tensor_diffs(dict(left.named_parameters()),dict(right.named_parameters())),
                                 state=tensor_diffs(left.state_dict(),right.state_dict()),
                                 cluster_ari=[float(adjusted_rand_score(l['labels'],r['labels'])) for l,r in zip(ld,rd)],
                                 labels_exact=[l['labels']==r['labels'] for l,r in zip(ld,rd)],logits={})
                    for mode in MODES:
                        x=logits_by[name][mode];y=logits_by[reference][mode]
                        compare['logits'][mode]=dict(
                            max_abs=float((x-y).abs().max()),
                            rmse=float((x-y).square().mean().sqrt()),
                            agreement=float((x.argmax(1)==y.argmax(1)).double().mean()),
                            top1_delta_pp=100*(metrics_by[name][mode]['top1']-metrics_by[reference][mode]['top1']))
                    pairs[name]=compare
                result=dict(status='completed',dataset=ds,budget=budget,tag=tag,seed=seed,gpu=args.gpu,
                            views_sha256=views_hash,plan=plan,architecture_max_abs=architecture_error,
                            controls_sha256=digest(controls_path),metrics=metrics_by,comparisons=pairs,
                            calibrations=calibrations,
                            params={k:sum(p.numel() for p in m.parameters()) for k,(m,_) in models.items()},
                            official_test_used=False,reserve_used=False)
                write(out/ds/tag/'summary.json',result)
                status['completed'].append(f'{ds}/{tag}');write(out/'status.json',status)
                del models,logits_by,metrics_by,calibrations
            del tv,up
        status.update(status='completed',finished_utc=utc(),runtime_seconds=time.monotonic()-start)
    except Exception as e:
        status.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:
        write(out/'status.json',status)


if __name__=='__main__':
    main()
