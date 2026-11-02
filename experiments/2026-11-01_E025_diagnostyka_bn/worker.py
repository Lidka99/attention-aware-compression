"""E025: jeden seed/GPU,11 nowych BN×4 stałe modele; OPIS.md.
Wejście: profile E024 i manifest; wyjście:44 nowe wyniki+4 źródłowe.
"""
import argparse
import copy
import os
import time
import traceback
from datetime import datetime,timezone
from torch.nn import functional as F
from common import *


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu) and torch.cuda.device_count()==1
    seed=42+args.gpu
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    folder=args.run_dir/f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    started=time.monotonic()
    state=dict(status='running',seed=seed,gpu=args.gpu,started_utc=utc(),new_completed=0,reused=0,
               official_test_used=False,reserve_used=False)
    write(folder/'status.json',state)
    try:
        ids=read(args.run_dir/'indices.json')
        source=source_model(seed)
        source_hash=weights_hash(source)
        geo=Geometry(source)
        old=read(SOURCE/f'seed_{seed}/summary.json')
        selection=read(SOURCE/f'seed_{seed}/selection.json')
        profiles=dict(teacher=geo.sizes,UNIFORM=selection['uniform'],TILT=selection['tilt'],SEARCH=selection['widths'])
        assert len({tuple(v) for v in profiles.values()})==4
        images=Images()
        views=Views(images,ids['L'])
        write(folder/'view_hashes.json',dict(L=views.hashes,A={k:tensor_hash(getattr(views,k)[:4096]) for k in ('plain','aug')}))
        cx=images.inputs(ids['C'])
        dx,dy=images.dev(ids['dev'])
        teacher=copy.deepcopy(source).to('cuda')
        teacher_c=predict(teacher,cx)
        del teacher
        def metrics(logits):
            return dict(top1=float((logits.argmax(1)==dy).double().mean()),nll=float(F.cross_entropy(logits,dy)))
        for name,widths in profiles.items():
            if name=='teacher':
                untouched=copy.deepcopy(source)
                baseline=old['controls']['teacher']
                reused_metrics=old['controls']['teacher_bn']
                reused_kl=None
            else:
                untouched,diag=fold(source,widths)
                assert diag==read(SOURCE/f'seed_{seed}/{name}_diagnostics.json'), 'Niezgodność operatora/klastrów ze źródłem'
                baseline=old['finals'][name]['pre_bn']
                reused_metrics=old['finals'][name]['post_bn']
                reused_kl=old['finals'][name]['kl_C']
            params=geo.count(widths)
            if name!='teacher':
                assert params==old['finals'][name]['params']
            write(folder/f'{name}_baseline.json',dict(model=name,seed=seed,metrics=baseline,source=str(SOURCE/f'seed_{seed}/summary.json')))
            pristine_hash=weights_hash(untouched)
            for cell in cells():
                item=dict(status='completed',seed=seed,gpu=args.gpu,model=name,cell=cell,params=params,widths=widths,
                          official_test_used=False,reserve_used=False)
                if cell['name']==REUSED:
                    item.update(reused=True,metrics=reused_metrics,kl_C=reused_kl,source=str(SOURCE/f'seed_{seed}/summary.json'),
                                source_sha256=digest(SOURCE/f'seed_{seed}/summary.json'),runtime_seconds=0)
                    state['reused']+=1
                else:
                    begin=time.monotonic()
                    model=copy.deepcopy(untouched).to('cuda')
                    n=4096 if cell['size']=='A' else 35904
                    bn=calibrate(model,views.batches(cell['view'],n),cell['estimator'])
                    assert bn['samples']==n and bn['batches']==(32 if n==4096 else 281)
                    logits=predict(model,cx)
                    kl=float(F.kl_div(F.log_softmax(logits,dim=1),F.softmax(teacher_c,dim=1),reduction='batchmean'))
                    measured=metrics(predict(model,dx))
                    item.update(reused=False,metrics=measured,kl_C=kl,bn=bn,runtime_seconds=time.monotonic()-begin)
                    state['new_completed']+=1
                    del model
                write(folder/'items'/f"{name}_{cell['name']}.json",item)
                write(folder/'status.json',state)
                print(seed,name,cell['name'],'reused' if item['reused'] else measured,flush=True)
            assert weights_hash(untouched)==pristine_hash
            del untouched
        assert weights_hash(source)==source_hash and state['new_completed']==44 and state['reused']==4
        state.update(status='completed',finished_utc=utc(),runtime_seconds=time.monotonic()-started,
                     gpu_name=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                     batch_size_per_gpu=128,effective_batch_per_run=128)
    except Exception as e:
        state.update(status='failed',error=str(e),traceback=traceback.format_exc(),finished_utc=utc())
        raise
    finally:
        write(folder/'status.json',state)

if __name__=='__main__':
    main()
