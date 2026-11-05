"""E031: launcher4GPU, seed42+i/GPUi, trzy zbiory×dwa budżety; OPIS.md.
Wejście check.json; wyjście unikalny audyt z hashami, logami i PID.
"""
import os
import argparse
import platform
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
import torch
from common import *


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reuse-from',type=Path)
    args=ap.parse_args()
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
    rows=gpu_info.strip().splitlines();assert len(rows)==4
    for i,line in enumerate(rows):
        r=[p.strip() for p in line.split(',')]
        assert int(r[0])==i and int(r[2])<256 and int(r[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    check=read(HERE/'check.json')
    assert check['status']=='passed' and check['protocol_sha256']==digest(HERE/'OPIS.md')
    assert check['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    assert check['inputs']==provenance()
    run=ROOT/'runs/E031'/f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}"
    run.mkdir(parents=True,exist_ok=False);(run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.suffix in ('.py','.json','.md'):
            (run/'source'/p.name).write_bytes(p.read_bytes())
    write(run/'plans.json',check['plans'])
    write(run/'indices.json',indices_all())
    reused={}
    if args.reuse_from:
        old=args.reuse_from.resolve()
        previous=read(old/'manifest.json')
        assert previous['protocol_sha256']==check['protocol_sha256']
        assert read(old/'plans.json')==check['plans']
        assert read(old/'indices.json')==read(run/'indices.json')
        assert all(check['inputs'].get(k)==v for k,v in previous['inputs'].items())
        for seed in range(42,46):
            for ds in DATASETS:
                control=old/f'seed_{seed}'/f'controls_{ds}.json'
                for b in BUDGETS:
                    cell=old/f'seed_{seed}'/ds/TAGS[b]
                    if not (cell/'summary.json').exists():
                        continue
                    summary=read(cell/'summary.json')
                    assert summary['status']=='completed' and summary['seed']==seed
                    assert summary['controls_sha256']==digest(control)
                    assert summary['plan']==check['plans'][ds][TAGS[b]]
                    for p in [control]+list(cell.glob('*.json')):
                        target=run/p.relative_to(old)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(p,target)
                        reused[str(p.relative_to(old))]=digest(p)
    expected={f'{ds}/{TAGS[b]}' for ds in DATASETS for b in BUDGETS}
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                  launcher_pid=os.getpid(),protocol_sha256=check['protocol_sha256'],
                  source_hashes=check['source_hashes'],inputs=check['inputs'],
                  upstream_commit=COMMIT,datasets=list(DATASETS),budgets=list(BUDGETS),
                  assignments={str(i):42+i for i in range(4)},official_test_used=False,
                  reserve_used=False,indices_sha256=digest(run/'indices.json'),
                  tiny_model_boundary='nasz torchvision ResNet18; operator upstream niezmieniony',
                  limits=dict(wall_seconds=5400,artifacts_bytes=536870912),
                  python=platform.python_version(),torch=torch.__version__,
                  cuda=torch.version.cuda,gpus=gpu_info)
    manifest['reuse_from']=str(args.reuse_from) if args.reuse_from else None
    manifest['reused_hashes']=reused
    write(run/'manifest.json',manifest)
    (run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(f'RUN={run}',flush=True)
    processes=[];started=time.monotonic()
    try:
        for gpu in range(4):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=(run/'logs'/f'gpu{gpu}.log').open('w')
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'worker.py'),'--run-dir',str(run),'--gpu',str(gpu)],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((p,log))
        manifest['worker_pids']=[p.pid for p,_ in processes];write(run/'manifest.json',manifest)
        codes=[p.wait(timeout=max(.1,5400-(time.monotonic()-started))) for p,_ in processes]
        states=[read(run/f'seed_{seed}/status.json') for seed in range(42,46)]
        ok=all(x==0 for x in codes) and all(
            s['status']=='completed' and set(s['completed'])==expected for s in states)
        manifest.update(status='completed' if ok else 'failed',exit_codes=codes)
    except Exception as e:
        manifest.update(status='failed',error=str(e));raise
    finally:
        for p,log in processes:
            if p.poll() is None:
                p.kill();p.wait()
            log.close()
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.monotonic()-started)
        write(run/'manifest.json',manifest)
    print(manifest['status'],manifest.get('exit_codes'),flush=True)
    if manifest['status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
