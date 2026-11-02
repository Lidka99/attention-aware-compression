"""E027: launcher4GPU dla diagnostyki celu; OPIS.md.
Wejście check.json; wyjście unikalny run z logami i hashami wejść.
"""
import os
import platform
import secrets
import shutil
import subprocess
import time
from datetime import datetime,timezone
import torch
from support import *


def main():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and torch.cuda.device_count()==4
    gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=gpu_info.strip().splitlines();assert len(rows)==4
    for i,line in enumerate(rows):
        row=[p.strip() for p in line.split(',')]
        assert int(row[0])==i and int(row[2])<256 and int(row[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    check=bn.read(HERE/'check.json')
    assert check['status']=='passed' and check['protocol_sha256']==digest(HERE/'OPIS.md')
    assert check['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    assert check['inputs']==sources()
    run=ROOT/'runs/E027'/f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}"
    run.mkdir(parents=True,exist_ok=False)
    (run/'source').mkdir();(run/'logs').mkdir()
    for p in HERE.iterdir():
        if p.suffix in ('.py','.json','.md'):
            (run/'source'/p.name).write_bytes(p.read_bytes())
    write(run/'indices.json',bn.read(bn.SOURCE/'indices.json'))
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),launcher_pid=os.getpid(),
                  protocol_sha256=check['protocol_sha256'],source_hashes=check['source_hashes'],inputs=check['inputs'],
                  augmented_views_sha256=check['augmented_views_sha256'],objectives=OBJECTIVES,
                  selection_rule='min disagreement on B',indices_sha256=digest(run/'indices.json'),
                  assignments={str(i):42+i for i in range(4)},official_test_used=False,reserve_used=False,
                  dev_used_as_posthoc_diagnostic=True,python=platform.python_version(),
                  torch=torch.__version__,cuda=torch.version.cuda,gpus=gpu_info)
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
        codes=[p.wait(timeout=max(.1,7200-(time.monotonic()-started))) for p,_ in processes]
        states=[bn.read(run/f'seed_{seed}/status.json') for seed in range(42,46)]
        ok=all(x==0 for x in codes) and all(s['status']=='completed' and 2<=s['evaluations']<=50 for s in states)
        manifest.update(status='completed' if ok else 'failed',exit_codes=codes,total_evaluations=sum(s.get('evaluations',0) for s in states))
    except Exception as e:
        manifest.update(status='failed',error=str(e));raise
    finally:
        for p,log in processes:
            if p.poll() is None:
                p.kill();p.wait()
            log.close()
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
        write(run/'manifest.json',manifest)
    print(manifest['status'],manifest['exit_codes'],flush=True)
    if manifest['status']!='completed':
        raise SystemExit(1)

if __name__=='__main__':
    main()
