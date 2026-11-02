"""E024: jeden launcher czterech seedów na GPU0–3; patrz OPIS.md.
Wejście: zweryfikowany kod/check.json; wyjście: unikalny runs/E024 z logami.
"""
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
import torch
from data import HERE,ROOT,digest,split_indices,write


def main():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV')=='attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3'
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    lines=status.strip().splitlines()
    assert len(lines)==4 and torch.cuda.device_count()==4
    for i,line in enumerate(lines):
        parts=[x.strip() for x in line.split(',')]
        assert int(parts[0])==i and int(parts[2])<256 and int(parts[3])==0
    assert shutil.disk_usage(ROOT).free>1024**3
    checks=json.loads((HERE/'check.json').read_text())
    assert checks['status']=='passed' and checks['protocol_sha256']==digest(HERE/'OPIS.md')
    assert checks['source_hashes']=={p.name:digest(p) for p in HERE.glob('*.py')}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'runs/E024'/f'{stamp}_{secrets.token_hex(4)}'
    run.mkdir(parents=True,exist_ok=False)
    (run/'logs').mkdir();(run/'source').mkdir()
    for p in HERE.iterdir():
        if p.suffix in ('.py','.md','.json'):
            (run/'source'/p.name).write_bytes(p.read_bytes())
    write(run/'indices.json',split_indices())
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),launcher_pid=os.getpid(),
                  protocol_sha256=digest(HERE/'OPIS.md'),source_hashes=checks['source_hashes'],
                  indices_sha256=digest(run/'indices.json'),assignments={str(i):42+i for i in range(4)},
                  python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpus=status,
                  official_test_used=False,reserve_used=False)
    write(run/'manifest.json',manifest)
    (run/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(f'RUN={run}',flush=True)
    processes=[]
    start=time.monotonic()
    try:
        for gpu in range(4):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=(run/'logs'/f'gpu{gpu}.log').open('w')
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'search.py'),'--run-dir',str(run),
                                '--gpu',str(gpu),'--seed',str(42+gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((p,log))
        manifest['worker_pids']=[p.pid for p,_ in processes]
        write(run/'manifest.json',manifest)
        codes=[]
        for p,log in processes:
            codes.append(p.wait(timeout=max(.1,5400-(time.monotonic()-start))))
        ok=all(code==0 for code in codes)
        manifest.update(status='completed' if ok else 'failed',exit_codes=codes)
    except Exception as e:
        manifest.update(status='failed',error=str(e))
        raise
    finally:
        for p,log in processes:
            if p.poll() is None:
                p.kill();p.wait()
            log.close()
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
        write(run/'manifest.json',manifest)
    print(manifest['status'],manifest['exit_codes'],flush=True)
    if manifest['status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
