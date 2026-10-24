"""E012: uruchamia cztery budżety foldingu na GPU0–3; wejście config.json.

Eksperyment 2026-10-24_E012_folding_referencja, pełny protokół OPIS.md.
Zapisuje unikalny run, snapshot, manifest i summary; bez fine-tuningu.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from evaluate import digest, write_json_atomic, E011_TRAIN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    assert cfg['status'] == 'frozen'
    assert digest(E011_TRAIN) == cfg['e011_train_sha256']
    assert not subprocess.check_output(
        ['git', 'status', '--porcelain', '--', str(HERE)], cwd=ROOT).strip(), 'Commit E012 first'
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3, 'Need 2 GiB free'
    import torch
    assert torch.cuda.device_count() == 4, 'Require exactly four visible GPUs'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True)
    assert not apps.strip(), 'GPUs occupied'
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    run = ROOT / 'runs/E012' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                               + '_' + git[:7] + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    source = run / 'source'
    source.mkdir()
    for p in HERE.iterdir():
        if p.is_file():
            shutil.copy2(p, source / p.name)
    shutil.copy2(E011_TRAIN, source / 'e011_train.py')
    manifest = dict(status='running', started_utc=now(), git_sha=git,
                    protocol_version=cfg['protocol_version'], run_id=run.name,
                    config_sha256=digest(source / 'config.json'),
                    source_sha256={p.name: digest(p) for p in source.iterdir()},
                    official_test_used=False, workers=[], independent_seeds=1,
                    gpu_assignment='GPU0 ratio0; GPU1 ratio.1; GPU2 ratio.3; GPU3 ratio.5',
                    batch_size_per_gpu=128, gradient_updates=0)
    (run / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    (run / 'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    write_json_atomic(run / 'manifest.json', manifest)
    print('E012_RUN=' + str(run), flush=True)
    procs, logs = [], []
    try:
        for gpu, ratio in enumerate(cfg['ratios']):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='42',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            cmd = [sys.executable, '-B', str(HERE / 'evaluate.py'), '--config',
                   str(source / 'config.json'), '--run-dir', str(run), '--ratio',
                   str(ratio), '--gpu', str(gpu)]
            log = (run / f'gpu{gpu}.log').open('w')
            logs.append(log)
            p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT)
            procs.append(p)
            manifest['workers'].append(dict(gpu=gpu, ratio=ratio, pid=p.pid, command=cmd))
        write_json_atomic(run / 'manifest.json', manifest)
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (None, 0) for p in procs):
                raise RuntimeError('Worker failed; see gpu*.log')
            time.sleep(2)
        assert all(p.returncode == 0 for p in procs)
        summaries = {}
        for ratio in cfg['ratios']:
            key = f'ratio_{round(ratio*100):02d}'
            summaries[key] = json.loads((run / key / 'summary.json').read_text())
            assert summaries[key]['status'] == 'completed'
        write_json_atomic(run / 'summary.json', summaries)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()
        manifest.update(finished_utc=now(), exit_codes=[p.returncode for p in procs])
        write_json_atomic(run / 'manifest.json', manifest)
        (run / 'artifacts.sha256').write_text(''.join(
            f'{digest(p)}  {p.relative_to(run)}\n' for p in sorted(run.rglob('*'))
            if p.is_file() and p.name != 'artifacts.sha256'))
        print('E012_STATUS=' + manifest['status'], flush=True)


if __name__ == '__main__':
    main()
