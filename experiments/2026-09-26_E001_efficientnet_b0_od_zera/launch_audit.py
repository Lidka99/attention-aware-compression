"""2026-09-26_E001_efficientnet_b0_od_zera: audyt i wykresy z czystego Git.
Wejścia: stały bieg E001, dane lokalne, Conda attention i cztery GPU.
Wyjścia: runs/E001/audits z logami; wykresy w folderze eksperymentu E001.
Protokół i komenda: AUDYT_OPIS_v1.md; kontekst: OPIS.md obok tego pliku.
"""
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'runs/E001/20260926T013203Z_7cc3f46_seed42_qHKULg'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def main():
    if os.environ.get('CONDA_DEFAULT_ENV') != 'attention':
        raise RuntimeError('Use conda run -n attention')
    if output(['git', 'status', '--porcelain']):
        raise RuntimeError('Commit audit code and protocol before running')
    git_sha = output(['git', 'rev-parse', 'HEAD'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONHASHSEED='42', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    # CUDA visibility must be set before any CUDA runtime calls in this process.
    os.environ.update({k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
                                          'PYTHONHASHSEED', 'OMP_NUM_THREADS')})
    if torch.cuda.device_count() != 4:
        raise RuntimeError('Four visible GPUs required')
    parent = ROOT / 'runs/E001/audits'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_seed42_', dir=parent))
    print(f'AUDIT_RUN={run}', flush=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=4', str(HERE / 'audit.py'), '--source', str(SOURCE),
               '--data-root', str(ROOT / 'data/cifar100'), '--run-dir', str(run)]
    plots = HERE / 'wykresy' / run.name
    plot_command = [sys.executable, str(HERE / 'plot_metrics.py'), '--metrics',
                    str(SOURCE / 'metrics.jsonl'), '--output-dir', str(plots)]
    manifest = {'experiment': HERE.name, 'protocol_version': '1.0', 'run_id': run.name,
                'status': 'running', 'git_sha': git_sha, 'dirty': False,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'cwd': str(ROOT), 'source_run': str(SOURCE), 'plots_dir': str(plots),
                'commands': [command, plot_command], 'kind': 'checkpoint_audit',
                'launcher_argv': sys.argv, 'official_test_used': False, 'reserve_evaluated': False}
    save(run / 'manifest.json', manifest)
    rc = 1
    try:
        (run / 'source').mkdir()
        for p in HERE.iterdir():
            if p.suffix in ('.py', '.md', '.json'):
                shutil.copy2(p, run / 'source' / p.name)
        manifest['source_sha256'] = {p.name: digest(p) for p in (run / 'source').iterdir()}
        save(run / 'manifest.json', manifest)
        original_cfg = json.loads((SOURCE / 'config.resolved.json').read_text())
        save(run / 'config.resolved.json', {
            'source_config': original_cfg, 'world_size': 4, 'batch_per_gpu': 32,
            'workers': 2, 'seed': 42, 'precision': 'FP32', 'amp': False, 'tf32': False,
            'partition': 'dev_ids[rank::4]', 'expected_correct': 3342, 'expected_count': 5000,
            'expected_epoch': 177, 'allowed_prediction_mismatches': 0, 'nll_atol': 1e-6})
        versions = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                    'torchvision': torchvision.__version__, 'numpy': np.__version__,
                    'pillow': PIL.__version__, 'cuda': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version(),
                    'gpu': [torch.cuda.get_device_name(i) for i in range(4)],
                    'gpu_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(4)],
                    'driver': output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
                    'deterministic_algorithms': True, 'tf32': False,
                    'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
                                               'PYTHONHASHSEED', 'OMP_NUM_THREADS')}}
        save(run / 'environment.json', versions)
        old = json.loads((SOURCE / 'environment.json').read_text())
        save(run / 'environment_comparison.json', {
            'differences': {k: {'training': v, 'audit': versions.get(k)} for k, v in old.items()
                            if versions.get(k) != v}})
        (run / 'pip-freeze.txt').write_text(output([sys.executable, '-m', 'pip', 'freeze']) + '\n')
        (run / 'conda-explicit.txt').write_text(output(['conda', 'list', '--explicit']) + '\n')
        (run / 'nvidia-smi.txt').write_text(output(['nvidia-smi']) + '\n')
        command_text = '# Cel: audyt E001; wejścia/wyjścia: source/AUDYT_OPIS_v1.md i source/OPIS.md.\n'
        command_text += f'# Working directory: {ROOT}\n'
        command_text += 'export ' + ' '.join(f'{k}={shlex.quote(versions["env"][k])}'
                                            for k in versions['env']) + '\n'
        command_text += shlex.join(command) + '\n' + shlex.join(plot_command) + '\n'
        (run / 'command.sh').write_text(command_text)
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
            # Curves remain informative even if new inference fails.
            plot_rc = subprocess.run(plot_command, cwd=ROOT, env=env,
                                     stdout=stdout, stderr=stderr).returncode
        manifest['inference_exit_code'] = rc
        manifest['plots_exit_code'] = plot_rc
        manifest['plots_sha256'] = {str(p.relative_to(ROOT)): digest(p)
                                    for p in sorted(plots.glob('*')) if p.is_file()}
        rc = rc or plot_rc
        manifest['status'] = 'completed' if rc == 0 else 'failed'
    except Exception:
        (run / 'launcher_error.txt').write_text(traceback.format_exc())
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(exit_code=rc, finished_utc=datetime.now(timezone.utc).isoformat())
        save(run / 'manifest.json', manifest)
        files = sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
        (run / 'artifacts.sha256').write_text(''.join(
            f'{digest(p)}  {p.relative_to(run)}\n' for p in files))
        print(f'AUDIT_STATUS={manifest["status"]} RUN={run}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
