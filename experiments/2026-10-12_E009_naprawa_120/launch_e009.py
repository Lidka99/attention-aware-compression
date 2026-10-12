"""E009: launcher naprawy 120 epok na 4 GPU (po etapie 1 kompresji 30%).
Eksperyment: 2026-10-12_E009_naprawa_120; protokol: PROTOKOL_v1.md i OPIS.md.
Wejscia: frozen config.json, modele etapu 1 E008 (pinowane hashem), Conda attention.
Wyjscia: nowy runs/E009 z manifestem, logami, wynikami i summary.json.
Launcher wymaga zakonczonego biegu E008 (wejscie) - sprawdzane przed startem.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/tiny-imagenet'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def main():
    if os.environ.get('CONDA_DEFAULT_ENV') != 'attention':
        raise RuntimeError('Use Conda attention')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit code/protocol before launch')
    cfg = json.loads((HERE / 'config.json').read_text())
    if cfg['status'] != 'frozen' or cfg['protocol_version'] != '1.0':
        raise RuntimeError('Unfrozen config')
    e008 = ROOT / cfg['input']['e008_run']
    e008_manifest = json.loads((e008 / 'manifest.json').read_text())
    if e008_manifest['status'] != 'completed' or e008_manifest.get('exit_code') != 0:
        raise RuntimeError('E008 run is not completed; E009 requires it as input')
    env_vars = {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTHONHASHSEED': '42', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONFAULTHANDLER': '1'}
    os.environ.update(env_vars)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly 4 working GPUs required')
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = ROOT / 'runs/E009'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_seed42_', dir=parent))
    commands = {
        'repair': [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node=4', str(HERE / 'repair_120.py'), '--run-dir', str(run)],
        'summarize': [sys.executable, str(HERE / 'summarize_e009.py'), '--run-dir', str(run)]}
    manifest = {'experiment': cfg['experiment'], 'protocol_version': '1.0', 'run_id': run.name,
                'git_sha': git_sha, 'dirty': False, 'status': 'running', 'cwd': str(ROOT),
                'started_utc': datetime.now(timezone.utc).isoformat(), 'commands': commands,
                'env': env_vars, 'official_test_used': False, 'reserve_used': False,
                'e008_input_run': e008.name, 'e008_git_sha': e008_manifest['git_sha'],
                'source_data': {'splits_sha256': digest(DATA / 'splits.json'),
                                'cache_train_sha256': digest(DATA / 'cache_train.npz'),
                                'cache_val_sha256': digest(DATA / 'cache_val.npz')}}
    write_json(run / 'manifest.json', manifest)
    exit_codes = {}
    print(f'E009_RUN={run}', flush=True)
    try:
        (run / 'source').mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.py', '.md', '.json'):
                shutil.copy2(p, run / 'source' / p.name)
        manifest['source_sha256'] = {p.name: digest(p) for p in sorted((run / 'source').iterdir())}
        manifest['config_sha256'] = digest(HERE / 'config.json')
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'config.resolved.json', cfg)
        write_json(run / 'environment.json', {
            'python': sys.version, 'platform': __import__('platform').platform(),
            'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
            'numpy': __import__('numpy').__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': [torch.cuda.get_device_name(i) for i in range(4)],
            'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                               '--format=csv,noheader'], text=True).strip(),
            'deterministic_algorithms': True, 'tf32': False, 'env': env_vars})
        for name, cmd in (('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                          ('conda-explicit.txt', ['conda', 'list', '--explicit']),
                          ('nvidia-smi.txt', ['nvidia-smi'])):
            (run / name).write_text(subprocess.check_output(cmd, text=True))
        (run / 'command.sh').write_text(
            '# E009: naprawa 120 epok; wejscia/wyjscia: source/PROTOKOL_v1.md.\n'
            f'# cwd: {ROOT}\nexport ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in env_vars.items())
            + '\n# repair\n' + shlex.join(commands['repair'])
            + '\n# summarize\n' + shlex.join(commands['summarize']) + '\n')
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            for stage in ('repair', 'summarize'):
                if stage == 'summarize' and exit_codes.get('repair') != 0:
                    break
                exit_codes[stage] = subprocess.run(commands[stage], cwd=ROOT,
                                                   env=os.environ.copy(),
                                                   stdout=stdout, stderr=stderr).returncode
        manifest['exit_codes'] = exit_codes
        manifest['status'] = 'completed' if exit_codes.get('summarize') == 0 else 'failed'
    except Exception:
        (run / 'launcher_error.txt').write_text(traceback.format_exc())
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(exit_code=exit_codes.get('summarize'),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'manifest.json', manifest)
        files = sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'artifacts.sha256'
                       and p.suffix != '.tmp')
        (run / 'artifacts.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n'
                                                      for p in files))
        print(f'E009_STATUS={manifest["status"]} RUN={run}', flush=True)
    sys.exit(0 if manifest['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
