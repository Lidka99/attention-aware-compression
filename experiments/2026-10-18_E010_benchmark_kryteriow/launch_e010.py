"""E010: launcher benchmarku kryteriow na 4 GPU (Tiny ImageNet + CIFAR-100).

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: PROTOKOL_v1.md i OPIS.md.
Wejscia: frozen config.json, baseline E007/E001, modele E008/E003, Conda attention.
Wyjscia: nowy runs/E010/<UTC>_<git>_seed42_<id>/ z manifestem, logami,
stats/, groups/, merge/, repair/ i summary.json; artifacts.sha256.
Etapy: stats -> criteria -> merge -> repair -> summarize, sekwencyjnie.
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
    config_path = HERE / 'config.json'
    if not config_path.exists():
        raise RuntimeError('Missing frozen config.json')
    cfg = json.loads(config_path.read_text())
    if cfg['status'] != 'frozen' or cfg['protocol_version'] not in ('1.0', '1.1'):
        raise RuntimeError('Unfrozen config')
    from support_e010 import verify_inputs
    inputs = verify_inputs(cfg)
    env_vars = {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTHONHASHSEED': '42', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONFAULTHANDLER': '1'}
    os.environ.update(env_vars)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly 4 working GPUs required')
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = ROOT / 'runs/E010'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_seed42_', dir=parent))
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc_per_node=4']
    commands = {
        'stats': torchrun + [str(HERE / 'stats_e010.py'), '--config', str(config_path),
                             '--run-dir', str(run)],
        'criteria': [sys.executable, str(HERE / 'criteria_e010.py'), '--config', str(config_path),
                     '--run-dir', str(run)],
        'merge': torchrun + [str(HERE / 'merge_e010.py'), '--config', str(config_path),
                             '--run-dir', str(run)],
        'repair': torchrun + [str(HERE / 'repair_e010.py'), '--config', str(config_path),
                              '--run-dir', str(run)],
        'summarize': [sys.executable, str(HERE / 'summarize_e010.py'), '--config',
                      str(config_path), '--run-dir', str(run)]}
    manifest = {'experiment': cfg['experiment'], 'protocol_version': '1.0',
                'run_id': run.name, 'git_sha': git_sha, 'dirty': False, 'status': 'running',
                'cwd': str(ROOT), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'commands': commands, 'env': env_vars, 'official_test_used': False,
                'reserve_used': False, 'inputs': inputs}
    write_json(run / 'manifest.json', manifest)
    exit_codes = {}
    print(f'E010_RUN={run}', flush=True)
    try:
        (run / 'source').mkdir()
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in ('.py', '.md', '.json'):
                shutil.copy2(path, run / 'source' / path.name)
        manifest['source_sha256'] = {p.name: digest(p) for p in sorted((run / 'source').iterdir())}
        manifest['config_sha256'] = digest(config_path)
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
            '# E010: benchmark kryteriow; wejscia/wyjscia: source/PROTOKOL_v1.md.\n'
            f'# cwd: {ROOT}\nexport ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in env_vars.items())
            + '\n' + '\n'.join(f'# {stage}\n' + shlex.join(cmd)
                               for stage, cmd in commands.items()) + '\n')
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            for stage in ('stats', 'criteria', 'merge', 'repair', 'summarize'):
                exit_codes[stage] = subprocess.run(commands[stage], cwd=ROOT,
                                                   env=os.environ.copy(),
                                                   stdout=stdout, stderr=stderr).returncode
                if exit_codes[stage] != 0:
                    break
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
        (run / 'artifacts.sha256').write_text(''.join(
            f'{digest(p)}  {p.relative_to(run)}\n' for p in files))
        print(f'E010_STATUS={manifest["status"]} RUN={run}', flush=True)
    sys.exit(0 if manifest['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
