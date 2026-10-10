"""E004: uruchamia wspólne dostrajanie na dokładnie 4 GPU i analizę CPU.
Eksperyment: 2026-10-10_E004_wspolne_dostrajanie; PROTOKOL_v1.md i OPIS.md.
Wejścia: frozen config/plan, kontrole TESTY_CPU_v1, modele E003, dane CIFAR.
Wyjścia: nowy runs/E004 oraz wykresy w folderze eksperymentu; bez nadpisywania.
Bez testu i rezerwy; źródłowe modele E003 pozostają niezmienione.
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from support_e004 import HERE, ROOT, DEPENDENCIES, digest, read_json, write_json, require, code_hashes


def main():
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention', 'Use Conda attention')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit code/protocol before launch')
    cfg = read_json(HERE / 'config.json')
    checks = read_json(HERE / 'TESTY_CPU_v1.json')
    require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.0', 'Unfrozen experiment')
    require(checks['status'] == 'passed' and checks['config_sha256'] == digest(HERE / 'config.json')
            and checks['source_sha256'] == code_hashes(), 'Code/config changed after CPU checks')
    env_vars = {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTHONHASHSEED': '42', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    os.environ.update(env_vars)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4,
            'Exactly 4 working GPUs required')
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = ROOT / 'runs/E004'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_seed42_', dir=parent))
    plots = HERE / 'wykresy' / run.name
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=4', str(HERE / 'train_finetune.py'), '--run-dir', str(run)]
    analysis = [sys.executable, str(HERE / 'summarize_finetune.py'),
                '--run-dir', str(run), '--plots-dir', str(plots)]
    manifest = {'experiment': cfg['experiment'], 'protocol_version': '1.0', 'run_id': run.name,
                'git_sha': git_sha, 'dirty': False, 'status': 'running', 'cwd': str(ROOT),
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'commands': [command, analysis], 'env': env_vars, 'plots_dir': str(plots),
                'source_experiment': cfg['source_experiment'], 'source_run': cfg['source_run'],
                'official_test_used': False, 'reserve_used': False}
    write_json(run / 'manifest.json', manifest)
    rc = 1
    print(f'E004_RUN={run}', flush=True)
    try:
        (run / 'source').mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.py', '.md', '.json'):
                shutil.copy2(p, run / 'source' / p.name)
        (run / 'source' / 'dependencies').mkdir()
        for relative in DEPENDENCIES:
            shutil.copy2(ROOT / relative, run / 'source' / 'dependencies' / Path(relative).name)
        manifest['source_sha256'] = {str(p.relative_to(run / 'source')): digest(p)
                                     for p in sorted((run / 'source').rglob('*')) if p.is_file()}
        manifest['config_sha256'] = digest(HERE / 'config.json')
        manifest['code_dependencies'] = code_hashes()['dependencies']
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'config.resolved.json', cfg)
        write_json(run / 'environment.json', {'python': sys.version,
                   'platform': __import__('platform').platform(), 'torch': torch.__version__,
                   'torchvision': __import__('torchvision').__version__, 'numpy': __import__('numpy').__version__,
                   'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'gpu': [torch.cuda.get_device_name(i) for i in range(4)],
                   'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                      '--format=csv,noheader'], text=True).strip(),
                   'deterministic_algorithms': True, 'tf32': False, 'env': env_vars})
        for name, cmd in (('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                          ('conda-explicit.txt', ['conda', 'list', '--explicit']),
                          ('nvidia-smi.txt', ['nvidia-smi'])):
            (run / name).write_text(subprocess.check_output(cmd, text=True))
        (run / 'command.sh').write_text(
            '# E004: wspólne dostrajanie; wejścia/wyjścia: source/PROTOKOL_v1.md i source/OPIS.md.\n'
            f'# cwd: {ROOT}\nexport ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in env_vars.items())
            + '\n' + shlex.join(command) + '\n' + shlex.join(analysis) + '\n')
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            rc = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                stdout=stdout, stderr=stderr).returncode
            manifest['train_exit_code'] = rc
            if rc == 0:
                rc = subprocess.run(analysis, cwd=ROOT, env=os.environ.copy(),
                                    stdout=stdout, stderr=stderr).returncode
                manifest['analysis_exit_code'] = rc
        if rc == 0:
            require(read_json(run / 'summary.json')['status'] == 'completed', 'Missing completed analysis')
            manifest['plots_sha256'] = {str(p.relative_to(ROOT)): digest(p)
                                        for p in sorted(plots.iterdir())}
            manifest['status'] = 'completed'
        else:
            manifest['status'] = 'failed'
    except Exception:
        (run / 'launcher_error.txt').write_text(traceback.format_exc())
        rc = 1
        manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(exit_code=rc, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'manifest.json', manifest)
        files = sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
        (run / 'artifacts.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n' for p in files))
        print(f'E004_STATUS={manifest["status"]} RUN={run}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
