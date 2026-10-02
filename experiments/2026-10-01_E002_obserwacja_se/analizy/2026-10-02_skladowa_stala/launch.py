"""2026-10-01_E002_obserwacja_se: uruchamia uzupełniającą analizę tła.
Wejścia: zamrożony config i zweryfikowane zapisane mapy E002 z 4 GPU.
Wyjścia: nowy runs/E002/analyses i nowe wykresy, logi, manifest, hashe.
Protokół: OPIS.md obok; kontekst: ../../OPIS.md. CPU, bez nowej inferencji.
"""
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
import matplotlib
from analyze import HERE, EXPERIMENT, ROOT, digest, read_json, require, write_json


def main():
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention', 'Use Conda attention')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    require(not dirty, 'Commit analysis code/protocol before launch')
    cfg = read_json(HERE / 'config.json')
    tests = read_json(HERE / 'TESTY_CPU.json')
    require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.0', 'Unfrozen analysis')
    require(tests['status'] == 'passed' and tests['config_sha256'] == digest(HERE / 'config.json'), 'CPU evidence/config mismatch')
    require(tests['source_sha256'] == {p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
            'Code changed after CPU checks')
    variables = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '42', 'CUDA_VISIBLE_DEVICES': ''}
    env = dict(os.environ, **variables)
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = ROOT / 'runs/E002/analyses'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_constant_', dir=parent))
    plots = EXPERIMENT / 'wykresy' / run.name
    command = [sys.executable, str(HERE / 'analyze.py'), '--run-dir', str(run), '--plots-dir', str(plots)]
    manifest = {'experiment': cfg['experiment'], 'analysis': cfg['analysis'], 'protocol_version': '1.0',
                'status': 'running', 'run_id': run.name, 'git_sha': git_sha, 'dirty': False,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'cwd': str(ROOT),
                'command': command, 'env': variables, 'source_run': cfg['source_run'],
                'post_hoc': True, 'device': 'CPU', 'new_gpu_inference': False, 'plots_dir': str(plots)}
    write_json(run / 'manifest.json', manifest)
    rc = 1
    print(f'CONSTANT_RUN={run}', flush=True)
    try:
        (run / 'source').mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.py', '.md', '.json'):
                shutil.copy2(p, run / 'source' / p.name)
        manifest['source_sha256'] = {p.name: digest(p) for p in sorted((run / 'source').iterdir())}
        manifest['config_sha256'] = digest(HERE / 'config.json')
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'config.resolved.json', cfg)
        write_json(run / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
                   'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'device': 'CPU',
                   'env': variables, 'new_gpu_inference': False})
        for name, cmd in (('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                          ('conda-explicit.txt', ['conda', 'list', '--explicit'])):
            (run / name).write_text(subprocess.check_output(cmd, text=True))
        (run / 'command.sh').write_text('# E002: analiza stałego tła; wejścia/wyjścia: source/OPIS.md.\n'
                    f'# cwd: {ROOT}\nexport ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in variables.items())
                    + '\n' + shlex.join(command) + '\n')
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        if rc == 0:
            require(read_json(run / 'summary.json')['status'] == 'completed', 'Missing completed analysis')
            manifest['plots_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(plots.iterdir())}
            manifest['status'] = 'completed'
        else:
            manifest['status'] = 'failed'
    except Exception:
        (run / 'launcher_error.txt').write_text(traceback.format_exc())
        manifest['status'] = 'failed'
        rc = 1
        raise
    finally:
        manifest.update(exit_code=rc, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'manifest.json', manifest)
        files = sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
        (run / 'artifacts.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n' for p in files))
        print(f'CONSTANT_STATUS={manifest["status"]}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
