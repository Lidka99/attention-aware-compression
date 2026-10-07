"""2026-10-01_E002_obserwacja_se: nowa analiza szablonu na zapisanych mapach.
Wejścia: frozen config, kontrola CPU, source E002 i kontrola średniej.
Wyjścia: nowy bieg analyses, logi/hashe, wykresy w folderze eksperymentu.
Protokół: OPIS.md obok; kontekst: ../../OPIS.md. Bez inferencji, CPU.
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
from analyze_template import HERE, EXPERIMENT, ROOT, digest, read_json, write_json, require, code_hashes


def main():
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention', 'Use Conda attention')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit protocol/code before launch')
    cfg = read_json(HERE / 'config.json')
    tests = read_json(HERE / 'TESTY_CPU.json')
    require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.0', 'Unfrozen analysis')
    require(tests['status'] == 'passed' and tests['config_sha256'] == digest(HERE / 'config.json')
            and tests['source_sha256'] == code_hashes(), 'Code/config changed after CPU checks')
    env_vars = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '42'}
    env = dict(os.environ, **env_vars)
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = ROOT / 'runs/E002/analyses'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_template_', dir=parent))
    plots = EXPERIMENT / 'wykresy' / run.name
    command = [sys.executable, str(HERE / 'analyze_template.py'), '--run-dir', str(run), '--plots-dir', str(plots)]
    manifest = {'experiment': cfg['experiment'], 'analysis': cfg['analysis'], 'protocol_version': '1.0',
                'run_id': run.name, 'status': 'running', 'git_sha': git_sha, 'dirty': False,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'cwd': str(ROOT),
                'command': command, 'env': env_vars, 'source_run': cfg['source_run'],
                'reference_run': cfg['constant_reference_run'], 'device': 'CPU',
                'new_gpu_inference': False, 'post_hoc': True, 'plots_dir': str(plots)}
    write_json(run / 'manifest.json', manifest)
    rc = 1
    print(f'TEMPLATE_RUN={run}', flush=True)
    try:
        (run / 'source').mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.py', '.md', '.json'):
                shutil.copy2(p, run / 'source' / p.name)
        (run / 'source' / 'dependencies').mkdir()
        for relative in code_hashes()['dependencies']:
            shutil.copy2(ROOT / relative, run / 'source' / 'dependencies' / Path(relative).name)
        manifest['source_sha256'] = {str(p.relative_to(run / 'source')): digest(p)
                                    for p in sorted((run / 'source').rglob('*')) if p.is_file()}
        manifest['config_sha256'] = digest(HERE / 'config.json')
        manifest['code_dependencies'] = code_hashes()['dependencies']
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'config.resolved.json', cfg)
        write_json(run / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
                   'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'device': 'CPU', 'env': env_vars})
        for name, cmd in (('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                          ('conda-explicit.txt', ['conda', 'list', '--explicit'])):
            (run / name).write_text(subprocess.check_output(cmd, text=True))
        (run / 'command.sh').write_text('# E002: analiza szablonu; wejścia/wyjścia: source/OPIS.md.\n'
            f'# cwd: {ROOT}\nexport ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in env_vars.items())
            + '\n' + shlex.join(command) + '\n')
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        if rc == 0:
            require(read_json(run / 'summary.json')['status'] == 'completed', 'Incomplete results')
            manifest['plots_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(plots.iterdir())}
            manifest['status'] = 'completed'
        else:
            manifest['status'] = 'failed'
    except Exception:
        (run / 'launcher_error.txt').write_text(traceback.format_exc())
        rc = 1; manifest['status'] = 'failed'
        raise
    finally:
        manifest.update(exit_code=rc, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'manifest.json', manifest)
        files = sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
        (run / 'artifacts.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n' for p in files))
        print(f'TEMPLATE_STATUS={manifest["status"]}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
