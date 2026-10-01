"""2026-10-01_E002_obserwacja_se: wykonuje zamrożony pomiar SE na 4 GPU.
Wejścia: config.json, SAMPLES.json/INPUTS.json, best.pt, Conda attention.
Wyjścia: nowy bieg runs/E002, logi i aktywacje; wykresy w folderze E002.
Protokół: PROTOKOL_v1.md; kontekst/szkic: OPIS.md. Bez treningu.
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone

from common import (HERE, ROOT, SOURCE, check_versions, digest, environment, git_output,
                    read_json, require, write_json)


def main():
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention', 'Use Conda attention')
    require(not git_output('status', '--porcelain'), 'Commit code and protocol before measurement')
    cfg = read_json(HERE / 'config.json')
    check_versions(cfg)
    require(cfg['status'] == 'frozen' and cfg['protocol_version'] == '1.0', 'Protocol must be frozen')
    tests = read_json(HERE / 'TESTY_CPU_v1.json')
    require(tests['status'] == 'passed' and tests['config_sha256'] == digest(HERE / 'config.json'),
            'CPU checks do not match frozen configuration')
    require(tests['source_sha256'] == {p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
            'Code changed after CPU checks; rerun checks to a new evidence file')
    variables = {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                 'PYTHONHASHSEED': '42', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    os.environ.update(variables)
    import torch
    require(torch.cuda.device_count() == 4, 'Exactly four visible GPUs required')
    git_sha = git_output('rev-parse', 'HEAD')
    parent = ROOT / 'runs/E002'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    from pathlib import Path
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}_{git_sha[:7]}_seed42_', dir=parent))
    plots = HERE / 'wykresy' / run.name
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=4', str(HERE / 'collect.py'), '--run-dir', str(run)]
    analysis = [sys.executable, str(HERE / 'analyze.py'), '--run-dir', str(run), '--plots-dir', str(plots)]
    manifest = {'experiment': cfg['experiment'], 'protocol_version': '1.0', 'git_sha': git_sha,
                'dirty': False, 'run_id': run.name, 'status': 'running', 'cwd': str(ROOT),
                'started_utc': datetime.now(timezone.utc).isoformat(), 'commands': [command, analysis],
                'env': variables, 'plots_dir': str(plots), 'input_manifest': 'INPUTS.json',
                'sample_manifest_sha256': digest(HERE / 'SAMPLES.json'),
                'checkpoint_sha256': cfg['checkpoint_sha256']}
    write_json(run / 'manifest.json', manifest)
    rc = 1
    print(f'E002_RUN={run}', flush=True)
    try:
        (run / 'source').mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.py', '.md', '.json'):
                shutil.copy2(p, run / 'source' / p.name)
        manifest['source_sha256'] = {p.name: digest(p) for p in (run / 'source').iterdir()}
        manifest['config_sha256'] = digest(HERE / 'config.json')
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'config.resolved.json', cfg)
        for name in ('SAMPLES.json', 'INPUTS.json'):
            shutil.copy2(HERE / name, run / name)
        current_environment = environment()
        write_json(run / 'environment.json', current_environment)
        original_environment = read_json(SOURCE / 'environment.json')
        write_json(run / 'environment_comparison.json', {
            'differences': {k: {'E001': v, 'E002': current_environment.get(k)}
                            for k, v in original_environment.items() if current_environment.get(k) != v}})
        for name, cmd in (('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                          ('conda-explicit.txt', ['conda', 'list', '--explicit']),
                          ('nvidia-smi.txt', ['nvidia-smi'])):
            (run / name).write_text(subprocess.check_output(cmd, text=True))
        (run / 'command.sh').write_text(
            '# E002: pomiar SE; wejścia/wyjścia: source/PROTOKOL_v1.md i source/OPIS.md.\n'
            f'# cwd: {ROOT}\nexport ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in variables.items())
            + '\n' + shlex.join(command) + '\n' + shlex.join(analysis) + '\n')
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            collect_rc = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                        stdout=stdout, stderr=stderr).returncode
            manifest['collection_exit_code'] = collect_rc
            rc = collect_rc
            if collect_rc == 0:
                rc = subprocess.run(analysis, cwd=ROOT, env=os.environ.copy(),
                                    stdout=stdout, stderr=stderr).returncode
                manifest['analysis_exit_code'] = rc
        if rc == 0:
            require(read_json(run / 'summary.json')['status'] == 'completed', 'Incomplete analysis')
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
        print(f'E002_STATUS={manifest["status"]} RUN={run}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
