"""E018: nadzór nad launcherm z twardym limitem czasu.

Eksperyment: 2026-10-29_E018_replikacja_cosine; pełny protokół: OPIS.md.
Uruchamia ``launch_e018.py`` z katalogu repo, czeka na konkretny PID,
zapisuje exit code i ``.status.json``; przy timeoucie kończy drzewo procesów.
Bez automatycznych restartów i bez wyszukiwania procesów po nazwie.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / 'launch_e018.py'


def kill_manifest_workers(log_path):
    """Domyka workery zapisane w manifeście biegu (po restarcie PID 1)."""
    try:
        text = Path(log_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return
    marker = 'E018_RUN='
    path = None
    for line in text.splitlines():
        if line.startswith(marker):
            path = line[len(marker):].strip()
    if not path:
        return
    manifest_path = Path(path) / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    for worker in manifest.get('workers', []):
        pid = int(worker.get('pid', 0))
        if pid <= 0:
            continue
        for number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, number)
            except ProcessLookupError:
                break
            time.sleep(0.2)


def stop_tree(process):
    """Kończy procesy potomne uruchomione przez launcher."""
    if process.poll() is not None:
        return
    try:
        os.kill(process.pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    parents = {int(pid): int(ppid) for pid, ppid in
               (line.split() for line in rows.splitlines())}
    children = {process.pid}
    while True:
        expanded = children | {pid for pid, ppid in parents.items()
                               if ppid in children}
        if expanded == children:
            break
        children = expanded
    children.discard(process.pid)
    for child in children:
        try:
            os.kill(child, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for child in children:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.kill()
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-seconds', type=int, default=5400)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('Timeout musi być dodatni')
    command = [sys.executable, '-u', '-B', str(LAUNCHER)]

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal={signum}')

    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    state = {
        'status': 'starting',
        'command': command,
        'timeout_seconds': args.timeout_seconds,
        'supervisor_pid': os.getpid(),
    }
    status_path = args.log.with_suffix('.status.json')
    with args.log.open('x') as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='running', launcher_pid=process.pid)
        status_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        try:
            code = process.wait(timeout=args.timeout_seconds)
            state.update(
                status='completed' if code == 0 else 'failed', exit_code=code)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            stop_tree(process)
            kill_manifest_workers(args.log)
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            code = 124 if timed_out else 130
            state.update(status='timeout' if timed_out else 'interrupted',
                         exit_code=code)
        finally:
            state['elapsed_seconds'] = time.monotonic() - start
            status_path.write_text(
                json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
