"""E016 S2: ograniczony czasowo nadzór launchera, bez wyszukiwania pgrep.

Przyjmuje --smoke, --timeout-seconds i nową ścieżkę --log; zapisuje log
oraz wynik procesu w pliku .status.json. Protokół: OPIS_S2.md.
Nie restartuje eksperymentu po błędzie. Timeout kończy nadzorowane drzewo.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def stop_tree(process):
    """Kończy tylko potomków uruchomionego launchera, także osobne sesje GPU."""
    if process.poll() is not None:
        return
    os.kill(process.pid, signal.SIGSTOP)
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    parents = {int(pid): int(ppid) for pid, ppid in
               (line.split() for line in rows.splitlines())}
    children = {process.pid}
    while True:
        expanded = children | {pid for pid, ppid in parents.items() if ppid in children}
        if expanded == children:
            break
        children = expanded
    children.remove(process.pid)
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
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=1200)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('Timeout musi być dodatni')
    command = [sys.executable, '-u', '-B',
               str(Path(__file__).with_name('launch_s2.py'))]
    if args.smoke:
        command.append('--smoke')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal={signum}')
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    state = {'status': 'starting', 'command': command,
             'timeout_seconds': args.timeout_seconds, 'supervisor_pid': os.getpid()}
    status_path = args.log.with_suffix('.status.json')
    with args.log.open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='running', launcher_pid=process.pid)
        status_path.write_text(json.dumps(state, indent=2))
        try:
            code = process.wait(timeout=args.timeout_seconds)
            state.update(status='completed' if code == 0 else 'failed', exit_code=code)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            stop_tree(process)
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            state.update(status='timeout' if code == 124 else 'interrupted',
                         exit_code=code)
        finally:
            state['elapsed_seconds'] = time.monotonic() - start
            status_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
