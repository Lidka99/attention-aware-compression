#!/usr/bin/env bash
# E001: uruchamia trening/test techniczny na 4 GPU i zapisuje logi w nowym folderze.
# Wejście: opcje train.py. Wyjście: runs/E001/<UTC>_<git>_seed42_<unikalny>/.
# Instrukcje: URUCHOMIENIE.md i OPIS.md w tym samym folderze.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "${CONDA_DEFAULT_ENV:-}" != "attention" ]]; then
  echo 'Najpierw: conda activate attention' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
RUN_PARENT="$PROJECT_ROOT/runs/E001"
mkdir -p "$RUN_PARENT"
RUN_DIR="$(mktemp -d "$RUN_PARENT/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_seed42_XXXXXX")"
printf '%q ' "$0" "$@" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
echo "Wyniki: $RUN_DIR"
set +e
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  "$SCRIPT_DIR/train.py" --run-dir "$RUN_DIR" "$@" \
  > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
RUN_EXIT=$?
wait
set -e
python - "$RUN_DIR" "$RUN_EXIT" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, code = Path(sys.argv[1]), int(sys.argv[2])
p = run / 'manifest.json'
m = json.loads(p.read_text()) if p.exists() else {'status': 'failed', 'run_id': run.name}
m['exit_code'] = code
if code:
    m['status'] = 'failed'
p.write_text(json.dumps(m, indent=2) + '\n')
with (run / 'artifacts.sha256').open('w') as out:
    for f in sorted(run.iterdir()):
        if f.is_file() and f.name != 'artifacts.sha256':
            h = hashlib.sha256()
            with f.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1048576), b''):
                    h.update(chunk)
            out.write(f'{h.hexdigest()}  {f.name}\n')
PY
exit "$RUN_EXIT"
