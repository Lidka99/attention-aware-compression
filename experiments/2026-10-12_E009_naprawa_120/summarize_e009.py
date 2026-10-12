"""E009: podsumowanie naprawy 120 epok z bootstrapem par (POST-PRE i pozostale).
Eksperyment: 2026-10-12_E009_naprawa_120; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: dev_final_<VARIANT>.npz z katalogu biegu oraz repair_metrics.json.
Wyjscia: summary.json z metrykami i porownaniami sparowanymi po obrazach.
Bootstrap: import paired_bootstrap z E003 summarize.py (hash przypiety w configu).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E003 = ROOT / 'experiments/2026-10-09_E003_kompresja_10proc_parametrow'
if str(E003) not in sys.path:
    sys.path.insert(0, str(E003))

from summarize import paired_bootstrap  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def correct_from(logits, targets):
    return (logits.argmax(1) == targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = read_json(HERE / 'config.json')
    require(cfg['status'] == 'frozen', 'Unfrozen config')
    require(hashlib.sha256((E003 / 'summarize.py').read_bytes()).hexdigest()
            == cfg['summarize_py_sha256'], 'E003 summarize.py changed')
    reps = cfg['bootstrap']['reps']
    seed = cfg['bootstrap']['seed']
    metrics = read_json(run / 'repair_metrics.json')['variants']
    preds = {}
    targets = None
    for name in ('BASE', 'PRE', 'POST', 'RANDOM'):
        with np.load(run / f'dev_final_{name}.npz', allow_pickle=False) as f:
            if targets is None:
                targets = f['targets']
            require(np.array_equal(f['targets'], targets), f'Targets mismatch: {name}')
            require(len(f['ids']) == cfg['data']['dev_count'], f'Dev count mismatch: {name}')
            preds[name] = f['logits']
    comparisons = {}
    for before, after in (('PRE', 'POST'), ('RANDOM', 'POST'), ('RANDOM', 'PRE'),
                          ('POST', 'BASE'), ('PRE', 'BASE'), ('RANDOM', 'BASE')):
        key = f'{after}_minus_{before}'
        comparisons[key] = paired_bootstrap(
            correct_from(preds[before], targets), correct_from(preds[after], targets),
            targets, repetitions=reps, seed=seed)
    primary = comparisons['POST_minus_PRE']
    summary = {'experiment': cfg['experiment'], 'run_id': run.name,
               'bootstrap': {'reps': reps, 'seed': seed, 'stratified_by': 'class'},
               'metrics': metrics, 'comparisons': comparisons,
               'primary': {'comparison': 'POST_minus_PRE',
                           'delta_top1': primary['delta_top1'],
                           'ci95_top1': primary['ci95_top1'],
                           'primary_passed': bool(primary['ci95_top1'][0] > 0)}}
    write_json(run / 'summary.json', summary)
    print('E009_SUMMARY=' + json.dumps(summary['primary'], sort_keys=True))


if __name__ == '__main__':
    main()
