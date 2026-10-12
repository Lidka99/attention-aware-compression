"""E008: podsumowanie etapow (szkoda 10%/30%, naprawa 30%) z bootstrapem par.
Eksperyment: 2026-10-12_E008_kompresja_224; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: stage1/dev_predictions.npz + stage1_metrics.json,
stage2/dev_final_<VARIANT>.npz + stage2_metrics.json.
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
    require((E003 / 'summarize.py').exists(), 'E003 summarize.py missing')
    require(hashlib.sha256((E003 / 'summarize.py').read_bytes()).hexdigest()
            == cfg['summarize_py_sha256'], 'E003 summarize.py changed')
    reps = cfg['bootstrap']['reps']
    seed = cfg['bootstrap']['seed']
    stage1 = run / 'stage1'
    stage2 = run / 'stage2'
    s1_metrics = read_json(stage1 / 'stage1_metrics.json')
    with np.load(stage1 / 'dev_predictions.npz', allow_pickle=False) as f:
        targets = f['targets']
        preds = {k[len('logits_'):]: f[k] for k in f.files if k.startswith('logits_')}
    require(len(targets) == cfg['data']['dev_count'], 'Stage 1 dev count mismatch')
    stage1_results = {'metrics': s1_metrics['modes'], 'comparisons': {}}
    for budget in cfg['budgets']:
        for before, after in (('PRE', 'POST'), ('RANDOM', 'POST'), ('RANDOM', 'PRE')):
            key = f'{budget}_{after}_minus_{budget}_{before}'
            stage1_results['comparisons'][key] = paired_bootstrap(
                correct_from(preds[f'{budget}_{before}'], targets),
                correct_from(preds[f'{budget}_{after}'], targets),
                targets, repetitions=reps, seed=seed)
    stage2_results = {'metrics': read_json(stage2 / 'stage2_metrics.json')['variants'],
                      'comparisons': {}}
    s2_preds = {}
    for name in ('BASE', 'PRE', 'POST', 'RANDOM'):
        with np.load(stage2 / f'dev_final_{name}.npz', allow_pickle=False) as f:
            require(np.array_equal(f['targets'], targets), f'Stage 2 targets mismatch: {name}')
            require(len(f['ids']) == cfg['data']['dev_count'], f'Stage 2 dev count mismatch: {name}')
            s2_preds[name] = f['logits']
    for before, after in (('PRE', 'POST'), ('RANDOM', 'POST'), ('RANDOM', 'PRE'),
                          ('POST', 'BASE'), ('PRE', 'BASE'), ('RANDOM', 'BASE')):
        key = f'{after}_minus_{before}'
        stage2_results['comparisons'][key] = paired_bootstrap(
            correct_from(s2_preds[before], targets),
            correct_from(s2_preds[after], targets),
            targets, repetitions=reps, seed=seed)
    primary = stage2_results['comparisons']['POST_minus_PRE']
    summary = {'experiment': cfg['experiment'], 'run_id': run.name,
               'bootstrap': {'reps': reps, 'seed': seed, 'stratified_by': 'class'},
               'stage1': stage1_results, 'stage2': stage2_results,
               'primary': {'comparison': 'stage2 POST_minus_PRE',
                           'delta_top1': primary['delta_top1'],
                           'ci95_top1': primary['ci95_top1'],
                           'primary_passed': bool(primary['ci95_top1'][0] > 0)}}
    write_json(run / 'summary.json', summary)
    print('E008_SUMMARY=' + json.dumps(summary['primary'], sort_keys=True))


if __name__ == '__main__':
    main()
