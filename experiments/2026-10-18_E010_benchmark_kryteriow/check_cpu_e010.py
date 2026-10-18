"""E010: testy CPU przed zamrozeniem (bez GPU i bez treningu).

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: PROTOKOL_v1.md.
Sprawdza: hashe wejsc, odtworzenie modeli E008/E003 przez folding_e010,
grupy nowych kryteriow na syntetycznych statystykach (liczby, rozlacznosc,
determinizm, budzet parametrow), zgodnosc bootstrapu z E003 oraz py_compile.
Wyjscia: TESTY_CPU_v1.json (passed/failed) i kod wyjscia 0/1.
"""
import argparse
import json
import py_compile
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from support_e010 import (ROOT, load_config, verify_inputs, dataset_cfg, plan_rows,
                          plan_spec, base_model, read_json, write_json, require,
                          digest, DATASET_KEYS)
from folding_e010 import fold_model, count_parameters
import criteria_e010
from summarize_e010 import bootstrap_exact

HERE = Path(__file__).resolve().parent
E008 = ROOT / 'experiments/2026-10-12_E008_kompresja_224'
E003 = ROOT / 'experiments/2026-10-09_E003_kompresja_10proc_parametrow'
E009 = ROOT / 'experiments/2026-10-12_E009_naprawa_120'
for path in (E008, E003, E009):
    if str(path) not in __import__('sys').path:
        __import__('sys').path.insert(0, str(path))


def check(name, function, results):
    started = time.time()
    try:
        detail = function()
        results[name] = {'status': 'passed', 'seconds': time.time() - started,
                         'detail': detail}
        print(f'[OK] {name}: {detail}')
    except Exception as error:
        results[name] = {'status': 'failed', 'seconds': time.time() - started,
                         'detail': f'{type(error).__name__}: {error}'}
        print(f'[FAIL] {name}: {error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.draft.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = {}
    started = time.time()
    failed = None
    temp = Path(tempfile.mkdtemp(prefix='e010_cpu_', dir='/tmp/opencode'))
    try:
        check('input_hashes', lambda: {'verified': True,
              'datasets': list(verify_inputs(cfg)['datasets'])}, results)

        def folding_equivalence():
            cases = []
            e008 = ROOT / 'runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim/stage1'
            e003 = ROOT / 'runs/E003/20261010T095706Z_5a2f233_seed42__0o_uckk'
            cases.append(('tiny', '30', e008, 'candidate_manifest_30.json', '30_'))
            cases.append(('tiny', '10', e008, 'candidate_manifest_10.json', '10_'))
            cases.append(('cifar100', '10', e003, 'candidate_manifest.json', ''))
            checked = 0
            for key, ratio, run, manifest_name, prefix in cases:
                ds = dataset_cfg(cfg, key)
                base = base_model(ds)
                rows = plan_rows(ds, ratio)
                manifest = read_json(run / manifest_name)
                for mode in ('PRE', 'POST', 'RANDOM'):
                    model = fold_model(base, rows, manifest['groups'][mode],
                                       int(ds['plans'][ratio]['base_parameters']))
                    reference = torch.load(run / 'models' / f'{prefix}{mode}.pt',
                                           map_location='cpu', weights_only=True)['state_dict']
                    state = model.state_dict()
                    require(set(state) == set(reference)
                            and all(torch.equal(state[k], reference[k]) for k in state),
                            f'{key} {ratio} {mode}: state_dict mismatch')
                    require(count_parameters(model)
                            == int(ds['plans'][ratio]['compressed_parameters']),
                            f'{key} {ratio} {mode}: parameter budget mismatch')
                    checked += 1
            return {'models_checked': checked}

        check('folding_equivalence', folding_equivalence, results)

        def synthetic_stats(run_dir):
            (run_dir / 'stats').mkdir(parents=True, exist_ok=True)
            for key in DATASET_KEYS:
                ds = dataset_cfg(cfg, key)
                rows = plan_rows(ds, sorted(ds['criteria']['ratios'])[0])
                rng = np.random.default_rng(20261018)
                arrays = {}
                for index, row in enumerate(rows):
                    channels = int(row['expanded_channels'])
                    arrays[f'b{index}_taylor'] = rng.random(channels)
                    arrays[f'b{index}_apoz'] = rng.random(channels) * 0.5
                    arrays[f'b{index}_count'] = np.asarray([1000.0])
                np.savez(run_dir / 'stats' / f'{key}.npz', **arrays)
                write_json(run_dir / 'stats' / f'{key}.json',
                           {'settings': {'kmeans_seed': 20261018}, 'dataset': key})

        def criteria_checks():
            run_a = temp / 'a'
            run_b = temp / 'b'
            synthetic_stats(run_a)
            shutil.copytree(run_a / 'stats', run_b / 'stats')
            first = {key: criteria_e010.generate_dataset(cfg, key, run_a)
                     for key in DATASET_KEYS}
            second = {key: criteria_e010.generate_dataset(cfg, key, run_b)
                      for key in DATASET_KEYS}
            checked = 0
            for key in DATASET_KEYS:
                ds = dataset_cfg(cfg, key)
                base = base_model(ds)
                for (ratio, criterion), path in first[key].items():
                    relative = Path(path).relative_to(run_a)
                    require((Path(path).read_bytes()
                             == (run_b / relative).read_bytes()),
                            f'{key} {ratio} {criterion}: nondeterministic groups')
                    payload = read_json(path)
                    rows = plan_rows(ds, ratio)
                    for row in rows:
                        pairs = payload['groups'][row['block']]
                        require(len(pairs) == int(row['remove_count']),
                                f'{key} {ratio} {criterion}: pair count')
                        removed = [int(p['d']) for p in pairs]
                        reps = [int(p['c']) for p in pairs]
                        require(len(set(removed)) == len(removed)
                                and not (set(removed) & set(reps)),
                                f'{key} {ratio} {criterion}: overlap')
                        require(all(np.isfinite(p['alpha']) and np.isfinite(p['beta'])
                                    for p in pairs), 'non-finite alpha/beta')
                    model = fold_model(base, rows, payload['groups'],
                                       int(ds['plans'][ratio]['base_parameters']))
                    require(count_parameters(model)
                            == int(ds['plans'][ratio]['compressed_parameters']),
                            f'{key} {ratio} {criterion}: folded budget')
                    checked += 1
            return {'group_files_checked': checked}

        check('criteria_groups', criteria_checks, results)

        def bootstrap_check():
            import summarize as e003_summarize
            stage2 = ROOT / 'runs/E008/20261012T042932Z_0fd9962_seed42_tn1zlyim/stage2'
            pre = np.load(stage2 / 'dev_final_PRE.npz', allow_pickle=False)
            post = np.load(stage2 / 'dev_final_POST.npz', allow_pickle=False)
            pre_correct = pre['logits'].argmax(1) == pre['targets']
            post_correct = post['logits'].argmax(1) == post['targets']
            reference = e003_summarize.paired_bootstrap(
                pre_correct, post_correct, pre['targets'], repetitions=10000, seed=20261016)
            ours = bootstrap_exact(pre_correct, post_correct,
                                   pre['targets'], 10000, 20261016)
            require(abs(reference['delta_top1'] - ours['delta_top1']) < 1e-12,
                    'E008 delta mismatch')
            require(np.allclose(reference['ci95_top1'], ours['ci95_top1'], atol=1e-12),
                    'E008 CI mismatch')
            e008_ci = ours['ci95_top1']
            e003 = ROOT / 'runs/E003/20261010T095706Z_5a2f233_seed42__0o_uckk'
            predictions = np.load(e003 / 'dev_predictions.npz', allow_pickle=False)
            pre_correct = predictions['logits_PRE'].argmax(1) == predictions['targets']
            post_correct = predictions['logits_POST'].argmax(1) == predictions['targets']
            reference = e003_summarize.paired_bootstrap(
                pre_correct, post_correct, predictions['targets'], repetitions=2000,
                seed=20261009)
            ours = bootstrap_exact(pre_correct, post_correct, predictions['targets'],
                                   2000, 20261009)
            require(abs(reference['delta_top1'] - ours['delta_top1']) < 1e-12,
                    'E003 delta mismatch')
            require(np.allclose(reference['ci95_top1'], ours['ci95_top1'], atol=1e-12),
                    'E003 CI mismatch')
            return {'E008_ci': e008_ci, 'E003_ci': ours['ci95_top1']}

        check('bootstrap_vs_E003', bootstrap_check, results)

        def compile_check():
            compiled = []
            for path in sorted(HERE.glob('*.py')):
                py_compile.compile(str(path), doraise=True)
                compiled.append(path.name)
            return {'modules': compiled}

        check('py_compile', compile_check, results)
    except Exception as error:
        failed = error
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    report = {'experiment': cfg['experiment'],
              'status': 'passed' if failed is None else 'failed',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'config': str(args.config), 'config_sha256': digest(args.config),
              'seconds': time.time() - started, 'checks': results,
              'versions': {'python': __import__('sys').version,
                           'torch': torch.__version__,
                           'numpy': np.__version__}}
    write_json(HERE / 'TESTY_CPU_v1.json', report)
    print('E010_TESTS=' + report['status'], flush=True)
    if failed is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
