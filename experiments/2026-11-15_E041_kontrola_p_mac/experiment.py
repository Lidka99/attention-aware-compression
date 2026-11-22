"""2026-11-15_E041_kontrola_p_mac: kontrola alokacji pod P i MAC; OPIS.md.
Wejście: geometria CPU i checkpointy E035; wyjście: check operatora, cztery
procesy oceny dev i audyt metryk. Bez treningu, testu/rezerwy i zapisu wag.
"""
import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-08_E038_operator_alokacja'))
import runtime as rt
import geometry as geom

ROOT = geom.ROOT
PAIRS = geom.PAIRS
CONTROL_IDS = geom.PERMUTATIONS
LIMIT_SECONDS = 1800


def cpu_only():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    rt.torch.set_num_threads(2)


def freeze(cpu_run):
    cpu_only()
    assert not (HERE / 'frozen.json').exists(), 'Zamrożone wejścia już istnieją'
    manifest = rt.read(cpu_run / 'manifest.json')
    geometry = rt.read(cpu_run / 'geometry.json')
    assert manifest['status'] == 'completed' and manifest['gate_A_passed']
    assert manifest['code_sha256'] == rt.digest(HERE / 'geometry.py')
    assert geometry['passed'] and tuple(geometry['names']) == geom.NAMES
    frozen = rt.input_freeze(PAIRS)
    frozen['geometry'] = geometry
    frozen['geometry_run'] = rt.rel(cpu_run)
    for path in (cpu_run / 'geometry.json', cpu_run / 'manifest.json', HERE / 'geometry.py'):
        frozen['hashes'][rt.rel(path)] = rt.digest(path)
    for pair in PAIRS:
        assert frozen['plans'][pair]['b50']['T'] == geometry['results'][pair]['tilt']
        assert [r['permutation'] for r in geometry['results'][pair]['controls']] == list(CONTROL_IDS)
    rt.write(HERE / 'frozen.json', frozen)
    print('FREEZE PASS', rt.digest(HERE / 'frozen.json'), flush=True)


def verify_model(pair, base, model, diagnostics, widths, expected):
    geo = rt.e37.geometry(pair, base)
    assert tuple(geo.names) == geom.GEOMETRY_NAMES
    assert list(geo.sizes) == geom.SIZES.tolist()
    assert [row['k'] for row in diagnostics] == widths
    for alias, row in zip(geo.names, diagnostics):
        assert row['first_axis'] == geo.members[alias][0][0]
        assert sorted(set(row['labels'])) == list(range(row['k']))
    actual = (sum(p.numel() for p in model.parameters()), rt.e37.mac(model, pair))
    assert actual == tuple(expected), (pair, actual, expected)
    assert actual[0] == geo.count(widths)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert rt.torch.equal(model.fc.bias, base.fc.bias)
    return actual


def check():
    cpu_only()
    frozen = rt.read(HERE / 'frozen.json')
    rt.e35.validate_hashes(frozen['hashes'])
    name = 'check_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(4)
    out = ROOT / 'runs/E041' / name
    out.mkdir()
    state = dict(status='running', phase='operator_cpu', pid=os.getpid(), started_utc=rt.utc(),
                 gpu_used=False, images_opened=False, quality_evaluated=False, rows=[])
    rt.write(out / 'manifest.json', state)
    started = time.monotonic()
    try:
        for pair in PAIRS:
            base = rt.e35.model_source(pair, 42)
            before = rt.e35.state_hash(base)
            axes = rt.e35.e29.get_axis_to_perm_ResNet18(override=False)
            groups = rt.e35.e29.axes2perm_to_perm2axes(axes)
            assert tuple(groups) == geom.NAMES
            geo = rt.e37.geometry(pair, base)
            for (_, pairs), alias in zip(groups.items(), geo.names):
                assert pairs == geo.members[alias]
            for plan in frozen['geometry']['results'][pair]['controls']:
                model, diag = rt.e37.fold_widths(pair, base, plan['widths'])
                actual = verify_model(pair, base, model, diag, plan['widths'], (plan['params'], plan['mac']))
                assert rt.e35.state_hash(base) == before
                state['rows'].append(dict(pair=pair, permutation=plan['permutation'],
                                          params=actual[0], mac=actual[1],
                                          state_sha256=rt.e35.state_hash(model)))
                rt.write(out / 'manifest.json', state)
                print('CPU OPERATOR', pair, plan['permutation'], actual, flush=True)
                assert time.monotonic() - started < 300, 'Limit kontroli operatora CPU300s'
                del model
            del base
        assert len(state['rows']) == 8 and not rt.DENIED
        state.update(status='passed', code_hashes=rt.hashes(HERE),
                     frozen_sha256=rt.digest(HERE / 'frozen.json'), data_open_attempts=rt.DENIED)
        rt.write(HERE / 'check.json', dict(state, check_run=rt.rel(out)))
    except BaseException:
        state.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        state.update(finished_utc=rt.utc(), seconds=time.monotonic() - started)
        rt.write(out / 'manifest.json', state)
    print('CPU CHECK PASS', rt.rel(out), flush=True)


def measure(pair, model, a, x, y):
    metrics, evidence = {}, {}
    for mode in rt.MODES:
        metrics[mode], evidence[mode] = rt.e35.evaluate(pair, model, a, x, y, mode)
    return metrics, evidence


def worker(run, gpu):
    rt.configure(gpu)
    seed = 42 + gpu
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', seed=seed, gpu=gpu, pid=os.getpid(),
                  started_utc=rt.utc(), completed=[])
    rt.write(out / 'status.json', status)
    frozen = rt.read(run / 'frozen.json')
    cached = {}
    try:
        # Wspólna bariera odtworzenia4procesów, przed oceną nowych kontroli.
        for pair in PAIRS:
            ds = pair.split('_')[1]
            a, x, y, views = rt.e35.e31.data(ds, frozen['indices'][ds])
            base = rt.e35.model_source(pair, seed)
            before = rt.e35.state_hash(base)
            ref = frozen['references'][pair]['b50'][str(seed)]
            assert views == ref['views_sha256']
            metrics, evidence = {}, {}
            metrics['teacher'], evidence['teacher'] = measure(pair, base, a, x, y)
            rt.check_reference(metrics['teacher'], evidence['teacher'],
                               ref['controls']['metrics'], ref['controls']['evidence'])
            for profile in ('U', 'T'):
                folded, _ = rt.e35.fold(pair, base, frozen['plans'][pair]['b50'], profile)
                metrics[profile], evidence[profile] = measure(pair, folded, a, x, y)
                rt.check_reference(metrics[profile], evidence[profile],
                                   ref['metrics'][profile], ref['evidence'][profile])
                del folded
            assert rt.e35.state_hash(base) == before
            rt.write(out / pair / 'references.json', dict(metrics=metrics, evidence=evidence,
                     views_sha256=views, phase='dev', official_test_used=False, reserve_used=False))
            cached[pair] = (base, a, x, y, views, before)
        rt.write(out / 'reference_gate.json', dict(passed=True, seed=seed, utc=rt.utc()))
        wait_start = time.monotonic()
        while True:
            gates = [run / f'seed_{s}/reference_gate.json' for s in rt.SEEDS]
            if all(p.exists() for p in gates):
                assert all(rt.read(p)['passed'] for p in gates)
                break
            if time.monotonic() - wait_start > 600:
                raise TimeoutError('Brak wspólnej bramki odtworzenia w600s')
            time.sleep(1)
        for pair in PAIRS:
            base, a, x, y, views, before = cached[pair]
            for plan in frozen['geometry']['results'][pair]['controls']:
                model, diagnostics = rt.e37.fold_widths(pair, base, plan['widths'])
                actual = verify_model(pair, base, model, diagnostics, plan['widths'],
                                      (plan['params'], plan['mac']))
                metrics, evidence = measure(pair, model, a, x, y)
                assert rt.e35.state_hash(base) == before
                rt.write(out / pair / f'control_{plan["permutation"]}.json', dict(
                    status='completed', pair=pair, permutation=plan['permutation'], seed=seed, gpu=gpu,
                    plan=plan, actual=dict(params=actual[0], mac=actual[1]),
                    metrics=metrics, evidence=evidence, clusters=diagnostics, views_sha256=views,
                    checkpoint_sha256=rt.digest(rt.e35.checkpoint(pair, seed)),
                    phase='dev', bias_preserved=True, official_test_used=False, reserve_used=False))
                status['completed'].append(f'{pair}/{plan["permutation"]}')
                rt.write(out / 'status.json', status)
                print(seed, pair, plan['permutation'], 'CMA', 100 * metrics['CMA']['top1'], flush=True)
                del model
        assert not rt.DENIED and len(status['completed']) == 8
        status.update(status='completed', finished_utc=rt.utc(), data_open_attempts=rt.DENIED)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc(), data_open_attempts=rt.DENIED)
        raise
    finally:
        cached.clear()
        rt.write(out / 'status.json', status)


def launch():
    assert 'envs/attention' in sys.executable and os.environ.get('CONDA_DEFAULT_ENV') == 'attention'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3' and rt.torch.cuda.device_count() == 4
    frozen = rt.validate_check(HERE)
    info = rt.telemetry()
    lines = info.strip().splitlines()
    assert len(lines) == 4
    for index, line in enumerate(lines):
        row = [v.strip() for v in line.split(',')]
        assert int(row[0]) == index and int(row[3]) < 256 and int(row[4]) == 0, info
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'], text=True)
    assert not apps.strip(), apps
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(4)
    run = ROOT / 'runs/E041' / name
    (run / 'source').mkdir(parents=True)
    for p in HERE.iterdir():
        if p.suffix in ('.py', '.md', '.json'):
            shutil.copy2(p, run / 'source' / p.name)
    for path in frozen['hashes']:
        p = ROOT / path
        if p.suffix == '.py':
            dest = run / 'source/dependencies' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.copy2(HERE / 'frozen.json', run / 'frozen.json')
    manifest = dict(status='running', pid=os.getpid(), run=rt.rel(run), started_utc=rt.utc(),
        limit_seconds=LIMIT_SECONDS, git_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        python=sys.version, executable=sys.executable, torch=rt.torch.__version__, cuda=rt.torch.version.cuda,
        gpu_start=info, code_hashes=rt.hashes(HERE), frozen_sha256=rt.digest(run / 'frozen.json'), workers=[],
        settings=dict(pairs=PAIRS, seeds=list(rt.SEEDS), control_ids=CONTROL_IDS,
                      modes=rt.MODES, batch_per_gpu=128, phase='dev', training=False,
                      global_training_batch=None, gradient_accumulation=None, evaluated_models=56),
        environment={k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')},
        command=[sys.executable, '-B', rt.rel(HERE / 'experiment.py'), 'run'])
    (run / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    rt.write(run / 'manifest.json', manifest)
    print('RUN', run, 'PID', os.getpid(), flush=True)
    processes, logs = [], []
    started = time.monotonic()
    try:
        for gpu in range(4):
            log = (run / f'worker_{gpu}.log').open('w')
            logs.append(log)
            process = subprocess.Popen([sys.executable, '-B', str(HERE / 'experiment.py'), 'worker',
                        '--run', str(run), '--gpu', str(gpu)], cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            manifest['workers'].append(dict(gpu=gpu, seed=42 + gpu, pid=process.pid))
        rt.write(run / 'manifest.json', manifest)
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in processes]
                monitor.write(json.dumps(dict(utc=rt.utc(), elapsed=time.monotonic() - started,
                                               exit_codes=codes, gpu=rt.telemetry())) + '\n')
                monitor.flush()
                assert all(c in (None, 0) for c in codes), codes
                if codes == [0] * 4:
                    break
                if time.monotonic() - started > LIMIT_SECONDS:
                    raise TimeoutError('Limit E041:30min')
                time.sleep(5)
        assert rt.hashes(HERE) == manifest['code_hashes']
        rt.e35.validate_hashes(frozen['hashes'])
        manifest['artifact_bytes'] = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
        assert manifest['artifact_bytes'] <= 100 * 1024 ** 2
        manifest['status'] = 'completed'
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for row, process in zip(manifest['workers'], processes):
            row['exit_code'] = process.poll()
        for log in logs:
            log.close()
        manifest.update(finished_utc=rt.utc(), elapsed_seconds=time.monotonic() - started)
        try:
            manifest['gpu_end'] = rt.telemetry()
        except Exception as error:
            manifest['gpu_end_error'] = repr(error)
        rt.write(run / 'manifest.json', manifest)
        print(manifest['status'], rt.rel(run), flush=True)
    return run


def analyze(run):
    manifest, frozen = rt.verify_run(run)
    rows, contrasts = [], []
    sources = {}
    for seed in rt.SEEDS:
        assert rt.read(run / f'seed_{seed}/reference_gate.json')['passed']
        for pair in PAIRS:
            reference = rt.read(run / f'seed_{seed}/{pair}/references.json')
            ref = frozen['references'][pair]['b50'][str(seed)]
            for key in ('teacher', 'U', 'T'):
                metrics = ref['controls']['metrics'] if key == 'teacher' else ref['metrics'][key]
                evidence = ref['controls']['evidence'] if key == 'teacher' else ref['evidence'][key]
                rt.check_reference(reference['metrics'][key], reference['evidence'][key], metrics, evidence)
            for plan in frozen['geometry']['results'][pair]['controls']:
                path = run / f'seed_{seed}/{pair}/control_{plan["permutation"]}.json'
                item = rt.read(path)
                assert item['status'] == 'completed' and item['plan'] == plan
                assert item['phase'] == 'dev' and not item['official_test_used'] and not item['reserve_used']
                assert item['bias_preserved'] and item['seed'] == seed and item['pair'] == pair
                assert [d['k'] for d in item['clusters']] == plan['widths']
                assert item['actual'] == dict(params=plan['params'], mac=plan['mac'])
                target = frozen['geometry']['results'][pair]['target']
                assert geom.inside([plan['params'], plan['mac']], rt.np.array(target))
                classes = 10 if pair.endswith('cifar10') else 100
                assert tuple(geom.costs(plan['widths'], classes)) == (plan['params'], plan['mac'])
                assert item['views_sha256'] == ref['views_sha256']
                assert item['checkpoint_sha256'] == frozen['hashes'][rt.rel(rt.e35.checkpoint(pair, seed))]
                sources[rt.rel(path)] = rt.digest(path)
                for mode in rt.MODES:
                    metric, evidence = item['metrics'][mode], item['evidence'][mode]
                    rt.verify_metric(metric, evidence)
                    rows.append(dict(pair=pair, seed=seed, control=plan['permutation'], mode=mode,
                        T=100 * reference['metrics']['T'][mode]['top1'], C=100 * metric['top1'],
                        U=100 * reference['metrics']['U'][mode]['top1'],
                        teacher=100 * reference['metrics']['teacher'][mode]['top1'],
                        T_nll=reference['metrics']['T'][mode]['nll'], C_nll=metric['nll']))
    for pair in PAIRS:
        for mode in rt.MODES:
            for comparator in ('mean4',) + CONTROL_IDS:
                values, nll = [], []
                for seed in rt.SEEDS:
                    selected = [r for r in rows if (r['pair'], r['mode'], r['seed']) == (pair, mode, seed)
                                and (comparator == 'mean4' or r['control'] == comparator)]
                    assert len(selected) == (4 if comparator == 'mean4' else 1)
                    values.append(selected[0]['T'] - float(rt.np.mean([r['C'] for r in selected])))
                    nll.append(selected[0]['T_nll'] - float(rt.np.mean([r['C_nll'] for r in selected])))
                stats = rt.stats(values)
                contrasts.append(dict(pair=pair, mode=mode, comparator=str(comparator), **stats,
                    direction=stats['mean'] > 0 and stats['positive'] >= 3,
                    magnitude_1pp=stats['mean'] >= 1 and stats['positive'] >= 3, nll=rt.stats(nll)))
    primary = [r for r in contrasts if r['mode'] == 'CMA' and r['comparator'] == 'mean4']
    assert len(rows) == 96 and len(primary) == 2
    rt.write(run / 'analysis/analysis.json', dict(rows=rows, contrasts=contrasts, primary=primary,
        primary_pass=all(r['direction'] for r in primary), magnitude_pass=all(r['magnitude_1pp'] for r in primary),
        phase='dev', independent_validation=False, evaluations=168, new_control_evaluations=96,
        elapsed_seconds=manifest['elapsed_seconds'], source_hashes=sources))
    rt.csv_write(run / 'analysis/per_seed.csv', rows)
    rt.write(run / 'analysis/audit.json', dict(passed=True, checks=dict(worker_exits=True,
        sources_unchanged=True, reference_gate=True, profiles_and_costs=True,
        weights_unchanged=True, samples_and_metrics=True, all_32_controls=True),
        manifest_sha256=rt.digest(run / 'manifest.json'), source_hashes=sources))
    for item in primary:
        print(item['pair'], 'T-mean4', item['mean'], 'SD', item['sd'], 'positive', item['positive'], flush=True)
    print('AUDIT PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'check', 'run', 'worker', 'analyze'))
    parser.add_argument('--cpu-run', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.cpu_run)
    elif args.action == 'check':
        check()
    elif args.action == 'run':
        launch()
    elif args.action == 'worker':
        worker(args.run, args.gpu)
    else:
        analyze(args.run)
