"""E040: zamrożenie i ograniczony pilot4GPU z barierami B/C/dev; OPIS.md.
Wejście: własne checkpointy, train i manifest; wyjście: predykcje/metryki/logi.
Stany modeli wyłącznie RAM; bez treningu i dostępu do testu/rezerwy.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import design as d
import support as s

np, torch = s.np, s.torch
HERE, ROOT = s.HERE, s.ROOT


def freeze():
    s.configure()
    assert not (HERE / 'frozen.json').exists(), 'Nie nadpisuj zamrożenia'
    checked = s.read(HERE / 'check_cpu.json')
    assert checked['status'] == 'passed' and checked['code_hashes'] == s.code_hashes()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=s.UP, text=True).strip() == s.UP_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=s.UP, text=True).strip()
    historical = s.read(ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna/frozen.json')
    hashes = s.code_hashes()
    hashes[s.relative(HERE / 'check_cpu.json')] = s.digest(HERE / 'check_cpu.json')
    dependency = ROOT / 'experiments/2026-11-08_E038_operator_alokacja/pruning.py'
    hashes[s.relative(dependency)] = s.digest(dependency)
    hashes[s.relative(ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna/frozen.json')] = s.digest(
        ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna/frozen.json')
    for path in sorted(s.UP.rglob('*.py')):
        hashes[s.relative(path)] = s.digest(path)
    plans, indices, views = {}, {}, {}
    for ds in d.DATASETS:
        raw_t = historical['plans'][f'resnet_{ds}']['b50']['T']
        target = [raw_t[i] for i in d.PERM]
        plans[ds] = d.make_plan(ds, target)
        indices[ds], _ = s.split_indices(ds)
        images = s.Images(ds, indices[ds])
        images.inputs('A', augment=True)
        images.inputs('B')
        views[ds] = images.access
        for path in [s.SPLITS[ds]] + s.TRAIN_FILES[ds]:
            hashes[s.relative(path)] = s.digest(path)
            assert hashes[s.relative(path)] == historical['provenance'][s.relative(path)], path
        for seed in d.SEEDS:
            path = s.checkpoint(ds, seed)
            assert s.digest(path) == s.CP_SHA[ds][seed]
            hashes[s.relative(path)] = s.CP_SHA[ds][seed]
    assert not s.DENIED
    obj = dict(created_utc=s.utc(), experiment='E040', code_hashes=s.code_hashes(),
               source_hashes=hashes, plans=plans, indices=indices, views=views,
               upstream_commit=s.UP_COMMIT, group_order='semantic_E040_named_adapter',
               gpu_assignment={str(seed): seed - 42 for seed in d.SEEDS},
               limits=dict(seconds=7200, configurations_per_checkpoint=200,
                           cma_per_checkpoint=200, optimization_seconds_per_checkpoint=60),
               gates=dict(prediction_mae_ratio=.8, prediction_positive=3,
                          quality_pp=1., quality_positive=3, quality_min_pp=-.5,
                          interaction_pp=.5, bn_pp=.25),
               training=False, test_used=False, reserve_used=False, save_weights=False)
    s.write(HERE / 'frozen.json', obj)
    print('FREEZE PASS', s.digest(HERE / 'frozen.json'), flush=True)
    for ds in d.DATASETS:
        print(ds, 'TILT_PROJ', plans[ds]['references'][-1], flush=True)


def check_frozen():
    s.configure()
    f = s.read(HERE / 'frozen.json')
    assert f['code_hashes'] == s.code_hashes()
    s.validate_hashes(f['source_hashes'])
    for ds in d.DATASETS:
        assert f['plans'][ds] == d.make_plan(ds, f['plans'][ds]['historical_tilt'])
    assert not s.DENIED
    s.write(HERE / 'check.json', dict(status='passed', utc=s.utc(), code_hashes=s.code_hashes(),
                                     frozen_sha256=s.digest(HERE / 'frozen.json'),
                                     data_open_attempts=s.DENIED))
    print('FROZEN CHECK PASS', flush=True)


class Session:
    """Jeden checkpoint: sondy strumieniowo; tylko terminalne modele w RAM."""
    def __init__(self, run, frozen, dataset, seed):
        self.run, self.frozen, self.dataset, self.seed = run, frozen, dataset, seed
        self.path = run / f'seed_{seed}' / dataset
        self.path.mkdir(parents=True)
        self.plan = frozen['plans'][dataset]
        self.images = s.Images(dataset, frozen['indices'][dataset])
        self.a, self.b = self.images.inputs('A', True), self.images.inputs('B')
        assert self.images.access == frozen['views'][dataset]
        self.base = s.source_model(dataset, seed)
        self.base_hash = s.state_hash(self.base)
        z = s.predict(self.base.to('cuda'), self.b, d.cost(d.SIZES, dataset)[1])
        self.teacher_b = s.prediction_array(z)
        self.base.cpu()
        self.records, self.models, self.arrays_b = {}, {}, {}
        self.aliases, self.profiles, self.selections, self.tilt_stars = {}, {}, {}, {}
        self.constructed, self.cma_count = 1, 0
        s.write_arrays(self.path / 'arrays/B_teacher.npz', {'pred': self.teacher_b})
        s.write(self.path / 'teacher.json', dict(state_sha256=self.base_hash,
                checkpoint_sha256=s.CP_SHA[dataset][seed], params=d.cost(d.SIZES, dataset)[0],
                mac=d.cost(d.SIZES, dataset)[1]))

    def key(self, op, widths):
        return op + '_' + hashlib.sha256(','.join(map(str, widths)).encode()).hexdigest()[:20]

    def counters(self):
        assert self.constructed <= 200 and self.cma_count <= 200
        return dict(constructions=self.constructed, unique_configurations=len(self.records) + 1,
                    cma_passes=self.cma_count, retained_models=len(self.models))

    def measure(self, op, widths, keep=False, stage='B'):
        widths = tuple(map(int, widths))
        key = self.key(op, widths)
        if key in self.models:
            return key, self.models[key]
        self.constructed += 1
        self.counters()
        start = time.monotonic()
        make = s.fold if op == 'FOLD' else s.prune
        assert op in ('FOLD', 'MAG2')
        model, diag = make(self.base, widths)
        s.validate_shapes(self.base, model, widths, self.dataset)
        p, mac = d.cost(widths, self.dataset)
        pre_hash = s.state_hash(model)
        model.to('cuda')
        pre = s.prediction_array(s.predict(model, self.b, mac))
        calibration = s.calibrate(model, self.a)
        self.cma_count += 1
        self.counters()
        post = s.prediction_array(s.predict(model, self.b))
        model.cpu()
        state = s.state_hash(model)
        counts = dict(pre=int(np.count_nonzero(pre != self.teacher_b)),
                      post=int(np.count_nonzero(post != self.teacher_b)))
        record = dict(id=key, operator=op, widths=list(widths), params=p, mac=mac,
                      feasible=d.feasible(widths, self.dataset), pre_state_sha256=pre_hash,
                      state_sha256=state, counts=counts, calibration=calibration,
                      forward_geometry=model._e040_geometry,
                      parameters_sha256=s.state_hash(model, True),
                      arrays_sha256=None, diagnostics=diag,
                      construction_seconds=time.monotonic() - start, first_stage=stage)
        if key in self.records:
            old = self.records[key]
            assert old['counts'] == counts and old['state_sha256'] == state
            assert old['pre_state_sha256'] == pre_hash and old['parameters_sha256'] == record['parameters_sha256']
            assert np.array_equal(self.arrays_b[key]['pre'], pre) and np.array_equal(self.arrays_b[key]['post'], post)
            old.setdefault('replays', []).append(dict(state_sha256=state, counts=counts,
                                                       seconds=time.monotonic() - start))
            s.write(self.path / f'B/{key}.json', old)
        else:
            arrays = dict(pre=pre, post=post)
            file = self.path / f'arrays/B_{key}.npz'
            s.write_arrays(file, arrays)
            record['arrays_sha256'] = s.digest(file)
            self.arrays_b[key], self.records[key] = arrays, record
            s.write(self.path / f'B/{key}.json', record)
        if keep:
            self.models[key] = model
        assert s.state_hash(self.base) == self.base_hash
        return key, model

    def phase_b(self):
        optimization_seconds = 0.
        for op in ('FOLD', 'MAG2'):
            probe_keys = []
            anchor = tuple(self.plan['anchor']['widths'])
            known = {tuple(r['widths']) for r in self.plan['references']} | {anchor}
            for index, z in enumerate(d.PROBES):
                widths = d.widths_at(self.plan, z)
                key, model = self.measure(op, widths, keep=widths in known)
                probe_keys.append(key)
                del model
                if (index + 1) % 10 == 0 or index == 78:
                    print(self.seed, self.dataset, op, 'sondy', index + 1, '/79', flush=True)
            coefs = {mode: d.coefficients([self.records[k]['counts'][mode] for k in probe_keys])
                     for mode in ('pre', 'post')}
            for mode in ('pre', 'post'):
                assert [d.predict_counts(coefs[mode], z) for z in d.PROBES] == [
                    self.records[k]['counts'][mode] for k in probe_keys]
            self.profiles[op] = dict(probe_keys=probe_keys, coefficients=coefs,
                bn_pair_difference=[a - b for a, b in zip(coefs['post']['pair'], coefs['pre']['pair'])])
            for ref in self.plan['references']:
                key, _ = self.measure(op, ref['widths'], keep=True)
                self.aliases[f'{op}/{ref["name"]}'] = key
            key, _ = self.measure(op, anchor, keep=True)
            self.aliases[f'{op}/ANCHOR'] = key
            choose = [('ADD_POST', 'post', 1), ('PAIR_POST', 'post', 2)]
            if op == 'FOLD':
                choose.append(('PAIR_PRE', 'pre', 2))
            for name, mode, order in choose:
                result = d.allocate(self.plan, coefs[mode], order)
                optimization_seconds += result['seconds']
                assert optimization_seconds <= 60
                key, _ = self.measure(op, result['widths'], keep=True)
                self.aliases[f'{op}/{name}'] = key
                self.selections[f'{op}/{name}'] = dict(result, id=key)
            choices = [(self.records[self.aliases[f'{op}/{ref["name"]}']]['counts']['post'],
                        ref['mac'], -ref['params'], i, ref['name'])
                       for i, ref in enumerate(self.plan['references'])]
            self.tilt_stars[op] = min(choices)[-1]
        for name, key in self.aliases.items():
            r = self.records[key]
            assert r['feasible'] and s.state_hash(self.models[key]) == r['state_sha256'], name
        obj = dict(dataset=self.dataset, seed=self.seed, profiles=self.profiles,
                   selections=self.selections, aliases=self.aliases, tilt_stars=self.tilt_stars,
                   terminal_states={key: self.records[key]['state_sha256'] for key in sorted(set(self.aliases.values()))},
                   counters=self.counters(), optimization_seconds=optimization_seconds,
                   diagnostics=self.plan['diagnostics'], views=self.images.access,
                   c_accessed=False, dev_accessed=False, utc=s.utc())
        s.write(self.path / 'selection.json', obj)
        s.write(self.path / 'access_B.json', self.images.access)
        print(self.seed, self.dataset, 'B ZAMROŻONE', self.counters(), flush=True)

    def phase_c(self):
        assert (self.run / 'B_GATE.json').exists()
        selection_hash = s.digest(self.path / 'selection.json')
        self.images.phase = 'C'
        c = self.images.inputs('C')
        teacher = s.prediction_array(s.predict(self.base.to('cuda'), c))
        self.base.cpu()
        s.write_arrays(self.path / 'arrays/C_teacher.npz', {'pred': teacher})
        rows, done = [], {}

        def evaluate(key, model):
            if key in done:
                return done[key]
            before = s.state_hash(model)
            assert before == self.records[key]['state_sha256']
            pred = s.prediction_array(s.predict(model.to('cuda'), c))
            model.cpu()
            assert s.state_hash(model) == before
            file = self.path / f'arrays/C_{key}.npz'
            s.write_arrays(file, {'pred': pred})
            done[key] = int(np.count_nonzero(pred != teacher))
            s.write(self.path / f'C/{key}.json', dict(id=key, count=done[key], n=2048,
                    state_sha256=before, arrays_sha256=s.digest(file)))
            return done[key]

        for mask in self.plan['diagnostics']:
            key, model = self.measure('FOLD', mask['widths'], stage='C')
            actual = evaluate(key, model)
            coef = self.profiles['FOLD']['coefficients']['post']
            q1, q2 = d.predict_counts(coef, mask['z'], 1), d.predict_counts(coef, mask['z'], 2)
            rows.append(dict(z=mask['z'], id=key, actual_count=actual,
                             q1_count=q1, q2_count=q2, error1=abs(q1 - actual), error2=abs(q2 - actual)))
            del model
        for key in sorted(set(self.aliases.values())):
            evaluate(key, self.models[key])
        assert s.digest(self.path / 'selection.json') == selection_hash
        obj = dict(dataset=self.dataset, seed=self.seed, rows=rows,
                   error1_sum=sum(r['error1'] for r in rows),
                   error2_sum=sum(r['error2'] for r in rows), denominator=2048, masks=16,
                   counters=self.counters(), selection_sha256=selection_hash,
                   terminal_counts={key: done[key] for key in sorted(set(self.aliases.values()))},
                   dev_accessed=False, utc=s.utc())
        s.write(self.path / 'prediction.json', obj)
        s.write(self.path / 'access_C.json', self.images.access)
        print(self.seed, self.dataset, 'C GOTOWE', obj['error1_sum'], obj['error2_sum'], flush=True)

    def phase_dev(self):
        assert s.read(self.run / 'C_GATE.json')['passed']
        assert (self.run / 'DEV_GATE.json').exists()
        self.images.phase = 'dev'
        x, y = self.images.inputs('dev'), self.images.labels_dev()
        s.write_arrays(self.path / 'arrays/dev_labels.npz', {'labels': y.numpy().astype(np.uint8)})
        models = {'teacher': self.base} | {key: self.models[key] for key in sorted(set(self.aliases.values()))}
        metrics = {}
        for key, model in models.items():
            before = s.state_hash(model)
            if key != 'teacher':
                assert before == self.records[key]['state_sha256']
            z = s.predict(model.to('cuda'), x)
            model.cpu()
            assert s.state_hash(model) == before
            m, pred, ce = s.metric(z, y)
            if key == 'teacher':
                assert abs(m['top1'] - s.EXPECTED_TOP1[self.dataset][self.seed]) < 1e-12
            metrics[key] = m
            file = self.path / f'arrays/dev_{key}.npz'
            s.write_arrays(file, dict(pred=pred, ce=ce))
            s.write(self.path / f'dev/{key}.json', dict(metrics=m, state_sha256=before,
                                                       arrays_sha256=s.digest(file)))
        s.write(self.path / 'dev.json', dict(dataset=self.dataset, seed=self.seed, metrics=metrics,
                aliases=self.aliases, counters=self.counters(), utc=s.utc()))
        s.write(self.path / 'access_dev.json', self.images.access)


def wait_file(run, name):
    while not (run / name).exists():
        if (run / 'ABORT.json').exists():
            raise RuntimeError('Supervisor przerwał próbę')
        time.sleep(.5)
    return s.read(run / name)


def worker(run, gpu):
    s.configure(gpu)
    frozen = s.read(run / 'frozen.json')
    assert frozen['code_hashes'] == s.code_hashes()
    out = run / f'seed_{42 + gpu}'
    out.mkdir()
    status = dict(status='running', phase='B', seed=42 + gpu, gpu=gpu, pid=os.getpid(),
                  started_utc=s.utc(), gpu_name=torch.cuda.get_device_name(0))
    s.write(out / 'status.json', status)
    try:
        sessions = [Session(run, frozen, ds, 42 + gpu) for ds in d.DATASETS]
        for session in sessions:
            session.phase_b()
        s.write(out / 'B_ready.json', {v.dataset: s.digest(v.path / 'selection.json') for v in sessions})
        status['phase'] = 'waiting_B'
        s.write(out / 'status.json', status)
        wait_file(run, 'B_GATE.json')
        status['phase'] = 'C'
        s.write(out / 'status.json', status)
        for session in sessions:
            session.phase_c()
        s.write(out / 'C_ready.json', {v.dataset: s.digest(v.path / 'prediction.json') for v in sessions})
        status['phase'] = 'waiting_C'
        s.write(out / 'status.json', status)
        gate = wait_file(run, 'C_GATE.json')
        if gate['passed']:
            wait_file(run, 'DEV_GATE.json')
            status['phase'] = 'dev'
            s.write(out / 'status.json', status)
            for session in sessions:
                session.phase_dev()
        status.update(status='completed', phase='completed' if gate['passed'] else 'stopped_C',
                      prediction_gate_pass=gate['passed'], dev_used=gate['passed'],
                      counters={v.dataset: v.counters() for v in sessions},
                      data_open_attempts=s.DENIED, opened_data=s.OPENED_DATA)
        assert not s.DENIED
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc(), data_open_attempts=s.DENIED)
        raise
    finally:
        status['finished_utc'] = s.utc()
        s.write(out / 'status.json', status)


def telemetry():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)


def launch(gpus):
    assert gpus == [0, 1, 2, 3]
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and 'envs/attention' in sys.executable
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3' and torch.cuda.device_count() == 4
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    checked, frozen = s.read(HERE / 'check.json'), s.read(HERE / 'frozen.json')
    assert checked['status'] == 'passed' and checked['code_hashes'] == s.code_hashes() == frozen['code_hashes']
    assert checked['frozen_sha256'] == s.digest(HERE / 'frozen.json')
    s.validate_hashes(frozen['source_hashes'])
    info = telemetry()
    rows = info.strip().splitlines()
    assert len(rows) == 4
    for index, row in enumerate(rows):
        fields = [f.strip() for f in row.split(',')]
        assert int(fields[0]) == index and int(fields[3]) < 256 and int(fields[4]) == 0, info
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert not apps.strip(), apps
    run = ROOT / 'runs/E040' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + secrets.token_hex(4))
    (run / 'source').mkdir(parents=True)
    for name in frozen['source_hashes']:
        path = ROOT / name
        if path.suffix in ('.py', '.md'):
            dest = run / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    shutil.copy2(HERE / 'frozen.json', run / 'frozen.json')
    shutil.copy2(HERE / 'check_cpu.json', run / 'check_cpu.json')
    shutil.copy2(HERE / 'check.json', run / 'check.json')
    manifest = dict(status='running', run=s.relative(run), pid=os.getpid(), started_utc=s.utc(),
        limit_seconds=7200, frozen_sha256=s.digest(run / 'frozen.json'), code_hashes=s.code_hashes(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        python=platform.python_version(), executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu_start=info, workers=[], training=False, test_used=False, reserve_used=False, save_weights=False,
        environment={key: os.environ.get(key) for key in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'CUBLAS_WORKSPACE_CONFIG')},
        command=[sys.executable, '-B', s.relative(HERE / 'experiment.py'), 'run', '--gpus', '0', '1', '2', '3'])
    (run / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    s.write(run / 'manifest.json', manifest)
    print('RUN', run, 'PID', os.getpid(), flush=True)
    procs, handles = [], []
    start = time.monotonic()
    try:
        for gpu in gpus:
            log = (run / f'worker_{gpu}.log').open('w')
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            proc = subprocess.Popen([sys.executable, '-B', str(HERE / 'experiment.py'), 'worker',
                '--run', str(run), '--gpu', str(gpu)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(proc)
            manifest['workers'].append(dict(gpu=gpu, seed=42 + gpu, pid=proc.pid))
        s.write(run / 'manifest.json', manifest)
        with (run / 'monitor.jsonl').open('w') as monitor:
            while True:
                codes = [p.poll() for p in procs]
                elapsed = time.monotonic() - start
                assert all(c in (None, 0) for c in codes), codes
                if elapsed > 7200:
                    raise TimeoutError('E040: limit120min')
                if shutil.disk_usage(run).free < 200 * 1024 ** 2:
                    raise RuntimeError('E040: wyczerpanie miejsca na metryki')
                if not (run / 'B_GATE.json').exists() and all((run / f'seed_{s0}/B_ready.json').exists() for s0 in d.SEEDS):
                    selections = {str(s0): s.read(run / f'seed_{s0}/B_ready.json') for s0 in d.SEEDS}
                    for seed in d.SEEDS:
                        for ds in d.DATASETS:
                            assert s.digest(run / f'seed_{seed}/{ds}/selection.json') == selections[str(seed)][ds]
                    s.write(run / 'B_GATE.json', dict(passed=True, utc=s.utc(), selections=selections))
                    print('BARIERA B PASS', flush=True)
                if not (run / 'C_GATE.json').exists() and all((run / f'seed_{s0}/C_ready.json').exists() for s0 in d.SEEDS):
                    for seed in d.SEEDS:
                        hashes_c = s.read(run / f'seed_{seed}/C_ready.json')
                        for ds in d.DATASETS:
                            assert s.digest(run / f'seed_{seed}/{ds}/prediction.json') == hashes_c[ds]
                    cells = [s.read(run / f'seed_{seed}/{ds}/prediction.json') for seed in d.SEEDS for ds in d.DATASETS]
                    gate = d.prediction_gate(cells)
                    gate['utc'] = s.utc()
                    s.write(run / 'C_GATE.json', gate)
                    if gate['passed']:
                        s.write(run / 'DEV_GATE.json', dict(passed=True, c_gate_sha256=s.digest(run / 'C_GATE.json'), utc=s.utc()))
                    print('BRAMKA C', gate['passed'], [(v['dataset'], v['ratio'], v['positive']) for v in gate['cells']], flush=True)
                monitor.write(json.dumps(dict(utc=s.utc(), seconds=elapsed, codes=codes, gpu=telemetry())) + '\n')
                monitor.flush()
                if codes == [0] * 4:
                    break
                time.sleep(5)
        assert s.code_hashes() == frozen['code_hashes']
        s.validate_hashes(frozen['source_hashes'])
        gate_passed = s.read(run / 'C_GATE.json')['passed']
        manifest.update(status='completed', prediction_gate_pass=gate_passed,
                        execution_outcome='dev_completed' if gate_passed else 'stopped_C')
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        s.write(run / 'ABORT.json', dict(utc=s.utc(), reason=manifest['traceback']))
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        raise
    finally:
        for item, p in zip(manifest['workers'], procs):
            item['exit_code'] = p.poll()
        for h in handles:
            h.close()
        manifest.update(finished_utc=s.utc(), elapsed_seconds=time.monotonic() - start, gpu_end=telemetry())
        s.write(run / 'manifest.json', manifest)
        print(manifest['status'], s.relative(run), 'seconds', manifest['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'check', 'run', 'worker'))
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'check':
        check_frozen()
    elif args.action == 'run':
        launch(args.gpus)
    else:
        assert args.run is not None and args.gpu in range(4)
        worker(args.run, args.gpu)
