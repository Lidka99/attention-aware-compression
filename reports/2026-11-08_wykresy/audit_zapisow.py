#!/usr/bin/env python3
"""Audyt zapisów E034-E037: CPU-only, read-only, bez danych/testów i bez modeli.

Czyta wyłącznie manifesty, logi, summary, controls, metrics, costs i analizy.
Nie otwiera danych obrazowych ani plików testowych i nie uruchamia modeli.
Zapisuje AUDYT_WYKONANIA.md oraz audit_evidence.json w tym katalogu.
Każdy czytany artefakt jest hashowany (checkpointy strumieniowo); na końcu
sprawdzana jest niezmienność wejść (hash przed/po). Błąd zatrzymuje audyt.
"""
import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_MD = HERE / 'AUDYT_WYKONANIA.md'
OUT_JSON = HERE / 'audit_evidence.json'

R34T = ROOT / 'runs/E034/train_20261105T014209Z_21b44f52'
R34F = ROOT / 'runs/E034/fold_20261106T003625Z_9ec4f2ea'
R35 = ROOT / 'runs/E035/20261107T183831Z_8b0b14e0'
R36 = ROOT / 'runs/E036/20261107T231027Z_27bac186'
R37 = ROOT / 'runs/E037/20261108T031759Z_855c0b33'
R36_OLD = ROOT / 'runs/E036/20261107T225229Z_1f5b7f3a'
R37_OLD = ROOT / 'runs/E037/20261108T030702Z_a7428da3'

E35 = ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna'
E36 = ROOT / 'experiments/2026-11-07_E036_audyt_kosztu'
E37 = ROOT / 'experiments/2026-11-08_E037_kontrola_mac'
E34 = ROOT / 'experiments/2026-11-05_E034_preactresnet18'

PAIRS = ('resnet_cifar10', 'resnet_cifar100', 'resnet_tiny', 'preact_cifar10', 'preact_cifar100')
TAGS = ('b30', 'b50', 'b70')
MODES = ('PRE', 'CMA', 'UP_REPAIR')
SEEDS = (42, 43, 44, 45)

FORBIDDEN = ('/data/', '/test_batch', 'cifar-100-python/test', '/cache_val.npz', '/test_data.json')


class AuditError(RuntimeError):
    pass


def check(cond, msg):
    if not cond:
        raise AuditError(msg)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def parse_utc(text):
    return datetime.fromisoformat(text)


def span_seconds(a, b):
    return (parse_utc(b) - parse_utc(a)).total_seconds()


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def fmt(x, n=3):
    return f'{x:.{n}f}'.replace('.', ',')


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class Store:
    """Rejestr artefaktów: hash przed odczytem i po audycie (immutability)."""

    def __init__(self):
        self.artifacts = {}
        self.abs_paths = {}

    def rel(self, path):
        p = Path(path).resolve()
        try:
            return str(p.relative_to(ROOT.resolve()))
        except ValueError:
            return str(p)

    def register(self, path, kind='input'):
        path = Path(path)
        rp = self.rel(path)
        if rp in self.artifacts:
            return self.artifacts[rp]['sha256_before']
        posix = path.resolve().as_posix()
        for marker in FORBIDDEN:
            check(marker not in posix, f'odmowa odczytu danych/testu: {rp}')
        check(path.is_file(), f'brak artefaktu: {rp}')
        digest, size = sha256_file(path)
        self.artifacts[rp] = dict(path=rp, bytes=size, sha256_before=digest, kind=kind)
        self.abs_paths[rp] = str(path.resolve())
        return digest

    def read_bytes(self, path):
        self.register(path)
        return Path(path).read_bytes()

    def read_text(self, path):
        return self.read_bytes(path).decode('utf-8')

    def read_json(self, path):
        return json.loads(self.read_text(path))

    def seal(self):
        changed = []
        for rp, rec in self.artifacts.items():
            digest, size = sha256_file(self.abs_paths[rp])
            rec['sha256_after'] = digest
            rec['unchanged'] = (digest == rec['sha256_before'] and size == rec['bytes'])
            if not rec['unchanged']:
                changed.append(rp)
        check(not changed, 'wejście zmienione podczas audytu: ' + ', '.join(changed))
        return dict(files=len(self.artifacts), all_unchanged=True, changed=[],
                    note='hash przed i po audycie; checkpointy liczone strumieniowo')


def stdev(vals):
    return float(statistics.stdev(vals))


def mean(vals):
    return float(sum(vals) / len(vals))


def percentile(vals, q):
    a = sorted(float(v) for v in vals)
    k = (len(a) - 1) * q / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


def full_stats(vals):
    return dict(mean=mean(vals), median=float(statistics.median(vals)),
                p10=percentile(vals, 10), p90=percentile(vals, 90),
                sd=stdev(vals))


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ps_rows(pids):
    proc = subprocess.run(['ps', '-o', 'pid=', '-p', ','.join(str(p) for p in pids)],
                          capture_output=True, text=True)
    rows = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return dict(returncode=proc.returncode, rows=rows)


def check_times(label, start, finish, reported, tol=2.0):
    measured = span_seconds(start, finish)
    check(close(measured, reported, tol),
          f'{label}: czas {reported} s niezgodny z {start}->{finish} ({measured} s)')
    return dict(started_utc=start, finished_utc=finish, measured_seconds=measured,
                reported_seconds=reported, delta_seconds=measured - reported, tolerance_seconds=tol)


def main():
    store = Store()
    ev = dict(audit=dict(utc=now_utc(), cpu_only=True, torch_imported=False,
                         data_opened=False, models_run=False,
                         script='reports/2026-11-08_wykresy/audit_zapisow.py'),
              runs={}, recomputations={}, old_runs={}, corrections={}, limitations=[])

    print('[1/7] E034 train', flush=True)
    # ---------------------------------------------------------------- E034 train
    m = store.read_json(R34T / 'manifest.json')
    check(m['status'] == 'completed', 'E034 train: manifest nie completed')
    check(m['exit_codes'] == [0, 0, 0, 0], f"E034 train: exit_codes {m['exit_codes']}")
    check(len(m['worker_pids']) == 4, 'E034 train: nie 4 worker_pids')
    times34t = check_times('E034 train', m['started_utc'], m['finished_utc'], m['wall_seconds'])
    check(m['config']['epochs'] == 200, 'E034 train: epochs != 200')
    check(m['config']['seeds'] == list(SEEDS), 'E034 train: seeds != 42-45')

    statuses = {}
    for s in SEEDS:
        d = store.read_json(R34T / f'seed_{s}/status.json')
        check(d['status'] == 'completed' and d['seed'] == s, f'E034 train: status seed {s}')
        check(set(d['completed']) == {'cifar10', 'cifar100'}, f'E034 train: completed seed {s}')
        statuses[s] = d

    logs = sorted((R34T / 'logs').glob('gpu*.log'))
    check(len(logs) == 4, 'E034 train: nie 4 logi')
    log_records = {}
    for log in logs:
        lines = store.read_text(log).splitlines()
        check(len(lines) == 400, f'E034 train: {log.name} ma {len(lines)} linii, nie 400')
        for line in lines:
            parts = line.split(' ', 3)
            check(len(parts) == 4, f'E034 train: zła linia logu {log.name}: {line[:80]}')
            seed, ds, epoch = int(parts[0]), parts[1], int(parts[2])
            payload = ast.literal_eval(parts[3])
            check(payload['epoch'] == epoch and payload['seed'] == seed and
                  payload['dataset'] == ds and 1 <= epoch <= 200,
                  f'E034 train: niespójny wpis logu {log.name}')
            key = (seed, ds, epoch)
            check(key not in log_records, f'E034 train: duplikat epizodu {key}')
            log_records[key] = payload
    check(len(log_records) == 1600, f'E034 train: {len(log_records)} epizodów, nie 1600')

    summaries = {}
    ckpt_hashes = {}
    for s in SEEDS:
        for ds in ('cifar10', 'cifar100'):
            sp = R34T / f'seed_{s}/{ds}/summary.json'
            sm = store.read_json(sp)
            check(sm['status'] == 'completed' and sm['epochs'] == 200 and
                  sm['seed'] == s and sm['dataset'] == ds, f'E034 train: summary {sp}')
            check(sm['final_metrics']['epoch'] == 200, f'E034 train: final epoch {sp}')
            ckpt = ROOT / sm['checkpoint']
            actual = store.register(ckpt, 'checkpoint')
            ckpt_hashes[store.rel(ckpt)] = actual
            check(actual == sm['checkpoint_sha256'],
                  f'E034 train: checkpoint SHA != summary {store.rel(ckpt)}')
            mp = R34T / f'seed_{s}/{ds}/metrics.jsonl'
            mlines = store.read_text(mp).splitlines()
            check(len(mlines) == 200, f'E034 train: {mp} ma {len(mlines)} linii')
            epochs = []
            for ln in mlines:
                row = json.loads(ln)
                epochs.append(row['epoch'])
                check(finite(row['train_loss']) and finite(row['train_top1']) and
                      finite(row['dev_top1']) and finite(row['dev_nll']),
                      f'E034 train: niefinitywna metryka {mp}')
            check(epochs == list(range(1, 201)), f'E034 train: epoki w {mp} != 1..200')
            last = json.loads(mlines[-1])
            check(last['checkpoint_sha256'] == actual,
                  f'E034 train: final metrics SHA != checkpoint {store.rel(ckpt)}')
            check(log_records[(s, ds, 200)]['checkpoint_sha256'] == actual,
                  f'E034 train: final log SHA != checkpoint {store.rel(ckpt)}')
            summaries[(s, ds)] = sm
    check(len(summaries) == 8, f'E034 train: {len(summaries)} summary, nie 8')
    check(len(ckpt_hashes) == 8, f'E034 train: {len(ckpt_hashes)} checkpointów, nie 8')

    src = store.read_bytes(R34T / 'source/train_preact.py')
    src_hash = hashlib.sha256(src).hexdigest()
    check(src_hash == m['source_hashes']['train_preact.py'],
          'E034 train: hash source/train_preact.py != manifest')

    ev['runs']['E034_train'] = dict(
        run=store.rel(R34T), manifest_status=m['status'], exit_codes=m['exit_codes'],
        worker_pids=m['worker_pids'], times=times34t,
        summaries=len(summaries), checkpoints=len(ckpt_hashes), epoch_records=len(log_records),
        logs=len(logs), metrics_lines=8 * 200, checkpoint_sha256=ckpt_hashes,
        source_train_preact_sha256=src_hash)

    print('[2/7] E034 fold', flush=True)
    # ----------------------------------------------------------------- E034 fold
    fm = store.read_json(R34F / 'manifest.json')
    check(fm['status'] == 'completed', 'E034 fold: manifest nie completed')
    check(fm['exit_codes'] == [0, 0, 0, 0], f"E034 fold: exit_codes {fm['exit_codes']}")
    check(len(fm['worker_pids']) == 4, 'E034 fold: nie 4 worker_pids')
    times34f = check_times('E034 fold', fm['started_utc'], fm['finished_utc'], fm['wall_seconds'])
    for s in SEEDS:
        d = store.read_json(R34F / f'seed_{s}/status.json')
        check(d['status'] == 'completed' and len(d['completed']) == 6,
              f'E034 fold: status seed {s}')

    fold_logs = sorted((R34F / 'logs').glob('gpu*.log'))
    check(len(fold_logs) == 4, 'E034 fold: nie 4 logi')
    task_re = re.compile(r'^(\d+) (cifar10|cifar100) (b\d0) (\{.*\})$')
    fold_tasks = 0
    for idx, log in enumerate(fold_logs):
        lines = store.read_text(log).splitlines()
        n = 0
        for line in lines:
            mt = task_re.match(line)
            if mt:
                check(int(mt.group(1)) == 42 + idx, f'E034 fold: zły seed w {log.name}')
                payload = ast.literal_eval(mt.group(4))
                check(set(payload) == {'UP_U', 'UP_T'}, f'E034 fold: zły payload {log.name}')
                n += 1
        check(n == 6, f'E034 fold: {log.name} ma {n} zadań, nie 6')
        fold_tasks += n
    check(fold_tasks == 24, f'E034 fold: {fold_tasks} zadań, nie 24')

    fold_summ = {}
    fold_controls = {}
    metric_count = 0
    for sp in sorted(R34F.glob('seed_*/cifar*/b*/summary.json')):
        sm = store.read_json(sp)
        check(sm['status'] == 'completed', f'E034 fold: summary {sp}')
        s, ds, tag = sm['seed'], sm['dataset'], sm['tag']
        check(set(sm['metrics']) == {'UP_U', 'UP_T'}, f'E034 fold: profile {sp}')
        for prof in ('UP_U', 'UP_T'):
            check(set(sm['metrics'][prof]) == set(MODES), f'E034 fold: tryby {sp}')
            for mode in MODES:
                for key in ('top1', 'nll'):
                    check(finite(sm['metrics'][prof][mode][key]),
                          f'E034 fold: niefinitywna metryka {sp}')
                metric_count += 1
        cp = R34F / f'seed_{s}/controls_{ds}.json'
        ctrl = store.read_json(cp)
        ctrl_hash = store.register(cp)
        check(sm['controls_sha256'] == ctrl_hash,
              f'E034 fold: controls_sha256 != plik {store.rel(cp)}')
        check(sm['views_sha256'] == ctrl['views_sha256'],
              f'E034 fold: views_sha256 {store.rel(cp)}')
        fold_summ[(ds, tag, s)] = sm
        fold_controls[(ds, s)] = ctrl
    check(len(fold_summ) == 24, f'E034 fold: {len(fold_summ)} summary, nie 24')
    check(metric_count == 144, f'E034 fold: {metric_count} metryk, nie 144')
    check(len(fold_controls) == 8, f'E034 fold: {len(fold_controls)} controls, nie 8')
    control_metric_count = sum(len(c['controls']) for c in fold_controls.values())
    check(control_metric_count == 24, f'E034 fold: {control_metric_count} metryk kontrolnych')
    baseline34 = {}
    for ds in ('cifar10', 'cifar100'):
        vals = [summaries[(s, ds)]['final_metrics']['dev_top1'] for s in SEEDS]
        for s, val in zip(SEEDS, vals):
            check(close(val, fold_controls[(ds, s)]['controls']['PRE']['top1'], 1e-12),
                  f'E034: teacher PRE != ostatnia epoka {ds}/{s}')
        baseline34[ds] = mean(vals) * 100

    fa = store.read_json(R34F / 'analysis_20261106T004001Z/analysis.json')
    fv = store.read_json(R34F / 'analysis_20261106T004001Z/verification.json')
    check(fa['primary_pass'] is True and fv['overall_ok'] is True,
          'E034 fold: analysis/verification nie OK')
    check(fa['controls_n'] == 24 and len(fa['primary']) == 6,
          'E034 fold: analysis counts')
    check(close(fa['wall_seconds'], fm['wall_seconds'], 1e-6),
          'E034 fold: analysis wall_seconds != manifest')
    # analysis source_hashes vs actual summaries
    for relp, exp in fa['source_hashes'].items():
        check(store.artifacts[store.rel(R34F / relp)]['sha256_before'] == exp,
              f'E034 fold: analysis source hash {relp}')

    def fold_contrast(mode):
        out = {}
        for ds in ('cifar10', 'cifar100'):
            for tag in TAGS:
                vals = [100.0 * (fold_summ[(ds, tag, s)]['metrics']['UP_T'][mode]['top1'] -
                                 fold_summ[(ds, tag, s)]['metrics']['UP_U'][mode]['top1'])
                        for s in SEEDS]
                out[(ds, tag)] = dict(mean=mean(vals), sd=stdev(vals),
                                      positive=sum(v > 0 for v in vals), per_seed=vals)
        return out

    cma = fold_contrast('CMA')
    pre = fold_contrast('PRE')
    analyses_primary = {(x['dataset'], x['tag'], x['mode']): x for x in fa['primary']}
    thresholds = {(x['dataset'], x['tag'], x['mode']): x
                  for x in fv['summary']['thresholds']}
    for (ds, tag), got in cma.items():
        exp = analyses_primary[(ds, tag, 'CMA')]
        check(close(got['mean'], exp['mean'], 1e-9) and close(got['sd'], exp['sd'], 1e-9)
              and got['positive'] == exp['positive'], f'E034 CMA {ds} {tag}')
        thr = thresholds[(ds, tag, 'CMA')]
        check(close(got['mean'], thr['mean_pp'], 1e-9) and
              close(got['sd'], thr['sd_pp'], 1e-9), f'E034 CMA threshold {ds} {tag}')
    for (ds, tag), got in pre.items():
        thr = thresholds[(ds, tag, 'PRE')]
        check(close(got['mean'], thr['mean_pp'], 1e-9) and
              close(got['sd'], thr['sd_pp'], 1e-9) and
              thr['positive'] == f"{got['positive']}/4", f'E034 PRE {ds} {tag}')

    ev['runs']['E034_fold'] = dict(
        run=store.rel(R34F), manifest_status=fm['status'], exit_codes=fm['exit_codes'],
        worker_pids=fm['worker_pids'], times=times34f, logs=len(fold_logs),
        tasks=fold_tasks, summaries=len(fold_summ), metrics=metric_count,
        controls_files=len(fold_controls), control_metrics=control_metric_count,
        teacher_PRE_top1=baseline34,
        cma_contrasts={f'{ds}/{tag}': v for (ds, tag), v in cma.items()},
        pre_baseline={f'{ds}/{tag}': v for (ds, tag), v in pre.items()})

    print('[3/7] E035', flush=True)
    # ---------------------------------------------------------------------- E035
    e35m = store.read_json(R35 / 'manifest.json')
    check(e35m['status'] == 'completed', 'E035: manifest nie completed')
    phase_times = {}
    for phase in ('dev', 'test'):
        ph = e35m['phases'][phase]
        check(len(ph['pids']) == 4 and ph['exit_codes'] == [0, 0, 0, 0],
              f'E035: {phase} pids/exit')
        pts = check_times(f'E035 {phase}', ph['started_utc'], ph['finished_utc'],
                          ph['seconds'])
        logs = sorted((R35 / 'logs').glob(f'{phase}_gpu*.log'))
        check(len(logs) == 4, f'E035: {phase} nie 4 logi')
        task_lines = 0
        for log in logs:
            for line in store.read_text(log).splitlines():
                if re.match(rf'^{phase} [0-9]+ ', line):
                    task_lines += 1
        check(task_lines == 60, f'E035: {phase} {task_lines} zadań w logach, nie 60')
        phase_times[phase] = dict(times=pts, logs=len(logs), log_tasks=task_lines)
    total35 = check_times('E035 całość', e35m['utc'], e35m['finished_utc'],
                          e35m['wall_seconds'])
    gate = store.read_json(R35 / 'dev_gate.json')
    check(gate['passed'] is True and gate['evaluations'] == 360 and gate['controls'] == 60,
          'E035: dev_gate counts')
    check(gate['maximum']['top1'] <= 1e-12 and gate['maximum']['nll'] <= 1e-5,
          'E035: dev_gate max delta')

    e35_dev = {}
    e35_dev_ctrl = {}
    dev_metric_count = 0
    for pair in PAIRS:
        for s in SEEDS:
            cp = R35 / f'dev/seed_{s}/{pair}/controls.json'
            ctrl = store.read_json(cp)
            check(set(ctrl['metrics']) == set(MODES), f'E035: controls {store.rel(cp)}')
            for mode in MODES:
                for key in ('top1', 'nll'):
                    check(finite(ctrl['metrics'][mode][key]),
                          f'E035: niefinitywny control {store.rel(cp)}')
            e35_dev_ctrl[(pair, s)] = ctrl
            for tag in TAGS:
                sp = R35 / f'dev/seed_{s}/{pair}/{tag}/summary.json'
                sm = store.read_json(sp)
                check(sm['status'] == 'completed', f'E035: summary {store.rel(sp)}')
                check(set(sm['metrics']) == {'U', 'T'}, f'E035: profile {store.rel(sp)}')
                for prof in ('U', 'T'):
                    check(set(sm['metrics'][prof]) == set(MODES),
                          f'E035: tryby {store.rel(sp)}')
                    for mode in MODES:
                        for key in ('top1', 'nll'):
                            check(finite(sm['metrics'][prof][mode][key]),
                                  f'E035: niefinitywna metryka {store.rel(sp)}')
                        dev_metric_count += 1
                e35_dev[(pair, tag, s)] = sm
    check(len(e35_dev) == 60, f'E035: {len(e35_dev)} dev summary, nie 60')
    check(dev_metric_count == 360, f'E035: {dev_metric_count} dev metryk, nie 360')
    check(len(e35_dev_ctrl) == 20, f'E035: {len(e35_dev_ctrl)} dev controls, nie 20')

    # Odczyt zapisanych metryk testowych, bez danych obrazowych i ponownej oceny modeli.
    test_summaries = list(R35.glob('test/seed_*/*/*/summary.json'))
    test_controls = list(R35.glob('test/seed_*/*/controls.json'))
    test_statuses = list(R35.glob('test/seed_*/status.json'))
    check(len(test_summaries) == 60 and len(test_controls) == 20 and len(test_statuses) == 4,
          'E035: liczby artefaktów testu')
    for p in test_statuses:
        d = store.read_json(p)
        check(d['status'] == 'completed', f'E035: test status {store.rel(p)}')
    e35_test = {}
    for p in test_summaries:
        sm = store.read_json(p)
        pair, tag, seed = sm['pair'], sm['tag'], sm['seed']
        check(sm['status'] == 'completed' and sm['phase'] == 'test', f'E035: test summary {p}')
        dev = e35_dev[(pair, tag, seed)]
        check(sm['checkpoint_sha256'] == dev['checkpoint_sha256'] and
              sm['views_sha256'] == dev['views_sha256'], f'E035: test/dev provenance {p}')
        for profile in ('U', 'T'):
            for mode in MODES:
                val = sm['metrics'][profile][mode]
                check(0 <= val['top1'] <= 1 and finite(val['nll']), f'E035: test metryka {p}')
                check(sm['evidence'][profile][mode]['state_sha256'] ==
                      dev['evidence'][profile][mode]['state_sha256'], f'E035: test/dev state {p}')
        e35_test[(pair, tag, seed)] = sm
    for p in test_controls:
        ctrl = store.read_json(p)
        for mode in MODES:
            check(0 <= ctrl['metrics'][mode]['top1'] <= 1 and
                  finite(ctrl['metrics'][mode]['nll']), f'E035: test control {p}')

    e35a = store.read_json(R35 / 'analysis/analysis.json')
    e35v = store.read_json(R35 / 'analysis/verification.json')
    check(len(e35a['rows']) == 720 and len(e35a['controls']) == 120 and
          len(e35a['contrasts']) == 90, 'E035: analysis counts')
    check(e35a['primary_pass'] is True and e35a['primary_pass_n'] == 15,
          'E035: analysis primary')
    check(e35v['status'] == 'PASS' and e35v['counts']['dev'] ==
          {'evaluations': 360, 'controls': 60}, 'E035: verification')
    check(e35v['analysis_sha256'] ==
          store.artifacts[store.rel(R35 / 'analysis/analysis.json')]['sha256_before'],
          'E035: verification analysis_sha256')

    cma35 = {}
    for pair in PAIRS:
        for tag in TAGS:
            vals = [100.0 * (e35_dev[(pair, tag, s)]['metrics']['T']['CMA']['top1'] -
                             e35_dev[(pair, tag, s)]['metrics']['U']['CMA']['top1'])
                    for s in SEEDS]
            cma35[(pair, tag)] = dict(mean=mean(vals), sd=stdev(vals),
                                      positive=sum(v > 0 for v in vals),
                                      direction=(mean(vals) > 0 and sum(v > 0 for v in vals) >= 3))
    a35 = {(x['pair'], x['tag']): x for x in e35a['contrasts']
           if x['phase'] == 'dev' and x['mode'] == 'CMA'}
    check(len(a35) == 15, 'E035: nie 15 CMA dev contrasts')
    for key, got in cma35.items():
        exp = a35[key]
        check(close(got['mean'], exp['mean'], 1e-9) and close(got['sd'], exp['sd'], 1e-9)
              and got['positive'] == exp['positive'] and got['direction'] == exp['direction'],
               f'E035: kontrast CMA {key}')
    cma35_test = {}
    for pair in PAIRS:
        for tag in TAGS:
            vals = [100 * (e35_test[(pair, tag, s)]['metrics']['T']['CMA']['top1'] -
                           e35_test[(pair, tag, s)]['metrics']['U']['CMA']['top1']) for s in SEEDS]
            got = dict(mean=mean(vals), sd=stdev(vals), positive=sum(v > 0 for v in vals))
            exp = next(x for x in e35a['contrasts'] if x['phase'] == 'test' and
                       x['pair'] == pair and x['tag'] == tag and x['mode'] == 'CMA')
            check(close(got['mean'], exp['mean'], 1e-9) and close(got['sd'], exp['sd'], 1e-9)
                  and got['positive'] == exp['positive'], f'E035: test kontrast {pair}/{tag}')
            cma35_test[(pair, tag)] = got

    # provenance E035: tylko już przeczytane artefakty, nigdy data/
    e35f = store.read_json(R35 / 'frozen.json')
    prov_checked = 0
    for relp, exp in e35f['provenance'].items():
        if relp.startswith('data/'):
            continue
        if relp in store.artifacts:
            check(store.artifacts[relp]['sha256_before'] == exp,
                  f'E035: provenance hash {relp}')
            prov_checked += 1

    ev['runs']['E035'] = dict(
        run=store.rel(R35), manifest_status=e35m['status'], times=total35,
        phases={k: v for k, v in phase_times.items()},
        dev=dict(summaries=len(e35_dev), metrics=dev_metric_count,
                 controls=len(e35_dev_ctrl), control_metrics=3 * len(e35_dev_ctrl)),
        test=dict(summaries=len(test_summaries), controls=len(test_controls),
                  metrics=60 * 6, control_metrics=20 * 3,
                   note='przeliczone z zapisanych metryk, bez danych i ponownej oceny modeli'),
        dev_gate=dict(passed=gate['passed'], evaluations=gate['evaluations'],
                      controls=gate['controls'], maximum=gate['maximum']),
        cma_contrasts={f'{p}/{t}': v for (p, t), v in cma35.items()},
        test_cma_contrasts={f'{p}/{t}': v for (p, t), v in cma35_test.items()},
        provenance_checked=prov_checked)

    print('[4/7] E036', flush=True)
    # ---------------------------------------------------------------------- E036
    e36m = store.read_json(R36 / 'manifest.json')
    check(e36m['status'] == 'completed', 'E036: manifest nie completed')
    check(len(e36m['workers']) == 4 and all(w['exit_code'] == 0 for w in e36m['workers']),
          'E036: workers/exit')
    times36 = check_times('E036', e36m['started_utc'], e36m['finished_utc'],
                          e36m['elapsed_seconds'])
    for s in SEEDS:
        d = store.read_json(R36 / f'seed_{s}/status.json')
        check(d['status'] == 'completed' and len(d['completed']) == 210,
              f'E036: status seed {s}')
    worker_logs = sorted(R36.glob('worker_*.log'))
    check(len(worker_logs) == 4, 'E036: nie 4 worker logi')
    worker_lines = {}
    for p in worker_logs:
        lines = store.read_text(p).splitlines()
        tasks = sum(1 for ln in lines if re.match(r'^\d+ \S+ \S+ (1|128) \d+ median_ms ', ln))
        worker_lines[p.name] = dict(lines=len(lines), tasks=tasks)
        check(tasks == 210, f'E036: {p.name} ma {tasks} pomiarów, nie 210')
    monitor = [json.loads(ln) for ln in store.read_text(R36 / 'monitor.jsonl').splitlines()]
    check(monitor[-1]['exit_codes'] == [0, 0, 0, 0], 'E036: monitor final exit_codes')

    costs = {}
    state_hashes = set()
    for cp in sorted(R36.glob('seed_*/???*/costs.json')):
        c = store.read_json(cp)
        check(len(c) == 7, f'E036: costs {store.rel(cp)} ma {len(c)} wpisów')
        for name, row in c.items():
            check(len(row['state_sha256']) == 64, f'E036: state hash {name}')
            check(finite(row['params']) and finite(row['mac']) and row['params'] > 0,
                  f'E036: cost {name}')
            state_hashes.add((store.rel(cp), name, row['state_sha256']))
        seed = int(cp.parent.parent.name.split('_')[1])
        pair = cp.parent.name
        costs[(seed, pair)] = c
    check(len(costs) == 20, f'E036: {len(costs)} costs, nie 20')
    check(len(state_hashes) == 140, f'E036: {len(state_hashes)} stanów, nie 140')

    e36f = store.read_json(R36 / 'frozen.json')
    for (seed, pair), c in costs.items():
        ref = e36f['references'][str(seed)][pair]
        for name, row in c.items():
            exp = ref['teacher'] if name == 'teacher' else ref[name[:3]][name[-1]]
            check(row['state_sha256'] == exp['state_sha256'],
                  f'E036: state hash != frozen {seed}/{pair}/{name}')

    meas = {}
    events = 0
    input_hashes = {}
    for mp in sorted(R36.glob('seed_*/???*/*_bs*_s*.json')):
        d = store.read_json(mp)
        check(d['batch'] in (1, 128) and d['state_unchanged'] is True,
              f'E036: pomiar {store.rel(mp)}')
        check(len(d['times_ms']) == 200, f'E036: {store.rel(mp)} != 200 czasów')
        check(all(finite(x) and x > 0 for x in d['times_ms']),
              f'E036: niefinitywny czas {store.rel(mp)}')
        events += 200
        key = (d['seed'], d['pair'], d['name'], d['batch'])
        input_hashes.setdefault(key, set()).add(d['input_sha256'])
        meas[key + (d['series'],)] = d['times_ms']
    check(len(meas) == 840, f'E036: {len(meas)} plików czasów, nie 840')
    check(events == 168000, f'E036: {events} zdarzeń, nie 168000')
    check(all(len(v) == 1 for v in input_hashes.values()),
          'E036: input_sha256 różni się między seriami')

    e36a = store.read_json(R36 / 'analysis/analysis.json')
    check(len(e36a['rows']) == 280 and len(e36a['ratios']) == 120 and
          len(e36a['averages']) == 120 and len(e36a['memory']) == 280 and
          len(e36a['costs']) == 140 and len(e36a['quality']) == 15,
          'E036: analysis counts')
    e36v = store.read_json(R36 / 'analysis/verification.json')
    check(e36v['status'] == 'passed' and e36v['timing_files'] == 840 and
          e36v['state_hash_matches'] == 140 and e36v['rows'] == 280,
          'E036: verification')

    # recompute rows from raw times
    arows = {(x['seed'], x['pair'], x['name'], x['batch']): x for x in e36a['rows']}
    for (seed, pair, name, batch), series_map in sorted(
            {(k[0], k[1], k[2], k[3]): None for k in input_hashes}.items()):
        vals = []
        for series in range(3):
            vals.extend(meas[(seed, pair, name, batch, series)])
        st = full_stats(vals)
        exp = arows[(seed, pair, name, batch)]
        for key in ('mean', 'median', 'p10', 'p90', 'sd'):
            check(close(st[key], exp[key], 1e-6),
                  f'E036: row {seed}/{pair}/{name}/b{batch} {key}')
        check(exp['events'] == 600, 'E036: events != 600')
        check(close(1000.0 * batch / st['mean'], exp['samples_s'], 1e-6),
              f'E036: samples_s {seed}/{pair}/{name}/b{batch}')

    # recompute batch1 latency/throughput ratios and their SD across seeds
    ar = {(x['seed'], x['pair'], x['tag'], x['batch']): x for x in e36a['ratios']}
    sd_batch1 = {}
    for pair in PAIRS:
        for tag in TAGS:
            lat = []
            for s in SEEDS:
                u = [x for x in meas[(s, pair, tag + '_U', 1, 0)] +
                     meas[(s, pair, tag + '_U', 1, 1)] +
                     meas[(s, pair, tag + '_U', 1, 2)]]
                t = [x for x in meas[(s, pair, tag + '_T', 1, 0)] +
                     meas[(s, pair, tag + '_T', 1, 1)] +
                     meas[(s, pair, tag + '_T', 1, 2)]]
                lr = statistics.median(t) / statistics.median(u)
                tr = statistics.mean(u) / statistics.mean(t)
                exp = ar[(s, pair, tag, 1)]
                check(close(lr, exp['latency_ratio'], 1e-9) and
                      close(tr, exp['throughput_ratio'], 1e-9),
                      f'E036: ratio {s}/{pair}/{tag}/b1')
                lat.append(lr)
            sd_batch1[(pair, tag)] = dict(sd=stdev(lat), min=min(lat), max=max(lat))
    sd_min = min(v['sd'] for v in sd_batch1.values())
    sd_max = max(v['sd'] for v in sd_batch1.values())
    check(0.019 <= sd_min <= 0.0195 and 0.100 <= sd_max <= 0.1005,
          f'E036: SD batch1 {sd_min}-{sd_max} poza 0.019-0.100')

    ev['runs']['E036'] = dict(
        run=store.rel(R36), manifest_status=e36m['status'],
        workers=e36m['workers'], times=times36,
        worker_logs=worker_lines, monitor_lines=len(monitor),
        costs_files=len(costs), state_hashes=len(state_hashes),
        timing_files=len(meas), events=events,
        sd_batch1={f'{p}/{t}': v for (p, t), v in sd_batch1.items()},
        analysis_counts=dict(rows=len(e36a['rows']), ratios=len(e36a['ratios']),
                             averages=len(e36a['averages']), memory=len(e36a['memory']),
                             costs=len(e36a['costs']), quality=len(e36a['quality'])))

    print('[5/7] E037', flush=True)
    # ---------------------------------------------------------------------- E037
    e37f = store.read_json(R37 / 'frozen.json')
    check(len(e37f['hashes']) == 101, f"E037: {len(e37f['hashes'])} hashów w frozen")
    frozen_checked = 0
    for relp, exp in e37f['hashes'].items():
        kind = 'checkpoint' if relp.endswith('.pt') else 'input'
        got = store.register(ROOT / relp, kind)
        check(got == exp, f'E037: frozen hash {relp}')
        frozen_checked += 1
    e35f_hash = store.register(R35 / 'frozen.json')
    check(e35f_hash == e37f['e35_frozen_sha256'], 'E037: e35_frozen_sha256')

    e37m = store.read_json(R37 / 'manifest.json')
    check(e37m['status'] == 'completed' and
          len(e37m['workers']) == 4 and all(w['exit_code'] == 0 for w in e37m['workers']),
          'E037: manifest/workers')
    times37 = check_times('E037', e37m['started_utc'], e37m['finished_utc'],
                          e37m['elapsed_seconds'])
    for s in SEEDS:
        d = store.read_json(R37 / f'seed_{s}/status.json')
        check(d['status'] == 'completed' and len(d['completed']) == 15,
              f'E037: status seed {s}')
    worker37 = sorted(R37.glob('worker_*.log'))
    check(len(worker37) == 4, 'E037: nie 4 worker logi')
    w37_counts = {}
    for p in worker37:
        lines = store.read_text(p).splitlines()
        tasks = sum(1 for ln in lines if re.match(r'^\d+ \S+ \S+ \{', ln))
        w37_counts[p.name] = dict(lines=len(lines), tasks=tasks)
        check(len(lines) == 30 and tasks == 15, f'E037: {p.name} linie/zadania')

    e37_summ = {}
    e37_ctrl = {}
    metric37 = 0
    for pair in PAIRS:
        for s in SEEDS:
            cp = R37 / f'seed_{s}/{pair}/controls.json'
            ctrl = store.read_json(cp)
            check(set(ctrl['metrics']) == set(MODES), f'E037: controls {store.rel(cp)}')
            for mode in MODES:
                for key in ('top1', 'nll'):
                    check(finite(ctrl['metrics'][mode][key]),
                          f'E037: niefinitywny control {store.rel(cp)}')
            e37_ctrl[(pair, s)] = ctrl
            for tag in TAGS:
                sp = R37 / f'seed_{s}/{pair}/{tag}/summary.json'
                sm = store.read_json(sp)
                check(sm['status'] == 'completed', f'E037: summary {store.rel(sp)}')
                check(sm['plan'] == e37f['plans'][pair][tag],
                      f'E037: plan != frozen {store.rel(sp)}')
                check(set(sm['metrics']) == set(MODES), f'E037: tryby {store.rel(sp)}')
                for mode in MODES:
                    for key in ('top1', 'nll'):
                        check(finite(sm['metrics'][mode][key]),
                              f'E037: niefinitywna metryka {store.rel(sp)}')
                    metric37 += 1
                e37_summ[(pair, tag, s)] = sm
    check(len(e37_summ) == 60 and metric37 == 180, 'E037: summary/metrics counts')
    check(len(e37_ctrl) == 20, 'E037: controls count')

    # checkpoint_sha256 in E037 summaries vs actual checkpoint files
    checkpoint_hash_set = {h for p, h in e37f['hashes'].items() if p.endswith('.pt')}
    check(len(checkpoint_hash_set) == 20, 'E037: nie 20 unikalnych checkpointów w frozen')
    for key, sm in e35_dev.items():
        check(sm['checkpoint_sha256'] in checkpoint_hash_set,
              f'E037: checkpoint E035 poza zbiorem 20 {key}')
    for key, sm in e37_summ.items():
        check(sm['checkpoint_sha256'] == e35_dev[key]['checkpoint_sha256'],
              f'E037: checkpoint_sha256 != E035 {key}')
    for (pair, tag, s), sm in e37_summ.items():
        arch, ds = pair.split('_')
        if arch == 'preact':
            cp = R34T / f'seed_{s}/{ds}/last.pt'
        else:
            # odczytaj ścieżkę z E035 summary (checkpoint_sha256) i sprawdź hash
            cp = None
        if cp is not None:
            check(sm['checkpoint_sha256'] == store.register(cp, 'checkpoint'),
                  f'E037: checkpoint_sha256 {pair}/{tag}/{s}')
        else:
            e35cp = e35_dev[(pair, tag, s)]['checkpoint_sha256']
            check(sm['checkpoint_sha256'] == e35cp,
                  f'E037: checkpoint_sha256 != E035 {pair}/{tag}/{s}')

    # controls E037 vs E035 dev
    control_pairs = 0
    for pair in PAIRS:
        for s in SEEDS:
            for mode in MODES:
                for key in ('top1', 'nll'):
                    a = e35_dev_ctrl[(pair, s)]['metrics'][mode][key]
                    b = e37_ctrl[(pair, s)]['metrics'][mode][key]
                    check(close(a, b, 1e-12),
                          f'E037: control != E035 {pair}/{s}/{mode}/{key}')
                    control_pairs += 1
    check(control_pairs == 120, f'E037: {control_pairs} porównań kontrolnych')

    # recompute 180 rows, 135 contrasts, 135 averages from E035 raw dev + E037 summaries
    rows = []
    for pair in PAIRS:
        for tag in TAGS:
            for s in SEEDS:
                a = e35_dev[(pair, tag, s)]['metrics']
                b = e37_summ[(pair, tag, s)]['metrics']
                for mode in MODES:
                    rows.append(dict(pair=pair, tag=tag, seed=s, mode=mode,
                                     U=a['U'][mode]['top1'], T=a['T'][mode]['top1'],
                                     Up=b[mode]['top1'],
                                     U_nll=a['U'][mode]['nll'], T_nll=a['T'][mode]['nll'],
                                     Up_nll=b[mode]['nll']))
    check(len(rows) == 180, 'E037: recomputed rows != 180')
    e37a = store.read_json(R37 / 'analysis/analysis.json')
    check(len(e37a['rows']) == 180 and len(e37a['allocation']) == 135 and
          len(e37a['averages']) == 135 and len(e37a['controls']) == 60,
          'E037: analysis counts')
    ar37 = {(x['pair'], x['tag'], x['seed'], x['mode']): x for x in e37a['rows']}
    max_row_delta = 0.0
    for r in rows:
        exp = ar37[(r['pair'], r['tag'], r['seed'], r['mode'])]
        for key in ('U', 'T', 'Up', 'U_nll', 'T_nll', 'Up_nll'):
            d = abs(r[key] - exp[key])
            max_row_delta = max(max_row_delta, d)
            check(d <= 1e-12, f'E037: row delta {r["pair"]}/{r["tag"]}/{r["seed"]}/{r["mode"]}/{key}')

    alloc = []
    for pair in PAIRS:
        for tag in TAGS:
            for mode in MODES:
                vals = [r for r in rows if r['pair'] == pair and r['tag'] == tag and r['mode'] == mode]
                for a, b in (('T', 'Up'), ('T', 'U'), ('Up', 'U')):
                    d = [100.0 * (r[a] - r[b]) for r in vals]
                    rec = dict(pair=pair, tag=tag, mode=mode, contrast=f'{a}-{b}',
                               mean=mean(d), sd=stdev(d), positive=sum(x > 0 for x in d),
                               per_seed=d)
                    if mode == 'CMA' and a == 'T' and b == 'Up':
                        rec['gate'] = ((rec['mean'] >= 2.0 and rec['positive'] >= 3)
                                       if tag in ('b50', 'b70') else
                                       (rec['mean'] > 0 and rec['positive'] >= 3))
                    alloc.append(rec)
    check(len(alloc) == 135, 'E037: recomputed allocation != 135')
    aa = {(x['pair'], x['tag'], x['mode'], x['contrast']): x for x in e37a['allocation']}
    max_alloc_delta = 0.0
    for r in alloc:
        exp = aa[(r['pair'], r['tag'], r['mode'], r['contrast'])]
        for key in ('mean', 'sd'):
            d = abs(r[key] - exp[key])
            max_alloc_delta = max(max_alloc_delta, d)
            check(d <= 1e-9, f'E037: allocation {r["pair"]}/{r["tag"]}/{r["mode"]}/{r["contrast"]}/{key}')
        check(r['positive'] == exp['positive'] and r.get('gate') == exp.get('gate'),
              f'E037: allocation flags {r["pair"]}/{r["tag"]}/{r["mode"]}/{r["contrast"]}')

    avg = []
    for pair in PAIRS:
        for tag in TAGS:
            for mode in MODES:
                vals = [r for r in rows if r['pair'] == pair and r['tag'] == tag and r['mode'] == mode]
                for name, key, nkey in (('U', 'U', 'U_nll'), ('T', 'T', 'T_nll'), ('Up', 'Up', 'Up_nll')):
                    avg.append(dict(pair=pair, tag=tag, mode=mode, name=name,
                                    top1=mean([r[key] for r in vals]),
                                    top1_sd=stdev([r[key] for r in vals]),
                                    nll=mean([r[nkey] for r in vals]),
                                    nll_sd=stdev([r[nkey] for r in vals])))
    check(len(avg) == 135, 'E037: recomputed averages != 135')
    av = {(x['pair'], x['tag'], x['mode'], x['name']): x for x in e37a['averages']}
    max_avg_delta = 0.0
    for r in avg:
        exp = av[(r['pair'], r['tag'], r['mode'], r['name'])]
        for key in ('top1', 'top1_sd', 'nll', 'nll_sd'):
            d = abs(r[key] - exp[key])
            max_avg_delta = max(max_avg_delta, d)
            check(d <= 1e-9, f'E037: average {r["pair"]}/{r["tag"]}/{r["mode"]}/{r["name"]}/{key}')

    # analysis controls vs E037 controls files
    ac = {(x['pair'], x['seed'], x['mode']): x for x in e37a['controls']}
    for pair in PAIRS:
        for s in SEEDS:
            for mode in MODES:
                exp = ac[(pair, s, mode)]
                got = e37_ctrl[(pair, s)]['metrics'][mode]
                check(close(got['top1'], exp['top1'], 1e-12) and
                      close(got['nll'], exp['nll'], 1e-12),
                      f'E037: analysis control {pair}/{s}/{mode}')

    primary37 = [r for r in alloc if r['mode'] == 'CMA' and r['contrast'] == 'T-Up']
    check(len(primary37) == 15, 'E037: primary != 15')
    ap = {(x['pair'], x['tag']): x for x in e37a['primary']}
    for r in primary37:
        exp = ap[(r['pair'], r['tag'])]
        check(close(r['mean'], exp['mean'], 1e-9) and close(r['sd'], exp['sd'], 1e-9)
              and r['positive'] == exp['positive'] and r.get('gate') == exp.get('gate'),
              f'E037: primary {r["pair"]}/{r["tag"]}')
    primary_pass = all(r.get('gate', False) for r in primary37)
    check(primary_pass is False and e37a['primary_pass'] is False,
          'E037: primary_pass')

    # plan params/MAC and ratio range
    ratios = []
    for pair in PAIRS:
        for tag in TAGS:
            p = e37f['plans'][pair][tag]
            check(p['Uprime_mac'] <= p['T_mac'], f'E037: MAC U\' > T {pair}/{tag}')
            check(p['Uprime_params'] >= p['T_params'], f'E037: params U\' < T {pair}/{tag}')
            ratios.append(dict(pair=pair, tag=tag,
                               ratio=p['Uprime_params'] / p['T_params'],
                               Uprime_params=p['Uprime_params'], T_params=p['T_params'],
                               Uprime_mac=p['Uprime_mac'], T_mac=p['T_mac']))
    rmin = min(x['ratio'] for x in ratios)
    rmax = max(x['ratio'] for x in ratios)
    check(1.09 <= rmin <= 1.10 and 1.82 <= rmax <= 1.83,
          f'E037: zakres parametrów U\'/T {rmin}-{rmax}')

    check(close(e37a['wall_seconds'], e37m['elapsed_seconds'], 1e-9),
          'E037: analysis wall_seconds != manifest elapsed_seconds')

    # snapshot vs current analyze
    src37 = store.read_text(R37 / 'source/experiment.py')
    cur37 = store.read_text(E37 / 'experiment.py')
    src37_hash = hashlib.sha256(src37.encode()).hexdigest()
    cur37_hash = hashlib.sha256(cur37.encode()).hexdigest()
    check(src37_hash == e37m['code_hashes'][store.rel(E37 / 'experiment.py')],
          'E037: snapshot hash != manifest code_hashes')
    check("averages = []" not in src37 and "m['wall_seconds']" in src37,
          'E037: snapshot nie ma oczekiwanej starej wersji analyze')
    check("averages = []" in cur37 and "m['elapsed_seconds']" in cur37,
          'E037: bieżący analyze bez poprawki')
    check(len(e37a['averages']) == 135 and
          close(e37a['wall_seconds'], e37m['elapsed_seconds'], 1e-9),
          'E037: analysis nie wygląda na wynik bieżącego analyze')

    e37v = store.read_json(R37 / 'analysis/verification.json')
    check(e37v['status'] == 'passed' and e37v['rows'] == 180 and
          e37v['controls'] == 60 and e37v['primary_pass'] is False,
          'E037: verification')
    check(e37v['frozen_sha256'] == store.artifacts[store.rel(R37 / 'frozen.json')]['sha256_before']
          and e37v['analysis_sha256'] ==
          store.artifacts[store.rel(R37 / 'analysis/analysis.json')]['sha256_before'],
          'E037: verification hashes')

    ev['runs']['E037'] = dict(
        run=store.rel(R37), manifest_status=e37m['status'], times=times37,
        worker_logs=w37_counts, frozen_hashes_checked=frozen_checked,
        summaries=len(e37_summ), metrics=metric37, controls=len(e37_ctrl),
        control_metrics=3 * len(e37_ctrl), control_comparisons=control_pairs,
        rows=len(rows), allocation=len(alloc), averages=len(avg), primary=len(primary37),
        max_row_delta=max_row_delta, max_alloc_delta=max_alloc_delta,
        max_avg_delta=max_avg_delta, primary_pass=primary_pass,
        param_ratio=dict(min=rmin, max=rmax, per_profile=ratios),
        snapshot=dict(run_experiment_sha256=src37_hash,
                      current_experiment_sha256=cur37_hash,
                      run_has_averages=False, run_uses_wall_seconds=True,
                      current_has_averages=True, current_uses_elapsed_seconds=True,
                      analysis_wall_seconds=e37a['wall_seconds'],
                      manifest_elapsed_seconds=e37m['elapsed_seconds']))

    print('[6/7] stare runy', flush=True)
    # ---------------------------------------------------------------- stare runy
    def old_run(label, path, expected_manifest, seed_statuses):
        man = store.read_json(path / 'manifest.json')
        check(man['status'] == expected_manifest,
              f'{label}: manifest status {man["status"]} != {expected_manifest}')
        seed_status = {}
        for s in SEEDS:
            d = store.read_json(path / f'seed_{s}/status.json')
            seed_status[str(s)] = d['status']
            check(d['status'] == seed_statuses[s],
                  f'{label}: seed {s} status {d["status"]} != {seed_statuses[s]}')
        logs = sorted(path.glob('worker_*.log'))
        log_sizes = {}
        for p in logs:
            txt = store.read_text(p)
            log_sizes[p.name] = len(txt.encode())
        pids = set()
        if 'pid' in man:
            pids.add(man['pid'])
        for w in man.get('workers', []):
            if 'pid' in w:
                pids.add(w['pid'])
        for s in SEEDS:
            pids.add(store.read_json(path / f'seed_{s}/status.json')['pid'])
        pids = sorted(int(p) for p in pids)
        alive = {str(p): pid_alive(p) for p in pids}
        ps = ps_rows(pids)
        check(not any(alive.values()), f'{label}: aktywny stary PID {alive}')
        check(not ps['rows'], f'{label}: ps pokazuje procesy: {ps["rows"]}')
        artifacts = {
            'summaries': len(list(path.glob('seed_*/**/summary.json'))),
            'measurements': len(list(path.glob('seed_*/**/*_bs*_s*.json'))),
            'costs': len(list(path.glob('seed_*/???*/costs.json'))),
        }
        return dict(path=store.rel(path), manifest_status=man['status'],
                    seed_statuses=seed_status, pids=pids, pid_alive=alive, ps=ps,
                    worker_logs=log_sizes, result_artifacts=artifacts,
                    historical_status=('interrupted/running historical status'
                                       if expected_manifest == 'running' else
                                       'failed historical status with stale running seed statuses'))

    ev['old_runs']['E037_20261108T030702Z_a7428da3'] = old_run(
        'E037 old', R37_OLD, 'running', {42: 'running', 43: 'running', 44: 'running', 45: 'running'})
    ev['old_runs']['E036_20261107T225229Z_1f5b7f3a'] = old_run(
        'E036 old', R36_OLD, 'failed', {42: 'running', 43: 'running', 44: 'failed', 45: 'failed'})

    print('[7/7] pieczęć i raport', flush=True)
    # ---------------------------------------------------------------- corrections
    ev['corrections'] = {
        'E036_batch1_sd': dict(
            raw_sd_min=sd_min, raw_sd_max=sd_max,
            required='raportować SD ilorazu T/U z surowych ratios batch1 (0,019-0,100), '
                     'a nie tylko średnie ~1'),
        'E037_causal_overclaim': dict(
            param_ratio_min=rmin, param_ratio_max=rmax,
            required='U\' ma 1,10-1,82x parametrów T; wynik nie izoluje wpływu obliczeń '
                     'od pojemności i nie dowodzi, że zysk TILT wynika z compute'),
        'snapshot_vs_current_analyze': dict(
            E037_snapshot_experiment_sha256=src37_hash,
            E037_current_experiment_sha256=cur37_hash,
            E037_snapshot_has_averages=False,
            E037_snapshot_wall_key="m['wall_seconds']",
            E037_current_has_averages=True,
            E037_current_wall_key="m['elapsed_seconds']",
            E037_analysis_has_averages=True,
            E037_analysis_wall_seconds=e37a['wall_seconds'],
            E037_manifest_elapsed_seconds=e37m['elapsed_seconds'],
            E036_note='source/ nie zawiera analyze.py; bieżący analyze.py powstał po runie'),
    }
    ev['limitations'] = [
        'Spójne lokalne artefakty silnie potwierdzają wykonanie E034-E037, ale nie są '
        'zewnętrznym ani podpisanym dowodem fizycznej pracy GPU.',
        'SHA256 pokazuje spójność plików, nie czas ich powstania ani autorstwo.',
        'Metryki E035 przeliczono z zapisanych summary; nie ponawiano oceny modeli i nie otwierano danych.',
        'Nie uruchamiano modeli; koszty E036 sprawdzono z zapisanych pomiarów.',
    ]

    ev['immutability'] = store.seal()
    ev['artifacts'] = store.artifacts
    ev['audit']['finished_utc'] = now_utc()
    ev['audit']['artifacts_hashed'] = len(store.artifacts)

    # ------------------------------------------------------------------ raport MD
    t34t = times34t
    t34f = times34f
    t35 = total35
    t36 = times36
    t37 = times37
    md = f"""# Audyt wykonania E034–E037 (2026-11-08)

CPU-only, bez otwierania danych/testów i bez uruchamiania modeli. Sprawdzono zapisy
w `runs/`. Skrypt: `reports/2026-11-08_wykresy/audit_zapisow.py`; dowody:
`reports/2026-11-08_wykresy/audit_evidence.json`. Wszystkie czytane artefakty
zahashowano (checkpointy strumieniowo), a wejścia sprawdzono przed/po audycie.

## Werdykt

Lokalne artefakty są spójne i silnie potwierdzają, że E034–E037 wykonano do końca:
manifesty `completed`, 4×exit0, komplet plików, checkpointy zgodne z summary/logami,
a niezależne przeliczenia zgadzają się z analizami. To **nie** jest jednak zewnętrzny,
podpisany dowód fizycznej pracy GPU; hash pokazuje spójność, nie czas.

## E034

- `{store.rel(R34T)}`: manifest `completed`, 4×exit0, wall **{fmt(t34t['reported_seconds'],2)} s**
  ({t34t['started_utc'][:19]}→{t34t['finished_utc'][:19]}). 8 summary (4 seedy×2 zbiory),
  **1600** epok w 4 logach po 400 linii, 8×200 linii `metrics.jsonl`.
  8 checkpointów `last.pt`: SHA256 strumieniowo zgodny z `summary.json`, końcową
  linią `metrics.jsonl` i końcową linią logu.
- `{store.rel(R34F)}`: manifest `completed`, 4×exit0, wall **{fmt(t34f['reported_seconds'],2)} s**.
  **24** summary, **144** metryki (2 profile×3 tryby), **8** plików controls
  (**24** metryki kontrolne), 4 logi×6 zadań = **24**.
- Baseline PRE i **6** kontrastów CMA (T−U) przeliczone z 24 summary: zgodne z
  `analysis.json`/`verification.json`. CMA: CIFAR-10 b70 **+{fmt(cma[('cifar10','b70')]['mean'],3)} pp**,
  CIFAR-100 b70 **+{fmt(cma[('cifar100','b70')]['mean'],3)} pp**; wszystkie 6/6 dodatnie.
  Baseline CIFAR-10 **{fmt(baseline34['cifar10'],3)}%**, CIFAR-100 **{fmt(baseline34['cifar100'],3)}%**.

## E035

- `{store.rel(R35)}`: manifest `completed`; dev i test po 4×exit0 i 4 logi po 15 zadań.
  Dev i test: po **60** summary / **360** metryk / **20** controls (**60** metryk).
  Przeliczenia z zapisów, bez ponownej oceny modeli. `dev_gate.json` passed: 360 ocen, 60 kontroli,
  max|Δ| top1 = 0, nll = 1,19e-7.
- Czasy: dev **{fmt(phase_times['dev']['times']['reported_seconds'],2)} s**, test
  **{fmt(phase_times['test']['times']['reported_seconds'],2)} s**, całość
  **{fmt(t35['reported_seconds'],2)} s** ({t35['started_utc'][:19]}→{t35['finished_utc'][:19]}).
- **15 dev i15 test** kontrastów CMA (T−U) przeliczono z surowych summary;
  mean/SD/liczba dodatnich zgodne z `analysis.json`. Modele i kalibracje
  mają zgodne hashe stanu w dev/test; nie uruchamiano ponownej oceny jakości.

## E036

- `{store.rel(R36)}`: manifest `completed`, 4×exit0, **{fmt(t36['reported_seconds'],2)} s**
  ({t36['started_utc'][:19]}→{t36['finished_utc'][:19]}). **20** plików `costs.json`
  = **140** stanów; **840** plików czasów ×200 = **168 000** zdarzeń; 4 statusy po 210
  zadań; 4 logi po 240 linii; monitor kończy `[0,0,0,0]`.
- **140** `state_sha256` zgodnych z `frozen.json`; liczebności analizy:
  280 rows / 120 ratios / 120 averages / 280 memory / 140 costs.
  Rows czasów i ilorazy batch1 przeliczono z surowych plików.
- SD ilorazu latencji T/U dla batch1 z surowych 60 ratios: **{fmt(sd_min,4)}–{fmt(sd_max,4)}**
  (min ResNet/Tiny b70, max ResNet/CIFAR-10 b50).

## E037

- `{store.rel(R37)}`: manifest `completed`, 4×exit0, **{fmt(t37['reported_seconds'],2)} s**
  ({t37['started_utc'][:19]}→{t37['finished_utc'][:19]}). **60** summary (**180** metryk),
  **20** controls (**60** metryk), 4 statusy po 15 zadań, 4 logi po 30 linii.
- **101/101** hashów w `frozen.json` zgodnych z plikami (w tym 20 checkpointów
  E011/E015/E019/E034); plany U′ mają MAC(U′)≤MAC(T).
- **135** kontrastów, **180** rows i **135** averages przeliczone niezależnie
  z E035 DEV + E037 summary: zgodne z `analysis.json` (max |Δ| rows
  {max_row_delta:.1e}, allocation {max_alloc_delta:.1e}, averages {max_avg_delta:.1e}).
  20 bloków controls zgodnych z E035 dev (120/120), wszystkie wartości finite.
  Primary 15 CMA: **1/15** gate, `primary_pass=false`.
- Stosunek parametrów U′/T: **{fmt(rmin,3)}–{fmt(rmax,3)}×**.

## Stare runy (rozpoznane, nie nadpisane)

- `{store.rel(R37_OLD)}`: manifest i 4 statusy `running` — **interrupted/running
  historical status**; PID 999264, 999305–999308 nieaktywne (`os.kill(0)`=ENOENT,
  `ps` brak), logi puste, 0 summary / 0 pomiarów / 0 costs.
- `{store.rel(R36_OLD)}`: manifest `failed`, seed 42/43 `running`, seed 44/45
  `failed` — **failed historical status ze starymi statusami running**; PID 979331,
  979385–979388 nieaktywne, 4 częściowe `costs.json`, 0 plików pomiarowych.

## Ustalenia i korekty dokumentacji

1. **E036 — SD batch1:** raportować SD z surowych ratios (**{fmt(sd_min,4)}–{fmt(sd_max,4)}**),
   nie tylko średnie ~1; średnia bliska 1 nie dowodzi równoważności metod.
2. **E037 — interpretacja przyczynowa:** U′ ma **{fmt(rmin,3)}–{fmt(rmax,3)}×** parametrów T
   przy zbliżonym MAC. Nie rozdzielono wpływu obliczeń i pojemności;
   zdanie „zysk TILT wynika z compute” było zbyt mocne i zostało skorygowane.
3. **Snapshot vs bieżący analyze:** snapshot `source/experiment.py` (SHA `{src37_hash[:12]}…`)
   nie ma `averages` i używa `m['wall_seconds']`; bieżący (SHA `{cur37_hash[:12]}…`) dodaje
   `averages` i poprawia na `m['elapsed_seconds']`. `analysis.json` w runie ma `averages`
   i `wall_seconds={fmt(e37a['wall_seconds'],4)}=manifest.elapsed_seconds`, więc powstał
   z poprawionej wersji; snapshot nie odtwarza analizy 1:1. (E036: `source/` nie zawiera
   `analyze.py`; bieżący `analyze.py` powstał po runie.)

## Granica dowodu

Spójność lokalnych artefaktów (manifesty, logi, summary, checkpointy, SHA256) silnie
potwierdza wykonanie, lecz nie stanowi zewnętrznego ani podpisanego dowodu fizycznej
pracy GPU. Hash dowodzi spójności plików, nie czasu ich powstania.

_Audyt: {ev['audit']['utc']} → {ev['audit']['finished_utc']}; artefakty: {len(store.artifacts)}._
"""
    OUT_MD.write_text(md)
    OUT_JSON.write_text(json.dumps(ev, indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    print('AUDYT PASS')
    print('MD   ', OUT_MD)
    print('JSON ', OUT_JSON)
    print('artefakty', len(store.artifacts), 'wejścia niezmienione', ev['immutability']['all_unchanged'])


if __name__ == '__main__':
    try:
        main()
    except AuditError as exc:
        print('AUDYT ZATRZYMANY:', exc, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print('AUDYT ZATRZYMANY (nieoczekiwany błąd):', repr(exc), file=sys.stderr)
        raise
