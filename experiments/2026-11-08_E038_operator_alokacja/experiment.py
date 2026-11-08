"""2026-11-08_E038_operator_alokacja: operator×profil na dev, patrz OPIS.md.
Wejście: checkpointy/profil E035; wyjście: MAG2, kontrole FOLD i interakcje.
Bez treningu; cztery niezależne zadania GPU, bez odczytu testu i rezerwy.
"""
import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime as rt
from pruning import keep_indices, prune

torch, np = rt.torch, rt.np


def freeze():
    assert not torch.cuda.is_available() and not (HERE / 'frozen.json').exists()
    torch.set_num_threads(2)
    frozen = rt.input_freeze()
    frozen['design'] = dict(operators=['FOLD', 'MAG2'], profiles=['U', 'T'], primary='I_b50_CMA')
    rt.write(HERE / 'frozen.json', frozen)
    print('FREEZE PASS', flush=True)


def shapes(model):
    return {n: list(t.shape) for n, t in model.state_dict().items()}


def check():
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    frozen = rt.read(HERE / 'frozen.json')
    rt.e35.validate_hashes(frozen['hashes'])
    # Test rzeczywistego znaczenia norm i reguły remisów.
    w = torch.tensor([[3., 0.], [2., 2.], [3., 0.]])
    assert keep_indices(w, 2, 2).tolist() == [0, 2]
    assert keep_indices(w, 2, 1).tolist() == [1, 0]
    from compression.mag_prune import ResNet18_MagnitudePruning, PreActResNet18_MagnitudePruning
    rows = []
    for pair in rt.PAIRS:
        base = rt.e35.model_source(pair, 42)
        before = rt.e35.state_hash(base)
        geo = rt.e37.geometry(pair, base)
        identity, _ = prune(base, pair, geo.sizes, rt)
        assert rt.e35.state_hash(identity) == before
        for tag in rt.TAGS:
            plan = frozen['plans'][pair][tag]
            for profile in ('U', 'T'):
                mag, diag = prune(base, pair, plan[profile], rt)
                fold, _ = rt.e35.fold(pair, base, plan, profile)
                assert shapes(mag) == shapes(fold), (pair, tag, profile)
                m1, m2 = rt.e37.mac(mag, pair), rt.e37.mac(fold, pair)
                assert m1 == m2
                assert sum(p.numel() for p in mag.parameters()) == plan['params'][profile]
                rows.append(dict(pair=pair, tag=tag, profile=profile, params=plan['params'][profile],
                                 mac=m1, shapes_equal=True, residual_forward_finite=True))
                if tag == 'b50' and profile == 'U':
                    cls = PreActResNet18_MagnitudePruning if pair.startswith('preact') else ResNet18_MagnitudePruning
                    lo = max(k / n for k, n in zip(plan['U'], geo.sizes))
                    hi = min((k + 1) / n for k, n in zip(plan['U'], geo.sizes))
                    assert lo < hi
                    keep = (lo + hi) / 2
                    assert [int(n * keep) for n in geo.sizes] == plan['U']
                    official = cls(copy.deepcopy(base), compression_ratio=1 - keep, p=2).apply().eval()
                    assert shapes(official) == shapes(mag)
                    for name, value in official.state_dict().items():
                        assert torch.equal(value, mag.state_dict()[name]), (pair, name)
                    print('UPSTREAM MAG2 EXACT', pair, flush=True)
        assert rt.e35.state_hash(base) == before
    assert not rt.DENIED
    rt.write(HERE / 'check.json', dict(status='passed', utc=rt.utc(), rows=rows,
        code_hashes=rt.hashes(HERE), frozen_sha256=rt.digest(HERE / 'frozen.json'), data_open_attempts=rt.DENIED))
    print('CPU PASS', len(rows), flush=True)


def worker(run, gpu):
    rt.configure(gpu)
    seed = 42 + gpu
    out = run / f'seed_{seed}'
    out.mkdir()
    status = dict(status='running', seed=seed, gpu=gpu, pid=rt.os.getpid(), started_utc=rt.utc(), completed=[])
    rt.write(out / 'status.json', status)
    frozen = rt.read(run / 'frozen.json')
    try:
        for pair in rt.PAIRS:
            ds = pair.split('_')[1]
            a, x, y, views = rt.e35.e31.data(ds, frozen['indices'][ds])
            base = rt.e35.model_source(pair, seed)
            base_hash = rt.e35.state_hash(base)
            controls, ctrl_ev = {}, {}
            for mode in rt.MODES:
                controls[mode], ctrl_ev[mode] = rt.e35.evaluate(pair, base, a, x, y, mode)
            ref_ctrl = frozen['references'][pair]['b50'][str(seed)]['controls']
            rt.check_reference(controls, ctrl_ev, ref_ctrl['metrics'], ref_ctrl['evidence'])
            rt.write(out / pair / 'controls.json', dict(metrics=controls, evidence=ctrl_ev, views_sha256=views))
            for tag in rt.TAGS:
                plan = frozen['plans'][pair][tag]
                ref = frozen['references'][pair][tag][str(seed)]
                assert views == ref['views_sha256']
                metrics, evidence, diagnostics, replay = {}, {}, {}, {}
                for profile in ('U', 'T'):
                    mag, diag = prune(base, pair, plan[profile], rt)
                    metrics[profile], evidence[profile], diagnostics[profile] = {}, {}, diag
                    for mode in rt.MODES:
                        metrics[profile][mode], evidence[profile][mode] = rt.e35.evaluate(pair, mag, a, x, y, mode)
                    if tag == 'b50':
                        folded, _ = rt.e35.fold(pair, base, plan, profile)
                        fm, fe = {}, {}
                        for mode in rt.MODES:
                            fm[mode], fe[mode] = rt.e35.evaluate(pair, folded, a, x, y, mode)
                        rt.check_reference(fm, fe, ref['metrics'][profile], ref['evidence'][profile])
                        replay[profile] = dict(metrics=fm, evidence=fe)
                        del folded
                    del mag
                assert rt.e35.state_hash(base) == base_hash
                rt.write(out / pair / tag / 'summary.json', dict(status='completed', pair=pair, tag=tag,
                    seed=seed, gpu=gpu, plan=plan, metrics=metrics, evidence=evidence, diagnostics=diagnostics,
                    replay=replay, views_sha256=views, checkpoint_sha256=rt.digest(rt.e35.checkpoint(pair, seed)),
                    phase='dev', bias_preserved=True, official_test_used=False))
                status['completed'].append(pair + '/' + tag)
                rt.write(out / 'status.json', status)
                print(seed, pair, tag, 'MAG2 CMA', {p: round(100 * metrics[p]['CMA']['top1'], 3) for p in ('U', 'T')}, flush=True)
            del a, x, y, base
        assert not rt.DENIED
        status.update(status='completed', finished_utc=rt.utc(), data_open_attempts=rt.DENIED)
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc(), data_open_attempts=rt.DENIED)
        raise
    finally:
        rt.write(out / 'status.json', status)


def analyze(run):
    manifest, frozen = rt.verify_run(run)
    rows, contrasts = [], []
    for pair in rt.PAIRS:
        for tag in rt.TAGS:
            cell = []
            for seed in rt.SEEDS:
                s = rt.read(run / f'seed_{seed}/{pair}/{tag}/summary.json')
                ref = frozen['references'][pair][tag][str(seed)]
                for mode in rt.MODES:
                    row = dict(pair=pair, tag=tag, seed=seed, mode=mode)
                    for op, source in [('F', ref['metrics']), ('M', s['metrics'])]:
                        for profile in ('U', 'T'):
                            v = source[profile][mode]
                            row[op + profile] = 100 * v['top1']
                            row[op + profile + '_nll'] = v['nll']
                    row['delta_F'] = row['FT'] - row['FU']
                    row['delta_M'] = row['MT'] - row['MU']
                    row['I'] = row['delta_F'] - row['delta_M']
                    rows.append(row)
                    cell.append(row)
            for mode in rt.MODES:
                vals = [r for r in cell if r['mode'] == mode]
                for key in ('delta_F', 'delta_M', 'I'):
                    st = rt.stats(r[key] for r in vals)
                    contrasts.append(dict(pair=pair, tag=tag, mode=mode, contrast=key, **st,
                        direction=st['mean'] > 0 and st['positive'] >= 3,
                        magnitude_1pp=st['mean'] >= 1 and st['positive'] >= 3))
    primary = [r for r in contrasts if r['tag'] == 'b50' and r['mode'] == 'CMA' and r['contrast'] == 'I']
    result = dict(utc=rt.utc(), rows=rows, contrasts=contrasts, primary=primary,
        primary_pass=all(r['direction'] for r in primary), evaluations_new=540, evaluations_reused=360,
        elapsed_seconds=manifest['elapsed_seconds'],
        source_hashes={str(p.relative_to(run)): rt.digest(p) for p in run.glob('seed_*/*/*/summary.json')})
    rt.write(run / 'analysis/analysis.json', result)
    rt.csv_write(run / 'analysis/per_seed.csv', rows)
    for r in primary:
        print(r['pair'], 'I', r['mean'], 'SD', r['sd'], 'positive', r['positive'], flush=True)
    print('PRIMARY', result['primary_pass'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['freeze', 'check', 'run', 'worker', 'analyze'])
    ap.add_argument('--run', type=Path)
    ap.add_argument('--gpu', type=int)
    args = ap.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'check':
        check()
    elif args.action == 'run':
        rt.launch(HERE, 'E038', dict(pairs=list(rt.PAIRS), tags=list(rt.TAGS), seeds=list(rt.SEEDS),
            modes=list(rt.MODES), batch=128, phase='dev', training=False, allocation='E035 frozen U/T'))
    elif args.action == 'analyze':
        analyze(args.run)
    else:
        worker(args.run, args.gpu)
