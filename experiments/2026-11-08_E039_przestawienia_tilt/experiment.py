"""2026-11-08_E039_przestawienia_tilt: stałe permutacje alokacji; OPIS.md.
Wejście: checkpointy E035, cztery permutacje; wyjście: plany i oceny dev4GPU.
Bez treningu, strojenia na jakości oraz dostępu do testu lub rezerwy.
"""
import argparse
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / '2026-11-08_E038_operator_alokacja'
sys.path.insert(0, str(SHARED))
import runtime as rt

PAIRS = rt.PAIRS[:3]
PERM_SEEDS = (101, 102, 103, 104)


def profile(geo, target, offsets):
    def widths(base):
        ratios = rt.np.clip(base + offsets, .1, .9)
        return [int(n * (1 - r)) for n, r in zip(geo.sizes, ratios)]
    lo, hi = -.3, 1.3
    assert geo.count(widths(lo)) >= target >= geo.count(widths(hi))
    for _ in range(60):
        mid = (lo + hi) / 2
        if geo.count(widths(mid)) > target:
            lo = mid
        else:
            hi = mid
    result = geo.fill(widths(hi), target)
    assert .999 * target <= geo.count(result) <= target
    return result


def freeze():
    assert not rt.torch.cuda.is_available() and not (HERE / 'frozen.json').exists()
    rt.torch.set_num_threads(2)
    frozen = rt.input_freeze(PAIRS)
    offsets = .3 * rt.np.linspace(-1, 1, 12)
    perms = {str(s): rt.np.random.default_rng(s).permutation(12).tolist() for s in PERM_SEEDS}
    assert len({tuple(p) for p in perms.values()}) == 4
    frozen['permutations'] = perms
    frozen['shuffled'] = {}
    frozen['costs'] = {}
    for pair in PAIRS:
        base = rt.e35.model_source(pair, 42)
        geo = rt.e37.geometry(pair, base)
        compute = rt.e37.mac_plan(pair, base, geo)
        frozen['shuffled'][pair], frozen['costs'][pair] = {}, {}
        for tag in rt.TAGS:
            plan = frozen['plans'][pair][tag]
            target = plan['params']['U']
            assert profile(geo, target, offsets) == plan['T']
            variants = {}
            for s in PERM_SEEDS:
                p = perms[str(s)]
                w = profile(geo, target, offsets[p])
                variants[str(s)] = dict(widths=w, params=geo.count(w), mac=compute(w), offsets=offsets[p].tolist())
            assert len({tuple(v['widths']) for v in variants.values()}) == 4
            frozen['shuffled'][pair][tag] = variants
            frozen['costs'][pair][tag] = dict(U=compute(plan['U']), T=compute(plan['T']),
                U_params=target, T_params=plan['params']['T'], group_sizes=geo.sizes, group_names=geo.names)
    rt.write(HERE / 'frozen.json', frozen)
    print('FREEZE PASS36', flush=True)


def check():
    assert not rt.torch.cuda.is_available()
    rt.torch.set_num_threads(2)
    frozen = rt.read(HERE / 'frozen.json')
    rt.e35.validate_hashes(frozen['hashes'])
    rows = []
    for pair in PAIRS:
        base = rt.e35.model_source(pair, 42)
        geo = rt.e37.geometry(pair, base)
        for tag in rt.TAGS:
            for s in PERM_SEEDS:
                v = frozen['shuffled'][pair][tag][str(s)]
                assert geo.count(v['widths']) == v['params']
                assert .999 * frozen['plans'][pair][tag]['params']['U'] <= v['params'] <= frozen['plans'][pair][tag]['params']['U']
                rows.append(dict(pair=pair, tag=tag, permutation=s, params=v['params'], mac=v['mac']))
        v = frozen['shuffled'][pair]['b50']['101']
        model, diag = rt.e37.fold_widths(pair, base, v['widths'])
        assert rt.e37.mac(model, pair) == v['mac']
        assert [d['k'] for d in diag] == v['widths']
        print('CPU FORWARD', pair, flush=True)
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
        for pair in PAIRS:
            a, x, y, views = rt.e35.e31.data(pair.split('_')[1], frozen['indices'][pair.split('_')[1]])
            base = rt.e35.model_source(pair, seed)
            before = rt.e35.state_hash(base)
            controls, evidence = {}, {}
            for mode in rt.MODES:
                controls[mode], evidence[mode] = rt.e35.evaluate(pair, base, a, x, y, mode)
            ctrl_ref = frozen['references'][pair]['b50'][str(seed)]['controls']
            rt.check_reference(controls, evidence, ctrl_ref['metrics'], ctrl_ref['evidence'])
            rt.write(out / pair / 'controls.json', dict(metrics=controls, evidence=evidence, views_sha256=views))
            for tag in rt.TAGS:
                ref = frozen['references'][pair][tag][str(seed)]
                assert views == ref['views_sha256']
                if tag == 'b50':
                    folded, _ = rt.e35.fold(pair, base, frozen['plans'][pair][tag], 'T')
                    fm, fe = {}, {}
                    for mode in rt.MODES:
                        fm[mode], fe[mode] = rt.e35.evaluate(pair, folded, a, x, y, mode)
                    rt.check_reference(fm, fe, ref['metrics']['T'], ref['evidence']['T'])
                    rt.write(out / pair / 'replay_T.json', dict(metrics=fm, evidence=fe))
                    del folded
                for s in PERM_SEEDS:
                    plan = frozen['shuffled'][pair][tag][str(s)]
                    folded, diag = rt.e37.fold_widths(pair, base, plan['widths'])
                    assert rt.e37.mac(folded, pair) == plan['mac']
                    metrics, ev = {}, {}
                    for mode in rt.MODES:
                        metrics[mode], ev[mode] = rt.e35.evaluate(pair, folded, a, x, y, mode)
                    assert rt.e35.state_hash(base) == before
                    rt.write(out / pair / tag / f'perm_{s}.json', dict(status='completed', pair=pair, tag=tag,
                        seed=seed, gpu=gpu, permutation=s, plan=plan, metrics=metrics, evidence=ev, clusters=diag,
                        views_sha256=views, phase='dev', checkpoint_sha256=rt.digest(rt.e35.checkpoint(pair, seed)),
                        bias_preserved=True, official_test_used=False))
                    print(seed, pair, tag, s, 'CMA', round(100 * metrics['CMA']['top1'], 3), flush=True)
                    del folded
                status['completed'].append(pair + '/' + tag)
                rt.write(out / 'status.json', status)
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
    for pair in PAIRS:
        for tag in rt.TAGS:
            for seed in rt.SEEDS:
                ref = frozen['references'][pair][tag][str(seed)]['metrics']
                for s in PERM_SEEDS:
                    summary = rt.read(run / f'seed_{seed}/{pair}/{tag}/perm_{s}.json')
                    for mode in rt.MODES:
                        v = summary['metrics'][mode]
                        rows.append(dict(pair=pair, tag=tag, seed=seed, permutation=s, mode=mode,
                            S=100 * v['top1'], T=100 * ref['T'][mode]['top1'], U=100 * ref['U'][mode]['top1'],
                            S_nll=v['nll'], T_nll=ref['T'][mode]['nll'], U_nll=ref['U'][mode]['nll']))
            for mode in rt.MODES:
                for which in ('mean4',) + PERM_SEEDS:
                    values, nll = [], []
                    for seed in rt.SEEDS:
                        selected = [r for r in rows if (r['pair'], r['tag'], r['seed'], r['mode']) == (pair, tag, seed, mode)
                                    and (which == 'mean4' or r['permutation'] == which)]
                        values.append(selected[0]['T'] - float(rt.np.mean([r['S'] for r in selected])))
                        nll.append(selected[0]['T_nll'] - float(rt.np.mean([r['S_nll'] for r in selected])))
                    st = rt.stats(values)
                    contrasts.append(dict(pair=pair, tag=tag, mode=mode, comparator=str(which), **st,
                        direction=st['mean'] > 0 and st['positive'] >= 3,
                        magnitude_1pp=st['mean'] >= 1 and st['positive'] >= 3, nll=rt.stats(nll)))
    primary = [r for r in contrasts if r['mode'] == 'CMA' and r['comparator'] == 'mean4']
    rt.write(run / 'analysis/analysis.json', dict(utc=rt.utc(), rows=rows, contrasts=contrasts, primary=primary,
        primary_pass=all(r['direction'] for r in primary), evaluations_new=504,
        elapsed_seconds=manifest['elapsed_seconds'],
        source_hashes={str(p.relative_to(run)): rt.digest(p) for p in run.glob('seed_*/*/*/perm_*.json')}))
    rt.csv_write(run / 'analysis/per_seed.csv', rows)
    for r in primary:
        print(r['pair'], r['tag'], r['mean'], 'SD', r['sd'], 'positive', r['positive'], flush=True)


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
        rt.launch(HERE, 'E039', dict(pairs=list(PAIRS), tags=list(rt.TAGS), seeds=list(rt.SEEDS),
            permutation_seeds=list(PERM_SEEDS), modes=list(rt.MODES), batch=128, phase='dev', training=False))
    elif args.action == 'analyze':
        analyze(args.run)
    else:
        worker(args.run, args.gpu)
