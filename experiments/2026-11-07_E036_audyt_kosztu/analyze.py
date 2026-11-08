"""E036: agregacja pomiarów kosztu z runu; wejście run, wyjście analysis.json/CSV. OPIS.md.
Nie otwiera danych, nie liczy nowej jakości; jakość CMA tylko z zamrożonych E035.
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PAIRS = ('resnet_cifar10', 'resnet_cifar100', 'resnet_tiny', 'preact_cifar10', 'preact_cifar100')
TAGS = ('b30', 'b50', 'b70')
SEEDS = tuple(range(42, 46))
BATCHES = (1, 128)
NAMES = ('teacher',) + tuple(t + '_' + p for t in TAGS for p in ('U', 'T'))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main(run):
    run = Path(run)
    m = read(run / 'manifest.json')
    assert m['status'] == 'completed' and [w['exit_code'] for w in m['workers']] == [0] * 4
    frozen = read(run / 'frozen.json')
    rows = []
    cost_rows = []
    for seed in SEEDS:
        status = read(run / f'seed_{seed}/status.json')
        assert status['status'] == 'completed' and len(status['completed']) == 210
        for pair in PAIRS:
            costs = read(run / f'seed_{seed}/{pair}/costs.json')
            assert tuple(costs) == NAMES
            ref = frozen['references'][str(seed)][pair]
            for name in NAMES:
                expected = ref['teacher'] if name == 'teacher' else ref[name[:3]][name[-1]]
                assert costs[name]['state_sha256'] == expected['state_sha256']
                cost_rows.append(dict(seed=seed, pair=pair, name=name, params=costs[name]['params'],
                                      mac=costs[name]['mac'],
                                      mac_per_param=costs[name]['mac'] / costs[name]['params']))
            for batch in BATCHES:
                for name in NAMES:
                    times = []
                    memory = []
                    hashes = set()
                    for series in range(3):
                        p = run / f'seed_{seed}/{pair}/{name}_bs{batch}_s{series}.json'
                        d = read(p)
                        assert d['batch'] == batch and d['name'] == name and d['series'] == series
                        assert d['seed'] == seed and d['pair'] == pair and d['state_unchanged']
                        assert len(d['times_ms']) == 200 and all(x > 0 and math.isfinite(x) for x in d['times_ms'])
                        times.extend(d['times_ms'])
                        memory.append(d)
                        hashes.add(d['input_sha256'])
                    assert len(hashes) == 1
                    st = stats(times)
                    st.update(seed=seed, pair=pair, name=name, batch=batch, events=len(times),
                              samples_s=1000.0 * batch / st['mean'],
                              input_sha256=next(iter(hashes)))
                    rows.append(st)
    # Per-seed ratios, najpierw sparowane w obrębie seeda, potem średnia.
    ratios = []
    for seed in SEEDS:
        for pair in PAIRS:
            for batch in BATCHES:
                for tag in TAGS:
                    u = next(r for r in rows if r['seed'] == seed and r['pair'] == pair and r['batch'] == batch and r['name'] == tag + '_U')
                    t = next(r for r in rows if r['seed'] == seed and r['pair'] == pair and r['batch'] == batch and r['name'] == tag + '_T')
                    cu = next(c for c in cost_rows if c['seed'] == seed and c['pair'] == pair and c['name'] == tag + '_U')
                    ct = next(c for c in cost_rows if c['seed'] == seed and c['pair'] == pair and c['name'] == tag + '_T')
                    ratios.append(dict(seed=seed, pair=pair, tag=tag, batch=batch,
                                       latency_ratio=t['median'] / u['median'],
                                       throughput_ratio=t['samples_s'] / u['samples_s'],
                                       mac_ratio=ct['mac'] / cu['mac'],
                                       params_ratio=ct['params'] / cu['params']))
    # Agregaty po seedach.
    averages = []
    for pair in PAIRS:
        for tag in TAGS:
            for batch in BATCHES:
                for name in ('teacher', tag + '_U', tag + '_T'):
                    vals = [r for r in rows if r['pair'] == pair and r['name'] == name and r['batch'] == batch]
                    med = [r['median'] for r in vals]
                    thr = [r['samples_s'] for r in vals]
                    averages.append(dict(pair=pair, tag=tag, name=name, batch=batch,
                                         median_ms=stats(med)['mean'], median_sd=stats(med)['sd'],
                                         samples_s=stats(thr)['mean'], samples_sd=stats(thr)['sd']))
            for batch in BATCHES:
                vals = [r for r in ratios if r['pair'] == pair and r['tag'] == tag and r['batch'] == batch]
                averages.append(dict(pair=pair, tag=tag, name='T/U', batch=batch,
                                     latency_ratio=stats([r['latency_ratio'] for r in vals])['mean'],
                                     throughput_ratio=stats([r['throughput_ratio'] for r in vals])['mean'],
                                     mac_ratio=stats([r['mac_ratio'] for r in vals])['mean'],
                                     params_ratio=stats([r['params_ratio'] for r in vals])['mean']))
    # Jakość wyłącznie reuse E035; nie jest to nowa ocena.
    quality = []
    for pair in PAIRS:
        for tag in TAGS:
            vals = []
            for seed in SEEDS:
                ref = frozen['references'][str(seed)][pair][tag]
                vals.append(dict(seed=seed, U=ref['U']['quality_CMA']['top1'],
                                 T=ref['T']['quality_CMA']['top1'],
                                 U_nll=ref['U']['quality_CMA']['nll'],
                                 T_nll=ref['T']['quality_CMA']['nll']))
            quality.append(dict(pair=pair, tag=tag, per_seed=vals,
                                mean_U=float(np.mean([v['U'] for v in vals])),
                                mean_T=float(np.mean([v['T'] for v in vals])),
                                mean_diff_pp=float(np.mean([100 * (v['T'] - v['U']) for v in vals]))))
    memory = []
    for seed in SEEDS:
        for pair in PAIRS:
            for batch in BATCHES:
                for name in NAMES:
                    ds = [read(run / f'seed_{seed}/{pair}/{name}_bs{batch}_s{s}.json') for s in range(3)]
                    memory.append(dict(seed=seed, pair=pair, name=name, batch=batch,
                                       baseline_mb=float(np.median([d['baseline_allocated'] for d in ds]) / 2**20),
                                       peak_mb=float(np.median([d['peak_allocated'] for d in ds]) / 2**20),
                                       delta_mb=float(np.median([d['delta_allocated'] for d in ds]) / 2**20),
                                       reserved_peak_mb=float(np.median([d['peak_reserved'] for d in ds]) / 2**20)))
    result = dict(run=str(run), manifest=m, rows=rows, ratios=ratios, averages=averages,
                  quality=quality, memory=memory, costs=cost_rows,
                  quality_note='CMA top1/NLL wyłącznie z E035; brak nowej oceny jakości',
                  timing_note='600 zdarzeń/model/batch:3serie×200; mediana, p10, p90; samples/s z średniej',
                  memory_note='mediana3 serii; peak allocated w trakcie200 forwardów, delta=peak−baseline po warmupie')
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(result, indent=1) + '\n')
    with (out / 'per_seed.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / 'ratios.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ratios[0])); w.writeheader(); w.writerows(ratios)
    with (out / 'memory.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(memory[0])); w.writeheader(); w.writerows(memory)
    print('ANALYSIS', out)
    print('rows', len(rows), 'ratios', len(ratios), 'memory', len(memory))


if __name__ == '__main__':
    main(sys.argv[1])
