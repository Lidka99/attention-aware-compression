"""E040: opisowe tabele/wykres z zakończonego audytu; protokół ../OPIS.md.
Wejście: zapisane audit.json i monitor.jsonl; wyjście: TABLES.md/PNG/PDF/metadane.
Analiza po zatrzymaniu próby; bez nowych modeli, danych i zmiany wyborów.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run):
    here = Path(__file__).resolve().parent
    experiment = here.parent
    audit = run / 'analysis/audit.json'
    result = json.loads(audit.read_text())
    frozen = json.loads((run / 'frozen.json').read_text())
    assert result['status'] == 'passed' and result['quality_evaluated'] is False
    cells = result['prediction_gate']['cells']
    rows = result['prediction_per_seed']
    gpu = {i: dict(max_util=0, max_memory_mib=0, positive_samples=0) for i in range(4)}
    for line in (run / 'monitor.jsonl').read_text().splitlines():
        record = json.loads(line)
        for value in record['gpu'].strip().splitlines():
            fields = [x.strip() for x in value.split(',')]
            i, memory, util = int(fields[0]), int(fields[3]), int(fields[4])
            gpu[i]['max_util'] = max(gpu[i]['max_util'], util)
            gpu[i]['max_memory_mib'] = max(gpu[i]['max_memory_mib'], memory)
            gpu[i]['positive_samples'] += int(util > 0)
    assert all(v['positive_samples'] > 0 for v in gpu.values())
    table = ['# E040 — tabele z zamrożonej próby', '',
             f'Run: `{result["run"]}`. Źródło: `analysis/audit.json`, audyt PASS.', '',
             '## Predykcja szkody na C', '',
             'MAE w pp niezgodności decyzji z teacher, nie w pp trafności top1.',
             'Średnia±SD(ddof=1) po4seedach; iloraz średnich, nie średnia ilorazów.', '',
             '| Zbiór | Addytywny po BN | Pary po BN | Iloraz | Poprawa |',
             '|---|---:|---:|---:|---:|']
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    summary = []
    for index, ds in enumerate(('cifar10', 'cifar100')):
        data = sorted([r for r in rows if r['dataset'] == ds], key=lambda r: r['seed'])
        a = np.asarray([100 * r['mae1'] for r in data])
        b = np.asarray([100 * r['mae2'] for r in data])
        cell = next(c for c in cells if c['dataset'] == ds)
        table.append(f'| {ds} | {a.mean():.4f}±{a.std(ddof=1):.4f} | {b.mean():.4f}±{b.std(ddof=1):.4f} | {cell["ratio"]:.4f} | {cell["positive"]}/4 |')
        summary.append(dict(dataset=ds, mae1_pp=float(a.mean()), sd1_pp=float(a.std(ddof=1)),
                            mae2_pp=float(b.mean()), sd2_pp=float(b.std(ddof=1)), ratio=cell['ratio']))
        ax.bar(index - .18, a.mean(), .32, yerr=a.std(ddof=1), capsize=4, color='#4C78A8',
               label='Addytywny po BN' if index == 0 else None, alpha=.8)
        ax.bar(index + .18, b.mean(), .32, yerr=b.std(ddof=1), capsize=4, color='#E45756',
               label='Pary po BN' if index == 0 else None, alpha=.8)
        for j, (left, right) in enumerate(zip(a, b)):
            offset = (j - 1.5) * .025
            ax.plot([index - .18 + offset, index + .18 + offset], [left, right],
                    color='#333333', marker='o', markersize=3.5, linewidth=.8, alpha=.8)
    ax.set_xticks([0, 1], ['CIFAR-10', 'CIFAR-100'])
    ax.set_ylabel('MAE [pp niezgodności decyzji]')
    ax.set_title('E040: model par pogarsza predykcję szkody po BN')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(here / f'e040_mae.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    table += ['', '| Zbiór | Seed | MAE addytywny [pp] | MAE pary [pp] | Iloraz |',
              '|---|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda x: (x['dataset'], x['seed'])):
        table.append(f'| {r["dataset"]} | {r["seed"]} | {100*r["mae1"]:.4f} | {100*r["mae2"]:.4f} | {r["ratio"]:.4f} |')
    table += ['', '## Faktyczne koszty wybranych alokacji', '',
              'Zakresy po4seedach. Wszystkie spełniają oba limity, bez ich przekroczenia.',
              'Wspólne ograniczenia nie oznaczają dokładnie dopasowanych P i MAC.', '',
              '| Zbiór | Operator/reguła | P min–max [M] | MAC min–max [M] |',
              '|---|---|---:|---:|']
    cost_rows = []
    methods = ('FOLD/ADD_POST', 'FOLD/PAIR_POST', 'FOLD/PAIR_PRE', 'MAG2/ADD_POST', 'MAG2/PAIR_POST')
    for ds in ('cifar10', 'cifar100'):
        for method in methods:
            selected = [r for r in result['selections'] if r['dataset'] == ds and r['method'] == method]
            assert len(selected) == 4
            p, m = [r['params'] for r in selected], [r['mac'] for r in selected]
            assert max(p) <= frozen['plans'][ds]['caps']['params'] and max(m) <= frozen['plans'][ds]['caps']['mac']
            table.append(f'| {ds} | {method} | {min(p)/1e6:.6f}–{max(p)/1e6:.6f} | {min(m)/1e6:.6f}–{max(m)/1e6:.6f} |')
            cost_rows.append(dict(dataset=ds, method=method, params_min=min(p), params_max=max(p),
                                  mac_min=min(m), mac_max=max(m)))
    table += ['', '## Referencje U/TILT i anchor', '',
              'Te same szerokości dla FOLD i MAG2; jakość dev nieoceniona po STOP_C.', '',
              '| Zbiór | Profil | P | MAC |', '|---|---|---:|---:|']
    for ds in ('cifar10', 'cifar100'):
        plan = frozen['plans'][ds]
        for r in plan['references'] + [dict(plan['anchor'], name='ANCHOR')]:
            table.append(f'| {ds} | {r["name"]} | {r["params"]} | {r["mac"]} |')
    table += ['', '## Wykonanie', '',
              f'Czas ścienny4GPU: {result["elapsed_seconds"]:.2f}s; rezerwacja≤{4*result["elapsed_seconds"]/3600:.4f}GPU-h.',
              f'Konstrukcje: {sum(r["constructions"] for r in result["counters"])}; CMA: {sum(r["cma_passes"] for r in result["counters"])}.', '',
              '| GPU | Max wykorzystanie w logu [%] | Próbki z aktywnością | Max pamięć [MiB] |',
              '|---:|---:|---:|---:|']
    for i, row in gpu.items():
        table.append(f'| {i} | {row["max_util"]} | {row["positive_samples"]} | {row["max_memory_mib"]} |')
    table += ['', 'To telemetria wykonania profilowania, nie benchmark latencji skompresowanych modeli.',
              'Brak nowych pomiarów jakości i brak wyborów na podstawie tej analizy.']
    (experiment / 'TABLES.md').write_text('\n'.join(table) + '\n')
    evidence = dict(run=result['run'], input_audit_sha256=digest(audit),
                    input_prediction_csv_sha256=digest(run / 'analysis/prediction_per_seed.csv'),
                    script_sha256=digest(Path(__file__)), summary=summary, costs=cost_rows, telemetry=gpu,
                    figures={name: digest(here / name) for name in ('e040_mae.png', 'e040_mae.pdf')},
                    table_sha256=digest(experiment / 'TABLES.md'))
    (here / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, gpu=gpu, gpu_hours=4*result['elapsed_seconds']/3600), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    main(ap.parse_args().run.resolve())
