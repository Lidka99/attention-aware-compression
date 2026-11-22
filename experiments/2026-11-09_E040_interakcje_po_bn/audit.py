"""E040: niezależne przeliczenie zapisów B/C/dev bez nowych ocen; OPIS.md.
Wejście: zakończony run i predykcje; wyjście: audyt, kontrasty i decyzja.
Nie odczytuje obrazów/wag, nie korzysta z GPU i nie wybiera nowych profili.
"""
import argparse
import csv
import itertools
import math
from pathlib import Path

import design as d
import support as s

np = s.np


def array(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def stats(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 4 and np.isfinite(a).all()
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                per_seed=a.tolist(), positive=int(np.count_nonzero(a > 0)))


def independent_coef(counts):
    base = int(counts[0])
    unary = [int(v) - base for v in counts[1:13]]
    interaction = {}
    for index, (g, h) in enumerate(itertools.combinations(range(12), 2)):
        interaction[(g, h)] = int(counts[13 + index]) - int(counts[1 + g]) - int(counts[1 + h]) + base
    return base, unary, interaction


def q_at(coef, z, order=2):
    base, unary, pairs = coef
    value = base + sum(unary[i] for i in range(12) if z[i])
    if order == 2:
        value += sum(v for (g, h), v in pairs.items() if z[g] and z[h])
    return value


def independent_solve(plan, coef, order):
    selected = None
    for z in itertools.product((0, 1), repeat=12):
        w = tuple(k if z[g] == 0 else plan['low'][g] + plan['high'][g] - k
                  for g, k in enumerate(plan['anchor']['widths']))
        p, mac = d.cost(w, plan['dataset'])
        if p > plan['caps']['params'] or mac > plan['caps']['mac']:
            continue
        candidate = (q_at(coef, z, order), -p, mac, z, w)
        if selected is None or candidate < selected:
            selected = candidate
    return selected


def audit(run):
    s.configure()
    manifest, frozen = s.read(run / 'manifest.json'), s.read(run / 'frozen.json')
    assert manifest['status'] == 'completed' and manifest['elapsed_seconds'] <= 7200
    assert [(r['gpu'], r['seed'], r['exit_code']) for r in manifest['workers']] == [(g, 42 + g, 0) for g in range(4)]
    assert manifest['frozen_sha256'] == s.digest(run / 'frozen.json')
    assert frozen['code_hashes'] == s.code_hashes() == manifest['code_hashes']
    s.validate_hashes(frozen['source_hashes'])
    b_gate, c_gate = s.read(run / 'B_GATE.json'), s.read(run / 'C_GATE.json')
    assert b_gate['passed']
    prediction_rows, model_rows, selections, counters, dev_cells = [], [], [], [], {}
    source_files = {}
    for seed in d.SEEDS:
        status = s.read(run / f'seed_{seed}/status.json')
        assert status['status'] == 'completed' and status['gpu'] == seed - 42
        assert status['data_open_attempts'] == [] and status['dev_used'] == c_gate['passed']
        assert status['gpu_name']
        for ds in d.DATASETS:
            root = run / f'seed_{seed}' / ds
            plan = frozen['plans'][ds]
            choice = s.read(root / 'selection.json')
            assert s.digest(root / 'selection.json') == b_gate['selections'][str(seed)][ds]
            assert choice['c_accessed'] is False and choice['dev_accessed'] is False
            assert choice['optimization_seconds'] <= 60
            expected_ids, reserve = s.split_indices(ds)
            assert expected_ids == frozen['indices'][ds]
            assert not set(sum(expected_ids.values(), [])).intersection(reserve)
            assert s.read(root / 'access_B.json') == frozen['views'][ds]
            access = s.read(root / ('access_dev.json' if c_gate['passed'] else 'access_C.json'))
            assert not any(row['part'].startswith('dev') for row in access) or c_gate['passed']
            assert all(row['part'] in ('A', 'B', 'C', 'dev', 'dev_labels') for row in access)
            counts = status['counters'][ds]
            assert counts['constructions'] <= 200 and counts['unique_configurations'] <= 200 and counts['cma_passes'] <= 200
            counters.append(dict(dataset=ds, seed=seed, **counts))
            teacher_b = array(root / 'arrays/B_teacher.npz')['pred']
            teacher_c = array(root / 'arrays/C_teacher.npz')['pred']
            assert teacher_b.shape == teacher_c.shape == (2048,)
            records = {p.stem: s.read(p) for p in (root / 'B').glob('*.json')}
            for key, record in records.items():
                assert record['id'] == key
                file = root / f'arrays/B_{key}.npz'
                assert s.digest(file) == record['arrays_sha256']
                pred = array(file)
                assert pred['pre'].shape == pred['post'].shape == (2048,)
                for mode in ('pre', 'post'):
                    assert record['counts'][mode] == int(np.count_nonzero(pred[mode] != teacher_b))
                p, mac = d.cost(record['widths'], ds)
                assert (record['params'], record['mac']) == (p, mac)
                evidence = record['forward_geometry']
                assert mac == evidence['mac'] == sum(row['mac'] for row in evidence['layers'])
                for layer in evidence['layers']:
                    w, output = layer['weight_shape'], layer['output_shape']
                    expected = math.prod(output) * math.prod(w[1:])
                    assert expected == layer['mac']
                assert record['calibration']['parameters_sha256'] == record['parameters_sha256']
                assert record['calibration']['samples'] == 4096 and record['calibration']['batches'] == 32
                assert all(r['batches'] == 32 for r in record['calibration']['layers'])
                assert len(record['diagnostics']) == 12
                assert {row['group']: row['k'] for row in record['diagnostics']} == dict(zip(d.NAMES, record['widths']))
                if record['operator'] == 'FOLD':
                    for row in record['diagnostics']:
                        assert len(row['labels']) == row['n'] and sorted(set(row['labels'])) == list(range(row['k']))
                else:
                    for row in record['diagnostics']:
                        assert len(row['keep']) == len(set(row['keep'])) == row['k']
                        assert min(row['keep']) >= 0 and max(row['keep']) < row['n']
                for replay in record.get('replays', []):
                    assert replay['state_sha256'] == record['state_sha256'] and replay['counts'] == record['counts']
            for alias, key in choice['aliases'].items():
                record = records[key]
                assert d.feasible(record['widths'], ds)
                assert choice['terminal_states'][key] == record['state_sha256']
                model_rows.append(dict(dataset=ds, seed=seed, alias=alias, id=key,
                    params=record['params'], mac=record['mac'],
                    p_fraction=record['params'] / plan['caps']['params'],
                    mac_fraction=record['mac'] / plan['caps']['mac'],
                    disagreement_B=record['counts']['post'] / 2048))
            coefs = {}
            for op in ('FOLD', 'MAG2'):
                profile = choice['profiles'][op]
                assert len(profile['probe_keys']) == 79
                coefs[op] = {}
                for mode in ('pre', 'post'):
                    vals = [records[key]['counts'][mode] for key in profile['probe_keys']]
                    c = independent_coef(vals)
                    coefs[op][mode] = c
                    saved = profile['coefficients'][mode]
                    assert saved['constant'] == c[0] and saved['unary'] == c[1]
                    assert saved['pair'] == list(c[2].values()) and saved['denominator'] == 2048
                    assert [q_at(c, z) for z in d.PROBES] == vals
                for name, mode, order in ([('ADD_POST', 'post', 1), ('PAIR_POST', 'post', 2)]
                                          + ([('PAIR_PRE', 'pre', 2)] if op == 'FOLD' else [])):
                    chosen = choice['selections'][op + '/' + name]
                    ref = independent_solve(plan, coefs[op][mode], order)
                    assert (chosen['q_count'], -chosen['params'], chosen['mac'], tuple(chosen['z']), tuple(chosen['widths'])) == ref
                    selections.append(dict(dataset=ds, seed=seed, method=op + '/' + name,
                        params=chosen['params'], mac=chosen['mac'], q_B=chosen['q_count'] / 2048,
                        observed_B=records[chosen['id']]['counts']['post'] / 2048,
                        predicted_out_of_range=chosen['out_of_range']))
                refs = [(records[choice['aliases'][op + '/' + r['name']]]['counts']['post'],
                         r['mac'], -r['params'], i, r['name']) for i, r in enumerate(plan['references'])]
                assert min(refs)[-1] == choice['tilt_stars'][op]
            result = s.read(root / 'prediction.json')
            assert result['selection_sha256'] == s.digest(root / 'selection.json')
            errors1, errors2 = [], []
            assert len(result['rows']) == 16
            for mask, measured in zip(plan['diagnostics'], result['rows']):
                assert mask['z'] == measured['z'] and sum(mask['z']) >= 3
                key = measured['id']
                assert records[key]['widths'] == mask['widths']
                pred = array(root / f'arrays/C_{key}.npz')['pred']
                actual = int(np.count_nonzero(pred != teacher_c))
                q1, q2 = q_at(coefs['FOLD']['post'], mask['z'], 1), q_at(coefs['FOLD']['post'], mask['z'], 2)
                assert (actual, q1, q2) == (measured['actual_count'], measured['q1_count'], measured['q2_count'])
                errors1.append(abs(actual - q1))
                errors2.append(abs(actual - q2))
            assert sum(errors1) == result['error1_sum'] and sum(errors2) == result['error2_sum']
            prediction_rows.append(dict(dataset=ds, seed=seed, error1_sum=sum(errors1), error2_sum=sum(errors2),
                mae1=sum(errors1) / (16 * 2048), mae2=sum(errors2) / (16 * 2048),
                ratio=sum(errors2) / sum(errors1) if sum(errors1) else None))
            for key in set(choice['aliases'].values()):
                record = s.read(root / f'C/{key}.json')
                assert record['state_sha256'] == choice['terminal_states'][key]
                assert s.digest(root / f'arrays/C_{key}.npz') == record['arrays_sha256']
            if c_gate['passed']:
                values = s.read(root / 'dev.json')
                labels = array(root / 'arrays/dev_labels.npz')['labels']
                assert labels.shape == (5000,)
                for key, metric in values['metrics'].items():
                    raw = array(root / f'arrays/dev_{key}.npz')
                    correct = int(np.count_nonzero(raw['pred'] == labels))
                    assert metric['n'] == 5000 and metric['correct'] == correct
                    assert metric['top1'] == correct / 5000
                    assert abs(metric['nll'] - float(np.mean(raw['ce']))) < 1e-12
                    saved = s.read(root / f'dev/{key}.json')
                    expected_state = choice['terminal_states'][key] if key != 'teacher' else s.read(root / 'teacher.json')['state_sha256']
                    assert saved['state_sha256'] == expected_state
                dev_cells[(ds, seed)] = values
            else:
                assert not (root / 'dev.json').exists() and not list((root / 'arrays').glob('dev*'))
            for path in root.rglob('*'):
                if path.is_file():
                    assert path.suffix not in ('.pt', '.pth', '.safetensors')
                    source_files[str(path.relative_to(run))] = s.digest(path)
    independently = d.prediction_gate(prediction_rows)
    assert independently['passed'] == c_gate['passed']
    for row, saved in zip(independently['cells'], c_gate['cells']):
        for key in ('dataset', 'passed', 'positive', 'mae1', 'mae2', 'ratio'):
            assert row[key] == saved[key], key
    assert sum(row['constructions'] for row in counters) <= 1600
    quality = []
    if c_gate['passed']:
        for ds in d.DATASETS:
            references = [op + '/' + ref['name'] for op in ('FOLD', 'MAG2') for ref in frozen['plans'][ds]['references']]
            references += ['FOLD/ANCHOR', 'MAG2/ANCHOR', 'MAG2/ADD_POST', 'MAG2/PAIR_POST']
            assert len(references) == 20
            for alias, threshold, minimum in [(r, 200, -25) for r in references] + [
                    ('FOLD/ADD_POST', 100, None), ('FOLD/PAIR_PRE', 50, None)]:
                delta = []
                for seed in d.SEEDS:
                    cell = dev_cells[(ds, seed)]
                    full = cell['metrics'][cell['aliases']['FOLD/PAIR_POST']]['correct']
                    ref = cell['metrics'][cell['aliases'][alias]]['correct']
                    delta.append(full - ref)
                passed = sum(delta) >= threshold and sum(v > 0 for v in delta) >= 3
                if minimum is not None:
                    passed = passed and min(delta) >= minimum
                quality.append(dict(dataset=ds, reference=alias, passed=passed,
                                    **stats([v / 50 for v in delta])))
    primary = c_gate['passed'] and all(v['passed'] for v in quality)
    result = dict(status='passed', utc=s.utc(), run=s.relative(run),
                  prediction_gate=independently, prediction_per_seed=prediction_rows,
                  quality_evaluated=c_gate['passed'], quality=quality, primary_pass=primary,
                  decision='extend' if primary else 'stop_and_return_to_article',
                  selections=selections, models=model_rows, counters=counters,
                  elapsed_seconds=manifest['elapsed_seconds'], input_hashes_verified=True,
                  code_hashes=frozen['code_hashes'], source_hashes=source_files,
                  no_new_weight_files=True, no_test_or_reserve=True)
    folder = run / 'analysis'
    folder.mkdir()
    s.write(folder / 'audit.json', result)
    csv_write(folder / 'prediction_per_seed.csv', prediction_rows)
    csv_write(folder / 'selected_profiles.csv', selections)
    csv_write(folder / 'model_costs.csv', model_rows)
    if quality:
        csv_write(folder / 'quality_contrasts.csv', quality)
    report = ['# E040 — audyt zapisów', '', f'Run: `{s.relative(run)}`.', '',
              '**Audyt integralności: PASS.**', f'Bramka predykcji: **{independently["passed"]}**.',
              f'Ocena dev wykonana: **{c_gate["passed"]}**.', f'Primary: **{primary}**.', '',
              '| Zbiór | MAE addytywny [pp] | MAE pary [pp] | Iloraz | Poprawa seedów | PASS |',
              '|---|---:|---:|---:|---:|---|']
    for cell in independently['cells']:
        ratio = f'{cell["ratio"]:.4f}' if cell['ratio'] is not None else 'n/a'
        report.append(f'| {cell["dataset"]} | {100 * cell["mae1"]:.4f} | {100 * cell["mae2"]:.4f} | {ratio} | {cell["positive"]}/4 | {cell["passed"]} |')
    report += ['', f'Czas4GPU: {manifest["elapsed_seconds"]:.2f}s.',
               f'Konstrukcje: {sum(v["constructions"] for v in counters)}; CMA: {sum(v["cma_passes"] for v in counters)}.',
               'Bez nowych plików wag, treningu, testów E035 i użycia rezerwy CIFAR-100.', '',
               'Decyzja: ' + ('osobny plan niezależnej oceny.' if primary else 'zakończyć próbę i wrócić do artykułu E035–E039.')]
    (folder / 'AUDYT.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    audit(parser.parse_args().run.resolve())
