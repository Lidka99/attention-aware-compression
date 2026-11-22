"""E040: sensowne kontrole CPU przed zamrożeniem; pełny protokół OPIS.md.
Wejście: geometria i własne checkpointy; wyjście: check_cpu.json, bez jakości.
Nie odczytuje obrazów, nie kalibruje BN, nie używa GPU ani nie zapisuje wag.
"""
import copy
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import design as d
import support as s

np, torch = s.np, s.torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run():
    s.configure()
    start = time.monotonic()
    assert len(d.PROBES) == 79 and len(set(d.PROBES)) == 79
    # Znany, nieaddytywny problem: zachowane znaki interakcji i dokładne remisy.
    coef = dict(constant=300, unary=[7 - g for g in range(12)],
                pair=[(-1) ** i * (1 + i % 9) for i in range(66)], denominator=2048)
    measured = [d.predict_counts(coef, z) for z in d.PROBES]
    assert d.coefficients(measured) == coef
    for z in d.ALL_BITS:
        expected = 300 + sum((7 - g) * z[g] for g in range(12))
        for i, (g, h) in enumerate(d.PAIRS):
            expected += ((-1) ** i * (1 + i % 9)) * z[g] * z[h]
        assert d.predict_counts(coef, z) == expected
    yes = [dict(dataset=ds, seed=seed, error1_sum=100, error2_sum=80)
           for ds in d.DATASETS for seed in d.SEEDS]
    assert d.prediction_gate(yes)['passed']
    no = copy.deepcopy(yes)
    no[-1]['error2_sum'] = 81
    assert not d.prediction_gate(no)['passed']
    zero = [dict(r, error1_sum=0, error2_sum=0) for r in yes]
    assert not d.prediction_gate(zero)['passed']
    tied = [dict(widths=list(d.LOW), params=1, mac=1, v=[0], z=[1]),
            dict(widths=list(d.LOW), params=1, mac=1, v=[1], z=[0])]
    assert d.projection(tied, d.LOW, 'v')['v'] == [0]
    assert d.projection(tied, d.LOW, 'z')['z'] == [0]
    # Ograniczenie modelu danych działa przed odczytem surowych obrazów.
    dummy = s.Images.__new__(s.Images)
    dummy.phase = 'B'
    for part in ('C', 'dev'):
        try:
            dummy.inputs(part)
        except AssertionError:
            pass
        else:
            raise AssertionError('Przedwczesny odczyt ' + part)
    old_prune = load(s.ROOT / 'experiments/2026-11-08_E038_operator_alokacja/pruning.py', '_e040_check_mag')
    historical = s.read(s.ROOT / 'experiments/2026-11-07_E035_ocena_niezalezna/frozen.json')
    geometry_rows, plans = [], {}
    for ds in d.DATASETS:
        raw_t = historical['plans'][f'resnet_{ds}']['b50']['T']
        plan = d.make_plan(ds, [raw_t[i] for i in d.PERM])
        plans[ds] = plan
        assert d.cost(d.UNIFORM[ds], ds) == d.CAPS[ds]
        model = s.source_model(ds, 42)
        before = s.state_hash(model)
        # Niezależne torchvision gwarantuje poprawny mapping własnego checkpointu.
        from torchvision.models import resnet18
        tv = resnet18(weights=None, num_classes=10 if ds == 'cifar10' else 100)
        tv.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        tv.maxpool = torch.nn.Identity()
        tv.load_state_dict(model.state_dict(), strict=True)
        tv.eval().requires_grad_(False)
        x = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
        assert torch.allclose(s.predict(tv, x), s.predict(model, x, d.cost(d.SIZES, ds)[1]), atol=1e-4, rtol=1e-5)
        _, _, axes, groups = s.upstream()

        class Geometry:
            names, sizes = list(groups), list(d.SIZES)

            def count(self, widths):
                return d.cost([widths[i] for i in d.PERM], ds)[0]

        rt = SimpleNamespace(e37=SimpleNamespace(geometry=lambda *args: Geometry()),
            e35=SimpleNamespace(e29=SimpleNamespace(
                get_axis_to_perm_ResNet18=lambda override=False: axes,
                axes2perm_to_perm2axes=lambda a: groups)))
        for which, widths in [('U', d.UNIFORM[ds]), ('ANCHOR', plan['anchor']['widths']),
                              ('TILT_PROJ', plan['references'][-1]['widths'])]:
            mag, _ = s.prune(model, widths)
            s.validate_shapes(model, mag, widths, ds)
            native_widths = [widths[i] for i in d.PERM]
            original, _ = old_prune.prune(model, f'resnet_{ds}', native_widths, rt)
            assert s.state_hash(mag) == s.state_hash(original)
            assert s.predict(mag, x, d.cost(widths, ds)[1]).shape == (2, 10 if ds == 'cifar10' else 100)
            geometry_rows.append(dict(dataset=ds, profile=which, operator='MAG2',
                widths=list(widths), params=d.cost(widths, ds)[0], mac=d.cost(widths, ds)[1],
                exact_E038=True))
            if which != 'U':
                folded, diag = s.fold(model, widths)
                s.validate_shapes(model, folded, widths, ds)
                assert s.predict(folded, x, d.cost(widths, ds)[1]).shape == (2, 10 if ds == 'cifar10' else 100)
                assert {r['group']: r['k'] for r in diag} == dict(zip(d.NAMES, widths))
                geometry_rows.append(dict(dataset=ds, profile=which, operator='FOLD',
                    widths=list(widths), params=d.cost(widths, ds)[0], mac=d.cost(widths, ds)[1],
                    group_mapping=True))
                del folded
            del mag, original
        assert before == s.state_hash(model)
        del model, tv
        print('CPU', ds, 'mapping/koszty/MAG2/forward PASS', flush=True)
    assert not s.DENIED and not s.OPENED_DATA
    obj = dict(status='passed', utc=s.utc(), seconds=time.monotonic() - start,
               code_hashes=s.code_hashes(), geometry=geometry_rows,
               design_counts={ds: dict(feasible=p['feasible_count'], masks=p['diagnostic_pool']) for ds, p in plans.items()},
               sources_unchanged=True, data_open_attempts=s.DENIED,
               actual_images_used=False, BN_calibration_used=False, gpu_used=False)
    s.write(s.HERE / 'check_cpu.json', obj)
    print('CPU CHECK PASS', obj['seconds'], flush=True)


if __name__ == '__main__':
    run()
