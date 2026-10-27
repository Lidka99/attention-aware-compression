"""E013: testy CPU alokacji budżetu foldingu (folding_budget.py), bez GPU.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Testy sprawdzają tożsamość dla ratios 0, dokładne budżety E012 dla ratios
0.1/0.3/0.5, solver ``base`` pod te budżety, własności alokacji TILT/RANDOM
oraz (warunkowo) folding z HKMeans i forward. Nie pobierają danych ani wag.

Model budujemy przez ``build_cifar_resnet18`` z E012, wczytywany po ścieżce.
Nie zmieniamy plików E012.

Uruchomienie z folderu E013 na CPU z jednym wątkiem:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES='' python -B -m unittest -v test_folding_budget
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import folding_budget as FB

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

EXPECTED_GROUPS = [
    'relu_conv',
    'layer1.0.relu1',
    'layer1.1.relu1',
    'layer2.0.relu1',
    'layer2.0.relu2',
    'layer2.1.relu1',
    'layer3.0.relu1',
    'layer3.0.relu2',
    'layer3.1.relu1',
    'layer4.0.relu1',
    'layer4.0.relu2',
    'layer4.1.relu1',
]
BASE_PARAMS = 11173962
EXPECTED_BUDGETS = {
    0.1: 9018358,
    0.3: 5459866,
    0.5: 2797610,
}


def build_model():
    reference = FB.load_reference_folding()
    return reference.build_cifar_resnet18()


def make_inputs(seed=0, batch_size=2):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, 32, 32, generator=generator)


def forward(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(inputs).detach().clone()


def hkmeans_available():
    return bool(FB.load_reference_folding().hkmeans_available())


class TestApiAndAllocations(unittest.TestCase):
    def test_group_names_match_e012(self):
        self.assertEqual(FB.group_names(), EXPECTED_GROUPS)
        self.assertEqual(
            FB.group_names(),
            FB.load_reference_folding().group_order())

    def test_tilt_formula_range_and_clipping(self):
        groups = FB.group_names()
        ratios = FB.allocation_from_tilt(groups, base=0.5, tilt=0.2)
        self.assertEqual(len(ratios), 12)
        self.assertAlmostEqual(ratios[0], 0.3)
        self.assertAlmostEqual(ratios[-1], 0.7)
        self.assertTrue(all(0.1 <= value <= 0.9 for value in ratios))
        self.assertEqual(
            ratios, FB.allocation_from_tilt(groups, base=0.5, tilt=0.2))

        low = FB.allocation_from_tilt(groups, base=0.05, tilt=0.9)
        high = FB.allocation_from_tilt(groups, base=0.95, tilt=0.9)
        self.assertTrue(all(value >= 0.1 for value in low))
        self.assertTrue(all(value <= 0.9 for value in high))

    def test_random_is_seeded_and_differs_from_uniform(self):
        groups = FB.group_names()
        uniform = FB.allocation_from_tilt(groups, base=0.5, tilt=0.0)
        first = FB.allocation_from_random(
            groups, base=0.5, sigma=0.05, seed=7)
        second = FB.allocation_from_random(
            groups, base=0.5, sigma=0.05, seed=7)
        other = FB.allocation_from_random(
            groups, base=0.5, sigma=0.05, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, uniform)
        self.assertTrue(all(0.1 <= value <= 0.9 for value in first))

        generator = np.random.default_rng(7)
        offsets = generator.normal(0.0, 0.05, size=len(groups))
        expected = [min(max(0.5 + value, 0.1), 0.9) for value in offsets]
        self.assertTrue(np.allclose(first, expected))

        constant = FB.allocation_from_random(
            groups, base=0.4, sigma=0.0, seed=0)
        self.assertTrue(np.allclose(constant, [0.4] * len(groups)))


class TestFoldingBudgets(unittest.TestCase):
    def test_zero_ratios_identity(self):
        model = build_model()
        ratios = [0.0] * len(FB.group_names())
        folded, diagnostics = FB.fold_with_ratios(
            model, ratios, preserve_fc_bias=True)

        self.assertEqual(diagnostics['params'], BASE_PARAMS)
        self.assertEqual(diagnostics['nparams_after'], BASE_PARAMS)
        self.assertTrue(diagnostics['finite'])
        self.assertTrue(diagnostics['fc_bias']['equal'])
        self.assertEqual(diagnostics['ratios'], ratios)

        original = model.state_dict()
        rebuilt = folded.state_dict()
        self.assertEqual(set(original), set(rebuilt))
        for name, tensor in original.items():
            self.assertEqual(
                tuple(tensor.shape), tuple(rebuilt[name].shape), name)
            self.assertTrue(torch.equal(tensor, rebuilt[name]), name)

    def test_deterministic_budget_counts_match_e012(self):
        model = build_model()
        for ratio, expected in EXPECTED_BUDGETS.items():
            ratios = [ratio] * len(FB.group_names())
            folded, diagnostics = FB.fold_with_ratios(
                model, ratios, seed=42, deterministic=True,
                preserve_fc_bias=True)
            self.assertEqual(diagnostics['params'], expected, ratio)
            self.assertTrue(diagnostics['finite'])
            self.assertEqual(len(diagnostics['widths']), 12)
            self.assertTrue(all(width >= 1 for width in diagnostics['widths']))
            logits = forward(folded, make_inputs(seed=int(ratio * 100)))
            self.assertEqual(tuple(logits.shape), (2, 10))
            self.assertTrue(bool(torch.isfinite(logits).all()))

        self.assertEqual(
            FB.parameter_count_for_ratios(
                model, [0.3] * len(FB.group_names())),
            EXPECTED_BUDGETS[0.3])

    def test_solver_hits_e012_budgets_under_cap(self):
        model = build_model()
        for target in EXPECTED_BUDGETS.values():
            result = FB.solve_base_for_budget(
                model, 'uniform', target)
            self.assertGreaterEqual(result['base'], 0.0)
            self.assertLessEqual(result['base'], 1.0)
            self.assertLessEqual(result['params'], target)
            self.assertTrue(result['ok'])
            self.assertLessEqual(abs(result['delta_params']), 5000)
            self.assertTrue(all(
                0.1 <= value <= 0.9 for value in result['ratios']))

    def test_solver_other_families_stay_under_cap(self):
        model = build_model()
        target = EXPECTED_BUDGETS[0.5]
        cases = (
            ('tilt', {'tilt': 0.2}),
            ('random', {'sigma': 0.05, 'random_seed': 3}),
        )
        for family, kwargs in cases:
            result = FB.solve_base_for_budget(
                model, family, target, **kwargs)
            self.assertGreaterEqual(result['base'], 0.0)
            self.assertLessEqual(result['base'], 1.0)
            self.assertLessEqual(result['params'], target)
            self.assertTrue(result['ok'])
            self.assertTrue(all(
                0.1 <= value <= 0.9 for value in result['ratios']))

    def test_repair_budget_fills_extreme_gap(self):
        model = build_model()
        target = int(round(0.70 * BASE_PARAMS))
        cases = (
            ('tilt', {'tilt': 0.30}),
            ('random', {'sigma': 0.20, 'random_seed': 102}),
        )
        for family, kwargs in cases:
            solved = FB.solve_base_for_budget(
                model, family, target, **kwargs)
            self.assertLess(solved['params'], target - 100000)
            repaired = FB.repair_budget(model, solved['ratios'], target)
            self.assertTrue(repaired['ok'])
            self.assertLessEqual(repaired['params'], target)
            self.assertGreaterEqual(repaired['delta_percent'], -0.5)
            self.assertGreaterEqual(repaired['repair_steps'], 1)
            self.assertTrue(all(
                0.1 <= value <= 0.9 for value in repaired['ratios']))

    def test_percent_to_target(self):
        self.assertAlmostEqual(FB.percent_to_target(100, 100), 0.0)
        self.assertAlmostEqual(FB.percent_to_target(101, 100), 1.0)
        self.assertAlmostEqual(FB.percent_to_target(99, 100), -1.0)


class TestHkmeansIntegration(unittest.TestCase):
    @unittest.skipUnless(hkmeans_available(), 'brak hkmeans/sklearn')
    def test_fold_all_03_hkmeans_forward(self):
        model = build_model()
        ratios = [0.3] * len(FB.group_names())
        folded, diagnostics = FB.fold_with_ratios(
            model, ratios, seed=42, deterministic=False,
            preserve_fc_bias=True)

        self.assertEqual(diagnostics['params'], EXPECTED_BUDGETS[0.3])
        self.assertTrue(diagnostics['finite'])
        for group in diagnostics['groups']:
            self.assertEqual(min(group['cluster_counts']), 1)
            self.assertEqual(
                sum(group['cluster_counts']), group['n_channels'])

        logits = forward(folded, make_inputs(seed=5))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
