"""E013: testy CPU miar grupowania i odległości (grouping.py), bez GPU.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Testy sprawdzają ręcznie policzone odległości L1/L2/cosine/correlation,
obsługę zerowych norm, własności macierzy odległości, determinizm i stałe
``k`` dla metod aglomeracyjnych oraz delegację ``hkmeans_l2`` do E012.
Nie pobierają danych ani wag. Testy HKMeans są pomijane, gdy brak hkmeans.

Uruchomienie z folderu E013:
    python -m unittest -v test_grouping
"""

import math
import sys
import types
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grouping as G


class TestDistanceL1L2(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.0, 2.0],
            [0.0, 0.0],
        ], dtype=torch.float64)

    def test_l1_hand_computed(self):
        expected = torch.tensor([
            [0.0, 3.0, 3.0, 0.0],
            [3.0, 0.0, 2.0, 3.0],
            [3.0, 2.0, 0.0, 3.0],
            [0.0, 3.0, 3.0, 0.0],
        ], dtype=torch.float64)
        self.assertTrue(torch.equal(
            G.distance_matrix(self.x, 'l1'), expected))

    def test_l2_hand_computed(self):
        root5 = math.sqrt(5.0)
        expected = torch.tensor([
            [0.0, root5, root5, 0.0],
            [root5, 0.0, 2.0, root5],
            [root5, 2.0, 0.0, root5],
            [0.0, root5, root5, 0.0],
        ], dtype=torch.float64)
        self.assertTrue(torch.allclose(
            G.distance_matrix(self.x, 'l2'), expected,
            rtol=0.0, atol=1e-15))

    def test_l1_l2_zero_rows_keep_metric_definition(self):
        distances_l1 = G.distance_matrix(self.x, 'l1')
        distances_l2 = G.distance_matrix(self.x, 'l2')
        self.assertEqual(float(distances_l1[0, 3]), 0.0)
        self.assertEqual(float(distances_l2[0, 3]), 0.0)
        self.assertAlmostEqual(float(distances_l1[0, 1]), 3.0)
        self.assertAlmostEqual(float(distances_l2[0, 1]), math.sqrt(5.0))


class TestDistanceCosine(unittest.TestCase):
    def test_cosine_hand_computed_without_abs(self):
        x = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 1.0],
        ], dtype=torch.float64)
        root2 = math.sqrt(2.0)
        expected = torch.tensor([
            [0.0, 1.0, 2.0, 1.0 - 1.0 / root2],
            [1.0, 0.0, 1.0, 1.0 - 1.0 / root2],
            [2.0, 1.0, 0.0, 1.0 + 1.0 / root2],
            [1.0 - 1.0 / root2, 1.0 - 1.0 / root2,
             1.0 + 1.0 / root2, 0.0],
        ], dtype=torch.float64)
        got = G.distance_matrix(x, 'cosine')
        self.assertTrue(torch.allclose(
            got, expected, rtol=0.0, atol=1e-15))
        self.assertAlmostEqual(float(got[0, 2]), 2.0)

    def test_cosine_zero_norm_policy(self):
        x = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ], dtype=torch.float64)
        expected = torch.tensor([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float64)
        self.assertTrue(torch.equal(
            G.distance_matrix(x, 'cosine'), expected))


class TestDistanceCorrelation(unittest.TestCase):
    def test_correlation_hand_computed_and_zero_norm(self):
        x = torch.tensor([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 2.0, 4.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
        ], dtype=torch.float64)
        root21 = math.sqrt(21.0)
        expected = torch.tensor([
            [0.0, 2.0, 1.0 - 9.0 / (2.0 * root21), 1.0, 1.0],
            [2.0, 0.0, 1.0 + 9.0 / (2.0 * root21), 1.0, 1.0],
            [1.0 - 9.0 / (2.0 * root21),
             1.0 + 9.0 / (2.0 * root21), 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        got = G.distance_matrix(x, 'correlation')
        self.assertTrue(torch.allclose(
            got, expected, rtol=0.0, atol=1e-15))

    def test_correlation_is_scale_and_shift_invariant(self):
        x = torch.tensor([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [10.0, 20.0, 30.0],
        ], dtype=torch.float64)
        got = G.distance_matrix(x, 'correlation')
        self.assertTrue(torch.allclose(
            got, torch.zeros((3, 3), dtype=torch.float64),
            rtol=0.0, atol=1e-14))


class TestDistanceProperties(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 3.0],
            [2.0, 2.0, 2.0],
            [-1.0, 2.0, -3.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32)

    def test_all_metrics_are_finite_symmetric_and_nonnegative(self):
        for metric in G.DISTANCE_METRICS:
            with self.subTest(metric=metric):
                distances = G.distance_matrix(self.x, metric)
                self.assertEqual(distances.dtype, torch.float64)
                self.assertEqual(tuple(distances.shape), (5, 5))
                self.assertTrue(bool(torch.isfinite(distances).all()))
                self.assertTrue(bool((distances >= 0.0).all()))
                self.assertTrue(torch.equal(distances, distances.T))
                self.assertTrue(bool((torch.diagonal(distances) == 0.0).all()))
                if metric in ('cosine', 'correlation'):
                    self.assertTrue(bool((distances <= 2.0 + 1e-12).all()))

    def test_method_names_resolve_to_their_metric(self):
        for method in ('average_l2', 'average_l1',
                       'average_cosine', 'average_correlation'):
            metric = method.removeprefix('average_')
            with self.subTest(method=method):
                self.assertTrue(torch.equal(
                    G.distance_matrix(self.x, method),
                    G.distance_matrix(self.x, metric)))
        self.assertTrue(torch.equal(
            G.distance_matrix(self.x, 'hkmeans_l2'),
            G.distance_matrix(self.x, 'l2')))

    def test_distance_matrix_is_permutation_equivariant(self):
        x = torch.tensor([
            [1.0, 0.0, 2.0],
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
        ], dtype=torch.float64)
        permutation = torch.tensor([2, 0, 3, 1])
        for metric in G.DISTANCE_METRICS:
            with self.subTest(metric=metric):
                base = G.distance_matrix(x, metric)
                permuted = G.distance_matrix(x[permutation], metric)
                expected = base[permutation][:, permutation]
                self.assertTrue(torch.allclose(
                    permuted, expected, rtol=0.0, atol=1e-14))

    def test_l2_unit_sphere_orders_like_cosine(self):
        root2 = math.sqrt(2.0)
        x = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / root2, 1.0 / root2],
            [-1.0, 0.0],
        ], dtype=torch.float64)
        l2 = G.distance_matrix(x, 'l2')
        cosine = G.distance_matrix(x, 'cosine')
        # Na sferze jednostkowej d_cos = d_l2**2 / 2; to nie jest nowa miara.
        self.assertTrue(torch.allclose(
            cosine, 0.5 * l2.square(), rtol=0.0, atol=1e-14))


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [10.1, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.1, 10.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ], dtype=torch.float64)

    def test_fixed_k_returns_exactly_k_nonempty_clusters(self):
        for method in G.GROUPING_METHODS:
            if method == 'hkmeans_l2' and not G.hkmeans_available():
                continue
            with self.subTest(method=method):
                labels = G.cluster(self.features, 4, method)
                self.assertEqual(labels.dtype, torch.int64)
                self.assertEqual(labels.device.type, 'cpu')
                self.assertEqual(int(labels.numel()), 8)
                self.assertTrue(torch.equal(
                    torch.unique(labels, sorted=True),
                    torch.arange(4, dtype=torch.int64)))
                counts = torch.bincount(labels, minlength=4)
                self.assertGreaterEqual(int(counts.min()), 1)

    def test_n_clusters_equal_n_is_identity(self):
        for method in G.GROUPING_METHODS:
            with self.subTest(method=method):
                labels = G.cluster(self.features, 8, method)
                self.assertTrue(torch.equal(
                    labels, torch.arange(8, dtype=torch.int64)))

    def test_one_cluster(self):
        for method in G.GROUPING_METHODS:
            if method == 'hkmeans_l2' and not G.hkmeans_available():
                continue
            with self.subTest(method=method):
                labels = G.cluster(self.features, 1, method)
                self.assertTrue(torch.equal(
                    labels, torch.zeros(8, dtype=torch.int64)))

    def test_deterministic_repeated_calls(self):
        for method in G.GROUPING_METHODS:
            if method == 'hkmeans_l2' and not G.hkmeans_available():
                continue
            with self.subTest(method=method):
                first = G.cluster(self.features, 4, method)
                second = G.cluster(self.features, 4, method)
                self.assertTrue(torch.equal(first, second))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            G.cluster(self.features, 0, 'average_l1')
        with self.assertRaises(ValueError):
            G.cluster(self.features, 9, 'average_l1')
        with self.assertRaises(ValueError):
            G.cluster(self.features, 2, 'nieznana')
        with self.assertRaises(ValueError):
            G.cluster(torch.tensor([1.0, 2.0]), 1, 'average_l1')
        with self.assertRaises(TypeError):
            G.cluster(self.features, 2.0, 'average_l1')


class TestBindMethod(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor([
            [0.0, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.2, 10.0],
            [0.0, 10.0],
            [0.2, 10.0],
        ], dtype=torch.float64)

    def test_average_method_is_bound_and_restored(self):
        def original(features, n_clusters):
            return torch.zeros(
                int(features.shape[0]), dtype=torch.int64)

        module = types.SimpleNamespace(cluster_group_features=original)
        with G.bind_method(module, 'average_l1') as yielded:
            self.assertIs(yielded, module)
            self.assertIsNot(module.cluster_group_features, original)
            got = module.cluster_group_features(self.features, 3)
            expected = G.cluster(self.features, 3, 'average_l1')
            self.assertTrue(torch.equal(got, expected))
        self.assertIs(module.cluster_group_features, original)

    def test_hkmeans_uses_saved_original_without_recursion(self):
        calls = []

        def original(features, n_clusters):
            calls.append((int(features.shape[0]), int(n_clusters)))
            return torch.arange(
                int(features.shape[0]), dtype=torch.int64) % int(n_clusters)

        module = types.SimpleNamespace(cluster_group_features=original)
        with G.bind_method(module, 'hkmeans_l2'):
            got = module.cluster_group_features(self.features, 3)
        self.assertEqual(calls, [(6, 3)])
        self.assertTrue(torch.equal(
            got, torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)))
        self.assertIs(module.cluster_group_features, original)

    def test_use_method_binds_default_e012_module(self):
        original = G.E012_FOLDING.cluster_group_features
        with G.use_method('average_cosine') as module:
            self.assertIs(module, G.E012_FOLDING)
            self.assertIsNot(module.cluster_group_features, original)
            got = module.cluster_group_features(self.features, 3)
            self.assertTrue(torch.equal(
                got, G.cluster(self.features, 3, 'average_cosine')))
        self.assertIs(G.E012_FOLDING.cluster_group_features, original)

    def test_unknown_method_rejected_before_patch(self):
        module = types.SimpleNamespace(
            cluster_group_features=lambda features, n_clusters: None)
        with self.assertRaises(ValueError):
            with G.bind_method(module, 'nieznana'):
                pass


class TestHkmeansDelegation(unittest.TestCase):
    @unittest.skipUnless(G.hkmeans_available(), 'brak hkmeans/sklearn')
    def test_hkmeans_l2_equals_e012_call(self):
        features = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0, 0.0],
            [10.1, 10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.1, 10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [10.1, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        got = G.cluster(features, 4, 'hkmeans_l2')
        expected = G.E012_CLUSTER_GROUP_FEATURES(features, 4)
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(
            torch.unique(got, sorted=True),
            torch.arange(4, dtype=torch.int64)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
