"""E012: testy CPU operatora foldingu ResNet18 (folding.py), bez danych i GPU.

Eksperyment: 2026-10-24_E012_folding_referencja; pełny opis: OPIS.md.
Testy nie pobierają danych ani wag. Domyślnie nie wymagają hkmeans/sklearn:
helpery grup i apply_fold_labels pozwalają wstrzyknąć etykiety klastrów.
Test integracyjny HKMeans uruchamia się tylko, gdy zależności są dostępne
i ustawiono E012_RUN_HKMEANS=1.

Uruchomienie:
    python -m unittest -v test_folding
    E012_RUN_HKMEANS=1 python -m unittest -v test_folding
"""

import copy
import io
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import folding as F

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


def build_model():
    return F.build_cifar_resnet18()


def labels_for(model, groups, keep_ratio, seed):
    labels = {}
    for group in groups:
        params = F.collect_group_params(model, group)
        n_channels = F.group_n_channels(params, group)
        n_clusters = max(int(n_channels * keep_ratio), 1)
        base = torch.arange(n_channels, dtype=torch.int64) % n_clusters
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n_channels, generator=generator)
        labels[group.name] = base[permutation]
    return labels


def singleton_labels_for(model, groups):
    labels = {}
    for group in groups:
        params = F.collect_group_params(model, group)
        n_channels = F.group_n_channels(params, group)
        labels[group.name] = torch.arange(n_channels, dtype=torch.int64)
    return labels


def make_inputs(seed=0, batch_size=2):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, 32, 32, generator=generator)


def forward(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(inputs).detach().clone()


class TestMapping(unittest.TestCase):
    def test_group_order_and_count(self):
        groups = F.resnet18_groups()
        self.assertEqual(len(groups), 12)
        self.assertEqual(F.group_order(), EXPECTED_GROUPS)

    def test_all_axes_covered_once(self):
        groups = F.resnet18_groups()
        covered = {}
        for group in groups:
            for pa in group.axes:
                key = (pa.full_name, pa.axis)
                self.assertNotIn(key, covered, f'Duplikat osi {key}')
                covered[key] = group.name

        model = build_model()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                for axis, perm in enumerate(F.resnet18_axis_to_perm()[name + '.weight']):
                    if perm is not None:
                        self.assertIn((name + '.weight', axis), covered)
            if isinstance(module, nn.BatchNorm2d):
                for pname in ('weight', 'bias', 'running_mean', 'running_var'):
                    self.assertIn((name + '.' + pname, 0), covered)
        self.assertIn(('fc.weight', 1), covered)
        self.assertNotIn(('fc.weight', 0), covered)
        self.assertNotIn(('fc.bias', 0), covered)

    def test_stage_inputs_belong_to_previous_group(self):
        membership = {}
        for group in F.resnet18_groups():
            for pa in group.axes:
                membership[(pa.full_name, pa.axis)] = group.name

        self.assertEqual(membership[('conv1.weight', 0)], 'relu_conv')
        self.assertEqual(membership[('layer1.0.conv1.weight', 1)], 'relu_conv')
        self.assertEqual(membership[('layer2.0.conv1.weight', 1)], 'relu_conv')
        self.assertEqual(membership[('layer2.0.downsample.0.weight', 1)], 'relu_conv')
        self.assertEqual(membership[('layer3.0.conv1.weight', 1)], 'layer2.0.relu2')
        self.assertEqual(
            membership[('layer3.0.downsample.0.weight', 1)], 'layer2.0.relu2')
        self.assertEqual(membership[('layer4.0.conv1.weight', 1)], 'layer3.0.relu2')
        self.assertEqual(
            membership[('layer4.0.downsample.0.weight', 1)], 'layer3.0.relu2')
        self.assertEqual(membership[('layer4.1.conv1.weight', 1)], 'layer4.0.relu2')
        self.assertEqual(membership[('fc.weight', 1)], 'layer4.0.relu2')


class TestHelpers(unittest.TestCase):
    def test_validate_labels_rejects_empty_cluster(self):
        with self.assertRaises(RuntimeError):
            F.validate_cluster_labels(torch.tensor([0, 0, 2]), 3, 3)
        with self.assertRaises(RuntimeError):
            F.validate_cluster_labels(torch.tensor([0, 0, 2]), 3)

    def test_build_merge_matrix_rows_sum_to_one(self):
        labels = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int64)
        merge = F.build_merge_matrix(labels, 3, 6)
        self.assertEqual(tuple(merge.shape), (3, 6))
        self.assertTrue(torch.allclose(
            merge.sum(dim=1), torch.ones(3), rtol=0, atol=0))
        self.assertTrue(torch.equal(
            merge[1], torch.tensor([0., 0.5, 0.5, 0., 0., 0.])))

    def test_axis0_sum_axis1_mean(self):
        group = F.resnet18_groups()[1]
        params = {
            'layer1.0.conv1.weight': torch.tensor(
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]]]),
            'layer1.0.bn1.weight': torch.tensor([10.0, 20.0, 30.0, 40.0]),
            'layer1.0.bn1.bias': torch.tensor([1.0, 1.0, 1.0, 1.0]),
            'layer1.0.bn1.running_mean': torch.tensor([0.0, 0.0, 0.0, 0.0]),
            'layer1.0.bn1.running_var': torch.tensor([1.0, 1.0, 1.0, 1.0]),
            'layer1.0.conv2.weight': torch.tensor(
                [[1.0, 2.0, 3.0, 4.0]] * 4
            ).reshape(4, 4, 1, 1),
        }
        labels = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
        merge = F.build_merge_matrix(labels, 3, 4)
        merged = F.merge_group_params(params, group, merge)
        self.assertTrue(torch.equal(
            merged['layer1.0.bn1.weight'],
            torch.tensor([10.0, 50.0, 40.0])))
        axis1 = merged['layer1.0.conv2.weight']
        self.assertEqual(tuple(axis1.shape), (4, 3, 1, 1))
        for out_channel in range(4):
            self.assertTrue(torch.equal(
                axis1[out_channel].flatten(),
                torch.tensor([1.0, 2.5, 4.0])))

    def test_singleton_labels_identity(self):
        group = F.FoldGroup('singleton', (
            F.ParamAxis('a.weight', 0),
            F.ParamAxis('b.weight', 0),
            F.ParamAxis('c.weight', 0),
        ))
        params = {
            'a.weight': torch.randn(4, 3, 3, 3),
            'b.weight': torch.randn(4),
            'c.weight': torch.randn(4, 2, 3, 3),
        }
        labels = torch.arange(4)
        merge = F.build_merge_matrix(labels, 4, 4)
        merged = F.merge_group_params(params, group, merge)
        for name, tensor in params.items():
            self.assertTrue(torch.equal(merged[name], tensor), name)


class TestSingletonAndBypass(unittest.TestCase):
    def test_singleton_preserves_model_exactly(self):
        model = build_model()
        labels = singleton_labels_for(model, F.resnet18_groups())
        folded, diagnostics = F.apply_fold_labels(
            model, labels, preserve_fc_bias=True)
        self.assertFalse(diagnostics['bypassed'])
        self.assertEqual(diagnostics['nparams_after'], diagnostics['nparams_before'])
        self.assertTrue(diagnostics['all_params_finite'])
        self.assertTrue(diagnostics['fc_bias']['equal'])
        for name, group in zip(EXPECTED_GROUPS, diagnostics['groups']):
            self.assertEqual(group['name'], name)
            self.assertEqual(group['n_clusters'], group['n_channels'])
            self.assertEqual(min(group['cluster_counts']), 1)

        original = model.state_dict()
        rebuilt = folded.state_dict()
        self.assertEqual(set(original), set(rebuilt))
        for name in original:
            self.assertEqual(tuple(original[name].shape), tuple(rebuilt[name].shape))
            self.assertTrue(torch.equal(original[name], rebuilt[name]), name)

        inputs = make_inputs()
        self.assertTrue(torch.equal(forward(model, inputs), forward(folded, inputs)))

    def test_ratio_zero_is_bypass_with_fc_bias(self):
        model = build_model()
        folded, diagnostics = F.fold(
            model, ratio=0.0, seed=42, preserve_fc_bias=False)
        self.assertTrue(diagnostics['bypassed'])
        self.assertEqual(diagnostics['groups'], [])
        self.assertEqual(diagnostics['fc_bias']['policy'], 'bypass_unchanged')
        self.assertTrue(diagnostics['fc_bias']['equal'])
        self.assertEqual(diagnostics['nparams_after'], diagnostics['nparams_before'])
        self.assertTrue(diagnostics['all_params_finite'])
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, folded.state_dict()[name]), name)
        self.assertIsNot(folded, model)


class TestSyntheticFold(unittest.TestCase):
    def test_random_labels_forward_shape_and_finite(self):
        model = build_model()
        labels = labels_for(model, F.resnet18_groups(), keep_ratio=0.7, seed=7)
        folded, diagnostics = F.apply_fold_labels(
            model, labels, preserve_fc_bias=True)
        self.assertEqual(diagnostics['groups'][0]['cluster_counts'][0] > 0, True)
        for group in diagnostics['groups']:
            self.assertEqual(min(group['cluster_counts']), 1)
            self.assertEqual(
                sum(group['cluster_counts']), group['n_channels'])
        self.assertLess(diagnostics['nparams_after'], diagnostics['nparams_before'])
        self.assertTrue(diagnostics['all_params_finite'])
        logits = forward(folded, make_inputs())
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_residual_shapes_consistent(self):
        model = build_model()
        labels = labels_for(model, F.resnet18_groups(), keep_ratio=0.7, seed=11)
        folded, _ = F.apply_fold_labels(model, labels, preserve_fc_bias=True)

        self.assertIsInstance(folded.maxpool, nn.Identity)
        self.assertEqual(folded.conv1.kernel_size, (3, 3))
        self.assertEqual(folded.conv1.stride, (1, 1))
        self.assertEqual(folded.conv1.padding, (1, 1))
        self.assertEqual(folded.conv1.out_channels, folded.bn1.num_features)

        stage_outputs = []
        previous_output = folded.conv1.out_channels
        for stage in range(1, 5):
            blocks = getattr(folded, f'layer{stage}')
            widths = set()
            for index, block in enumerate(blocks):
                conv1, bn1 = block.conv1, block.bn1
                conv2, bn2 = block.conv2, block.bn2
                self.assertEqual(conv1.out_channels, bn1.num_features)
                self.assertEqual(conv2.in_channels, bn1.num_features)
                self.assertEqual(conv2.out_channels, bn2.num_features)
                self.assertEqual(conv1.groups, 1)
                self.assertEqual(conv2.groups, 1)
                widths.add(conv2.out_channels)
                if index == 0:
                    self.assertEqual(conv1.in_channels, previous_output)
                if block.downsample is not None:
                    down_conv, down_bn = block.downsample[0], block.downsample[1]
                    self.assertEqual(down_conv.in_channels, conv1.in_channels)
                    self.assertEqual(down_conv.out_channels, conv2.out_channels)
                    self.assertEqual(down_bn.num_features, conv2.out_channels)
                    self.assertEqual(down_conv.stride, conv1.stride)
                else:
                    self.assertEqual(conv1.in_channels, conv2.out_channels)
                    self.assertEqual(conv1.stride, (1, 1))
                    self.assertEqual(conv2.stride, (1, 1))
            self.assertEqual(len(widths), 1)
            previous_output = widths.pop()
            stage_outputs.append(previous_output)
        self.assertEqual(folded.fc.in_features, stage_outputs[-1])
        self.assertEqual(folded.fc.out_features, 10)

    def test_rebuilt_bn_uses_source_defaults(self):
        model = build_model()
        labels = labels_for(model, F.resnet18_groups(), keep_ratio=0.7, seed=13)
        folded, _ = F.apply_fold_labels(model, labels, preserve_fc_bias=True)
        for name, module in folded.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertAlmostEqual(module.eps, 1e-5)
                self.assertEqual(module.momentum, 0.1)
                self.assertEqual(module.num_batches_tracked.item(), 0)
                self.assertTrue(module.affine)
                self.assertTrue(module.track_running_stats)

    def test_fc_bias_policies(self):
        model = build_model()
        self.assertTrue(bool((model.fc.bias != 0).any()))
        labels = labels_for(model, F.resnet18_groups(), keep_ratio=0.7, seed=17)

        official_a, diag_a = F.apply_fold_labels(
            model, labels, preserve_fc_bias=False, seed=42)
        official_b, _ = F.apply_fold_labels(
            model, labels, preserve_fc_bias=False, seed=42)
        self.assertEqual(diag_a['fc_bias']['policy'], 'regenerated_official_style')
        self.assertFalse(diag_a['fc_bias']['equal'])
        self.assertTrue(torch.equal(official_a.fc.bias, official_b.fc.bias))
        self.assertTrue(bool(torch.isfinite(official_a.fc.bias).all()))

        preserved, diag_p = F.apply_fold_labels(
            model, labels, preserve_fc_bias=True, seed=42)
        self.assertEqual(diag_p['fc_bias']['policy'], 'preserved')
        self.assertTrue(diag_p['fc_bias']['equal'])
        self.assertTrue(torch.equal(preserved.fc.bias, model.fc.bias))

    def test_state_dict_roundtrip(self):
        model = build_model()
        labels = labels_for(model, F.resnet18_groups(), keep_ratio=0.7, seed=19)
        folded, _ = F.apply_fold_labels(model, labels, preserve_fc_bias=True)
        buffer = io.BytesIO()
        torch.save(folded.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location='cpu', weights_only=True)

        reloaded, _ = F.apply_fold_labels(
            build_model(), labels, preserve_fc_bias=True)
        reloaded.load_state_dict(state, strict=True)
        inputs = make_inputs(seed=3)
        self.assertTrue(torch.equal(forward(folded, inputs), forward(reloaded, inputs)))
        for name, tensor in folded.state_dict().items():
            self.assertTrue(torch.equal(tensor, reloaded.state_dict()[name]), name)


class TestHkmeansIntegration(unittest.TestCase):
    @unittest.skipUnless(F.hkmeans_available(), 'brak hkmeans/sklearn')
    def test_cluster_tiny_features_no_empty_clusters(self):
        centers = torch.arange(10, dtype=torch.float32).reshape(10, 1)
        features = torch.cat([
            centers + 0.01 * torch.randn(10, 4, generator=torch.Generator().manual_seed(0)),
            centers + 0.01 * torch.randn(10, 4, generator=torch.Generator().manual_seed(1)),
        ], dim=0)
        labels = F.cluster_group_features(features, n_clusters=10)
        k = F.validate_cluster_labels(labels, 20, 10)
        self.assertEqual(k, 10)
        # HKMeans does not guarantee equally sized clusters; require nonempty.
        self.assertGreaterEqual(int(torch.bincount(labels).min()), 1)

    @unittest.skipUnless(
        F.hkmeans_available() and os.environ.get('E012_RUN_HKMEANS') == '1',
        'ustaw E012_RUN_HKMEANS=1 po instalacji hkmeans/sklearn')
    def test_fold_ratio_03_hkmeans(self):
        model = build_model()
        folded, diagnostics = F.fold(
            model, ratio=0.3, seed=42, preserve_fc_bias=False)
        self.assertFalse(diagnostics['bypassed'])
        for group in diagnostics['groups']:
            self.assertEqual(min(group['cluster_counts']), 1)
            self.assertEqual(
                sum(group['cluster_counts']), group['n_channels'])
        logits = forward(folded, make_inputs(seed=5))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(diagnostics['all_params_finite'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
