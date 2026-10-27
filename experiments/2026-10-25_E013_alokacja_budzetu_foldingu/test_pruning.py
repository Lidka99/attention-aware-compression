"""E013: testy CPU pruningu magnitudowego ResNet18 (pruning.py), bez GPU.

Eksperyment: 2026-10-25_E013_alokacja_budzetu_foldingu; pełny opis: OPIS.md.
Testy sprawdzają ręcznie policzone normy wierszy i stabilne remisy, zachowanie
biasu fc, dokładny bypass ``ratio=0``, budżety parametrów E012
(9018358, 5459866, 2797610), forward dla ``ratio`` 0.1/0.3/0.5, kształty
residualne, wycinanie BN, serializację diagnostyki oraz to, że ocena conv
korzysta z oryginalnych wierszy przed obcięciem wejścia.

Model budujemy przez ``build_cifar_resnet18`` z E012, wczytywany po ścieżce.
Nie zmieniamy plików E012 i nie pobieramy danych ani wag.

Uruchomienie z folderu E013 na CPU z jednym wątkiem:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES='' python -B -m unittest -v test_pruning
"""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pruning as P

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

E012_FOLDING_PATH = (
    HERE.parent / '2026-10-24_E012_folding_referencja' / 'folding.py'
)


def _load_module(path, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Nie można wczytać modułu {module_name} z {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


E012_FOLDING = _load_module(E012_FOLDING_PATH, '_e013_pruning_e012_folding')

BASE_PARAMS = 11173962
EXPECTED_BUDGETS = {
    0.1: 9018358,
    0.3: 5459866,
    0.5: 2797610,
}


def build_model():
    return E012_FOLDING.build_cifar_resnet18()


def make_inputs(seed=0, batch_size=2):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, 32, 32, generator=generator)


def forward(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(inputs).detach().clone()


class TestMagnitudeHelpers(unittest.TestCase):
    def test_row_norms_manual_l1_and_l2(self):
        weight = torch.tensor([
            [[[3.0, 4.0]]],
            [[[0.0, 2.0]]],
            [[[0.0, -2.0]]],
            [[[-5.0, 0.0]]],
        ])
        self.assertTrue(torch.allclose(
            P._row_norms(weight, 1),
            torch.tensor([7.0, 2.0, 2.0, 5.0]),
            rtol=0.0, atol=0.0))
        self.assertTrue(torch.allclose(
            P._row_norms(weight, 2),
            torch.tensor([5.0, 2.0, 2.0, 5.0]),
            rtol=0.0, atol=0.0))

    def test_keep_indices_stable_ties_lower_index_wins(self):
        scores = torch.tensor([5.0, 2.0, 2.0, 5.0])
        self.assertEqual(P._keep_indices(scores, 2).tolist(), [0, 3])
        self.assertEqual(P._keep_indices(scores, 3).tolist(), [0, 3, 1])
        scores = torch.tensor([1.0, 3.0, 3.0, 2.0])
        self.assertEqual(P._keep_indices(scores, 3).tolist(), [1, 2, 3])

    def test_keep_count_floor_and_min_channels(self):
        self.assertEqual(P._keep_count(64, 0.1), 57)
        self.assertEqual(P._keep_count(128, 0.3), 89)
        self.assertEqual(P._keep_count(512, 0.5), 256)
        self.assertEqual(P._keep_count(64, 1.0), 1)
        self.assertEqual(P._keep_count(1, 0.9), 1)

    def test_mag1_and_mag2_differ_in_selection(self):
        weight = torch.tensor([
            [[[1.0, 1.0]]],
            [[[2.0, 0.0]]],
        ])
        mag1_scores = P._row_norms(weight, 1)
        mag2_scores = P._row_norms(weight, 2)
        self.assertEqual(P._keep_indices(mag1_scores, 1).tolist(), [0])
        self.assertEqual(P._keep_indices(mag2_scores, 1).tolist(), [1])


class TestRatioZeroBypass(unittest.TestCase):
    def test_ratio_zero_is_exact_bypass(self):
        model = build_model()
        with torch.no_grad():
            model.fc.bias.copy_(torch.linspace(-2.0, 2.0, 10))
        before = {
            name: tensor.clone() for name, tensor in model.state_dict().items()
        }

        pruned, diagnostics = P.prune(model, 0.0, p=2)

        self.assertIsNot(pruned, model)
        self.assertTrue(diagnostics['bypassed'])
        self.assertEqual(diagnostics['modules'], [])
        self.assertEqual(diagnostics['groups'], [])
        self.assertEqual(
            diagnostics['nparams_after'], diagnostics['nparams_before'])
        self.assertEqual(diagnostics['nparams_before'], BASE_PARAMS)
        self.assertTrue(diagnostics['all_params_finite'])
        self.assertTrue(diagnostics['fc']['bypassed'])
        self.assertTrue(diagnostics['fc']['bias_preserved'])
        self.assertTrue(torch.equal(pruned.fc.bias, model.fc.bias))

        for name, tensor in pruned.state_dict().items():
            self.assertEqual(tensor.device.type, 'cpu')
            self.assertTrue(torch.equal(tensor, before[name]), name)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)


class TestBudgets(unittest.TestCase):
    def test_parameter_budgets_match_e012(self):
        for ratio, expected in EXPECTED_BUDGETS.items():
            for p in (1, 2):
                with self.subTest(ratio=ratio, p=p):
                    model = build_model()
                    before = {
                        name: tensor.clone()
                        for name, tensor in model.state_dict().items()
                    }
                    pruned, diagnostics = P.prune(model, ratio, p=p)
                    self.assertEqual(diagnostics['nparams_before'], BASE_PARAMS)
                    self.assertEqual(diagnostics['nparams_after'], expected)
                    self.assertEqual(P.count_parameters(pruned), expected)
                    self.assertTrue(diagnostics['all_params_finite'])
                    self.assertEqual(
                        diagnostics['method'], 'MAG1' if p == 1 else 'MAG2')
                    for name, tensor in pruned.state_dict().items():
                        self.assertEqual(tensor.device.type, 'cpu')
                    for name, tensor in model.state_dict().items():
                        self.assertTrue(torch.equal(tensor, before[name]), name)


class TestFcBias(unittest.TestCase):
    def test_fc_bias_is_preserved_full(self):
        model = build_model()
        bias = torch.linspace(-1.0, 1.0, 10)
        with torch.no_grad():
            model.fc.bias.copy_(bias)
        old_fc = copy.deepcopy(model.fc)

        pruned, diagnostics = P.prune(model, 0.3, p=2)

        self.assertEqual(tuple(pruned.fc.weight.shape), (10, pruned.fc.in_features))
        self.assertEqual(tuple(pruned.fc.bias.shape), (10,))
        self.assertTrue(torch.equal(pruned.fc.bias, old_fc.bias))
        self.assertTrue(torch.equal(pruned.fc.bias, bias))
        self.assertTrue(diagnostics['fc']['bias_preserved'])

        keep = torch.tensor(diagnostics['fc']['in_keep'], dtype=torch.int64)
        self.assertEqual(pruned.fc.in_features, int(keep.numel()))
        self.assertTrue(torch.equal(
            pruned.fc.weight,
            old_fc.weight.detach().index_select(1, keep)))


class TestForward(unittest.TestCase):
    def test_forward_all_budgets_mag2(self):
        for ratio in EXPECTED_BUDGETS:
            with self.subTest(ratio=ratio):
                pruned, diagnostics = P.prune(build_model(), ratio, p=2)
                inputs = make_inputs(seed=int(round(ratio * 100)) + 1)
                logits = forward(pruned, inputs)
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertTrue(diagnostics['all_params_finite'])

    def test_forward_mag1(self):
        pruned, diagnostics = P.prune(build_model(), 0.3, p=1)
        logits = forward(pruned, make_inputs(seed=31))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertEqual(diagnostics['method'], 'MAG1')

    def test_ratio_one_keeps_at_least_one_channel(self):
        pruned, diagnostics = P.prune(build_model(), 1.0, p=2)
        self.assertEqual(pruned.conv1.out_channels, 1)
        for stage in range(1, 5):
            for block in getattr(pruned, f'layer{stage}'):
                self.assertGreaterEqual(block.conv1.out_channels, 1)
                self.assertGreaterEqual(block.conv2.out_channels, 1)
                if block.downsample is not None:
                    self.assertGreaterEqual(
                        block.downsample[0].out_channels, 1)
        self.assertEqual(pruned.fc.in_features, 1)
        logits = forward(pruned, make_inputs(seed=41, batch_size=1))
        self.assertEqual(tuple(logits.shape), (1, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(diagnostics['all_params_finite'])


class TestResidualShapes(unittest.TestCase):
    def test_residual_and_downsample_shapes(self):
        pruned, _ = P.prune(build_model(), 0.3, p=2)

        self.assertIsInstance(pruned.maxpool, nn.Identity)
        self.assertEqual(pruned.conv1.kernel_size, (3, 3))
        self.assertEqual(pruned.conv1.stride, (1, 1))
        self.assertEqual(pruned.conv1.padding, (1, 1))
        self.assertEqual(pruned.conv1.out_channels, pruned.bn1.num_features)

        previous_width = pruned.conv1.out_channels
        for stage in range(1, 5):
            blocks = getattr(pruned, f'layer{stage}')
            widths = set()
            for index, block in enumerate(blocks):
                self.assertEqual(block.conv1.out_channels, block.bn1.num_features)
                self.assertEqual(block.conv2.in_channels, block.bn1.num_features)
                self.assertEqual(block.conv2.out_channels, block.bn2.num_features)
                self.assertEqual(block.conv1.groups, 1)
                self.assertEqual(block.conv2.groups, 1)
                widths.add(block.conv2.out_channels)

                if index == 0:
                    self.assertEqual(block.conv1.in_channels, previous_width)
                else:
                    self.assertEqual(
                        block.conv1.in_channels,
                        blocks[index - 1].conv2.out_channels)

                if block.downsample is not None:
                    down_conv, down_bn = block.downsample[0], block.downsample[1]
                    self.assertEqual(down_conv.in_channels, block.conv1.in_channels)
                    self.assertEqual(
                        down_conv.out_channels, block.conv2.out_channels)
                    self.assertEqual(
                        down_bn.num_features, block.conv2.out_channels)
                    self.assertEqual(down_conv.stride, block.conv1.stride)
                else:
                    self.assertEqual(
                        block.conv1.in_channels, block.conv2.out_channels)
                    self.assertEqual(block.conv1.stride, (1, 1))
                    self.assertEqual(block.conv2.stride, (1, 1))

            self.assertEqual(len(widths), 1)
            previous_width = widths.pop()

        self.assertEqual(pruned.fc.in_features, previous_width)
        self.assertEqual(pruned.fc.out_features, 10)


class TestBatchNormSlicing(unittest.TestCase):
    def test_bn_slices_tensors_and_uses_source_defaults(self):
        model = build_model()
        original_bns = {
            name: copy.deepcopy(module)
            for name, module in model.named_modules()
            if isinstance(module, nn.BatchNorm2d)
        }

        pruned, diagnostics = P.prune(model, 0.3, p=2)
        entries = [
            entry for entry in diagnostics['modules']
            if entry['kind'] == 'batch_norm'
        ]
        self.assertEqual(len(entries), 20)

        for entry in entries:
            name = entry['name']
            keep = torch.tensor(entry['keep'], dtype=torch.int64)
            old = original_bns[name]
            new = P.get_module_by_name(pruned, name)
            self.assertIsInstance(new, nn.BatchNorm2d)
            self.assertEqual(new.num_features, int(keep.numel()))
            self.assertAlmostEqual(new.eps, 1e-5)
            self.assertEqual(new.momentum, 0.1)
            self.assertTrue(new.affine)
            self.assertTrue(new.track_running_stats)
            self.assertEqual(int(new.num_batches_tracked.item()), 0)
            self.assertTrue(torch.equal(
                new.weight.detach(),
                old.weight.detach().index_select(0, keep)))
            self.assertTrue(torch.equal(
                new.bias.detach(),
                old.bias.detach().index_select(0, keep)))
            self.assertTrue(torch.equal(
                new.running_mean,
                old.running_mean.detach().index_select(0, keep)))
            self.assertTrue(torch.equal(
                new.running_var,
                old.running_var.detach().index_select(0, keep)))


class TestScoreBeforeInputSlice(unittest.TestCase):
    def test_conv_scores_use_original_rows_before_input_slice(self):
        model = build_model()
        with torch.no_grad():
            for index in range(model.conv1.out_channels):
                model.conv1.weight[index].fill_(
                    float(model.conv1.out_channels - index))

            conv = model.layer2[0].conv1
            conv.weight.zero_()
            conv.weight[0, 32:, :, :] = 100.0
            conv.weight[1:, :32, :, :] = 1.0

        pruned, diagnostics = P.prune(model, 0.5, p=2)
        entry = next(
            item for item in diagnostics['modules']
            if item['name'] == 'layer2.0.conv1')

        self.assertEqual(entry['in_channels_before'], 64)
        self.assertEqual(entry['in_channels_after'], 32)
        self.assertEqual(entry['out_keep'], list(range(64)))
        self.assertEqual(pruned.layer2[0].conv1.out_channels, 64)


class TestDiagnosticsSerialization(unittest.TestCase):
    def test_diagnostics_are_strictly_json_serializable(self):
        pruned, diagnostics = P.prune(build_model(), 0.3, p=2)
        text = json.dumps(diagnostics, allow_nan=False)
        restored = json.loads(text)

        self.assertEqual(restored['experiment'], P.EXPERIMENT)
        self.assertEqual(restored['reference_commit'], P.REFERENCE_COMMIT)
        self.assertEqual(restored['ratio'], 0.3)
        self.assertEqual(restored['p'], 2.0)
        self.assertEqual(restored['method'], 'MAG2')
        self.assertEqual(restored['nparams_after'], EXPECTED_BUDGETS[0.3])
        self.assertTrue(restored['all_params_finite'])
        self.assertTrue(restored['fc']['bias_preserved'])
        self.assertEqual(len(restored['groups']), 5)

        for entry in restored['modules']:
            if entry['kind'] == 'conv':
                self.assertIsInstance(entry['out_keep'], list)
                self.assertGreaterEqual(len(entry['out_keep']), 1)
            if entry['kind'] == 'batch_norm':
                self.assertIsInstance(entry['keep'], list)
                self.assertEqual(entry['num_batches_tracked'], 0)

        self.assertEqual(
            restored['nparams_after'], P.count_parameters(pruned))


class TestInvalidArguments(unittest.TestCase):
    def test_invalid_ratio_and_p(self):
        model = build_model()
        with self.assertRaises(ValueError):
            P.prune(model, -0.1, p=2)
        with self.assertRaises(ValueError):
            P.prune(model, 1.1, p=2)
        with self.assertRaises(ValueError):
            P.prune(model, 0.3, p=0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
