"""E019: testy CPU danych, modelu i operatora alokacji (bez GPU i treningu).

Eksperyment: 2026-10-29_E019_tinyimagenet_alokacja; pełny opis: OPIS.md.
Wejście: wyłącznie lokalny Tiny ImageNet, splits.json oraz kod E012/E014
importowany bez zmian. Wyjście: wynik kontroli na stdout; brak plików
wynikowych, brak GPU, brak treningu, bez danych dev. Sprawdza hashe danych,
podział 90 000/10 000/10 000, model 200 klas (11 279 112 parametrów),
transformacje, wspólny target per budżet (faktyczny UNIFORM) dla wszystkich
12 kombinacji 3 budżetów × 4 wariantów oraz odrzucanie smoke runów przez
folding.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_tiny as data  # noqa: E402

ROOT = data.ROOT
E012_FOLDING = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/folding.py')
E014_FOLDING_BUDGET = ROOT / (
    'experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py')
E014_PRUNING = ROOT / (
    'experiments/2026-10-27_E014_replikacja_alokacji/pruning.py')
WORKER = HERE / 'folding_worker.py'
LAUNCH_FOLDING = HERE / 'launch_folding.py'
PREPARE_FOLDING = HERE / 'prepare_folding_config.py'
ALLOCATION = HERE / 'configs' / 'allocation.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f'Nie można wczytać {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forward(model, seed=0, batch_size=2):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(batch_size, 3, 64, 64, generator=generator)
    model.eval()
    with torch.inference_mode():
        return model(inputs)


def check_smoke_rejection(launch_folding, prepare_folding):
    """Folding musi odrzucić manifest ukończonego smoke runu."""
    manifest = {
        'experiment': data.EXPERIMENT,
        'status': 'completed',
        'smoke': True,
        'data_only': False,
        'effective_epochs': 200,
        'official_test_used': False,
        'reserve_used': False,
        'seeds': [42, 43, 44, 45],
        'gpu_by_seed': {'42': 0, '43': 1, '44': 2, '45': 3},
    }
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        data.write_json_atomic(run / 'manifest.json', manifest)
        for label, function in (
                ('launch_folding.load_training_run',
                 launch_folding.load_training_run),
                ('prepare_folding_config.load_training_manifest',
                 prepare_folding.load_training_manifest)):
            try:
                function(run)
            except RuntimeError as error:
                require('smoke' in str(error).lower(),
                        f'{label}: oczekiwano odmowy z powodu smoke, '
                        f'otrzymano {error!r}')
            else:
                raise RuntimeError(f'{label} zaakceptował smoke run')
    return True


def main():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data.verify_source_files()
    images, labels, classes = data.load_tiny_train(data.DATA_ROOT)
    split = data.load_split(data.SPLITS_FILE)
    data.validate_split(split, labels, data.SPLIT_SEED)
    require(images.shape == (100000, 64, 64, 3), 'Zły kształt obrazów')
    require(len(classes) == 200, 'Zła liczba klas')
    train_indices = np.asarray(split['splits']['train'], dtype=np.int64)
    dev_indices = np.asarray(split['splits']['dev'], dtype=np.int64)
    reserve_indices = np.asarray(split['splits']['reserve'], dtype=np.int64)
    require(len(train_indices) == 90000 and len(dev_indices) == 10000
            and len(reserve_indices) == 10000, 'Złe rozmiary podziału')
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), 5000):
        chunk = images[train_indices[start:start + 5000]].astype(
            np.float64) / 255.0
        total += chunk.sum(axis=(0, 1, 2))
        squares += (chunk * chunk).sum(axis=(0, 1, 2))
        count += chunk.shape[0] * chunk.shape[1] * chunk.shape[2]
    measured_mean = total / count
    measured_var = squares / count - measured_mean ** 2
    measured_std = np.sqrt(np.maximum(measured_var, 0.0))
    require(np.allclose(measured_mean, data.NORMALIZATION_MEAN, atol=5e-7),
            'Zamrożona średnia nie zgadza się z 90k train')
    require(np.allclose(measured_std, data.NORMALIZATION_STD, atol=5e-7),
            'Zamrożone odchylenie nie zgadza się z 90k train')

    model = data.build_model()
    parameters = data.count_parameters(model)
    require(parameters == data.EXPECTED_PARAMETERS,
            f'Nieoczekiwana liczba parametrów: {parameters}')
    logits = forward(model)
    require(tuple(logits.shape) == (2, 200), 'Zły kształt logitów')
    require(logits.dtype == torch.float32
            and bool(torch.isfinite(logits).all()), 'Niefinitywne logity')

    from PIL import Image
    image = Image.fromarray(np.random.default_rng(0).integers(
        0, 256, size=(64, 64, 3), dtype=np.uint8))
    require(tuple(data.train_transform()(image).shape) == (3, 64, 64),
            'Zła transformacja train')
    require(tuple(data.dev_transform()(image).shape) == (3, 64, 64),
            'Zła transformacja dev')
    dataset = data.CifarTrainImages(
        images, labels, train_indices[:8], data.dev_transform())
    sample, target = dataset[0]
    require(tuple(sample.shape) == (3, 64, 64) and 0 <= target < 200,
            'Zły sample datasetu (kontrola bez danych dev)')

    E012 = load_module(E012_FOLDING, '_e019_check_e012_folding')
    FB = load_module(E014_FOLDING_BUDGET, '_e019_check_folding_budget')
    PRUNING = load_module(E014_PRUNING, '_e019_check_pruning')
    worker = load_module(WORKER, '_e019_check_folding_worker')
    LAUNCH = load_module(LAUNCH_FOLDING, '_e019_check_launch_folding')
    PREPARE = load_module(PREPARE_FOLDING, '_e019_check_prepare_folding')
    require(FB.group_names() == E012.group_order(),
            'Grupy E014 nie zgadzają się z E012')

    ratios = FB.allocation_from_tilt(FB.group_names(), base=0.5, tilt=0.30)
    require(len(ratios) == 12 and all(0.1 <= value <= 0.9 for value in ratios),
            'Zła alokacja TILT')
    folded, diagnostics = FB.fold_with_ratios(
        model, [0.5] * 12, seed=42, deterministic=True,
        preserve_fc_bias=True)
    require(0 < diagnostics['params'] < parameters, 'Zły budżet foldingu')
    require(bool(diagnostics['finite']), 'Niefinitywny folding')
    folded_logits = forward(folded)
    require(tuple(folded_logits.shape) == (2, 200),
            'Zły kształt logitów po foldingu')
    require(bool(torch.isfinite(folded_logits).all()),
            'Niefinitywne logity po foldingu')

    allocation = data.read_json(ALLOCATION)
    require(allocation.get('status') == 'frozen',
            'Config alokacji nie jest zamrożony')
    specs = LAUNCH.build_specs(allocation)
    require(len(specs) == 12, 'Oczekiwano 12 speców')
    budget_checks = []
    for budget in [float(value) for value in allocation['budgets']]:
        nominal = int(round((1.0 - budget) * parameters))
        budget_specs = [
            spec for spec in specs
            if abs(float(spec['budget']) - budget) < 1e-12]
        require(len(budget_specs) == 4,
                f'Budżet {budget}: oczekiwano 4 wariantów')
        reference = None
        for spec in budget_specs:
            resolved, solved_reference = worker.resolve_for_spec(
                model, spec, nominal)
            if reference is None:
                reference = int(solved_reference)
            require(int(solved_reference) == reference,
                    f"{spec['name']}: niespójny target UNIFORM")
            require(int(resolved['nominal_target_params']) == nominal,
                    f"{spec['name']}: zły nominalny cel")
            require(int(resolved['uniform_reference_params']) == reference,
                    f"{spec['name']}: zły wspólny target")
            require(int(resolved['target_params']) == reference,
                    f"{spec['name']}: target != faktyczny UNIFORM")
            params = int(resolved['params'])
            require(params <= reference,
                    f"{spec['name']}: przekroczono wspólny target")
            require(params <= nominal,
                    f"{spec['name']}: przekroczono nominalny cel")
            require(-1.0 <= float(resolved['delta_percent']) <= 0.0,
                    f"{spec['name']}: odchylenie od targetu poza tolerancją")
            folded_spec, spec_diagnostics, kind = worker.apply_allocation(
                model, spec, resolved,
                {'fold_seed': int(allocation['fold_seed'])})
            require(data.count_parameters(folded_spec) == params,
                    f"{spec['name']}: parametry po kompresji != solver")
            require(FB.load_reference_folding().state_dict_all_finite(
                folded_spec), f"{spec['name']}: niefinitywne parametry")
            require(tuple(forward(folded_spec).shape) == (2, 200),
                    f"{spec['name']}: zły kształt logitów")
            budget_checks.append({
                'budget': budget,
                'variant': spec['name'],
                'kind': kind,
                'nominal_target_params': nominal,
                'uniform_reference_params': reference,
                'target_params': reference,
                'params': params,
                'nominal_reduction_pct': 100.0 * (
                    parameters - nominal) / parameters,
                'target_reduction_pct': 100.0 * (
                    parameters - reference) / parameters,
                'realized_reduction_pct': 100.0 * (
                    parameters - params) / parameters,
                'delta_percent': float(resolved['delta_percent']),
                'nominal_delta_percent': float(
                    resolved['nominal_delta_percent']),
                'mag2_method': (
                    spec_diagnostics.get('method') if kind == 'magnitude'
                    else None),
            })
        uniform = next(
            item for item in budget_checks
            if item['budget'] == budget and item['variant'].endswith('_uniform'))
        require(uniform['params'] == reference,
                f'Budżet {budget}: UNIFORM != wspólny target')
        for item in budget_checks:
            if item['budget'] != budget:
                continue
            require(item['target_params'] == reference,
                    f"Budżet {budget}: {item['variant']} ma inny target")

    smoke_rejected = check_smoke_rejection(LAUNCH, PREPARE)

    print('E019_CPU_CHECKS=passed')
    print(json.dumps({
        'parameters': parameters,
        'num_classes': data.NUM_CLASSES,
        'train_count': int(len(train_indices)),
        'dev_count': int(len(dev_indices)),
        'reserve_count': int(len(reserve_indices)),
        'split_sha256': data.SPLITS_SHA256,
        'fold_uniform_50_params': int(diagnostics['params']),
        'hkmeans_available': bool(E012.hkmeans_available()),
        'smoke_run_rejected_by_folding': bool(smoke_rejected),
        'dev_images_loaded': False,
        'gpu_used': False,
        'budget_checks': budget_checks,
        'official_test_used': False,
        'reserve_used': False,
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
