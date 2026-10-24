"""E012: ograniczony test CPU pojedynczej operacji odbudowy Linear (fc) i biasu.

Eksperyment: 2026-10-24_E012_folding_referencja; pełny opis: OPIS.md.
Wejścia: opcjonalny checkpoint E011 (--checkpoint, domyślnie ostatni znany
seed42), seed i liczba syntetycznych obrazów; bez danych CIFAR-10, bez sieci,
bez GPU. Wyjścia: JSON na stdout oraz opcjonalnie --output; zawiera hash
checkpointu, seed, wersje środowiska i maksymalne różnice logitów/biasu.

Test dotyczy WYŁĄCZNIE pojedynczej operacji odbudowy warstwy Linear
z compression/base_resnet.py (commit 694691556e8314a835178a741938dd3d4d80c171,
metoda _rebuild_module): nowy Linear(in, out), kopia weight, bias pominięty,
bo oficjalny folding nie umieszcza biasu w param_dict. Nie testuje całej
ścieżki foldingu przy zerowej kompresji: ta ścieżka może w ogóle pominąć
odbudowę fc, więc wynik nie mówi, czy logity oficjalnego pipeline'u z r=0
się zmieniają.
"""
import argparse
import copy
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models import resnet18

EXPERIMENT = '2026-10-24_E012_folding_referencja'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL_CHECKPOINT = ROOT / (
    'runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012/seed_42/last.pt')
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def load_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state_dict = payload.get('state_dict', payload)
    model.load_state_dict(state_dict, strict=True)
    metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
    return metadata


def rebuild_linear_fc(old_fc, param_dict, device):
    """Kopia ścieżki Linear z _rebuild_module (base_resnet.py)."""
    weight = param_dict['weight']
    new_fc = nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    new_fc.weight.data.copy_(weight)
    if 'bias' in param_dict:
        new_fc.bias.data.copy_(param_dict['bias'])
    return new_fc


def make_inputs(seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, 32, 32, generator=generator)


def forward_logits(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(inputs).detach().clone()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='E012: CPU test pojedynczej odbudowy fc (weight tak, bias nie).')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='checkpoint E011 (state_dict + metadata)')
    parser.add_argument('--output', type=Path, default=None,
                        help='opcjonalny plik JSON z wynikiem')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='seed syntetycznych wejść i inicjalizacji nowego fc')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='liczba syntetycznych obrazów 3x32x32')
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cpu')
    checkpoint = args.checkpoint
    if checkpoint is None and CANONICAL_CHECKPOINT.is_file():
        checkpoint = CANONICAL_CHECKPOINT
    checkpoint_metadata = {}
    checkpoint_sha256 = None
    if checkpoint is not None:
        require(Path(checkpoint).is_file(), f'Brak checkpointu: {checkpoint}')
        checkpoint_sha256 = sha256_file(checkpoint)

    model = build_model().to(device)
    if checkpoint is not None:
        checkpoint_metadata = load_checkpoint(model, checkpoint)
    model.eval()

    inputs = make_inputs(args.seed, args.batch_size)
    original_logits = forward_logits(model, inputs)
    original_fc = model.fc

    official_model = copy.deepcopy(model)
    corrected_model = copy.deepcopy(model)

    torch.manual_seed(args.seed)
    official_param_dict = {'weight': original_fc.weight.detach().clone()}
    official_fc = rebuild_linear_fc(original_fc, official_param_dict, device)
    official_model.fc = official_fc

    corrected_param_dict = {
        'weight': original_fc.weight.detach().clone(),
        'bias': original_fc.bias.detach().clone(),
    }
    corrected_fc = rebuild_linear_fc(original_fc, corrected_param_dict, device)
    corrected_model.fc = corrected_fc

    official_logits = forward_logits(official_model, inputs)
    corrected_logits = forward_logits(corrected_model, inputs)

    weights_equal = (torch.equal(official_fc.weight, original_fc.weight)
                     and torch.equal(corrected_fc.weight, original_fc.weight))
    original_bias_nonzero = bool((original_fc.bias != 0).any())
    official_bias_differs = not torch.equal(official_fc.bias, original_fc.bias)
    corrected_bias_equal = torch.equal(corrected_fc.bias, original_fc.bias)

    logit_diff = official_logits - original_logits
    expected_shift = (official_fc.bias - original_fc.bias).detach().reshape(1, -1)
    expected_shift = expected_shift.expand_as(logit_diff)
    shift_residual = (logit_diff - expected_shift).abs()
    max_abs_shift_residual = float(shift_residual.max())
    max_abs_logit_diff = float(logit_diff.abs().max())
    max_abs_expected_shift = float(expected_shift.abs().max())

    require(tuple(official_logits.shape) == tuple(original_logits.shape),
            f'Zły kształt logitów: {tuple(official_logits.shape)}')
    require(tuple(corrected_logits.shape) == tuple(original_logits.shape),
            f'Zły kształt logitów poprawionych: {tuple(corrected_logits.shape)}')
    require(bool(torch.isfinite(original_logits).all())
            and bool(torch.isfinite(official_logits).all())
            and bool(torch.isfinite(corrected_logits).all()),
            'Niefinitywne logity')
    require(weights_equal, 'Wagi po odbudowie fc nie są równe oryginałowi')
    require(original_bias_nonzero, 'Oryginalny bias fc jest zerowy')
    require(official_bias_differs, 'Bias oficjalnej odbudowy nie różni się od oryginału')
    require(corrected_bias_equal, 'Poprawiona odbudowa nie skopiowała biasu')
    require(max_abs_shift_residual < 1e-5,
            f'Różnica logitów != nowy_bias - oryginalny_bias: '
            f'max_residual={max_abs_shift_residual:.3e}')
    require(torch.equal(corrected_logits, original_logits),
            'Poprawione logity nie są bitowo równe oryginalnym')

    result = {
        'experiment': EXPERIMENT,
        'scope': 'single_fc_rebuild_operation_only',
        'full_official_pipeline_tested': False,
        'full_pipeline_r0_logits_claim': 'not_tested_pipeline_may_bypass_fc_rebuild',
        'tested_operation': 'rebuild_linear_weight_copied_bias_not_copied',
        'result': 'passed_single_operation',
        'device': 'cpu',
        'cpu_threads': 1,
        'torch_num_threads': int(torch.get_num_threads()),
        'seed': int(args.seed),
        'batch_size': int(args.batch_size),
        'input_shape': list(inputs.shape),
        'checkpoint': str(checkpoint) if checkpoint is not None else None,
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_seed': checkpoint_metadata.get('seed'),
        'checkpoint_epoch': checkpoint_metadata.get('epoch'),
        'checkpoint_architecture': checkpoint_metadata.get('architecture'),
        'weights_equal': bool(weights_equal),
        'original_bias_nonzero': bool(original_bias_nonzero),
        'official_bias_differs': bool(official_bias_differs),
        'corrected_bias_equal': bool(corrected_bias_equal),
        'max_abs_official_logit_diff': max_abs_logit_diff,
        'max_abs_expected_bias_shift': max_abs_expected_shift,
        'max_abs_shift_residual': max_abs_shift_residual,
        'corrected_logits_bit_equal': bool(torch.equal(corrected_logits, original_logits)),
        'official_logits_bit_equal': bool(torch.equal(official_logits, original_logits)),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'platform': platform.platform(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
