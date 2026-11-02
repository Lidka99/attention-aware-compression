"""E020: wspólne helpery mechanizmu alokacji TILT (dane, folding, BN, metryki).

Eksperyment: 2026-10-30_E020_mechanizm_alokacji; pełny opis: OPIS.md.
Wejście: zamrożone ``configs/inventory.json``, ``configs/sensitivity.json``
i ``configs/ablation.json``; checkpointy E011/E015/E019; moduły danych tych
eksperymentów oraz operator E012/E014 importowany bez zmian.
Wyjście: helpery do budowy przebiegów, oceny pre/post-BN, pomiaru szkody
i predyktorów. Moduł nie modyfikuje E011–E019 i nie trenuje wag.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = '2026-10-30_E020_mechanizm_alokacji'
OPIS_PATH = 'experiments/2026-10-30_E020_mechanizm_alokacji/OPIS.md'
INVENTORY_PATH = HERE / 'configs' / 'inventory.json'
SENSITIVITY_PATH = HERE / 'configs' / 'sensitivity.json'
ABLATION_PATH = HERE / 'configs' / 'ablation.json'

E012_FOLDING = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/folding.py')
E012_EVALUATE = ROOT / (
    'experiments/2026-10-24_E012_folding_referencja/evaluate.py')
E014_FOLDING_BUDGET = ROOT / (
    'experiments/2026-10-27_E014_replikacja_alokacji/folding_budget.py')

GROUPS = (
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
)

_MODULES = {}


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def digest(path):
    """Zwraca SHA-256 pliku."""
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    """Wczytuje JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_atomic(path, value):
    """Zapisuje JSON atomowo, sortując klucze."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def utc_now():
    """Zwraca bieżący czas UTC w ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def git_head():
    """Zwraca skrót HEAD, jeśli repozytorium jest dostępne."""
    import subprocess

    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def git_dirty():
    """Zwraca liczbę zmodyfikowanych plików (bez treści)."""
    import subprocess

    try:
        text = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True)
        return len([line for line in text.splitlines() if line.strip()])
    except Exception:
        return -1


def load_module(path, name):
    """Wczytuje moduł z pełnej ścieżki (cache w ``sys.modules``)."""
    if name in _MODULES:
        return _MODULES[name]
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f'Nie można wczytać modułu {name} z {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


def e012_evaluate():
    """Zwraca moduł oceny E012."""
    return load_module(E012_EVALUATE, '_e020_e012_evaluate')


def folding_budget():
    """Zwraca moduł alokacji E014."""
    return load_module(E014_FOLDING_BUDGET, '_e020_e014_folding_budget')


def dataset_module(dataset):
    """Wczytuje moduł danych dla ``cifar10``, ``cifar100`` lub ``tiny``."""
    inventory = load_inventory()
    require(dataset in inventory['datasets'],
            f'Nieznany zbiór: {dataset!r}')
    path = ROOT / inventory['datasets'][dataset]['data_module']
    return load_module(path, f'_e020_data_{dataset}')


def load_inventory():
    """Wczytuje zamrożony inwentarz wejść."""
    inventory = read_json(INVENTORY_PATH)
    require(inventory.get('status') == 'frozen', 'Inwentarz musi być frozen')
    return inventory


def load_sensitivity_config():
    """Wczytuje zamrożony protokół pomiaru Tiny S1."""
    cfg = read_json(SENSITIVITY_PATH)
    require(cfg.get('status') == 'frozen', 'Config S1 musi być frozen')
    return cfg


def load_ablation_config():
    """Wczytuje zamrożony protokół ablacji ANTI-TILT."""
    cfg = read_json(ABLATION_PATH)
    require(cfg.get('status') == 'frozen', 'Config ablacji musi być frozen')
    return cfg


def load_dataset(dataset, split_path=None):
    """Wczytuje obrazy, etykiety, klasy i split dla zbioru.

    Zwraca ``(module, images, labels, classes, split)``. Nie używa testu
    ani rezerwy.
    """
    module = dataset_module(dataset)
    inventory = load_inventory()
    meta = inventory['datasets'][dataset]
    split = read_json(split_path or meta['split'])
    if isinstance(split.get('splits'), dict):
        split_map = split['splits']
    else:
        split_map = split
    require(set(split_map) >= {'train', 'dev'},
            'Split musi zawierać train i dev')
    if dataset == 'cifar10':
        images, labels, classes = module.load_cifar10_train(module.DATA_ROOT)
    elif dataset == 'cifar100':
        images, labels, classes = module.load_cifar100_train(module.DATA_ROOT)
    elif dataset == 'tiny':
        images, labels, classes = module.load_tiny_train(module.DATA_ROOT)
    else:
        raise ValueError(f'Nieznany zbiór: {dataset!r}')
    require(len(classes) == int(meta['num_classes']),
            'Niezgodna liczba klas')
    train_indices = np.asarray(split_map['train'], dtype=np.int64)
    dev_indices = np.asarray(split_map['dev'], dtype=np.int64)
    require(len(train_indices) == int(meta['train_count']),
            'Niezgodna liczba train')
    require(len(dev_indices) == int(meta['dev_count']),
            'Niezgodna liczba dev')
    return module, np.asarray(images), np.asarray(labels), classes, split


def load_checkpoint(dataset, seed, inventory=None):
    """Buduje model i wczytuje checkpoint dla zbioru i seeda."""
    inventory = inventory or load_inventory()
    meta = inventory['datasets'][dataset]
    require(str(seed) in meta['seeds'], f'Brak seeda {seed} dla {dataset}')
    entry = meta['seeds'][str(seed)]
    checkpoint = Path(entry['checkpoint'])
    require(checkpoint.is_file(), f'Brak checkpointu: {checkpoint}')
    require(digest(checkpoint) == entry['checkpoint_sha256'],
            'Hash checkpointu nie zgadza się z inwentarzem')
    module = dataset_module(dataset)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(isinstance(payload, dict)
            and isinstance(payload.get('state_dict'), dict),
            'Checkpoint bez state_dict')
    model = module.build_model()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    require(count_parameters(model) == int(meta['params']),
            'Niezgodna liczba parametrów checkpointu')
    return model, entry


def count_parameters(model):
    """Liczy parametry modelu."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def make_loaders(module, images, labels, split, batch_size, num_workers,
                 seed, bn_seed):
    """Buduje loader dev i loader BN (pełny train)."""
    e012 = e012_evaluate()
    cfg = {'batch_size': int(batch_size), 'num_workers': int(num_workers),
           'bn_seed': int(bn_seed)}
    split_map = split['splits'] if isinstance(split.get('splits'), dict) else split
    dev_indices = np.asarray(split_map['dev'], dtype=np.int64)
    train_indices = np.asarray(split_map['train'], dtype=np.int64)
    dev_loader = e012.make_eval_loader(
        module, images, labels, dev_indices, cfg, int(seed))
    bn_loader = e012.make_bn_loader(
        module, images, labels, train_indices, cfg)
    return dev_loader, bn_loader, train_indices


def evaluate(model, loader, device):
    """Ocenia model i zwraca metryki oraz logity/etykiety/indeksy."""
    return e012_evaluate().evaluate_model(
        model, loader, device, collect=True)


def adapt_bn(model, loader, device, expected_samples, seed):
    """Wykonuje korektę BN bez gradientów na pełnym train."""
    return e012_evaluate().adapt_bn(
        model, loader, device, int(expected_samples), int(seed))


def logit_metrics(candidate, baseline):
    """Metryki zmiany logitów (bez etykiet)."""
    candidate = torch.as_tensor(candidate, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    require(candidate.shape == baseline.shape, 'Niezgodne kształty logitów')
    diff = candidate - baseline
    logp_candidate = torch.log_softmax(candidate, dim=1)
    logp_baseline = torch.log_softmax(baseline, dim=1)
    return {
        'prediction_agreement': float(
            (candidate.argmax(1) == baseline.argmax(1)).float().mean()),
        'logit_mae': float(diff.abs().mean()),
        'logit_rmse': float(diff.pow(2).mean().sqrt()),
        'logit_max_abs': float(diff.abs().max()),
        'kl_baseline_candidate': float(torch.nn.functional.kl_div(
            logp_candidate, logp_baseline.exp(), reduction='batchmean')),
        'kl_candidate_baseline': float(torch.nn.functional.kl_div(
            logp_baseline, logp_candidate.exp(), reduction='batchmean')),
    }


def fold_with_ratios(source_cpu, ratios, fold_seed=42):
    """Wykonuje folding 12 grup z podanymi ratios."""
    require(len(ratios) == len(GROUPS), 'Oczekiwano 12 ratios')
    return folding_budget().fold_with_ratios(
        source_cpu, [float(value) for value in ratios],
        seed=int(fold_seed), deterministic=False, preserve_fc_bias=True)


def ratios_for_single_group(group_index, level):
    """Buduje wektor ratios z jedną skompresowaną grupą."""
    ratios = [0.0] * len(GROUPS)
    if group_index is not None:
        ratios[int(group_index)] = float(level)
    return ratios


def weight_reconstruction_error(source_cpu, group_index, diagnostics):
    """Względny błąd Frobeniusa projekcji parametrów jednej grupy."""
    if group_index is None:
        return 0.0
    reference = folding_budget().load_reference_folding()
    group = reference.resnet18_groups()[int(group_index)]
    entry = next(
        (value for value in diagnostics['groups']
         if value['name'] == group.name), None)
    require(entry is not None, f'Brak diagnostyki grupy {group.name}')
    n_channels = int(entry['n_channels'])
    n_clusters = int(entry['n_clusters'])
    if n_clusters >= n_channels:
        return 0.0
    labels = torch.as_tensor(entry['cluster_labels'], dtype=torch.int64)
    params = reference.collect_group_params(source_cpu, group)
    features = reference.concat_group_features(params, group).detach().float()
    merge = reference.build_merge_matrix(labels, n_clusters, n_channels)
    reconstructed = merge.transpose(0, 1) @ (merge @ features)
    numerator = torch.linalg.vector_norm(features - reconstructed)
    denominator = torch.linalg.vector_norm(features)
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator / denominator)


def group_parameter_counts(model):
    """Zwraca liczbę parametrów w każdej z 12 grup E012.

    Grupy E012 są rozłączne na osiach parametrów; sumaryczna liczba może
    różnić się od liczby parametrów modelu o fc i nieprzypisane bufory.
    """
    reference = folding_budget().load_reference_folding()
    result = []
    for group in reference.resnet18_groups():
        params = reference.collect_group_params(model, group)
        total = 0
        seen = set()
        for name, tensor in params.items():
            key = (name, id(tensor))
            if key in seen:
                continue
            seen.add(key)
            total += int(tensor.numel())
        result.append(total)
    return result


def single_group_saving(model, group_index, ratio):
    """Liczba parametrów zaoszczędzonych przez kompresję jednej grupy."""
    if group_index is None or float(ratio) == 0.0:
        return 0
    ratios = ratios_for_single_group(group_index, ratio)
    after = folding_budget().parameter_count_for_ratios(model, ratios)
    return int(count_parameters(model) - after)


def interpolate_damage(levels, damages, ratio, zero_anchor=True):
    """Liniowo interpoluje szkodę w funkcji ratio grupy.

    ``levels`` posortowane rosnąco; ``damages`` odpowiadają poziomom.
    Przy ``zero_anchor`` dokładamy punkt (0, 0). Poza zakresem ekstrapolacja
    liniowa ostatnim odcinkiem (TILT i tak mieści się w zakresie).
    """
    xs = [float(value) for value in levels]
    ys = [float(value) for value in damages]
    if zero_anchor:
        xs = [0.0] + xs
        ys = [0.0] + ys
    order = np.argsort(xs)
    xs = np.asarray(xs, dtype=np.float64)[order]
    ys = np.asarray(ys, dtype=np.float64)[order]
    ratio = float(ratio)
    if ratio <= xs[0]:
        return float(ys[0])
    if ratio >= xs[-1]:
        if len(xs) == 1:
            return float(ys[-1])
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + slope * (ratio - xs[-1]))
    return float(np.interp(ratio, xs, ys))


def environment():
    """Zwraca wersje środowiska istotne dla reprodukcji."""
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': __import__('torchvision').__version__,
        'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }


def sha256_sources():
    """Hashuje pliki źródłowe E020 i importowane operatory."""
    sources = {
        'e020_common.py': HERE / 'e020_common.py',
        'sensitivity_worker.py': HERE / 'sensitivity_worker.py',
        'ablation_worker.py': HERE / 'ablation_worker.py',
        'E012_folding.py': E012_FOLDING,
        'E012_evaluate.py': E012_EVALUATE,
        'E014_folding_budget.py': E014_FOLDING_BUDGET,
    }
    return {name: digest(path) for name, path in sources.items()
            if Path(path).is_file()}


def create_run_dir(kind, token, sources=None):
    """Tworzy unikalny katalog ``runs/E020/<UTC>_<git>_<kind>_<token>``."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = ROOT / 'runs' / 'E020' / f'{stamp}_{git_head()}_{kind}_{token}'
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'logs').mkdir()
    (run_dir / 'source').mkdir()
    if sources:
        for name, path in sources.items():
            target = run_dir / 'source' / name
            target.write_bytes(Path(path).read_bytes())
    return run_dir


def write_manifest(run_dir, config, items, assignments):
    """Zapisuje manifest przebiegu z przypisaniem itemów do GPU."""
    manifest = {
        'experiment': EXPERIMENT,
        'opis': OPIS_PATH,
        'status': 'running',
        'started_utc': utc_now(),
        'git_head': git_head(),
        'git_dirty_files': git_dirty(),
        'config': config,
        'items': items,
        'assignments': assignments,
        'environment': environment(),
        'official_test_used': False,
        'reserve_used': False,
        'source_sha256': sha256_sources(),
    }
    write_json_atomic(run_dir / 'manifest.json', manifest)
    return manifest


def finish_manifest(run_dir, status, summary_extra=None):
    """Zamyka manifest statusem i krótkim podsumowaniem."""
    path = run_dir / 'manifest.json'
    manifest = read_json(path)
    manifest['status'] = status
    manifest['finished_utc'] = utc_now()
    if summary_extra:
        manifest['summary'] = summary_extra
    write_json_atomic(path, manifest)
    return manifest


def shard_items(items, gpu_count=4):
    """Dzieli itemy round-robin na rozłączne, kompletne shardy GPU."""
    shards = {index: [] for index in range(int(gpu_count))}
    for offset, item in enumerate(items):
        shards[offset % int(gpu_count)].append(item)
    names = [item['name'] for shard in shards.values() for item in shard]
    require(len(names) == len(items) and len(set(names)) == len(items),
            'Shardy nie są rozłączne i kompletne')
    return shards


def gpu_status():
    """Zwraca listę słowników ze stanem GPU (nvidia-smi)."""
    import subprocess

    query = ('index,name,memory.used,memory.total,utilization.gpu')
    text = subprocess.check_output(
        ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
        text=True)
    result = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(',')]
        result.append({
            'index': int(parts[0]),
            'name': parts[1],
            'memory_used_mib': int(parts[2]),
            'memory_total_mib': int(parts[3]),
            'utilization_gpu': int(parts[4]),
        })
    return result


def preflight(gpu_count=4, max_used_mib=256, min_disk_gib=1.5):
    """Sprawdza cztery wolne GPU, środowisko i miejsce na dysku."""
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention',
            'Wymagane środowisko Conda attention')
    require('envs/attention' in sys.executable,
            f'Zły interpreter: {sys.executable}')
    status = gpu_status()
    require(len(status) >= int(gpu_count),
            f'Wymagane {gpu_count} GPU, widoczne {len(status)}')
    for gpu in status[:int(gpu_count)]:
        require(gpu['index'] == len([g for g in status[:int(gpu_count)]
                                     if g['index'] <= gpu['index']]) - 1,
                'GPU nie są w oczekiwanej kolejności 0..N-1')
        require(gpu['memory_used_mib'] <= int(max_used_mib),
                f"GPU {gpu['index']} zajęte: {gpu['memory_used_mib']} MiB")
        require(gpu['utilization_gpu'] == 0,
                f"GPU {gpu['index']} ma aktywne obliczenia")
    free_bytes = __import__('shutil').disk_usage(str(ROOT)).free
    free_gib = free_bytes / (1024 ** 3)
    require(free_gib >= float(min_disk_gib),
            f'Za mało miejsca: {free_gib:.2f} GiB < {min_disk_gib} GiB')
    return {'gpus': status, 'free_gib': free_gib}


def write_run_environment(run_dir, preflight_info):
    """Zapisuje środowisko, GPU i pakiety do katalogu przebiegu."""
    import subprocess

    write_json_atomic(run_dir / 'environment.json', {
        'environment': environment(),
        'preflight': preflight_info,
        'executable': sys.executable,
        'conda_default_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'cwd': str(ROOT),
        'git_head': git_head(),
        'git_dirty_files': git_dirty(),
    })
    try:
        (run_dir / 'pip-freeze.txt').write_text(
            subprocess.check_output(
                [sys.executable, '-m', 'pip', 'freeze'], text=True),
            encoding='utf-8')
    except Exception:
        pass
    try:
        (run_dir / 'nvidia-smi.txt').write_text(
            subprocess.check_output(['nvidia-smi'], text=True),
            encoding='utf-8')
    except Exception:
        pass


def launch_shards(run_dir, worker_script, config_path, items, gpu_count=4,
                  timeout_seconds=5400):
    """Uruchamia po jednym workerze na GPU; zwraca kody wyjścia."""
    import subprocess

    shards = shard_items(items, gpu_count)
    processes = []
    for gpu in range(int(gpu_count)):
        shard_path = run_dir / 'shards' / f'gpu_{gpu}.json'
        write_json_atomic(shard_path, shards[gpu])
        log_path = run_dir / 'logs' / f'gpu_{gpu}.log'
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        stream = log_path.open('w', encoding='utf-8')
        process = subprocess.Popen(
            [sys.executable, '-B', str(worker_script),
             '--run-dir', str(run_dir),
             '--config', str(config_path),
             '--items', str(shard_path),
             '--gpu', str(gpu)],
            cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT,
            text=True)
        processes.append((gpu, process, stream))
    deadline = __import__('time').monotonic() + float(timeout_seconds)
    codes = {}
    try:
        for gpu, process, stream in processes:
            remaining = max(0.0, deadline - __import__('time').monotonic())
            try:
                codes[gpu] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                codes[gpu] = -9
            finally:
                stream.close()
    finally:
        for gpu, process, stream in processes:
            if process.poll() is None:
                process.kill()
            try:
                stream.close()
            except Exception:
                pass
    return codes
