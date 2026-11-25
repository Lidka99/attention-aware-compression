"""E048: trening ResNet18 od zera na CIFAR-100/Tiny, po jednym seedzie na GPU.

Czyta tylko kanoniczny cache/train i indeksy train/dev. Zapisuje manifest,
metryki, ostatnie checkpointy i audyt w nowym runs/E048/. Protokół: OPIS.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs/E048'
CONFIG = HERE / 'config.json'
HELPERS = {
    'cifar100': ROOT / 'experiments/2026-10-27_E015_cifar100_alokacja/data_cifar100.py',
    'tiny': ROOT / 'experiments/2026-10-29_E019_tinyimagenet_alokacja/data_tiny.py',
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = {ds: load_module(path, '_e048_' + ds) for ds, path in HELPERS.items()}
read = DATA['tiny'].read_json
write = DATA['tiny'].write_json_atomic
digest = DATA['tiny'].digest
utc = DATA['tiny'].utc_now
require = DATA['tiny'].require


def new_run(prefix):
    run = RUNS / f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{secrets.token_hex(4)}'
    run.mkdir(parents=True, exist_ok=False)
    return run


def hashes():
    paths = [Path(__file__), CONFIG, HERE / 'OPIS.md', HERE / 'URUCHOMIENIE.md', *HELPERS.values()]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def inputs():
    c, t = DATA['cifar100'], DATA['tiny']
    return {str(p.relative_to(ROOT)): digest(p) for p in
            (c.TRAIN_FILE, c.META_FILE, c.SPLITS_FILE, t.CACHE_TRAIN, t.SPLITS_FILE)}


def raw_data(dataset):
    source = DATA[dataset]
    load = source.load_tiny_train if dataset == 'tiny' else source.load_cifar100_train
    images, labels, _ = load()
    split = source.load_split()
    source.validate_split(split, labels)
    return images, labels, np.asarray(split['splits']['train']), np.asarray(split['splits']['dev'])


def model_for(dataset, cfg):
    spec = cfg['datasets'][dataset]
    if dataset == 'cifar100':
        return DATA[dataset].build_model()
    model = resnet18(weights=None, num_classes=spec['classes'])
    model.conv1 = nn.Conv2d(3, 64, 3, stride=spec['stem_stride'], padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.maxpool = nn.Identity()
    return model


def size_cost(model, size, classes):
    mac = []

    def hook(module, _args, out):
        per_output = module.weight[0].numel()
        mac.append(int(out[0].numel()) * per_output)

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode():
            output = model.eval()(torch.zeros(2, 3, size, size))
        require(output.shape == (2, classes) and torch.isfinite(output).all(), 'Błędny forward CPU')
    finally:
        for h in hooks:
            h.remove()
    return sum(p.numel() for p in model.parameters()), sum(mac)


def transform(dataset, cfg, train):
    source = DATA[dataset]
    ops = []
    if train:
        if dataset == 'tiny':
            spec = cfg['datasets'][dataset]
            ops = [T.RandomResizedCrop(spec['input_size'], scale=tuple(spec['crop_scale']), antialias=True)]
        else:
            ops = [T.RandomCrop(32, padding=4, padding_mode='reflect')]
        ops += [T.RandomHorizontalFlip(), T.RandAugment(
            num_ops=cfg['randaugment_ops'], magnitude=cfg['randaugment_magnitude'],
            interpolation=T.InterpolationMode.BILINEAR,
            fill=[round(255*x) for x in source.NORMALIZATION_MEAN])]
    return T.Compose(ops + [T.PILToTensor()])


def loader(dataset, images, labels, indices, cfg, seed, train):
    data = DATA[dataset].CifarTrainImages(images, labels, indices, transform(dataset, cfg, train))
    return DataLoader(data, batch_size=cfg['batch_per_gpu'], shuffle=train,
                      num_workers=cfg['workers_per_gpu'], pin_memory=True, persistent_workers=True,
                      generator=torch.Generator().manual_seed(seed))


def normalizer(dataset, device):
    source = DATA[dataset]
    mean = torch.tensor(source.NORMALIZATION_MEAN, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(source.NORMALIZATION_STD, device=device).reshape(1, 3, 1, 1)

    def normalize(x):
        return x.to(device, dtype=torch.float32, non_blocking=True,
                    memory_format=torch.channels_last).div_(255).sub_(mean).div_(std)
    return normalize


def mixers(cfg, classes):
    return (v2.MixUp(alpha=cfg['mixup_alpha'], num_classes=classes),
            v2.CutMix(alpha=cfg['cutmix_alpha'], num_classes=classes))


@torch.inference_mode()
def evaluate(model, data, norm, device):
    model.eval()
    loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    count = 0
    for x, y in data:
        y = y.to(device, non_blocking=True)
        z = model(norm(x))
        loss += F.cross_entropy(z, y, reduction='sum')
        correct += (z.argmax(1) == y).sum()
        count += len(y)
    hits, nll = int(correct), float(loss) / count
    require(count == len(data.dataset) and math.isfinite(nll) and 0 <= hits <= count, 'Błędna ewaluacja')
    return {'correct': hits, 'count': count, 'top1_pct': 100*hits/count, 'nll': nll}


def check():
    require(not torch.cuda.is_available(), "check wymaga CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = read(CONFIG)
    require(cfg['seeds'] == [42, 43, 44, 45] and cfg['wave_order'] == ['cifar100', 'tiny'] and
            cfg['batch_per_gpu'] == cfg['global_batch_per_model'] and cfg['gradient_accumulation'] == 1,
            'Niezgodna siatka protokołu')
    records = {}
    for ds in cfg['wave_order']:
        spec = cfg['datasets'][ds]
        image_size, classes = spec['input_size'], spec['classes']
        p, mac = size_cost(model_for(ds, cfg), image_size, classes)
        require((p, mac) == (spec['parameters'], spec['mac']), f'{ds}: nieoczekiwane P/MAC: {p}/{mac}')
        images, labels, train, dev = raw_data(ds)
        src = DATA[ds].CifarTrainImages(images, labels, train[:4], transform(ds, cfg, True))
        x = normalizer(ds, torch.device('cpu'))(torch.stack([src[i][0] for i in range(4)]))
        y = torch.tensor([src[i][1] for i in range(4)])
        for mixer in mixers(cfg, classes):
            a, b = mixer(x, y)
            require(a.shape == (4, 3, image_size, image_size) and torch.isfinite(a).all() and
                    b.shape == (4, classes) and torch.allclose(b.sum(1), torch.ones(4)), f'{ds}: augmentation FAIL')
        records[ds] = {'parameters': p, 'mac': mac, 'train': len(train), 'dev': len(dev)}
        del images, labels, train, dev, src
    run = new_run('cpu')
    write(run/'check.json', {'status': 'passed', 'utc': utc(), 'sources': hashes(),
                            'inputs': inputs(), 'datasets': records})
    print(f'CHECK={run.relative_to(ROOT)}', records, flush=True)


def epoch_lr(epoch, epochs, cfg):
    if epoch <= cfg['warmup_epochs']:
        return cfg['lr'] * epoch / cfg['warmup_epochs']
    t = (epoch-cfg['warmup_epochs']) / (epochs-cfg['warmup_epochs'])
    return cfg['min_lr'] + (cfg['lr']-cfg['min_lr']) * (1+math.cos(math.pi*t))/2


def worker(run, dataset, gpu):
    manifest = read(run/'manifest.json')
    cfg = read(run/'config.json')
    require(hashes() == manifest['sources'] and digest(run/'config.json') == manifest['config_sha256'], 'Niezgodna kopia wejść')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and torch.cuda.device_count() == 1, 'Złe mapowanie CUDA')
    seed = cfg['seeds'][gpu]
    smoke = manifest['mode'] == 'smoke'
    epochs = 1 if smoke else cfg['epochs']
    spec = cfg['datasets'][dataset]
    out = run / dataset / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'dataset': dataset, 'seed': seed, 'gpu': gpu, 'pid': os.getpid(), 'started_utc': utc()}
    write(out/'status.json', status)
    try:
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device('cuda:0')
        images, labels, train, dev = raw_data(dataset)
        source = DATA[dataset]
        train_probe = source.stratified_subset(train, labels, 2 if smoke else 20, 20261124)
        if smoke:
            train = source.stratified_subset(train, labels, 8, 20261124)
            dev = source.stratified_subset(dev, labels, 2, 20261124)
        train_dl = loader(dataset, images, labels, train, cfg, seed+1000, True)
        dev_dl = loader(dataset, images, labels, dev, cfg, seed+2000, False)
        norm = normalizer(dataset, device)
        model = model_for(dataset, cfg).to(device, memory_format=torch.channels_last)
        require(sum(p.numel() for p in model.parameters()) == spec['parameters'], 'Niezgodne P na GPU')
        opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'], nesterov=cfg['nesterov'])
        scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp_fp16'], init_scale=4096.)
        mix = mixers(cfg, spec['classes'])
        start = time.monotonic()
        for epoch in range(1, epochs+1):
            lr = epoch_lr(epoch, epochs, cfg)
            for group in opt.param_groups:
                group['lr'] = lr
            model.train()
            losses = torch.zeros((), device=device)
            count, mixed = 0, 0
            for x, y in train_dl:
                x = norm(x)
                y = y.to(device, non_blocking=True)
                if (smoke or epoch <= epochs-cfg['unmixed_tail_epochs']) and random.random() < cfg['mix_probability']:
                    x, y = mix[random.randrange(2)](x, y)
                    mixed += 1
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp_fp16']):
                    loss = F.cross_entropy(model(x), y, label_smoothing=cfg['label_smoothing'])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses += loss.detach() * len(y)
                count += len(y)
            train_loss = float(losses) / count
            require(count == len(train) and math.isfinite(train_loss) and scaler.get_scale() >= 1., f'{dataset}/{seed}/{epoch}: trening niestabilny')
            dev_metrics = evaluate(model, dev_dl, norm, device) if epoch in (1, epochs) or epoch % cfg['eval_every'] == 0 else None
            if dev_metrics is not None:
                meta = {'experiment': HERE.name, 'dataset': dataset, 'architecture': 'ResNet18', 'seed': seed,
                        'epoch': epoch, 'epochs': epochs, 'smoke': smoke, 'geometry': spec,
                        'config_sha256': manifest['config_sha256'], 'split_sha256': source.SPLITS_SHA256}
                torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'metadata': meta}, out/'last.pt.tmp')
                os.replace(out/'last.pt.tmp', out/'last.pt')
            row = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'train_count': count, 'mixed_batches': mixed,
                   'amp_scale': scaler.get_scale(), 'dev': dev_metrics, 'elapsed_seconds': time.monotonic()-start}
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        probe_dl = loader(dataset, images, labels, train_probe, cfg, seed+3000, False)
        probe = evaluate(model, probe_dl, norm, device)
        write(out/'summary.json', {**meta, 'dev': dev_metrics, 'clean_train_probe': probe,
                                  'checkpoint_sha256': digest(out/'last.pt'), 'wall_seconds': time.monotonic()-start,
                                  'gpu': gpu, 'peak_bytes': torch.cuda.max_memory_allocated()})
        status['status'] = 'completed'
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status['finished_utc'] = utc()
        write(out/'status.json', status)


def require_idle_gpus():
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'attention' and 'envs/attention' in sys.executable,
            'Wymagane Conda attention')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3' and torch.cuda.device_count() == 4,
            'Wymagane jawne CUDA_VISIBLE_DEVICES=0,1,2,3 i cztery GPU')
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    require(len(info) == 4, 'Inny zestaw urządzeń')
    for i, line in enumerate(info):
        index, name, memory, utilization = (v.strip() for v in line.split(','))
        require(int(index) == i and '2080 Ti' in name and int(memory) < 256 and int(utilization) == 0,
                f'GPU nie jest wolne lub ma inny model: {line}')
    require(shutil.disk_usage(ROOT).free > 1024**3, 'Mniej niż 1 GiB wolnego dysku')
    return info


def audit(run):
    m = read(run/'manifest.json')
    cfg = read(run/'config.json')
    smoke = m['mode'] == 'smoke'
    require(digest(run/'config.json') == m['config_sha256'], 'Zmieniony config')
    require(m['inputs'] == inputs(), 'Niezgodne wejścia danych')
    epochs = 1 if smoke else cfg['epochs']
    results = {}
    require(len(m['waves']) == 2 and all(w['exit_codes'] == [0]*4 for w in m['waves']), 'Brak 4GPU/exit0 obu fal')
    for ds in cfg['wave_order']:
        rows = []
        spec = cfg['datasets'][ds]
        expected_train = 8*spec['classes'] if smoke else DATA[ds].EXPECTED_TRAIN
        expected_dev = 2*spec['classes'] if smoke else DATA[ds].EXPECTED_DEV
        for gpu, seed in enumerate(cfg['seeds']):
            out = run/ds/f'seed_{seed}'
            summary = read(out/'summary.json')
            history = [json.loads(p) for p in (out/'metrics.jsonl').read_text().splitlines()]
            require(read(out/'status.json')['status'] == 'completed' and summary['epochs'] == summary['epoch'] == epochs and
                    summary['gpu'] == gpu and summary['seed'] == seed and summary['dataset'] == ds and summary['smoke'] == smoke and
                    summary['config_sha256'] == m['config_sha256'] and summary['split_sha256'] == DATA[ds].SPLITS_SHA256 and
                    summary['geometry'] == spec and digest(out/'last.pt') == summary['checkpoint_sha256'], f'{ds}/{seed}: niespójny checkpoint')
            require([r['epoch'] for r in history] == list(range(1, epochs+1)) and
                    all(r['train_count'] == expected_train and math.isfinite(r['train_loss']) for r in history) and
                    history[-1]['dev'] == summary['dev'] and summary['dev']['count'] == expected_dev and
                    summary['dev']['top1_pct'] == 100*summary['dev']['correct']/expected_dev,
                    f'{ds}/{seed}: niepełna lub niespójna historia')
            rows.append(summary)
        scores = [r['dev']['top1_pct'] for r in rows]
        diffs = None if smoke else [x-y for x, y in zip(scores, spec['baseline_pct'], strict=True)]
        mean = statistics.mean(scores)
        results[ds] = {'per_seed_pct': scores, 'mean_pct': mean, 'sample_sd_pp': statistics.stdev(scores),
                       'old_mean_pct': None if smoke else statistics.mean(spec['baseline_pct']),
                       'per_seed_delta_pp': diffs, 'positive_pairs': None if smoke else sum(x>0 for x in diffs),
                       'target_pct': spec['target_pct'],
                       'quality_gate_met': None if smoke else mean >= spec['target_pct'] and sum(x>0 for x in diffs) >= 3}
    report = {'status': 'passed', 'utc': utc(), 'mode': m['mode'], 'validated_checkpoints': 8,
              'official_test_used': False, 'reserve_used': False, 'results': results}
    write(run/'audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def supervise(run):
    m = read(run/'manifest.json')
    cfg = read(run/'config.json')
    start = time.monotonic()
    processes = []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        require(m['status'] == 'queued' and m['sources'] == hashes() and digest(run/'config.json') == m['config_sha256'],
                'Niezgodne lub użyte wejście supervise')
        m.update(status='running', started_utc=utc(), launcher_pid=os.getpid(), gpus=require_idle_gpus())
        write(run/'manifest.json', m)
        for ds in cfg['wave_order']:
            wave = {'dataset': ds, 'started_utc': utc(), 'workers': []}
            m['waves'].append(wave)
            for gpu, seed in enumerate(cfg['seeds']):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (run/'logs'/f'{ds}_seed_{seed}.log').open('w') as log:
                    p = subprocess.Popen([sys.executable, '-B', str(HERE/'train.py'), 'worker', '--run-dir', str(run),
                                          '--dataset', ds, '--gpu', str(gpu)], cwd=ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(p)
                wave['workers'].append({'gpu': gpu, 'seed': seed, 'pid': p.pid})
            write(run/'manifest.json', m)
            while True:
                codes = [p.poll() for p in processes]
                require(not any(c not in (None, 0) for c in codes), f'{ds}: worker failed {codes}')
                if codes == [0]*4:
                    break
                require(time.monotonic()-start < m['limit_seconds'], 'Limit czasu E048')
                time.sleep(2)
            wave.update(exit_codes=codes, finished_utc=utc())
            write(run/'manifest.json', m)
            processes = []
        audit(run)
        m['status'] = 'completed'
    except BaseException:
        m.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()
        if processes and m['waves']:
            m['waves'][-1]['exit_codes'] = [p.returncode for p in processes]
        m.update(finished_utc=utc(), wall_seconds=time.monotonic()-start)
        write(run/'manifest.json', m)


def launch(mode, check_dir, smoke_dir, detach):
    gpus = require_idle_gpus()
    checked = read(check_dir/'check.json')
    require(checked['status'] == 'passed' and checked['sources'] == hashes() and checked['inputs'] == inputs(), 'CPU check nieaktualny')
    if mode == 'train':
        require(smoke_dir is not None, 'Brak --smoke-run')
        sm = read(smoke_dir/'manifest.json')
        require(sm['mode'] == 'smoke' and sm['status'] == 'completed' and sm['sources'] == checked['sources'] and
                sm['inputs'] == checked['inputs'] and read(smoke_dir/'audit.json')['status'] == 'passed', 'Smoke niezgodny lub niekompletny')
    run = new_run(mode)
    (run/'logs').mkdir()
    for p in checked['sources']:
        dest = run/'source'/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/p, dest)
    shutil.copy2(CONFIG, run/'config.json')
    cfg = read(run/'config.json')
    write(run/'manifest.json', {'status': 'queued', 'mode': mode, 'run': str(run.relative_to(ROOT)), 'frozen_utc': utc(),
                                'cpu_check': str(check_dir), 'smoke_run': None if smoke_dir is None else str(smoke_dir),
                                'sources': checked['sources'], 'inputs': checked['inputs'], 'gpus': gpus, 'waves': [],
                                'config_sha256': digest(run/'config.json'), 'python': sys.version, 'torch': torch.__version__,
                                'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
                                'official_test_used': False, 'reserve_used': False,
                                'limit_seconds': cfg['smoke_wall_seconds' if mode == 'smoke' else 'wall_seconds']})
    print(f'RUN={run.relative_to(ROOT)}', flush=True)
    if detach:
        with (run/'launch.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-B', str(HERE/'train.py'), 'supervise', '--run-dir', str(run)],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'PID={proc.pid}', flush=True)
    else:
        supervise(run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'smoke', 'train', 'supervise', 'worker', 'audit'])
    parser.add_argument('--check', type=Path)
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--dataset', choices=DATA)
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == 'check':
        check()
    elif args.action in ('smoke', 'train'):
        launch(args.action, args.check.resolve(), args.smoke_run.resolve() if args.smoke_run else None, args.detach)
    elif args.action == 'supervise':
        supervise(args.run_dir.resolve())
    elif args.action == 'worker':
        worker(args.run_dir.resolve(), args.dataset, args.gpu)
    else:
        audit(args.run_dir.resolve())
