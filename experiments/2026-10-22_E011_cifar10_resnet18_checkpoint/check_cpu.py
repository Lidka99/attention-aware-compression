"""E011: testy CPU podziału danych, architektury i forwardu bez pobierania danych.

Eksperyment: 2026-10-22_E011_cifar10_resnet18_checkpoint; pełny opis: OPIS.md.
Wejścia: wyłącznie kod train.py i syntetyczne etykiety/obrazy w pamięci.
Wyjścia: wynik kontroli na stdout (bez plików, bez GPU, bez CIFAR-10 i bez sieci).
"""
import json

import numpy as np
import torch
from PIL import Image
from torch import nn

from train import (DEV_PER_CLASS, EXPECTED_PARAMETERS, SMOKE_DEV_PER_CLASS,
                   SMOKE_TRAIN_PER_CLASS, TRAIN_PER_CLASS, build_model,
                   count_parameters, dev_transform, make_split,
                   stratified_subset, train_transform)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def class_counts(indices, labels):
    return np.bincount(np.asarray(labels)[np.asarray(indices)], minlength=10)


def main():
    labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
    train_indices, dev_indices = make_split(labels, 20261022)
    train_again, dev_again = make_split(labels, 20261022)
    train_other, dev_other = make_split(labels, 20261023)
    require(len(train_indices) == TRAIN_PER_CLASS * 10 == 45000, 'Zła liczba train')
    require(len(dev_indices) == DEV_PER_CLASS * 10 == 5000, 'Zła liczba dev')
    require(np.array_equal(train_indices, train_again)
            and np.array_equal(dev_indices, dev_again), 'Podział nie jest deterministyczny')
    require(not np.array_equal(dev_indices, dev_other)
            or not np.array_equal(train_indices, train_other), 'Seed nie zmienia podziału')
    require(not np.intersect1d(train_indices, dev_indices).size,
            'Train i dev nie są rozłączne')
    require(np.array_equal(np.sort(np.concatenate([train_indices, dev_indices])),
                           np.arange(50000)), 'Podział nie pokrywa 50 000 obrazów')
    require(np.all(class_counts(train_indices, labels) == TRAIN_PER_CLASS),
            'Train nie jest stratyfikowany')
    require(np.all(class_counts(dev_indices, labels) == DEV_PER_CLASS),
            'Dev nie jest stratyfikowany')

    smoke_train = stratified_subset(train_indices, labels, SMOKE_TRAIN_PER_CLASS, 20261022)
    smoke_dev = stratified_subset(dev_indices, labels, SMOKE_DEV_PER_CLASS, 20261022)
    require(len(smoke_train) == 1000 and len(smoke_dev) == 200, 'Zły podzbiór smoke')
    require(not np.intersect1d(smoke_train, smoke_dev).size, 'Smoke train/dev nachodzą')
    require(np.all(class_counts(smoke_train, labels) == SMOKE_TRAIN_PER_CLASS),
            'Smoke train nie jest stratyfikowany')
    require(np.all(class_counts(smoke_dev, labels) == SMOKE_DEV_PER_CLASS),
            'Smoke dev nie jest stratyfikowany')
    require(np.array_equal(smoke_train, stratified_subset(
        train_indices, labels, SMOKE_TRAIN_PER_CLASS, 20261022)),
        'Podzbiór smoke nie jest deterministyczny')

    model = build_model()
    parameters = count_parameters(model)
    require(EXPECTED_PARAMETERS == 11173962, 'Stała liczby parametrów jest zła')
    require(parameters == EXPECTED_PARAMETERS,
            f'Nieoczekiwana liczba parametrów: {parameters}')
    require(isinstance(model.conv1, nn.Conv2d)
            and model.conv1.in_channels == 3 and model.conv1.out_channels == 64
            and model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
            and model.conv1.padding == (1, 1) and model.conv1.bias is None,
            'Zła warstwa conv1')
    require(isinstance(model.maxpool, nn.Identity), 'maxpool nie jest Identity')
    require(isinstance(model.fc, nn.Linear) and model.fc.in_features == 512
            and model.fc.out_features == 10, 'Zła warstwa fc')

    torch.manual_seed(0)
    synthetic = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.inference_mode():
        output = model(synthetic)
    require(tuple(output.shape) == (2, 10), f'Zły kształt wyjścia: {tuple(output.shape)}')
    require(output.dtype == torch.float32 and bool(torch.isfinite(output).all()),
            'Forward zwrócił zły typ lub niefinitywne wartości')

    image = Image.fromarray(np.random.default_rng(0).integers(
        0, 256, size=(32, 32, 3), dtype=np.uint8))
    train_sample = train_transform()(image)
    dev_sample = dev_transform()(image)
    require(tuple(train_sample.shape) == (3, 32, 32), 'Zły kształt transformacji train')
    require(tuple(dev_sample.shape) == (3, 32, 32), 'Zły kształt transformacji dev')

    print('E011_CPU_CHECKS=passed')
    print(json.dumps({
        'parameters': parameters,
        'split_seed': 20261022,
        'train_count': int(len(train_indices)),
        'dev_count': int(len(dev_indices)),
        'train_per_class': int(TRAIN_PER_CLASS),
        'dev_per_class': int(DEV_PER_CLASS),
        'smoke_train_count': int(len(smoke_train)),
        'smoke_dev_count': int(len(smoke_dev)),
        'official_test_used': False,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
