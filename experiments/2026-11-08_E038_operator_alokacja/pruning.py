"""2026-11-08_E038_operator_alokacja: MAG2 dla zadanych szerokości; OPIS.md.
Wejście: CNN CPU i profil E035; wyjście: przycięty model i indeksy grup.
Ranking oryginalnych filtrów; bez treningu, bez dostępu do danych.
"""
import copy

import torch
from torch import nn


def keep_indices(weight, k, p=2):
    scores = weight.detach().flatten(1).norm(p=p, dim=1)
    assert 1 <= k <= len(scores) and torch.isfinite(scores).all()
    return torch.argsort(scores, descending=True, stable=True)[:k]


def prune(base, pair, widths, rt):
    preact = pair.startswith('preact')
    geo = rt.e37.geometry(pair, base)
    assert len(widths) == len(geo.names) and all(1 <= k <= n for k, n in zip(widths, geo.sizes))
    if list(widths) == list(geo.sizes):
        return copy.deepcopy(base).eval().requires_grad_(False), []
    axis_map = (rt.e35.e34.get_axis_to_perm_PreActResNet18(override=False) if preact
                else rt.e35.e29.get_axis_to_perm_ResNet18(override=False))
    groups = rt.e35.e29.axes2perm_to_perm2axes(axis_map)
    assert list(groups) == (geo.names if preact else list(groups))
    assert len(groups) == len(widths)
    assignment = dict(zip(groups, widths))
    selected, diag = {}, []
    for g, axes in groups.items():
        if preact and g == 'relu1_out':
            assert assignment[g] == assignment['relu_conv'], 'Brak projekcji residual w layer1'
            selected[g] = selected['relu_conv'].clone()
            anchor = 'conv1.weight (identity-tied)'
        else:
            anchors = [name for name, axis in axes if axis == 0 and name.endswith('.weight')
                       and isinstance(base.get_submodule(name.rsplit('.', 1)[0]), nn.Conv2d)]
            assert anchors, g
            anchor = anchors[0]
            selected[g] = keep_indices(base.state_dict()[anchor], assignment[g])
        ids = selected[g]
        assert len(ids) == assignment[g] and len(torch.unique(ids)) == len(ids)
        diag.append(dict(group=g, k=len(ids), n=int(base.state_dict()[axes[0][0]].shape[axes[0][1]]),
                         anchor=anchor, keep=ids.tolist()))

    result = copy.deepcopy(base)
    # Wycina każdą oś według oryginalnych indeksów, bez przeliczania score po cięciu.
    for name, tensor in base.state_dict().items():
        value = tensor.detach().clone()
        for axis, g in enumerate(axis_map.get(name, ())):
            if g is not None:
                value = value.index_select(axis, selected[g])
        path, field = name.rsplit('.', 1)
        module = result.get_submodule(path)
        if field in module._parameters:
            module._parameters[field] = nn.Parameter(value.contiguous(), requires_grad=False)
        else:
            module._buffers[field] = value.contiguous()
    for module in result.modules():
        if isinstance(module, nn.Conv2d):
            assert module.groups == 1
            module.out_channels, module.in_channels = module.weight.shape[:2]
        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = module.weight.numel()
            module.num_batches_tracked.zero_()  # tak jak przebudowa BN w upstream MAG
        elif isinstance(module, nn.Linear):
            module.out_features, module.in_features = module.weight.shape
    result.eval().requires_grad_(False)
    assert sum(p.numel() for p in result.parameters()) == geo.count(widths)
    bias = 'linear' if preact else 'fc'
    assert torch.equal(getattr(result, bias).bias, getattr(base, bias).bias)
    return result, diag
