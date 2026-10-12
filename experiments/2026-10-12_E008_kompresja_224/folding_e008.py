"""E008: operator fizycznego foldingu reprezentantow par kanalow SE.

Eksperyment: 2026-10-12_E008_kompresja_224. Fork lokalny E003 folding.py;
semantyka operatora bez zmian.
Cel: z modelu EfficientNet-B0 (4263748 parametrow, 200 klas) zbudowac nowa,
gleboka kopie o mniejszych szerokosciach wewnetrznych blokow MBConv, usuwajac
fizycznie producenta d pary (c<d) i wchłaniajac jego wklad: alpha do SE fc1,
beta do projekcji. To nie jest maskowanie kanalow ani zmiana w miejscu.
Wejscia: wytrenowany lub losowy model bazowy, plan szerokosci (rows: lista
slownikow z PLAN_STRUKTURY_*.json albo mapa block->new_channels) oraz grupy par
(groups: block -> lista {c, d, alpha, beta}).
Wyjscia: nowy obiekt nn.Module (deepcopy) z compression_applied=True; funkcje
zwracaja tez liczbe parametrow. Bez GPU, bez danych i bez checkpointow.
Dokumenty: OPIS.md oraz PROTOKOL_v1.md w tym samym folderze eksperymentu.
"""
import copy
import math

import torch
from torch import nn
from torchvision.models.efficientnet import MBConv

BASE_PARAMETERS = 4263748

BASE_EXPANDED_CHANNELS = {
    'features.2.0': 96,
    'features.2.1': 144,
    'features.3.0': 144,
    'features.3.1': 240,
    'features.4.0': 240,
    'features.4.1': 480,
    'features.4.2': 480,
    'features.5.0': 480,
    'features.5.1': 672,
    'features.5.2': 672,
    'features.6.0': 672,
    'features.6.1': 1152,
    'features.6.2': 1152,
    'features.6.3': 1152,
    'features.7.0': 1152,
}


def count_parameters(model):
    """Zwraca liczbe parametrow (suma numel), bez buforow BN."""
    return int(sum(p.numel() for p in model.parameters()))


def per_removed_channel(block):
    """Koszt usuniecia jednego kanalu rozszerzonego w bloku MBConv.

    Zgodny z plan_structure.py: wejscie ekspansji, jadro depthwise, oba BN,
    kolumna fc1, wiersz i bias fc2 oraz kolumna projekcji.
    """
    expansion, depthwise, se, projection = block.block
    conv, exp_bn = expansion[0], expansion[1]
    dw, dw_bn = depthwise[0], depthwise[1]
    fc1, fc2 = se.fc1, se.fc2
    proj = projection[0]
    return int(
        conv.in_channels
        + int(exp_bn.weight is not None) + int(exp_bn.bias is not None)
        + dw.weight[0].numel()
        + int(dw_bn.weight is not None) + int(dw_bn.bias is not None)
        + fc1.out_channels + fc1.out_channels
        + int(fc2.bias is not None)
        + proj.out_channels
    )


def _check_base(model):
    """Sprawdza niezmieniony model bazowy: flaga, liczba i szerokosci."""
    if bool(getattr(model, 'compression_applied', False)):
        raise RuntimeError('Base model already has compression_applied=True; refusing to fold again.')
    total = count_parameters(model)
    if total != BASE_PARAMETERS:
        raise RuntimeError(
            f'Base model has {total} parameters, expected {BASE_PARAMETERS}.')
    found = {}
    for name, module in model.named_modules():
        if isinstance(module, MBConv) and len(module.block) == 4:
            found[name] = module
    if set(found) != set(BASE_EXPANDED_CHANNELS):
        raise RuntimeError('Base model MBConv blocks do not match the frozen E008 plan.')
    for path, expected in BASE_EXPANDED_CHANNELS.items():
        expansion, depthwise, se, _ = found[path].block
        conv, dw = expansion[0], depthwise[0]
        if conv.out_channels != expected or dw.groups != expected or dw.out_channels != expected:
            raise RuntimeError(f'Base expanded width mismatch in {path}.')
        if se.fc1.in_channels != expected or se.fc2.out_channels != expected:
            raise RuntimeError(f'Base SE width mismatch in {path}.')
        if se.fc1.out_channels != max(1, conv.in_channels // 4):
            raise RuntimeError(f'Base squeeze width mismatch in {path}.')
    return found


def _normalize_rows(rows):
    """Zamienia rows na mape block -> new_channels (dict lub lista slownikow)."""
    plan = {}
    if isinstance(rows, dict):
        for path, value in rows.items():
            if isinstance(value, dict):
                if 'new_channels' not in value:
                    raise RuntimeError(f'Row {path} lacks new_channels.')
                plan[str(path)] = int(value['new_channels'])
            else:
                plan[str(path)] = int(value)
    else:
        for row in rows:
            if 'block' not in row or 'new_channels' not in row:
                raise RuntimeError('Each row must contain block and new_channels.')
            plan[str(row['block'])] = int(row['new_channels'])
    if not plan:
        raise RuntimeError('Empty width plan.')
    return plan


def _validate_plan(plan):
    for path, new_channels in plan.items():
        if path not in BASE_EXPANDED_CHANNELS:
            raise RuntimeError(f'Unknown block path: {path}.')
        base_channels = BASE_EXPANDED_CHANNELS[path]
        if not 0 < new_channels <= base_channels:
            raise RuntimeError(f'Invalid new_channels={new_channels} for {path}.')
        if base_channels - new_channels > base_channels // 2:
            raise RuntimeError(f'Removing more than half of {path} is not allowed.')
    return plan


def _validate_groups(plan, groups):
    if not isinstance(groups, dict):
        raise RuntimeError('groups must be a mapping block -> list of pairs.')
    for path in groups:
        if path not in plan:
            raise RuntimeError(f'groups reference block {path} not present in rows.')
    for path, new_channels in plan.items():
        base_channels = BASE_EXPANDED_CHANNELS[path]
        remove_count = base_channels - new_channels
        entries = groups.get(path, [])
        if len(entries) != remove_count:
            raise RuntimeError(
                f'{path}: {len(entries)} pairs declared but remove_count={remove_count}.')
        seen = set()
        for entry in entries:
            c, d = int(entry['c']), int(entry['d'])
            alpha, beta = float(entry['alpha']), float(entry['beta'])
            if not 0 <= c < d < base_channels:
                raise RuntimeError(f'{path}: invalid pair ({c}, {d}).')
            if c in seen or d in seen:
                raise RuntimeError(f'{path}: overlapping pairs around channel {c}/{d}.')
            if not (math.isfinite(alpha) and math.isfinite(beta)):
                raise RuntimeError(f'{path}: non-finite coefficient for pair ({c}, {d}).')
            seen.add(c)
            seen.add(d)


def _set_param(module, name, tensor):
    old = module._parameters[name]
    requires_grad = bool(old.requires_grad) if old is not None else False
    module._parameters[name] = nn.Parameter(tensor, requires_grad=requires_grad)


def _slice_bn(bn, keep):
    """Przycina affine i bufory BN, zachowujac klase, eps, momentum i track_running_stats."""
    idx_cpu = torch.as_tensor(keep, dtype=torch.long)
    for name in ('weight', 'bias'):
        param = getattr(bn, name)
        if param is not None:
            idx = idx_cpu.to(param.device)
            _set_param(bn, name, param.detach().index_select(0, idx).clone())
    for name in ('running_mean', 'running_var'):
        buf = getattr(bn, name)
        if buf is not None:
            idx = idx_cpu.to(buf.device)
            setattr(bn, name, buf.detach().index_select(0, idx).clone())
    bn.num_features = len(keep)


def _transform_block(block, keep, pairs):
    """Wspolny rdzen: sklada alpha/beta (jesli pairs) i przycina kanaly `keep`.

    Dla materialize pairs=[] i keep=[:new_channels]. Dla foldingu pairs zawiera
    (c, d, alpha, beta), a keep pomija usuwane d. Modyfikuje block w miejscu.
    """
    expansion, depthwise, se, projection = block.block
    conv, exp_bn = expansion[0], expansion[1]
    dw, dw_bn = depthwise[0], depthwise[1]
    fc1, fc2 = se.fc1, se.fc2
    proj = projection[0]
    channels = conv.out_channels
    keep = [int(i) for i in keep]
    if len(keep) != len(set(keep)) or any(i < 0 or i >= channels for i in keep):
        raise RuntimeError('Invalid kept channel indices.')

    fc1_weight = fc1.weight.detach().clone()
    proj_weight = proj.weight.detach().clone()
    for c, d, alpha, beta in pairs:
        fc1_weight[:, c] = fc1_weight[:, c] + alpha * fc1_weight[:, d]
        proj_weight[:, c] = proj_weight[:, c] + beta * proj_weight[:, d]

    new_channels = len(keep)

    exp_idx = torch.as_tensor(keep, dtype=torch.long, device=conv.weight.device)
    _set_param(conv, 'weight', conv.weight.detach().index_select(0, exp_idx).clone())
    conv.out_channels = new_channels
    expansion.out_channels = new_channels
    _slice_bn(exp_bn, keep)

    dw_idx = torch.as_tensor(keep, dtype=torch.long, device=dw.weight.device)
    _set_param(dw, 'weight', dw.weight.detach().index_select(0, dw_idx).clone())
    dw.groups = new_channels
    dw.in_channels = new_channels
    dw.out_channels = new_channels
    depthwise.out_channels = new_channels
    _slice_bn(dw_bn, keep)

    fc1_idx = torch.as_tensor(keep, dtype=torch.long, device=fc1_weight.device)
    _set_param(fc1, 'weight', fc1_weight.index_select(1, fc1_idx).clone())
    fc1.in_channels = new_channels

    fc2_idx = torch.as_tensor(keep, dtype=torch.long, device=fc2.weight.device)
    _set_param(fc2, 'weight', fc2.weight.detach().index_select(0, fc2_idx).clone())
    fc2.out_channels = new_channels
    if fc2.bias is not None:
        _set_param(fc2, 'bias', fc2.bias.detach().index_select(0, fc2_idx).clone())

    proj_idx = torch.as_tensor(keep, dtype=torch.long, device=proj_weight.device)
    _set_param(proj, 'weight', proj_weight.index_select(1, proj_idx).clone())
    proj.in_channels = new_channels
    return block


def fold_model(base, rows, groups):
    """Zwraca nowa, skompresowana kopie modelu (deepcopy) z foldingiem alpha/beta.

    base: niezmieniony model 4263748 parametrow.
    rows: plan szerokosci (PLAN_STRUKTURY.json lub mapa block->new_channels).
    groups: block -> lista {c, d, alpha, beta}; liczba par = remove_count,
    pary rozlaczne, c < d, wspolczynniki skonczone. Model bazowy nie jest
    zmieniany; kazda proba ponownego foldingu jest odrzucana.
    """
    _check_base(base)
    plan = _validate_plan(_normalize_rows(rows))
    _validate_groups(plan, groups)

    expected = BASE_PARAMETERS
    for path, new_channels in plan.items():
        expected -= (BASE_EXPANDED_CHANNELS[path] - new_channels) * per_removed_channel(
            base.get_submodule(path))

    model = copy.deepcopy(base)
    for path, new_channels in plan.items():
        base_channels = BASE_EXPANDED_CHANNELS[path]
        entries = groups.get(path, [])
        removed = {int(entry['d']) for entry in entries}
        keep = [i for i in range(base_channels) if i not in removed]
        pairs = [(int(entry['c']), int(entry['d']), float(entry['alpha']), float(entry['beta']))
                 for entry in entries]
        _transform_block(model.get_submodule(path), keep, pairs)

    actual = count_parameters(model)
    if actual != expected:
        raise RuntimeError(f'Folded model has {actual} parameters, expected {expected}.')
    model.compression_applied = True
    model.folding_widths = dict(plan)
    model.folding_pairs = {path: [dict(entry) for entry in groups.get(path, [])] for path in plan}
    return model


def materialize_model(base, rows):
    """Buduje nowy graf o szerokosciach rows.new_channels, bez foldingu alpha/beta.

    Wybiera pierwsze new_channels indeksow tylko po to, by ustalic ksztalty;
    wartosci sa pozniejsze (strict load state_dict skompresowanego modelu).
    Model bazowy pozostaje niezmieniony, a wynik ma compression_applied=True.
    """
    _check_base(base)
    plan = _validate_plan(_normalize_rows(rows))
    model = copy.deepcopy(base)
    for path, new_channels in plan.items():
        _transform_block(model.get_submodule(path), list(range(new_channels)), [])
    model.compression_applied = True
    model.folding_widths = dict(plan)
    return model
