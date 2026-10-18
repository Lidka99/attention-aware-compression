"""E010: generowanie grup kanalow dla nowych kryteriow benchmarku.

Eksperyment: 2026-10-18_E010_benchmark_kryteriow; protokol: OPIS.md i PROTOKOL_v1.md.
Wejscia: model bazowy, plan struktury, momenty Grama (b{i}_h/z/g), statystyki
APoZ/Taylor (b{i}_apoz, b{i}_taylor) oraz config frozen.
Wyjscia: JSON z grupami par per (dataset, ratio, kryterium) w katalogu biegu;
pary rozlaczne, c = reprezentant (zostaje), d = usuwany, alpha/beta z LS.
Kryteria: L1, BN_GAMMA, TAYLOR, APOZ (per kanal), REC (reszta LS),
KMEANS (grupowanie wag po BN, medoid; Gram macierzy filtrow dla taniosci).
Bez GPU i bez treningu.
"""
import argparse
from pathlib import Path

import numpy as np

from support_e010 import (dataset_cfg, plan_rows, plan_spec, block_map, load_moments,
                          load_stats, base_model, write_json, require, digest,
                          load_config, DATASET_KEYS)


def eligible_mask(moment, threshold):
    diag = [np.sqrt(np.maximum(np.diag(moment[key]), 0.0)) for key in ('h', 'z', 'g')]
    return (diag[0] > threshold) & (diag[1] > threshold) & (diag[2] > threshold)


def bn_folded_vectors(model, block):
    expansion = model.get_submodule(block).block[0]
    conv, bn = expansion[0], expansion[1]
    weight = conv.weight.detach().double().reshape(conv.out_channels, -1).cpu().numpy()
    gamma = bn.weight.detach().double().cpu().numpy()
    var = bn.running_var.detach().double().cpu().numpy()
    scale = gamma / np.sqrt(var + float(bn.eps))
    return weight * scale[:, None]


def alpha_beta(moment, c, d):
    den_g = float(moment['g'][c, c])
    den_z = float(moment['z'][c, c])
    require(den_g > 1e-12 and den_z > 1e-12,
            f'Zero Gram denominator for representative {c}')
    alpha = float(moment['g'][c, d] / den_g)
    beta = float(moment['z'][c, d] / den_z)
    require(np.isfinite(alpha) and np.isfinite(beta), 'Non-finite alpha/beta')
    return alpha, beta


def _pair_entry(moment, c, d, score):
    alpha, beta = alpha_beta(moment, c, d)
    return {'c': int(c), 'd': int(d), 'alpha': alpha, 'beta': beta, 'score': float(score)}


def _cosine_from_gram(gram, diag, a, b):
    den = np.sqrt(np.maximum(diag[a] * diag[b], 1e-24))
    return float(np.clip(gram[a, b] / den, -1.0, 1.0))


def pairs_rec(moment, eligible, n_pairs):
    """Zachlanne rozlaczne pary minimalizujace wzgledna reszte rekonstrukcji."""
    require(2 * n_pairs <= len(eligible), 'Too few eligible channels for REC pairs')
    idx = np.asarray(eligible)
    require(bool(np.all(np.diag(moment['g'])[idx] > 0)
                 and np.all(np.diag(moment['z'])[idx] > 0)), 'Non-positive Gram diagonal')
    scored = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = int(idx[a]), int(idx[b])
            rg = float(moment['g'][j, j] - moment['g'][i, j] ** 2 / moment['g'][i, i])
            rz = float(moment['z'][j, j] - moment['z'][i, j] ** 2 / moment['z'][i, i])
            score = -0.5 * (rg / float(moment['g'][j, j]) + rz / float(moment['z'][j, j]))
            scored.append((score, i, j))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    used, pairs = set(), []
    for score, i, j in scored:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        pairs.append(_pair_entry(moment, i, j, score))
        if len(pairs) == n_pairs:
            break
    require(len(pairs) == n_pairs, 'REC could not find disjoint pairs')
    return pairs


def pairs_from_importance(moment, eligible, remove_count, importance, gram, diag):
    """Usuwa najslabsze kanaly; przypisanie do unikalnych reprezentantow po cosine."""
    require(len(eligible) >= remove_count + 1, 'Too few eligible channels for removal')
    order = sorted(eligible, key=lambda i: (float(importance[i]), int(i)))
    removed = np.asarray(order[:remove_count], dtype=np.int64)
    removed_set = set(removed.tolist())
    kept = np.asarray([i for i in eligible if i not in removed_set], dtype=np.int64)
    sims = gram[np.ix_(removed, kept)] / np.sqrt(
        np.maximum(np.outer(diag[removed], diag[kept]), 1e-24))
    candidates = []
    for r in range(len(removed)):
        for k in range(len(kept)):
            candidates.append((float(np.clip(sims[r, k], -1.0, 1.0)),
                               int(removed[r]), int(kept[k])))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_d, used_c, pairs = set(), set(), []
    for score, d, c in candidates:
        if d in used_d or c in used_c:
            continue
        used_d.add(d); used_c.add(c)
        pairs.append(_pair_entry(moment, c, d, score))
        if len(pairs) == remove_count:
            break
    require(len(pairs) == remove_count, 'Importance assignment could not find a matching')
    return pairs


def _fix_empty(labels, distances, k):
    """Przydziela najdalszy punkt do pustych klastrow (deterministycznie)."""
    n = len(labels)
    counts = np.bincount(labels, minlength=k)
    for cluster in range(k):
        if counts[cluster] == 0:
            far = int(np.argmax(distances[np.arange(n), labels]))
            labels[far] = cluster
            counts = np.bincount(labels, minlength=k)
    return labels


def _kmeans_gram(gram, k, seed, max_iter=100):
    """k-means++/Lloyd na wektorach opisanych macierza Grama (bez D-wymiarow)."""
    n = gram.shape[0]
    diag = np.diag(gram).copy()
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(n))
    centers = [first]
    distances = diag - 2.0 * gram[:, first] + diag[first]
    for _ in range(1, k):
        total = float(distances.sum())
        probs = distances / total if total > 0 else np.full(n, 1.0 / n)
        index = int(rng.choice(n, p=probs))
        centers.append(index)
        distances = np.minimum(distances, diag - 2.0 * gram[:, index] + diag[index])
    initial = diag[:, None] - 2.0 * gram[:, centers] + diag[centers][None, :]
    labels = _fix_empty(initial.argmin(axis=1), initial, k)
    for _ in range(max_iter):
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        sums = np.zeros((n, k), dtype=np.float64)
        np.add.at(sums.T, labels, gram.T)
        mean_dot = sums / np.maximum(counts[None, :], 1.0)
        mask = np.zeros((n, k), dtype=bool)
        mask[np.arange(n), labels] = True
        within = (sums * mask).sum(axis=0)
        norm2 = within / np.maximum(counts ** 2, 1.0)
        distances = diag[:, None] - 2.0 * mean_dot + norm2[None, :]
        new = _fix_empty(distances.argmin(axis=1), distances, k)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def pairs_kmeans(moment, eligible, gram, new_channels, seed):
    diag = np.diag(gram)
    ineligible = int(gram.shape[0]) - len(eligible)
    k = int(new_channels) - ineligible
    require(k >= 1, 'No eligible cluster representatives left')
    require(k <= len(eligible), 'Too many clusters for eligible channels')
    sub = gram[np.ix_(np.asarray(eligible), np.asarray(eligible))]
    labels = _kmeans_gram(sub, k, seed)
    pairs = []
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        require(len(members) > 0, 'Empty k-means cluster')
        member_idx = np.asarray(eligible)[members]
        if len(members) == 1:
            continue
        block_gram = gram[np.ix_(member_idx, member_idx)]
        block_diag = diag[member_idx]
        costs = (block_diag[:, None] + block_diag[None, :] - 2.0 * block_gram).sum(axis=1)
        medoid = int(member_idx[int(np.argmin(costs))])
        for channel in member_idx:
            if int(channel) == medoid:
                continue
            score = _cosine_from_gram(gram, diag, medoid, int(channel))
            pairs.append(_pair_entry(moment, medoid, int(channel), score))
    require(len(pairs) + k + ineligible == int(gram.shape[0]),
            'K-means budget mismatch')
    return pairs


def groups_for_block(criterion, moment, gram, row, threshold, stats):
    base_channels = int(row['expanded_channels'])
    require(gram.shape == (base_channels, base_channels), 'Gram width mismatch')
    diag = np.diag(gram)
    eligible = np.flatnonzero(eligible_mask(moment, threshold))
    remove_count = int(row['remove_count'])
    new_channels = base_channels - remove_count
    if criterion == 'REC':
        return pairs_rec(moment, eligible, remove_count)
    if criterion == 'KMEANS':
        return pairs_kmeans(moment, eligible, gram, new_channels,
                            seed=int(stats['settings']['kmeans_seed']) + 1000 * int(stats['index']))
    if criterion == 'L1':
        importance = stats['l1']
    elif criterion == 'BN_GAMMA':
        importance = stats['bn_gamma']
    elif criterion == 'TAYLOR':
        importance = stats['taylor']
    elif criterion == 'APOZ':
        importance = 1.0 - stats['apoz']
    else:
        raise RuntimeError(f'Unknown criterion: {criterion}')
    require(len(importance) == base_channels, 'Importance width mismatch')
    return pairs_from_importance(moment, eligible, remove_count, importance, gram, diag)


def generate_dataset(cfg, key, run_dir, max_blocks=None):
    ds = dataset_cfg(cfg, key)
    moments = load_moments(ds)
    stats = load_stats(cfg, key, run_dir)
    base = base_model(ds)
    outputs = {}
    for ratio, spec in ds['criteria']['ratios'].items():
        rows = plan_rows(ds, ratio)
        plan = plan_spec(ds, ratio)
        for criterion in spec['criteria']:
            if criterion in ds['criteria']['reused']:
                continue
            groups = {}
            for index, row in enumerate(rows):
                path = row['block']
                moment = {name: moments[f'b{index}_{name}'] for name in ('h', 'z', 'g')}
                vectors = bn_folded_vectors(base, path)
                gram = vectors @ vectors.T
                expansion = base.get_submodule(path).block[0]
                raw = expansion[0].weight.detach().double().reshape(
                    expansion[0].out_channels, -1).cpu().numpy()
                st = {'settings': stats['settings'],
                      'index': index,
                      'l1': np.abs(raw).sum(axis=1),
                      'bn_gamma': np.abs(expansion[1].weight.detach()
                                         .double().cpu().numpy()),
                      'taylor': np.asarray(stats[f'b{index}_taylor']),
                      'apoz': np.asarray(stats[f'b{index}_apoz'])}
                groups[path] = groups_for_block(criterion, moment, gram, row,
                                                float(cfg['norm_zero_threshold']), st)
                if max_blocks is not None and index + 1 >= max_blocks:
                    break
            payload = {
                'experiment': cfg['experiment'], 'dataset': key, 'ratio': ratio,
                'criterion': criterion, 'plan_file': plan['file'],
                'plan_sha256': plan['sha256'],
                'moments_sha256': ds['moments']['sha256'],
                'stats_sha256': digest(Path(run_dir) / 'stats' / f'{key}.npz'),
                'settings': {
                    'assignment': ('greedy unique cosine of BN-folded expansion filters'
                                   if criterion not in ('REC', 'KMEANS') else criterion),
                    'coefficients': 'alpha=S_g[c,d]/S_g[c,c]; beta=S_z[c,d]/S_z[c,c]',
                    'eligibility': 'sqrt(diag(S_h)), sqrt(diag(S_z)), sqrt(diag(S_g)) > threshold',
                    'norm_zero_threshold': float(cfg['norm_zero_threshold']),
                    'pair_order': 'free (c kept, d removed)'},
                'groups': groups}
            path_out = Path(run_dir) / 'groups' / f'{key}_{ratio}_{criterion}.json'
            path_out.parent.mkdir(parents=True, exist_ok=True)
            write_json(path_out, payload)
            outputs[(ratio, criterion)] = str(path_out)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASET_KEYS, action='append')
    args = parser.parse_args()
    cfg = load_config(args.config)
    keys = args.dataset or list(DATASET_KEYS)
    all_outputs = {}
    for key in keys:
        all_outputs[key] = generate_dataset(cfg, key, args.run_dir)
    print('E010_CRITERIA=' + str({k: len(v) for k, v in all_outputs.items()}), flush=True)


if __name__ == '__main__':
    main()
