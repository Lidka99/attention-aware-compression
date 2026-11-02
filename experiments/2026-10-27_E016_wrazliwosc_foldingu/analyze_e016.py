"""E016: analiza CPU etapu S1 — wrażliwość pojedynczych grup na folding.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu; pełny opis: OPIS.md
(``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS.md``).

Wejście: ukończony katalog ``runs/E016/s1_*`` z 98 itemami. Wyjście:
``analysis.json``, ``analysis.md``, ``group_level.csv`` i decyzja C1–C5
o kontynuacji do etapu S2. Analiza jest opisowa; nie stroi poziomów ani
predyktorów na dev. Etap S2 nie jest uruchamiany przez ten skrypt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402

PREDICTORS = (
    'weight_reconstruction_rel_error',
    'calib_logit_rmse_pre_bn',
    'calib_logit_rmse_post_bn',
    'depth_norm',
)


def spearman(x, y):
    """Zwraca (rho, p) Spearmana; przy braku zmienności zwraca (nan, nan)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return float('nan'), float('nan')
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
        return rho, float('nan')


def mean(values):
    """Średnia z pominięciem NaN; pusta lista daje NaN."""
    array = np.asarray([value for value in values if value is not None],
                       dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float('nan')


def load_rows(run_dir):
    """Wczytuje wszystkie niekontrolne summary S1 do listy wierszy."""
    manifest = common.read_json(run_dir / 'manifest.json')
    common.require(manifest.get('kind') == 'stage1_single_group_sensitivity',
                   'To nie jest bieg S1 E016')
    common.require(manifest.get('smoke') is False,
                   'Analiza odrzuca bieg smoke')
    common.require(manifest.get('status') == 'completed',
                   'Bieg S1 nie jest ukończony')
    common.require(manifest.get('official_test_used') is False,
                   'Bieg użył oficjalnego testu')
    common.require(manifest.get('reserve_used') is False,
                   'Bieg użył rezerwy')
    rows = []
    controls = []
    for item in manifest.get('items', []):
        path = run_dir / 'items' / item['dataset'] / item['name'] / 'summary.json'
        common.require(path.is_file(), f'Brak summary: {path}')
        summary = common.read_json(path)
        common.require(summary.get('status') == 'completed',
                       f"Nieukończony item: {item['name']}")
        if item['control']:
            controls.append(summary)
            continue
        rows.append({
            'dataset': item['dataset'],
            'group_index': int(item['group_index']),
            'group_name': item['group_name'],
            'group_stage': summary['fold']['group_stage'],
            'depth_norm': float(summary['predictors']['depth_norm']),
            'level': float(item['level']),
            'params_before': int(summary['fold']['params_before']),
            'params_after': int(summary['fold']['params_after']),
            'group_width': summary['fold']['group_width'],
            'group_n_channels': summary['fold']['group_n_channels'],
            'damage_pre_bn_pp': float(summary['damage']['pre_bn_top1_pp']),
            'damage_post_bn_pp': float(summary['damage']['post_bn_top1_pp']),
            'bn_recovery_pp': float(summary['damage']['bn_recovery_top1_pp']),
            'nll_increase_pre_bn': float(
                summary['damage']['pre_bn_nll_delta']),
            'nll_increase_post_bn': float(
                summary['damage']['post_bn_nll_delta']),
            'baseline_top1': float(summary['baseline']['metrics']['top1']),
            'pre_top1': float(summary['compressed']['pre_bn']['top1']),
            'post_top1': float(summary['compressed']['post_bn']['top1']),
            'weight_reconstruction_rel_error': float(
                summary['predictors']['weight_reconstruction_rel_error']),
            'calib_logit_rmse_pre_bn': float(
                summary['predictors']['calib_logit_rmse_pre_bn']),
            'calib_logit_rmse_post_bn': float(
                summary['predictors']['calib_logit_rmse_post_bn']),
            'calib_prediction_agreement_pre_bn': float(
                summary['predictors']['calib_prediction_agreement_pre_bn']),
            'calib_prediction_agreement_post_bn': float(
                summary['predictors']['calib_prediction_agreement_post_bn']),
        })
    return manifest, rows, controls


def group_table(rows, dataset):
    """Buduje tabelę średnich po grupach dla jednego datasetu."""
    subset = [row for row in rows if row['dataset'] == dataset]
    groups = sorted({row['group_index'] for row in subset})
    table = []
    for group_index in groups:
        group_rows = [row for row in subset if row['group_index'] == group_index]
        levels = {}
        for level in sorted({row['level'] for row in group_rows}):
            level_rows = [row for row in group_rows if row['level'] == level]
            levels[str(level)] = {
                'damage_pre_bn_pp': mean(
                    row['damage_pre_bn_pp'] for row in level_rows),
                'damage_post_bn_pp': mean(
                    row['damage_post_bn_pp'] for row in level_rows),
                'bn_recovery_pp': mean(
                    row['bn_recovery_pp'] for row in level_rows),
                'weight_reconstruction_rel_error': mean(
                    row['weight_reconstruction_rel_error']
                    for row in level_rows),
                'calib_logit_rmse_pre_bn': mean(
                    row['calib_logit_rmse_pre_bn'] for row in level_rows),
                'calib_logit_rmse_post_bn': mean(
                    row['calib_logit_rmse_post_bn'] for row in level_rows),
            }
        table.append({
            'group_index': int(group_index),
            'group_name': group_rows[0]['group_name'],
            'group_stage': group_rows[0]['group_stage'],
            'depth_norm': group_rows[0]['depth_norm'],
            'mean_damage_pre_bn_pp': mean(
                row['damage_pre_bn_pp'] for row in group_rows),
            'mean_damage_post_bn_pp': mean(
                row['damage_post_bn_pp'] for row in group_rows),
            'mean_bn_recovery_pp': mean(
                row['bn_recovery_pp'] for row in group_rows),
            'mean_weight_reconstruction_rel_error': mean(
                row['weight_reconstruction_rel_error'] for row in group_rows),
            'mean_calib_logit_rmse_pre_bn': mean(
                row['calib_logit_rmse_pre_bn'] for row in group_rows),
            'mean_calib_logit_rmse_post_bn': mean(
                row['calib_logit_rmse_post_bn'] for row in group_rows),
            'levels': levels,
        })
    table.sort(key=lambda item: item['group_index'])
    return table


def level_table(rows, dataset):
    """Buduje tabelę średnich po poziomach dla jednego datasetu."""
    subset = [row for row in rows if row['dataset'] == dataset]
    table = []
    for level in sorted({row['level'] for row in subset}):
        level_rows = [row for row in subset if row['level'] == level]
        table.append({
            'level': float(level),
            'n': len(level_rows),
            'mean_damage_pre_bn_pp': mean(
                row['damage_pre_bn_pp'] for row in level_rows),
            'mean_damage_post_bn_pp': mean(
                row['damage_post_bn_pp'] for row in level_rows),
            'mean_bn_recovery_pp': mean(
                row['bn_recovery_pp'] for row in level_rows),
            'mean_weight_reconstruction_rel_error': mean(
                row['weight_reconstruction_rel_error'] for row in level_rows),
            'mean_calib_logit_rmse_pre_bn': mean(
                row['calib_logit_rmse_pre_bn'] for row in level_rows),
            'mean_calib_logit_rmse_post_bn': mean(
                row['calib_logit_rmse_post_bn'] for row in level_rows),
        })
    return table


def correlations(rows, dataset):
    """Liczy Spearmana predyktorów ze szkodą pre/post-BN w jednym datasecie."""
    subset = [row for row in rows if row['dataset'] == dataset]
    result = {}
    for predictor in PREDICTORS:
        entry = {}
        for target in ('damage_pre_bn_pp', 'damage_post_bn_pp'):
            rho, pvalue = spearman(
                [row[predictor] for row in subset],
                [row[target] for row in subset])
            entry[target] = {'rho': rho, 'pvalue': pvalue, 'n': len(subset)}
        result[predictor] = entry
    return result


def cross_dataset(rows):
    """Porównuje rangi grup między datasetami dla szkody i predyktorów."""
    result = {'n_groups': 12}
    for metric in ('damage_post_bn_pp', 'weight_reconstruction_rel_error',
                   'calib_logit_rmse_pre_bn', 'calib_logit_rmse_post_bn',
                   'depth_norm'):
        values = {}
        for dataset in ('cifar10', 'cifar100'):
            table = group_table(rows, dataset)
            values[dataset] = [item[f'mean_{metric}']
                               if f'mean_{metric}' in item
                               else item[metric] for item in table]
        rho, pvalue = spearman(values['cifar10'], values['cifar100'])
        result[metric] = {'rho': rho, 'pvalue': pvalue}
    return result


def dataset_criteria(rows, controls, cfg, manifest):
    """Ocena C1–C5 i tabel dla obu datasetów."""
    criteria = cfg['criteria']
    dataset_results = {}
    c1 = (len(rows) == 96 and len(controls) == 2)
    for dataset in ('cifar10', 'cifar100'):
        table = group_table(rows, dataset)
        levels = level_table(rows, dataset)
        corr = correlations(rows, dataset)
        level = float(criteria['dynamic_range_level'])
        chosen = [item for item in table
                  if str(level) in item['levels']]
        values = [item['levels'][str(level)]['damage_post_bn_pp']
                  for item in chosen]
        dynamic_range = (max(values) - min(values)) if values else float('nan')
        dynamic_groups = sum(
            1 for value in values
            if value >= float(criteria['dynamic_group_min_damage_pp']))
        mean_recovery = mean(
            row['bn_recovery_pp'] for row in rows
            if row['dataset'] == dataset)
        primary = corr['weight_reconstruction_rel_error']['damage_post_bn_pp']
        dataset_results[dataset] = {
            'n_rows': len([row for row in rows if row['dataset'] == dataset]),
            'group_table': table,
            'level_table': levels,
            'correlations': corr,
            'dynamic_range_level': level,
            'dynamic_range_pp': dynamic_range,
            'dynamic_groups_ge_05_pp': int(dynamic_groups),
            'mean_bn_recovery_pp': mean_recovery,
            'primary_predictor_post_bn': primary,
        }
    controls_ok = True
    control_rows = []
    for summary in controls:
        pre = float(summary['damage']['pre_bn_top1_pp'])
        agreement = float(
            summary['predictors']['calib_prediction_agreement_pre_bn'])
        reconstruction = float(
            summary['predictors']['weight_reconstruction_rel_error'])
        ok = (
            abs(pre) <= float(criteria['control_pre_bn_top1_abs_pp_max'])
            and agreement >= float(criteria['control_pre_bn_agreement_min'])
            and reconstruction == 0.0
        )
        controls_ok = controls_ok and ok
        control_rows.append({
            'dataset': summary['dataset'],
            'pre_bn_top1_pp': pre,
            'agreement_pre_bn': agreement,
            'weight_reconstruction_rel_error': reconstruction,
            'ok': bool(ok),
        })
    c1 = c1 and controls_ok
    c2_values = []
    for dataset in ('cifar10', 'cifar100'):
        item = dataset_results[dataset]
        c2_values.append(
            item['dynamic_range_pp'] >= float(criteria['dynamic_range_min_pp'])
            and item['dynamic_groups_ge_05_pp']
            >= int(criteria['dynamic_group_min_count']))
    c4_values = []
    for dataset in ('cifar10', 'cifar100'):
        primary = dataset_results[dataset]['primary_predictor_post_bn']
        c4_values.append(
            abs(primary['rho']) >= float(criteria['predictor_abs_spearman_min']))
    signs = [np.sign(dataset_results[dataset]['primary_predictor_post_bn']['rho'])
             for dataset in ('cifar10', 'cifar100')]
    c4 = all(c4_values) and signs[0] == signs[1] and signs[0] != 0
    started = manifest.get('started_utc')
    finished = manifest.get('finished_utc')
    wall_minutes = float('nan')
    if started and finished:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished)
        wall_minutes = (end - start).total_seconds() / 60.0
    disk_bytes = sum(
        path.stat().st_size for path in manifest['_run_dir'].rglob('*')
        if path.is_file()) if '_run_dir' in manifest else None
    c5 = (
        (not math.isfinite(wall_minutes)
         or wall_minutes <= float(criteria['stage1_max_wall_minutes']))
        and (disk_bytes is None
             or disk_bytes <= int(criteria['stage1_max_disk_bytes']))
    )
    return {
        'datasets': dataset_results,
        'C1_completeness': bool(c1),
        'C2_dynamic_range': bool(all(c2_values)),
        'C3_bn_recovery_reported': True,
        'C4_primary_predictor': bool(c4),
        'C5_budget': bool(c5),
        'controls': control_rows,
        'wall_minutes': wall_minutes,
        'disk_bytes': disk_bytes,
        'stage2_recommended': bool(c1 and all(c2_values) and c4 and c5),
    }


def write_csv(path, rows):
    """Zapisuje tabelę group×level do CSV."""
    fields = [
        'dataset', 'group_index', 'group_name', 'group_stage', 'depth_norm',
        'level', 'damage_pre_bn_pp', 'damage_post_bn_pp', 'bn_recovery_pp',
        'nll_increase_pre_bn', 'nll_increase_post_bn',
        'weight_reconstruction_rel_error', 'calib_logit_rmse_pre_bn',
        'calib_logit_rmse_post_bn', 'calib_prediction_agreement_pre_bn',
        'calib_prediction_agreement_post_bn',
    ]
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def fmt(value, digits=3):
    """Formatuje liczbę do Markdown albo zwraca 'n/a'."""
    if value is None or not math.isfinite(float(value)):
        return 'n/a'
    return f'{float(value):.{digits}f}'


def write_markdown(path, manifest, rows, controls, dataset_results, decision):
    """Zapisuje czytelny raport Markdown."""
    lines = [
        '# E016 — etap S1: wrażliwość pojedynczych grup na folding',
        '',
        f"Run: `{manifest['run_id']}`. Status: **{manifest['status']}**.",
        f"Itemy: {len(rows)} grupowych + {len(controls)} kontrolnych.",
        '',
        '## Decyzja o etapie S2',
        '',
    ]
    for key in ('C1_completeness', 'C2_dynamic_range', 'C4_primary_predictor',
                'C5_budget'):
        lines.append(f"- {key}: **{'TAK' if decision[key] else 'NIE'}**")
    lines.append(
        f"- **S2 rekomendowany: "
        f"{'TAK' if decision['stage2_recommended'] else 'NIE'}**")
    lines.append('')
    lines.append('C3 (BN recovery) jest raportowany opisowo i nie jest '
                 'warunkiem go/no-go.')
    lines.append('')
    lines.append('## Kontrole ratio 0')
    lines.append('')
    lines.append('| dataset | pre-BN top1 pp | agreement | recon | ok |')
    lines.append('|---|---:|---:|---:|---|')
    for row in decision['controls']:
        lines.append(
            f"| {row['dataset']} | {fmt(row['pre_bn_top1_pp'])} | "
            f"{fmt(row['agreement_pre_bn'], 6)} | "
            f"{fmt(row['weight_reconstruction_rel_error'], 6)} | "
            f"{'tak' if row['ok'] else 'NIE'} |")
    lines.append('')
    for dataset in ('cifar10', 'cifar100'):
        result = dataset_results[dataset]
        lines.append(f'## {dataset}')
        lines.append('')
        lines.append('### Średnia szkoda po poziomach')
        lines.append('')
        lines.append('| poziom | n | pre-BN pp | post-BN pp | BN recovery pp | recon | calib pre RMSE | calib post RMSE |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
        for item in result['level_table']:
            lines.append(
                f"| {int(round(item['level'] * 100))}% | {item['n']} | "
                f"{fmt(item['mean_damage_pre_bn_pp'])} | "
                f"{fmt(item['mean_damage_post_bn_pp'])} | "
                f"{fmt(item['mean_bn_recovery_pp'])} | "
                f"{fmt(item['mean_weight_reconstruction_rel_error'], 4)} | "
                f"{fmt(item['mean_calib_logit_rmse_pre_bn'], 4)} | "
                f"{fmt(item['mean_calib_logit_rmse_post_bn'], 4)} |")
        lines.append('')
        lines.append('### Grupy przy 75% (posortowane po szkodzie post-BN)')
        lines.append('')
        lines.append('| grupa | stage | depth | pre-BN pp | post-BN pp | BN recovery pp | recon |')
        lines.append('|---|---|---:|---:|---:|---:|---:|')
        chosen = sorted(
            result['group_table'],
            key=lambda item: item['levels']['0.75']['damage_post_bn_pp'],
            reverse=True)
        for item in chosen:
            level = item['levels']['0.75']
            lines.append(
                f"| {item['group_name']} | {item['group_stage']} | "
                f"{fmt(item['depth_norm'], 2)} | "
                f"{fmt(level['damage_pre_bn_pp'])} | "
                f"{fmt(level['damage_post_bn_pp'])} | "
                f"{fmt(level['bn_recovery_pp'])} | "
                f"{fmt(level['weight_reconstruction_rel_error'], 4)} |")
        lines.append('')
        lines.append('### Korelacje Spearmana (48 przebiegów grupowych)')
        lines.append('')
        lines.append('| predyktor | vs pre-BN rho | p | vs post-BN rho | p |')
        lines.append('|---|---:|---:|---:|---:|')
        for predictor in PREDICTORS:
            entry = result['correlations'][predictor]
            lines.append(
                f"| {predictor} | {fmt(entry['damage_pre_bn_pp']['rho'])} | "
                f"{fmt(entry['damage_pre_bn_pp']['pvalue'], 4)} | "
                f"{fmt(entry['damage_post_bn_pp']['rho'])} | "
                f"{fmt(entry['damage_post_bn_pp']['pvalue'], 4)} |")
        lines.append('')
        lines.append(
            f"C2: rozstęp przy 75% = {fmt(result['dynamic_range_pp'])} pp; "
            f"grup z mean post-BN ≥ 0,5 pp = "
            f"{result['dynamic_groups_ge_05_pp']}.")
        lines.append(
            f"C3: średnie BN recovery = "
            f"{fmt(result['mean_bn_recovery_pp'])} pp.")
        lines.append('')
    lines.append('## Rangi grup między datasetami')
    lines.append('')
    lines.append('| metryka | Spearman rho | p |')
    lines.append('|---|---:|---:|')
    for metric, entry in decision['cross_dataset'].items():
        if isinstance(entry, dict):
            lines.append(
                f"| {metric} | {fmt(entry['rho'])} | "
                f"{fmt(entry['pvalue'], 4)} |")
    lines.append('')
    lines.append('## Budżet')
    lines.append('')
    lines.append(
        f"Wall-time: {fmt(decision['wall_minutes'], 1)} min; "
        f"rozmiar katalogu: {decision['disk_bytes']} B.")
    lines.append('')
    lines.append('## Ograniczenia')
    lines.append('')
    lines.append('- Jeden seed42 na dataset; to etap rozwojowy, nie replikacja.')
    lines.append('- Ocena na dev; oficjalny test i rezerwa nietknięte.')
    lines.append('- Etap S2 nie został uruchomiony tym skryptem.')
    lines.append('')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir
        else run_dir / f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = common.load_config(run_dir / 'config.json')
    manifest, rows, controls = load_rows(run_dir)
    manifest = dict(manifest)
    manifest['_run_dir'] = run_dir
    decision = dataset_criteria(rows, controls, cfg, manifest)
    cross = cross_dataset(rows)
    decision['cross_dataset'] = cross
    decision['run_id'] = manifest['run_id']
    dataset_results = decision['datasets']
    result = {
        'experiment': common.EXPERIMENT,
        'opis': common.OPIS_PATH,
        'run_id': manifest['run_id'],
        'analysis_utc': datetime.now(timezone.utc).isoformat(),
        'n_group_rows': len(rows),
        'n_controls': len(controls),
        'decision': {key: value for key, value in decision.items()
                     if key != 'datasets'},
        'datasets': dataset_results,
        'cross_dataset': cross,
    }
    common.write_json_atomic(output_dir / 'analysis.json', result)
    write_csv(output_dir / 'group_level.csv', rows)
    write_markdown(output_dir / 'analysis.md', manifest, rows, controls,
                   dataset_results, decision)
    print(f'E016_ANALYSIS={output_dir}', flush=True)
    print(
        f"E016_S2_RECOMMENDED={'YES' if decision['stage2_recommended'] else 'NO'}",
        flush=True)


if __name__ == '__main__':
    main()
