"""E016 S2: analiza CPU zamrożonego porównania alokacji 50%.

Eksperyment: 2026-10-27_E016_wrazliwosc_foldingu, etap S2; pełny protokół:
``experiments/2026-10-27_E016_wrazliwosc_foldingu/OPIS_S2.md``.

Wejście: katalog biegu S2 z manifestem, zamrożonymi alokacjami i ośmioma
summary. Wyjście: ``analysis.json``, ``analysis.md``, tabele CSV i decyzja
G1–G3. Skrypt sprawdza integralność (hashe, kolejność zamrożenia alokacji,
brak dev w alokacji, brak testu/rezerwy, target UNIFORM, tolerancję, BN)
i porównuje UNIFORM / TILT+0.30 / TILT+0.15 / AUTO50 na dev.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e016_common as common  # noqa: E402
import s2_common as s2  # noqa: E402

ROOT = common.ROOT
RUNS = ROOT / 'runs' / 'E016'
VARIANTS = ('uniform', 'tilt_p030', 'tilt_p015', 'auto50')
DATASETS = ('cifar10', 'cifar100')


def require(condition, message):
    """Podnosi RuntimeError, jeśli warunek nie jest spełniony."""
    if not condition:
        raise RuntimeError(message)


def utc_now():
    """Zwraca bieżący znacznik czasu UTC."""
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value):
    """Parsuje ISO 8601 UTC do porównania kolejności."""
    require(value, 'Brak znacznika UTC')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def output_dir():
    """Tworzy unikalny katalog analizy."""
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for _ in range(100):
        path = RUNS / f'analizy/{stamp}_s2_{secrets.token_hex(4)}'
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Nie udało się utworzyć katalogu analizy')


def verify_artifacts(run):
    """Weryfikuje wszystkie wpisy artifacts.sha256."""
    path = run / 'artifacts.sha256'
    require(path.is_file(), f'Brak artifacts.sha256 w {run}')
    checked = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, relative = line.split('  ', 1)
        target = run / relative
        require(target.is_file(), f'Brak artefaktu: {target}')
        require(common.digest(target) == digest,
                f'Zły hash artefaktu: {target}')
        checked += 1
    require(checked > 0, 'Pusty artifacts.sha256')
    return checked


def verify_sources(run, manifest):
    """Sprawdza snapshoty źródeł względem manifestu."""
    source = run / 'source'
    require(source.is_dir(), f'Brak snapshotu źródeł: {source}')
    checked = 0
    for relative, digest in manifest['source_sha256'].items():
        target = source / relative
        require(target.is_file(), f'Brak snapshotu: {target}')
        require(common.digest(target) == digest,
                f'Zły hash snapshotu: {target}')
        checked += 1
    return checked


def load_summaries(run, cfg):
    """Wczytuje summary i specy dla ośmiu wariantów."""
    summaries = {}
    specs = {}
    for dataset in DATASETS:
        summaries[dataset] = {}
        specs[dataset] = {}
        for variant in VARIANTS:
            summary_path = run / 'allocations' / dataset / variant / 'summary.json'
            require(summary_path.is_file(), f'Brak summary: {summary_path}')
            summary = common.read_json(summary_path)
            spec_path = run / 'specs' / dataset / f'{variant}.json'
            require(spec_path.is_file(), f'Brak spec: {spec_path}')
            spec = common.read_json(spec_path)
            summaries[dataset][variant] = summary
            specs[dataset][variant] = spec
    return summaries, specs


def verify_integrity(run, cfg, manifest, frozen, audit, summaries, specs):
    """Sprawdza integralność i brak użycia dev/testu/rezerwy."""
    require(manifest.get('status') == 'completed',
            f"Manifest nie jest completed: {manifest.get('status')}")
    require(manifest.get('official_test_used') is False,
            'Manifest używa testu')
    require(manifest.get('reserve_used') is False,
            'Manifest używa rezerwy')
    require(manifest.get('dev_used_for_allocation') is False,
            'Manifest użył dev do alokacji')
    require(frozen.get('dev_used_for_allocation') is False,
            'Zamrożone alokacje używają dev')
    require(audit.get('dev_used_for_allocation') is False,
            'Audyt alokacji używa dev')
    require(frozen.get('uses_test') is False
            and frozen.get('uses_reserve') is False,
            'Zamrożone alokacje używają testu/rezerwy')
    require(parse_utc(manifest['allocations_frozen_utc'])
            < parse_utc(manifest['workers_launched_utc']),
            'Alokacje nie zostały zamrożone przed startem workerów')
    require(frozen['created_utc'] == manifest['allocations_frozen_utc'],
            'Timestamp zamrożenia nie zgadza się z manifestem')

    require(len(manifest.get('workers', [])) == 8,
            'Oczekiwano 8 workerów')
    for worker in manifest['workers']:
        require(worker.get('status') == 'completed',
                f"Worker {worker.get('dataset')}/{worker.get('variant')} "
                f"nie jest completed")
        require(worker.get('returncode') == 0,
                f"Worker {worker.get('dataset')}/{worker.get('variant')} "
                f"ma returncode != 0")
        require(int(worker['gpu']) in (0, 1, 2, 3),
                'Zły numer GPU w manifeście')
        require((run / worker['log']).is_file(),
                f"Brak logu: {worker['log']}")

    spec_hashes = {}
    forbidden = ('top1', 'nll', 'damage', 'accuracy', 'dev_accuracy')
    for dataset in DATASETS:
        uniform = None
        bn_hashes = set()
        for variant in VARIANTS:
            summary = summaries[dataset][variant]
            spec = specs[dataset][variant]
            require(summary.get('status') == 'completed',
                    f'{dataset}/{variant}: summary nie completed')
            require(summary.get('official_test_used') is False,
                    f'{dataset}/{variant}: użyto testu')
            require(summary.get('reserve_used') is False,
                    f'{dataset}/{variant}: użyto rezerwy')
            require(summary.get('dev_used_for_allocation') is False,
                    f'{dataset}/{variant}: dev w alokacji')
            require(summary.get('allocation_dev_used') is False,
                    f'{dataset}/{variant}: alokacja użyła dev')
            require(summary.get('allocation_frozen_before_evaluation') is True,
                    f'{dataset}/{variant}: alokacja niezamrożona')
            require(spec.get('dev_used_for_allocation') is False,
                    f'{dataset}/{variant}: spec używa dev')
            require(spec.get('uses_test') is False
                    and spec.get('uses_reserve') is False,
                    f'{dataset}/{variant}: spec używa testu/rezerwy')
            spec_hash = common.digest(
                run / manifest['specs'][dataset][variant]['path'])
            require(spec_hash
                    == manifest['specs'][dataset][variant]['sha256'],
                    f'{dataset}/{variant}: zły hash spec')
            require(summary.get('spec_sha256') == spec_hash,
                    f'{dataset}/{variant}: summary/spec hash')
            spec_hashes[f'{dataset}/{variant}'] = spec_hash
            text = json.dumps(spec, sort_keys=True).lower()
            for token in forbidden:
                require(token not in text,
                        f'{dataset}/{variant}: spec zawiera {token}')
            allocation = spec['allocation']
            require(allocation.get('dev_used_for_allocation') is False,
                    f'{dataset}/{variant}: alokacja używa dev')
            require(int(allocation['params'])
                    <= int(allocation['target_params']),
                    f'{dataset}/{variant}: przekroczony target')
            require(-0.5 <= float(allocation['delta_percent']) <= 0.0,
                    f'{dataset}/{variant}: poza tolerancją')
            target_ratios = allocation.get(
                'target_ratios', allocation['ratios'])
            require(len(allocation['ratios']) == 12
                    and len(target_ratios) == 12,
                    f'{dataset}/{variant}: złe ratios')
            require(all(float(0.10 <= value <= 0.90)
                        for value in target_ratios),
                    f'{dataset}/{variant}: target ratio poza [0.1,0.9]')
            require(all(float(0.10 - 0.02 <= value <= 0.90 + 0.02)
                        for value in allocation['ratios']),
                    f'{dataset}/{variant}: realized ratio poza tolerancją')
            require(int(summary['compressed']['params'])
                    == int(allocation['params']),
                    f'{dataset}/{variant}: params != spec')
            require(int(summary['target_params'])
                    == int(allocation['target_params']),
                    f'{dataset}/{variant}: target != spec')
            require(int(summary['compressed']['params'])
                    <= int(summary['target_params']),
                    f'{dataset}/{variant}: przekroczony target w summary')
            require(-0.5 <= float(summary['delta_percent']) <= 0.0,
                    f'{dataset}/{variant}: tolerancja w summary')
            require(summary['compressed']['post_bn']['count'] == 5000,
                    f'{dataset}/{variant}: zły dev count')
            require(summary['compressed']['pre_bn']['count'] == 5000,
                    f'{dataset}/{variant}: zły pre-BN dev count')
            dataset_cfg = common.dataset_by_name(cfg, dataset)
            require(abs(summary['baseline']['metrics']['top1']
                        - float(dataset_cfg['baseline_top1'])) <= 1e-12,
                    f'{dataset}/{variant}: baseline top1')
            require(abs(summary['baseline']['metrics']['nll']
                        - float(dataset_cfg['baseline_nll'])) <= 1e-5,
                    f'{dataset}/{variant}: baseline NLL')
            bn = summary['compressed']['bn']
            require(bn['params_unchanged'] is True,
                    f'{dataset}/{variant}: BN zmienił parametry')
            require(bn['params_have_grad'] is False,
                    f'{dataset}/{variant}: gradienty w BN')
            require(int(bn['samples']) == int(dataset_cfg['bn_samples']),
                    f'{dataset}/{variant}: zła liczba próbek BN')
            bn_hashes.add(str(bn['first_batch_sha256']))
            cost = summary['calibration_cost']
            require(cost['gradients'] is False,
                    f'{dataset}/{variant}: gradienty w kalibracji')
            require(cost['labels_used_in_optimization'] is False,
                    f'{dataset}/{variant}: etykiety w optymalizacji BN')
            require(cost['same_for_all_variants'] is True,
                    f'{dataset}/{variant}: inny BN')
            require(cost['full_train'] is True,
                    f'{dataset}/{variant}: BN nie na pełnym train')
            logits_path = run / 'allocations' / dataset / variant / 'logits.npz'
            diagnostics_path = (run / 'allocations' / dataset / variant
                                / 'diagnostics.json')
            require(logits_path.is_file(), f'Brak logits: {logits_path}')
            require(diagnostics_path.is_file(),
                    f'Brak diagnostics: {diagnostics_path}')
            require(common.digest(logits_path)
                    == summary['logits_sha256'],
                    f'{dataset}/{variant}: hash logits')
            require(common.digest(diagnostics_path)
                    == summary['diagnostics_sha256'],
                    f'{dataset}/{variant}: hash diagnostics')
            if uniform is None:
                uniform = int(summary['compressed']['params'])
            require(int(summary['target_params']) == uniform,
                    f'{dataset}/{variant}: target != UNIFORM')
            require(int(summary['uniform_reference_params']) == uniform,
                    f'{dataset}/{variant}: uniform_reference != UNIFORM')
            require(int(summary['allocation']['params'])
                    <= uniform,
                    f'{dataset}/{variant}: AUTO/TILT > UNIFORM')
        require(len(bn_hashes) == 1,
                f'{dataset}: warianty nie użyły identycznego BN')
        require(uniform is not None and uniform > 0,
                f'{dataset}: zły UNIFORM')
    code = (HERE / 's2_common.py').read_text(encoding='utf-8').lower()
    for token in ('damage_post_bn_pp', 'damage_pre_bn_pp',
                  'baseline_top1', 'dev_top1'):
        require(token not in code,
                f's2_common.py zawiera {token} w logice alokacji')
    worker = (HERE / 's2_worker.py').read_text(encoding='utf-8')
    require(worker.index("allocation = spec['allocation']")
            < worker.index('baseline_model = copy.deepcopy'),
            'Worker musi odczytać alokację przed oceną dev')
    return spec_hashes


def results_table(summaries):
    """Buduje tabelę metryk i różnic."""
    results = {}
    for dataset in DATASETS:
        rows = {}
        for variant in VARIANTS:
            summary = summaries[dataset][variant]
            rows[variant] = {
                'pre_bn_top1': float(summary['compressed']['pre_bn']['top1']),
                'post_bn_top1': float(summary['compressed']['post_bn']['top1']),
                'pre_bn_nll': float(summary['compressed']['pre_bn']['nll']),
                'post_bn_nll': float(summary['compressed']['post_bn']['nll']),
                'pre_vs_baseline_agreement': float(
                    summary['compressed']['pre_bn_vs_baseline'][
                        'prediction_agreement']),
                'post_vs_baseline_agreement': float(
                    summary['compressed']['post_bn_vs_baseline'][
                        'prediction_agreement']),
                'params': int(summary['compressed']['params']),
                'target_params': int(summary['target_params']),
                'delta_params': int(summary['delta_params']),
                'delta_percent': float(summary['delta_percent']),
                'baseline_params': int(summary['baseline']['params']),
            }
        base = rows['uniform']
        best_name = max(
            ('tilt_p030', 'tilt_p015'),
            key=lambda name: (rows[name]['post_bn_top1'],
                              name == 'tilt_p030'))
        for variant in VARIANTS:
            rows[variant]['delta_vs_uniform_pp'] = 100.0 * (
                rows[variant]['post_bn_top1'] - base['post_bn_top1'])
        rows['auto50']['delta_vs_tilt030_pp'] = 100.0 * (
            rows['auto50']['post_bn_top1'] - rows['tilt_p030']['post_bn_top1'])
        rows['auto50']['best_fixed_rule'] = best_name
        rows['auto50']['best_fixed_post_bn_top1'] = rows[best_name][
            'post_bn_top1']
        rows['auto50']['delta_vs_best_fixed_pp'] = 100.0 * (
            rows['auto50']['post_bn_top1']
            - rows[best_name]['post_bn_top1'])
        rows['auto50']['delta_vs_tilt015_pp'] = 100.0 * (
            rows['auto50']['post_bn_top1'] - rows['tilt_p015']['post_bn_top1'])
        rows['tilt_p030']['delta_vs_uniform_pp'] = 100.0 * (
            rows['tilt_p030']['post_bn_top1'] - base['post_bn_top1'])
        rows['tilt_p015']['delta_vs_uniform_pp'] = 100.0 * (
            rows['tilt_p015']['post_bn_top1'] - base['post_bn_top1'])
        results[dataset] = rows
    return results


def predictor_audit(cfg, audit):
    """Opisowa kontrola predyktora: stały poziom vs korelacja łączna."""
    analysis = common.read_json(
        ROOT / cfg['s1']['analysis_dir'] / 'analysis.json')
    try:
        from scipy.stats import spearmanr
    except Exception:  # pragma: no cover
        spearmanr = None
    out = {}
    for dataset in DATASETS:
        table = sorted(
            analysis['datasets'][dataset]['group_table'],
            key=lambda item: int(item['group_index']))
        levels = ('0.25', '0.5', '0.75', '0.9')
        fixed = {}
        for level in levels:
            x = [float(item['levels'][level][
                'weight_reconstruction_rel_error']) for item in table]
            y = [float(item['levels'][level]['damage_post_bn_pp'])
                 for item in table]
            fixed[level] = (
                None if spearmanr is None else {
                    'rho': float(spearmanr(x, y).statistic),
                    'pvalue': float(spearmanr(x, y).pvalue),
                })
        x = [float(item['levels'][level][
            'weight_reconstruction_rel_error'])
             for item in table for level in levels]
        y = [float(item['levels'][level]['damage_post_bn_pp'])
             for item in table for level in levels]
        pooled = None if spearmanr is None else {
            'rho': float(spearmanr(x, y).statistic),
            'pvalue': float(spearmanr(x, y).pvalue),
        }
        out[dataset] = {
            'fixed_level': fixed,
            'pooled_48_runs': pooled,
            'pooled_note': (
                'Korelacja łączna jest zmieszana z poziomem kompresji i NIE '
                'jest dowodem alokacji; AUTO używa standaryzacji w obrębie '
                'poziomu i tylko primary predyktora.'),
            'fixed_level_recon': {
                level: [float(item['levels'][level][
                    'weight_reconstruction_rel_error']) for item in table]
                for level in levels
            },
            'fixed_level_damage_post_bn': {
                level: [float(item['levels'][level]['damage_post_bn_pp'])
                        for item in table]
                for level in levels
            },
            'auto_z': audit['datasets'][dataset]['z'],
            'auto_score': audit['datasets'][dataset]['score'],
            'fixed_level_z': audit['datasets'][dataset]['fixed_level_z'],
        }
    return out


def read_group_rows(run, dataset, variant):
    """Zwraca wiersze ratio per grupa z diagnostics.json."""
    path = run / 'allocations' / dataset / variant / 'diagnostics.json'
    diagnostics = common.read_json(path)
    rows = []
    for index, group in enumerate(diagnostics['groups']):
        rows.append({
            'dataset': dataset,
            'variant': variant,
            'group_index': int(index),
            'group_name': group['name'],
            'n_channels': int(group['n_channels']),
            'width': int(group['width']),
            'ratio': float(group['ratio']),
        })
    require(len(rows) == 12, f'{dataset}/{variant}: oczekiwano 12 grup')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    require(run.is_dir(), f'Brak katalogu biegu: {run}')
    cfg = s2.load_config(run / 'config.json')
    s2.verify_sources(cfg)
    manifest = common.read_json(run / 'manifest.json')
    frozen = common.read_json(run / 'allocations_frozen.json')
    audit = common.read_json(run / 'allocation_audit.json')
    summaries, specs = load_summaries(run, cfg)
    n_artifacts = verify_artifacts(run)
    n_sources = verify_sources(run, manifest)
    spec_hashes = verify_integrity(
        run, cfg, manifest, frozen, audit, summaries, specs)
    results = results_table(summaries)
    predictor = predictor_audit(cfg, audit)

    integrity = {
        'artifacts_verified': n_artifacts,
        'sources_verified': n_sources,
        'specs_verified': len(spec_hashes),
        'allocations_frozen_utc': manifest['allocations_frozen_utc'],
        'workers_launched_utc': manifest['workers_launched_utc'],
        'freeze_before_workers': True,
        'dev_used_for_allocation': False,
        'official_test_used': False,
        'reserve_used': False,
        'same_bn_for_all_variants': True,
        'actual_model_parameter_count': True,
        'target_from_actual_uniform': True,
        'auto_not_above_uniform': True,
        'tolerance_percent': 0.5,
    }

    datasets = {}
    for dataset in DATASETS:
        uniform = summaries[dataset]['uniform']
        target = int(uniform['compressed']['params'])
        baseline_params = int(uniform['baseline']['params'])
        budgets = {}
        calibration = {}
        ratios = []
        for variant in VARIANTS:
            summary = summaries[dataset][variant]
            budgets[variant] = {
                'baseline_params': baseline_params,
                'nominal_target_params': int(
                    summary['nominal_target_params']),
                'uniform_target_params': target,
                'params': int(summary['compressed']['params']),
                'delta_params': int(summary['delta_params']),
                'delta_percent': float(summary['delta_percent']),
                'actual_reduction_fraction': float(
                    summary['compressed']['actual_reduction_vs_baseline']),
                'target_reduction_fraction': float(
                    summary['compressed']['target_reduction_fraction']),
            }
            calibration[variant] = summary['calibration_cost']
            ratios.extend(read_group_rows(run, dataset, variant))
        datasets[dataset] = {
            'results': results[dataset],
            'budgets': budgets,
            'calibration': calibration,
            'predictor': predictor[dataset],
        }

    g1 = all(
        results[dataset]['auto50']['post_bn_top1']
        > results[dataset]['uniform']['post_bn_top1']
        for dataset in DATASETS)
    g2 = all(
        results[dataset]['auto50']['post_bn_top1']
        >= results[dataset]['tilt_p030']['post_bn_top1'] - 0.005
        for dataset in DATASETS)
    g3 = True
    success = bool(g1 and g2 and g3)

    decision = {
        'G1_auto_gt_uniform_both': bool(g1),
        'G2_auto_not_worse_than_tilt030_by_0_5pp': bool(g2),
        'G3_integrity': bool(g3),
        'success_development': bool(success),
        'best_fixed_rule_per_dataset': {
            dataset: results[dataset]['auto50']['best_fixed_rule']
            for dataset in DATASETS
        },
        'auto_vs_best_fixed_pp': {
            dataset: results[dataset]['auto50']['delta_vs_best_fixed_pp']
            for dataset in DATASETS
        },
        'publication_claim': False,
        'next_step_if_success': 'recommend additional seeds only',
    }

    runtime = {
        'run_started_utc': manifest.get('started_utc'),
        'run_finished_utc': manifest.get('finished_utc'),
        'worker_total_seconds': {
            f'{dataset}/{variant}': float(
                summaries[dataset][variant]['timing']['total_seconds'])
            for dataset in DATASETS for variant in VARIANTS
        },
        'worker_bn_seconds': {
            f'{dataset}/{variant}': float(
                summaries[dataset][variant]['compressed']['bn']['seconds']
                if 'seconds' in summaries[dataset][variant]['compressed']['bn']
                else summaries[dataset][variant]['calibration_cost']['seconds'])
            for dataset in DATASETS for variant in VARIANTS
        },
    }

    analysis = {
        'experiment': cfg['experiment'],
        'stage': cfg['stage'],
        'run_id': run.name,
        'analysis_utc': utc_now(),
        'protocol': cfg['opis'],
        'integrity': integrity,
        'decision': decision,
        'datasets': datasets,
        'runtime': runtime,
        'limitations': [
            'Jeden seed42 na dataset; to etap rozwojowy, nie replikacja.',
            'Dev był używany w S1; S2 nie jest niezależnym testem.',
            'Ocena tylko na dev; test i rezerwa nietknięte.',
            'Jeden przepis treningowy na dataset; checkpointy nie są niezależne.',
            'Brak Tiny ImageNet/ImageNet i innych architektur.',
            'Sukces oznacza tylko rekomendację dalszych seedów, nie publikację.',
        ],
    }
    out = output_dir()
    common.write_json_atomic(out / 'analysis.json', analysis)

    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'dataset', 'variant', 'pre_bn_top1', 'post_bn_top1',
            'pre_bn_nll', 'post_bn_nll', 'params', 'target_params',
            'delta_params', 'delta_percent', 'delta_vs_uniform_pp',
        ])
        for dataset in DATASETS:
            for variant in VARIANTS:
                row = results[dataset][variant]
                writer.writerow([
                    dataset, variant, f"{row['pre_bn_top1']:.6f}",
                    f"{row['post_bn_top1']:.6f}",
                    f"{row['pre_bn_nll']:.6f}",
                    f"{row['post_bn_nll']:.6f}",
                    row['params'], row['target_params'],
                    row['delta_params'], f"{row['delta_percent']:.6f}",
                    f"{row['delta_vs_uniform_pp']:.4f}",
                ])

    with (out / 'budgets.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'dataset', 'variant', 'baseline_params', 'nominal_target_params',
            'uniform_target_params', 'params', 'delta_params',
            'delta_percent', 'actual_reduction_fraction',
            'target_reduction_fraction',
        ])
        for dataset in DATASETS:
            for variant in VARIANTS:
                row = datasets[dataset]['budgets'][variant]
                writer.writerow([
                    dataset, variant, row['baseline_params'],
                    row['nominal_target_params'],
                    row['uniform_target_params'], row['params'],
                    row['delta_params'], f"{row['delta_percent']:.6f}",
                    f"{row['actual_reduction_fraction']:.8f}",
                    f"{row['target_reduction_fraction']:.8f}",
                ])

    ratio_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            ratio_rows.extend(read_group_rows(run, dataset, variant))
    with (out / 'ratios.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=['dataset', 'variant', 'group_index',
                                'group_name', 'n_channels', 'width', 'ratio'])
        writer.writeheader()
        writer.writerows(ratio_rows)

    with (out / 'calibration.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'dataset', 'variant', 'samples', 'batches', 'batch_size',
            'seconds', 'gradients', 'full_train',
            'labels_used_in_optimization',
        ])
        for dataset in DATASETS:
            for variant in VARIANTS:
                row = datasets[dataset]['calibration'][variant]
                writer.writerow([
                    dataset, variant, row['samples'], row['batches'],
                    row['batch_size'], f"{row['seconds']:.4f}",
                    row['gradients'], row['full_train'],
                    row['labels_used_in_optimization'],
                ])

    with (out / 'predictors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'dataset', 'group_index', 'group_name', 'score', 'z',
            'fixed_level_z', 'recon_025', 'recon_050', 'recon_075',
            'recon_090',
        ])
        for dataset in DATASETS:
            audit_ds = audit['datasets'][dataset]
            matrix = audit_ds['recon_matrix']
            for index, name in enumerate(audit_ds['group_names']):
                writer.writerow([
                    dataset, index, name,
                    f"{audit_ds['score'][index]:.8f}",
                    f"{audit_ds['z'][index]:.8f}",
                    f"{audit_ds['fixed_level_z'][index]:.8f}",
                    f"{matrix[index][0]:.8f}",
                    f"{matrix[index][1]:.8f}",
                    f"{matrix[index][2]:.8f}",
                    f"{matrix[index][3]:.8f}",
                ])

    lines = [
        '# E016 S2 — automatyczna alokacja 50%: analiza',
        '',
        f'Run: `{run.name}`. Status: **{manifest["status"]}**.',
        f'Alokacje zamrożone: {manifest["allocations_frozen_utc"]}; '
        f'workery start: {manifest["workers_launched_utc"]}.',
        '',
        '## Decyzja',
        '',
        f'- G1 (AUTO > UNIFORM na obu): **{"TAK" if g1 else "NIE"}**',
        f'- G2 (AUTO nie gorszy od TILT+0.30 o >0,5 pp): '
        f'**{"TAK" if g2 else "NIE"}**',
        f'- G3 (integralność): **{"TAK" if g3 else "NIE"}**',
        f'- Sukces rozwojowy: **{"TAK" if success else "NIE"}**',
        '',
        '## Wynik główny (dev top-1 po BN)',
        '',
        '| dataset | UNIFORM | TILT+0.30 | TILT+0.15 | AUTO50 | AUTO−UNIFORM | AUTO−TILT+0.30 | AUTO−best fixed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        r = results[dataset]
        best = r['auto50']['best_fixed_rule']
        lines.append(
            f"| {dataset} | {100*r['uniform']['post_bn_top1']:.3f}% | "
            f"{100*r['tilt_p030']['post_bn_top1']:.3f}% | "
            f"{100*r['tilt_p015']['post_bn_top1']:.3f}% | "
            f"{100*r['auto50']['post_bn_top1']:.3f}% | "
            f"{r['auto50']['delta_vs_uniform_pp']:+.3f} pp | "
            f"{r['auto50']['delta_vs_tilt030_pp']:+.3f} pp | "
            f"{r['auto50']['delta_vs_best_fixed_pp']:+.3f} pp "
            f"({best}) |")
    lines += [
        '',
        '## Metryki przed/po BN',
        '',
        '| dataset | wariant | pre-BN | post-BN | NLL post | agreement post |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            r = results[dataset][variant]
            lines.append(
                f"| {dataset} | {variant} | "
                f"{100*r['pre_bn_top1']:.3f}% | "
                f"{100*r['post_bn_top1']:.3f}% | "
                f"{r['post_bn_nll']:.4f} | "
                f"{r['post_vs_baseline_agreement']:.4f} |")
    lines += [
        '',
        '## Budżety i target UNIFORM',
        '',
        '| dataset | wariant | N | nominal | UNIFORM target | params | delta | delta % | redukcja faktyczna |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            b = datasets[dataset]['budgets'][variant]
            lines.append(
                f"| {dataset} | {variant} | {b['baseline_params']} | "
                f"{b['nominal_target_params']} | "
                f"{b['uniform_target_params']} | {b['params']} | "
                f"{b['delta_params']} | {b['delta_percent']:+.4f}% | "
                f"{100*b['actual_reduction_fraction']:.3f}% |")
    lines += [
        '',
        '## AUTO — ratio per grupa',
        '',
        '| grupa | CIFAR-10 ratio | CIFAR-10 width | CIFAR-100 ratio | CIFAR-100 width |',
        '|---|---:|---:|---:|---:|',
    ]
    auto_rows = {
        dataset: read_group_rows(run, dataset, 'auto50')
        for dataset in DATASETS
    }
    for index in range(12):
        a = auto_rows['cifar10'][index]
        b = auto_rows['cifar100'][index]
        lines.append(
            f"| {a['group_name']} | {a['ratio']:.4f} | {a['width']} | "
            f"{b['ratio']:.4f} | {b['width']} |")
    lines += [
        '',
        '## Koszt kalibracji BN',
        '',
        '| dataset | wariant | samples | batches | seconds | pełny train | gradienty |',
        '|---|---|---:|---:|---:|---|---|',
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            c = datasets[dataset]['calibration'][variant]
            lines.append(
                f"| {dataset} | {variant} | {c['samples']} | {c['batches']} | "
                f"{c['seconds']:.2f} | {c['full_train']} | {c['gradients']} |")
    lines += [
        '',
        '## Kontrola predyktora',
        '',
        'AUTO używa wyłącznie `weight_reconstruction_rel_error` z S1, po '
        'standaryzacji w obrębie poziomu i uśrednieniu po poziomach. '
        'Poniżej korelacje opisowe ze szkodą post-BN S1 **przy stałym '
        'poziomie** oraz korelacja łączna (ta ostatnia jest zmieszana z '
        'poziomem kompresji i nie jest dowodem alokacji).',
        '',
        '| dataset | poziom | Spearman recon vs damage post-BN | p |',
        '|---|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        p = predictor[dataset]
        for level, value in p['fixed_level'].items():
            if value is None:
                lines.append(f'| {dataset} | {level} | n/a | n/a |')
            else:
                lines.append(
                    f"| {dataset} | {level} | {value['rho']:.3f} | "
                    f"{value['pvalue']:.4f} |")
        pooled = p['pooled_48_runs']
        if pooled is not None:
            lines.append(
                f"| {dataset} | łącznie 48 | {pooled['rho']:.3f} | "
                f"{pooled['pvalue']:.4f} |")
    lines += [
        '',
        '**Uwaga:** korelacja łączna 48 przebiegów nie jest dowodem '
        'alokacji; jest zmieszana z poziomem kompresji. AUTO nie używa dev.',
        '',
        '## Integralność i czas',
        '',
        f"- Artefakty sprawdzone: {n_artifacts}",
        f"- Snapshoty źródeł sprawdzone: {n_sources}",
        f"- Specy sprawdzone: {len(spec_hashes)}",
        '- Alokacje zamrożone przed startem workerów: **TAK**',
        '- Dev nieużywany do alokacji: **TAK**',
        '- Test/rezerwa nietknięte: **TAK**',
        '- Ten sam BN dla metod: **TAK**',
        '- Target = faktyczne parametry UNIFORM: **TAK**',
        '- AUTO nie przekracza UNIFORM: **TAK**',
        f"- Tolerancja: 0,5% poniżej targetu, 0 powyżej",
        '',
        '## Ograniczenia',
        '',
        '- Jeden seed42 na dataset; to etap rozwojowy, nie replikacja.',
        '- Dev był używany w S1; S2 nie jest niezależnym testem.',
        '- Ocena tylko na dev; test i rezerwa nietknięte.',
        '- Jeden przepis treningowy; checkpointy nie są niezależne.',
        '- Sukces oznacza tylko rekomendację dalszych seedów, nie publikację.',
    ]
    (out / 'analysis.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(f'E016_S2_ANALYSIS={out}', flush=True)
    print(f'E016_S2_G1={"YES" if g1 else "NO"}', flush=True)
    print(f'E016_S2_G2={"YES" if g2 else "NO"}', flush=True)
    print(f'E016_S2_SUCCESS={"YES" if success else "NO"}', flush=True)
    print('E016_S2_ANALYSIS_OK', flush=True)


if __name__ == '__main__':
    main()
