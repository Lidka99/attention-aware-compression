"""E006: kontrola FWER sygnalow wtornych (analiza po biegu, CPU).
Eksperyment: 2026-10-11_E006_kryterium_bramkowe_predykcja; protokol: OPIS.md v1.1.
Wejscia: pairs.csv z zamrozonego biegu
``runs/E006/20261011T053842Z_eb6b660_seed42_c5gen8pp`` (pary, sygnaly, szkody).
Wyjscia: wynik_fwer.json w tym katalogu (rho, p permutacyjne, p maksimum).
Rola: dopelnienie zadeklarowanego w OPIS punktu 4 kryterium oceny (rodzina
wtorna, kontrola FWER w przyblizeniu) bez wplywu na decyzje pierwszorzedna.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from e006_metrics import spearman

ROOT = HERE.parents[3]
RUN = ROOT / 'runs/E006/20261011T053842Z_eb6b660_seed42_c5gen8pp'
SIGNALS = ('s_pre', 'g_corr', 'g_lstd', 'g_flip', 'g_amp')
DAMAGE = 'block_relative_mse'
REPS = 10000
SEED = 20261014

rows = list(csv.DictReader((RUN / 'pairs.csv').open()))
damage = np.asarray([float(row[DAMAGE]) for row in rows])
x = {key: np.asarray([float(row[key]) for row in rows]) for key in SIGNALS}
observed = {key: float(spearman(x[key], damage)) for key in SIGNALS}
max_observed = max(abs(value) for value in observed.values())

rng = np.random.default_rng(SEED)
counts = {key: 0 for key in SIGNALS}
max_null = np.empty(REPS, dtype=np.float64)
for rep in range(REPS):
    permuted = damage[rng.permutation(damage.size)]
    values = {key: float(spearman(x[key], permuted)) for key in SIGNALS}
    for key in SIGNALS:
        if abs(values[key]) >= abs(observed[key]):
            counts[key] += 1
    max_null[rep] = max(abs(value) for value in values.values())

result = {
    'experiment': '2026-10-11_E006_kryterium_bramkowe_predykcja',
    'run_id': RUN.name,
    'damage': DAMAGE,
    'signals': list(SIGNALS),
    'observed_spearman': observed,
    'permutation_p_two_sided': {key: (1 + counts[key]) / (REPS + 1) for key in SIGNALS},
    'max_statistic_observed': float(max_observed),
    'max_statistic_p_fwer': float((1 + int(np.count_nonzero(max_null >= max_observed)))
                                  / (REPS + 1)),
    'reps': REPS,
    'seed': SEED,
    'note': 'rodzina wtorna; kontrola FWER w przyblizeniu; bez wplywu na decyzje',
}
(HERE / 'wynik_fwer.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print('E006_FWER=' + json.dumps(result, sort_keys=True))
