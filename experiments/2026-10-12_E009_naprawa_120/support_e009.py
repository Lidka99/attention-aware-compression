"""E009: wsparcie naprawy 120 epok po etapie 1 kompresji 30% (projekt E008).
Eksperyment: 2026-10-12_E009_naprawa_120; protokol: OPIS.md i PROTOKOL_v1.md.
Importuje sprawdzone helpery E008 (pin hashem w configu) i definiuje pochodzenie
wejsc: modele etapu 1 E008 (BASE/PRE/POST/RANDOM przy 30% redukcji).
Bez GPU i bez treningu - stale, hashe i kontrole wejsc.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E008_DIR = ROOT / 'experiments/2026-10-12_E008_kompresja_224'
if str(E008_DIR) not in sys.path:
    sys.path.insert(0, str(E008_DIR))

from support_e008 import (DATA, digest, read_json, write_json, require, model_digest,
                          seed_all, seed_worker, load_data, loader, base_model,
                          eval_transform)  # noqa: E402
from folding_e008 import materialize_model, count_parameters  # noqa: E402


def e008_run(cfg):
    return ROOT / cfg['input']['e008_run']


def verify_inputs(cfg):
    run = e008_run(cfg)
    manifest = read_json(run / 'manifest.json')
    require(manifest['status'] == 'completed' and manifest.get('exit_code') == 0,
            'E008 run not completed/exit nonzero')
    stage1 = run / 'stage1'
    for name, expected in cfg['input']['sha256'].items():
        require(digest(stage1 / name) == expected, f'E008 input changed: {name}')
    require(digest(E008_DIR / 'support_e008.py') == cfg['input']['support_e008_sha256'],
            'support_e008.py changed')
    require(digest(E008_DIR / 'folding_e008.py') == cfg['input']['folding_e008_sha256'],
            'folding_e008.py changed')
    return {'e008_run': run.name, 'e008_git_sha': manifest['git_sha'],
            'inputs': cfg['input']['sha256'],
            'support_e008_sha256': cfg['input']['support_e008_sha256'],
            'folding_e008_sha256': cfg['input']['folding_e008_sha256']}
