"""E004: wspólne ścieżki, hashe kodu i zależności (tylko CPU, bez modeli).
Eksperyment: 2026-10-10_E004_wspolne_dostrajanie; protokół: PROTOKOL_v1.md i OPIS.md.
Wejścia: pliki E004, E003 i E002 (tylko do odczytu i hashowania).
Wyjścia: słowniki ścieżek i SHA-256 używane przez kontrole i launcher.
To nie uruchamia treningu ani nie wczytuje wag.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E003 = ROOT / 'experiments/2026-10-09_E003_kompresja_10proc_parametrow'
E002 = ROOT / 'experiments/2026-10-01_E002_obserwacja_se'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + '\n')


DEPENDENCIES = (
    'experiments/2026-10-09_E003_kompresja_10proc_parametrow/support.py',
    'experiments/2026-10-09_E003_kompresja_10proc_parametrow/folding.py',
    'experiments/2026-10-09_E003_kompresja_10proc_parametrow/summarize.py',
    'experiments/2026-10-01_E002_obserwacja_se/common.py',
)


def code_hashes():
    """Hashe kodu E004 i przypiętych zależności; używa ich kontrola i launcher."""
    return {'local': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
            'dependencies': {relative: digest(ROOT / relative) for relative in DEPENDENCIES}}
