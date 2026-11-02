"""E022: adaptuje worker E020 bez zmiany operatora; patrz OPIS.md.
Wejście: shard brakujących ablacji; wyjście: summary/diagnostics bez logitów.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'experiments/2026-10-30_E020_mechanizm_alokacji'))
import e020_common as c
import ablation_worker as w

c.EXPERIMENT = HERE.name
c.OPIS_PATH = str((HERE/'OPIS.md').relative_to(c.ROOT))
# Zachowujemy ocenę i wszystkie metryki, pomijamy kosztowny zapis logitów.
original_write = c.write_json_atomic
original_digest = c.digest
w.np.savez_compressed = lambda *args, **kwargs: None
c.digest = lambda p: None if str(p).endswith('logits.npz') else original_digest(p)
def write(path, value):
    if Path(path).name == 'summary.json':
        value.pop('logits_file', None)
        value.pop('logits_sha256', None)
        value['logits_saved'] = False
        print(value.get('dataset'), value.get('seed'), value.get('budget'), value.get('status'), flush=True)
    return original_write(path, value)
c.write_json_atomic = write

if __name__ == '__main__':
    w.main()
