"""E034: kontrola zakończenia i hashy treningu; wejście run, wynik stdout. OPIS.md."""
import hashlib
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
m = read(run / 'manifest.json')
assert m['status'] == 'completed' and m['exit_codes'] == [0] * 4
assert sha(run / 'config.json') == m['config_sha256']
for seed in (42, 43, 44, 45):
    assert read(run / f'seed_{seed}/status.json')['status'] == 'completed'
    for ds in ('cifar10', 'cifar100'):
        base = run / f'seed_{seed}' / ds
        s = read(base / 'summary.json')
        rows = [json.loads(line) for line in (base / 'metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows] == list(range(1, 201))
        assert s['status'] == 'completed' and s['epochs'] == 200
        assert not s['official_test_used'] and not s['reserve_used']
        assert sha(base / 'last.pt') == s['checkpoint_sha256'] == rows[-1]['checkpoint_sha256']
        top = s['final_metrics']['dev_top1']
        assert math.isfinite(top) and top >= (0.90 if ds == 'cifar10' else 0.65)
        print(seed, ds, f'{100 * top:.3f}%', 'PASS')
