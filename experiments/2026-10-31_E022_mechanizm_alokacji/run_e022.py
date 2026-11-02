"""E022: uzupełnia osiem brakujących ablacji E020 na 4 GPU; OPIS.md.
Wejście: zamrożone E020 i istniejące checkpointy; wyjście: nowy runs/E022.
Bez treningu, bez modyfikacji poprzednich wyników.
"""
import os
import sys
import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/2026-10-30_E020_mechanizm_alokacji'
sys.path.insert(0, str(OLD))
import e020_common as c
import launch_ablation as launch

SOURCE = ROOT / 'runs/E020/20261031T061615Z_cb9ffaf_abl_bbb35b29'


def main():
    info = c.preflight(gpu_count=4, min_disk_gib=1.0)
    cfg = c.load_ablation_config()
    all_items = launch.build_items(cfg)
    reused, missing = {}, []
    for item in all_items:
        p = SOURCE / 'allocations' / item['dataset'] / f"seed_{item['seed']}" / item['name'] / 'summary.json'
        if p.exists() and c.read_json(p).get('status') == 'completed':
            reused[item['name']] = {'path': str(p.relative_to(ROOT)), 'sha256': c.digest(p)}
        else:
            missing.append(item)
    c.require(len(reused) == 28 and len(missing) == 8, 'Nieoczekiwany stan źródeł')
    c.EXPERIMENT = HERE.name
    c.OPIS_PATH = str((HERE / 'OPIS.md').relative_to(ROOT))
    run = ROOT / 'runs/E022' / (c.utc_now().replace(':', '').replace('-', '')[:15] + '_' + secrets.token_hex(4))
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    cfg = dict(cfg, experiment=HERE.name, opis=c.OPIS_PATH)
    c.write_json_atomic(run / 'config.json', cfg)
    shards = c.shard_items(missing, 4)
    manifest = c.write_manifest(run, cfg, missing, {str(k): [i['name'] for i in v] for k,v in shards.items()})
    manifest.update(reused=reused, launcher_pid=os.getpid(), protocol_sha256=c.digest(HERE/'OPIS.md'),
                    e022_sources={p.name:c.digest(p) for p in HERE.glob('*.py')})
    c.write_json_atomic(run/'manifest.json', manifest)
    c.write_run_environment(run, info)
    print(f'RUN={run}', flush=True)
    codes = c.launch_shards(run, HERE/'worker.py', run/'config.json', missing, 4, 5400)
    results = list((run/'allocations').glob('**/summary.json'))
    ok = len(results) == 8 and all(c.read_json(p)['status']=='completed' for p in results) and all(v==0 for v in codes.values())
    c.finish_manifest(run, 'completed' if ok else 'failed', {'exit_codes':codes, 'new_completed':sum(c.read_json(p)['status']=='completed' for p in results), 'reused_completed':28})
    print(f'completed={ok}; exit_codes={codes}', flush=True)
    raise SystemExit(0 if ok else 1)

if __name__ == '__main__':
    main()
