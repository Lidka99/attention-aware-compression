"""E022: analiza CPU mechanizmu z istniejących JSON; pełny protokół OPIS.md.
Wejście: E016/E020/E014/E015/E019 i --run-dir E022. Wyjście: unikalna
analiza JSON, CSV, tabele Markdown, wykres i hashe wszystkich wejść.
Nie wczytuje danych obrazowych ani nie uruchamia treningu.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S1 = ROOT/'runs/E016/s1_20261027T215948Z_cb9ffaf_d98f0ed9'
TINY = ROOT/'runs/E020/20261031T055920Z_cb9ffaf_sens_e650f55f'
ABL = ROOT/'runs/E020/20261031T061615Z_cb9ffaf_abl_bbb35b29'
REF = {'cifar10': ROOT/'runs/E014/20261027T094707Z_267904d_6a0e93eb',
       'cifar100': ROOT/'runs/E015/fold_20261027T210324Z_cb9ffaf_36a01249',
       'tiny': ROOT/'runs/E019/fold_20261030T220139Z_cb9ffaf_51bc405f'}
LEVELS = [.25, .5, .75, .9]
HASHES = {}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    HASHES[str(p.relative_to(ROOT))] = sha(p)
    return json.loads(p.read_text())


def rho(x,y):
    return float(spearmanr(x,y).statistic)


def table(rows):
    keys = list(rows[0])
    def fmt(v):
        return f'{v:.3f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |', '| '+' | '.join(['---']*len(keys))+' |']+
                     ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])


def save_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def refpath(ds, seed, budget, variant):
    base = REF[ds]
    if ds == 'cifar10':
        base = base/'seeds'/f'seed_{seed}'/'allocations'
    else:
        base = base/'allocations'/f'seed_{seed}'
    return base/f'b{budget}_{variant}'/'summary.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    manifest = load(run/'manifest.json')
    assert manifest['status'] == 'completed'
    assert manifest['protocol_sha256'] == sha(HERE/'OPIS.md')
    assert manifest['summary']['exit_codes'] == {str(i):0 for i in range(4)}
    for v in manifest['reused'].values():
        assert sha(ROOT/v['path']) == v['sha256']
    for name, expected in manifest['e022_sources'].items():
        assert sha(HERE/name) == expected
    old = ROOT/'experiments/2026-10-30_E020_mechanizm_alokacji'
    inventory = load(old/'configs/inventory.json')
    for ds, meta in inventory['datasets'].items():
        assert sha(Path(meta['split'])) == meta['split_sha256']
        for entry in meta['seeds'].values():
            assert sha(Path(entry['checkpoint'])) == entry['checkpoint_sha256']
    out = run/('analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    rows, controls = [], []
    for ds in REF:
        base = TINY/'items' if ds=='tiny' else S1/'items'/ds
        paths = sorted(base.glob('*/summary.json'))
        assert len(paths)==49
        for p in paths:
            s = load(p)
            assert s['status']=='completed' and s['seed']==42
            assert not s.get('official_test_used',False) and not s.get('reserve_used',False)
            b = s['baseline']['metrics']['top1']
            pre, post = [s['compressed'][k]['top1'] for k in ('pre_bn','post_bn')]
            assert np.isclose(100*(b-pre), s['damage']['pre_bn_top1_pp'])
            assert np.isclose(100*(b-post), s['damage']['post_bn_top1_pp'])
            if s['item']['control']:
                agreement = s['compressed']['pre_bn_vs_baseline']['prediction_agreement']
                assert abs(100*(b-pre)) <= .05 and agreement >= .999999
                controls.append(dict(dataset=ds, baseline=100*b, pre_damage=100*(b-pre), post_damage=100*(b-post), agreement=agreement))
                continue
            rows.append(dict(dataset=ds, group=s['item']['group_name'], g=s['item']['group_index'], level=s['item']['level'],
                             baseline=100*b, pre=100*(b-pre), post=100*(b-post), recovery=100*(post-pre),
                             reconstruction=s['predictors']['weight_reconstruction_rel_error'],
                             nll_pre=s['damage']['pre_bn_nll_delta'], nll_post=s['damage']['post_bn_nll_delta'],
                             source=str(p.relative_to(ROOT))))
    assert len(rows)==144 and len(controls)==3
    for r in rows:
        r['post_vs_bn_control'] = r['post']-next(c['post_damage'] for c in controls if c['dataset']==r['dataset'])
    profiles, critical, pooled = [], [], []
    for ds in REF:
        d = [r for r in rows if r['dataset']==ds]
        assert len({(r['g'],r['level']) for r in d})==48
        pooled.append(dict(dataset=ds, rho_reconstruction_pooled=rho([r['reconstruction'] for r in d],[r['post'] for r in d])))
        for level in LEVELS:
            sub = sorted([r for r in d if r['level']==level], key=lambda r:r['g'])
            assert len(sub)==12
            pre,post,rec,depth,err = [np.array([r[k] for r in sub]) for k in ('pre','post','recovery','g','reconstruction')]
            profiles.append(dict(dataset=ds, level=level, pre=float(pre.mean()), post=float(post.mean()), recovery=float(rec.mean()),
                                 recovery_fraction=float(rec.sum()/pre.sum()), relative_post=float(post.mean()/sub[0]['baseline']),
                                 post_vs_bn_control=float(np.mean([r['post_vs_bn_control'] for r in sub])),
                                 rho_depth_pre=rho(depth,pre), rho_depth_post=rho(depth,post),
                                 rho_reconstruction_pre=rho(err,pre), rho_reconstruction_post=rho(err,post),
                                 rho_reconstruction_nll=rho(err,[r['nll_post'] for r in sub])))
            if level==.75:
                for r in sorted(sub,key=lambda r:r['post'],reverse=True)[:3]:
                    critical.append({k:r[k] for k in ('dataset','group','pre','post','recovery','post_vs_bn_control')})
    ablations, alignment, allocations = [], [], []
    for ds in REF:
        d = [r for r in rows if r['dataset']==ds]
        for seed in range(42,46):
            for budget in (30,50,70):
                u,t = [load(refpath(ds,seed,budget,v)) for v in ('uniform','tilt_p030')]
                name = f'{ds}_seed{seed}_b{budget}_antitilt_p030'
                rel = Path('allocations')/ds/f'seed_{seed}'/name/'summary.json'
                p = ABL/rel if (ABL/rel).exists() else run/rel
                a = load(p)
                for s in (u,t,a):
                    assert s['status']=='completed'
                    assert np.isclose(s['baseline']['metrics']['top1'],u['baseline']['metrics']['top1'])
                delta = 100*(a['allocation']['params']/u['allocation']['params']-1)
                assert -1 <= delta <= 0
                ablations.append(dict(dataset=ds,seed=seed,budget=budget,
                                      anti_minus_uniform=100*(a['compressed']['post_bn']['top1']-u['compressed']['post_bn']['top1']),
                                      anti_minus_tilt=100*(a['compressed']['post_bn']['top1']-t['compressed']['post_bn']['top1']),
                                      tilt_minus_uniform=100*(t['compressed']['post_bn']['top1']-u['compressed']['post_bn']['top1']),
                                      budget_delta_percent=delta))
                for variant,s in [('uniform',u),('tilt',t),('anti',a)]:
                    allocations.append(dict(dataset=ds,seed=seed,budget=budget,variant=variant,
                                            pre=100*s['compressed']['pre_bn']['top1'],post=100*s['compressed']['post_bn']['top1']))
                ur,tr = [np.array(s['allocation']['ratios']) for s in (u,t)]
                # Operator: width=floor(n_channels*(1-ratio)); ratio to kompresja.
                # U-T>0 oznacza mniej kompresji (więcej kanałów) w TILT.
                protection = ur-tr
                fixed = sorted([r for r in d if r['level']==.75],key=lambda r:r['g'])
                def damage_at(ratios, key):
                    vals=[]
                    for g,ratio in enumerate(ratios):
                        assert 0<=ratio<=.9
                        sub=sorted([r for r in d if r['g']==g],key=lambda r:r['level'])
                        vals.append(float(np.interp(ratio,[0]+LEVELS,[0]+[r[key] for r in sub])))
                    return np.array(vals)
                alignment.append(dict(dataset=ds,seed=seed,budget=budget,
                                      rho_protection_pre=rho(protection,[r['pre'] for r in fixed]),
                                      rho_protection_post=rho(protection,[r['post'] for r in fixed]),
                                      interpolated_post_saving=float((damage_at(ur,'post')-damage_at(tr,'post')).sum()),
                                      top3_protection=float(np.mean([protection[r['g']] for r in sorted(fixed,key=lambda r:r['post'],reverse=True)[:3]]))))
    aggregate=[]
    for ds in REF:
        for budget in (30,50,70):
            sub=[r for r in ablations if r['dataset']==ds and r['budget']==budget]
            al=[r for r in alignment if r['dataset']==ds and r['budget']==budget]
            aggregate.append(dict(dataset=ds,budget=budget,
                                  anti_uniform=float(np.mean([r['anti_minus_uniform'] for r in sub])),
                                  anti_uniform_sd=float(np.std([r['anti_minus_uniform'] for r in sub],ddof=1)),
                                  anti_uniform_negative=sum(r['anti_minus_uniform']<0 for r in sub),
                                  anti_tilt=float(np.mean([r['anti_minus_tilt'] for r in sub])),
                                  tilt_uniform=float(np.mean([r['tilt_minus_uniform'] for r in sub])),
                                  protection_pre=float(np.mean([r['rho_protection_pre'] for r in al])),
                                  protection_post=float(np.mean([r['rho_protection_post'] for r in al])),
                                  interpolated_saving=float(np.mean([r['interpolated_post_saving'] for r in al]))))
            aggregate[-1]['anti_tilt_sd']=float(np.std([r['anti_minus_tilt'] for r in sub],ddof=1))
            aggregate[-1]['tilt_uniform_sd']=float(np.std([r['tilt_minus_uniform'] for r in sub],ddof=1))
    pred_support={ds:sum(r['rho_reconstruction_post']>=.5 for r in profiles if r['dataset']==ds)>=3 for ds in REF}
    b50={r['dataset']:r for r in aggregate if r['budget']==50}
    gates=dict(direction_cifar10_tiny=all(b50[ds]['anti_uniform']<0 and b50[ds]['anti_uniform_negative']>=3 for ds in ('cifar10','tiny')),
               reverse_helps_cifar100=b50['cifar100']['anti_uniform']>=0 or b50['cifar100']['anti_tilt']>0,
               strong_post_protection_at50={ds:b50[ds]['protection_post']>=.5 for ds in REF})
    allocation_means=[]
    for ds in REF:
        for budget in (30,50,70):
            for variant in ('uniform','tilt','anti'):
                sub=[r for r in allocations if r['dataset']==ds and r['budget']==budget and r['variant']==variant]
                allocation_means.append(dict(dataset=ds,budget=budget,variant=variant,
                                             pre=float(np.mean([r['pre'] for r in sub])),
                                             pre_sd=float(np.std([r['pre'] for r in sub],ddof=1)),
                                             post=float(np.mean([r['post'] for r in sub])),
                                             post_sd=float(np.std([r['post'] for r in sub],ddof=1))))
    new = [load(p) for p in sorted((run/'allocations').glob('**/summary.json'))]
    assert len(new)==8 and {s['gpu'] for s in new}==set(range(4))
    elapsed=(datetime.fromisoformat(manifest['finished_utc'])-datetime.fromisoformat(manifest['started_utc'])).total_seconds()
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    result=dict(profiles=profiles,critical=critical,controls=controls,pooled=pooled,aggregate=aggregate,gates=gates,allocation_means=allocation_means,
                predictor_support=pred_support,predictor_joint_support=sum(pred_support.values())>=2,
                integrity=dict(group_items=147,ablations=36,new_items=8,gpus=sorted({s['gpu'] for s in new}),
                               elapsed_seconds=elapsed,run_bytes=size,resource_gate=elapsed<=5400 and size<=.5*1024**3),
                source_hashes=HASHES,analysis_code_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'OPIS.md'))
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rs in [('groups',rows),('profiles',profiles),('ablations',ablations),('alignment',alignment),('allocations',allocations)]:
        save_csv(out/f'{name}.csv',rs)
    text='\n\n'.join(['# E022 — tabele wygenerowane',str(run.relative_to(ROOT)), '## Profile',table(profiles),
                       '## Krytyczne grupy (75%)',table(critical),'## Kontrole',table(controls),
                       '## Korelacje pooled (kontrast)',table(pooled),'## Ablacja i ochrona',table(aggregate),
                       'Ochrona i interpolacja używają tylko profili seed42; alokacje szerokości nie stanowią replikacji szkody. Suma interpolowana jest opisowa, nie predykcją addytywną.',
                       '## Alokacje: średnia i SD po 4 seedach',table(allocation_means),
                       '## Kryteria',str(gates),'## Wsparcie predyktora',str(pred_support), '## Integralność',str(result['integrity'])])
    (out/'TABLES.md').write_text(text+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharex=True)
    for ax,ds in zip(axes,REF):
        sub=sorted([r for r in rows if r['dataset']==ds and r['level']==.75],key=lambda r:r['g'])
        for key in ('pre','post'):
            ax.plot([r['g'] for r in sub],[r[key] for r in sub],marker='o',label=key+' BN')
        ax.set(title=ds,xlabel='Indeks grupy (0–11)',ylabel='Szkoda [pp]')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'profile75.png',dpi=160)
    print(text)
    print(f'ANALYSIS={out}')

if __name__=='__main__':
    main()
