"""E025: audyt post hoc przejściowego efektu EMA, bez GPU; ../OPIS.md.
Wejście: zapisane statystyki BN; wyjście: ema_audit.json/Markdown w runie.
Nie zmienia kryteriów ani wyników; sprawdza analityczną zależność wariancji.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    run=p.parse_args().run_dir.resolve()
    residuals=[];examples=[];source_hashes={}
    for seed in range(42,46):
        for model in ('teacher','UNIFORM','TILT','SEARCH'):
            for size in ('A','L'):
                for view in ('plain','aug'):
                    pair=[]
                    for est in ('EMA_reset','EMA_retain'):
                        path=run/f'seed_{seed}/items/{model}_{size}_{view}_{est}.json'
                        pair.append(json.loads(path.read_text()))
                        source_hashes[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    reset,retain=pair
                    alpha=reset['bn']['initial_weight_ema']
                    for a,b in zip(reset['bn']['layers'],retain['bn']['layers']):
                        assert a['name']==b['name'] and a['var_before_sha256']==b['var_before_sha256']
                        predicted=alpha*(1-a['var_before_mean'])
                        observed=a['var_after_mean']-b['var_after_mean']
                        residuals.append(abs(predicted-observed))
                        if model=='teacher' and view=='plain' and a['name']=='layer4.1.bn2':
                            examples.append(dict(seed=seed,size=size,layer=a['name'],weight=alpha,
                                                 variance_source=a['var_before_mean'],variance_reset=a['var_after_mean'],
                                                 variance_retain=b['var_after_mean'],reset_retain_ratio=a['var_after_mean']/b['var_after_mean'],
                                                 equation_error=abs(predicted-observed)))
    assert max(residuals)<1e-5
    result=dict(post_hoc=True,equation='v_reset_N - v_retain_N = 0.9**N * (1 - v_source)',
                comparisons=len(residuals),max_abs_error=max(residuals),examples=examples,source_hashes=source_hashes,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=run/'ema_audit'
    out.mkdir(exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    text='# E025 — audyt post hoc EMA\n\n'+result['equation']+'\n\n'
    text+=f"Porównania warstw: {len(residuals)}; max błąd: {max(residuals):.3g}.\n\n"
    text+='| seed | size | waga inicjalizacji | var źródła | var reset | var retain | iloraz |\n|---|---|---|---|---|---|---|\n'
    for r in examples:
        text+=f"| {r['seed']} | {r['size']} | {r['weight']:.6g} | {r['variance_source']:.6g} | {r['variance_reset']:.6g} | {r['variance_retain']:.6g} | {r['reset_retain_ratio']:.3f} |\n"
    (out/'AUDYT.md').write_text(text)
    print(text)

if __name__=='__main__':
    main()
