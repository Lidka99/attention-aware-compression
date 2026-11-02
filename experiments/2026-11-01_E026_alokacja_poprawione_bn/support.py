"""E026: zweryfikowane wejścia i wspólny operator E024/BN E025; OPIS.md.
Wejście: zapisane źródła; wyjście: hashe i narzędzia bez zmiany historii.
"""
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E025=ROOT/'experiments/2026-11-01_E025_diagnostyka_bn'
REF=ROOT/'runs/E025/20261101T121011Z_9c3d4231'
spec=importlib.util.spec_from_file_location('_e026_bn',E025/'common.py')
bn=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=bn
spec.loader.exec_module(bn)
from data import digest,write


def sources():
    manifest=bn.read(REF/'manifest.json')
    assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
    for name,expected in manifest['source_hashes'].items():
        assert digest(E025/name)==expected
    result=bn.inputs_manifest()
    for p in (bn.TOOLKIT/'folding.py',bn.TOOLKIT/'data.py',bn.TOOLKIT/'search.py',E025/'common.py',REF/'manifest.json'):
        result[str(p.relative_to(ROOT))]=digest(p)
    for seed in range(42,46):
        folder=REF/f'seed_{seed}'
        for p in [folder/'view_hashes.json']+[folder/'items'/f'{model}_A_aug_CMA_reset.json' for model in bn.MODELS]:
            result[str(p.relative_to(ROOT))]=digest(p)
    return result


def optimize(geometry,target,uniform,tilt,measure,progress=lambda trace:None):
    """Reguła E024; measure widzi tylko szerokości, nie wyniki C/dev."""
    cache={};trace=[]
    def evaluate(widths,origin):
        key=tuple(widths)
        if key in cache:
            return cache[key]
        assert .999*target<=geometry.count(widths)<=target
        measured=measure(widths)
        entry=dict(measured,index=len(trace),origin=origin,widths=list(widths),params=geometry.count(widths))
        cache[key]=entry;trace.append(entry);progress(trace)
        return entry
    u=evaluate(uniform,'UNIFORM');t=evaluate(tilt,'TILT')
    incumbent=min((u,t),key=lambda r:r['kl_B'])
    trajectory=[incumbent['index']]
    for round_index in range(2):
        candidates=[]
        for widths in geometry.proposals(incumbent['widths'],target,round_index):
            if tuple(widths) in cache:
                continue
            candidates.append(evaluate(widths,f'round{round_index+1}'))
            if len(candidates)==24:
                break
        best=min([incumbent]+candidates,key=lambda r:r['kl_B'])
        if incumbent['kl_B']-best['kl_B']<=1e-6:
            break
        incumbent=best;trajectory.append(incumbent['index'])
    assert len(trace)<=50
    return incumbent,trajectory,trace
