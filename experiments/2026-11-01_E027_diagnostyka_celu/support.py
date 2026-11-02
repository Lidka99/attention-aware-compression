"""E027: wejścia i trzy bezetykietowe kryteria pełnej sieci; OPIS.md.
Wejście: zweryfikowane E024/E025/E026; wyjście: helpery search i pomiaru.
Bez treningu, bez modyfikacji historii.
"""
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E026=ROOT/'experiments/2026-11-01_E026_alokacja_poprawione_bn'
E026_RUN=ROOT/'runs/E026/20261101T133903Z_bbd75e05'
spec=importlib.util.spec_from_file_location('_e027_e026',E026/'support.py')
e26=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=e26
spec.loader.exec_module(e26)
bn=e26.bn
write=e26.write
digest=e26.digest

OBJECTIVES=('disagreement','kl','teacher_ce')


def sources():
    result=e26.sources()
    manifest=bn.read(E026_RUN/'manifest.json')
    assert manifest['status']=='completed' and manifest['exit_codes']==[0]*4
    assert manifest['source_hashes']=={p.name:digest(p) for p in E026.glob('*.py')}
    for seed in range(42,46):
        for p in (E026_RUN/f'seed_{seed}'/name for name in ('trace.json','selection.json','summary.json')):
            result[str(p.relative_to(ROOT))]=digest(p)
    result[str(E026_RUN/'manifest.json')]=digest(E026_RUN/'manifest.json')
    return result


def measure_values(teacher_logits,candidate_logits):
    """Trzy zamrożone miary zgodności; wejście: logity bez etykiet."""
    import torch
    from torch.nn import functional as F
    assert teacher_logits.shape==candidate_logits.shape
    t_label=teacher_logits.argmax(1)
    return dict(
        kl=float(F.kl_div(F.log_softmax(candidate_logits,dim=1),F.softmax(teacher_logits,dim=1),reduction='batchmean')),
        disagreement=float((candidate_logits.argmax(1)!=t_label).double().mean()),
        teacher_ce=float(F.cross_entropy(candidate_logits,t_label)))


def optimize(geometry,target,uniform,tilt,measure,key='disagreement',progress=lambda trace:None):
    """Ta sama reguła co E024; wybór według jednego z trzech kluczy."""
    assert key in OBJECTIVES
    cache={};trace=[]
    def evaluate(widths,origin):
        cache_key=tuple(widths)
        if cache_key in cache:
            return cache[cache_key]
        assert .999*target<=geometry.count(widths)<=target
        values=measure(widths)
        assert all(k in values and values[k]==values[k] for k in OBJECTIVES)
        entry=dict(values,index=len(trace),origin=origin,widths=list(widths),params=geometry.count(widths))
        cache[cache_key]=entry;trace.append(entry);progress(trace)
        return entry
    u=evaluate(uniform,'UNIFORM');t=evaluate(tilt,'TILT')
    incumbent=min((u,t),key=lambda r:r[key])
    trajectory=[incumbent['index']]
    for round_index in range(2):
        candidates=[]
        for widths in geometry.proposals(incumbent['widths'],target,round_index):
            if tuple(widths) in cache:
                continue
            candidates.append(evaluate(widths,f'round{round_index+1}'))
            if len(candidates)==24:
                break
        best=min([incumbent]+candidates,key=lambda r:r[key])
        if incumbent[key]-best[key]<=1e-6:
            break
        incumbent=best;trajectory.append(incumbent['index'])
    assert len(trace)<=50
    return incumbent,trajectory,trace
