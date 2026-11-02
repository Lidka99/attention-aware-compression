"""E028: transfer zamrożonej reguły E027 na CIFAR-10; OPIS.md.
Wejście: niezależne dane CIFAR-10 i checkpointy E011; wyjście: helpery.
Reguła wyszukiwania i BN są ponownie użyte bez zmian naukowych.
"""
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E027=ROOT/'experiments/2026-11-01_E027_diagnostyka_celu'
_spec=importlib.util.spec_from_file_location('_e028_e027',E027/'support.py')
_e27=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_e27
_spec.loader.exec_module(_e27)
e27=_e27
bn=e27.bn
optimize=e27.optimize
measure_values=e27.measure_values
write=e27.write
digest=e27.digest
OBJECTIVES=e27.OBJECTIVES

from data10 import (HERE as DATA_HERE,Images,architecture,source_model,indices,augment_uint8,
                    normalize,tensor_hash,HASHES,BASELINE,SPLIT,SPLIT_SHA256,DATA)


def sources():
    result={}
    for seed in range(42,46):
        p=ROOT/f'runs/E011/20261024T192205Z_018f3f2_seeds42-45_b2da2012/seed_{seed}/last.pt'
        assert digest(p)==HASHES[seed]
        result[str(p.relative_to(ROOT))]=HASHES[seed]
    assert digest(SPLIT)==SPLIT_SHA256
    result[str(SPLIT.relative_to(ROOT))]=SPLIT_SHA256
    for index in range(1,6):
        p=DATA/f'data_batch_{index}'
        result[str(p.relative_to(ROOT))]=digest(p)
    for p in (E027/'OPIS.md',E027/'support.py',E027/'worker.py'):
        result[str(p.relative_to(ROOT))]=digest(p)
    return result
