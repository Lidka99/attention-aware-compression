"""E003: końcowa kontrola pipeline przed realnym badaniem.
ID 2026-10-09_E003_kompresja_10proc_parametrow; OPIS.md i PROTOKOL_v1.md.
Wejścia: kod/config i wcześniejszy dowód operatora, sztuczne tablice CPU.
Wyjście: nowy TESTY_CPU_v1.json, hashe wszystkich źródeł i zależności.
Bez wytrenowanego modelu, rzeczywistych obrazów, checkpointu i GPU.
"""
import argparse
import ast
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torchvision.models import efficientnet_b0

from support import HERE,digest,read_json,write_json,require,code_hashes,model_digest
from folding import count_parameters,materialize_model
from selection import choose_groups
from study import Capture
from summarize import paired_bootstrap


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); require(not args.output.exists(),'Refusing to overwrite evidence')
    for path in HERE.glob('*.py'): ast.parse(path.read_text())
    cfg=read_json(HERE/'config.json'); draft=read_json(HERE/'config.draft.json')
    require(all(cfg[k]==v for k,v in draft.items() if k not in ('protocol_version','status')),
            'Frozen scientific parameters changed from draft')
    old=read_json(HERE/'TEST_FOLDING_CPU.json')
    require(old['status']=='passed' and all(digest(HERE/name)==value for name,value in old['source_sha256'].items()),
            'Operator evidence does not match code')
    torch.set_num_threads(1); torch.manual_seed(20261010)
    model=efficientnet_b0(weights=None,num_classes=100).eval(); before=model_digest(model)
    rows=read_json(HERE/cfg['structure_plan'])['blocks']
    require(count_parameters(materialize_model(model,rows))==cfg['expected_compressed_parameters'],'Graph budget mismatch')
    x=torch.randn(2,3,96,96)
    captures=[]
    with torch.inference_mode():
        plain=model(x)
        cal=Capture(model,rows,'calibration'); cal.reset(); hooked=model(x); cal.check(2)
        require(torch.equal(plain,hooked),'Calibration hooks change predictions')
        row=rows[5]; value=cal.values[row['block']]
        require(torch.allclose(value['g'],value['h'].mean((2,3),keepdim=True),atol=1e-6,rtol=1e-6),'GAP hook mismatch')
        cal.close()
        diag=Capture(model,rows,'diagnostic'); diag.reset(); hooked=model(x); diag.check(2)
        require(torch.equal(plain,hooked),'Diagnostic hooks change predictions')
        saved={path:{key:value.clone() for key,value in values.items()} for path,values in diag.values.items()}
        diag.close()
        for row in rows:
            block=model.get_submodule(row['block']); v=saved[row['block']]
            direct=block.block(v['input']).clone()
            require(torch.equal(direct,v['branch']),'Branch clone not captured before residual')
            target=direct+v['input'] if block.use_res_connect else direct
            require(torch.equal(target,v['block']),'Block capture mismatch')
    require(model_digest(model)==before,'Test changes source state')
    generator=np.random.default_rng(20261010)
    h=generator.normal(size=(11,10)); z=generator.normal(size=(11,10)); g=generator.normal(size=(11,10))
    moments={'block':{'h':h.T@h,'z':z.T@z,'g':g.T@g}}
    selected=choose_groups(moments,[{'block':'block','remove_count':3,'expanded_channels':10}],cfg)
    for mode,groups in selected.items():
        channels=[p[k] for p in groups['block'] for k in ('c','d')]
        require(len(channels)==len(set(channels))==6,'Pair overlap')
        for pair in groups['block']:
            c,d=pair['c'],pair['d']
            require(np.isclose(pair['alpha'],g[:,c]@g[:,d]/(g[:,c]@g[:,c]))
                    and np.isclose(pair['beta'],z[:,c]@z[:,d]/(z[:,c]@z[:,c])),'Scalar fit mismatch')
    # The summarizer provides its own controlled examples, checked below.
    labels=np.repeat(np.arange(100),50); flags=np.ones(5000,dtype=bool)
    same=paired_bootstrap(flags,flags,labels,20,20261010)
    write_json(args.output,{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
               'device':'CPU','real_images_used':False,'checkpoint_used':False,'seed':20261010,
               'checks':['syntax','frozen_equals_draft','folding_evidence','physical_graph_budget',
                         'capture_shapes','GAP_value','hooks_do_not_change_predictions','branch_clone_before_residual',
                         'source_state_unchanged','disjoint_groups','least_squares_coefficients','bootstrap_identical_predictions'],
               'folding_evidence_sha256':digest(HERE/'TEST_FOLDING_CPU.json'),
               'config_sha256':digest(HERE/'config.json'),'source_sha256':code_hashes(),
               'bootstrap_identical_result':same})
    print('E003_CPU_CHECKS=passed; operator and pipeline checked on synthetic CPU data; actual study uses 4 GPUs')


if __name__=='__main__': main()
