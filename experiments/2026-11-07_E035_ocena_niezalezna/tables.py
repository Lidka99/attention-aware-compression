"""E035: formatowanie ukończonej analizy do tabel; wejście run, wyjście stdout.
Bez zmiany ocen i progów; pełny protokół OPIS.md.
"""
import json
import sys
from pathlib import Path

a=json.loads((Path(sys.argv[1])/'analysis/analysis.json').read_text())
test=[r for r in a['contrasts'] if r['phase']=='test']
print('CMA: para | budżet | dev delta | U | T | test delta ±SD | teacher | T loss')
for r in test:
    if r['mode']!='CMA':continue
    dev=next(d for d in a['contrasts'] if d['phase']=='dev' and all(d[k]==r[k] for k in ('pair','tag','mode')))
    print(f"| {r['pair']} | {r['tag'][1:]} | {dev['mean']:+.4f} | {r['absolute']['U']['mean']:.4f} | {r['absolute']['T']['mean']:.4f} | {r['mean']:+.4f} ±{r['sd']:.4f} | {r['teacher']['mean']:.4f} | {r['teacher_loss']['T']['mean']:.4f} |")
print('SECONDARY: para | budżet | repair mean ±SD | positive | PRE mean | CMA NLL U/T | repair NLL delta')
for r in test:
    if r['mode']!='UP_REPAIR':continue
    pre=next(d for d in test if d['mode']=='PRE' and all(d[k]==r[k] for k in ('pair','tag')))
    nll={k:sum(x['nll'] for x in a['rows'] if x['phase']=='test' and x['mode']=='CMA' and x['profile']==k and x['pair']==r['pair'] and x['tag']==r['tag'])/4 for k in ('U','T')}
    print(f"| {r['pair']} | {r['tag'][1:]} | {r['mean']:+.4f} ±{r['sd']:.4f} | {r['positive']}/4 | {pre['mean']:+.4f} | {nll['U']:.6f}/{nll['T']:.6f} | {r['nll_delta']['mean']:+.6f} |")
print('PER SEED: para | budżet | CMA | repair')
for r in test:
    if r['mode']!='CMA':continue
    repair=next(d for d in test if d['mode']=='UP_REPAIR' and all(d[k]==r[k] for k in ('pair','tag')))
    print('| '+r['pair']+' | '+r['tag'][1:]+' | '+' / '.join(f'{v:+.2f}' for v in r['per_seed'])+' | '+' / '.join(f'{v:+.2f}' for v in repair['per_seed'])+' |')
print('TEACHER: para | PRE/CMA/repair')
for pair in dict.fromkeys(r['pair'] for r in test):
    vals=[sum(c['top1']*100 for c in a['controls'] if c['phase']=='test' and c['pair']==pair and c['mode']==mode)/4 for mode in ('PRE','CMA','UP_REPAIR')]
    print(pair,vals)
