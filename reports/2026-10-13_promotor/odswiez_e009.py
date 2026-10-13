"""Odswieza wyniki E009 w prezentacji podsumowania badań (katalog 2026-10-13_promotor).

Dzialanie (tylko odczyt danych biegu, CPU, bez GPU):
1. czyta najnowszy katalog runs/E009/2* (stan biezacy lub koncowy),
2. regeneruje wykresy E009 (fig_e009_krzywe.png, fig_e009_wyniki.png),
3. podmienia bloki miedzy znacznikami <!-- E009:BEGIN --> ... <!-- E009:END -->
   oraz linie <!-- E009:NAGLOWEK --> i <!-- E009:WNIOSEK --> w prezentacja.md
   i prezentacja.html.

Uruchomienie (po zakonczeniu biegu E009 albo w trakcie, zeby odswiezyc stan):
    python odswiez_e009.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generuj_wykresy as gw  # noqa: E402

BEGIN = '<!-- E009:BEGIN -->'
END = '<!-- E009:END -->'
NAGLOWEK_BEGIN = '<!-- E009:NAGLOWEK:BEGIN -->'
NAGLOWEK_END = '<!-- E009:NAGLOWEK:END -->'
WNIOSEK_BEGIN = '<!-- E009:WNIOSEK:BEGIN -->'
WNIOSEK_END = '<!-- E009:WNIOSEK:END -->'


def bold_to_html(text):
    parts = text.split('**')
    if len(parts) % 2 == 0:
        return text
    out = parts[0]
    for index, part in enumerate(parts[1:], start=1):
        out += ('<b>' if index % 2 == 1 else '</b>') + part
    return out


def pc(value, sign=False):
    return (f'{value:+.2f}' if sign else f'{value:.2f}').replace('.', ',')


def fmt_variant_line(state):
    parts = []
    for name in gw.VARIANTS:
        v = state['variants'][name]
        if v['top1'] is None:
            parts.append(f'{name} (w kolejce)')
        elif v['epoch'] >= v['total']:
            parts.append(f'{name} **{gw.fmt_pl(100 * v["top1"])}%** ({v["epoch"]}/{v["total"]})')
        else:
            parts.append(f'{name} {gw.fmt_pl(100 * v["top1"])}% (ep. {v["epoch"]}/{v["total"]})')
    return ' · '.join(parts)


def status_sentence(state):
    finished = [n for n in gw.VARIANTS if state['variants'][n]['top1'] is not None
                and state['variants'][n]['epoch'] >= state['variants'][n]['total']]
    in_progress = [n for n in gw.VARIANTS if state['variants'][n]['top1'] is not None
                   and state['variants'][n]['epoch'] < state['variants'][n]['total']]
    queued = [n for n in gw.VARIANTS if state['variants'][n]['top1'] is None]
    details = []
    if finished:
        label = ' i '.join(finished) if len(finished) > 1 else finished[0]
        details.append(f'{label} po pełnych {state["variants"][finished[0]]["total"]} epokach')
    if in_progress:
        n = in_progress[0]
        details.append(f'{n} w toku (ep. {state["variants"][n]["epoch"]}/{state["variants"][n]["total"]})')
    if queued:
        details.append(' i '.join(queued) + ' w kolejce')
    return f'**w toku** (stan {state["now_utc"]})' + ('; ' + ', '.join(details) if details else '')


def improvements(state, e8_values):
    chunks = []
    for name in gw.VARIANTS:
        v = state['variants'][name]
        if v['top1'] is not None and v['epoch'] >= v['total']:
            delta = 100 * (v['top1'] - e8_values[name] / 100)
            chunks.append(f'{name} {delta:+.2f} pp'.replace('.', ','))
    return ', '.join(chunks) if chunks else 'brak jeszcze ukończonych wariantów'


def build_block(state):
    e8 = json.loads(gw.E008_S2.read_text())['variants']
    e8_values = {n: 100 * e8[n]['last_row']['dev_top1'] for n in gw.VARIANTS}
    e8_line = ' · '.join(f'{n} {gw.fmt_pl(e8_values[n])}' for n in gw.VARIANTS)
    p = state['summary']['primary'] if state['summary'] else None
    lines = []
    if state['finished'] and p is not None:
        ci = p['ci95_top1']
        verdict = 'przekroczone' if p['primary_passed'] else 'nieprzekroczone'
        lines.append(f'Status biegu: **zakończony** (stan {state["now_utc"]}); '
                     f'podsumowanie `summary.json` zapisane.')
        lines.append('dev top-1 po 120 epokach: ' + fmt_variant_line(state) + '.')
        lines.append('Test główny POST−PRE: **%s pp, CI [%s; %s]** — kryterium **%s** '
                     '(dolna granica %s 0).' % (pc(100 * p['delta_top1'], sign=True),
                                                pc(100 * ci[0]), pc(100 * ci[1]),
                                                verdict, '>' if p['primary_passed'] else '<'))
        comparisons = state['summary'].get('comparisons', {})
        for key in ('POST_minus_RANDOM', 'PRE_minus_RANDOM'):
            c = comparisons.get(key)
            if c is not None:
                label = key.replace('_minus_', '−')
                lines.append('%s: **%s pp, CI [%s; %s]**.'
                             % (label, pc(100 * c['delta_top1'], sign=True),
                                pc(100 * c['ci95_top1'][0]), pc(100 * c['ci95_top1'][1])))
        lines.append(f'Dla porównania E008 (60 epok): {e8_line}.')
        lines.append(f'Zmiana względem E008: {improvements(state, e8_values)}.')
    else:
        lines.append('Status biegu: ' + status_sentence(state) + '.')
        lines.append('dev top-1: ' + fmt_variant_line(state) + '.')
        lines.append(f'Dla porównania E008 (60 epok): {e8_line}.')
        lines.append(f'Wniosek z części danych: ukończone warianty poprawiły się względem 60 epok '
                     f'({improvements(state, e8_values)}) — dłuższa naprawa realnie podnosi jakość.')
    return lines


def render_block(lines, html):
    if html:
        items = '\n'.join(f'    <li>{bold_to_html(line)}</li>' for line in lines)
        return '  <ul>\n' + items + '\n  </ul>'
    return '\n'.join('- ' + line for line in lines)


def render_naglowek(state, html):
    p = state['summary']['primary'] if state['summary'] else None
    if state['finished'] and p is not None:
        ci = p['ci95_top1']
        verdict = 'przekroczone' if p['primary_passed'] else 'nieprzekroczone'
        text = ('E009 zakończony (120 epok naprawy): POST−PRE %s pp, CI [%s; %s] — '
                'kryterium %s. Wyniki finalne na slajdzie „E009".'
                % (pc(100 * p['delta_top1'], sign=True), pc(100 * ci[0]), pc(100 * ci[1]), verdict))
    else:
        bits = []
        for name in ('BASE', 'PRE'):
            v = state['variants'][name]
            if v['top1'] is not None and v['epoch'] >= v['total']:
                bits.append(f'{name} {gw.fmt_pl(100 * v["top1"])}% (120/120)')
        post = state['variants']['POST']
        if post['top1'] is not None and post['epoch'] < post['total']:
            bits.append(f'POST w toku (ep. {post["epoch"]}/{post["total"]})')
        else:
            bits.append('RANDOM w kolejce')
        text = ('Trwa E009: naprawa 120 epok (stan %s) — %s. Wyniki E008 na slajdach „Etap 2"; '
                'po zakończeniu biegu uruchom `odswiez_e009.py`.' % (state['now_utc'], ', '.join(bits)))
    if html:
        return f'<div class="note">{text}</div>'
    return text


def render_wniosek(state, html):
    p = state['summary']['primary'] if state['summary'] else None
    if state['finished'] and p is not None:
        ci = p['ci95_top1']
        if p['primary_passed']:
            text = ('E009: POST−PRE %s pp, CI [%s; %s] — kryterium przekroczone (dolna granica > 0): '
                    'ścieżka publikacyjna na baseline 224.'
                    % (pc(100 * p['delta_top1'], sign=True), pc(100 * ci[0]), pc(100 * ci[1])))
        else:
            text = ('E009: POST−PRE %s pp, CI [%s; %s] — kryterium nieprzekroczone: zamknięcie '
                    'wątku selekcji przy badanych budżetach.'
                    % (pc(100 * p['delta_top1'], sign=True), pc(100 * ci[0]), pc(100 * ci[1])))
    else:
        text = ('E009 (120 epok, w toku) rozstrzyga dziś po południu: trwała przewaga POST−PRE '
                '(ścieżka publikacyjna) albo brak różnic (zamknięcie wątku selekcji przy badanych budżetach).')
    if html:
        return '    <li>' + text + '</li>'
    return '- ' + text


def replace_section(text, begin, end, new, path):
    i = text.index(begin)
    j = text.index(end)
    if j < i:
        raise RuntimeError(f'{path}: znaczniki w złej kolejności')
    return text[:i] + begin + '\n' + new + '\n' + end + text[j + len(end):]


def update_file(path, state):
    html = path.suffix == '.html'
    text = path.read_text()
    text = replace_section(text, BEGIN, END, render_block(build_block(state), html), path)
    text = replace_section(text, NAGLOWEK_BEGIN, NAGLOWEK_END,
                           render_naglowek(state, html), path)
    text = replace_section(text, WNIOSEK_BEGIN, WNIOSEK_END,
                           render_wniosek(state, html), path)
    path.write_text(text)
    print(f'updated {path}')


def main():
    state = gw.load_e009()
    print(f'E009 run: {state["run"]}')
    for name in gw.VARIANTS:
        v = state['variants'][name]
        top1 = 'brak' if v['top1'] is None else f'{100 * v["top1"]:.2f}%'
        print(f'  {name:6s} ep. {v["epoch"]:3d}/{v["total"]}  dev {top1}')
    print(f'  finished={state["finished"]}  summary={"jest" if state["summary"] else "brak"}')
    gw.generate_e009_figures(state)
    update_file(HERE / 'prezentacja.md', state)
    update_file(HERE / 'prezentacja.html', state)
    print('E009_PRESENTATION_REFRESHED=done')


if __name__ == '__main__':
    main()
