"""Offline integrity and numerical consistency checks for the research report."""
import json
import math
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from PIL import Image
from research_data import OUT, sha, dump

def validate():
    d=json.loads((OUT/'data/analysis.json').read_text());manifest=json.loads((OUT/'data/source_manifest.json').read_text())
    checks=[]
    def check(test,label):
        if not test:raise AssertionError(label)
        checks.append(label)
    expected={'O-A':(12,40,1,0,20),'O-Q':(10.4,31.64,0,1,16),'V-DH':(13.22,39.26,0,3,20),'V-DL':(4,18.84,0,2,10),'V-QH':(4,24.88,0,2,13),'V-QL':(4,16.8,0,1,9),'V-A':(12,32.94,0,3,17)}
    token_expected={'O-Q':565031,'V-DH':255034,'V-DL':87968,'V-QH':132070,'V-QL':55712}
    for r in d['runs']:
        m=r['metric'];ident=r['id'];ref=expected[ident]
        check(all(math.isclose(x,y) for x,y in zip([m['score_observed'],m['end_seconds'],m['hp_a'],m['hp_b'],m['decisions']],ref)),ident+' endpoint matches archive')
        check((m['final_score'] is not None)==m['completed'],ident+' incomplete final_score is null')
        check(math.isclose(sum(e['points'] for e in d['episodes'] if e['run']==ident),m['score_observed'],abs_tol=1e-7),ident+' target-instance points sum to score')
        check(math.isclose(sum(s['wait_seconds'] for s in r['steps'])/len(r['steps']),m['wait_mean']),ident+' wait mean recomputation')
        check(m['stationary_seen_seconds']<=m['human_sees_a_seconds']<=m['end_seconds'],ident+' exposure subsets bounded')
        if r['usage']:
            u=r['usage'];check(u['prompt_tokens']+u['completion_tokens']==u['total_tokens']==token_expected[ident],ident+' provider token accounting')
            check((u.get('reasoning_tokens') or 0)<=u['completion_tokens'],ident+' reasoning not double counted')
        for s in r['steps']:
            check(s['time']<=s['end']<=m['end_seconds'],ident+f' step {s["decision"]} within observed trajectory')
            check(all(f['simulation_seconds']<=s['time']+1e-8 for f in s['frames']),ident+f' step {s["decision"]} no future frame')
            if r['cohort']=='vision':
                check(set(s['raw'])=={'observation_id','skill','params','assessment'},ident+f' step {s["decision"]} raw schema')
                allowed={'observation_id','simulation_seconds','self_state','public_state','frames','perception','skills'}
                if s['decision']==0:allowed.add('opponent_initial_xy')
                check(set(s['observation'])==allowed,ident+f' step {s["decision"]} public-only observation schema')
                if s['accepted_utc']:
                    delta=(datetime.fromisoformat(s['accepted_utc'])-datetime.fromisoformat(s['observation_ready_utc'])).total_seconds()
                    check(abs(delta-s['wait_seconds']-s['compile_seconds'])<.025,ident+f' step {s["decision"]} UTC includes compile')
            if s['compiled_state']=='UNSAFE_ROUTE':check(s['compiled_action']['skill']=='HOLD',ident+f' rejected step {s["decision"]} really HOLD')
        for a in r['annotations']:
            check(math.isclose(a['error_m'],math.hypot(a['projected_x']-a['truth_x'],a['projected_y']-a['truth_y'])),ident+' projection-error arithmetic')
    decision_count=sum(len(r['steps']) for r in d['runs'])
    check(decision_count==105,'105 accepted decisions, not 105 independent matches')
    check(len(d['annotations'])==13,'13 submitted annotations')
    check(all(r['metric']['completed'] for r in d['runs']),'all seven primary runs reached true terminal states')
    rerun=next(r for r in d['runs'] if r['id']=='V-DH');old=next(h for h in d['history'] if h['id']=='DH16')
    check(Path(rerun['root']).name=='20261022-095614-313039','primary V-DH is the new independent complete run')
    check(len(d['history'])==7 and Path(old['path']).name=='20261021-200759-757048','old V-DH retained separately among seven history records')
    check(rerun['failed_response'] is None and old['failed_response']['finish_reason']=='length','historical failure is not assigned to new V-DH')
    check(rerun['profile']['max_tokens']==393216 and rerun['profile']['wall_timeout_seconds'] is None and rerun['profile']['launch_timeout_seconds'] is None,'new max output and timeout policy retained')
    responses=[json.loads(p.read_text()) for p in sorted((Path(rerun['root'])/'api_calls').glob('*_response.json'))]
    check(len(responses)==20 and all(r['finish_reason']=='stop' for r in responses),'all twenty new max requests completed normally')
    check([r['usage']['completion_tokens'] for r in responses if r['usage']['completion_tokens']>16384]==[17520,26319],'two successful responses exceed previous cap')
    check(math.isclose(old['endpoint']['score'],10.56) and not old['endpoint']['finished'],'old observed score remains nonterminal')
    check(len(rerun['overhead'])==21 and rerun['overhead'][-1]['time']==39.26,'new max third-person frames include every decision and terminal state')
    check(all('20261021-200759-757048' not in f['source'] for f in manifest['assets'] if f['file'].startswith('assets/V-DH/')),'new max assets never point at the old run')
    check(all(a['visual_identical'] for a in d['audit']),'five shared file types hash-identical across five visual runs')
    for f in manifest['sources']:check(sha(Path(f['path']))==f['sha256'],'original unchanged: '+f['path'])
    for f in manifest['assets']:
        p=OUT/f['file'];check(sha(p)==f['sha256'],'asset hash: '+f['file'])
        if p.suffix=='.png':
            with Image.open(p) as im:im.verify()
        if f['kind']=='original online RGB; unchanged bytes':
            with Image.open(p) as im:check(im.size==(768,432),'native onboard size: '+f['file'])
    text=(OUT/'index.html').read_text()
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            for key in ['src','href']:
                if key in a:self.links.append(a[key])
    parser=Links();parser.feed(text)
    check(len(parser.ids)==len(set(parser.ids)),'unique DOM IDs')
    for link in parser.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:continue
        if u.path and u.path!='data/validation.json':check((OUT/unquote(u.path)).exists(),'local link resolves: '+link)
        elif u.fragment:check(u.fragment in parser.ids,'section resolves: '+link)
    for p in [OUT/'index.html',*list((OUT/'archive').rglob('*.txt')),*list((OUT/'archive').rglob('*.json'))]:
        check(re.search(r'\bsk-[A-Za-z0-9_.-]{20,}',p.read_text()) is None,'no API credential pattern: '+str(p.relative_to(OUT)))
    figures=list((OUT/'figures').glob('*.svg'));check(len(figures)==6,'six editable vector figures')
    for p in figures:
        check('<text' in p.read_text(),'editable SVG text: '+p.name)
        check(p.with_suffix('.pdf').exists() and p.with_suffix('.png').exists() and p.with_suffix('.json').exists(),'four exports: '+p.stem)
        with Image.open(p.with_suffix('.png')) as im:check(abs(im.info['dpi'][0]-600)<1,'600 dpi metadata: '+p.stem)
    result=dict(status='passed',checked='2026-10-22',checks_count=len(checks),source_files=len(manifest['sources']),asset_files=len(manifest['assets']),runs=len(d['runs']),completed=sum(r['metric']['completed'] for r in d['runs']),interrupted=sum(not r['metric']['completed'] for r in d['runs']),accepted_decisions=decision_count,annotations=len(d['annotations']),history_records=len(d['history']),checks=checks)
    dump(OUT/'data/validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))

if __name__=='__main__':validate()
