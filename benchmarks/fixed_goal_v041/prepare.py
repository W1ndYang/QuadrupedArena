"""Map-based pre-registration; no model calls and no outcome-based scene selection."""
import copy, json, math, tempfile
import numpy as np
from ds_fixed_goal import R,scenario,WAYPOINTS,read,dump,utc
from fixed_goal_probe import boxes,clear_ray,sha
from trial_motion import Planner,path_length,clearance
from peek_braking_fix import SequenceProbe
from sequence_skills import FEATURES
from v04_model import prompt

def sampled(start,route,spacing=.05):
    pts=[];at=np.array(start,float)
    for q in route:
        pts.extend((at*(1-u)+q*u).tolist() for u in np.linspace(0,1,max(2,math.ceil(math.dist(at,q)/spacing)+1)))
        at=q
    return pts

class OpeningProbe(SequenceProbe):
    """Scripted first 2s route probe only; never a model outcome."""
    def __init__(self,*args,destination,**kwargs):
        super().__init__(*args,**kwargs);self.destination=destination
    def execute(self,i,command,t):
        if i==0:command=dict(mode='ADVANCE',posture='upright',target=self.destination)
        return super().execute(i,command,t)

def plan(p):
    return dict(observation_id=p['observation_id'],skill='PLAN',params=dict(horizon_seconds=8,steps=[dict(skill='MOVE',params=dict(target='SPAWN',posture='upright',fire='off'),on_contact='continue')]),assessment={k:'几何起步检查，不是模型决策。' for k in ('observation_evidence','state_evidence','decision_summary','uncertainty')})

def main():
    cases=read(R/'cases.json');specs=read(R/'layouts.json');checks=[]
    (R/'prompts').mkdir(exist_ok=True)
    for case in cases:
        name=case['layout'];spec=specs[name];base,rules=scenario(name)
        rules.update(version='fixed-goal-v0.4.1-peek-braking',decision_period_seconds=None,decision_policy='event_driven_plan_or_peek_completion')
        folder=R/f"run_{case['run']}";folder.mkdir(exist_ok=True)
        assert not (folder/'api_calls').exists(),'Do not rewrite preregistration after paid calls'
        dump(folder/'case.json',case);dump(folder/'trial_config.json',dict(base=base,rules=rules,waypoints=copy.deepcopy(WAYPOINTS),features=FEATURES))
        (R/'prompts'/f'{name}.txt').write_text(prompt(base,rules)+'\n')
        p=Planner(base,base['arena']['covers']);walls=boxes(base['arena']['covers']);a=base['start_a'];b=base['start_b'];g=spec['goal']
        los=lambda x,y:clear_ray((*x,.4),(*y,1.05),walls)
        assert all(p.free(x) for x in [a,b,g,*spec['patrol'],*spec['escape_points']]),name
        assert math.dist(a,b)>=10 and math.dist(a,g)>=10 and not los(a,b),name
        blocked=[c['name'] for c in base['arena']['covers'] if not clear_ray((*a,.4),(*b,1.05),boxes([c]))]
        patrol_positions=sampled(spec['patrol'][0],p.route(*spec['patrol']))
        assert all(not los(a,q) for q in patrol_positions),('spawn exposed along patrol',name)
        exits=[]
        for j,q in enumerate(spec['escape_points']):
            path=p.route(a,q);assert path is not None
            pts=sampled(a,path)
            assert all(p.segment(x,y) for x,y in zip([a]+path,path))
            prefix=0.;prev=pts[0]
            for v in pts[1:]:
                if los(v,b):break
                prefix+=math.dist(prev,v);prev=v
            r2=copy.deepcopy(rules);r2['duration']=2.
            # Exact 50 Hz engine, stopped at 2 seconds.
            with tempfile.TemporaryDirectory() as td:
                probe=OpeningProbe(base,r2,plan,Path(td),destination=q)
                result=probe.run(f'opening_{name}_{j}')
            assert result['summary']['seconds']<=2.02
            tag=any(e['type']=='tag' and 1 in e['actors'] for e in result['events'])
            assert not tag,(name,j,'opening tag')
            exits.append(dict(target=q,path=[list(map(float,v)) for v in path],length_m=path_length(a,path),
                initially_occluded_prefix_m=prefix,first_two_seconds_human_tag=tag,
                first_two_seconds_max_human_lock=max(s['aim'][1] for s in result['states']),
                first_two_seconds_distance_m=result['summary']['distance_m'][0]))
        assert all(e['initially_occluded_prefix_m']>=1.2 for e in exits),(name,'too little covered departure')
        route=p.route(a,g);assert route is not None
        checks.append(dict(layout=name,label=case['label'],start_a=a,start_b=b,goal=g,initial_separation_m=math.dist(a,b),
            initial_goal_distance_m=math.dist(a,g),goal_route_length_m=path_length(a,route),initial_los=False,
            blocking_walls=blocked,spawn_wall_clearance_m=min(clearance(a,c) for c in base['arena']['covers']),
            spawn_occluded_against_sampled_patrol=True,patrol_sample_count=len(patrol_positions),departures=exits,
            limitation='两条预设起步路径各执行2秒，另检查静态巡逻路径采样；不保证任意模型路线安全，也不是整局胜负预筛选。'))
    dump(R/'layout_checks.json',checks)
    frozen=[*[R/n for n in ('run_batch.py','ds_fixed_goal.py','qwen_fixed_goal.py','sequence_skills.py','peek_geometry.py','peek_braking_fix.py','run_goal_only.py','run_study.py','v04_model.py','prepare.py')],R/'layouts.json',R/'cases.json',R/'waypoints.json',R/'api_profile.json',R/'design_basis.json',*R.glob('prompts/*.txt'),*R.glob('run_*/trial_config.json'),*R.glob('source_snapshot/scripts/*.py'),*R.glob('source_snapshot/configs/*.json')]
    dump(R/'preregistration.json',dict(created_utc=utc(),source_sha256={str(p.relative_to(R)):sha(p) for p in frozen},
        hypothesis='DS LOW在有遮挡且具有两条出路的三种新布局中，能否获取信息后承诺并执行绕行路线。',
        changes=['Three new start/defender/goal/patrol layouts; SPAWN tracks each actual start.', 'Previously validated v0.4.1 brake-before-reverse patch only; no policy/prompt advice changes.'],
        trials=3,provider='deepseek_official',model='deepseek-flash',reasoning_effort='low',
        no_outcome_filter=True,no_old_match_history_to_model=True,no_scene_redesign_after_paid_start=True))
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':
    from pathlib import Path
    main()
