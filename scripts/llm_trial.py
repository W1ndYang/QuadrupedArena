"""Pauseable R0 trial: external high-level decisions, fixed motion primitives.

Private checkpoints contain RNG and are never included in observations. Only
load checkpoints created locally by this script; pickle is not an import format.
No future rollout or automated tactical selector is exposed to the planner.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import numpy as np
from adaptive_motion import load_config, HumanPolicy, ReversibleMover, safe_step, visible_observation
from tactical_logic import Mover, block
from trial_motion import Planner, clearance, sweep_clear, path_length
from trial_rules import Referee, covers_at, aim_valid

PROJECT=Path(__file__).resolve().parents[1]


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def public_events(events):
    return [{k:v for k,v in e.items() if k not in ('threshold','hazard')} for e in events]


class Match:
    def __init__(self,cfg):
        self.cfg=copy.deepcopy(cfg);self.ref=Referee(self.cfg);self.tick=0;self.t=0.
        self.a=ReversibleMover(cfg['start_a'],0,*[cfg[k][0] for k in ('max_speed','acceleration','yaw_rate')])
        self.b=Mover(cfg['start_b'],math.pi,*[cfg[k][1] for k in ('max_speed','acceleration','yaw_rate')])
        self.human=HumanPolicy(cfg);self.planner=Planner(cfg,covers_at(cfg,0));self.collapsed=False
        self.states=[];self.decisions=[];self.events=[];self.active=None;self.skill_status='IDLE'
        self.min_clear=99.;self.min_separation=99.;self.safety_stops=[0,0]
        self.observation_cursor=0;self.previous_human=None
        self.human.update(0,self.b,visible_observation(cfg,self.b.xy,self.b.yaw,self.a.xy,self.planner.covers,0),self.planner)
        self.ref.step(0,[self.a.xy,self.b.xy],[self.a.yaw,self.b.yaw],self.planner.covers,attack_enabled=[True,self.human.visible])
        self.record()

    def compile(self,action):
        if not isinstance(action,dict):raise ValueError('Action must be an object')
        allowed={'skill','goal','via','facing','reverse','attack','speed_limit','target_generation'}
        if set(action)-allowed:raise ValueError('Unknown action fields')
        skill=action.get('skill')
        if skill not in ('MOVE','HOLD','AIM','SCAN','STOP'):raise ValueError('Unknown skill')
        for key in ('reverse','attack'):
            if key in action and type(action[key]) is not bool:raise ValueError(key+' must be boolean')
        speed=float(action.get('speed_limit',self.cfg['max_speed'][0]))
        if not 0<speed<=self.cfg['max_speed'][0]:raise ValueError('Invalid speed')
        facing=action.get('facing','keep')
        if isinstance(facing,str):
            if facing not in ('keep','enemy'):raise ValueError('Invalid facing')
        elif not isinstance(facing,(int,float)) or not math.isfinite(facing):raise ValueError('Invalid facing')
        route=[];start=self.a.xy.copy()
        if skill=='MOVE':
            if 'goal' not in action:raise ValueError('MOVE requires goal')
            waypoints=action.get('via',[])+[action['goal']]
            if len(waypoints)>8:raise ValueError('Too many waypoints')
            for value in waypoints:
                point=np.asarray(value,dtype=float)
                if point.shape!=(2,) or not np.isfinite(point).all():raise ValueError('Invalid waypoint')
                segment=self.planner.route(start,point)
                if segment is None:raise ValueError('Goal not reachable with original clearance')
                route.extend(segment);start=point
        elif 'goal' in action or 'via' in action:raise ValueError('Only MOVE accepts waypoints')
        generation=action.get('target_generation')
        if generation is not None and generation!=self.ref.generation:raise ValueError('Expired target generation')
        return route

    def apply(self,action,route):
        self.active=copy.deepcopy(action);self.active['initial_yaw']=self.a.yaw
        self.a.navigate(route);self.a.vmax=float(action.get('speed_limit',self.cfg['max_speed'][0]))
        self.a.reverse_allowed=action.get('reverse',False)
        self.a.reverse_limit=self.cfg['policy']['reverse_speed_limit']
        self.skill_status='RUNNING'

    def advance(self,seconds=2.):
        for _ in range(round(seconds/self.cfg['dt'])):
            if self.ref.finished:break
            self.tick+=1;self.t=round(self.tick*self.cfg['dt'],8);dt=self.cfg['dt']
            changed=not self.collapsed and self.t>=self.cfg['collapse']['time']
            if changed:
                if not sweep_clear(self.cfg,[m.xy for i,m in enumerate((self.a,self.b)) if self.ref.hp[i]>0]):raise RuntimeError('Unsafe collapse')
                self.collapsed=True;self.planner=Planner(self.cfg,covers_at(self.cfg,self.t));self.b.route=[]
                previous=self.a.xy
                for dest in self.a.route:
                    if not self.planner.segment(previous,dest):
                        self.cancel('map_changed_path_invalid');break
                    previous=dest
            if self.active.get('target_generation',self.ref.generation)!=self.ref.generation:self.cancel('target_expired')
            obs=visible_observation(self.cfg,self.b.xy,self.b.yaw,self.a.xy,self.planner.covers,self.t) if min(self.ref.hp)>0 else None
            self.human.update(self.t,self.b,obs,self.planner,self.ref.cooldown_until[1],self.ref.hp[1]>0)
            if self.human.state!=self.previous_human:
                self.events.append(dict(time=self.t,type='human_decision',state=self.human.state));self.previous_human=self.human.state
            facing=self.active.get('facing','keep');look=self.active['initial_yaw']
            if isinstance(facing,(int,float)):look=float(facing)
            if facing=='enemy' or self.active['skill']=='AIM':look=math.atan2(*(self.b.xy-self.a.xy)[::-1])
            if self.active['skill']=='SCAN':look=self.a.yaw+.6*dt
            # Same collapse-step motion pause as the existing baseline.
            if not changed:
                for i,(m,heading) in enumerate(((self.a,look),(self.b,self.human.look))):
                    if self.ref.hp[i]>0:
                        unsafe=safe_step(m,dt,heading,self.cfg,self.planner,(self.b,self.a)[i].xy,self.ref.hp[1-i]>0)
                        self.safety_stops[i]+=int(unsafe)
                        if i==0 and unsafe:self.cancel('safety_stop')
            self.ref.step(self.t,[self.a.xy,self.b.xy],[self.a.yaw,self.b.yaw],self.planner.covers,changed,
                          attack_enabled=[self.active.get('attack',True),self.human.visible])
            if not self.a.route and self.active['skill']=='MOVE':self.skill_status='ARRIVED'
            self.record()

    def cancel(self,reason):
        if self.skill_status==reason:return
        self.a.route=[];self.skill_status=reason
        self.active.update(skill='STOP',facing='keep',initial_yaw=self.a.yaw)
        self.active.pop('target_generation',None)
        self.events.append(dict(time=self.t,type='skill_cancelled',reason=reason))

    def record(self):
        for i,m in enumerate((self.a,self.b)):
            if self.ref.hp[i]>0:self.min_clear=min(self.min_clear,*[clearance(m.xy,c) for c in self.planner.covers])
        if min(self.ref.hp)>0:self.min_separation=min(self.min_separation,float(np.linalg.norm(self.a.xy-self.b.xy)))
        skill='IDLE' if self.active is None else self.active['skill']
        self.states.append(dict(t=self.t,xy=[self.a.xy.tolist(),self.b.xy.tolist()],yaw=[self.a.yaw,self.b.yaw],
            speed=[abs(self.a.v),abs(self.b.v)],signed_speed=[self.a.v,self.b.v],wheels=self.a.wheels.tolist(),
            selected=self.ref.active_target,human_state=self.human.state,a_state='HOLD' if skill in ('HOLD','AIM','SCAN') else 'RISK_ROUTE',
            a_reason='外部大模型调用固定运动技能',a_threat=0.,a_route=[p.tolist() for p in self.a.route],
            human_visible=self.human.visible,human_memory=None if self.human.memory is None else self.human.memory.tolist(),
            human_memory_age=None if self.human.seen_at is None else self.t-self.human.seen_at,
            collapsed=self.collapsed,llm_decision=len(self.decisions),llm_skill=skill,skill_status=self.skill_status,**self.ref.snapshot(self.t)))

    def observe(self):
        events=sorted(public_events(self.ref.events)+self.events,key=lambda e:e['time'])
        geometry=dict(distance=float(np.linalg.norm(self.a.xy-self.b.xy)),
            representative_point_los=block((*self.a.xy,.4),(*self.b.xy,1.05),self.planner.covers) is None,
            aim_geometry=[aim_valid(self.cfg,[self.a.xy,self.b.xy],[self.a.yaw,self.b.yaw],i,self.planner.covers) for i in range(2)])
        return dict(observation_id=len(self.decisions),state=self.states[-1],geometry=geometry,
            human_policy=dict(state=self.human.state,patrol_index=self.human.patrol_index,scan_until=self.human.scan_until,
                              route=[p.tolist() for p in self.b.route]),
            covers=self.planner.covers,task_points=self.cfg['arena']['task_points'],
            next_collapse_time=None if self.collapsed else self.cfg['collapse']['time'],
            recent_events=[e for e in events if e['time']>=max(0,self.t-2)],
            epistemic_limits='current state only; no RNG thresholds, future trajectories or rollout tool')

    def validation(self):
        assert self.min_clear>=self.cfg['planner_clearance']-1e-6
        assert self.min_separation>=self.cfg['policy']['safety_stop_distance']-1e-6
        return dict(score=self.ref.score,finish=self.ref.finished,min_center_to_cover=self.min_clear,
            min_actor_separation=self.min_separation,collapse_sweep_checked=self.collapsed,
            safety_stop_steps=self.safety_stops,controller='External LLM selects fixed skills every 2 simulated seconds',
            no_training=True,no_future_rollouts=True,token_usage=None,token_usage_status='not_exposed_by_session_tools')


def map_image(observation,path,route=None):
    from PIL import Image,ImageDraw,ImageFont
    im=Image.new('RGB',(1000,760),'#142333');d=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    def p(xy):return (80+(xy[0]+6)*70,640-(xy[1]+4)*70)
    d.rectangle((*p([-6,4]),*p([6,-4])),fill='#d5d9d9')
    for x in range(-6,7):d.line([p([x,-4]),p([x,4])],fill='#b4bebf')
    for y in range(-4,5):d.line([p([-6,y]),p([6,y])],fill='#b4bebf')
    for c in observation['covers']:
        x,y=c['center'];sx,sy,h=c['size']
        d.rectangle((*p([x-sx/2,y+sy/2]),*p([x+sx/2,y-sy/2])),fill='#546f86' if h>1 else '#b0a075')
        d.text(p([x-.25,y]),c['name'],font=small,fill='#11202d')
    s=observation['state']
    for k,target in enumerate(observation['task_points']):
        if not s['eligible'][k]:continue
        x,y=p(target['center']);r=target['radius']*70
        d.ellipse((x-r,y-r,x+r,y+r),outline='#16834a',width=4)
        d.text((x-35,y+r),target['name']+' '+('待触发' if s['target_remaining'] is None else f"{s['target_remaining']:.2f}s"),font=small,fill='#075c30')
    if observation['geometry']['representative_point_los']:d.line([p(s['xy'][0]),p(s['xy'][1])],fill='#e16956',width=2)
    for i,col in enumerate(('#0074b8','#e35a15')):
        x,y=p(s['xy'][i]);yaw=s['yaw'][i]
        d.ellipse((x-11,y-11,x+11,y+11),fill=col)
        d.line([(x,y),(x+38*math.cos(yaw),y-38*math.sin(yaw))],fill=col,width=4)
        for a in (yaw-math.pi/6,yaw+math.pi/6):d.line([(x,y),(x+65*math.cos(a),y-65*math.sin(a))],fill=col,width=2)
        d.text((x+12,y-18),f"{'A' if i==0 else 'B'} HP{s['hp'][i]}",font=small,fill=col)
    if route:d.line([p(s['xy'][0])]+[p(v) for v in route],fill='#0aa4bc',width=3)
    d.text((28,12),f"决策 {observation['observation_id']+1:02}  |  比赛 {s['t']:.2f}s  |  得分 {s['score']:.2f}  |  当前真实状态",font=font,fill='white')
    d.text((28,44),f"人形 {s['human_state']}   连续瞄准 A/B {s['aim'][0]:.2f}/{s['aim'][1]:.2f}s   冷却 {s['cooldown'][0]:.2f}/{s['cooldown'][1]:.2f}s",font=small,fill='#c4d8e8')
    d.text((30,670),'坐标：左下 (-6,-4)，右上 (6,4)，网格 1 米；箭头为朝向，小扇线为 60° 射击范围。',font=small,fill='white')
    d.text((30,700),'这是程序绘制的观测示意图，不是相机图像。蓝色高墙／棕色低墙；青线为已下发路径。',font=small,fill='white')
    im.save(path)


def save(match,directory):
    (directory/'.private_state.pkl').write_bytes(pickle.dumps(match))


def observation_packet(match,directory):
    obs=match.observe();index=len(match.decisions)
    path=directory/'observations'/f'{index:03d}.json';dump(path,obs)
    map_image(obs,path.with_suffix('.png'),[p.tolist() for p in match.a.route])
    ready=dict(observation_id=index,ready_monotonic_ns=time.monotonic_ns(),ready_unix_ns=time.time_ns(),
               ready_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(directory/'pending.json',ready)
    return dict(observation=obs,image=str(path.with_suffix('.png')),ready=ready)


def export(match,directory):
    dump(directory/'timeline.json',match.states)
    dump(directory/'events.json',sorted(public_events(match.ref.events)+match.events,key=lambda e:e['time']))
    dump(directory/'validation.json',match.validation())
    dump(directory/'decisions.json',match.decisions)
    dump(directory/'trial_config.json',match.cfg)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['init','act','observe','probe'])
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--request',type=Path)
    parser.add_argument('--points',type=str)
    args=parser.parse_args();directory=args.directory
    if args.command=='init':
        directory.mkdir(parents=True,exist_ok=False)
        for name in ('observations','decisions','requests','source_snapshot'): (directory/name).mkdir()
        cfg=load_config(PROJECT/'configs/trial_rotating.json');cfg['seed']=2026102107
        match=Match(cfg)
        for folder in ('scripts','configs'):
            dest=directory/'source_snapshot'/folder;dest.mkdir()
            for file in (PROJECT/folder).glob('*.py' if folder=='scripts' else '*.json'):(dest/file.name).write_bytes(file.read_bytes())
        dump(directory/'protocol.json',dict(planner='current assistant; exact backend model identifier not exposed',
            decision_period_sim_seconds=2.,seed=cfg['seed'],rules='v0.4',
            timing='observation ready to action accepted; includes reasoning, tool latency and intervening work; not pure model inference',
            tokens=None,token_status='not exposed; no invented token estimate',
            first_run_only=True,no_rewinds=True,skill_selection='external model, not QuadrupedPolicy',
            human_policy='unchanged HumanPolicy; all original limits',
            reference_baseline='run only after the LLM match finishes; unequal observation/control interfaces disclosed'))
        save(match,directory);export(match,directory);print(json.dumps(observation_packet(match,directory),ensure_ascii=False));return
    match=pickle.loads((directory/'.private_state.pkl').read_bytes())
    if args.command=='observe':
        print(json.dumps(dict(observation=match.observe(),pending=json.loads((directory/'pending.json').read_text())),ensure_ascii=False));return
    if args.command=='probe':
        result=[]
        for point in json.loads(args.points):
            route=match.planner.route(match.a.xy,np.asarray(point,float))
            result.append(dict(point=point,reachable=route is not None,
                path_length=None if route is None else path_length(match.a.xy,route),
                route=None if route is None else [p.tolist() for p in route],
                current_los_to_b=block((*point,.4),(*match.b.xy,1.05),match.planner.covers) is None))
        print(json.dumps(result));return
    if match.ref.finished:raise ValueError('Match already finished; no further actions permitted')
    request=json.loads(args.request.read_text());pending=json.loads((directory/'pending.json').read_text())
    if request['observation_id']!=len(match.decisions) or request['observation_id']!=pending['observation_id']:raise ValueError('Stale observation')
    for field in ('evidence','choice_summary','alternatives_summary','expected_result'):
        if not request.get(field):raise ValueError('Missing decision explanation '+field)
    route=match.compile(request['action'])
    accepted_ns=time.monotonic_ns()
    entry=dict(**request,decision_wall_seconds=(accepted_ns-pending['ready_monotonic_ns'])/1e9,
        observation_ready_utc=pending['ready_utc'],accepted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tokens=None,token_status='unavailable',compiled_route=[p.tolist() for p in route],simulation_start=match.t)
    # Exclusive immutable pre-execution commitment. A failed execution must be
    # audited, not silently overwritten or retried with another tactic.
    commit=directory/'decisions'/f"{len(match.decisions):03d}.json"
    with commit.open('x') as file:json.dump(entry,file,ensure_ascii=False,indent=2)
    match.decisions.append(entry);match.apply(request['action'],route)
    started=time.monotonic();match.advance(2.);elapsed=time.monotonic()-started
    save(match,directory);export(match,directory)
    outcome=dict(simulation_end=match.t,execution_wall_seconds=elapsed,state=match.states[-1])
    dump(commit.with_name(commit.stem+'_outcome.json'),outcome)
    packet=observation_packet(match,directory)
    print(json.dumps(dict(committed_decision_seconds=entry['decision_wall_seconds'],execution_wall_seconds=elapsed,**packet),ensure_ascii=False))


if __name__=='__main__':main()
