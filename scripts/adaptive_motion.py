"""Asymmetric observation and risk-aware scripted policies; no learned control.

Only the observation adapter and referee see both actors' true positions.
HumanPolicy receives its own mover plus an optional visible observation, never A.
"""
import copy
import heapq
import json
import math
from pathlib import Path
import numpy as np
from tactical_logic import Mover, block, wrap
from trial_motion import Planner, clearance, path_length, sweep_clear
from trial_rules import Referee, covers_at


def load_config(path=None):
    path=Path(path or Path(__file__).resolve().parents[1]/'configs/trial_adaptive.json')
    overrides=json.loads(path.read_text())
    if 'base_config' not in overrides:return overrides
    base=load_config(path.parent/overrides.pop('base_config'))
    def merge(target,source):
        for key,value in source.items():
            if isinstance(value,dict) and isinstance(target.get(key),dict):merge(target[key],value)
            else:target[key]=value
    merge(base,overrides)
    return base


def visible_observation(config,own_xy,own_yaw,target_xy,covers,t):
    """Ideal geometric sensor, not a rendered camera or neural detector."""
    d=np.asarray(target_xy)-own_xy;p=config['policy']
    if np.linalg.norm(d)>p['human_sensor_range']:return None
    if abs(wrap(math.atan2(d[1],d[0])-own_yaw))>math.radians(p['human_sensor_half_angle_degrees']):return None
    h=config['representative_heights']
    if block((*own_xy,h[1]),(*target_xy,h[0]),covers):return None
    return dict(time=float(t),xy=list(target_xy))


class HumanPolicy:
    def __init__(self,config):
        self.cfg=config;self.memory=None;self.seen_at=None;self.state='PATROL'
        self.patrol_index=0;self.scan_until=None;self.last_plan=-99.;self.look=None
        self.visible=False;self.last_goal=None

    def update(self,t,mover,observation,planner,own_cooldown_until=0.,alive=True):
        # No opponent truth, HP, target occupancy, or referee aim accumulator here.
        p=self.cfg['policy'];self.visible=observation is not None
        previous=self.state
        if not alive:
            self.state='ELIMINATED';mover.route=[];mover.v=0.;self.look=mover.yaw;return
        if observation is not None:
            self.memory=np.asarray(observation['xy'],float).copy();self.seen_at=observation['time']
            d=self.memory-mover.xy;self.look=math.atan2(d[1],d[0]);mover.route=[]
            self.state='VISIBLE_COOLDOWN' if t<own_cooldown_until else 'VISIBLE_AIM'
            self.scan_until=None;return
        if self.memory is not None and t-self.seen_at<=p['human_memory_seconds']:
            self.state='SEARCH_LAST_SEEN'
            if previous.startswith('VISIBLE'):mover.route=[];self.last_plan=-99.
            d=self.memory-mover.xy
            if np.linalg.norm(d)>.3:
                if t-self.last_plan>=1.0 or not mover.route:
                    route=planner.route(mover.xy,self.memory)
                    if route:mover.navigate(route)
                    self.last_plan=t
                self.look=math.atan2(d[1],d[0])
            else:
                mover.route=[];self.look=mover.yaw+p.get('human_search_scan_rate',.75)*self.cfg['dt'];self.state='SEARCH_SCAN'
            return
        if self.memory is not None:
            self.memory=None;self.seen_at=None;mover.route=[];self.scan_until=None
        if self.scan_until is not None:
            if t<self.scan_until:
                mover.route=[];self.state='PATROL_SCAN';self.look=mover.yaw+p.get('human_patrol_scan_rate',.9)*self.cfg['dt'];return
            self.scan_until=None;self.patrol_index=(self.patrol_index+1)%len(p['human_patrol'])
        goal=np.asarray(p['human_patrol'][self.patrol_index])
        if np.linalg.norm(goal-mover.xy)<.12:
            mover.route=[];self.scan_until=t+p['human_scan_seconds'];self.state='PATROL_SCAN'
            self.look=mover.yaw+p.get('human_patrol_scan_rate',.9)*self.cfg['dt'];return
        self.state='PATROL'
        if not mover.route or previous not in ('PATROL','PATROL_SCAN'):
            route=planner.route(mover.xy,goal)
            if route:mover.navigate(route)
        self.look=mover.yaw  # Route heading drives the body, never hidden target truth.


def exposure(config,point,enemy_xy,enemy_yaw,covers,enemy_alive=True):
    if not enemy_alive:return 0.
    d=np.asarray(point)-enemy_xy;dist=float(np.linalg.norm(d));h=config['representative_heights']
    if dist>config['range_m'] or block((*enemy_xy,h[1]),(*point,h[0]),covers):return 0.
    angle=abs(wrap(math.atan2(d[1],d[0])-enemy_yaw))
    facing=1.0 if angle<=math.pi/6 else .45 if angle<=math.pi/3 else .12
    # Even a back-facing enemy may turn: a smaller, nonzero LOS cost remains.
    return facing+.15+max(0.,2.5-dist)*.25


class RiskPlanner:
    """Weighted grid search, preserving exposure costs during route shortcuts."""
    def __init__(self,base,enemy_xy,enemy_yaw,hp,cooldown_window=False,enemy_alive=True):
        self.base=base;self.cfg=base.cfg;self.grid=base.grid;self.covers=base.covers
        self.enemy=np.asarray(enemy_xy);self.enemy_alive=enemy_alive
        self.weight=self.cfg['policy']['risk_weight']*(self.cfg['policy']['low_hp_risk_multiplier'] if hp<=1 else 1.)
        self.weight*=.15 if cooldown_window else 1.
        self.nodes={n for n in base.free_nodes if self.dynamic_free(np.asarray(n)*self.grid)}
        self.risk={n:exposure(self.cfg,np.asarray(n)*self.grid,self.enemy,enemy_yaw,self.covers,enemy_alive) for n in self.nodes}
        self.cost={};self.parent={};self.start=None;self.source=None

    def dynamic_free(self,p):
        return not self.enemy_alive or np.linalg.norm(p-self.enemy)>=self.cfg['policy']['dynamic_avoid_radius']

    def free(self,p):return self.base.free(p) and self.dynamic_free(np.asarray(p))

    def segment(self,a,b):
        return all(self.free(np.asarray(a)*(1-u)+np.asarray(b)*u) for u in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(np.asarray(a)-b)/.08)+1)))

    def local_risk(self,p):
        n=tuple(np.rint(np.asarray(p)/self.grid).astype(int))
        return self.risk.get(n,0.)

    def segment_cost(self,a,b):
        distance=float(np.linalg.norm(np.asarray(a)-b))
        samples=[self.local_risk(np.asarray(a)*(1-u)+np.asarray(b)*u) for u in np.linspace(0,1,max(2,math.ceil(distance/.1)+1))]
        return distance*(1+self.weight*float(np.mean(samples)))

    def nearest(self,p):
        nodes=sorted(self.nodes,key=lambda n:(np.linalg.norm(np.asarray(n)*self.grid-p),n))
        return next((n for n in nodes[:30] if self.segment(p,np.asarray(n)*self.grid)),None)

    def solve(self,start):
        self.start=np.asarray(start).copy();s=self.nearest(self.start);self.source=s
        if s is None:return
        self.cost={s:self.segment_cost(self.start,np.asarray(s)*self.grid)};queue=[(self.cost[s],s)]
        while queue:
            value,n=heapq.heappop(queue)
            if value>self.cost[n]+1e-9:continue
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                v=(n[0]+dx,n[1]+dy)
                if v not in self.nodes:continue
                if dx and dy and ((n[0]+dx,n[1]) not in self.nodes or (n[0],n[1]+dy) not in self.nodes):continue
                nc=value+self.grid*math.hypot(dx,dy)*(1+self.weight*(self.risk[n]+self.risk[v])/2)
                if nc<self.cost.get(v,float('inf'))-1e-9:
                    self.cost[v]=nc;self.parent[v]=n;heapq.heappush(queue,(nc,v))

    def route(self,goal):
        goal=np.asarray(goal)
        if not self.free(goal):return None
        n=self.nearest(goal)
        if n not in self.cost:return None
        chain=[n]
        while chain[-1]!=self.source:chain.append(self.parent[chain[-1]])
        raw=[self.start,*[np.asarray(k)*self.grid for k in reversed(chain)],goal]
        prefix=[0.]
        for a,b in zip(raw,raw[1:]):prefix.append(prefix[-1]+self.segment_cost(a,b))
        route=[];i=0
        while i<len(raw)-1:
            j=len(raw)-1
            while j>i+1 and (not self.segment(raw[i],raw[j]) or self.segment_cost(raw[i],raw[j])>1.04*(prefix[j]-prefix[i])+1e-8):j-=1
            route.append(raw[j]);i=j
        return route

    def target_route(self,point):
        center=np.asarray(point['center']);candidates=[center]
        candidates += [center+point['radius']*.6*np.array([math.cos(a),math.sin(a)]) for a in np.linspace(0,2*math.pi,8,endpoint=False)]
        scored=[]
        for g in candidates:
            if not self.free(g):continue
            n=self.nearest(g)
            # Prefer the point interior when exposure is comparable: stopping
            # at the nearest circle edge often throws away the wall's protection.
            if n in self.cost:scored.append((self.cost[n]+self.segment_cost(np.asarray(n)*self.grid,g)+2*np.linalg.norm(g-center),g))
        if not scored:return None
        _,goal=min(scored,key=lambda v:v[0]);return self.route(goal)

    def cover_route(self,target):
        candidates=[]
        for n,value in self.cost.items():
            p=np.asarray(n)*self.grid;distance=np.linalg.norm(p-self.start)
            if .65<distance<3.5 and self.risk[n]==0.:
                candidates.append((value+.3*np.linalg.norm(p-target),n))
        for _,n in sorted(candidates)[:8]:
            route=self.route(np.asarray(n)*self.grid)
            if route:return route
        return None


class QuadrupedPolicy:
    def __init__(self,config):
        self.cfg=config;self.selected=None;self.state='ADVANCE';self.reason='选择有效目标'
        self.last_plan=-99.;self.evade_since=-99.;self.threat_since=None;self.known_b_cooldown=0.
        self.look=None;self.threat=0.;self.last_hp=config['initial_hp'];self.route_goal=None

    def public_hit(self,event):
        if event['type']=='hit' and event['attacker']==1:
            self.known_b_cooldown=event['time']+self.cfg['cooldown_seconds']

    def update(self,t,mover,enemy_xy,enemy_yaw,enemy_alive,eligible,hp,planner,map_changed=False):
        p=self.cfg['policy'];goal_changed=False
        if self.selected is None or not eligible[self.selected]:
            options=[(np.linalg.norm(mover.xy-point['center']),i) for i,point in enumerate(self.cfg['arena']['task_points']) if eligible[i]]
            if not options:return
            self.selected=min(options)[1];goal_changed=True;mover.route=[]
        point=self.cfg['arena']['task_points'][self.selected];goal=np.asarray(point['center'])
        self.threat=exposure(self.cfg,mover.xy,np.asarray(enemy_xy),enemy_yaw,planner.covers,enemy_alive)
        window=t<self.known_b_cooldown-.15
        danger=self.threat>=p['warning_risk_threshold'] and not window
        if danger and self.threat_since is None:self.threat_since=t
        if not danger:self.threat_since=None
        wounded=hp<self.last_hp;self.last_hp=hp
        in_target=np.linalg.norm(mover.xy-goal)<=point['radius']-.04
        settled=not mover.route or np.linalg.norm(mover.xy-mover.route[-1])<.08
        evade=enemy_alive and (wounded or (danger and t-self.threat_since>=p['a_react_seconds']))
        if self.state=='EVADE' and t-self.evade_since<p['a_min_evasion_seconds']:evade=True
        if in_target and settled and not evade and not goal_changed:
            mover.route=[];self.state='HOLD';self.reason='目标安全，驻留计分'
            self.look=mover.yaw+.6*self.cfg['dt'] if self.threat<.2 else math.atan2(enemy_xy[1]-mover.xy[1],enemy_xy[0]-mover.xy[0])
            return
        urgent=map_changed or goal_changed or wounded or evade and self.state!='EVADE' or window and self.state=='EVADE'
        if urgent or t-self.last_plan>=p['a_replan_seconds']:
            risk=RiskPlanner(planner,enemy_xy,enemy_yaw,hp,window,enemy_alive);risk.solve(mover.xy)
            route=risk.cover_route(goal) if evade else None
            if route:
                if self.state!='EVADE':self.evade_since=t
                self.state='EVADE';self.reason='受到威胁，先切断人形视线'
            else:
                route=risk.target_route(point)
                self.state='WINDOW_ADVANCE' if window else 'RISK_ROUTE'
                self.reason='根据公开命中事件利用冷却窗口' if window else '路径代价包含暴露风险，绕开火力方向'
            if route:
                mover.navigate(route);self.route_goal=np.asarray(route[-1]).tolist()
            else:
                mover.route=[];self.state='WAIT_SCAN';self.reason='暂时无安全通路，等待并重规划'
            self.last_plan=t
        # No perpetual aim-at-truth while protected. Body follows the route;
        # only exposed stationary actors orient for defensive return fire.
        self.look=(math.atan2(enemy_xy[1]-mover.xy[1],enemy_xy[0]-mover.xy[0])
                   if self.threat>=.8 and not mover.route else mover.yaw+.6*self.cfg['dt'])


class NearestPointPolicy(QuadrupedPolicy):
    """Ablation only: old nearest-point behavior against the same limited B."""
    def update(self,t,mover,enemy_xy,enemy_yaw,enemy_alive,eligible,hp,planner,map_changed=False):
        if self.selected is None or not eligible[self.selected]:
            options=[(np.linalg.norm(mover.xy-point['center']),i) for i,point in enumerate(self.cfg['arena']['task_points']) if eligible[i]]
            if not options:return
            self.selected=min(options)[1];mover.route=[]
        goal=self.cfg['arena']['task_points'][self.selected]['center']
        if not mover.route and np.linalg.norm(mover.xy-goal)>.08:
            route=planner.route(mover.xy,goal)
            if route:mover.navigate(route)
        self.state='NEAREST_BASELINE';self.reason='对照组：最近目标与最短几何路径'
        self.threat=exposure(self.cfg,mover.xy,np.asarray(enemy_xy),enemy_yaw,planner.covers,enemy_alive)
        self.look=math.atan2(enemy_xy[1]-mover.xy[1],enemy_xy[0]-mover.xy[0]) if enemy_alive else mover.yaw


class ReversibleMover(Mover):
    """Bounded differential-drive kinematics; reverse only while evading."""
    reverse_allowed=False
    reverse_limit=.8

    def step(self,dt,look=None,radius=.089,track=.4):
        while self.route and np.linalg.norm(self.route[0]-self.xy)<.055:self.route.pop(0)
        desired=0.;distance=float('inf');target=self.yaw if look is None else look
        if self.route:
            d=self.route[0]-self.xy;distance=float(np.linalg.norm(d));bearing=math.atan2(d[1],d[0])
            reverse=self.reverse_allowed and abs(wrap(bearing-self.yaw))>math.pi/2
            target=bearing+(math.pi if reverse else 0.)
            speed=min(self.reverse_limit if reverse else self.vmax,math.sqrt(2*self.accel*max(0.,distance-.035)))
            desired=-speed if reverse else speed
        error=wrap(target-self.yaw)
        if abs(error)>.25:desired=0.
        old_yaw=self.yaw;self.yaw+=float(np.clip(error,-self.turn*dt,self.turn*dt));self.omega=wrap(self.yaw-old_yaw)/dt
        self.v+=float(np.clip(desired-self.v,-self.accel*dt,self.accel*dt))
        ds=float(np.clip(self.v*dt,-distance,distance));self.xy+=ds*np.array([math.cos(self.yaw),math.sin(self.yaw)])
        self.distance+=abs(ds);self.wheels+=np.array([self.v-self.omega*track/2,self.v+self.omega*track/2]*2)/radius*dt


def safe_step(mover,dt,look,config,planner,other_xy,other_alive):
    xy=mover.xy.copy();wheels=mover.wheels.copy();distance=mover.distance;old_yaw=mover.yaw
    mover.step(dt,look,config['wheel_radius'],config['wheel_track'])
    unsafe=not planner.free(mover.xy) or other_alive and np.linalg.norm(mover.xy-other_xy)<config['policy']['safety_stop_distance']
    if unsafe:
        mover.xy=xy;mover.distance=distance;mover.v=0.
        delta=wrap(mover.yaw-old_yaw)*config['wheel_track']/2/config['wheel_radius']
        mover.wheels=wheels+np.array([-delta,delta,-delta,delta])
    return unsafe


def build(config,quadruped_mode='risk_aware'):
    dt=config['dt'];ref=Referee(config)
    movers=[(ReversibleMover if i==0 and quadruped_mode=='risk_aware' else Mover)(config[key],yaw,config['max_speed'][i],config['acceleration'][i],config['yaw_rate'][i])
            for i,(key,yaw) in enumerate((('start_a',0.),('start_b',math.pi)))]
    if quadruped_mode not in ('risk_aware','nearest_point'):raise ValueError(quadruped_mode)
    a,b=movers;ac=(QuadrupedPolicy if quadruped_mode=='risk_aware' else NearestPointPolicy)(config);bc=HumanPolicy(config)
    planner=Planner(config,covers_at(config,0));states=[];decisions=[];collapsed=False
    min_clear=float('inf');min_separation=float('inf');safety_stops=[0,0];event_cursor=0;previous=(None,None,None)
    for tick in range(round(config['duration']/dt)+1):
        t=round(tick*dt,8);changed=not collapsed and t>=config['collapse']['time']
        if changed:
            if not sweep_clear(config,[m.xy for i,m in enumerate(movers) if ref.hp[i]>0]):raise RuntimeError('Unsafe collapse sweep')
            collapsed=True;planner=Planner(config,covers_at(config,t));a.route=[];b.route=[]
        for event in ref.events[event_cursor:]:ac.public_hit(event)
        event_cursor=len(ref.events)
        observation=visible_observation(config,b.xy,b.yaw,a.xy,planner.covers,t) if min(ref.hp)>0 else None
        bc.update(t,b,observation,planner,ref.cooldown_until[1],ref.hp[1]>0)
        ac.update(t,a,b.xy.copy(),b.yaw,ref.hp[1]>0,ref.eligible.copy(),ref.hp[0],planner,changed)
        if isinstance(a,ReversibleMover):
            a.reverse_allowed=ac.state=='EVADE';a.reverse_limit=config['policy']['reverse_speed_limit']
        current=(ac.state,bc.state,ac.selected)
        if current[0]!=previous[0]:decisions.append(dict(time=t,type='quadruped_decision',state=ac.state,reason=ac.reason,goal=ac.route_goal))
        if current[1]!=previous[1]:decisions.append(dict(time=t,type='human_decision',state=bc.state,memory=None if bc.memory is None else bc.memory.tolist()))
        if current[2]!=previous[2]:decisions.append(dict(time=t,type='select_target',point=ac.selected))
        previous=current
        if tick and not changed:
            for i,(m,look) in enumerate(((a,ac.look),(b,bc.look))):
                if ref.hp[i]>0:safety_stops[i]+=int(safe_step(m,dt,look,config,planner,movers[1-i].xy,ref.hp[1-i]>0))
        xy=[m.xy.tolist() for m in movers];yaw=[m.yaw for m in movers]
        # Require B's own sensor to have acquired a target; the referee still
        # independently checks current geometric validity and the 60-degree FOV.
        ref.step(t,xy,yaw,planner.covers,map_changed=changed,attack_enabled=[True,bc.visible])
        for i,m in enumerate(movers):
            if ref.hp[i]>0:min_clear=min(min_clear,*[clearance(m.xy,c) for c in planner.covers])
        if min(ref.hp)>0:min_separation=min(min_separation,float(np.linalg.norm(a.xy-b.xy)))
        states.append(dict(t=t,xy=xy,yaw=yaw,speed=[abs(m.v) for m in movers],signed_speed=[m.v for m in movers],wheels=a.wheels.tolist(),
            selected=ac.selected,human_state=bc.state,a_state=ac.state,a_reason=ac.reason,a_threat=ac.threat,
            a_route=[v.tolist() for v in a.route],human_visible=bc.visible,
            human_memory=None if bc.memory is None else bc.memory.tolist(),
            human_memory_age=None if bc.seen_at is None else t-bc.seen_at,collapsed=collapsed,**ref.snapshot(t)))
        if ref.finished:break
    assert min_clear>=config['planner_clearance']-1e-6,min_clear
    assert min_separation>=config['policy']['safety_stop_distance']-1e-6,min_separation
    events=sorted(ref.events+decisions,key=lambda e:e['time'])
    validation=dict(min_center_to_cover=min_clear,min_actor_separation=min_separation,collapse_sweep_checked=collapsed,
        score=ref.score,finish=ref.finished,max_speed=[max(s['speed'][i] for s in states) for i in range(2)],
        safety_stop_steps=safety_stops,human_visible_seconds=sum(s['human_visible'] for s in states)*dt,
        a_exposed_seconds=sum(s['a_threat']>=.8 for s in states)*dt,
        quadruped_mode=quadruped_mode,
        controller='A: '+quadruped_mode+'; B: visibility-only pursuit, last-seen search and patrol',
        scope='ideal geometric sensor and kinematic center-clearance checks, not vision or contact dynamics')
    return states,events,validation


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--compare',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();config=load_config(args.config);states,events,validation=build(config)
    result=dict(validation=validation,last=states[-1],events=[e for e in events if e['type'] not in ('aim_started','aim_broken')])
    if args.compare:
        bs,be,bv=build(config,quadruped_mode='nearest_point')
        result['same_human_baseline']=dict(validation=bv,last=bs[-1])
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        for name,data in (('comparison.json',result),('timeline.json',states),('events.json',events),('config.json',config)):
            (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
