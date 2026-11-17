"""Exploratory three-gate Skill game on the formal CPU referee.

The attacker may select any Skill at 10 Hz throughout a single objective run.
Crossing gates is physical navigation, not a scripted decision phase. The
defender has a continuous private multi-waypoint patrol sampled per episode.
This is a rule-search candidate; no PPO or validation claim is made here.
"""
import math

import numpy as np

from corridor_geometry import GATES,GOAL,SHELTER_PAIRS,make_config
from peek_env import PeekController
from env import RLProbe,clear_ray
from trial_motion import path_length


ACTIONS=('CONTINUE','GOAL')+tuple(
    f'G{i}_{side}_{part}'
    for i in range(1,4) for side in ('S','N') for part in ('COVER','PEEK')
)+tuple(
    f'G{i}_{side}_GATE' for i in range(1,4) for side in ('S','N')
)+('SMOKE_NEXT_S','SMOKE_NEXT_N','GOAL_LOW','TAG_TO_GOAL','AIM')
ALARM_START=(-10.,-3.)


def waypoint_catalog():
    points={'GOAL':list(GOAL)}
    for i,x in enumerate(GATES,1):
        for side,y in (('S',-3.),('N',3.)):
            cx,cy,px,py=SHELTER_PAIRS[i,'south' if y<0 else 'north']
            prefix=f'G{i}_{side}'
            points[prefix+'_COVER']=[cx,cy]
            points[prefix+'_PEEK']=[px,py]
            points[prefix+'_GATE']=[x,y]
    return points


def sample_route(rng,planner):
    route=[]
    for _ in range(6):
        for attempt in range(100):
            x=float(rng.uniform(-4.,4.))
            y=float(3*(-1 if rng.integers(2)==0 else 1)+rng.uniform(-.35,.35))
            if planner.free((x,y)):
                route.append([x,y]);break
        else:raise RuntimeError('Could not sample a defender waypoint')
    return route


def patrol_sample(seed,planner,patrol_seed=None):
    rng=np.random.default_rng(int(seed))
    side=-1. if rng.integers(2)==0 else 1.
    start=[float(rng.uniform(-.8,.8)),float(3*side+rng.uniform(-.3,.3))]
    if not planner.free(start):raise RuntimeError('Generated defender start is blocked')
    route=sample_route(rng,planner)
    if patrol_seed is not None:
        route=sample_route(np.random.default_rng(int(patrol_seed)),planner)
    return start,route,float(rng.uniform(1.15,1.85)),float(rng.uniform(-math.pi,math.pi))


def varied_stage_route(rng,planner,gate,count=10):
    """Continuous waypoints and a private cyclic patrol phase at one region."""
    side=-1. if rng.integers(2)==0 else 1.
    switch_probability=float(rng.uniform(.25,.65))
    route=[];sides=[]
    for index in range(count):
        if index and rng.random()<switch_probability:side=-side
        for _ in range(100):
            point=[float(gate+rng.uniform(.55,2.2)),
                   float(3*side+rng.uniform(-.6,.6))]
            if planner.free(point):
                route.append(point);sides.append(side);break
        else:raise RuntimeError('Could not sample varied patrol waypoint')
    if len(set(sides))<2:
        index=int(rng.integers(1,count))
        for _ in range(100):
            point=[float(gate+rng.uniform(.55,2.2)),
                   float(-3*side+rng.uniform(-.6,.6))]
            if planner.free(point):route[index]=point;break
        else:raise RuntimeError('Could not sample opposite patrol lane')
    phase=int(rng.integers(count))
    return route[phase:]+route[:phase]


def varied_defender_start(public,planner):
    side=-1. if public.integers(2)==0 else 1.
    for _ in range(100):
        start=[float(GATES[0]+public.uniform(.8,2.)),
               float(3*side+public.uniform(-.55,.55))]
        if planner.free(start):return start
    raise RuntimeError('Could not sample varied defender start')


NETWORK_ANCHOR_X=(-5.95,-3.5,-1.,1.4,4.)


def network_patrol_sample(rng,planner,attacker_start):
    """Draw a spatial patrol and spawn at a random point on its cyclic path."""
    anchors=[(x,y) for x in NETWORK_ANCHOR_X for y in (-3.,3.)]
    selected=rng.permutation(len(anchors))[:8]
    route=[]
    for index in selected:
        x,y=anchors[int(index)]
        for _ in range(100):
            point=[float(x+rng.uniform(-.28,.28)),
                   float(y+rng.uniform(-.28,.28))]
            if planner.free(point):
                route.append(point)
                break
        else:raise RuntimeError('Could not jitter network patrol anchor')
    # The waypoint order is a private permutation, not a fixed north/south cycle.
    for segment_index in rng.permutation(len(route)):
        before=route[int(segment_index)]
        after=route[(int(segment_index)+1)%len(route)]
        path=planner.route(before,after)
        if path is None:raise RuntimeError('Disconnected network patrol')
        length=path_length(before,path)
        for _ in range(12):
            distance=float(rng.uniform(0.,length))
            previous=np.asarray(before,float)
            for endpoint in path:
                step=float(np.linalg.norm(endpoint-previous))
                if distance<=step:
                    start=(previous+(endpoint-previous)*distance/max(step,1e-9)).tolist()
                    break
                distance-=step
                previous=endpoint
            else:start=np.asarray(after,float).tolist()
            if planner.free(start) and math.dist(start,attacker_start)>=5.:
                ordered=route[(int(segment_index)+1):]+route[:(int(segment_index)+1)]
                return start,ordered
    raise RuntimeError('Could not place network guard 5 m from attacker')


def alarm_patrol_sample(seed,planner,patrol_seed=None,variant='free',attacker_start=None):
    """One moving guard follows publicly announced gate crossings."""
    public=np.random.default_rng(int(seed))
    private=np.random.default_rng(int(seed)+41731 if patrol_seed is None else int(patrol_seed))
    if variant=='network':
        if attacker_start is None:raise ValueError('Network patrol requires attacker start')
        start,route=network_patrol_sample(private,planner,attacker_start)
        speed=float(private.uniform(1.05,2.05))
        return start,[route.copy() for _ in GATES],speed,float(private.uniform(-math.pi,math.pi))
    if variant=='free':
        side=-1. if public.integers(2)==0 else 1.
        start=[float(GATES[0]+public.uniform(.7,1.4)),
               float(3*side+public.uniform(-.25,.25))]
    elif variant=='committed':
        start=[float(GATES[0]+public.uniform(1.2,1.8)),
               float(public.uniform(-.3,.3))]
    elif variant in ('lane_phase','depart_hold'):
        start=[float(GATES[0]+public.uniform(1.15,1.45)),
               float(-3.2+public.uniform(-.08,.08))]
    elif variant=='varied':
        start=varied_defender_start(public,planner)
    else:raise ValueError(variant)
    if not planner.free(start):raise RuntimeError('Alarm defender start is blocked')
    stages=[]
    for gate in GATES:
        if variant=='varied':
            stages.append(varied_stage_route(private,planner,gate))
            continue
        route=[]
        if variant=='committed':
            side=-1. if private.integers(2)==0 else 1.
            for offset in (.7,2.0,.7,2.0):
                for attempt in range(100):
                    point=[float(gate+offset+private.uniform(-.15,.15)),
                           float(3*side+private.uniform(-.25,.25))]
                    if planner.free(point):
                        route.append(point);break
                else:raise RuntimeError('Could not sample a committed guard waypoint')
        if variant=='lane_phase':
            side=-1. if gate==GATES[0] else (-1. if private.integers(2)==0 else 1.)
            direction=-1. if private.integers(2)==0 else 1.
            offsets=(.7,.55) if direction<0 else (1.7,1.95)
            for offset in offsets:
                for attempt in range(100):
                    point=[float(gate+offset+private.uniform(-.08,.08)),
                           float(3.2*side+private.uniform(-.08,.08))]
                    if planner.free(point):
                        route.append(point);break
                else:raise RuntimeError(f'Could not sample lane-phase waypoint gate={gate} offset={offset} side={side}')
        if variant=='depart_hold':
            if gate==GATES[0]:
                depart=bool(private.integers(2))
                side=1. if depart else -1.
            else:side=-1. if private.integers(2)==0 else 1.
            for offset in (.7,1.8):
                for attempt in range(100):
                    point=[float(gate+offset+private.uniform(-.12,.12)),
                           float(3.2*side+private.uniform(-.12,.12))]
                    if planner.free(point):
                        route.append(point);break
                else:raise RuntimeError(f'Could not sample depart-hold waypoint gate={gate} side={side}')
        for _ in range(6):
            for attempt in range(100):
                point=[float(gate+private.uniform(-1.2,1.6)),
                       float(3*(-1 if private.integers(2)==0 else 1)+private.uniform(-.4,.4))]
                if planner.free(point):
                    route.append(point);break
            else:raise RuntimeError('Could not sample an alarm patrol waypoint')
        stages.append(route)
    speed=(public.uniform(1.05,2.05) if variant=='varied'
           else public.uniform(1.15,1.85))
    return (start,stages,float(speed),
            float(public.uniform(-math.pi,math.pi)))


class CorridorController(PeekController):
    action_names=ACTIONS
    smoke_uses_track=True

    def next_gate(self,own,side):
        for i,x in enumerate(GATES,1):
            if own[0]<x-.2:return f'G{i}_{side}_GATE'
        return 'GOAL'

    def smoke_location_for(self,packet,target):
        own=np.asarray(packet['self']['position'],float)
        destination=np.asarray(self.waypoints[target],float)
        track=(packet['visible'] or packet['memory']) if self.smoke_uses_track else None
        candidates=[]
        if track is not None:
            enemy=np.asarray(track['position'],float)
            direction=enemy-own
            length=float(np.linalg.norm(direction))
            if length>1e-8:candidates.append(own+direction/length*min(3.,length/2.))
        direction=destination-own
        length=float(np.linalg.norm(direction))
        if length>1e-8:candidates.append(own+direction/length*min(3.,length/2.))
        for point in candidates:
            q=point.tolist()
            if self.planner.free(q) and math.dist(own,q)<=6. and clear_ray((*own,1.),(*q,1.),self.walls):
                return q
        return own.tolist()

    def request(self,action,packet):
        if action==1 or action==22:
            return dict(skill='MOVE',params=dict(target='GOAL',
                        posture='low' if action==22 else 'upright',fire='off'))
        if 2<=action<=13:
            name=ACTIONS[action]
            if name.endswith('_COVER'):
                return dict(skill='TAKE_COVER',params=dict(target=name,posture='upright'))
            return dict(skill='PEEK',params=dict(target=name,
                         return_to=name.removesuffix('_PEEK')+'_COVER',
                         posture='upright',hold_seconds=.1))
        if 14<=action<=19:
            return dict(skill='MOVE',params=dict(target=ACTIONS[action],
                                                posture='upright',fire='off'))
        if action in (20,21):
            target=self.next_gate(packet['self']['position'],'S' if action==20 else 'N')
            return dict(skill='SMOKE_AND_MOVE',params=dict(
                location=self.smoke_location_for(packet,target),target=target,posture='upright'))
        if action==23:
            return dict(skill='TAG_AND_ADVANCE',params=dict(target='GOAL',posture='upright'))
        if action==24:return dict(skill='AIM',params=dict(posture='upright'))
        raise ValueError(action)


class CorridorProbe(RLProbe):
    def __init__(self,seed,policy,start_a=None,duration=25.,camera_half_fov=45.,
                 patrol_seed=None,alarm_patrol=False,alarm_variant='free',
                 smoke_uses_track=True,alarm_speed=None,
                 extra_covers=None,extra_waypoints=None,attacker_yaw=None):
        base,rules=make_config()
        if extra_covers:base['arena']['covers'].extend(extra_covers)
        if start_a is not None:base['start_a']=list(start_a)
        elif alarm_patrol:
            base['start_a']=list((-8.,-3.5) if alarm_variant in ('lane_phase','depart_hold') else ALARM_START)
        # Construct once for start and patrol feasibility; the formal probe
        # builds its own planner after configuration is fixed.
        from trial_motion import Planner
        planner=Planner(base,base['arena']['covers'])
        if alarm_patrol:
            b,stages,speed,yaw=alarm_patrol_sample(
                seed,planner,patrol_seed,alarm_variant,base['start_a'])
            patrol=stages[0]
        else:
            b,patrol,speed,yaw=patrol_sample(seed,planner,patrol_seed)
            stages=None
        base['start_b']=b;base['guard_patrol']=patrol
        base['max_speed'][1]=speed;base['guard_yaw']=yaw
        rules['duration']=float(duration)
        rules['robot_half_fov_degrees']=float(camera_half_fov)
        if alarm_variant in ('lane_phase','depart_hold','varied','network'):rules['tag_range_m']=4.
        points=waypoint_catalog()
        if extra_waypoints:points.update(extra_waypoints)
        super().__init__(base,rules,points,policy)
        if alarm_variant in ('lane_phase','depart_hold'):
            self.actor[0].yaw=-math.pi/2
        if attacker_yaw is not None:self.actor[0].yaw=float(attacker_yaw)
        self.controllers[0]=CorridorController(self.cfg,self.rules,points,policy)
        self.controllers[0].smoke_uses_track=bool(smoke_uses_track)
        self.alarm_patrol=bool(alarm_patrol)
        self.alarm_variant=alarm_variant
        self.alarm_stages=stages
        self.alarm_phase=0
        self.alarm_speed=None if alarm_speed is None else float(alarm_speed)
        self.alarm_history=[dict(phase=0,time=0.,route=patrol)] if alarm_patrol else []
        self.camera_pan=0.;self.last_camera_t=0.;self.camera_pan_rate=math.radians(120.)
        self.metadata.update(candidate=('three_gate_corridor_v4_depart_hold' if alarm_variant=='depart_hold'
                                        else 'three_gate_corridor_v7_network' if alarm_variant=='network'
                                        else 'three_gate_corridor_v6_varied' if alarm_variant=='varied'
                                        else 'three_gate_corridor_v3_lane_phase' if alarm_variant=='lane_phase'
                                        else 'three_gate_corridor_v2_committed' if alarm_variant=='committed'
                                        else 'three_gate_corridor_v1_alarm') if alarm_patrol
                             else 'three_gate_corridor_v0',seed=int(seed),
                             patrol_seed=None if patrol_seed is None else int(patrol_seed),
                             private_patrol=stages if alarm_patrol else patrol,defender_speed=speed,
                             camera_pan_rate_degrees_per_second=120.,
                             attacker_start_yaw=self.actor[0].yaw,
                             peek_retreat_reverse_limit_mps=1.6)
        self.metadata['smoke_uses_track']=bool(smoke_uses_track)
        self.metadata['alarm_speed']=self.alarm_speed

    def execute(self,i,command,t):
        if i==0:
            controller=self.controllers[0]
            action=controller.current_intent_action
            retreat=self.is_peek_action(action) and controller.phase=='retreat'
            self.actor[0].reverse_allowed=retreat
            self.actor[0].reverse_limit=1.6
        return super().execute(i,command,t)

    def is_peek_action(self,action):
        return 2<=action<=13 and ACTIONS[action].endswith('_PEEK')

    def peek_scan_heading(self,action,body_yaw):
        return 0. if self.is_peek_action(action) else body_yaw

    def observation(self,i,t):
        if i!=0:return super().observation(i,t)
        if self.alarm_patrol:
            while (self.alarm_phase<len(GATES)-1 and
                   self.actor[0].xy[0]>=GATES[self.alarm_phase]+.05):
                self.alarm_phase+=1
                if self.alarm_speed is not None:
                    self.actor[1].vmax=self.alarm_speed
                defender=self.controllers[1]
                route=self.alarm_stages[self.alarm_phase]
                if self.alarm_variant!='network':
                    defender.patrol=[self.actor[1].xy.tolist(),*route]
                    defender.patrol_index=0
                self.alarm_history.append(dict(phase=self.alarm_phase,time=t,route=route))
        body_yaw=self.actor[0].yaw
        action=self.controllers[0].current_intent_action
        desired=self.peek_scan_heading(action,body_yaw)
        difference=math.atan2(math.sin(desired-body_yaw),math.cos(desired-body_yaw))
        target=float(np.clip(difference,-math.pi/2,math.pi/2))
        travel=self.camera_pan_rate*max(0.,t-self.last_camera_t)
        self.camera_pan+=float(np.clip(target-self.camera_pan,-travel,travel))
        self.last_camera_t=t
        self.actor[0].yaw=body_yaw+self.camera_pan
        try:packet=super().observation(i,t)
        finally:self.actor[0].yaw=body_yaw
        packet['self']['yaw']=body_yaw
        packet['self']['camera_pan']=self.camera_pan
        if self.public_now is not None:
            self.public_now['self']['yaw']=body_yaw
            self.public_now['self']['camera_pan']=self.camera_pan
        return packet

    def run(self,identifier):
        result=super().run(identifier)
        states=result['states']
        crossing=[]
        for x in GATES:
            state=next((s for s in states if s['xy'][0][0]>=x),None)
            crossing.append(None if state is None else state['t'])
        result['summary'].update(gate_crossing_times=crossing,
                                 defender_patrol=self.metadata['private_patrol'],
                                 alarm_history=self.alarm_history)
        return result
