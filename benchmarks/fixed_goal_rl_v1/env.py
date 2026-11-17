"""10 Hz Skill policy adapter for the published fixed-goal v0.4.1 referee.

This module never alters the frozen v0.4.1 publication. Actor observations are
built only from the probe's public packet and the known map/initial position.
"""
import copy
import math
from pathlib import Path
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
V041=PROJECT/'benchmarks/fixed_goal_v041'
sys.path.insert(0,str(V041))

from ds_fixed_goal import IntentController,SkillProbe,WAYPOINTS,read  # noqa: E402
from fixed_goal_probe import boxes,clear_ray  # noqa: E402
from trial_motion import Planner  # noqa: E402


ACTION_NAMES=(
    'CONTINUE','GOAL_UPRIGHT','ESCAPE_A','ESCAPE_B','CENTRAL_NORTH',
    'CENTRAL_SOUTH','GOAL_LOW','TAG_AND_GOAL','AIM_AT_VISIBLE','SMOKE_AND_GOAL',
)
LAYOUTS=('SE','EW','WC')
FEEDBACK_NAMES=('initial','ROUTE_ACTIVE','CONTINUOUS_ROUTE_ACTIVE',
                'NO_ROUTE_TO_REQUESTED_TARGET','SMOKE_NOT_CURRENTLY_LEGAL_CONTINUE_DESTINATION')


def load_case(layout,perturb_seed=None):
    if layout not in LAYOUTS:raise ValueError(f'Unknown layout: {layout}')
    number=LAYOUTS.index(layout)+1
    cfg=read(V041/f'run_{number}/trial_config.json')
    spec=read(V041/'layouts.json')[layout]
    waypoints=copy.deepcopy(cfg['waypoints'])
    waypoints.update(ESCAPE_A=spec['escape_points'][0],ESCAPE_B=spec['escape_points'][1])
    if perturb_seed is not None:
        rng=np.random.default_rng(int(perturb_seed))
        base=cfg['base'];planner=Planner(base,base['arena']['covers'])
        walls=boxes(base['arena']['covers'])
        for _ in range(100):
            da=rng.uniform(-.22,.22,2);db=rng.uniform(-.22,.22,2)
            a=(np.asarray(base['start_a'])+da).tolist()
            b=(np.asarray(base['start_b'])+db).tolist()
            patrol=(np.asarray(base['guard_patrol'])+db).tolist()
            if (planner.free(a) and planner.free(b) and all(planner.free(q) for q in patrol)
                    and math.dist(a,b)>=10 and not clear_ray((*a,.4),(*b,1.05),walls)):
                base['start_a']=a;base['start_b']=b;base['guard_patrol']=patrol
                base['guard_yaw']+=float(rng.uniform(-.12,.12))
                waypoints['SPAWN']=a
                break
        else:raise RuntimeError(f'Could not sample legal {layout} perturbation')
    return cfg['base'],cfg['rules'],waypoints


class RLIntentController(IntentController):
    """Select at 10 Hz; execute each chosen formal Skill continuously at 50 Hz."""
    def __init__(self,base,rules,waypoints,policy):
        super().__init__(lambda _:None,base,rules,Path('/tmp'))
        self.waypoints=waypoints;self.policy=policy;self.records=[]
        # The environment starts with an explicit, public direct-goal Skill.
        # CONTINUE at the first decision therefore has an honest meaning.
        self.intent=dict(observation_id=-1,**self.request(1,None))
        self.last_action=None;self.current_intent_action=1;self.last_commit_time=0.
        self.next_decision=float('inf')

    def vector(self,packet):
        own=packet['self'];xy=np.asarray(own['position']);yaw=own['yaw'];t=packet['time']
        goal=np.asarray(self.waypoints['GOAL']);delta=goal-xy
        visible=packet['visible'];memory=packet['memory']
        track=visible or memory
        if track is None:
            track_fields=[0.]*8
        else:
            relative=np.asarray(track['position'])-xy
            bearing=math.atan2(relative[1],relative[0])-yaw
            track_fields=[1.,relative[0]/24.,relative[1]/16.,
                          float(np.linalg.norm(relative))/30.,math.sin(bearing),math.cos(bearing),
                          max(0.,t-track.get('last_seen',t))/4.,
                          math.sin(track.get('yaw',0.))]
        feedback=own.get('execution_feedback','initial')
        values=[xy[0]/12.,xy[1]/8.,math.sin(yaw),math.cos(yaw),own['speed']/3.2,
                delta[0]/24.,delta[1]/16.,float(np.linalg.norm(delta))/30.,
                t/self.rules['duration'],
                float(own.get('posture','upright')=='low'),
                own.get('transition_remaining',0.)/.4,
                own.get('tag_cooldown_remaining',0.)/4.,
                float(packet['smoke_available']),
                min(own.get('collision_stop_ticks',0),100)/100.,
                float(visible is not None),
                min(max(0.,t-self.last_commit_time),5.)/5.,
                (self.base['start_b'][0]-xy[0])/24.,(self.base['start_b'][1]-xy[1])/16.]
        values.extend(float(self.base['layout_id']==layout) for layout in LAYOUTS)
        values.extend(track_fields)
        values.extend(float(feedback==name) for name in FEEDBACK_NAMES)
        values.extend(float(self.current_intent_action==i) for i in range(len(ACTION_NAMES)))
        values.extend(float(self.last_action==i) for i in range(len(ACTION_NAMES)))
        result=np.asarray(values,dtype=np.float32)
        if not np.isfinite(result).all():raise RuntimeError('Non-finite actor observation')
        return result

    def smoke_location(self,packet):
        own=np.asarray(packet['self']['position']);goal=np.asarray(self.waypoints['GOAL'])
        track=packet['visible'] or packet['memory']
        candidates=[]
        if track is not None:
            enemy=np.asarray(track['position']);delta=enemy-own
            distance=float(np.linalg.norm(delta))
            if distance>1e-8:candidates.append(own+delta/distance*min(3.,distance/2.))
        delta=goal-own;distance=float(np.linalg.norm(delta))
        if distance>1e-8:candidates.append(own+delta/distance*min(3.,distance/2.))
        for candidate in candidates:
            q=candidate.tolist()
            if self.planner.free(q) and math.dist(own,q)<=6.:
                return q
        return own.tolist()  # The formal Skill validates the throw and reports failure.

    def request(self,action,packet):
        if action in (1,6):
            return dict(skill='MOVE',params=dict(target='GOAL',posture='low' if action==6 else 'upright',fire='off'))
        if action in (2,3,4,5):
            target={2:'ESCAPE_A',3:'ESCAPE_B',4:'CENTRAL_NORTH',5:'CENTRAL_SOUTH'}[action]
            return dict(skill='MOVE',params=dict(target=target,posture='upright',fire='off'))
        if action==7:return dict(skill='TAG_AND_ADVANCE',params=dict(target='GOAL',posture='upright'))
        if action==8:return dict(skill='AIM',params=dict(posture='upright'))
        if action==9:return dict(skill='SMOKE_AND_MOVE',params=dict(location=self.smoke_location(packet),target='GOAL',posture='upright'))
        raise ValueError(f'Unknown non-continue action {action}')

    def decide(self,packet):
        t=packet['time']
        if round(t/self.rules['dt'])%5==0:
            observation=self.vector(packet)
            action,logp,value=self.policy(observation)
            action=int(action)
            if not 0<=action<len(ACTION_NAMES):raise ValueError('Action outside catalog')
            self.records.append(dict(time=t,observation=observation,action=action,
                                     logp=float(logp),value=float(value),
                                     own_xy=list(packet['self']['position']),
                                     visible=packet['visible'] is not None))
            if action and action!=self.current_intent_action:
                intent=self.request(action,packet)
                self.intent=dict(observation_id=len(self.records)-1,**intent)
                self.current_intent_action=action;self.phase='approach'
                self.hold_started=None;self.tag_done=False;self.last_commit_time=t
            self.last_action=action
            self.samples=[]
        return super().decide(packet)


class RLProbe(SkillProbe):
    def __init__(self,base,rules,waypoints,policy):
        WAYPOINTS.clear();WAYPOINTS.update(copy.deepcopy(waypoints))
        super().__init__(base,rules,lambda _:None,Path('/tmp'))
        self.controllers[0]=RLIntentController(base,rules,waypoints,policy)


def run_episode(layout,policy,identifier=None,perturb_seed=None):
    base,rules,waypoints=load_case(layout,perturb_seed)
    probe=RLProbe(base,rules,waypoints,policy)
    result=probe.run(identifier or f'rl_{layout}')
    records=probe.controllers[0].records
    goal=np.asarray(waypoints['GOAL'])
    positions=[np.asarray(row['own_xy']) for row in records]+[np.asarray(result['states'][-1]['xy'][0])]
    rewards=[]
    for i in range(len(records)):
        progress=float(np.linalg.norm(positions[i]-goal)-np.linalg.norm(positions[i+1]-goal))
        rewards.append(.12*progress-.002)
    if rewards:
        rewards[-1]+=10. if result['summary']['winner']=='attacker' else 0. if result['summary']['winner']=='draw' else -10.
    return result,records,np.asarray(rewards,dtype=np.float32)
