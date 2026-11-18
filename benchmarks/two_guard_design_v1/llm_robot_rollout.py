"""Own-state Skill rollout to the actual next safe boundary, without a referee.

Reuse the runtime's navigation, rounded paths, motion and smoke executor. No
opponent state, seed, observation memory or future outcome enters this model.
"""
from functools import lru_cache
import math
import time
import numpy as np

from two_node_geometry import make_config, add_two_node_geometry, SECOND_COVER
from ds_fixed_goal import SkillProbe
from fixed_goal_probe import ContinuousMover, boxes, clear_ray
from trial_motion import Planner
from continuous_motion import rounded_route


@lru_cache(maxsize=1)
def geometry():
    base,rules=make_config()
    add_two_node_geometry(base)
    rules=dict(rules,tag_range_m=4.,tag_seconds=.6,goal_radius_m=1.7)
    return base,rules,Planner(base,base['arena']['covers'])


@lru_cache(maxsize=512)
def _rounded_path(start,target):
    _,_,planner=geometry()
    route=planner.route(start,target)
    if route is None:return None
    return tuple(tuple(map(float,point)) for point in rounded_route(start,route,planner))


class RobotOnlyProbe(SkillProbe):
    def __init__(self,public,smoke):
        base,rules,planner=geometry()
        self.cfg=base;self.rules=rules;self.planner=planner
        self.obstacles=boxes(base['arena']['covers'])
        mover=ContinuousMover(public['own_xy'],public['own_yaw'],
            base['postures']['upright']['speed'],
            base['postures']['upright']['acceleration'],base['yaw_rate'][0])
        mover.v=float(public['own_speed'])
        self.actor=[mover]
        self.heights=[.4];self.pose=self.pose_target='upright'
        self.pose_until=0.;self.pose_start=0.;self.pose_from=.4
        self.goal=[10.,0.];self.move_goal=[None];self.progress=0;self.waypoints=[]
        self.inventory=int(smoke);self.cast=None;self.smokes=[];self.events=[]
        self.feedback='initial';self.last_route_check=-99.;self.route_build_seconds=0.
        self.public_now=dict(visible=None)
        self.last_clock_time=0.
        self.exact_planning=False

    def route_to(self,i,target):
        if self.exact_planning:return SkillProbe.route_to(self,i,target)
        key=tuple(target);mover=self.actor[0]
        if self.move_goal[0]==key and mover.route:return
        started=time.perf_counter()
        route=_rounded_path(tuple(round(float(x),6) for x in mover.xy),key)
        self.route_build_seconds+=time.perf_counter()-started
        if route is None:
            self.stop(0);self.feedback='NO_ROUTE_TO_REQUESTED_TARGET';return
        mover.navigate(route);self.move_goal[0]=key;self.feedback='ROUTE_ACTIVE'


def rollout(public,stage,side,wait,smoke,passed_gate=False,limit=20.,retreat=False):
    probe=RobotOnlyProbe(public,smoke)
    mover=probe.actor[0];dt=.02
    gate=[6. if stage else -5.,1.6 if side=='north' else -1.6]
    destination=[10.,gate[1]] if stage else [3.01,0.]
    cover=[3. if stage else -8.,0.]
    if retreat:destination=cover
    direction=[gate[i]-public['own_xy'][i] for i in (0,1)]
    distance=math.hypot(*direction)
    factor=min(3.,distance/2.)/max(distance,1e-9)
    location=[public['own_xy'][i]+factor*direction[i] for i in (0,1)]
    # Match CorridorController.smoke_location_for, including its fallback.
    if not (probe.planner.free(location) and
            clear_ray((*public['own_xy'],1.),(*location,1.),probe.obstacles)):
        location=list(public['own_xy'])
    passed=bool(passed_gate);gate_time=0. if passed else None
    samples=[];complete=False
    for tick in range(round(limit/dt)+1):
        t=round(tick*dt,8)
        if tick%5==0 and not passed and math.dist(mover.xy,gate)<.35:
            passed=True;gate_time=t
        complete=(math.dist(mover.xy,cover)<.15 if retreat else
                  math.dist(mover.xy,SECOND_COVER)<.15 if stage==0 and passed
                  else math.dist(mover.xy,[10.,0.])<=1.7 if stage==1 else False)
        samples.append(dict(dt=t,xy=mover.xy.tolist(),yaw=float(mover.yaw),
                            speed=float(mover.v)))
        if complete:break
        probe.last_clock_time=t
        if t<wait-1e-8:
            command=dict(mode='WAIT_FIXED',look_degrees=0.,posture='upright')
        elif not passed and smoke and probe.inventory>0 and probe.cast is None:
            command=dict(mode='THROW',target=location,destination=gate,posture='upright')
        else:
            command=dict(mode='ADVANCE',target=destination if passed else gate,
                         posture='upright')
        look,_,_=probe.execute(0,command,t)
        before=mover.xy.copy();yaw=mover.yaw;omega=mover.omega
        following=bool(mover.route)
        mover.step(dt,look)
        # Route segments were already swept-clearance checked by the same
        # planner. Explicit braking can leave a route and needs a fresh check.
        if not following and not probe.planner.segment(before,mover.xy):
            mover.xy=before;mover.yaw=yaw;mover.omega=omega;mover.v=0.
        if probe.feedback=='NO_ROUTE_TO_REQUESTED_TARGET':break
    return dict(side=side,wait=wait,smoke=smoke,stage=stage,
                states=samples,seconds=samples[-1]['dt'],complete=complete,
                gate_seconds=gate_time,smokes=probe.smokes,
                route_build_seconds=probe.route_build_seconds,
                maneuver='retreat' if retreat else 'route',
                end_boundary='own_cover' if retreat else 'second_cover' if stage==0 else 'goal_circle',
                initial_state=dict(xy=list(public['own_xy']),yaw=public['own_yaw'],
                                   speed=public['own_speed']))


@lru_cache(maxsize=64)
def _candidates(key):
    stage,xy,yaw,speed,available,passed,side,limit,allow_retreat=key
    public=dict(own_xy=list(xy),own_yaw=yaw,own_speed=speed)
    rows=[rollout(public,stage,s,w,sm,passed,limit)
            for s in ((side,) if passed else ('north','south'))
            for w in (0.,1.,2.)
            for sm in ((False,True) if available and not passed else (False,))]
    if passed and allow_retreat:
        rows.append(rollout(public,stage,side,0.,False,True,limit,retreat=True))
    return rows


def candidates(public,stage,limit=20.):
    passed=bool(public['action_mask'][7])
    side=('south' if public.get('last_action') in (5,8) else 'north'
          if public.get('last_action') in (6,9) else
          'south' if public['own_xy'][1]<0. else 'north')
    return _candidates((stage,tuple(public['own_xy']),float(public['own_yaw']),
                        float(public['own_speed']),bool(public['smoke_available']),
                        passed,side,float(limit),bool(public.get('retreat_allowed',True))))


def prediction_horizon(public,stage):
    remaining=float(public['remaining_seconds'])
    rows=candidates(public,stage,min(20.,remaining))
    return min(remaining,max(row['seconds'] for row in rows)+.5)


def selected_prediction(public,stage,selection):
    row=next(row for row in candidates(public,stage,min(20.,public['remaining_seconds']))
             if all(row[key]==selection[key] for key in ('side','wait','smoke')) and
             row.get('maneuver','route')==selection.get('maneuver','route'))
    states=row['states'][::5]
    if states[-1]['dt']!=row['states'][-1]['dt']:states=states+[row['states'][-1]]
    return dict(initial_state=row['initial_state'],states=states,
                seconds=row['seconds'],complete=row['complete'],
                end_boundary=row['end_boundary'])


def _peek_return(initial,side):
    """Predict the public, deterministic opening PEEK using the same executor."""
    probe=RobotOnlyProbe(initial,False);probe.exact_planning=True
    mover=probe.actor[0];target=[-8.,-2.1 if side=='south' else 2.1]
    phase='approach';hold=None;dt=.02
    origin=initial.get('game_time',0.)
    for tick in range(600):
        t=round(origin+tick*dt,8)
        if tick%5==0 and phase=='retreat' and math.dist(mover.xy,[-8.,0.])<.15:
            return dict(own_xy=mover.xy.tolist(),own_yaw=mover.yaw,own_speed=mover.v,game_time=t)
        if phase=='approach' and math.dist(mover.xy,target)<.16:
            phase='observe';hold=t
        if phase=='observe' and t-hold>=.4:phase='retreat'
        mover.reverse_allowed=phase=='retreat';mover.reverse_limit=1.6
        probe.last_clock_time=t
        command=(dict(mode='WAIT_FIXED',look_degrees=math.degrees(mover.yaw),posture='upright')
                 if phase=='observe' else dict(mode='ADVANCE',posture='upright',
                    target=[-8.,0.] if phase=='retreat' else target))
        look,_,_=probe.execute(0,command,t);mover.step(dt,look)
    raise RuntimeError('Nominal opening peek did not return')


def _sensor_state(raw):
    return dict(own_xy=[float(np.float32(raw['own_xy'][0]/12.))*12.,
                        float(np.float32(raw['own_xy'][1]/8.))*8.],
        own_yaw=math.atan2(float(np.float32(math.sin(raw['own_yaw']))),
                           float(np.float32(math.cos(raw['own_yaw'])))),
        own_speed=float(np.float32(raw['own_speed']/3.2))*3.2,
        smoke_available=True,action_mask=[False]*10,last_action=2)


@lru_cache(maxsize=1)
def prewarm():
    """Pay nominal opening rollout costs before game time starts.

    Cache keys still include the exact sensor-rounded own state. A different
    pose, speed, inventory or stage falls back to a fresh rollout.
    """
    started=time.perf_counter()
    first=_peek_return(dict(own_xy=[-8.,0.],own_yaw=0.,own_speed=0.),'south')
    raw_states=[first,_peek_return(first,'north'),_peek_return(first,'south')]
    for raw in raw_states:candidates(_sensor_state(raw),0)
    return dict(seconds=time.perf_counter()-started,opening_states=len(raw_states))
