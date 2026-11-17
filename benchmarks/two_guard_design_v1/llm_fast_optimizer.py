"""Bounded online LOS proxy; formal candidate rollout remains an offline audit."""
from functools import lru_cache
import json
import math
from pathlib import Path
import time

from env import boxes, clear_ray
from llm_trajectory import interpolate as guard_at


HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _templates():
    data=json.loads((HERE/'llm_fast_templates.json').read_text())
    if data['schema']!='llm-fast-action-templates-v1':
        raise ValueError('Fast action templates have an unknown schema')
    return data


def _robot_at(trace, t):
    if t<=trace[0][0]:return trace[0][1]
    if t>=trace[-1][0]:return trace[-1][1]
    for (a,xy0),(b,xy1) in zip(trace,trace[1:]):
        if a<=t<=b:
            f=(t-a)/(b-a)
            return [(1-f)*xy0[i]+f*xy1[i] for i in (0,1)]
    raise RuntimeError('Unsorted robot action template')


def _segment_hits_disk(a,b,center,radius):
    dx=b[0]-a[0];dy=b[1]-a[1]
    factor=max(0.,min(1.,((center[0]-a[0])*dx+(center[1]-a[1])*dy)/
                        max(1e-9,dx*dx+dy*dy)))
    return math.hypot(a[0]+factor*dx-center[0],
                      a[1]+factor*dy-center[1])<=radius


def _score_trace(trajectory, robot, smoke, smoke_xy, wait, walls, horizon):
    if trajectory['covered_until']+1e-6<horizon:
        return False,trajectory['covered_until'],None,1.
    consecutive=0;exposed=0;tag_time=None
    for tick,(t,own_xy) in enumerate(robot):
        guard=guard_at(trajectory['states'],t)
        opponent=guard['xy']
        visible=math.dist(own_xy,opponent)<=4.
        if visible and smoke and wait+.75<=t<wait+5.75:
            visible=not _segment_hits_disk(own_xy,opponent,smoke_xy,2.)
        if visible:
            visible=clear_ray((*own_xy,.4),(*opponent,1.05),walls)
        if visible and consecutive==0:
            bearing=math.atan2(own_xy[1]-opponent[1],
                               own_xy[0]-opponent[0])
            facing_error=(bearing-guard['yaw']+math.pi)%(2*math.pi)-math.pi
            visible=abs(facing_error)<=math.pi/2
        if visible:
            exposed+=1;consecutive+=1
            if consecutive>=9:
                tag_time=t;break
        else:consecutive=0
    return tag_time is None,(tag_time if tag_time is not None else robot[-1][0]),tag_time,exposed/max(1,len(robot))


def optimize(trajectories, stage, smoke_available, unknown_probability,
             horizon, semantic_map, own_xy, maximum_particles=6):
    if not trajectories:
        raise ValueError('No trajectories')
    if not 0.<=unknown_probability<1.:
        raise ValueError('Invalid unknown probability')
    library=_templates()
    if library['map_id']!=semantic_map['map_id']:
        raise ValueError('Fast action templates do not match the map')
    chosen=sorted(trajectories,key=lambda row:row['probability'],reverse=True)[:maximum_particles]
    represented=sum(row['probability'] for row in chosen)
    residual=max(0.,1.-represented)
    walls=boxes(semantic_map['obstacles'])
    candidates=[];started=time.perf_counter()
    for template in library['rows']:
        if template['stage']!=stage or (template['smoke'] and not smoke_available):
            continue
        side,wait,smoke=(template[key] for key in ('side','wait','smoke'))
        start=template['attacker_trace'][0][1]
        offset=[own_xy[i]-start[i] for i in (0,1)]
        trace=[(t,[xy[i]+offset[i] for i in (0,1)])
               for t,xy in template['attacker_trace']]
        ticks=math.ceil(template['seconds']/.1)
        robot=[(round(i*.1,6),_robot_at(trace,min(i*.1,trace[-1][0])))
               for i in range(ticks+1)]
        gate=[(-5. if stage==0 else 6.),1.6 if side=='north' else -1.6]
        distance=math.dist(own_xy,gate)
        fraction=min(3.,distance/2.)/max(distance,1e-9)
        smoke_xy=[own_xy[i]+fraction*(gate[i]-own_xy[i]) for i in (0,1)]
        outcomes=[];exposure=0.
        for trajectory in chosen:
            survived,seconds,tag_time,exposed=_score_trace(
                trajectory,robot,smoke,smoke_xy,wait,walls,horizon)
            weight=trajectory['probability']
            exposure+=weight*exposed
            outcomes.append(dict(hypothesis_id=trajectory['hypothesis_id'],
                variant=trajectory['variant'],weight=weight,survived=survived,
                seconds=seconds,reason='proxy_clear' if survived else 'proxy_tag',
                tag_time=tag_time))
        success=sum(row['weight']*row['survived'] for row in outcomes)
        risk=min(1.,max(0.,1.-success))
        expected_time=sum(row['weight']*row['seconds'] for row in outcomes)
        objective=(risk+.25*exposure+.015*wait+.08*smoke+
                   .002*expected_time)
        candidates.append(dict(side=side,wait=wait,smoke=smoke,risk=risk,
            expected_success=success,time_cost_seconds=expected_time,
            resource_cost=int(smoke),objective=objective,
            residual_uncertainty_mass=residual,outcomes=outcomes,
            proxy_exposure=exposure))
    best=min(candidates,key=lambda c:(c['objective'],c['smoke'],c['wait'],
                                      c['side']!='north'))
    return dict(selected=dict(side=best['side'],wait=best['wait'],
                              smoke=best['smoke']),
                candidates=candidates,selected_risk=best['risk'],
                score_seconds=time.perf_counter()-started,
                trajectories_used=len(chosen),
                represented_probability=represented,
                scoring_method='fast_los_proxy_v1',
                reason='Bounded LOS and lock proxy over formal robot action templates')
