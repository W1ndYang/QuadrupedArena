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
             horizon, semantic_map, own_xy, maximum_particles=6, own_state=None):
    if own_state is not None:
        return optimize_aligned(trajectories,stage,unknown_probability,
                                semantic_map,own_state,maximum_particles)
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


def _reactive_outcome(trajectory,robot,walls,rules):
    """10 Hz interactive guard model with the runtime's reaction and lock rules.

    Facing is unobserved. Detection uses omnidirectional LOS as a conservative
    envelope; aiming still obeys turn, braking and tracking limits. This is
    explicitly a risk model, not a calibrated failure probability.
    """
    from formal_central_screen import HypothesisGuardController
    from humanoid_motion import HumanoidMover
    from fixed_goal_probe import smoke_blocks,valid_lock,TagClock,memory_update
    if not robot['complete'] or trajectory['covered_until']+1e-6<robot['seconds']:
        return dict(survived=False,seconds=robot['seconds'],reason='uncovered_future',
                    tag_time=None,exposure=1.)
    states=robot['states'][::5]
    if states[-1]['dt']!=robot['states'][-1]['dt']:states.append(robot['states'][-1])
    base_state=trajectory['states'][0]
    mover=HumanoidMover(base_state['xy'],base_state['yaw'],
                       trajectory['speed_mps'],1.8,4.5)
    mover.v=trajectory['initial_speed']
    controller=HypothesisGuardController(rules,mover.xy.tolist(),
        trajectory['waypoints'],[i for i,node in enumerate(trajectory['waypoint_ids'])
                                if node in trajectory['dwell_node_ids']],
        trajectory['dwell_seconds'])
    if trajectory.get('initial_dwell_remaining',0.)>0.:
        controller.dwell_until=trajectory['initial_dwell_remaining']
    if trajectory.get('possible_reaction_elapsed_seconds',0.)>0.:
        controller.first_visible=-trajectory['possible_reaction_elapsed_seconds']
        # The torso is unobserved and could already be tracking the robot.
        # Retain travel_yaw for feet, and include this adverse facing state
        # instead of granting a fresh turn-away delay at every boundary.
        mover.yaw=math.atan2(states[0]['xy'][1]-mover.xy[1],
                             states[0]['xy'][0]-mover.xy[0])
    clock=TagClock(.6);requested=False;memory=None;previous=0.;exposed=0
    for state in states:
        t=state['dt'];delta=t-previous;previous=t;own=state['xy']
        visible=(clear_ray((*own,.4),(*mover.xy,1.05),walls) and
                 not smoke_blocks((*own,.4),(*mover.xy,1.05),robot['smokes'],t))
        distance=math.dist(own,mover.xy)
        exposed+=int(visible and distance<=4.)
        bearing=math.atan2(own[1]-mover.xy[1],own[0]-mover.xy[0])
        error=(bearing-mover.yaw+math.pi)%(2*math.pi)-math.pi
        valid=valid_lock(visible,distance,error,mover.v,mover.omega,requested,rules)
        winner,_,_=clock.step(delta,[False,valid])
        if winner is not None:
            return dict(survived=False,seconds=t,tag_time=t,reason='reactive_proxy_tag',
                        exposure=exposed/max(1,len(states)))
        detection=dict(position=list(own),yaw=state['yaw']) if visible else None
        memory=memory_update(memory,detection,t,rules['memory_seconds'][1])
        packet=dict(time=t,self=dict(position=mover.xy.tolist(),yaw=mover.yaw,
                                    speed=mover.v),visible=detection,memory=memory)
        command=controller.decide(packet)
        requested=command['mode']=='AIM'
        if command['mode']=='PATROL':
            target=command['target']
            if not mover.route or math.dist(mover.route[-1],target)>1e-7:
                mover.navigate([target])
        else:mover.navigate([])
        look=command.get('look')
        look_yaw=(math.atan2(look[1]-mover.xy[1],look[0]-mover.xy[0])
                  if look is not None else mover.yaw+2.4*.1)
        mover.step(.1,look_yaw)
    return dict(survived=True,seconds=robot['seconds'],tag_time=None,
                reason='reactive_proxy_clear',exposure=exposed/max(1,len(states)))


def optimize_aligned(trajectories,stage,unknown_probability,semantic_map,public,maximum_particles):
    from llm_robot_rollout import candidates,geometry,selected_prediction
    started=time.perf_counter()
    chosen=sorted(trajectories,key=lambda row:row['probability'],reverse=True)[:maximum_particles]
    represented=sum(row['probability'] for row in chosen)
    walls=boxes(semantic_map['obstacles']);_,rules,_=geometry()
    rows=[]
    for robot in candidates(public,stage,min(20.,public['remaining_seconds'])):
        outcomes=[];exposure=0.
        for trajectory in chosen:
            outcome=_reactive_outcome(trajectory,robot,walls,rules)
            weight=trajectory['probability'];exposure+=weight*outcome.pop('exposure')
            outcomes.append(dict(outcome,weight=weight,
                hypothesis_id=trajectory['hypothesis_id'],variant=trajectory['variant']))
        success=sum(row['weight']*row['survived'] for row in outcomes)
        risk=max(0.,min(1.,1.-success));expected=sum(row['weight']*row['seconds'] for row in outcomes)
        objective=(risk+.25*exposure+.015*robot['wait']+.08*robot['smoke']+
                   .002*expected+.04*int(robot['maneuver']=='retreat'))
        rows.append(dict(side=robot['side'],wait=robot['wait'],smoke=robot['smoke'],
            risk=risk,expected_success=success,objective=objective,outcomes=outcomes,
            resource_cost=int(robot['smoke']),time_cost_seconds=expected,
            residual_uncertainty_mass=max(0.,1.-represented),proxy_exposure=exposure,
            robot_seconds=robot['seconds'],robot_complete=robot['complete'],
            maneuver=robot['maneuver'],
            end_boundary=robot['end_boundary'],gate_seconds=robot['gate_seconds']))
    best=min(rows,key=lambda row:(row['objective'],row['smoke'],row['wait'],row['side']!='north'))
    return dict(selected={key:best[key] for key in ('side','wait','smoke','maneuver')},
        candidates=rows,selected_risk=best['risk'],score_seconds=time.perf_counter()-started,
        trajectories_used=len(chosen),represented_probability=represented,
        robot_prediction=selected_prediction(public,stage,best),
        scoring_method='aligned_skill_reactive_v2',
        risk_interpretation='Conservative uncalibrated proxy with unknown guard facing',
        reason='Own-state Skill rollout to next safe boundary with reactive guard scoring')
