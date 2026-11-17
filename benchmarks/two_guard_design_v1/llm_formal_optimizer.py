"""Score hypothesis-conditioned actions through the existing formal referee."""
import math
import time

from formal_central_screen import run as formal_local_run
from two_node_geometry import add_two_node_geometry, make_config, Planner
from trial_motion import path_length


def horizon_for(stage, own_xy, remaining_seconds):
    base, _ = make_config()
    add_two_node_geometry(base)
    planner = Planner(base, base['arena']['covers'])
    gate_x = -5. if stage == 0 else 6.
    end_x = -1. if stage == 0 else 10.
    lengths = []
    for y in (-1.6, 1.6):
        one = planner.route(own_xy, (gate_x, y))
        two = planner.route((gate_x, y), (end_x, y))
        if one is not None and two is not None:
            lengths.append(path_length(own_xy, one)+path_length((gate_x,y),two))
    if not lengths:
        raise ValueError('No reachable candidate route')
    # Travel at 3.2 m/s, plus acceleration, 2 s wait and prediction margin.
    return min(float(remaining_seconds), max(6., min(16.,max(lengths)/3.2+5.)))


def _formal_spec(trajectory):
    return dict(start_xy=trajectory['start_xy'],
                start_yaw=trajectory['start_yaw'],
                initial_speed=trajectory['initial_speed'],
                speed_mps=trajectory['speed_mps'],
                waypoints=trajectory['waypoints'],
                dwell_node_indices=[i for i,node in enumerate(trajectory['waypoint_ids'])
                                    if node in trajectory['dwell_node_ids']],
                dwell_seconds=trajectory['dwell_seconds'])


def optimize(trajectories, stage, smoke_available, unknown_probability,
             horizon, maximum_particles=6):
    if not trajectories:
        raise ValueError('No trajectories')
    if not 0. <= unknown_probability < 1.:
        raise ValueError('Invalid unknown probability')
    chosen = sorted(trajectories, key=lambda row: row['probability'],
                    reverse=True)[:maximum_particles]
    represented = sum(row['probability'] for row in chosen)
    residual = max(0.,1.-represented)
    candidates = []
    started = time.perf_counter()
    for side in ('north','south'):
        for wait in (0.,1.,2.):
            for smoke in ((False,True) if smoke_available else (False,)):
                outcomes=[]
                for trajectory in chosen:
                    if trajectory['covered_until']+1e-6<horizon:
                        outcomes.append(dict(hypothesis_id=trajectory['hypothesis_id'],
                            variant=trajectory['variant'],
                            weight=trajectory['probability'],
                            survived=False,seconds=trajectory['covered_until'],
                            reason='uncovered_future',tag_time=None))
                        continue
                    row = formal_local_run(
                        seed=0,side=side,wait=wait,
                        duration=max(12.,min(18.,horizon+2.)),
                        tag_range=4.,guard_x=7.1 if stage else -3.9,
                        south_dwell_max=4.,middle_height=.6,
                        local_goal=True,second_peek=False,skip_peek=True,
                        route_y=1.6,tag_seconds=.6,
                        peek_if_unseen=False,stage2_local=bool(stage),
                        patrol_style='hypothesis',
                        hypothesis_guard=_formal_spec(trajectory),
                        use_smoke=smoke)
                    outcomes.append(dict(hypothesis_id=trajectory['hypothesis_id'],
                        variant=trajectory['variant'],
                        weight=trajectory['probability'],
                        survived=bool(row['local_survived']),
                        seconds=row['seconds'], reason=row['reason'],
                        tag_time=row['tag_time']))
                success=sum(item['weight']*item['survived'] for item in outcomes)
                risk=min(1.,max(0.,1.-success))
                # Unknown and omitted trajectory mass is conservatively unsafe.
                assert residual+1e-7 >= unknown_probability
                expected_time=sum(item['weight']*item['seconds'] for item in outcomes)
                objective=risk+.015*wait+.08*smoke+.002*expected_time
                candidates.append(dict(side=side,wait=wait,smoke=smoke,
                    risk=risk,expected_success=success,
                    time_cost_seconds=expected_time,
                    resource_cost=int(smoke), objective=objective,
                    residual_uncertainty_mass=residual,outcomes=outcomes))
    best=min(candidates,key=lambda c:(c['objective'],c['smoke'],c['wait'],
                                      c['side']!='north'))
    return dict(selected=dict(side=best['side'],wait=best['wait'],
                              smoke=best['smoke']),
                candidates=candidates,
                selected_risk=best['risk'],
                score_seconds=time.perf_counter()-started,
                trajectories_used=len(chosen),
                represented_probability=represented,
                scoring_method='formal_referee',
                reason=('Lowest conservative formal-referee risk plus '
                        'waiting, time and smoke costs'))
