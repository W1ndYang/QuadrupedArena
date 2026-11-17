"""Turn connected LLM motion events into weighted, timed guard trajectories."""
import math

import numpy as np

from humanoid_motion import HumanoidMover


def interpolate(states, seconds):
    if seconds <= states[0]['dt']:
        return states[0]
    if seconds >= states[-1]['dt']:
        return states[-1]
    index = min(len(states)-2, max(0, int(seconds/.1)))
    left, right = states[index:index+2]
    f = (seconds-left['dt']) / max(1e-9, right['dt']-left['dt'])
    return dict(dt=seconds,
                xy=[(1.-f)*a+f*b for a,b in zip(left['xy'],right['xy'])],
                yaw=left['yaw']+f*((right['yaw']-left['yaw']+math.pi)%(2*math.pi)-math.pi),
                speed=(1.-f)*left['speed']+f*right['speed'])


def _start_state(track, now, region):
    observations = track['observations']
    if not observations:
        raise ValueError('No legal guard observation for trajectory generation')
    latest = observations[-1]
    age = now-latest['time']
    if age < -1e-5 or age > 4.:
        raise ValueError('Latest guard sighting is too old or in the future')
    xy = np.asarray(latest['xy'], float)
    velocity = np.asarray(track.get('estimated_velocity', [0., 0.]), float)
    if len(observations) >= 2:
        xy += velocity*min(age, .8)
    xlo, xhi = ((-3.9,-2.1) if region == 'first' else (7.1,8.9))
    xy[0] = np.clip(xy[0], xlo, xhi)
    xy[1] = np.clip(xy[1], -3.35, 3.35)
    speed = float(np.linalg.norm(velocity))
    yaw = math.atan2(velocity[1],velocity[0]) if speed > .15 else 0.
    return xy, yaw, min(speed,1.5)


def generate(semantic_map, response, track, now, horizon, rules):
    region = track['region']
    start, yaw, observed_speed = _start_state(track, now, region)
    dt = .1
    count = math.ceil(horizon/dt)
    traces = []
    for hypothesis in response['hypotheses']:
        if hypothesis['state'] == 'UNKNOWN' or hypothesis['probability'] == 0.:
            continue
        edges = [semantic_map['edges'][edge_id]
                 for edge_id in hypothesis['edge_ids']]
        first = np.asarray(semantic_map['nodes'][edges[0]['start']]['xy'],float)
        if np.linalg.norm(start-first) > 2.25:
            raise ValueError('LLM route does not connect to observed guard location')
        waypoints = [first.tolist(), *[
            semantic_map['nodes'][edge['end']]['xy'] for edge in edges]]
        waypoint_ids = [edges[0]['start'], *[edge['end'] for edge in edges]]
        dwells = {event['at_id'] for event in hypothesis['events']
                  if event['type'] == 'DWELL' and event.get('at_id') in semantic_map['nodes']}
        lo, hi = hypothesis['speed_range_mps']
        d0, d1 = hypothesis['dwell_range_seconds']
        variants = ((lo,d0),((lo+hi)/2.,(d0+d1)/2.),(hi,d1))
        for variant_index, (max_speed,dwell_seconds) in enumerate(variants):
            mover = HumanoidMover(start, yaw, float(max_speed),
                                   rules['public_kinematics']['acceleration_mps2'],
                                   rules['public_kinematics']['yaw_rate_radps'])
            mover.v = min(observed_speed, max_speed)
            mover.travel_yaw = yaw
            targets = list(waypoints)
            if math.dist(start, first) < .15:
                targets.pop(0)
            dwell_left = 0.
            covered_until = horizon
            states = [dict(dt=0.,xy=mover.xy.tolist(),yaw=float(mover.yaw),
                           speed=float(mover.v))]
            for tick in range(1, count+1):
                if dwell_left > 0.:
                    mover.navigate([])
                    dwell_left = max(0., dwell_left-dt)
                elif targets:
                    target = targets[0]
                    if math.dist(mover.xy,target) < .15:
                        arrived = targets.pop(0)
                        node = next((n for n in semantic_map['regions'][region]['node_ids']
                                     if semantic_map['nodes'][n]['xy'] == arrived),None)
                        if node in dwells:
                            dwell_left = dwell_seconds
                    if targets and dwell_left == 0.:
                        mover.navigate([targets[0]])
                    else:
                        mover.navigate([])
                else:
                    mover.navigate([])
                    if covered_until == horizon:
                        covered_until = round((tick-1)*dt,6)
                scan = rules['public_kinematics']['yaw_rate_radps']
                mover.step(dt,mover.yaw+min(scan,2.4)*dt)
                states.append(dict(dt=round(tick*dt,6),
                    xy=mover.xy.tolist(),yaw=float(mover.yaw),
                    speed=float(mover.v)))
            traces.append(dict(hypothesis_id=hypothesis['id'],
                probability=hypothesis['probability']/len(variants),
                variant=variant_index, speed_mps=max_speed,
                dwell_seconds=dwell_seconds, start_xy=start.tolist(),
                start_yaw=yaw, initial_speed=observed_speed,
                waypoint_ids=waypoint_ids,
                waypoints=waypoints, dwell_node_ids=sorted(dwells),
                states=states, horizon=horizon, generated_at=now,
                covered_until=covered_until))
    if not traces:
        raise ValueError('No concrete motion hypothesis to simulate')
    return traces
