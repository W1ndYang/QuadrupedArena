"""Turn connected LLM motion events into weighted, timed guard trajectories."""
import math

import numpy as np

from humanoid_motion import HumanoidMover

MAX_OBSERVATION_AGE_SECONDS = 4.

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
    if age < -1e-5 or age > MAX_OBSERVATION_AGE_SECONDS:
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


def generate(semantic_map, response, track, now, horizon, rules, mode='legacy'):
    if mode=='observed_anchor_v2':
        return generate_aligned(semantic_map,response,track,now,horizon,rules)
    if mode!='legacy':raise ValueError(f'Unknown trajectory mode: {mode}')
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


def _remaining_path(semantic_map,hypothesis,position,velocity):
    edges=[semantic_map['edges'][key] for key in hypothesis['edge_ids']]
    speed=float(np.linalg.norm(velocity))
    ranked=[]
    for i,edge in enumerate(edges):
        a=np.asarray(semantic_map['nodes'][edge['start']]['xy'],float)
        b=np.asarray(semantic_map['nodes'][edge['end']]['xy'],float)
        direction=b-a
        fraction=float(np.clip((position-a)@direction/max(1e-9,direction@direction),0.,1.))
        distance=float(np.linalg.norm(position-a-fraction*direction))
        alignment=(1.-float(direction@velocity)/max(1e-9,float(np.linalg.norm(direction))*speed)
                   if speed>.15 else 0.)
        ranked.append((distance+.35*alignment+.015*i,i,distance))
    _,index,distance=min(ranked)
    if distance>2.25:
        raise ValueError('LLM route does not connect to observed guard location')
    chosen=edges[index:]
    # Stay at the measured position: graph starts describe connectivity, not
    # a mandatory visit before the prediction may begin.
    return [edge['end'] for edge in chosen]


def generate_aligned(semantic_map,response,track,now,horizon,rules):
    latest=track['observations'][-1]
    age=float(now-latest['time'])
    if not -1e-5<=age<=MAX_OBSERVATION_AGE_SECONDS:
        raise ValueError('Latest guard sighting is too old or in the future')
    position=np.asarray(latest['xy'],float)
    velocity=np.asarray(track.get('estimated_velocity',[0.,0.]),float)
    observed_speed=min(1.5,float(np.linalg.norm(velocity)))
    dt=.02;age_ticks=round(age/dt);count=math.ceil(horizon/dt)
    if abs(age-age_ticks*dt)>1e-5:
        raise ValueError('Observation age must align with the simulation clock')
    result=[]
    for hypothesis in response['hypotheses']:
        if hypothesis['state']=='UNKNOWN' or hypothesis['probability']==0.:continue
        ids=_remaining_path(semantic_map,hypothesis,position,velocity)
        points=[semantic_map['nodes'][node]['xy'] for node in ids]
        heading=(math.atan2(velocity[1],velocity[0]) if observed_speed>.15 else
                 math.atan2(points[0][1]-position[1],points[0][0]-position[0]))
        dwell_nodes={event['at_id'] for event in hypothesis['events']
                     if event['type']=='DWELL'}
        lo,hi=hypothesis['speed_range_mps'];d0,d1=hypothesis['dwell_range_seconds']
        variants=((lo,d0),((lo+hi)/2.,(d0+d1)/2.),(hi,d1))
        for variant,(speed,dwell) in enumerate(variants):
            mover=HumanoidMover(position,heading,float(speed),
                rules['public_kinematics']['acceleration_mps2'],
                rules['public_kinematics']['yaw_rate_radps'])
            mover.v=min(observed_speed,speed)
            index=0
            # Initial dwell belongs to the currently observed endpoint, not
            # to a graph point that must first be reached by walking backward.
            initial_dwell=(hypothesis['state']=='DWELL' and observed_speed<.15)
            dwell_left=dwell if initial_dwell else 0.
            states=[];covered_until=horizon;request_index=0;request_dwell=0.
            for tick in range(age_ticks+count+1):
                relative=(tick-age_ticks)*dt
                if tick==age_ticks:
                    request_index=index;request_dwell=dwell_left
                if tick>=age_ticks and ((tick-age_ticks)%5==0 or tick==age_ticks+count):
                    states.append(dict(dt=round(relative,6),xy=mover.xy.tolist(),
                                       yaw=float(mover.yaw),speed=float(mover.v)))
                if dwell_left>1e-8:
                    mover.navigate([]);dwell_left=max(0.,dwell_left-dt)
                else:
                    if index<len(points) and math.dist(mover.xy,points[index])<.15:
                        arrived=ids[index];index+=1
                        if arrived in dwell_nodes:dwell_left=dwell
                    if index<len(points) and dwell_left<=1e-8:
                        if not mover.route or math.dist(mover.route[-1],points[index])>1e-7:
                            mover.navigate([points[index]])
                    else:mover.navigate([])
                if index>=len(points) and dwell_left<=1e-8:
                    covered_until=min(covered_until,max(0.,relative))
                mover.step(dt,mover.yaw+2.4*dt)
            if not states:raise ValueError('Empty aligned forecast')
            result.append(dict(hypothesis_id=hypothesis['id'],
                probability=hypothesis['probability']/3.,variant=variant,
                speed_mps=speed,dwell_seconds=dwell,
                start_xy=states[0]['xy'],start_yaw=states[0]['yaw'],
                initial_speed=states[0]['speed'],waypoint_ids=ids[request_index:],
                waypoints=points[request_index:],initial_dwell_remaining=request_dwell,
                dwell_node_ids=sorted(dwell_nodes),states=states,horizon=horizon,
                generated_at=now,source_observed_at=latest['time'],
                source_age_seconds=age,covered_until=covered_until,
                trajectory_method='observed_anchor_v2'))
    if not result:raise ValueError('No concrete motion hypothesis to simulate')
    return result


def local_motion_envelope(semantic_map,track,now,horizon,rules):
    """Short-horizon update using new legal measurements, without another API.

    Continue measured velocity with a small direction/speed envelope; reflect
    at the public patrol bounds. This is a numerical fallback, not LLM output.
    """
    latest=track['observations'][-1]
    velocity=np.asarray(track['estimated_velocity'],float)
    speed=float(np.linalg.norm(velocity))
    if speed>.15:
        heading=math.atan2(velocity[1],velocity[0]);unknown=.2
    else:
        old=next((row for row in reversed(track['observations'][:-1])
                  if math.dist(row['xy'],latest['xy'])>.1),None)
        heading=(math.atan2(latest['xy'][1]-old['xy'][1],latest['xy'][0]-old['xy'][0])
                 if old is not None else -math.pi/2 if latest['xy'][1]>0 else math.pi/2)
        unknown=.6
    local_map=dict(semantic_map,nodes=dict(semantic_map['nodes']),edges=dict(semantic_map['edges']))
    xlo,xhi=(-3.9,-2.1) if track['region']=='first' else (7.1,8.9)
    hypotheses=[]
    for variant,offset in enumerate((-.25,0.,.25)):
        point=np.asarray(latest['xy'],float).copy()
        point[0]=np.clip(point[0],xlo+1e-5,xhi-1e-5)
        point[1]=np.clip(point[1],-3.35+1e-5,3.35-1e-5)
        direction=np.array([math.cos(heading+offset),math.sin(heading+offset)])
        ids=[];edge_ids=[]
        for step in range(13):
            node=f'local_{variant}_{step}';ids.append(node)
            local_map['nodes'][node]=dict(id=node,xy=point.tolist(),region=track['region'])
            if step:
                edge=f'{ids[-2]}__{node}';edge_ids.append(edge)
                local_map['edges'][edge]=dict(id=edge,start=ids[-2],end=node,
                                            traversable_by=['guard'])
            hits=[]
            for axis,(lo,hi) in enumerate(((xlo,xhi),(-3.35,3.35))):
                if abs(direction[axis])>1e-9:
                    distance=((hi if direction[axis]>0 else lo)-point[axis])/direction[axis]
                    if distance>1e-7:hits.append((distance,axis))
            distance,axis=min(hits)
            point=point+distance*direction;direction[axis]*=-1.
        hypotheses.append(dict(id=f'local-motion-{variant}',probability=(1.-unknown)/3.,
            state='MOVE' if speed>.15 else 'DWELL',edge_ids=edge_ids,
            events=[dict(type='MOVE',at_id=edge) for edge in edge_ids],
            speed_range_mps=[max(.5,min(1.5,speed*.8)) if speed>.15 else .8,
                             max(.8,min(1.5,speed*1.2)) if speed>.15 else 1.5],
            dwell_range_seconds=[0.,0.] if speed>.15 else [0.,2.],
            next_target_id=ids[1],reason='Updated measured motion envelope',
            uncertainty='Unobserved future turns remain possible'))
    response=dict(hypotheses=hypotheses)
    traces=generate_aligned(local_map,response,track,now,horizon,rules)
    earliest=latest['time']
    if now-latest['time']<=.15:
        for row in reversed(track['observations'][:-1]):
            if earliest-row['time']>.151:break
            earliest=row['time']
        possible_reaction=min(rules['public_kinematics']['reaction_seconds'],now-earliest)
    else:possible_reaction=0.
    for trace in traces:
        trace['trajectory_method']='local_motion_envelope_v1'
        # Visual LOS is reciprocal, while guard facing is unobserved. Treat
        # continuous recent samples as possible prior detection in the same
        # conservative sensing envelope used by the risk model.
        trace['possible_reaction_elapsed_seconds']=possible_reaction
    return traces,unknown
