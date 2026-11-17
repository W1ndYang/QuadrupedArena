"""Candidate four-waypoint patrols inside the verified observation envelopes."""
import math

import numpy as np

from formal_gate_decisions import SouthDwellGuard


REGIONS={'first':(-3.9,-2.1),'second':(7.1,8.9)}
Y_BANDS=((-3.35,-3.0),(-1.8,-.3),(.3,1.8),(3.0,3.35))


def sample(seed,region):
    if region not in REGIONS:raise ValueError(region)
    rng=np.random.default_rng(int(seed))
    x_lo,x_hi=REGIONS[region]
    points=[[float(rng.uniform(x_lo,x_hi)),float(rng.uniform(*band))]
            for band in Y_BANDS]
    order=rng.permutation(4).tolist()
    waypoints=[points[i] for i in order]
    segment=int(rng.integers(4))
    u=float(rng.uniform(.05,.95))
    start=(np.asarray(waypoints[segment])*(1-u)+
           np.asarray(waypoints[(segment+1)%4])*u)
    patrol=waypoints[(segment+1)%4:]+waypoints[:(segment+1)%4]
    heading=math.atan2(patrol[0][1]-start[1],patrol[0][0]-start[0])
    return dict(seed=int(seed),region=region,start=start.tolist(),
                patrol=patrol,heading=heading,
                speed=float(rng.uniform(.8,1.5)),
                dwell=float(rng.uniform(0.,4.)),
                dwell_endpoint=('south' if rng.integers(2)==0 else 'north'),
                waypoint_order=order,segment=segment,segment_fraction=u)


def place(probe,seed,region):
    state=sample(seed,region)
    guard=probe.actor[1]
    guard.xy=np.asarray(state['start'],float)
    guard.yaw=state['heading']
    guard.travel_yaw=state['heading']
    guard.v=0.;guard.omega=0.
    guard.vmax=state['speed']
    guard.navigate([])
    controller=probe.controllers[1]
    controller.patrol=state['patrol']
    controller.patrol_index=0
    if state['dwell']>0:
        probe.controllers[1]=SouthDwellGuard(controller,state['dwell'],
                                              state['dwell_endpoint'])
    return state
