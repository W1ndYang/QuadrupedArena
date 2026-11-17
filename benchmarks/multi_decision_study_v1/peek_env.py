"""10 Hz formal Skill interface with cover, peek, route, tag and smoke choices."""
import copy
import math

import numpy as np

from sequential import SequenceProbe,WAYPOINTS
from env import RLIntentController,IntentController


ACTIONS=(
    'CONTINUE','GOAL_UPRIGHT','WEST_COVER','PEEK_WEST','SOUTH_COVER',
    'PEEK_SOUTH','NORTH_COVER','PEEK_NORTH','GOAL_LOW','TAG_TO_GOAL',
    'SMOKE_TO_GOAL','AIM_AT_VISIBLE','VIA_SOUTH_ENTRY','VIA_NORTH_WINDOW',
)
COVER_WAYPOINTS={
    'WEST_COVER':[-5.5,-3.0],
    'WEST_PEEK':[-5.5,-3.5],
    'SOUTH_COVER':[-.5,-3.5],
    'SOUTH_PEEK':[.5,-4.5],
    'NORTH_COVER':[-1.,3.],
    'NORTH_PEEK':[1.,3.],
}


class PeekController(RLIntentController):
    action_names=ACTIONS

    def vector(self,packet):
        base=super().vector(packet)
        current=[float(self.current_intent_action==i) for i in range(len(self.action_names))]
        last=[float(self.last_action==i) for i in range(len(self.action_names))]
        pan=float(packet['self'].get('camera_pan',0.))/math.pi
        return np.concatenate((base[:-20],np.asarray([pan]+current+last,dtype=np.float32)))

    def request(self,action,packet):
        if action in (1,8):
            return dict(skill='MOVE',params=dict(target='GOAL',
                         posture='low' if action==8 else 'upright',fire='off'))
        if action in (2,4,6):
            target={2:'WEST_COVER',4:'SOUTH_COVER',6:'NORTH_COVER'}[action]
            return dict(skill='TAKE_COVER',params=dict(target=target,posture='upright'))
        if action in (3,5,7):
            target,ret={3:('WEST_PEEK','WEST_COVER'),
                        5:('SOUTH_PEEK','SOUTH_COVER'),
                        7:('NORTH_PEEK','NORTH_COVER')}[action]
            return dict(skill='PEEK',params=dict(target=target,return_to=ret,
                                                posture='upright',hold_seconds=.5))
        if action==9:return dict(skill='TAG_AND_ADVANCE',params=dict(target='GOAL',posture='upright'))
        if action==10:
            return dict(skill='SMOKE_AND_MOVE',params=dict(location=self.smoke_location(packet),
                                                           target='GOAL',posture='upright'))
        if action==11:return dict(skill='AIM',params=dict(posture='upright'))
        if action in (12,13):
            return dict(skill='MOVE',params=dict(target='SOUTH_ENTRY' if action==12
                                                else 'NORTH_WINDOW',posture='upright',fire='off'))
        raise ValueError(f'Unknown action {action}')

    def decide(self,packet):
        t=packet['time']
        if round(t/self.rules['dt'])%5==0:
            observation=self.vector(packet)
            action,logp,value=self.policy(observation)
            action=int(action)
            if not 0<=action<len(self.action_names):raise ValueError('Action outside catalog')
            self.records.append(dict(time=t,observation=observation,action=action,
                                     logp=float(logp),value=float(value),
                                     own_xy=list(packet['self']['position']),
                                     visible=packet['visible'] is not None))
            if action and action!=self.current_intent_action:
                self.intent=dict(observation_id=len(self.records)-1,
                                 **self.request(action,packet))
                self.current_intent_action=action;self.phase='approach'
                self.hold_started=None;self.tag_done=False;self.last_commit_time=t
            self.last_action=action
            self.samples=[]
        return IntentController.decide(self,packet)


class PeekSequenceProbe(SequenceProbe):
    def __init__(self,layout,seed,policy,duration=30.,attacker_start=None,
                 defender_start=None,defender_speed=None,hold_seconds=0.,
                 goal_names=None,defender_destinations=None,defender_yaw=None,
                 camera_half_fov=None):
        kwargs={}
        if goal_names is not None:kwargs['goal_names']=goal_names
        if defender_destinations is not None:kwargs['defender_destinations']=defender_destinations
        super().__init__(layout,seed,policy,duration,attacker_start,defender_start,
                         defender_speed,hold_seconds,defender_yaw=defender_yaw,**kwargs)
        self.camera_half_fov=camera_half_fov
        self.camera_pan=0.
        self.last_camera_t=0.
        self.camera_pan_rate=math.radians(120.)
        if camera_half_fov is not None:
            if not 0<camera_half_fov<=180:raise ValueError('Invalid camera half-FOV')
            self.rules['robot_half_fov_degrees']=float(camera_half_fov)
            self.metadata['rules']['robot_half_fov_degrees']=float(camera_half_fov)
            self.metadata['camera_pan_rate_degrees_per_second']=120.
        waypoints=copy.deepcopy(WAYPOINTS)
        waypoints.update(COVER_WAYPOINTS)
        WAYPOINTS.update(COVER_WAYPOINTS)
        self.controllers[0]=PeekController(self.cfg,self.rules,waypoints,policy)

    def observation(self,i,t):
        if i!=0 or self.camera_half_fov is None:return super().observation(i,t)
        body_yaw=self.actor[0].yaw
        action=self.controllers[0].current_intent_action
        desired_yaw=0. if action in (3,5,7) else body_yaw
        difference=math.atan2(math.sin(desired_yaw-body_yaw),
                              math.cos(desired_yaw-body_yaw))
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


def geometric_cover_visibility(base,waypoints,clear_ray,boxes):
    """Visibility from configured cover and peek points to selected patrol sites."""
    walls=boxes(base['arena']['covers'])
    pairs=(('WEST_COVER','WEST_PEEK'),('SOUTH_COVER','SOUTH_PEEK'),
           ('NORTH_COVER','NORTH_PEEK'))
    sites=('SOUTH_ENTRY','EAST_COURT_WEST','NORTH_WINDOW','CENTRAL_EAST','GOAL')
    rows=[]
    for cover,peek in pairs:
        for site in sites:
            other=waypoints[site]
            def sight(point):return clear_ray((*waypoints[point],.4),(*other,1.05),walls)
            rows.append(dict(cover=cover,peek=peek,site=site,
                             cover_visible=bool(sight(cover)),peek_visible=bool(sight(peek))))
    return rows
