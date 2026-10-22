"""Camera scheduling and strict public observations; no Isaac imports."""
import math
from llm_trial import Match


class VisionMatch(Match):
    def compile(self,action):
        if action.get('skill')=='AIM' or action.get('facing')=='enemy':
            raise ValueError('Privileged enemy tracking is unavailable in visual mode')
        return super().compile(action)


class CameraRig:
    def __init__(self):self.pan=0.;self.maximum_rate=120.

    @staticmethod
    def validate(command):
        if not isinstance(command,dict):raise ValueError('Camera command must be an object')
        mode=command.get('mode')
        if mode=='hold':
            if set(command)!={'mode','yaw_degrees'}:raise ValueError('hold requires yaw_degrees')
            values=[command['yaw_degrees']]
        elif mode=='sweep':
            if set(command)!={'mode','center_degrees','amplitude_degrees'}:raise ValueError('Invalid sweep fields')
            values=[command['center_degrees'],command['amplitude_degrees']]
        else:raise ValueError('Unknown camera mode')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Invalid camera angle')
        if mode=='hold' and abs(values[0])>90:raise ValueError('Camera yaw outside joint limits')
        if mode=='sweep' and (not 0<=values[1]<=60 or abs(values[0])+values[1]>90):raise ValueError('Sweep outside joint limits')
        return dict(command)

    def step(self,command,elapsed,dt):
        command=self.validate(command)
        if command['mode']=='hold':target=command['yaw_degrees']
        else:
            # Four real-time half-second legs: center->left->center->right->center.
            u=min(2.,max(0.,elapsed));leg=min(3,int(u/.5));alpha=(u-leg*.5)/.5
            values=[0,1,0,-1,0]
            target=command['center_degrees']+command['amplitude_degrees']*(values[leg]*(1-alpha)+values[leg+1]*alpha)
        change=max(-self.maximum_rate*dt,min(self.maximum_rate*dt,target-self.pan))
        self.pan=max(-90.,min(90.,self.pan+change));return self.pan


def public_packet(index,state,frames,initial_b):
    packet=dict(observation_id=index,simulation_seconds=state['t'],
        self_state=dict(xy=list(state['xy'][0]),yaw=state['yaw'][0],hp=state['hp'][0],
            speed=state['speed'][0],cooldown=state['cooldown'][0],skill_status=state['skill_status']),
        public_state={k:state[k] for k in ('score','active_target','target_generation','target_remaining','dwell','finished')},
        frames=frames,perception='Use only these raw onboard RGB frames and prior visual memory; no enemy state or automatic detections supplied')
    if index==0:packet['opponent_initial_xy']=list(initial_b)
    return packet


def validate_request(request,match):
    fields={'observation_id','action','camera','evidence','choice_summary','alternatives_summary','expected_result'}
    if not isinstance(request,dict) or set(request)!=fields:raise ValueError('Expected exactly seven request fields')
    if type(request['observation_id']) is not int or request['observation_id']!=len(match.decisions):raise ValueError('Wrong observation_id')
    for name in ('evidence','choice_summary','alternatives_summary','expected_result'):
        if not isinstance(request[name],str) or not request[name].strip():raise ValueError('Missing decision summary')
    CameraRig.validate(request['camera'])
    return match.compile(request['action'])
