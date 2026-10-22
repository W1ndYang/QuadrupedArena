"""Pure allowlist boundary for onboard-RGB R0 observations.

No enemy state, visibility bit, raycast, detection, aim timer or policy is copied.
Known initial opponent position is disclosed only in observation zero. The model
must form its own visual memory; this module does not update it using truth.
"""
import math


def finite_vector(value,length):
    if not isinstance(value,(list,tuple)) or len(value)!=length:
        raise ValueError('Invalid vector size')
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in value):
        raise ValueError('Vector must be finite numeric values')
    return list(value)


def observation(index,sim_time,own,public,image_name,initial_opponent):
    if type(index) is not int or index<0:raise ValueError('Invalid observation index')
    packet=dict(observation_id=index,simulation_seconds=float(sim_time),
        self_state=dict(position=finite_vector(own['position'],3),yaw=float(own['yaw']),
            linear_velocity=finite_vector(own['linear_velocity'],3),hp=own.get('hp')),
        public_state={k:public[k] for k in ('score','active_target','target_generation','target_remaining','finished') if k in public},
        camera=dict(file=image_name,type='Isaac onboard RGB',privileged_overlays=False),
        opponent_information='Infer only from camera and your own prior observations; not seeing is not proof of absence')
    if index==0:packet['opponent_initial_position']=finite_vector(initial_opponent,2)
    return packet


def validate_action(value):
    if not isinstance(value,dict) or set(value)-{'skill','goal','yaw','attack'}:
        raise ValueError('Unknown action fields')
    skill=value.get('skill')
    if skill not in ('MOVE','TURN','HOLD','STOP'):raise ValueError('Unsupported embodied skill')
    if 'attack' in value and type(value['attack']) is not bool:raise ValueError('attack must be boolean')
    if skill=='MOVE':
        point=finite_vector(value.get('goal'),2)
        if not (-5.5<=point[0]<=5.5 and -3.5<=point[1]<=3.5):raise ValueError('Goal outside safe map bounds')
    elif 'goal' in value:raise ValueError('Only MOVE accepts goal')
    if skill=='TURN':
        if type(value.get('yaw')) not in (int,float) or not math.isfinite(value['yaw']):raise ValueError('TURN needs finite world yaw')
    elif 'yaw' in value:raise ValueError('Only TURN accepts yaw')
    return dict(value)
