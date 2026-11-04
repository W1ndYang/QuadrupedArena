"""10 Hz Skill-level RL environment over the formal CPU match.

The actor never receives opponent truth.  A declared simulator-native sensor
uses truth internally to produce a noisy, visibility-gated estimate; it is not
an RGB detector.  Referee truth is otherwise restricted to reward/evaluation.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

from adaptive_motion import load_config
from tactical_logic import block,wrap
from trial_rules import covers_at
from vision_core import CameraRig,VisionMatch,public_packet
from semantic_skills import SKILLS,SemanticSkills


PROJECT=Path(__file__).resolve().parents[1]
ACTION_NAMES=(
    'CONTINUE',
    'CAPTURE_POINT_SAFE',
    'CAPTURE_POINT_FAST',
    'TAKE_COVER_SAFE',
    'AIM_AT_TRACK',
    'OBSERVE_SCAN',
    'OBSERVE_THREAT',
)
BLOCKED_STATES={
    'TARGET_STALE','TARGET_INACTIVE','TARGET_EXPIRED','NO_SAFE_COVER',
    'UNSAFE_ROUTE','NO_ROUTE','BLOCKED_SAFETY_STOP',
}
STATUS_NAMES=(
    'IDLE','RUNNING','HOLDING_POINT','AT_DESTINATION','EMERGENCY_COVER',
    'TARGET_STALE','TARGET_INACTIVE','TARGET_EXPIRED','NO_SAFE_COVER',
    'UNSAFE_ROUTE','NO_ROUTE','BLOCKED_SAFETY_STOP','OTHER',
)
LOW_LEVEL_NAMES=('HOLD','MOVE','SCAN','STOP','OTHER')


def load_rl_config(path=None):
    path=Path(path or PROJECT/'configs/rl_skill_v1.json')
    overrides=json.loads(path.read_text())
    base=load_config(path.parent/overrides.pop('base_config'))

    def merge(target,source):
        for key,value in source.items():
            if isinstance(value,dict) and isinstance(target.get(key),dict):merge(target[key],value)
            else:target[key]=value
    merge(base,overrides)
    return base


class NativeGeometrySensor:
    """Visibility-gated simulator sensor; truth is contained inside this class."""
    def __init__(self,cfg,rng):
        self.cfg=cfg;self.rng=rng;self.last_sample_time=None

    def sample(self,match,camera_pan_degrees):
        t=match.t;own=match.a.xy;target=match.b.xy
        if match.ref.hp[1]<=0:return None
        d=np.asarray(target)-own;distance=float(np.linalg.norm(d));sensor=self.cfg['sensor']
        heading=match.a.yaw+math.radians(camera_pan_degrees)
        visible=(distance<=sensor['range_m'] and
            abs(wrap(math.atan2(d[1],d[0])-heading))<=math.radians(sensor['horizontal_fov_degrees'])/2 and
            block((*own,self.cfg['representative_heights'][0]),(*target,self.cfg['representative_heights'][1]),covers_at(self.cfg,t)) is None)
        self.last_sample_time=t
        if not visible or self.rng.random()<sensor['dropout_probability']:return None
        estimate=np.asarray(target)+self.rng.normal(0,sensor['position_noise_std_m'],2)
        estimate=np.clip(estimate,[-6.,-4.],[6.,4.])
        return dict(time=t,xy=estimate.tolist(),uncertainty=sensor['base_uncertainty_m'])


class SkillEnv:
    """Small Gym-like API without a gym dependency."""
    action_names=ACTION_NAMES

    def __init__(self,config=None,seed=0,record_trajectory=False):
        self.cfg=copy.deepcopy(config or load_rl_config())
        ratio=self.cfg['decision_interval_seconds']/self.cfg['dt']
        sensor_ratio=self.cfg['sensor_interval_seconds']/self.cfg['dt']
        if abs(ratio-round(ratio))>1e-9 or round(ratio)<1:raise ValueError('Decision interval must be an integer number of referee steps')
        if abs(sensor_ratio-round(sensor_ratio))>1e-9 or round(sensor_ratio)<1:raise ValueError('Sensor interval must be an integer number of referee steps')
        self.microsteps=int(round(ratio));self.sensor_microsteps=int(round(sensor_ratio))
        self.base_seed=int(seed);self.record_trajectory=record_trajectory
        self.episode_index=0;self._obs_size=None

    @property
    def observation_size(self):
        if self._obs_size is None:self.reset(self.base_seed)
        return self._obs_size

    @property
    def action_size(self):return len(ACTION_NAMES)

    def reset(self,seed=None):
        episode_seed=self.base_seed+self.episode_index if seed is None else int(seed)
        self.episode_index+=1
        self.cfg['seed']=episode_seed
        self.rng=np.random.default_rng(np.random.SeedSequence(episode_seed).spawn(1)[0])
        self.match=VisionMatch(self.cfg);self.engine=SemanticSkills(self.cfg)
        self.rig=CameraRig();self.sensor=NativeGeometrySensor(self.cfg,self.rng)
        initial=dict(skill='HOLD',facing='keep',attack=False)
        self.match.apply(initial,self.match.compile(initial))
        self.camera_command=dict(mode='sweep',center_degrees=0,amplitude_degrees=60)
        self.camera_elapsed=0.;self.decision_index=0;self.last_signature=None
        # No fabricated previous action at reset: the persistence prior becomes
        # active only after the policy has actually selected an action.
        self.last_action=None;self.last_feedback=self._empty_feedback()
        self.last_target_distance=self._target_distance();self.terminal_bonus_paid=False
        self.trajectory=[]
        self._sample_sensor()
        observation=self._observation()
        if self._obs_size is None:self._obs_size=len(observation)
        elif len(observation)!=self._obs_size:raise RuntimeError('Observation shape changed')
        return observation,dict(seed=episode_seed,contract_version=self.cfg['contract_version'])

    def _empty_feedback(self):
        return dict(requested='RESET',validated=False,committed=False,continued=False,
            compiled_low_level='HOLD',status='IDLE',blocked=False,reason=None)

    def _packet(self):
        return public_packet(self.decision_index,self.match.states[-1],[],self.cfg['start_b'])

    def _sample_sensor(self):
        sighting=self.sensor.sample(self.match,self.rig.pan)
        if sighting is not None:self.engine.observe_simulator_sensor(sighting)
        return sighting

    def _request(self,action):
        mapping={
            1:('CAPTURE_POINT',{'target':'active','risk':'safe'}),
            2:('CAPTURE_POINT',{'target':'active','risk':'fast'}),
            3:('TAKE_COVER',{'risk':'safe'}),
            4:('AIM_AT_TRACK',{}),
            5:('OBSERVE',{'direction':'scan'}),
            6:('OBSERVE',{'direction':'threat'}),
        }
        skill,params=mapping[action]
        return dict(observation_id=self.decision_index,skill=skill,params=params)

    def _apply_policy_action(self,action):
        if type(action) not in (int,np.int32,np.int64) or not 0<=int(action)<self.action_size:
            raise ValueError('Action index outside catalog')
        action=int(action);feedback=self._empty_feedback();feedback['requested']=ACTION_NAMES[action]
        if action==0:
            feedback.update(validated=True,continued=True,compiled_low_level=self.match.active['skill'],
                status=self.engine.status.get('state','IDLE'))
            return feedback
        request=self._request(action);signature=(request['skill'],json.dumps(request['params'],sort_keys=True),self.match.ref.generation)
        state=self.engine.status.get('state')
        # Repeating an intent always means "keep executing".  A target refresh
        # changes the signature, so CAPTURE_POINT still replans immediately.
        # To retry an otherwise blocked intent the policy must select a distinct
        # action first; this prevents a 10 Hz stream from rebuilding the planner.
        if signature==self.last_signature:
            feedback.update(validated=True,continued=True,compiled_low_level=self.match.active['skill'],status=state)
            return feedback
        try:
            prepared=self.engine.validate(request,self._packet())
            if prepared is None:raise RuntimeError('Skill validator returned no prepared memory')
            feedback['validated']=True
            action_dict,route,self.camera_command,status=self.engine.commit(request,self._packet())
            self.match.compile(action_dict)
            self.match.apply(action_dict,route);self.camera_elapsed=0.;self.last_signature=signature
            feedback.update(committed=True,compiled_low_level=action_dict['skill'],status=status['state'],
                blocked=status['state'] in BLOCKED_STATES,
                reason=status['state'] if status['state'] in BLOCKED_STATES else None)
        except (ValueError,RuntimeError) as exc:
            feedback.update(blocked=True,reason=type(exc).__name__+': '+str(exc),status='VALIDATION_ERROR')
        return feedback

    def _microstep(self):
        packet=self._packet();interruption=self.engine.tick(packet)
        if interruption:
            action,route=interruption;self.match.compile(action);self.match.apply(action,route)
        elif self.engine.action is not None:
            for key in ('attack','facing'):self.match.active[key]=self.engine.action[key]
        if self.engine.intent is not None:self.camera_command=self.engine.camera(packet)
        self.match.advance(self.cfg['dt'])
        self.camera_elapsed+=self.cfg['dt']
        self.rig.step(self.camera_command,self.camera_elapsed%2.,self.cfg['dt'])
        if abs(self.match.t-self.cfg['collapse']['time'])<1e-8 and not self.match.ref.finished and self.engine.intent is not None:
            action,route,self.camera_command,_=self.engine.plan(self._packet());self.match.compile(action);self.match.apply(action,route)
        if self.match.tick%self.sensor_microsteps==0:self._sample_sensor()

    def _target_distance(self):
        target=self.cfg['arena']['task_points'][self.match.ref.active_target]
        return float(np.linalg.norm(self.match.a.xy-np.asarray(target['center'])))

    def _reward(self,previous):
        reward_cfg=self.cfg['reward'];score_delta=self.match.ref.score-previous['score']
        own_loss=previous['own_hp']-self.match.ref.hp[0];opponent_loss=previous['opponent_hp']-self.match.ref.hp[1]
        distance=self._target_distance();same_target=previous['generation']==self.match.ref.generation
        progress=(previous['distance']-distance) if same_target else 0.
        active=self.match.ref.active_target
        occupancy=float(self.match.ref.inside[active] and self.match.ref.hp[0]>0)
        components=dict(
            official_score=reward_cfg['official_score_scale']*score_delta,
            own_hp=reward_cfg['own_hp_loss']*own_loss,
            opponent_hp=reward_cfg['opponent_hp_loss']*opponent_loss,
            target_progress=reward_cfg['target_progress_per_m']*float(np.clip(progress,-.25,.25)),
            occupancy=reward_cfg['active_target_occupancy']*occupancy,
            blocked=reward_cfg['blocked_action']*float(self.last_feedback['blocked']),
            switch=reward_cfg['skill_switch']*float(self.last_feedback['committed']),
            terminal=0.,
        )
        if self.match.ref.finished=='time_limit' and not self.terminal_bonus_paid:
            components['terminal']=reward_cfg['alive_at_time_limit'];self.terminal_bonus_paid=True
        return sum(components.values()),components,distance

    @staticmethod
    def _one_hot(value,names):
        result=[0.]*len(names);result[names.index(value if value in names else names[-1])]=1.;return result

    def _observation(self):
        s=self.match.states[-1];own=s['xy'][0];yaw=s['yaw'][0];public=self._packet()['public_state']
        target=self.cfg['arena']['task_points'][public['active_target']]
        d=np.asarray(target['center'])-np.asarray(own);belief=self.engine.memory.belief(self.match.t)
        obs=[
            own[0]/6.,own[1]/4.,math.sin(yaw),math.cos(yaw),self.match.a.v/self.cfg['max_speed'][0],
            s['hp'][0]/self.cfg['initial_hp'],s['cooldown'][0]/self.cfg['cooldown_seconds'],
            self.match.t/self.cfg['duration'],s['score']/40.,
            d[0]/12.,d[1]/8.,float(np.linalg.norm(d))/14.,
            (public['target_remaining'] or 0.)/self.cfg['target_rotation']['lifetime_seconds'],
            float(public['target_remaining'] is not None),
            s['dwell'][public['active_target']]/max(1.,self.cfg['hold_seconds']),
            min(public['target_generation'],20)/20.,
            max(0.,self.cfg['collapse']['time']-self.match.t)/self.cfg['collapse']['time'],
            self.rig.pan/90.,
        ]
        obs += [float(public['active_target']==i) for i in range(len(self.cfg['arena']['task_points']))]
        if belief is None:
            obs += [0.]*9
        else:
            relative=np.asarray(belief['xy'])-np.asarray(own);angle=wrap(math.atan2(relative[1],relative[0])-yaw)
            obs += [1.,relative[0]/12.,relative[1]/8.,float(np.linalg.norm(relative))/14.,
                math.sin(angle),math.cos(angle),belief['age']/4.,min(belief['uncertainty_radius'],4.)/4.,float(belief['aim_usable'])]
        intent='NONE' if self.engine.intent is None else self.engine.intent
        obs += self._one_hot(intent,('NONE',)+SKILLS)
        obs += self._one_hot(self.engine.status.get('state','IDLE'),STATUS_NAMES)
        obs += self._one_hot(self.last_feedback['compiled_low_level'],LOW_LEVEL_NAMES)
        obs += [float(self.last_feedback['blocked']),float(self.last_feedback['continued']),float(self.last_feedback['committed'])]
        obs += [float(self.last_action==i) for i in range(self.action_size)]
        result=np.asarray(obs,dtype=np.float32)
        if not np.isfinite(result).all():raise RuntimeError('Non-finite actor observation')
        return result

    def step(self,action):
        if self.match.ref.finished:raise RuntimeError('Episode is finished; call reset')
        previous=dict(score=self.match.ref.score,own_hp=self.match.ref.hp[0],opponent_hp=self.match.ref.hp[1],
            generation=self.match.ref.generation,distance=self._target_distance())
        self.last_feedback=self._apply_policy_action(action);self.last_action=int(action)
        for _ in range(self.microsteps):
            if self.match.ref.finished:break
            self._microstep()
        reward,components,self.last_target_distance=self._reward(previous)
        self.decision_index+=1;done=self.match.ref.finished is not None
        info=dict(time=self.match.t,official_score=self.match.ref.score,finish=self.match.ref.finished,
            own_hp=self.match.ref.hp[0],opponent_hp=self.match.ref.hp[1],reward_components=components,
            feedback=copy.deepcopy(self.last_feedback),sensor_last_sample_time=self.sensor.last_sample_time)
        if self.record_trajectory:
            self.trajectory.append(dict(step=self.decision_index,action=int(action),action_name=ACTION_NAMES[int(action)],reward=reward,**copy.deepcopy(info)))
        return self._observation(),float(reward),done,False,info

    def episode_summary(self):
        requested=len(self.trajectory) if self.record_trajectory else self.decision_index
        blocked=sum(int(x['feedback']['blocked']) for x in self.trajectory) if self.record_trajectory else None
        committed=sum(int(x['feedback']['committed']) for x in self.trajectory) if self.record_trajectory else None
        return dict(seed=self.cfg['seed'],score=self.match.ref.score,finish=self.match.ref.finished,
            final_hp=self.match.ref.hp.copy(),decisions=requested,blocked_actions=blocked,committed_actions=committed,
            simulation_seconds=self.match.t,trajectory=self.trajectory if self.record_trajectory else None)
