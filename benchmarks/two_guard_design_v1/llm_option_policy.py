"""Public-sensor Option policy talking to an independent decision service."""
import json
import math
import urllib.error
import urllib.request

from online_env import ACTION_NAMES
from ppo_observation import FEEDBACK_NAMES, VisualHistoryEncoder


class DecisionServiceClient:
    def __init__(self,base_url='http://127.0.0.1:8765',timeout_seconds=900.):
        if not base_url.startswith('http://127.0.0.1:'):
            raise ValueError('Only a loopback decision service is supported')
        self.base_url=base_url.rstrip('/')
        self.timeout_seconds=float(timeout_seconds)

    def _post(self,path,payload):
        body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
        request=urllib.request.Request(self.base_url+path,data=body,
            headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=self.timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:detail=json.load(exc)
            except Exception:detail={'error':f'HTTP {exc.code}'}
            raise RuntimeError(f'Decision service rejected request: {detail}') from None

    def decide(self,payload):
        return self._post('/decide',payload)

    def finish(self,payload):
        return self._post('/finish',payload)


class LLMOptionPolicy:
    def __init__(self,episode_id,client,duration=48.):
        self.episode_id=episode_id
        self.client=client
        self.duration=float(duration)
        self.index=VisualHistoryEncoder(duration,len(ACTION_NAMES)).index
        self.stage=0
        self.stage_origin=0.
        self.observation_origin=0.
        self.request_count=0
        self.peek_count=[0,0]
        self.samples=[]
        self.own_trajectory=[]
        self.last_plan=None
        self.waits_left=0
        self.waits_done=0
        self.plan_time=None
        self.last_action=None
        self.responses=[]
        self.last_time=-1.
        self.recovery=False
        self.boundary_recheck=False
        self.boundary_waits=0
        self.retreats=0

    def _time(self,obs):
        return round((1.-float(obs[self.index['remaining_time']]))*self.duration,2)

    def _feedback(self,obs):
        ix=self.index
        return next((name for name in FEEDBACK_NAMES
                     if obs[ix['feedback_'+name]]>.5),'other')

    def observe(self,observation,mask):
        ix=self.index;t=self._time(observation)
        stage=int(observation[ix['stage_second']]>.5)
        if stage!=self.stage:
            if stage!=1 or self.stage!=0:raise ValueError('Invalid stage transition')
            self.stage=1;self.stage_origin=t
            self.observation_origin=t
            self.last_plan=None;self.waits_left=0;self.waits_done=0
            self.plan_time=None;self.recovery=False
            self.boundary_recheck=False;self.boundary_waits=0
            self.retreats=0
        if t+1e-4<self.last_time:raise ValueError('Game time moved backward')
        self.last_time=t
        self.own_trajectory.append(dict(time=t,stage=stage,
            xy=[float(observation[ix['self_x']])*12.,float(observation[ix['self_y']])*8.],
            yaw=math.atan2(float(observation[ix['body_yaw_sin']]),
                           float(observation[ix['body_yaw_cos']])),
            speed=float(observation[ix['self_speed']])*3.2))
        if observation[ix['detected_now']]>.5:
            own_x=float(observation[ix['self_x']])*12.
            own_y=float(observation[ix['self_y']])*8.
            measured_x=own_x+float(observation[ix['measured_dx']])*24.
            measured_y=own_y+float(observation[ix['measured_dy']])*16.
            yaw=math.atan2(float(observation[ix['body_yaw_sin']]),
                           float(observation[ix['body_yaw_cos']]))
            yaw+=float(observation[ix['camera_pan']])*math.pi
            self.samples.append(dict(time=t,xy=[measured_x,measured_y],
                observer_xy=[own_x,own_y],observer_yaw=yaw,
                uncertainty_m=float(observation[ix['measurement_uncertainty']])*2.,
                stage=stage,frame_valid=bool(observation[ix['frame_valid']]>.5)))

    def _request(self,observation,mask,decision_type):
        ix=self.index;t=self._time(observation)
        own=[float(observation[ix['self_x']])*12.,
             float(observation[ix['self_y']])*8.]
        yaw=math.atan2(float(observation[ix['body_yaw_sin']]),
                       float(observation[ix['body_yaw_cos']]))
        self.request_count+=1
        request=dict(episode_id=self.episode_id,
            request_id=f'{self.episode_id}:{self.request_count:04d}',
            time=t,stage=self.stage,decision_type=decision_type,
            public=dict(own_xy=own,own_yaw=yaw,
                own_speed=float(observation[ix['self_speed']])*3.2,
                smoke_available=bool(observation[ix['smoke_available']]>.5),
                remaining_seconds=float(observation[ix['remaining_time']])*self.duration,
                action_mask=[bool(x) for x in mask],
                peek_count=self.peek_count[self.stage],
                stage_elapsed_seconds=t-self.stage_origin,
                last_action=self.last_action,skill_feedback=self._feedback(observation)),
            observations=self.samples,
            skill_state=dict(current_option=self.last_action,
                             plan_request_id=(self.last_plan['request_id']
                                              if self.last_plan else None),
                             waits_done=self.waits_done,recovery=self.recovery,
                             observation_window_elapsed=t-self.observation_origin))
        if self.boundary_recheck:
            request['skill_state']['boundary_waits']=self.boundary_waits
            request['skill_state']['retreats']=self.retreats
        response=self.client.decide(request)
        if (response.get('request_id')!=request['request_id'] or
                response.get('episode_id')!=self.episode_id):
            raise RuntimeError('Decision service returned mismatched ID')
        if response.get('status')=='error':
            raise RuntimeError(f"Decision service failed: {response.get('error')}")
        self.samples=[]
        self.responses.append(dict(request_id=request['request_id'],
                                   status=response['status'],
                                   game_time=t,
                                   action=response.get('action'),
                                   plan=response.get('plan')))
        return response

    def __call__(self,observation,mask):
        ix=self.index;t=self._time(observation)
        if self.recovery and self.last_action==1 and not mask[7]:
            self.peek_count[self.stage]=0
            self.boundary_waits=0
            self.observation_origin=t
        feedback=self._feedback(observation)
        failed=feedback in ('NO_ROUTE_TO_REQUESTED_TARGET',
                            'SMOKE_NOT_CURRENTLY_LEGAL_CONTINUE_DESTINATION')
        collision=float(observation[ix['collision_stop']])>=.1-1e-6
        if failed or collision:
            self.last_plan=None;self.waits_left=0;self.recovery=True
            if mask[1]:
                self.last_action=1
                return 1,0.,0.
            raise RuntimeError('Skill failure with no legal safe-cover recovery')
        if mask[7]:
            self.last_plan=None
            if self.boundary_recheck:
                response=self._request(observation,mask,'gate_boundary')
                if response['status']!='boundary' or response.get('action') not in (1,4,7):
                    raise RuntimeError('Invalid boundary reassessment response')
                action=int(response['action'])
                if action==4:self.boundary_waits+=1
                if action==1:self.retreats+=1;self.recovery=True
            else:action=7
            self.last_action=action
            return action,0.,0.
        new_visual_after_plan=(self.plan_time is not None and
                               any(sample['time']>self.plan_time+1e-5
                                   for sample in self.samples))
        if self.last_plan and new_visual_after_plan and self.last_action==4:
            self.last_plan=None;self.waits_left=0
        if self.last_plan is None:
            decision_type=('execution_anomaly' if self.recovery else
                           'new_visual_after_wait' if new_visual_after_plan else
                           'observation_complete' if self.peek_count[self.stage]
                           else 'initial')
            response=self._request(observation,mask,decision_type)
            self.recovery=False
            if response['status']=='observe':
                action=int(response['action'])
                if action not in (2,3):raise ValueError('Invalid observe action')
                self.peek_count[self.stage]+=1
                self.last_action=action
                if not mask[action]:raise ValueError('Illegal observation action')
                return action,0.,0.
            if response['status']!='plan':
                raise RuntimeError('Unexpected decision status')
            plan=response['plan']
            if (plan['side'] not in ('south','north') or
                    plan['wait'] not in (0.,1.,2.) or type(plan['smoke']) is not bool):
                raise ValueError('Invalid plan schema')
            self.last_plan=dict(plan,request_id=response['request_id'])
            self.boundary_recheck=bool(response.get('boundary_recheck',False))
            self.waits_left=int(plan['wait'])
            self.waits_done=0
            self.plan_time=t
        if self.waits_left:
            action=4;self.waits_left-=1;self.waits_done+=1
        else:
            plan=self.last_plan
            action=(9 if plan['side']=='north' else 8) if plan['smoke'] else (
                    6 if plan['side']=='north' else 5)
        if not mask[action]:
            raise ValueError(f'Planned Option {ACTION_NAMES[action]} not legal')
        self.last_action=action
        return action,0.,0.

    def finish(self,summary):
        return self.client.finish(dict(episode_id=self.episode_id,
                                       observations=self.samples,
                                       summary=dict(winner=summary['winner'],
                                                    reason=summary['reason'],
                                                    seconds=summary['seconds'],
                                                    handoff_time=summary['handoff_time'],
                                                    smoke_used=summary['smoke_used'],
                                                    early_second_visible_ticks=(
                                                        summary['early_second_visible_ticks']))))
