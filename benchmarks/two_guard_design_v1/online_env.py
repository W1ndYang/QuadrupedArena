"""10 Hz continuous Skill interface for the varied two-guard CPU candidate.

The attacker policy receives only VisualHistoryEncoder output and a public
action mask. The second patrol runs independently from t=0 and is handed to
the formal two-actor referee at the second concealed room.
"""
import math

import numpy as np

from formal_two_stage import (
    FIRST_POINTS,HandoffProbe,ShadowGuard,place_varied_guard,
)
from formal_central_screen import CentralController
from env import IntentController,WAYPOINTS,boxes
from ppo_observation import GeometryDetectionAdapter,VisualHistoryEncoder
from trial_motion import Planner
from two_node_geometry import SECOND_COVER,add_two_node_geometry


ACTION_NAMES = (
    'CONTINUE', 'TAKE_COVER', 'PEEK_SOUTH', 'PEEK_NORTH', 'WAIT',
    'SOUTH_GATE', 'NORTH_GATE', 'NEXT_COVER_OR_GOAL',
    'SMOKE_SOUTH', 'SMOKE_NORTH',
)
ACTION_TO_FORMAL = {1:25,2:26,3:27,4:28,5:14,6:15,7:29,8:20,9:21}


class OnlineSession:
    def __init__(self, policy, encoder):
        self.policy = policy
        self.encoder = encoder
        self.stage = 0
        self.probe = None
        self.handoff_time = None

    def act(self, observation, mask):
        return self.policy(observation, mask)

    def ready_for_handoff(self,own_xy):
        return self.probe.controllers[0].passed_gate

    def start_second(self,t):
        self.stage = 1
        self.handoff_time = t
        controller = self.probe.controllers[0]
        controller.reset_stage(t)
        controller.waypoints['LOCAL_EXIT'] = [10.,0.]
        WAYPOINTS['LOCAL_EXIT'] = [10.,0.]


class OnlineController(CentralController):
    action_names = ACTION_NAMES

    def __init__(self,cfg,rules,waypoints,session):
        super().__init__(cfg,rules,waypoints,session)
        self.session = session
        self.encoder = session.encoder
        self.passed_gate = False
        # The Skill executor does not use a 50 Hz referee memory to aim smoke.
        self.smoke_uses_track = False

    def reset_stage(self,t):
        self.passed_gate = False
        self.chosen_gate = None
        self.current_intent_action = 4
        self.last_action = None
        self.intent = dict(observation_id=len(self.records),
                           **self.request(4,None))
        self.phase = 'approach'
        self.hold_started = None
        self.tag_done = False
        self.last_commit_time = t
        self.samples = []

    def request(self,action,packet):
        if action not in ACTION_TO_FORMAL:raise ValueError(action)
        if action in (5,6,8,9):
            self.chosen_gate = 'south' if action in (5,8) else 'north'
        if action == 7 and self.session.stage == 1 and self.chosen_gate:
            point = [10., -1.6 if self.chosen_gate == 'south' else 1.6]
            self.waypoints['LOCAL_EXIT'] = point
            WAYPOINTS['LOCAL_EXIT'] = point
        return super().request(ACTION_TO_FORMAL[action],packet)

    def legal_actions(self,packet):
        own=np.asarray(packet['self']['position'],float)
        stage=self.session.stage
        gate_x=-5. if stage==0 else 6.
        cover_x=-8. if stage==0 else 3.
        completed_peek=(self.phase=='retreat' and
                        math.dist(own,(cover_x,0.))<.15)
        if not self.passed_gate:
            if any(math.dist(own,point)<.35
                   for point in ((gate_x,-1.6),(gate_x,1.6))):
                self.passed_gate=True
        before_gate=not self.passed_gate
        smoke=bool(packet.get('smoke_available',False))
        return np.asarray((
            True,True,
            before_gate and not(completed_peek and self.current_intent_action==2),
            before_gate and not(completed_peek and self.current_intent_action==3),
            True,
            before_gate,before_gate,self.passed_gate,
            before_gate and smoke,before_gate and smoke,
        ),dtype=np.bool_)

    def decide(self,packet):
        t=float(packet['time'])
        if round(t/self.rules['dt'])%5==0:
            mask=self.legal_actions(packet)
            observation=self.encoder.encode(
                packet,self.session.stage,self.current_intent_action,
                self.last_action,self.last_commit_time,
                skill_phase=self.phase)
            action,logp,value=self.session.act(observation,mask.copy())
            action=int(action)
            if not 0<=action<len(ACTION_NAMES) or not mask[action]:
                raise ValueError(f'Policy selected illegal action {action}')
            self.records.append(dict(
                time=t,stage=self.session.stage,observation=observation,
                action_mask=mask,action=action,logp=float(logp),
                value=float(value),own_xy=list(packet['self']['position']),
                visual=self.encoder.last_details.copy()))
            if action and action!=self.current_intent_action:
                self.intent=dict(observation_id=len(self.records)-1,
                                 **self.request(action,packet))
                self.current_intent_action=action
                self.phase='approach'
                self.hold_started=None
                self.tag_done=False
                self.last_commit_time=t
            self.last_action=action
            self.samples=[]
        return IntentController.decide(self,packet)


class OnlineProbe(HandoffProbe):
    def is_peek_action(self,action):
        return action in (2,3)


def referee_telemetry(states, handoff_time, dt):
    """Evaluation-only exposure intervals with no positions or guard commands."""
    rows = []
    for stage in (0,1):
        trace = [row for row in states
                 if int(handoff_time is not None and row['t'] >= handoff_time) == stage]
        if not trace:
            continue
        masks = {
            'guard_visibility': lambda row: row['visible'][1],
            'valid_guard_lock': lambda row: row['lock_valid'][1],
        }
        report = dict(stage=stage)
        for name, valid in masks.items():
            intervals = []
            for row in trace:
                if not valid(row):
                    continue
                t = row['t']
                if intervals and t-intervals[-1][1] <= dt+1e-7:
                    intervals[-1][1] = t
                else:
                    intervals.append([t,t])
            report[name+'_intervals'] = intervals
            report[name+'_sampled_seconds'] = sum(max(0., b-a) for a,b in intervals)
        rows.append(report)
    return rows


def run_episode(first_seed,second_seed,policy,duration=48.,
                sensor=None,controller_class=OnlineController):
    encoder=VisualHistoryEncoder(duration,len(ACTION_NAMES),
                                  sensor or GeometryDetectionAdapter())
    session=OnlineSession(policy,encoder)
    probe=OnlineProbe(71000,session,start_a=(-8.,0.),duration=duration,
                      camera_half_fov=60.,alarm_patrol=False,
                      extra_covers=[],extra_waypoints=FIRST_POINTS)
    add_two_node_geometry(probe.cfg)
    probe.planner=Planner(probe.cfg,probe.cfg['arena']['covers'])
    probe.obstacles=boxes(probe.cfg['arena']['covers'])
    probe.rules.update(tag_range_m=4.,tag_seconds=.6,goal_radius_m=1.7)
    probe.controllers[1].rules.update(tag_range_m=4.,tag_seconds=.6)
    probe.clock.seconds=.6
    probe.goal=[10.,0.]
    probe.metadata['goal']=[10.,0.]
    probe.metadata['goal_radius']=1.7
    probe.metadata['rules'].update(tag_range_m=4.,tag_seconds=.6,
                                    goal_radius_m=1.7)
    points=probe.controllers[0].waypoints
    points.update(FIRST_POINTS,GOAL=[10.,0.],LOCAL_EXIT=[3.01,0.],
                  G1_S_GATE=[-5.,-1.6],G1_N_GATE=[-5.,1.6])
    WAYPOINTS.update(points)
    probe.controllers[0]=controller_class(probe.cfg,probe.rules,points,session)
    probe.controllers[0].chosen_gate=None
    probe.controllers[0].peek_hold_seconds=.4
    probe.second_initial_travel_yaw=float(probe.actor[1].travel_yaw)
    place_varied_guard(probe,first_seed,'first')
    probe.stage_policy=session
    session.probe=probe
    probe.second_seed=second_seed
    probe.handed_off=False
    probe.parallel_second=True
    probe.patrol_style='varied'
    probe.early_second_sightings=[]
    probe.shadow_guard=ShadowGuard(probe,second_seed,'varied')
    try:
        result=probe.run(f'online_{first_seed}_{second_seed}')
    except Exception as error:
        error.closed_options = [row for row in probe.controllers[0].records
                                if 'end_time' in row]
        if getattr(policy,'record_referee_telemetry',False):
            error.referee_telemetry = referee_telemetry(
                probe.trace, getattr(probe,'handoff_time',None), probe.rules['dt'])
        raise
    controller=probe.controllers[0]
    if hasattr(controller,'finalize_options'):
        controller.finalize_options(float(result['summary']['seconds']))
    records=controller.records
    visual_ticks=getattr(controller,'visual_tick_counts',None)
    summary=dict(first_seed=first_seed,second_seed=second_seed,
                 winner=result['summary']['winner'],
                 reason=result['summary']['reason'],
                 seconds=result['summary']['seconds'],
                 handoff_time=getattr(probe,'handoff_time',None),
                 early_second_visible_ticks=len(probe.early_second_sightings),
                 decisions=len(records),
                 first_visual_ticks=(visual_ticks[0] if visual_ticks is not None
                                     else sum(r['visual']['detected_now']
                                              for r in records if r['stage']==0)),
                 second_visual_ticks=(visual_ticks[1] if visual_ticks is not None
                                      else sum(r['visual']['detected_now']
                                               for r in records if r['stage']==1)),
                 smoke_used=result['summary']['smoke_used'])
    if hasattr(controller,'visual_sample_counts'):
        summary['visual_sample_counts']=controller.visual_sample_counts.copy()
    if getattr(policy,'record_referee_telemetry',False):
        summary['referee_telemetry'] = referee_telemetry(
            result['states'], summary['handoff_time'], probe.rules['dt'])
    return summary,records
