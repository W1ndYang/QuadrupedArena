"""Formal local gate screen with a central two-sided observation shelter."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'/'multi_decision_study_v2'))
from corridor_env import ACTIONS,CorridorController,CorridorProbe  # noqa: E402
from corridor_geometry import cover  # noqa: E402
from env import WAYPOINTS,boxes  # noqa: E402
from formal_gate_decisions import place_guard  # noqa: E402
from two_node_geometry import add_two_node_geometry,SECOND_COVER  # noqa: E402
from varied_patrol import place as place_varied_guard  # noqa: E402
from trial_motion import Planner  # noqa: E402
from fixed_goal_probe import PublicController  # noqa: E402


PEEK_OFFSET=.55
EXTRA_POINTS={
    'CENTER_COVER':[-8.,0.],
    'CENTER_S_PEEK':[-8.,-PEEK_OFFSET],
    'CENTER_N_PEEK':[-8.,PEEK_OFFSET],
}
EXTRA_ACTIONS=ACTIONS+('CENTER_COVER','CENTER_S_PEEK','CENTER_N_PEEK',
                       'WAIT','LOCAL_EXIT')


class CentralController(CorridorController):
    action_names=EXTRA_ACTIONS

    def request(self,action,packet):
        if action in (20,21):
            target='G1_S_GATE' if action==20 else 'G1_N_GATE'
            return dict(skill='SMOKE_AND_MOVE',params=dict(
                location=self.smoke_location_for(packet,target),target=target,
                posture='upright'))
        if action==25:
            return dict(skill='TAKE_COVER',params=dict(target='CENTER_COVER',
                                                      posture='upright'))
        if action in (26,27):
            return dict(skill='PEEK',params=dict(
                target='CENTER_S_PEEK' if action==26 else 'CENTER_N_PEEK',
                return_to='CENTER_COVER',posture='upright',
                hold_seconds=getattr(self,'peek_hold_seconds',.4)))
        if action==28:
            return dict(skill='WAIT',params=dict(posture='upright',look_degrees=0.))
        if action==29:
            return dict(skill='MOVE',params=dict(target='LOCAL_EXIT',
                                                 posture='upright',fire='off'))
        return super().request(action,packet)


class CentralProbe(CorridorProbe):
    def is_peek_action(self,action):
        return action in (26,27) or super().is_peek_action(action)


class HypothesisGuardController(PublicController):
    """Execute one LLM-supplied path once; retain the formal AIM transition.

    Only behavior selection changes here. Sight, motion, aiming and lock are
    still computed by the existing formal Probe and HumanoidMover.
    """
    def __init__(self, rules, start, waypoints, dwell_node_indices, dwell_seconds):
        super().__init__('patrol', 1, [], start, waypoints, rules)
        self.index = 0
        self.dwell_node_indices = set(dwell_node_indices)
        self.dwell_seconds = float(dwell_seconds)
        self.dwell_until = None

    def decide(self, packet):
        t, own, visible = packet['time'], packet['self'], packet['visible']
        if visible is None:
            self.first_visible = None
        elif self.first_visible is None:
            self.first_visible = t
        ready = (visible is not None and
                 t-self.first_visible >= self.rules['reaction_seconds']-1e-8)
        if ready and math.dist(own['position'], visible['position']) <= self.rules['tag_range_m']:
            return dict(mode='AIM', look=visible['position'])
        memory = packet['memory']
        look = (memory['position'] if memory and
                t-memory['last_seen'] < 1. else None)
        if self.dwell_until is not None:
            if t < self.dwell_until-1e-8:
                return dict(mode='WATCH', look=look)
            self.dwell_until = None
        if self.index < len(self.patrol) and math.dist(own['position'], self.patrol[self.index]) < .15:
            arrived = self.index
            self.index += 1
            if arrived in self.dwell_node_indices and self.dwell_seconds > 0.:
                self.dwell_until = t+self.dwell_seconds
                return dict(mode='WATCH', look=look)
        if self.index < len(self.patrol):
            return dict(mode='PATROL', target=self.patrol[self.index], look=look)
        return dict(mode='WATCH', look=look)


class CentralScout:
    def __init__(self,side,wait,duration,second_peek=True,skip_peek=False,
                 route_y=3.,peek_if_unseen=False,origin_x=-8.,
                 decision_callback=None,use_smoke=False):
        self.side=side;self.wait=float(wait);self.duration=float(duration)
        self.route_y=float(route_y)
        self.origin_x=float(origin_x)
        self.decision_callback=decision_callback
        self.use_smoke=bool(use_smoke)
        self.second_peek=second_peek
        self.peek_if_unseen=peek_if_unseen
        self.skip_peek=skip_peek
        self.phase='wait' if skip_peek else 'south_peek'
        self.reached_south=False;self.reached_north=False
        self.south_track=[];self.north_track=[];self.cover_sightings=0
        self.south_track_checkpoint=0
        self.choice_time=0. if skip_peek else None
        self.final_decision=bool(skip_peek)
        self.gate1_done=False

    def __call__(self,obs):
        x=float(obs[0])*12.;y=float(obs[1])*8.;t=float(obs[8])*self.duration
        own=(x,y)
        if obs[14]>.5 and self.phase!='advance':
            sight=(round(t,2),round(x+float(obs[22])*24.,3),
                   round(y+float(obs[23])*16.,3))
            if self.phase=='south_peek':self.south_track.append(sight)
            elif self.phase=='north_peek':self.north_track.append(sight)
            else:self.cover_sightings+=1
        if self.phase=='south_peek':
            if math.dist(own,(self.origin_x,-PEEK_OFFSET))<.19:self.reached_south=True
            if not(self.reached_south and math.dist(own,(self.origin_x,0.))<.15):
                return 26,0.,0.
            need_north=(self.second_peek or
                        (self.peek_if_unseen and
                         len(self.south_track)==self.south_track_checkpoint))
            if not need_north:
                self._finish_observation(t)
            else:self.phase='north_peek'
        if self.phase=='north_peek':
            if math.dist(own,(self.origin_x,PEEK_OFFSET))<.19:self.reached_north=True
            if not(self.reached_north and math.dist(own,(self.origin_x,0.))<.15):
                return 27,0.,0.
            self._finish_observation(t)
        if self.phase=='wait':
            if t<self.choice_time+self.wait-1e-8:return 28,0.,0.
            self.phase='advance'
        gate=(self.origin_x+3.,
              self.route_y if self.side=='north' else -self.route_y)
        if math.dist(own,gate)<.35:self.gate1_done=True
        if not self.gate1_done:
            action=(21 if self.side=='north' else 20) if self.use_smoke else (
                    15 if self.side=='north' else 14)
            return action,0.,0.
        if self.route_y!=3.:return 29,0.,0.
        return (17 if self.side=='north' else 16),0.,0.

    def _finish_observation(self,t):
        self.choice_time=t
        self.final_decision=True
        if self.decision_callback is not None:
            self.side,self.wait=self.decision_callback(self)
        self.phase='wait'


class RepeatScout(CentralScout):
    """Make one timed, independent revisit to the shelter's observation peek."""
    def __init__(self,*args,repeek_gap,**kwargs):
        super().__init__(*args,**kwargs)
        if self.skip_peek:raise ValueError('Repeat observation requires a peek')
        self.repeek_gap=float(repeek_gap)
        self.requested_wait=self.wait
        self.pending_repeek=False
        self.repeek_done=False

    def _finish_observation(self,t):
        if not self.repeek_done:
            self.choice_time=t
            self.wait=self.repeek_gap
            self.phase='wait'
            self.pending_repeek=True
            self.repeek_done=True
            return
        super()._finish_observation(t)

    def __call__(self,obs):
        t=float(obs[8])*self.duration
        if (self.pending_repeek and self.phase=='wait' and
                t>=self.choice_time+self.repeek_gap-1e-8):
            self.pending_repeek=False
            self.phase='south_peek'
            self.reached_south=False
            self.reached_north=False
            self.south_track_checkpoint=len(self.south_track)
            self.wait=self.requested_wait
        return super().__call__(obs)


def run(seed,side,wait,duration,tag_range,guard_x,south_dwell_max,
        middle_height,local_goal,balanced_dwell=False,second_peek=True,
        camera_half_fov=45.,skip_peek=False,route_y=3.,tag_seconds=1.,
        peek_if_unseen=False,transfer_goal=False,stage2_local=False,
        peek_hold_seconds=.4,patrol_style='line',use_smoke=False,
        repeek_gap=None,hypothesis_guard=None):
    assert not(transfer_goal and stage2_local)
    origin_x=SECOND_COVER[0] if stage2_local else -8.
    scout_class=RepeatScout if repeek_gap is not None else CentralScout
    scout_kwargs={'repeek_gap':repeek_gap} if repeek_gap is not None else {}
    policy=scout_class(side,wait,duration,second_peek,skip_peek,route_y,
                       peek_if_unseen,origin_x,use_smoke=use_smoke,
                       **scout_kwargs)
    extra_points={
        'CENTER_COVER':[origin_x,0.],
        'CENTER_S_PEEK':[origin_x,-PEEK_OFFSET],
        'CENTER_N_PEEK':[origin_x,PEEK_OFFSET],
    }
    probe=CentralProbe(71000,policy,start_a=(origin_x,0.),duration=duration,
                       camera_half_fov=camera_half_fov,
                       alarm_patrol=False,
        extra_covers=([] if transfer_goal or stage2_local or
                         patrol_style in ('varied','hypothesis') else
                                     [cover('CentralObserver',-7.3,0.,.5,1.1)]),
                       extra_waypoints=extra_points)
    if transfer_goal or stage2_local or patrol_style in ('varied','hypothesis'):
        assert local_goal and route_y==1.6
        add_two_node_geometry(probe.cfg)
    if route_y!=3.:
        assert route_y in (2.,1.6),'Only checked narrow openings are supported'
        for label in ('lower','upper'):
            wall=next(c for c in probe.cfg['arena']['covers']
                      if c['name']==f'Gate1_{label}')
            center=(route_y+9.)/2.
            wall['center'][1]=(-center if label=='lower' else center)
            wall['size'][1]=7.-route_y
        middle=next(c for c in probe.cfg['arena']['covers']
                    if c['name']=='Gate1_middle')
        middle['size'][1]=2.*(route_y-1.)
    if middle_height!=2.4 or route_y!=3.:
        middle=next(c for c in probe.cfg['arena']['covers']
                    if c['name']=='Gate1_middle')
        middle['size'][2]=float(middle_height)
        probe.planner=Planner(probe.cfg,probe.cfg['arena']['covers'])
        probe.obstacles=boxes(probe.cfg['arena']['covers'])
    probe.rules['tag_range_m']=float(tag_range)
    probe.controllers[1].rules['tag_range_m']=float(tag_range)
    assert probe.controllers[1].rules['tag_range_m']==probe.rules['tag_range_m']
    probe.rules['tag_seconds']=float(tag_seconds)
    probe.controllers[1].rules['tag_seconds']=float(tag_seconds)
    probe.clock.seconds=float(tag_seconds)
    assert probe.clock.seconds==probe.rules['tag_seconds']
    probe.metadata['rules']['tag_range_m']=float(tag_range)
    probe.metadata['rules']['tag_seconds']=float(tag_seconds)
    points=probe.controllers[0].waypoints
    points.update(extra_points)
    WAYPOINTS.update(extra_points)
    gate_x=origin_x+3.
    points['G1_S_GATE']=[gate_x,-route_y]
    points['G1_N_GATE']=[gate_x,route_y]
    WAYPOINTS['G1_S_GATE']=[gate_x,-route_y]
    WAYPOINTS['G1_N_GATE']=[gate_x,route_y]
    if local_goal:
        goal=(list(SECOND_COVER) if transfer_goal else
              [origin_x+7.,route_y if side=='north' else -route_y])
        probe.goal=goal
        points['GOAL']=goal
        WAYPOINTS['GOAL']=goal
        exit_target=([SECOND_COVER[0]+.01,SECOND_COVER[1]]
                     if transfer_goal else goal)
        points['LOCAL_EXIT']=exit_target
        WAYPOINTS['LOCAL_EXIT']=exit_target
        if transfer_goal:
            probe.rules['goal_radius_m']=.35
            probe.metadata['goal_radius']=.35
            probe.metadata['rules']['goal_radius_m']=.35
    probe.controllers[0]=CentralController(probe.cfg,probe.rules,points,policy)
    probe.controllers[0].peek_hold_seconds=float(peek_hold_seconds)
    if patrol_style=='hypothesis':
        # Match OnlineController: a synthetic referee sighting must not guide
        # the actual robot's smoke location at this public decision boundary.
        probe.controllers[0].smoke_uses_track=False
    if patrol_style=='line':
        y,direction,speed,dwell=place_guard(probe,seed,guard_x,south_dwell_max,
                                            balanced_dwell=balanced_dwell)
        private_patrol=None
    elif patrol_style=='varied':
        private_patrol=place_varied_guard(
            probe,seed,'second' if stage2_local else 'first')
        y=private_patrol['start'][1]
        direction=(1 if private_patrol['patrol'][0][1]>y else -1)
        speed=private_patrol['speed'];dwell=private_patrol['dwell']
        probe.dwell_endpoint=private_patrol['dwell_endpoint']
    elif patrol_style=='hypothesis':
        if hypothesis_guard is None:
            raise ValueError('hypothesis_guard required')
        guard=probe.actor[1]
        guard.xy=np.asarray(hypothesis_guard['start_xy'],float)
        guard.yaw=float(hypothesis_guard['start_yaw'])
        guard.travel_yaw=guard.yaw
        guard.vmax=float(hypothesis_guard['speed_mps'])
        guard.v=float(hypothesis_guard.get('initial_speed',0.))
        guard.omega=0.
        guard.navigate([])
        points=hypothesis_guard['waypoints']
        if not points or not all(probe.planner.free(p) for p in points):
            raise ValueError('Hypothesized guard path is blocked or empty')
        probe.controllers[1]=HypothesisGuardController(
            probe.rules,guard.xy.tolist(),points,
            hypothesis_guard.get('dwell_node_indices',[]),
            hypothesis_guard.get('dwell_seconds',0.))
        private_patrol=None
        y=guard.xy[1]
        direction=(1 if points[0][1]>y else -1)
        speed=guard.vmax
        dwell=float(hypothesis_guard.get('dwell_seconds',0.))
        probe.dwell_endpoint=None
    else:raise ValueError(patrol_style)
    try:
        result=probe.run(f'central_{seed}_{side}_{wait}')
    except AssertionError as exc:
        raise RuntimeError(
            f'Formal clearance invariant failed: seed={seed} side={side} '
            f'wait={wait} patrol_style={patrol_style} '
            f'minimum_clearance={probe.minimum_clearance:.6f} '
            f'required={probe.cfg["planner_clearance"]:.6f} '
            f'guard_xy={probe.actor[1].xy.tolist()} '
            f'attacker_xy={probe.actor[0].xy.tolist()}') from exc
    summary=result['summary']
    guard_trace=[(state['t'],[round(v,3) for v in state['xy'][1]],
                  state['modes'][1]) for state in result['states']
                 if abs(state['t']-round(state['t'],1))<1e-8
                 and state['t']<=5.]
    attacker_trace=[(state['t'],[round(v,3) for v in state['xy'][0]])
                    for state in result['states']
                    if abs(state['t']*2.-round(state['t']*2.))<1e-8]
    exit_time=next((state['t'] for state in result['states']
                    if state['xy'][0][0]>=-2.),None)
    tag_time=next((event['time'] for event in result['events']
                   if event['type']=='tag' and 1 in event['actors']),None)
    return dict(seed=seed,side=side,wait=wait,guard_x=guard_x,
                middle_height=middle_height,peek_offset=PEEK_OFFSET,
                route_y=route_y,
                local_goal=local_goal,second_peek=second_peek,
                transfer_goal=transfer_goal,
                stage2_local=stage2_local,
                patrol_style=patrol_style,private_patrol=private_patrol,
                repeek_gap=repeek_gap,
                use_smoke=use_smoke,smoke_used=summary['smoke_used'],
                peek_if_unseen=peek_if_unseen,
                skip_peek=skip_peek,
                camera_half_fov=camera_half_fov,
                peek_hold_seconds=peek_hold_seconds,
                tag_range=tag_range,initial_y=round(y,3),direction=direction,
                tag_seconds=tag_seconds,
                guard_speed=round(speed,3),endpoint_dwell=round(dwell,3),
                dwell_endpoint=probe.dwell_endpoint,
                south_sightings=len(policy.south_track),
                north_sightings=len(policy.north_track),
                cover_sightings=policy.cover_sightings,
                guard_trace=guard_trace,
                attacker_trace=attacker_trace,
                south_track=policy.south_track,north_track=policy.north_track,
                choice_time=(policy.choice_time if policy.final_decision else None),
                exit_time=exit_time,tag_time=tag_time,
                local_survived=(summary['winner']=='attacker' if local_goal else
                                exit_time is not None and
                                (tag_time is None or exit_time<tag_time)),
                reason=summary['reason'],seconds=summary['seconds'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed-start',type=int,default=97000)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--sides',nargs='+',choices=('north','south'),
                        default=('north','south'))
    parser.add_argument('--waits',nargs='+',type=float,default=(0.,))
    parser.add_argument('--duration',type=float,default=14.)
    parser.add_argument('--tag-range',type=float,default=3.)
    parser.add_argument('--tag-seconds',type=float,default=1.)
    parser.add_argument('--guard-x',type=float,default=-3.9)
    parser.add_argument('--south-dwell-max',type=float,default=4.)
    parser.add_argument('--balanced-dwell',action='store_true')
    parser.add_argument('--single-peek',action='store_true')
    parser.add_argument('--peek-if-unseen',action='store_true')
    parser.add_argument('--skip-peek',action='store_true')
    parser.add_argument('--camera-half-fov',type=float,default=45.)
    parser.add_argument('--peek-hold-seconds',type=float,default=.4)
    parser.add_argument('--middle-height',type=float,default=2.4)
    parser.add_argument('--peek-offset',type=float,default=.55)
    parser.add_argument('--route-y',type=float,default=3.)
    parser.add_argument('--local-goal',action='store_true')
    parser.add_argument('--transfer-goal',action='store_true')
    parser.add_argument('--stage2-local',action='store_true')
    parser.add_argument('--patrol-style',choices=('line','varied'),default='line')
    parser.add_argument('--use-smoke',action='store_true')
    parser.add_argument('--repeek-gap',type=float)
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    global PEEK_OFFSET
    PEEK_OFFSET=args.peek_offset
    EXTRA_POINTS['CENTER_S_PEEK'][1]=-PEEK_OFFSET
    EXTRA_POINTS['CENTER_N_PEEK'][1]=PEEK_OFFSET
    rows=[]
    for seed in range(args.seed_start,args.seed_start+args.count):
        for side in args.sides:
            for wait in args.waits:
                row=run(seed,side,wait,args.duration,args.tag_range,
                        args.guard_x,args.south_dwell_max,args.middle_height,
                        args.local_goal,args.balanced_dwell,
                        second_peek=not args.single_peek,
                        camera_half_fov=args.camera_half_fov,
                        skip_peek=args.skip_peek,route_y=args.route_y,
                        tag_seconds=args.tag_seconds,
                        peek_if_unseen=args.peek_if_unseen,
                        transfer_goal=args.transfer_goal,
                        stage2_local=args.stage2_local,
                        peek_hold_seconds=args.peek_hold_seconds,
                        patrol_style=args.patrol_style,
                        use_smoke=args.use_smoke,
                        repeek_gap=args.repeek_gap)
                rows.append(row)
                if not args.quiet:print(json.dumps(row),flush=True)
    summary={f'{side}_wait{wait}':sum(r['local_survived'] for r in rows
                                     if r['side']==side and r['wait']==wait)
             for side in args.sides for wait in args.waits}
    result=dict(status='formal_single_zone_central_observer_screen',
                warning='One guard and local exit only; not two-guard outcome',
                summary=summary,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
