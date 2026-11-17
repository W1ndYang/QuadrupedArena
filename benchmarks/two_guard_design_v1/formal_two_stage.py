"""Continuous two-node Skill screen with an isolated guard handoff.

Both patrol states are independently sampled.  Guard 2 starts patrolling when
the attacker reaches its concealed observation room; the baffle geometry
prevents any observation of guard 2 before that point and of guard 1 after it.
This is a candidate rule, not the legacy simultaneous two-actor referee.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import formal_central_screen as central
from formal_central_audit import best_action,event_key,event_table,load_rows
from formal_central_knn_audit import choose
from formal_gate_decisions import place_guard
from formal_varied_knn_audit import choose as choose_varied
from two_node_geometry import add_two_node_geometry,SECOND_COVER
from varied_patrol import place as place_varied_guard

from env import WAYPOINTS,boxes,clear_ray
from fixed_goal_probe import PublicController
from humanoid_motion import HumanoidMover
from trial_motion import Planner


FIRST_POINTS={'CENTER_COVER':[-8.,0.],
              'CENTER_S_PEEK':[-8.,-2.1],
              'CENTER_N_PEEK':[-8.,2.1]}
SECOND_POINTS={'CENTER_COVER':[3.,0.],
               'CENTER_S_PEEK':[3.,-2.1],
               'CENTER_N_PEEK':[3.,2.1]}


def training(paths,stage):
    cases=load_rows(paths)
    expected=dict(route_y=1.6,tag_seconds=.6,tag_range=4.,
                  peek_offset=2.1,camera_half_fov=60.,
                  guard_x=7.1 if stage else -3.9,
                  middle_height=.6,local_goal=True,
                  peek_if_unseen=True)
    if stage:expected['peek_hold_seconds']=2.
    for actions in cases.values():
        sample=next(iter(actions.values()))
        if (any(sample.get(key)!=value for key,value in expected.items()) or
                bool(sample.get('stage2_local',False))!=bool(stage)):
            raise ValueError(f'Training sample does not match stage {stage} rules: '
                             f"seed {sample['seed']}")
    rows=[(seed,next(iter(actions.values())),actions)
          for seed,actions in cases.items()]
    styles={row.get('patrol_style','line') for _,row,_ in rows}
    peek_holds={row['peek_hold_seconds'] for _,row,_ in rows}
    if len(styles)!=1:raise ValueError('Mixed patrol styles in training')
    if len(peek_holds)!=1:raise ValueError('Mixed peek durations in training')
    fallback,events=event_table(cases)
    return dict(cases=cases,rows=rows,fixed=best_action(list(cases.values())),
                event_fallback=fallback,events=events,
                patrol_style=styles.pop(),peek_hold_seconds=peek_holds.pop())


class TwoStagePolicy:
    def __init__(self,duration,mode,first_training,second_training,
                 forced_second=None,forced_first=None,skip_peek=False):
        self.duration=duration;self.mode=mode
        self.forced_second=forced_second
        self.forced_first=forced_first
        self.skip_peek=skip_peek
        if skip_peek and mode!='fixed':
            raise ValueError('Unobserved baseline only supports fixed mode')
        self.training=(first_training,second_training)
        self.stage=0;self.handoff_time=None;self.probe=None
        self.actions=[None,None]
        self.scouts=[self._scout(0),None]

    def _scout(self,stage):
        if self.skip_peek:
            forced=self.forced_first if stage==0 else self.forced_second
            side,wait=forced or self.training[stage]['fixed']
            self.actions[stage]=(side,wait)
            return central.CentralScout(side,wait,self.duration,
                                        skip_peek=True,
                                        origin_x=-8. if stage==0 else 3.,
                                        route_y=1.6)
        return central.CentralScout('north',0.,self.duration,
                                   second_peek=False,peek_if_unseen=True,
                                   origin_x=-8. if stage==0 else 3.,
                                   route_y=1.6,
                                   decision_callback=lambda scout:
                                       self._decide(stage,scout))

    def _decide(self,stage,scout):
        offset=0. if stage==0 else self.handoff_time
        observation={name:[(round(t-offset,2),x,y) for t,x,y in getattr(scout,name)]
                     for name in ('south_track','north_track')}
        observation['south_sightings']=len(observation['south_track'])
        observation['north_sightings']=len(observation['north_track'])
        observation['stage2_local']=bool(stage)
        data=self.training[stage]
        stage_mode=self.mode
        observed_mode=('planar' if data['patrol_style']=='varied' else 'motion')
        if stage_mode=='first_only':stage_mode=observed_mode if stage==0 else 'fixed'
        if stage_mode=='second_only':stage_mode='fixed' if stage==0 else observed_mode
        if stage_mode in ('timing','position','motion','planar'):
            chooser=choose_varied if data['patrol_style']=='varied' else choose
            action=chooser(data['rows'],observation,stage_mode)
        elif stage_mode=='event':
            action=data['events'].get(event_key(observation),
                                      data['event_fallback'])
        else:action=data['fixed']
        if stage==0 and self.forced_first is not None:
            action=self.forced_first
        if stage==1 and self.forced_second is not None:
            action=self.forced_second
        self.actions[stage]=action
        if stage==1:
            side=action[0]
            destination=[10.,1.6 if side=='north' else -1.6]
            WAYPOINTS['LOCAL_EXIT']=destination
            self.probe.controllers[0].waypoints['LOCAL_EXIT']=destination
        return action

    def start_second(self,t):
        self.stage=1;self.handoff_time=t;self.scouts[1]=self._scout(1)
        if self.skip_peek:
            self.scouts[1].choice_time=t
            side=self.actions[1][0]
            destination=[10.,1.6 if side=='north' else -1.6]
            WAYPOINTS['LOCAL_EXIT']=destination
            self.probe.controllers[0].waypoints['LOCAL_EXIT']=destination

    def __call__(self,obs):
        return self.scouts[self.stage](obs)


class ShadowGuard:
    """Advance an unseen second patrol from t=0 with the formal guard motion."""
    def __init__(self,probe,seed,patrol_style='line'):
        original=probe.actor[1]
        self.actor=HumanoidMover([7.1,0.],
                                 probe.second_initial_travel_yaw,original.vmax,
                                 original.accel,original.turn)
        controller=PublicController('patrol',1,[],[7.1,0.],
                                    [[7.1,3.35],[7.1,-3.35]],probe.rules)
        holder=SimpleNamespace(actor=[None,self.actor],
                               controllers=[None,controller],dwell_endpoint=None)
        if patrol_style=='varied':
            self.private=place_varied_guard(holder,seed,'second')
        else:
            y,direction,speed,dwell=place_guard(holder,seed,7.1,4.,
                                                balanced_dwell=True)
            self.private=dict(y=y,direction=direction,speed=speed,dwell=dwell,
                              endpoint=holder.dwell_endpoint)
        self.controller=holder.controllers[1]
        self.planner=probe.planner
        self.rules=probe.rules
        self.t=0.
        self.move_goal=None

    def advance_until(self,target_t):
        dt=self.rules['dt']
        while self.t+dt/2<target_t:
            own=self.actor
            packet=dict(time=self.t,
                        self=dict(position=own.xy.tolist(),yaw=own.yaw,
                                  speed=own.v),visible=None,memory=None)
            command=self.controller.decide(packet)
            if command['mode']=='PATROL':
                target=command['target'];key=tuple(target)
                if self.move_goal!=key or not own.route:
                    route=self.planner.route(own.xy,target)
                    if route is None:raise RuntimeError('Shadow patrol route is blocked')
                    own.navigate(route);self.move_goal=key
            else:
                if own.route:own.navigate([])
                self.move_goal=None
            look_target=command.get('look')
            look=(math.atan2(look_target[1]-own.xy[1],
                             look_target[0]-own.xy[0])
                  if look_target is not None else
                  own.yaw+self.rules['scan_radps']*dt)
            previous=own.xy.copy()
            own.step(dt,look)
            if not self.planner.segment(previous,own.xy):
                raise RuntimeError('Shadow patrol crossed a wall')
            self.t=round(self.t+dt,8)


class HandoffProbe(central.CentralProbe):
    def observation(self,i,t):
        if i==0 and not getattr(self,'handed_off',False) and getattr(self,'parallel_second',False):
            self.shadow_guard.advance_until(t)
            other=self.shadow_guard.actor
            if (math.dist(self.actor[0].xy,other.xy)<=8. and
                    clear_ray((*self.actor[0].xy,.4),(*other.xy,1.05),
                              self.obstacles)):
                self.early_second_sightings.append(t)
        policy=getattr(self,'stage_policy',None)
        ready=(policy.ready_for_handoff(self.actor[0].xy)
               if policy is not None and hasattr(policy,'ready_for_handoff')
               else policy.scouts[0].gate1_done if policy is not None else False)
        if (i==0 and not getattr(self,'handed_off',False) and ready
                and math.dist(self.actor[0].xy,SECOND_COVER)<.15):
            self._handoff(t)
        return super().observation(i,t)

    def _handoff(self,t):
        self.handed_off=True
        self.stage_policy.start_second(t)
        WAYPOINTS.update(SECOND_POINTS)
        WAYPOINTS['G1_S_GATE']=[6.,-1.6]
        WAYPOINTS['G1_N_GATE']=[6.,1.6]
        self.controllers[0].waypoints.update(SECOND_POINTS)
        self.controllers[0].waypoints['G1_S_GATE']=[6.,-1.6]
        self.controllers[0].waypoints['G1_N_GATE']=[6.,1.6]
        training=getattr(self.stage_policy,'training',None)
        self.controllers[0].peek_hold_seconds=(
            2. if training is None else training[1]['peek_hold_seconds'])
        if getattr(self,'parallel_second',False):
            self.actor[1]=self.shadow_guard.actor
            self.controllers[1]=self.shadow_guard.controller
            self.second_private=dict(self.shadow_guard.private,
                                     handoff_y=float(self.actor[1].xy[1]))
            self.move_goal[1]=self.shadow_guard.move_goal
        else:
            old_guard=self.actor[1]
            self.actor[1]=HumanoidMover([7.1,0.],
                                        self.second_initial_travel_yaw,
                                        old_guard.vmax,
                                        old_guard.accel,old_guard.turn)
            self.controllers[1]=PublicController(
                'patrol',1,[],[7.1,0.],[[7.1,3.35],[7.1,-3.35]],self.rules)
            if getattr(self,'patrol_style','line')=='varied':
                self.second_private=place_varied_guard(self,self.second_seed,
                                                       'second')
            else:
                y,direction,speed,dwell=place_guard(
                    self,self.second_seed,7.1,4.,balanced_dwell=True)
                self.second_private=dict(y=y,direction=direction,speed=speed,
                                         dwell=dwell,endpoint=self.dwell_endpoint)
            self.move_goal[1]=None
        self.memory=[None,None]
        self.detected=[False,False]
        self.first_detection=[None,None]
        self.clock.age=[0.,0.]
        self.clock.previous=[False,False]
        self.handoff_time=t
        self.handoff_position=self.actor[0].xy.tolist()
        self.handoff_yaw=float(self.actor[0].yaw)
        self.handoff_camera_pan=float(self.camera_pan)


def run(first_seed,second_seed,mode,first_training,second_training,
        duration=36.,forced_second=None,parallel_second=False,
        patrol_style='line',forced_first=None,skip_peek=False):
    if (first_training['patrol_style']!=patrol_style or
            second_training['patrol_style']!=patrol_style):
        raise ValueError('Training patrol style differs from test patrol style')
    central.PEEK_OFFSET=2.1
    policy=TwoStagePolicy(duration,mode,first_training,second_training,
                          forced_second,forced_first,skip_peek)
    probe=HandoffProbe(71000,policy,start_a=(-8.,0.),duration=duration,
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
    probe.controllers[0]=central.CentralController(
        probe.cfg,probe.rules,points,policy)
    probe.controllers[0].peek_hold_seconds=first_training['peek_hold_seconds']
    probe.second_initial_travel_yaw=float(probe.actor[1].travel_yaw)
    if patrol_style=='varied':
        place_varied_guard(probe,first_seed,'first')
    else:place_guard(probe,first_seed,-3.9,4.,balanced_dwell=True)
    probe.stage_policy=policy;policy.probe=probe
    probe.second_seed=second_seed;probe.handed_off=False
    probe.parallel_second=parallel_second
    probe.patrol_style=patrol_style
    probe.early_second_sightings=[]
    if parallel_second:
        probe.shadow_guard=ShadowGuard(probe,second_seed,patrol_style)
    result=probe.run(f'two_stage_{first_seed}_{second_seed}_{mode}')
    handoff=getattr(probe,'handoff_time',None)
    second_choice=(None if policy.scouts[1] is None else
                   policy.scouts[1].choice_time)
    visible_after_handoff=([] if handoff is None or second_choice is None else
                           [(round(state['t'],2),
                             [round(v,2) for v in state['xy'][0]],
                             [round(v,2) for v in state['xy'][1]])
                            for state in result['states']
                            if handoff<=state['t']<=second_choice
                            and state['visible'][0]])
    return dict(first_seed=first_seed,second_seed=second_seed,mode=mode,
                patrol_style=patrol_style,
                skip_peek=skip_peek,
                parallel_second=parallel_second,
                early_second_visible_ticks=len(probe.early_second_sightings),
                winner=result['summary']['winner'],
                reason=result['summary']['reason'],
                seconds=result['summary']['seconds'],
                first_action=policy.actions[0],second_action=policy.actions[1],
                handoff_time=getattr(probe,'handoff_time',None),
                handoff_position=getattr(probe,'handoff_position',None),
                handoff_yaw=getattr(probe,'handoff_yaw',None),
                handoff_camera_pan=getattr(probe,'handoff_camera_pan',None),
                second_choice_time=second_choice,
                second_visible_ticks=len(visible_after_handoff),
                second_visible_samples=visible_after_handoff[::10],
                second_south_track=(None if policy.scouts[1] is None else
                                    policy.scouts[1].south_track),
                second_north_track=(None if policy.scouts[1] is None else
                                    policy.scouts[1].north_track),
                first_track_samples=(len(policy.scouts[0].south_track)+
                                     len(policy.scouts[0].north_track)),
                second_track_samples=(None if policy.scouts[1] is None else
                                      len(policy.scouts[1].south_track)+
                                      len(policy.scouts[1].north_track)),
                second_private=getattr(probe,'second_private',None))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--first-training',nargs='+',required=True,type=Path)
    parser.add_argument('--second-training',nargs='+',required=True,type=Path)
    parser.add_argument('--first-seed-start',type=int,default=109000)
    parser.add_argument('--first-seed-fixed',type=int)
    parser.add_argument('--second-seed-start',type=int,default=219000)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--modes',nargs='+',choices=(
        'fixed','event','timing','position','motion','planar',
        'first_only','second_only'),
        default=('motion',))
    parser.add_argument('--duration',type=float,default=36.)
    parser.add_argument('--force-second-side',choices=('north','south'))
    parser.add_argument('--force-second-wait',type=float,default=0.)
    parser.add_argument('--force-first-side',choices=('north','south'))
    parser.add_argument('--force-first-wait',type=float,default=0.)
    parser.add_argument('--parallel-second',action='store_true')
    parser.add_argument('--skip-peek',action='store_true')
    parser.add_argument('--patrol-style',choices=('line','varied'),default='line')
    parser.add_argument('--progress',action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    first=training(args.first_training,0)
    second=training(args.second_training,1)
    forced_second=(None if args.force_second_side is None else
                   (args.force_second_side,args.force_second_wait))
    forced_first=(None if args.force_first_side is None else
                  (args.force_first_side,args.force_first_wait))
    rows=[]
    for offset in range(args.count):
        for mode in args.modes:
            first_seed=(args.first_seed_fixed if args.first_seed_fixed is not None
                        else args.first_seed_start+offset)
            rows.append(run(first_seed,
                            args.second_seed_start+offset,mode,first,second,
                            args.duration,forced_second,args.parallel_second,
                            args.patrol_style,forced_first,args.skip_peek))
        if args.progress:
            print(f'completed {offset+1}/{args.count} seed pairs',flush=True)
    result=dict(status='isolated_guard_handoff_two_stage_formal_screen',
                warning=('Guard 2 patrols unseen from t=0, then replaces guard 1 at the room; no simultaneous three-actor referee'
                         if args.parallel_second else
                         'Guard 2 private patrol phase is sampled at room entry; no simultaneous three-actor engine'),
                parallel_second=args.parallel_second,
                patrol_style=args.patrol_style,
                forced_second=forced_second,
                forced_first=forced_first,
                skip_peek=args.skip_peek,
                counts={mode:sum(r['winner']=='attacker' for r in rows if r['mode']==mode)
                        for mode in args.modes},rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()
