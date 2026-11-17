"""Formal first-zone screen with two legal peeks, route and wait actions."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'/'multi_decision_study_v2'))
from corridor_env import ACTIONS,CorridorController,CorridorProbe  # noqa: E402


class SouthDwellGuard:
    """Pause at a patrol endpoint while preserving formal AIM behavior."""
    def __init__(self,delegate,dwell,endpoint='south'):
        self.delegate=delegate
        self.dwell=float(dwell)
        self.endpoint=endpoint
        self.until=None
        self.completed=False

    def decide(self,packet):
        d=self.delegate
        t=packet['time'];own=packet['self'];visible=packet['visible']
        if visible is None:d.first_visible=None
        elif d.first_visible is None:d.first_visible=t
        ready=(visible is not None and t-d.first_visible>=d.rules['reaction_seconds']-1e-8)
        if ready and math.dist(own['position'],visible['position'])<=d.rules['tag_range_m']:
            return dict(mode='AIM',look=visible['position'])
        target=d.patrol[d.patrol_index]
        selected=(target[1]<-3. if self.endpoint=='south' else target[1]>3.)
        at_target=math.dist(own['position'],target)<.15
        if selected and at_target and self.dwell>0:
            if not self.completed:
                self.completed=True
                self.until=t+self.dwell
            if t<self.until-1e-8:
                memory=packet['memory']
                look=(memory['position'] if memory and t-memory['last_seen']<1.
                      else None)
                return dict(mode='WATCH',look=look)
        if not selected:self.completed=False
        return d.decide(packet)


class WaitController(CorridorController):
    action_names=ACTIONS+('WAIT',)

    def request(self,action,packet):
        if action==25:
            return dict(skill='WAIT',params=dict(posture='upright',look_degrees=0.))
        return super().request(action,packet)


class DualPeekRoute:
    def __init__(self,side,wait,duration,second_peek,repeat_peek):
        self.side=side
        self.wait=float(wait)
        self.duration=float(duration)
        self.second_peek=bool(second_peek)
        self.repeat_peek=bool(repeat_peek)
        self.phase='south_peek'
        self.reached_south=False
        self.reached_north=False
        self.track=[]
        self.approach_sightings=0
        self.south_sightings=0
        self.north_sightings=0
        self.choice_time=None
        self.used_north_peek=False
        self.used_repeat_peek=False
        self.repeat_until=None
        self.gate1_done=False

    def __call__(self,obs):
        x=float(obs[0])*12.;y=float(obs[1])*8.
        t=float(obs[8])*self.duration
        if self.phase!='advance' and obs[14]>.5:
            enemy=(x+float(obs[22])*24.,y+float(obs[23])*16.)
            self.track.append((round(t,2),round(enemy[0],3),round(enemy[1],3)))
            if self.phase=='north_cover':self.approach_sightings+=1
            if self.phase=='south_peek':self.south_sightings+=1
            if self.phase=='north_peek':self.north_sightings+=1
        own=(x,y)
        if self.phase=='south_peek':
            if math.dist(own,(-8.,-4.))<.2:self.reached_south=True
            if not (self.reached_south and math.dist(own,(-8.,-3.5))<.15):
                return 3,0.,0.
            if (self.repeat_peek and not self.used_repeat_peek
                    and self.south_sightings<2):
                self.used_repeat_peek=True
                self.repeat_until=t+.3
                self.phase='repeat_wait'
            else:
                self.phase=('choose' if self.south_sightings>=2 or not self.second_peek
                            else 'north_cover')
        if self.phase=='repeat_wait':
            if t<self.repeat_until-1e-8:return 25,0.,0.
            self.reached_south=False
            self.phase='south_peek'
            return 3,0.,0.
        if self.phase=='north_cover':
            if math.dist(own,(-8.,2.5))>.35:return 4,0.,0.
            self.phase='north_peek';self.used_north_peek=True
        if self.phase=='north_peek':
            if math.dist(own,(-8.,2.))<.2:self.reached_north=True
            if not (self.reached_north and math.dist(own,(-8.,2.5))<.15):
                return 5,0.,0.
            self.phase='choose'
        if self.phase=='choose':
            self.choice_time=t
            self.phase='wait'
        if self.phase=='wait':
            if t<self.choice_time+self.wait-1e-8:return 25,0.,0.
            self.phase='advance'
        gate=(-5.,3. if self.side=='north' else -3.)
        if math.dist(own,gate)<.35:self.gate1_done=True
        if not self.gate1_done:
            return (15 if self.side=='north' else 14),0.,0.
        return 16,0.,0.  # Matched south-side local exit after either first opening.


def place_guard(probe,seed,guard_x,south_dwell_max,balanced_dwell=False):
    rng=np.random.default_rng(int(seed))
    y=float(rng.uniform(-3.35,3.35))
    direction=1 if rng.integers(2) else -1
    speed=float(rng.uniform(.8,1.5))
    south_dwell=float(rng.uniform(0.,south_dwell_max))
    dwell_endpoint=('north' if rng.integers(2) else 'south') if balanced_dwell else 'south'
    guard=probe.actor[1]
    guard.xy=np.asarray([guard_x,y],float)
    guard.vmax=speed
    guard.yaw=math.pi/2 if direction>0 else -math.pi/2
    guard.navigate([])
    controller=probe.controllers[1]
    controller.patrol=[[guard_x,3.35],[guard_x,-3.35]] if direction>0 else [
        [guard_x,-3.35],[guard_x,3.35]]
    controller.patrol_index=0
    if south_dwell>0:
        probe.controllers[1]=SouthDwellGuard(controller,south_dwell,dwell_endpoint)
    probe.dwell_endpoint=dwell_endpoint
    return y,direction,speed,south_dwell


def run(seed,side,wait,duration,tag_range,second_peek,repeat_peek,guard_x,
        south_dwell_max):
    policy=DualPeekRoute(side,wait,duration,second_peek,repeat_peek)
    probe=CorridorProbe(71000,policy,start_a=(-8.,-3.5),duration=duration,
                        alarm_patrol=False)
    probe.rules['tag_range_m']=float(tag_range)
    probe.controllers[1].rules['tag_range_m']=float(tag_range)
    assert probe.controllers[1].rules['tag_range_m']==probe.rules['tag_range_m']
    points=probe.controllers[0].waypoints
    probe.controllers[0]=WaitController(probe.cfg,probe.rules,points,policy)
    initial_y,direction,speed,south_dwell=place_guard(
        probe,seed,guard_x,south_dwell_max)
    result=probe.run(f'local_decision_{seed}_{side}_{wait}')
    summary=result['summary']
    exit_time=next((state['t'] for state in result['states']
                    if state['xy'][0][0]>=-2.),None)
    tag_time=next((event['time'] for event in result['events']
                   if event['type']=='tag' and 1 in event['actors']),None)
    sampled_path=[(state['t'],[round(x,2) for x in state['xy'][0]])
                  for state in result['states']
                  if abs(state['t']-round(state['t']))<1e-8]
    return dict(seed=seed,side=side,wait=wait,tag_range=tag_range,
                guard_x=guard_x,south_dwell=round(south_dwell,3),
                second_peek=second_peek,
                start_y=round(initial_y,3),
                repeated_south_peek=policy.used_repeat_peek,
                direction=direction,speed=round(speed,3),
                winner=summary['winner'],reason=summary['reason'],
                local_survived=exit_time is not None and
                    (tag_time is None or exit_time<tag_time),
                exit_time=exit_time,tag_time=tag_time,
                seconds=summary['seconds'],choice_time=policy.choice_time,
                used_north_peek=policy.used_north_peek,
                south_sightings=policy.south_sightings,
                north_sightings=policy.north_sightings,
                approach_sightings=policy.approach_sightings,
                last_track=policy.track[-2:],
                sampled_path=sampled_path,
                gate1_time=summary['gate_crossing_times'][0])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed-start',type=int,default=94000)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--waits',nargs='+',type=float,default=(0.,))
    parser.add_argument('--sides',nargs='+',choices=('north','south'),
                        default=('north','south'))
    parser.add_argument('--duration',type=float,default=14.)
    parser.add_argument('--tag-range',type=float,default=3.)
    parser.add_argument('--guard-x',type=float,default=-3.9)
    parser.add_argument('--south-dwell-max',type=float,default=0.)
    parser.add_argument('--second-peek',action='store_true')
    parser.add_argument('--repeat-peek',action='store_true')
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    rows=[]
    for seed in range(args.seed_start,args.seed_start+args.count):
        for side in args.sides:
            for wait in args.waits:
                row=run(seed,side,wait,args.duration,args.tag_range,
                        args.second_peek,args.repeat_peek,args.guard_x,
                        args.south_dwell_max)
                rows.append(row)
                if not args.quiet:
                    print(json.dumps({k:v for k,v in row.items() if k!='sampled_path'}),flush=True)
    result=dict(status='formal_single_zone_decision_screen',
                warning='One guard and local goal only; not two-guard outcome',
                rows=rows,
                summary={f'{side}_wait{wait}':sum(r['local_survived'] for r in rows
                                                   if r['side']==side and r['wait']==wait)
                         for side in args.sides for wait in args.waits})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
