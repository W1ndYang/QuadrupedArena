"""Three sequential goals over the frozen fixed-goal Skill/referee mechanics.

This is a rule-search pilot. The defender chooses a fresh private patrol
destination at each public goal transition. The attacker may request any of
the ten formal Skills at every 10 Hz decision throughout the match.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'benchmarks/fixed_goal_rl_v1'))
from env import RLProbe,WAYPOINTS,load_case  # noqa: E402
sys.path.insert(0,str(PROJECT/'benchmarks/fixed_goal_v041'))
from run_goal_only import GoalOnlyClock  # noqa: E402


GOAL_SEQUENCE=('CENTRAL_SOUTH','CENTRAL_NORTH','GOAL')
DEFENDER_DESTINATIONS=(('SOUTH_ENTRY','EAST_COURT_WEST','NORTH_WINDOW'),
                       ('CENTRAL_SOUTH','CENTRAL_NORTH','EAST_COURT_WEST'),
                       ('SOUTH_ENTRY','EAST_COURT_WEST','GOAL'))


class SequenceClock(GoalOnlyClock):
    def __init__(self,goal_count,on_transition,hold_seconds=0.):
        super().__init__()
        self.goal_count=goal_count;self.stage=0;self.on_transition=on_transition
        self.hold_seconds=float(hold_seconds);self.dwell=0.

    def step(self,dt,valid,goal=False,timeout=False):
        self.dwell=self.dwell+dt if goal else 0.
        complete=goal and self.dwell+1e-9>=self.hold_seconds
        final=complete and self.stage==self.goal_count-1
        winner,reason,hits=super().step(dt,valid,final,timeout)
        if winner is not None:return winner,reason,hits
        if complete:
            self.stage+=1
            self.dwell=0.
            self.on_transition(self.stage)
        return None,None,hits


class SequenceProbe(RLProbe):
    def __init__(self,layout,seed,policy,duration=30.,attacker_start=None,
                 defender_start=None,defender_speed=None,
                 hold_seconds=0.,goal_names=GOAL_SEQUENCE,
                 defender_destinations=DEFENDER_DESTINATIONS,defender_yaw=None):
        base,rules,waypoints=load_case(layout,seed)
        if attacker_start is not None:
            base['start_a']=list(attacker_start)
            waypoints['SPAWN']=list(attacker_start)
        if defender_start is not None:base['start_b']=list(defender_start)
        if defender_speed is not None:base['max_speed'][1]=float(defender_speed)
        if defender_yaw is not None:base['guard_yaw']=float(defender_yaw)
        waypoints=copy.deepcopy(waypoints)
        if len(goal_names)!=len(defender_destinations):
            raise ValueError('Every goal stage needs defender destinations')
        goals=[list(waypoints[name]) for name in goal_names]
        waypoints['GOAL']=goals[0]
        rules['duration']=float(duration)
        super().__init__(base,rules,waypoints,policy)
        self.goal_sequence=goals
        self.goal_names=tuple(goal_names)
        self.defender_destinations=tuple(tuple(stage) for stage in defender_destinations)
        self.goal=goals[0]
        self.patrol_rng=np.random.default_rng(np.random.SeedSequence([0 if seed is None else seed,5971]))
        self.patrol_history=[]
        self.clock=SequenceClock(len(goals),self.set_stage,hold_seconds)
        self.set_stage(0)

    def set_stage(self,stage):
        if stage<len(self.goal_sequence):
            self.goal=list(self.goal_sequence[stage])
            self.controllers[0].waypoints['GOAL']=list(self.goal)
            WAYPOINTS['GOAL']=list(self.goal)
            options=self.defender_destinations[stage]
            choice=int(self.patrol_rng.integers(len(options)))
            destination=options[choice]
            route_names=(destination,) if isinstance(destination,str) else tuple(destination)
            defender=self.controllers[1]
            defender.patrol=[self.actor[1].xy.tolist(),
                             *(list(WAYPOINTS[name]) for name in route_names)]
            defender.patrol_index=0
            self.patrol_history.append(dict(stage=stage,time=self.clock.time,
                                            destination=route_names,choice=choice))

    def run(self,identifier):
        result=super().run(identifier)
        result['summary'].update(goals_completed=self.clock.stage+int(result['summary']['winner']=='attacker'),
                                 patrol_history=self.patrol_history,
                                 goal_sequence=self.goal_sequence,goal_names=self.goal_names)
        return result


def policy_factory(name):
    if name=='direct':return lambda obs:(1,0.,0.)
    if name=='tag_and_goal':return lambda obs:(7,0.,0.)
    if name=='smoke_and_goal':return lambda obs:(9,0.,0.)
    if name=='reactive_tag':return lambda obs:(7 if obs[14]>.5 else 1,0.,0.)
    if name=='reactive_smoke':return lambda obs:(9 if obs[14]>.5 else 1,0.,0.)
    if name=='reactive_low':return lambda obs:(6 if obs[14]>.5 else 1,0.,0.)
    if name=='smoke_then_low':
        return lambda obs:(9 if obs[14]>.5 and obs[12]>.5 else
                           6 if obs[14]>.5 else 1,0.,0.)
    if name=='smoke_then_tag':
        return lambda obs:(9 if obs[14]>.5 and obs[12]>.5 else
                           7 if obs[14]>.5 else 1,0.,0.)
    raise ValueError(name)


def episode(layout,seed,name,duration=30.,defender_start=None,defender_speed=None,
            hold_seconds=0.,goal_names=GOAL_SEQUENCE,
            defender_destinations=DEFENDER_DESTINATIONS):
    probe=SequenceProbe(layout,seed,policy_factory(name),duration,None,defender_start,
                        defender_speed,hold_seconds,goal_names,defender_destinations)
    result=probe.run(f'sequence_{layout}_{seed}_{name}')
    summary=result['summary']
    records=probe.controllers[0].records
    return dict(layout=layout,seed=seed,policy=name,defender_start=defender_start,
                defender_speed=defender_speed,hold_seconds=hold_seconds,
                goal_names=summary['goal_names'],winner=summary['winner'],
                reason=summary['reason'],seconds=summary['seconds'],
                goals_completed=summary['goals_completed'],
                patrol_history=summary['patrol_history'],
                actor_decisions=len(records),legal_sighting_steps=sum(row['visible'] for row in records),
                actor_switches=sum(row['action']!=records[i-1]['action'] for i,row in enumerate(records) if i))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[51001,51002,51003])
    parser.add_argument('--duration',type=float,default=30.)
    args=parser.parse_args()
    rows=[]
    for seed in args.seeds:
        for name in ('direct','tag_and_goal','smoke_and_goal','reactive_tag'):
            row=episode('SE',seed,name,args.duration)
            rows.append(row);print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status='exploratory_pilot',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
