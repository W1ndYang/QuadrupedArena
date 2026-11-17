"""Matched-policy audit for the formal single-zone central-observer screen."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

ACTIONS=tuple((side,float(wait)) for side in ('north','south')
              for wait in (0,1,2))


def load_rows(paths):
    rows=[]
    for path in paths:rows+=json.loads(path.read_text())['rows']
    by_seed=defaultdict(dict)
    for row in rows:
        action=(row['side'],float(row['wait']))
        if action in by_seed[row['seed']]:raise ValueError('Duplicate action')
        by_seed[row['seed']][action]=row
    for seed,actions in by_seed.items():
        if set(actions)!=set(ACTIONS):raise ValueError(f'Incomplete seed {seed}')
        prefixes={(tuple(map(tuple,row['south_track'])),
                   tuple(map(tuple,row['north_track'])),row['choice_time'])
                  for row in actions.values()}
        if len(prefixes)!=1:raise ValueError(f'Unmatched observation prefix {seed}')
    return dict(by_seed)


def event_key(row):
    return (bool(row['south_sightings']),bool(row['north_sightings']))


def latest(track):
    if len(track)<2:return None
    a,b=track[-2:]
    return b[2],(b[2]-a[2])/max(b[0]-a[0],1e-9)


def motion_action(row):
    """Exploratory hand-written rule using legal tracks; no performance claim."""
    north=latest(row['north_track'])
    south=latest(row['south_track'])
    if north is not None:
        return ('south',0.) if north[1]>.4 else ('north',1.)
    if south is not None:
        if south[0]<-2.5 and abs(south[1])<.25:return ('south',0.)
        if south[1]<-.3:return ('north',0.)
        return ('south',0.)
    return ('north',0.)


def best_action(cases):
    ranked=[]
    for action in ACTIONS:
        wins=sum(rows[action]['local_survived'] for rows in cases)
        ranked.append((wins,-action[1],action))
    return max(ranked)[2]


def event_table(design):
    groups=defaultdict(list)
    for actions in design.values():
        groups[event_key(next(iter(actions.values())))].append(actions)
    fallback=best_action(list(design.values()))
    return fallback,{key:best_action(cases) for key,cases in groups.items()}


def event_conflicts(cases):
    """Check whether one action can solve every state sharing an event pattern."""
    groups=defaultdict(list)
    for seed,actions in cases.items():
        groups[event_key(next(iter(actions.values())))].append((seed,actions))
    result={}
    for key,members in groups.items():
        winning_sets=[(seed,{a for a,r in actions.items() if r['local_survived']})
                      for seed,actions in members]
        universal=set(ACTIONS)
        for _,wins in winning_sets:universal &= wins
        pair=next(((s1,s2) for i,(s1,w1) in enumerate(winning_sets)
                   for s2,w2 in winning_sets[i+1:] if not (w1 & w2)),None)
        result[str(key)]=dict(episodes=len(members),
                              common_winning_actions=sorted(universal),
                              disjoint_winner_pair=pair)
    return result


def audit(cases,fallback,events):
    totals=Counter()
    rows=[]
    route_cases=Counter()
    for seed,actions in sorted(cases.items()):
        observation=next(iter(actions.values()))
        event_action=events.get(event_key(observation),fallback)
        motion=motion_action(observation)
        wins={action:bool(row['local_survived']) for action,row in actions.items()}
        north=any(wins[('north',wait)] for wait in (0.,1.,2.))
        south=any(wins[('south',wait)] for wait in (0.,1.,2.))
        route_case=('both' if north and south else 'north_only' if north
                    else 'south_only' if south else 'neither')
        route_cases[route_case]+=1
        wait_needed=(any(wins.values()) and
                     not wins[('north',0.)] and not wins[('south',0.)])
        totals['episodes']+=1
        totals['oracle']+=any(wins.values())
        totals['fixed']+=wins[fallback]
        totals['event']+=wins[event_action]
        totals['motion']+=wins[motion]
        totals['wait_needed']+=wait_needed
        totals['cover_sightings']+=observation['cover_sightings']
        totals['no_track']+=not observation['south_track'] and not observation['north_track']
        totals['prechoice_tag']+=observation['choice_time'] is None
        totals['motion_only_vs_event']+=wins[motion] and not wins[event_action]
        totals['event_only_vs_motion']+=wins[event_action] and not wins[motion]
        rows.append(dict(seed=seed,event_key=event_key(observation),
                         event_action=event_action,motion_action=motion,
                         event_win=wins[event_action],motion_win=wins[motion],
                         route_case=route_case,wait_needed=wait_needed,
                         legal_south_samples=len(observation['south_track']),
                         legal_north_samples=len(observation['north_track'])))
    return dict(counts=dict(totals),route_cases=dict(route_cases),rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--design',nargs=3,required=True,type=Path)
    parser.add_argument('--test',nargs=3,required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    design=load_rows(args.design)
    test=load_rows(args.test)
    fallback,events=event_table(design)
    result=dict(status='formal_single_zone_matched_audit',
                warning='Not yet a two-guard match or PPO result',
                design_seeds=sorted(design),test_seeds=sorted(test),
                trained_fixed_action=fallback,
                posthoc_test_best_fixed_action=best_action(list(test.values())),
                posthoc_test_best_fixed_wins=sum(
                    actions[best_action(list(test.values()))]['local_survived']
                    for actions in test.values()),
                trained_event_table={str(key):action for key,action in events.items()},
                design_event_conflicts=event_conflicts(design),
                test_event_conflicts=event_conflicts(test),
                design=audit(design,fallback,events),
                test=audit(test,fallback,events))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('design','test','design_seeds','test_seeds')},indent=2))
    print('DESIGN',json.dumps(result['design']['counts']))
    print('TEST',json.dumps(result['test']['counts']))
    print('TEST_ROUTE_CASES',json.dumps(result['test']['route_cases']))


if __name__=='__main__':main()
