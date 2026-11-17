"""Matched timing-only versus motion-aware decision rule on a frozen seed split."""
import argparse
import json
import math
from pathlib import Path

from formal_central_audit import ACTIONS,event_key,load_rows


def features(row,mode):
    if mode not in ('timing','position','motion'):raise ValueError(mode)
    result=[]
    for name in ('south_track','north_track'):
        track=row[name]
        result.extend((int(bool(track)),track[0][0]/6. if track else 0.,
                       track[-1][0]/6. if track else 0.,len(track)/20.))
        if mode in ('position','motion'):
            velocity=((track[-1][2]-track[-2][2])/
                      max(track[-1][0]-track[-2][0],1e-9)
                      if len(track)>1 else 0.)
            result.append(track[-1][2]/3.35 if track else 0.)
            if mode=='motion':result.append(velocity/1.5)
    return result


def choose(training,row,mode,k=5):
    target=features(row,mode)
    neighbors=sorted((math.dist(target,features(obs,mode)),seed,actions)
                     for seed,obs,actions in training)[:k]
    scores={action:sum((1./(.15+distance))*int(actions[action]['local_survived'])
                       for distance,_,actions in neighbors)
            for action in ACTIONS}
    return max(ACTIONS,key=lambda action:(scores[action],-action[1],
                                           action[0]=='north'))


def evaluate(training,test):
    wins={'timing':0,'position':0,'motion':0}
    paired={'motion_only_vs_timing':0,'timing_only_vs_motion':0,
            'motion_only_vs_position':0,'position_only_vs_motion':0,
            'position_only_vs_timing':0,'timing_only_vs_position':0}
    rows=[]
    for seed,actions in sorted(test.items()):
        obs=next(iter(actions.values()))
        selected={key:choose(training,obs,key)
                  for key in wins}
        outcome={key:bool(actions[action]['local_survived'])
                 for key,action in selected.items()}
        for key in wins:wins[key]+=outcome[key]
        paired['motion_only_vs_timing']+=outcome['motion'] and not outcome['timing']
        paired['timing_only_vs_motion']+=outcome['timing'] and not outcome['motion']
        paired['motion_only_vs_position']+=outcome['motion'] and not outcome['position']
        paired['position_only_vs_motion']+=outcome['position'] and not outcome['motion']
        paired['position_only_vs_timing']+=outcome['position'] and not outcome['timing']
        paired['timing_only_vs_position']+=outcome['timing'] and not outcome['position']
        rows.append(dict(seed=seed,event_key=event_key(obs),
                         timing_action=selected['timing'],
                         position_action=selected['position'],
                         motion_action=selected['motion'],
                         timing_win=outcome['timing'],
                         position_win=outcome['position'],
                         motion_win=outcome['motion']))
    return dict(wins=wins,paired=paired,rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--design',nargs=3,required=True,type=Path)
    parser.add_argument('--test',nargs=3,required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    design=load_rows(args.design)
    test=load_rows(args.test)
    training=[(seed,next(iter(actions.values())),actions)
              for seed,actions in design.items()]
    result=dict(status='matched_knn_audit',
                warning='Exploratory small-sample model; one guard and local exit only',
                design_seeds=sorted(design),test_seeds=sorted(test),
                feature_scales=dict(time=6.,sample_count=20.,position_y=3.35,
                                    velocity_y=1.5),k=5,
                test=evaluate(training,test))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(test=result['test']['wins'],
                          paired=result['test']['paired']),indent=2))


if __name__=='__main__':main()
