"""Frozen KNN feature comparison for four-waypoint patrols."""
import argparse
import json
import math
from pathlib import Path

from formal_central_audit import ACTIONS,load_rows
from formal_central_knn_audit import features as line_features


def features(row,mode):
    if mode!='planar':return line_features(row,mode)
    result=[]
    guard_x=7.1 if row.get('stage2_local') else -3.9
    for name in ('south_track','north_track'):
        track=row[name]
        result.extend((int(bool(track)),track[0][0]/6. if track else 0.,
                       track[-1][0]/6. if track else 0.,len(track)/20.))
        if track:
            a,b=track[-2:] if len(track)>1 else (track[-1],track[-1])
            dt=max(b[0]-a[0],1e-9)
            result.extend(((b[1]-guard_x)/1.8,b[2]/3.35,
                           (b[1]-a[1])/dt/1.5,(b[2]-a[2])/dt/1.5))
        else:result.extend((0.,0.,0.,0.))
    return result


def choose(training,observation,mode,k=5):
    target=features(observation,mode)
    neighbors=sorted((math.dist(target,features(obs,mode)),seed,actions)
                     for seed,obs,actions in training)[:k]
    scores={action:sum(int(actions[action]['local_survived'])/
                       (.15+distance) for distance,_,actions in neighbors)
            for action in ACTIONS}
    return max(ACTIONS,key=lambda action:(scores[action],-action[1],
                                           action[0]=='north'))


def audit(design,test):
    training=[(seed,next(iter(actions.values())),actions)
              for seed,actions in design.items()]
    modes=('timing','position','motion','planar')
    wins={mode:0 for mode in modes}
    rows=[]
    for seed,actions in sorted(test.items()):
        obs=next(iter(actions.values()))
        selected={mode:choose(training,obs,mode) for mode in modes}
        outcomes={mode:bool(actions[action]['local_survived'])
                  for mode,action in selected.items()}
        for mode in modes:wins[mode]+=outcomes[mode]
        rows.append(dict(seed=seed,actions=selected,outcomes=outcomes,
                         observed_samples=len(obs['south_track'])+
                                          len(obs['north_track'])))
    return dict(design_seeds=sorted(design),test_seeds=sorted(test),
                k=5,wins=wins,rows=rows,
                feature_scales=dict(time=6.,count=20.,x_offset=1.8,y=3.35,
                                    velocity_x=1.5,velocity_y=1.5))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--design',required=True,type=Path)
    parser.add_argument('--test',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=audit(load_rows([args.design]),load_rows([args.test]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['wins'],indent=2))


if __name__=='__main__':main()
