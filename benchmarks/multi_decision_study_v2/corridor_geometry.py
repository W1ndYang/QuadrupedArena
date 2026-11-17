"""Three-gate arena geometry admission check, before adding any game rules.

This candidate keeps the formal fixed-goal motion and 3D sight algorithms, but
uses a new arena. Each gate has north/south openings and a shelter before it.
No decision time, patrol class, smoke obligation, or reward is encoded here.
"""
import copy
import itertools
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'multi_decision_study_v1'))
from sequential import load_case  # noqa: E402
from env import boxes,clear_ray  # noqa: E402
from trial_motion import Planner,path_length  # noqa: E402


GATES=(-5.,0.,5.)
START=(-10.,0.)
GOAL=(10.,0.)
OPENINGS=(-3.,3.)
SHELTER_PAIRS={
    (1,'south'):(-8.,-3.5,-8.,-4.0),
    (1,'north'):(-8.,2.5,-8.,2.0),
    (2,'south'):(-3.,-3.5,-3.,-4.0),
    (2,'north'):(-3.,2.5,-3.,2.0),
    (3,'south'):(2.,-3.5,2.,-4.0),
    (3,'north'):(2.,2.5,2.,2.0),
}


def cover(name,x,y,width,height):
    return dict(name=name,center=[float(x),float(y)],
                size=[float(width),float(height),2.4])


def make_config():
    base,rules,_=load_case('SE')
    base=copy.deepcopy(base)
    base['start_a']=list(START)
    base['arena']['covers']=[]
    for i,x in enumerate(GATES):
        for label,y in (('lower',-6.),('middle',0.),('upper',6.)):
            base['arena']['covers'].append(cover(f'Gate{i+1}_{label}',x,y,.7,4.))
        for side,y in (('south',-3.),('north',3.)):
            base['arena']['covers'].append(cover(f'Shelter{i+1}_{side}',x-2.,y,.75,1.))
    # The inherited collapse is disabled by the formal Skill probe; its
    # controller-only future sweep must not reserve part of this new arena.
    base['collapse']['planner_reserved_sweep']=cover('InactiveFutureSweep',100.,100.,.1,.1)
    return base,rules


def route_combinations(planner):
    rows=[]
    for signs in itertools.product(OPENINGS,repeat=len(GATES)):
        checkpoints=[(x,y) for x,y in zip(GATES,signs)]
        current=START;length=0.;reachable=True
        for point in [*checkpoints,GOAL]:
            route=planner.route(current,point)
            if route is None:
                reachable=False;break
            length+=path_length(current,route)
            current=point
        rows.append(dict(openings=['south' if y<0 else 'north' for y in signs],
                         reachable=reachable,length_m=round(length,3) if reachable else None))
    return rows


def cover_peek_pairs(base,planner):
    """Find 1 m peeks where the shelter changes legal B visibility."""
    walls=boxes(base['arena']['covers'])
    rows=[]
    for i,x in enumerate(GATES):
        for side in OPENINGS:
            opponent=(x+1.3,side)
            candidates=[]
            for ax in [x-3.25+j*.25 for j in range(11)]:
                for ay in [side-1.5+j*.25 for j in range(13)]:
                    a=(ax,ay)
                    if not planner.free(a) or clear_ray((*a,.4),(*opponent,1.05),walls):continue
                    for dx,dy in ((.5,0.),(-.5,0.),(0.,.5),(0.,-.5),
                                  (.75,0.),(-.75,0.),(0.,.75),(0.,-.75),
                                  (1.,0.),(-1.,0.),(0.,1.),(0.,-1.)):
                        q=(ax+dx,ay+dy)
                        if not planner.free(q) or not planner.segment(a,q):continue
                        distance=math.dist(a,q)
                        stop=(q[0]+.16*(a[0]-q[0])/distance,
                              q[1]+.16*(a[1]-q[1])/distance)
                        if (planner.free(stop) and
                                clear_ray((*stop,.4),(*opponent,1.05),walls)):
                            score=math.dist(a,(x-2.,side))+math.dist(q,a)
                            candidates.append((score,a,q))
            best=min(candidates,key=lambda c:c[0]) if candidates else None
            rows.append(dict(gate=i+1,side='south' if side<0 else 'north',
                             defender_point=opponent,
                             cover_point=best[1] if best else None,
                             peek_point=best[2] if best else None,
                             peek_distance_m=round(math.dist(best[1],best[2]),3) if best else None))
    return rows


def evaluate():
    base,rules=make_config()
    planner=Planner(base,base['arena']['covers'])
    routes=route_combinations(planner)
    peeks=cover_peek_pairs(base,planner)
    walls=boxes(base['arena']['covers'])
    configured=[]
    for (i,side),(cx,cy,px,py) in SHELTER_PAIRS.items():
        distance=math.dist((cx,cy),(px,py))
        stop=(px+.16*(cx-px)/distance,py+.16*(cy-py)/distance)
        other=(GATES[i-1]+1.3,-3. if side=='south' else 3.,1.05)
        configured.append(dict(gate=i,side=side,cover=[cx,cy],peek_target=[px,py],
                               actual_observation_stop=stop,
                               navigation_clear=bool(planner.segment((cx,cy),(px,py))),
                               cover_visible=bool(clear_ray((cx,cy,.4),other,walls)),
                               stop_visible=bool(clear_ray((*stop,.4),other,walls))))
    return dict(status='geometry_only_candidate',gate_x=GATES,start=START,goal=GOAL,
                route_combinations=routes,cover_peeks=peeks,configured_peeks=configured,
                reachable_combinations=sum(r['reachable'] for r in routes),
                valid_cover_peeks=sum(p['cover_point'] is not None for p in peeks),
                valid_configured_peeks=sum(p['navigation_clear'] and not p['cover_visible']
                                           and p['stop_visible'] for p in configured),
                rules_reference=rules['version'])


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=evaluate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(reachable_combinations=result['reachable_combinations'],
                          valid_cover_peeks=result['valid_cover_peeks'],
                          valid_configured_peeks=result['valid_configured_peeks'],
                          route_combinations=result['route_combinations'],
                          cover_peeks=result['cover_peeks'])))


if __name__=='__main__':main()
