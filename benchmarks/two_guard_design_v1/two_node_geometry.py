"""Physical transfer candidate between two independent guard regions.

This module checks navigability and line-of-sight isolation of the two patrol
lines.  The continuous formal screen uses an isolated guard handoff.
"""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'/'multi_decision_study_v2'))
from corridor_geometry import cover,make_config  # noqa: E402
from env import boxes,clear_ray  # noqa: E402
from trial_motion import Planner,path_length  # noqa: E402

FIRST_EXIT_X=-1.
SECOND_COVER=(3.,0.)
SECOND_GUARD_X=7.1


def narrow_gate(covers,prefix,low_middle,outer_boundary=2.6):
    for wall in covers:
        if not wall['name'].startswith(prefix):continue
        if wall['name'].endswith('middle'):
            wall['size'][1]=1.2
            if low_middle:wall['size'][2]=.6
        elif wall['name'].endswith('lower'):
            wall['center'][1]=-(8.+outer_boundary)/2.
            wall['size'][1]=8.-outer_boundary
        elif wall['name'].endswith('upper'):
            wall['center'][1]=(8.+outer_boundary)/2.
            wall['size'][1]=8.-outer_boundary


def add_two_node_geometry(base):
    covers=base['arena']['covers']
    covers[:]=[c for c in covers
               if not c['name'].startswith(('Gate2_','Shelter2_','Shelter3_'))]
    for wall in covers:
        if wall['name'].startswith('Gate3_'):
            wall['center'][0]+=1.
    narrow_gate(covers,'Gate1_',True)
    narrow_gate(covers,'Gate3_',True,outer_boundary=3.4)
    covers.extend((
        cover('CentralObserver1',-7.3,0.,.5,1.1),
        cover('CentralObserver2',3.7,0.,.5,3.),
        cover('TransferBaffleA',0.,0.,.5,4.),
        cover('TransferBaffleBLower',2.,-4.5,.5,7.),
        cover('TransferBaffleBUpper',2.,4.5,.5,7.),
        cover('PrivacySouth',2.3,-1.3,.3,1.2),
        cover('PrivacyNorth',2.3,1.3,.3,1.2),
    ))
    return base


def sample_path(start,route,step=.1):
    samples=[];previous=start
    for endpoint in route:
        n=max(1,math.ceil(math.dist(previous,endpoint)/step))
        for j in range(n+1):
            u=j/n
            samples.append((previous[0]+u*(endpoint[0]-previous[0]),
                            previous[1]+u*(endpoint[1]-previous[1])))
        previous=endpoint
    return samples


def audit():
    base,_=make_config();add_two_node_geometry(base)
    planner=Planner(base,base['arena']['covers'])
    walls=boxes(base['arena']['covers'])
    guard_ys=[-3.35+j*.1 for j in range(68)]
    rows=[]
    for side in (-1.,1.):
        start=(FIRST_EXIT_X,side*1.6)
        route=planner.route(start,SECOND_COVER)
        if route is None:raise RuntimeError(f'No transfer route from {start}')
        samples=sample_path(start,route)
        early_second=[];first_tag_exposure=[]
        for guard_y in guard_ys:
            second=(SECOND_GUARD_X,guard_y)
            if any(math.dist(q,second)<=8. and
                   clear_ray((*q,.4),(*second,1.05),walls)
                   for q in samples):early_second.append(round(guard_y,2))
            first=(-3.9,guard_y)
            if any(math.dist(q,first)<=4. and
                   clear_ray((*q,.4),(*first,1.05),walls)
                   for q in samples):first_tag_exposure.append(round(guard_y,2))
        rows.append(dict(side='south' if side<0 else 'north',
                         transfer_route=[list(map(float,q)) for q in route],
                         transfer_length_m=round(path_length(start,route),3),
                         early_second_guard_states=len(early_second),
                         first_guard_in_range_states=len(first_tag_exposure)))
    second_peek={}
    for side,peek_y in (('south',-1.94),('north',1.94)):
        second_peek[side]=sum(clear_ray((3.,peek_y,.4),
                                        (SECOND_GUARD_X,y,1.05),walls)
                              for y in guard_ys)
    peek_union=sum(any(clear_ray((3.,peek_y,.4),
                                 (SECOND_GUARD_X,y,1.05),walls)
                       for peek_y in (-1.94,1.94))
                   for y in guard_ys)
    first_guard_after_second={}
    for side in (-1.,1.):
        samples=[];current=SECOND_COVER
        for target in ((6.,side*1.6),(10.,side*1.6)):
            route=planner.route(current,target)
            if route is None:raise RuntimeError(f'No second-zone route to {target}')
            samples+=sample_path(current,route)
            current=target
        first_guard_after_second['south' if side<0 else 'north']=sum(
            any(clear_ray((*q,.4),(-3.9,y,1.05),walls) for q in samples)
            for y in guard_ys)
    handoff_visibility={}
    for label,gx in (('first_guard',-3.9),('second_guard',SECOND_GUARD_X)):
        handoff_visibility[label]=sum(
            math.dist(q,(gx,y))<=8. and
            clear_ray((*q,.4),(gx,y,1.05),walls)
            for ix in range(-5,6) for iy in range(-5,6)
            for q in ((SECOND_COVER[0]+ix*.03,iy*.03),)
            if math.dist(q,SECOND_COVER)<=.15+1e-9
            for y in guard_ys)
    candidate_patrol_envelope={}
    for label,cover_x,x0 in (('first',-8.,-3.9),
                             ('second',3.,7.1)):
        points=[(round(x0+ix*.2,2),round(-3.35+iy*.1,2))
                for ix in range(10) for iy in range(68)]
        legal=[q for q in points if planner.free(q)]
        cover_hidden=[q for q in legal
                      if not clear_ray((cover_x,0.,.4),(*q,1.05),walls)]
        peek_observable=[q for q in legal if any(
            math.dist((cover_x,peek_y),q)<=8. and
            abs(math.degrees(math.atan2(q[1]-peek_y,q[0]-cover_x)))<=60. and
            clear_ray((cover_x,peek_y,.4),(*q,1.05),walls)
            for peek_y in (-1.94,1.94))]
        candidate_patrol_envelope[label]=dict(
            x_range_m=[x0,round(x0+1.8,2)],y_range_m=[-3.35,3.35],
            samples=len(points),free_samples=len(legal),
            hidden_from_cover_samples=len(cover_hidden),
            visible_from_at_least_one_peek_samples=len(peek_observable))
    result=dict(status='two_node_geometry_only',
                warning='First guard remains a transfer hazard; no three-actor referee yet',
                transfer=rows,second_guard_patrol_samples=len(guard_ys),
                second_peek_visible_samples=second_peek,
                second_peek_union_visible_samples=peek_union,
                first_guard_visible_after_second_cover=first_guard_after_second,
                handoff_disc_visible_rays=handoff_visibility,
                candidate_patrol_envelope=candidate_patrol_envelope,
                second_cover_hides_full_patrol=all(
                    not clear_ray((3.,0.,.4),(SECOND_GUARD_X,y,1.05),walls)
                    for y in guard_ys),
                second_gate_routes_reachable={side:planner.route(
                    SECOND_COVER,(6.,1.6 if side=='north' else -1.6)) is not None
                    for side in ('south','north')})
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
