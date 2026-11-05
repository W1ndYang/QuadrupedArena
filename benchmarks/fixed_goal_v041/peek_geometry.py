"""Map/last-observation-only peek candidates and exact local-motion preview."""
import math
import copy
import numpy as np
from ds_fixed_goal import boxes,clear_ray,Planner
from continuous_motion import ContinuousMover
from trial_motion import path_length

PEEK=dict(speed=1.6,reverse_speed=1.6,acceleration=6.,hold=.04,
          hidden_pause=.18,max_exposure=.70,dt=.02,max_cycles=3)

def ray_open(q,threat,walls,height=.4):
    return clear_ray((*q,height),(*threat,1.05),walls)

def preview(home,out,threat,planner,hold=.04):
    """Finite acceleration/turning; no teleport. Same mover and caps as execution."""
    h=np.array(home,float);o=np.array(out,float);delta=o-h
    yaw=math.atan2(delta[1],delta[0]);m=ContinuousMover(h,yaw,PEEK['speed'],6.,2.5)
    m.reverse_allowed=True;m.reverse_limit=PEEK['reverse_speed'];m.navigate([o])
    walls=boxes(planner.covers);phase='out';at_out=None;trace=[];exposure=maximum=0.
    for k in range(251):
        t=k*.02;visible=ray_open(m.xy,threat,walls)
        exposure=exposure+.02 if visible else 0.;maximum=max(maximum,exposure)
        trace.append(dict(t=round(t,2),xy=m.xy.tolist(),yaw=m.yaw,v=m.v,visible=visible,phase=phase))
        if phase=='out' and math.dist(m.xy,o)<1e-7 and abs(m.v)<1e-7:
            phase='hold';at_out=t
        if phase=='hold' and t-at_out>=hold-1e-9:phase='back';m.navigate([h])
        if phase=='back' and math.dist(m.xy,h)<1e-7 and abs(m.v)<1e-7:
            return dict(feasible=True,duration=round(t,2),max_exposure=round(maximum,3),trace=trace)
        before=m.xy.copy();m.step(.02)
        if not planner.segment(before,m.xy):return dict(feasible=False,reason='collision')
    return dict(feasible=False,reason='local_motion_timeout')

class PeekCatalog:
    def __init__(self,base):
        self.base=base;self.planner=Planner(base,base['arena']['covers']);self.walls=boxes(base['arena']['covers'])
        self.registry={};self.serial=0

    def candidates(self,position,threat,goal,observed_at,t,limit=6):
        if threat is None:return []
        p=self.planner;raw=[];seen=set()
        for wall in p.covers:
            x,y=wall['center'];w,d,_=wall['size']
            sides=[([x-w/2-.47,y],[0,1],d,'W'),([x+w/2+.47,y],[0,1],d,'E'),
                   ([x,y-d/2-.47],[1,0],w,'S'),([x,y+d/2+.47],[1,0],w,'N')]
            for center,axis,length,side in sides:
                axis=np.array(axis,float);points=[np.array(center)+axis*u for u in np.arange(-length/2-1.1,length/2+1.1001,.10)]
                last=None
                for q in points:
                    state=ray_open(q,threat,self.walls) if p.free(q) else None
                    if last is not None and state is not None and last[1] is not None and state!=last[1]:
                        sign=1 if state else -1;boundary=(q+last[0])/2
                        home=boundary-sign*.20*axis;out=boundary+sign*.10*axis
                        key=tuple(np.round(np.r_[home,out],2))
                        robust_home=all(not ray_open(home+v,threat,self.walls) for v in ([0,0],[.04,0],[-.04,0],[0,.04],[0,-.04]))
                        if key not in seen and p.segment(home,out) and robust_home and ray_open(out,threat,self.walls):
                            seen.add(key);raw.append((math.dist(position,home)+.12*math.dist(home,goal),wall['name'],side,home,out))
                    last=(q,state)
        result=[]
        for _,wall,side,home,out in sorted(raw,key=lambda v:v[0])[:28]:
            route=p.route(position,home)
            if route is None:continue
            check=preview(home,out,threat,p)
            if not check['feasible'] or check['max_exposure']>PEEK['max_exposure']+1e-9:continue
            length=path_length(position,route);exposed=0.
            for a,b in zip([np.array(position)]+route,route):
                n=max(1,math.ceil(math.dist(a,b)/.15))
                exposed+=sum(ray_open(np.array(a)*(1-u)+np.array(b)*u,threat,self.walls) and math.dist(np.array(a)*(1-u)+np.array(b)*u,threat)<=8 for u in np.linspace(0,1,n))*math.dist(a,b)/n
            ident=f'PK{self.serial:03d}';self.serial+=1
            item=dict(id=ident,wall=wall,side=side,home=[round(float(v),5) for v in home],out=[round(float(v),5) for v in out],
                      expected_threat=list(threat),observed_at=observed_at,created_at=t,
                      approach_m=round(length,2),approach_exposed_m=round(exposed,2),predicted_burst_seconds=check['duration'],
                      predicted_exposure_seconds=check['max_exposure'],assumption='Static last-observed/initial target; not a safety guarantee for a moving hidden opponent.')
            self.registry[ident]=item;result.append(copy.deepcopy(item))
            if len(result)>=limit:break
        return result

    def nearest_hidden(self,position,threat,max_distance=4.):
        """Emergency destination from public threat only, map free-space and LOS."""
        candidates=[]
        for node in self.planner.free_nodes:
            q=np.array(node)*self.planner.grid;d=math.dist(position,q)
            if .12<d<=max_distance and not ray_open(q,threat,self.walls) and self.planner.segment(position,q):
                candidates.append((d,q))
        return min(candidates,key=lambda v:v[0])[1].tolist() if candidates else None
