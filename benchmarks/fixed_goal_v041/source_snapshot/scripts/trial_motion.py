"""Known-map A* and kinematic baselines, separate from virtual-mark referee."""
import heapq
import math
import numpy as np
from tactical_logic import Mover, block
from trial_rules import Referee, covers_at, collapse_pieces


def clearance(xy,cover):
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(xy)-cover["center"])-np.asarray(cover["size"][:2])/2,0)))


class Planner:
    def __init__(self,config,covers):
        self.cfg=config;self.covers=covers;self.grid=config["planner_grid"]
        # Controller-only exclusion: known fall footprint, not an invisible LOS wall.
        fallen=(all(any(c['name']==p['name'] for c in covers) for p in config['collapse']['debris_pieces'])
                if config['collapse'].get('debris_pieces') else
                any(c['name']==config['collapse']['wall'] and c['size'][2]<1. for c in covers))
        release=fallen and config['collapse'].get('release_sweep_after_collapse',False)
        self.obstacles=covers+([] if release else [config["collapse"]["planner_reserved_sweep"]])
        self.margin=config["planner_clearance"]
        self.bounds=np.asarray(config['arena']['floor_size'][:2],float)/2-config.get('planner_boundary_margin',.45)
        nx,ny=np.floor((self.bounds+1e-9)/self.grid).astype(int)
        self.free_nodes={(x,y) for x in range(-nx,nx+1) for y in range(-ny,ny+1) if self.free(np.array([x,y])*self.grid)}
        self._edge_cache={}

    def free(self,p):
        return abs(p[0])<=self.bounds[0]+1e-10 and abs(p[1])<=self.bounds[1]+1e-10 and all(clearance(p,c)>=self.margin for c in self.obstacles)

    def neighbors(self,node):
        """Reusable map-only graph, with exact swept edges for the new arena."""
        if node not in self._edge_cache:
            result=[]
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt=(node[0]+dx,node[1]+dy)
                if nxt not in self.free_nodes:continue
                if dx and dy and ((node[0]+dx,node[1]) not in self.free_nodes or (node[0],node[1]+dy) not in self.free_nodes):continue
                if self.cfg.get('motion',{}).get('validate_grid_edges') and not self.segment(np.asarray(node)*self.grid,np.asarray(nxt)*self.grid):continue
                result.append((nxt,math.hypot(dx,dy)*self.grid))
            self._edge_cache[node]=result
        return self._edge_cache[node]

    def distance_field(self,start):
        return DistanceField(self,start)

    def segment(self,a,b):
        if self.cfg.get('motion',{}).get('exact_segment_clearance'):
            # Swept disk against wall boxes: endpoint checks plus segment/box
            # intersection and corner-to-segment distance. Sampling can miss a
            # sub-millimetre incursion at a rounded clearance corner.
            ax,ay=map(float,a);bx,by=map(float,b);dx,dy=bx-ax,by-ay;length2=dx*dx+dy*dy
            if not self.free(a) or not self.free(b):return False
            if length2<1e-20:return True
            for c in self.obstacles:
                x,y=c['center'];w,h=c['size'][:2];lo=(x-w/2,y-h/2);hi=(x+w/2,y+h/2)
                lower,upper=0.,1.
                for origin,delta,mn,mx in zip((ax,ay),(dx,dy),lo,hi):
                    if abs(delta)<1e-12:
                        if not mn<=origin<=mx:lower,upper=2.,-1.;break
                    else:
                        u,v=sorted(((mn-origin)/delta,(mx-origin)/delta));lower=max(lower,u);upper=min(upper,v)
                if lower<=upper:return False
                for px,py in ((lo[0],lo[1]),(lo[0],hi[1]),(hi[0],lo[1]),(hi[0],hi[1])):
                    u=max(0.,min(1.,((px-ax)*dx+(py-ay)*dy)/length2))
                    if math.hypot(ax+u*dx-px,ay+u*dy-py)<self.margin-1e-10:return False
            return True
        return all(self.free(np.asarray(a)*(1-u)+np.asarray(b)*u) for u in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(np.asarray(a)-b)/.05)+1)))

    def route(self,start,goal):
        if not self.free(goal):return None
        if self.segment(start,goal):return [np.asarray(goal,float)]
        def nearest(p):
            candidates=sorted(self.free_nodes,key=lambda n:np.linalg.norm(np.asarray(n)*self.grid-p))
            return next((n for n in candidates[:30] if self.segment(p,np.asarray(n)*self.grid)),None)
        s=nearest(np.asarray(start));g=nearest(np.asarray(goal))
        if s is None or g is None:return None
        queue=[(0.,s)];cost={s:0.};parent={};closed=set()
        while queue:
            _,node=heapq.heappop(queue)
            if node in closed:continue
            if node==g:break
            closed.add(node)
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt=(node[0]+dx,node[1]+dy)
                if nxt not in self.free_nodes:continue
                if dx and dy and ((node[0]+dx,node[1]) not in self.free_nodes or (node[0],node[1]+dy) not in self.free_nodes):continue
                nc=cost[node]+math.hypot(dx,dy)
                if nc<cost.get(nxt,float("inf")):
                    cost[nxt]=nc;parent[nxt]=node
                    heapq.heappush(queue,(nc+math.dist(nxt,g),nxt))
        if g not in cost:return None
        nodes=[g]
        while nodes[-1]!=s:nodes.append(parent[nodes[-1]])
        raw=[np.asarray(start),*[np.asarray(n)*self.grid for n in reversed(nodes)],np.asarray(goal)]
        route=[];index=0
        while index<len(raw)-1:
            end=len(raw)-1
            while end>index+1 and not self.segment(raw[index],raw[end]):end-=1
            route.append(raw[end]);index=end
        return route


def path_length(start,route):
    return sum(float(np.linalg.norm(b-a)) for a,b in zip([np.asarray(start)]+route,route))


class DistanceField:
    """One Dijkstra expansion for many defender viewpoints; no actor truth."""
    def __init__(self,planner,start):
        self.planner=planner;self.start=np.asarray(start,float);self.cost={};self.parent={}
        ordered=sorted(planner.free_nodes,key=lambda n:(math.dist(np.asarray(n)*planner.grid,self.start),n))
        self.source=next((n for n in ordered[:30] if planner.segment(self.start,np.asarray(n)*planner.grid)),None)
        if self.source is None:return
        self.cost[self.source]=math.dist(self.start,np.asarray(self.source)*planner.grid)
        queue=[(self.cost[self.source],self.source)]
        while queue:
            value,n=heapq.heappop(queue)
            if value>self.cost[n]+1e-9:continue
            for nxt,weight in planner.neighbors(n):
                new=value+weight
                if new<self.cost.get(nxt,float('inf'))-1e-9:
                    self.cost[nxt]=new;self.parent[nxt]=n;heapq.heappush(queue,(new,nxt))

    def route(self,goal):
        p=self.planner;goal=np.asarray(goal,float)
        if not p.free(goal):return None
        if p.segment(self.start,goal):return [goal]
        node=tuple(np.rint(goal/p.grid).astype(int))
        if node not in self.cost or not p.segment(np.asarray(node)*p.grid,goal):
            ordered=sorted(self.cost,key=lambda n:(math.dist(np.asarray(n)*p.grid,goal),n))
            node=next((n for n in ordered[:30] if p.segment(np.asarray(n)*p.grid,goal)),None)
        if node is None:return None
        chain=[node]
        while chain[-1]!=self.source:chain.append(self.parent[chain[-1]])
        raw=[self.start,*[np.asarray(n)*p.grid for n in reversed(chain)],goal];route=[];i=0
        while i<len(raw)-1:
            j=len(raw)-1
            while j>i+1 and not p.segment(raw[i],raw[j]):j-=1
            if not p.segment(raw[i],raw[j]):return None
            if math.dist(raw[i],raw[j])>1e-8:route.append(raw[j])
            i=j
        return route


def attack_route(config,planner,b,a):
    candidates=[]
    for radius in (2.2,3.5,5.0):
        for theta in np.linspace(0,2*math.pi,12,endpoint=False):
            p=a.xy+radius*np.array([math.cos(theta),math.sin(theta)])
            if not planner.free(p):continue
            if block((*p,1.05),(*a.xy,.4),planner.covers):continue
            candidates.append((float(np.linalg.norm(p-b.xy)),p))
    best=None
    for _,p in sorted(candidates,key=lambda c:c[0])[:10]:
        route=planner.route(b.xy,p)
        if route is None:continue
        value=path_length(b.xy,route)
        if best is None or value<best[0]:best=(value,route)
    return None if best is None else best[1]


def collapse_rotation(config, progress=1.):
    """Shared fixed-axis presentation/sweep rotation; legacy maps rotate about X."""
    theta=math.radians(config['collapse']['angle_degrees'])*progress
    c,s=math.cos(theta),math.sin(theta)
    axis=config['collapse'].get('axis','x')
    if isinstance(axis,str):
        if axis not in ('x','y','z'):raise ValueError('Unknown collapse axis')
        axis={'x':[1.,0.,0.],'y':[0.,1.,0.],'z':[0.,0.,1.]}[axis]
    axis=np.asarray(axis,dtype=float)
    if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis)<1e-12:
        raise ValueError('Collapse axis must be a nonzero finite 3-vector')
    x,y,z=axis/np.linalg.norm(axis)
    return c*np.eye(3)+(1-c)*np.outer([x,y,z],[x,y,z])+s*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def sweep_clear(config,poses):
    c=next(c for c in config["arena"]["covers"] if c["name"]==config["collapse"]["wall"])
    if config['collapse'].get('mode')=='rubble':
        # In-place collapse: keep both old and new footprints clear; no rotating sweep.
        return all(clearance(xy,cover)>=config['planner_clearance']
                   for xy in poses for cover in [c,*collapse_pieces(config)])
    pivot=np.asarray(config["collapse"]["pivot"])
    original=np.array([*c["center"],c["size"][2]/2])
    for progress in np.linspace(0,1,41):
        rot=collapse_rotation(config,progress)
        center=pivot+rot@(original-pivot)
        half=np.abs(rot)@(np.asarray(c["size"])/2)
        projected={"center":center[:2],"size":[2*half[0],2*half[1],2*half[2]]}
        for xy in poses:
            if clearance(xy,projected)<config["planner_clearance"]:return False
    return True


def build(config):
    dt=config["dt"];ref=Referee(config)
    movers=[Mover(config[key],yaw,config["max_speed"][i],config["acceleration"][i],config["yaw_rate"][i])
            for i,(key,yaw) in enumerate((("start_a",0.),("start_b",math.pi)))]
    a,b=movers;planner=Planner(config,covers_at(config,0))
    states=[];decisions=[];selected=None;last_plan_b=-99.;last_state_b=None;map_changed=False
    minimum=float("inf");separation=float("inf");collapse_checked=False
    for tick in range(round(config["duration"]/dt)+1):
        t=round(tick*dt,8)
        changed=not map_changed and t>=config["collapse"]["time"]
        if changed:
            if not sweep_clear(config,[m.xy for i,m in enumerate(movers) if ref.hp[i]>0]):
                raise RuntimeError(f"Unsafe collapse sweep at {t}: {[m.xy.tolist() for m in movers]}")
            collapse_checked=True;map_changed=True
            planner=Planner(config,covers_at(config,t));a.route=[];b.route=[];last_plan_b=-99.
            decisions.append(dict(time=t,type="replan",actor=0,reason="map_changed"))
        if selected is None or not ref.eligible[selected]:
            candidates=[(np.linalg.norm(a.xy-p["center"]),k) for k,p in enumerate(config["arena"]["task_points"]) if ref.eligible[k]]
            if candidates:
                selected=min(candidates)[1];a.route=[]
                decisions.append(dict(time=t,type="select_target",point=selected))
        goal=config["arena"]["task_points"][selected]["center"]
        if ref.hp[0]>0 and not a.route and np.linalg.norm(a.xy-goal)>.08:
            route=planner.route(a.xy,goal)
            if route is None:raise RuntimeError("A target unreachable")
            a.navigate(route)
        if ref.hp[1]==0:
            b.route=[];b.v=0.;state_b="ELIMINATED"
        elif t<ref.cooldown_until[1]:
            b.route=[];state_b="COOLDOWN_TRACK"
        elif block((*b.xy,1.05),(*a.xy,.4),planner.covers) is None and np.linalg.norm(b.xy-a.xy)<=config["range_m"]:
            b.route=[];state_b="AIM"
        else:
            state_b="FIND_ATTACK_POSITION"
            if t-last_plan_b>2.0 or not b.route:
                route=attack_route(config,planner,b,a)
                if route is not None:b.navigate(route)
                last_plan_b=t
        if state_b!=last_state_b:
            decisions.append(dict(time=t,type="human_decision",state=state_b));last_state_b=state_b
        if tick and not changed:
            for i,m in enumerate(movers):
                if ref.hp[i]<=0:continue
                d=movers[1-i].xy-m.xy
                look=math.atan2(d[1],d[0]) if ref.hp[1-i]>0 else m.yaw
                before=(m.xy.copy(),m.wheels.copy(),m.distance)
                m.step(dt,look,config["wheel_radius"],config["wheel_track"])
                if ref.hp[1-i]>0 and np.linalg.norm(m.xy-movers[1-i].xy)<.85:
                    # Conservative traffic stop; keeps bodies separated, no contact solve.
                    m.xy,m.wheels,m.distance=before;m.v=0.
        xy=[m.xy.tolist() for m in movers];yaw=[m.yaw for m in movers]
        ref.step(t,xy,yaw,planner.covers,map_changed=changed)
        for i,m in enumerate(movers):
            if ref.hp[i]<=0:continue
            assert abs(m.xy[0])<5.65 and abs(m.xy[1])<3.65,(t,m.xy)
            minimum=min(minimum,*[clearance(m.xy,c) for c in planner.covers])
        if min(ref.hp)>0:separation=min(separation,float(np.linalg.norm(a.xy-b.xy)))
        states.append(dict(t=t,xy=xy,yaw=yaw,speed=[m.v for m in movers],wheels=a.wheels.tolist(),
            selected=selected,human_state=state_b,collapsed=map_changed,**ref.snapshot(t)))
        if ref.finished:break
    assert minimum>.33,("center cover clearance",minimum)
    assert separation>.75,("actor separation",separation)
    events=sorted(ref.events+decisions,key=lambda e:e["time"])
    validation=dict(min_center_to_cover=minimum,min_actor_separation=separation,
        collapse_sweep_checked=collapse_checked,score=ref.score,finish=ref.finished,
        max_speed=[max(s["speed"][i] for s in states) for i in range(2)],
        controller="nearest eligible target with A*; humanoid attack-priority tree; no training",
        scope="kinematic center-clearance checks, not full-body contact dynamics")
    return states,events,validation


if __name__=="__main__":
    import json
    from pathlib import Path
    config=json.loads((Path(__file__).resolve().parents[1]/"configs/trial_v3.json").read_text())
    states,events,validation=build(config)
    print(json.dumps(dict(events=[e for e in events if e["type"] not in ("aim_started","aim_broken","human_decision")],validation=validation,last=states[-1]),ensure_ascii=False,indent=2))
