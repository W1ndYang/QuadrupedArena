"""Known-map A* and kinematic baselines, separate from virtual-mark referee."""
import heapq
import math
import numpy as np
from tactical_logic import Mover, block
from trial_rules import Referee, covers_at


def clearance(xy,cover):
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(xy)-cover["center"])-np.asarray(cover["size"][:2])/2,0)))


class Planner:
    def __init__(self,config,covers):
        self.cfg=config;self.covers=covers;self.grid=config["planner_grid"]
        # Controller-only exclusion: known fall footprint, not an invisible LOS wall.
        self.obstacles=covers+[config["collapse"]["planner_reserved_sweep"]]
        self.margin=config["planner_clearance"]
        self.free_nodes={(x,y) for x in range(-27,28) for y in range(-17,18) if self.free(np.array([x,y])*self.grid)}

    def free(self,p):
        return abs(p[0])<=5.55 and abs(p[1])<=3.55 and all(clearance(p,c)>=self.margin for c in self.obstacles)

    def segment(self,a,b):
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


def sweep_clear(config,poses):
    c=next(c for c in config["arena"]["covers"] if c["name"]==config["collapse"]["wall"])
    pivot=np.asarray(config["collapse"]["pivot"])
    original=np.array([*c["center"],c["size"][2]/2])
    for angle in np.linspace(0,math.pi/2,41):
        rot=np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])
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
