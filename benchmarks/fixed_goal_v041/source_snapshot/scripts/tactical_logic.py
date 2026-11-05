"""Reactive cover-selection demo with separated agent observations and referee truth.

This is game/sport virtual marking, not a projectile model. Motion is kinematic.
"""
import json
import math
from pathlib import Path
import numpy as np


def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


def block(a,b,covers):
    """First 3D cover intersection, including the hit location for referee overlay."""
    a,b = np.asarray(a,float),np.asarray(b,float)
    best = None
    for c in covers:
        x,y = c["center"]; w,d,h = c["size"]
        lo,hi = (x-w/2,y-d/2,0),(x+w/2,y+d/2,h)
        lower,upper = 0.,1.
        for k in range(3):
            delta=b[k]-a[k]
            if abs(delta)<1e-9:
                if not lo[k]<=a[k]<=hi[k]: lower,upper=2.,-1.; break
            else:
                u,v=sorted(((lo[k]-a[k])/delta,(hi[k]-a[k])/delta))
                lower,upper=max(lower,u),min(upper,v)
        if lower<=upper and (best is None or lower<best[0]):
            best=(lower,c["name"],(a+lower*(b-a)).tolist())
    return best


def observe(xy,yaw,own_height,other,other_height,covers):
    delta=np.asarray(other)-xy
    bearing=math.atan2(delta[1],delta[0])
    if np.linalg.norm(delta)>8 or abs(wrap(bearing-yaw))>math.pi/3:
        return None
    if abs(math.atan2(other_height-own_height,np.linalg.norm(delta)))>math.pi/4:
        return None
    if block((*xy,own_height),(*other,other_height),covers):
        return None
    return np.asarray(other,float).copy()


class Mover:
    def __init__(self,xy,yaw,vmax,accel,turn):
        self.xy=np.array(xy,float); self.yaw=yaw; self.v=0.; self.omega=0.
        self.vmax=vmax; self.accel=accel; self.turn=turn; self.route=[]
        self.distance=0.; self.wheels=np.zeros(4)

    def navigate(self,route):
        self.route=[np.array(p,float) for p in route]

    def step(self,dt,look=None,radius=.089,track=.40):
        while self.route and np.linalg.norm(self.route[0]-self.xy)<.055:
            self.route.pop(0)
        if self.route:
            delta=self.route[0]-self.xy; dist=float(np.linalg.norm(delta))
            target=math.atan2(delta[1],delta[0])
            desired=min(self.vmax,math.sqrt(2*self.accel*max(0.,dist-.035)))
        else:
            target=self.yaw if look is None else float(look); desired=0.
        error=wrap(target-self.yaw)
        # Turn at a waypoint instead of sliding sideways through it.
        if abs(error)>.25: desired=0.
        old_yaw=self.yaw
        self.yaw+=float(np.clip(error,-self.turn*dt,self.turn*dt))
        self.omega=wrap(self.yaw-old_yaw)/dt
        self.v+=float(np.clip(desired-self.v,-self.accel*dt,self.accel*dt))
        ds=self.v*dt
        if self.route: ds=min(ds,float(np.linalg.norm(self.route[0]-self.xy)))
        self.xy+=ds*np.array([math.cos(self.yaw),math.sin(self.yaw)])
        self.distance+=ds
        self.wheels+=np.array([self.v-self.omega*track/2,self.v+self.omega*track/2]*2)/radius*dt


class CoverController:
    """Receives only visible opponent samples, never hidden current opponent position."""
    def __init__(self,config,arena,mover):
        self.cfg=config; self.arena=arena; self.mover=mover
        self.state="APPROACH"; self.age=0.; self.memory=None; self.memory_age=None
        self.cover_index=None; self.hits=0; self.transitions=[]; self.reason="搜索对手，准备进入掩体"
        self.mover.navigate([[-4.0,-3.2]])

    def change(self,state,t,reason,route=None):
        self.state=state; self.age=0.; self.reason=reason
        self.transitions.append({"time":round(t,2),"type":"decision","state":state,"reason":reason})
        if route is not None: self.mover.navigate(route)

    def choose_cover(self,seen,exclude=None):
        candidates=[]
        for i,node in enumerate(self.cfg["cover_nodes"]):
            if i!=exclude and block((*seen,1.05),(*node,.40),self.arena["covers"]):
                candidates.append((np.linalg.norm(np.asarray(node)-self.mover.xy),i))
        return min(candidates)[1] if candidates else None

    def update(self,t,dt,observation,marked,active_point):
        self.age+=dt
        if observation is not None:
            self.memory=observation.copy(); self.memory_age=0.
        elif self.memory_age is not None: self.memory_age+=dt
        if self.state=="APPROACH" and observation is not None:
            self.cover_index=self.choose_cover(observation)
            if self.cover_index is not None:
                self.change("TAKE_COVER",t,"发现对手，选择可切断视线的掩体",[[-3.85,-3.15],self.cfg["cover_nodes"][self.cover_index]])
        elif self.state=="TAKE_COVER" and not self.mover.route:
            self.change("HIDDEN",t,"掩体挡住视线；双方均不能穿墙标记")
        elif self.state=="HIDDEN" and self.age>1.5:
            self.change("PEEK",t,"沿预设安全通道探出，重新获取可见观测",[[-3.85,-3.15],self.cfg["peek_nodes"][0]])
        elif self.state=="PEEK" and not self.mover.route:
            self.change("AIM",t,"看见对手后停稳，持续锁定才可标记")
        elif self.state=="AIM" and marked:
            self.hits+=1
            self.change("WITHDRAW",t,"标记成功，撤回掩体；不是原地持续对射",[[-3.85,-3.15],self.cfg["cover_nodes"][0]])
        elif self.state=="WITHDRAW" and not self.mover.route:
            self.change("WATCH",t,"失去对手观测，仅保留旧位置；扫描墙边通道")
        elif self.state=="WATCH" and observation is not None:
            chosen=self.choose_cover(observation,exclude=0)
            if chosen is not None:
                self.cover_index=chosen
                self.change("RELOCATE",t,"重新看到侧面来敌：旧掩体失效，改选另一掩体",
                            [[-3.85,-3.15],[-3.85,-3.5],[-.65,-3.5],self.cfg["cover_nodes"][chosen]])
        elif self.state=="RELOCATE" and not self.mover.route:
            self.change("NEW_COVER",t,"到达新掩体，切断追踪；等待任务机会")
        elif self.state=="NEW_COVER" and self.age>1.8 and active_point==1:
            self.change("TO_POINT",t,"P2已激活，沿墙外侧通道快速转移到任务区",[[2.65,-3.5],[2.65,-1.5],[1.8,-.5]])
        elif self.state=="TO_POINT" and not self.mover.route:
            self.change("CAPTURE",t,"在任务区停留计分；掩体机动服务于任务目标")
        if self.state=="WATCH":
            look=math.pi/2+.25*math.sin(self.age*1.5)
        elif self.memory is not None:
            delta=self.memory-self.mover.xy; look=math.atan2(delta[1],delta[0])
        else: look=0.
        return look,self.state in ("PEEK","AIM","CAPTURE") and observation is not None


def build(config,arena,dt=.02):
    a=Mover(config["start_a"],0.,config["max_speed_a"],config["acceleration_a"],config["yaw_rate_a"])
    b=Mover(config["start_b"],0.,config["max_speed_b"],config["acceleration_b"],config["yaw_rate_b"])
    controller=CoverController(config,arena,a)
    hp=[3,3]; scores=[0.,0.]; dwell=[0.,0.]; locks=[0.,0.]; cooldown=-1.
    events=[]; states=[]; last_hit_a=False; flank=False; b_memory=None
    clearance=float("inf"); separation=float("inf"); last_point=0
    for tick in range(round(config["duration"]/dt)+1):
        t=tick*dt; point=int(t>=config["point_switch"])
        if point!=last_point:
            events.append({"time":round(t,2),"type":"point_switch","point":"P2"}); dwell=[0.,0.]
        last_point=point
        obs_a=observe(a.xy,a.yaw,.40,b.xy,1.05,arena["covers"])
        obs_b=observe(b.xy,b.yaw,1.05,a.xy,.40,arena["covers"])
        look_a,request_a=controller.update(t,dt,obs_a,last_hit_a,point)
        last_hit_a=False
        if hp[1]<3 and not flank and config.get("opponent_flank",True):
            flank=True
            b.navigate([[-.3,-1.9],[-.3,.25],[-2.2,.4],[-3.85,.4]])
            events.append({"time":round(t,2),"type":"opponent_flank","reason":"Scripted opponent changes approach after being marked"})
        if obs_b is not None: b_memory=obs_b.copy()
        if b_memory is not None:
            delta=b_memory-b.xy; look_b=math.atan2(delta[1],delta[0])
        else:
            look_b=-math.pi+.18*math.sin(t*.5)
        if tick:
            a.step(dt,look_a,config["wheel_radius"],config["wheel_track"])
            b.turn=config["yaw_rate_b"] if b.route else config["scan_yaw_rate_b"]
            b.step(dt,look_b)
        # Referee independently evaluates post-motion geometry and both mark attempts.
        hit=block((*a.xy,.40),(*b.xy,1.05),arena["covers"])
        distance=np.linalg.norm(a.xy-b.xy)
        candidates=[]
        for i,(own,other,request) in enumerate(((a,b,request_a),(b,a,obs_b is not None))):
            delta=other.xy-own.xy; angle=abs(wrap(math.atan2(delta[1],delta[0])-own.yaw))
            valid=request and hit is None and distance<=8 and math.atan2(.65,distance)<=math.pi/4 and angle<=math.radians(10) and t>=cooldown
            locks[i]=locks[i]+dt if valid else 0.
            if locks[i]+1e-9>=config["lock_seconds"]: candidates.append(i)
        if candidates:
            for i in candidates:
                hp[1-i]-=1
                events.append({"time":round(t,2),"type":"mark","attacker":"AB"[i],"target":"AB"[1-i],"clear_los":hit is None})
                if i==0: last_hit_a=True
            cooldown=t+config["pause_seconds"];locks=[0.,0.];dwell=[0.,0.]
        target=arena["task_points"][point]
        inside=[np.linalg.norm(m.xy-target["center"])<=target["radius"] for m in (a,b)]
        scoring=[False,False]
        for i in range(2):
            if tick and inside[i] and not inside[1-i] and t>=cooldown and min(hp)>0:
                dwell[i]+=dt
                if dwell[i]>.5: scores[i]+=dt;scoring[i]=True
            else: dwell[i]=0.
        for mover in (a,b):
            for c in arena["covers"]:
                delta=np.maximum(np.abs(mover.xy-c["center"])-np.array(c["size"][:2])/2,0)
                clearance=min(clearance,float(np.linalg.norm(delta)))
            assert np.all(np.abs(mover.xy)<[5.6,3.7]),(t,mover.xy)
        separation=min(separation,float(distance))
        states.append({"t":round(t,3),"xy":[a.xy.tolist(),b.xy.tolist()],"yaw":[a.yaw,b.yaw],"speed":[a.v,b.v],
            "omega":[a.omega,b.omega],"wheels":a.wheels.tolist(),"state":controller.state,"reason":controller.reason,
            "observation_a":None if obs_a is None else obs_a.tolist(),"memory_a":None if controller.memory is None else controller.memory.tolist(),
            "memory_age":controller.memory_age,"los":hit is None,"block":hit,"lock":locks.copy(),"hp":hp.copy(),
            "score":[round(s,2) for s in scores],"point":point,"pause":max(0.,cooldown-t),"scoring":scoring,"contested":all(inside)})
        if min(hp)<=0: break
    events+=controller.transitions
    events.sort(key=lambda e:e["time"])
    result_a=(hp[0]>0,scores[0],hp[0]); result_b=(hp[1]>0,scores[1],hp[1])
    winner="A" if result_a>result_b else "B" if result_b>result_a else "draw"
    events.append({"time":states[-1]["t"],"type":"finish","winner":winner,"scores":states[-1]["score"]})
    return states,events,{"min_center_cover_distance":clearance,"min_actor_separation":separation,
        "max_speed_a":max(s["speed"][0] for s in states),"states":list(dict.fromkeys(s["state"] for s in states)),
        "winner":winner,"observation_contract":"Controller receives visible samples only; hidden position retained only as stale memory"}


if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    s,e,v=build(json.loads((root/"configs/demo_tactical_v2.json").read_text()),json.loads((root/"configs/arena_v0.json").read_text()))
    print(json.dumps({"events":e,"validation":v,"last":s[-1]},ensure_ascii=False,indent=2))
