"""Collision-checked kinematic path references, with speed-limited curved turns.

This is still a kinematic reference follower, not wheel/contact dynamics.
Only the quadruped opts in; the defender and historical configs are unchanged.
"""
import math
import numpy as np
from tactical_logic import Mover,wrap


def rounded_route(start,route,planner):
    raw=[np.asarray(start,dtype=float)]+[np.asarray(p,dtype=float) for p in route]
    raw=[p for i,p in enumerate(raw) if i==0 or math.dist(p,raw[i-1])>1e-8]
    if len(raw)<3:return [p.copy() for p in raw[1:]]
    result=[raw[0]]
    for i in range(1,len(raw)-1):
        a,b,c=raw[i-1:i+2];u=b-a;v=c-b;la=np.linalg.norm(u);lb=np.linalg.norm(v)
        u=u/la;v=v/lb;angle=math.acos(float(np.clip(u@v,-1,1)))
        selected=None
        if .04<angle<2.7:
            # Quadratic fillet inside the already-valid two-segment corridor.
            # Never move a point through a wall to make a visually nicer curve.
            radius=min(.85,.42*la,.42*lb)
            for scale in (1.,.7,.45,.25):
                r=radius*scale;p=b-u*r;q=b+v*r
                n=max(8,math.ceil(2*r/.045))
                curve=[(1-t)**2*p+2*t*(1-t)*b+t*t*q for t in np.linspace(0,1,n+1)]
                chain=[result[-1],*curve,c]
                if all(planner.segment(x,y) for x,y in zip(chain,chain[1:])):
                    selected=curve;break
        result.extend(selected if selected is not None else [b])
    result.append(raw[-1]);clean=[result[0]]
    for p in result[1:]:
        if math.dist(clean[-1],p)>1e-8:clean.append(p)
    if not all(planner.segment(x,y) for x,y in zip(clean,clean[1:])):
        return [p.copy() for p in raw[1:]]
    return [p.copy() for p in clean[1:]]


class ContinuousMover(Mover):
    continuous_path=True
    reverse_allowed=False
    reverse_limit=.8

    def navigate(self,route):
        super().navigate(route)
        self.travel_reverse=None
        self.corner_curvatures=[];previous=self.xy
        for i,p in enumerate(self.route):
            curvature=float('inf')
            if i+1<len(self.route):
                u=p-previous;v=self.route[i+1]-p;la=np.linalg.norm(u);lb=np.linalg.norm(v)
                angle=math.acos(float(np.clip(u@v/max(1e-9,la*lb),-1,1)))
                curvature=float('inf') if angle>.4 else angle/max(.015,.5*(la+lb))
            self.corner_curvatures.append(curvature);previous=p

    def speed_bound(self,dt):
        # Future corner limits are propagated backwards by braking distance.
        bound=self.vmax;distance=0.;previous=self.xy
        for i,p in enumerate(self.route):
            incoming=p-previous;length=float(np.linalg.norm(incoming));distance+=length
            curvature=self.corner_curvatures[i]
            cap=min(self.vmax,.72*self.turn/curvature) if curvature>1e-6 else self.vmax
            # Reserve this integration step as well as the remaining braking
            # distance. sqrt(2*a*d) alone overshoots in discrete simulation and
            # used to force an abrupt velocity reset at the final point.
            bound=min(bound,max(0.,math.sqrt(cap*cap+2*self.accel*distance+(self.accel*dt)**2)-self.accel*dt))
            previous=p
        return bound

    def step(self,dt,look=None,radius=.089,track=.4):
        while self.route and math.dist(self.route[0],self.xy)<1e-8:
            self.route.pop(0);self.corner_curvatures.pop(0)
        desired=0.;heading=self.yaw if look is None else look
        if self.route:
            d=self.route[0]-self.xy;bearing=math.atan2(d[1],d[0])
            if getattr(self,'travel_reverse',None) is None:
                self.travel_reverse=self.reverse_allowed and abs(wrap(bearing-self.yaw))>math.pi/2
            reverse=self.travel_reverse
            heading=bearing+(math.pi if reverse else 0.)
            desired=min(self.speed_bound(dt),self.reverse_limit if reverse else self.vmax)
            if reverse:desired=-desired
            if abs(wrap(heading-self.yaw))>.22:desired=0.
        error=wrap(heading-self.yaw);old_yaw=self.yaw
        self.yaw+=float(np.clip(error,-self.turn*dt,self.turn*dt));self.omega=wrap(self.yaw-old_yaw)/dt
        self.v+=float(np.clip(desired-self.v,-self.accel*dt,self.accel*dt))
        remaining=abs(self.v)*dt;travelled=0.
        if self.route:
            while self.route and remaining>1e-10:
                delta=self.route[0]-self.xy;length=float(np.linalg.norm(delta))
                if length<1e-8:self.route.pop(0);self.corner_curvatures.pop(0);continue
                step=min(remaining,length);self.xy+=delta*(step/length)
                travelled+=step;remaining-=step
                if length-step<1e-8:
                    stop=math.isinf(self.corner_curvatures[0])
                    self.route.pop(0);self.corner_curvatures.pop(0)
                    if stop:self.v=0.;remaining=0.
            # Reaching the final reference cannot overshoot the destination.
            if not self.route:self.v=0.
        else:
            # A new explicit stop brakes rather than resetting velocity.
            travelled=abs(self.v)*dt;self.xy+=self.v*dt*np.array([math.cos(self.yaw),math.sin(self.yaw)])
        self.distance+=travelled
        signed=math.copysign(travelled/dt,self.v) if self.v else 0.
        self.wheels+=np.array([signed-self.omega*track/2,signed+self.omega*track/2]*2)/radius*dt
