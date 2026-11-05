"""Kinematic walking with independently bounded torso facing.

Travel keeps the existing stop-before-corner rule and safe_step collision checks.
The torso (mover.yaw, shared by sensing and firing) may turn while the feet brake
or follow a route. This is a walking/strafe reference, not contact dynamics.
"""
import math
import numpy as np
from tactical_logic import Mover,wrap


class HumanoidMover(Mover):
    independent_facing=True

    def __init__(self,*args):
        super().__init__(*args)
        self.travel_yaw=self.yaw

    def step(self,dt,look=None,radius=.089,track=.4):
        torso=self.yaw;self.yaw=self.travel_yaw
        while self.route and math.dist(self.route[0],self.xy)<.055:self.route.pop(0)
        heading=self.travel_yaw
        if self.route:
            delta=self.route[0]-self.xy;heading=math.atan2(delta[1],delta[0])
        if abs(wrap(heading-self.travel_yaw))>.025:
            route=self.route;self.route=[]
            super().step(dt,self.travel_yaw if abs(self.v)>.001 else heading,radius,track)
            self.route=route
        else:super().step(dt,self.travel_yaw,radius,track)
        self.travel_yaw=self.yaw
        desired=self.travel_yaw if look is None else float(look)
        change=float(np.clip(wrap(desired-torso),-self.turn*dt,self.turn*dt))
        self.yaw=torso+change;self.omega=change/dt
