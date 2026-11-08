"""Kinematic reference motion: speed/acceleration limits, braking before turns, yaw-rate limit.

Translation follows a polyline independent of body yaw (the yaw is steered toward the
look target, or the travel direction when no look target is given). This is not a
contact-dynamics gait; it is the same level of abstraction as the earlier fixed-goal
prototypes in this repository.
"""
import math

import numpy as np

from .geometry import wrap


def corner_speed(vmax, a, b, c, floor=0.8):
    """Speed allowed at vertex b for a turn a->b->c (no limit for straight lines)."""
    u = np.subtract(b, a)
    w = np.subtract(c, b)
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-9 or nw < 1e-9:
        return vmax
    turn = math.acos(max(-1.0, min(1.0, float(u @ w) / (nu * nw))))
    if turn < math.radians(12):
        return vmax
    return max(min(floor, vmax), vmax * (1.0 - turn / math.pi))


class Mover:
    def __init__(self, xy, yaw, vmax, accel, yaw_rate):
        self.xy = np.asarray(xy, float)
        self.yaw = float(yaw)
        self.v = 0.0
        self.omega = 0.0
        self.vmax, self.accel, self.yaw_rate = vmax, accel, yaw_rate
        self.path = []
        self.distance = 0.0

    def set_limits(self, vmax, accel):
        self.vmax, self.accel = vmax, accel

    def navigate(self, path):
        self.path = [np.asarray(p, float) for p in path]

    def stop(self):
        self.path = []

    def _allowed_speed(self):
        """Braking envelope from upcoming corners and the end of the path."""
        allowed = self.vmax
        prev = self.xy
        dist = 0.0
        for i, p in enumerate(self.path):
            dist += float(np.linalg.norm(p - prev))
            nxt = self.path[i + 1] if i + 1 < len(self.path) else None
            vc = 0.0 if nxt is None else corner_speed(self.vmax, prev if i else self.xy, p, nxt)
            allowed = min(allowed, math.sqrt(vc * vc + 2 * self.accel * dist))
            prev = p
            if dist > self.vmax * self.vmax / (2 * self.accel) + 0.5:
                break
        return allowed

    def step(self, dt, look_yaw=None):
        """Advance one tick; returns nothing. Caller validates collisions."""
        if self.path:
            target_v = min(self.vmax, self._allowed_speed())
        else:
            target_v = 0.0
        if self.v < target_v:
            self.v = min(target_v, self.v + self.accel * dt)
        else:
            self.v = max(target_v, self.v - self.accel * dt)
        move = self.v * dt
        heading = None
        while move > 1e-12 and self.path:
            seg = self.path[0] - self.xy
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                self.path.pop(0)
                continue
            heading = math.atan2(seg[1], seg[0])
            if move >= L:
                self.xy = self.path.pop(0).copy()
                self.distance += L
                move -= L
            else:
                self.xy = self.xy + seg / L * move
                self.distance += move
                move = 0.0
        if not self.path and self.v > 0 and move > 0:
            self.v = 0.0
        desired = look_yaw if look_yaw is not None else heading
        if desired is None:
            self.omega = 0.0
            return
        err = wrap(desired - self.yaw)
        turn = max(-self.yaw_rate * dt, min(self.yaw_rate * dt, err))
        self.yaw = wrap(self.yaw + turn)
        self.omega = turn / dt
