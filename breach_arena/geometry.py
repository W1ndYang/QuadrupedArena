"""Exact geometry shared by referee, planner and design checks.

Walls are axis-aligned boxes from z = 0 to their height. Line of sight is a 3D
segment between representative points; smoke is a vertical cylinder that blocks
any segment passing through it (zero penetration).
"""
import math

import numpy as np


def boxes(covers):
    """[(x0, x1, y0, y1, z0, z1)] from map covers with centre and size [sx, sy, h]."""
    out = []
    for c in covers:
        (cx, cy), (sx, sy, sz) = c['center'], c['size']
        out.append((cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, 0.0, sz))
    return out


def clear_ray(a, b, obstacles):
    """True if the closed 3D segment a-b touches no box."""
    for box in obstacles:
        lo, hi = 0.0, 1.0
        for k in range(3):
            o, d = a[k], b[k] - a[k]
            mn, mx = box[2 * k], box[2 * k + 1]
            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    lo, hi = 2.0, -1.0
                    break
            else:
                t0, t1 = sorted(((mn - o) / d, (mx - o) / d))
                lo, hi = max(lo, t0), min(hi, t1)
        if lo <= hi:
            return False
    return True


def smoke_chord(a, b, center, radius, height):
    """Length of segment a-b inside a vertical smoke cylinder."""
    dx, dy, dz = (b[k] - a[k] for k in range(3))
    px, py = a[0] - center[0], a[1] - center[1]
    aa = dx * dx + dy * dy
    if aa < 1e-18:
        if px * px + py * py >= radius * radius:
            return 0.0
        lo, hi = 0.0, 1.0
    else:
        bb = 2 * (px * dx + py * dy)
        cc = px * px + py * py - radius * radius
        disc = bb * bb - 4 * aa * cc
        if disc <= 0:
            return 0.0
        lo = max(0.0, (-bb - math.sqrt(disc)) / (2 * aa))
        hi = min(1.0, (-bb + math.sqrt(disc)) / (2 * aa))
    if abs(dz) > 1e-12:
        t0, t1 = sorted((-a[2] / dz, (height - a[2]) / dz))
        lo, hi = max(lo, t0), min(hi, t1)
    elif not 0 <= a[2] <= height:
        return 0.0
    return max(0.0, hi - lo) * math.sqrt(dx * dx + dy * dy + dz * dz)


def line_of_sight(a_xy, a_h, b_xy, b_h, obstacles, smokes=()):
    """Walls and active smokes (dicts with center/radius/height) between two points."""
    a, b = (a_xy[0], a_xy[1], a_h), (b_xy[0], b_xy[1], b_h)
    if not clear_ray(a, b, obstacles):
        return False
    return not any(smoke_chord(a, b, s['center'], s['radius'], s['height']) > 1e-9 for s in smokes)


def wall_clearance(p, obstacles):
    """Planar distance from p to the nearest wall box (0 inside)."""
    if not obstacles:
        return float('inf')
    arr = np.asarray(obstacles)
    dx = np.maximum(np.maximum(arr[:, 0] - p[0], p[0] - arr[:, 1]), 0.0)
    dy = np.maximum(np.maximum(arr[:, 2] - p[1], p[1] - arr[:, 3]), 0.0)
    return float(np.min(np.hypot(dx, dy)))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bearing(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])
