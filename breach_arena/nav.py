"""Grid A* with wall clearance, string pulling and exact segment checks."""
import heapq
import math

import numpy as np

from .geometry import boxes, wall_clearance

SQ2 = math.sqrt(2.0)
MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


class Planner:
    def __init__(self, covers, arena_size=(24.0, 16.0), clearance=0.42, grid=0.1):
        self.obstacles = boxes(covers)
        self.clearance, self.grid = clearance, grid
        hx, hy = arena_size[0] / 2, arena_size[1] / 2
        self.bounds = (-hx, hx, -hy, hy)
        self.nx = int(round(2 * hx / grid)) + 1
        self.ny = int(round(2 * hy / grid)) + 1
        xs = -hx + np.arange(self.nx) * grid
        ys = -hy + np.arange(self.ny) * grid
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        free = (X >= -hx + clearance) & (X <= hx - clearance) & (Y >= -hy + clearance) & (Y <= hy - clearance)
        for b in self.obstacles:
            dx = np.maximum(np.maximum(b[0] - X, X - b[1]), 0)
            dy = np.maximum(np.maximum(b[2] - Y, Y - b[3]), 0)
            free &= np.hypot(dx, dy) >= clearance
        self.free_grid = free

    # ------------------------------------------------------------------ queries
    def point_ok(self, p, slack=0.0):
        x0, x1, y0, y1 = self.bounds
        c = self.clearance - slack
        if not (x0 + c - 1e-9 <= p[0] <= x1 - c + 1e-9 and y0 + c - 1e-9 <= p[1] <= y1 - c + 1e-9):
            return False
        return wall_clearance(p, self.obstacles) >= c - 1e-9

    def segment_ok(self, a, b, slack=0.0, step=0.04):
        n = max(1, int(math.ceil(math.dist(a, b) / step)))
        return all(self.point_ok((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n), slack)
                   for k in range(n + 1))

    def cell(self, p):
        return (int(round((p[0] - self.bounds[0]) / self.grid)), int(round((p[1] - self.bounds[2]) / self.grid)))

    def xy(self, c):
        return (self.bounds[0] + c[0] * self.grid, self.bounds[2] + c[1] * self.grid)

    def snap(self, p, radius=0.6):
        """Nearest exactly-free point (grid cell centre) within radius, else None."""
        if self.point_ok(p):
            return tuple(p)
        c = self.cell(p)
        r = int(math.ceil(radius / self.grid))
        best = None
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                n = (c[0] + di, c[1] + dj)
                if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny and self.free_grid[n]:
                    q = self.xy(n)
                    d = math.dist(q, p)
                    if d <= radius and (best is None or d < best[0]):
                        best = (d, q)
        return None if best is None else best[1]

    # ------------------------------------------------------------------ planning
    def free_cell(self, p, radius=0.6):
        c = self.cell(p)
        if 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and self.free_grid[c]:
            return c
        r = int(math.ceil(radius / self.grid))
        best = None
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                n = (c[0] + di, c[1] + dj)
                if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny and self.free_grid[n]:
                    d = math.dist(self.xy(n), p)
                    if best is None or d < best[0]:
                        best = (d, n)
        return None if best is None else best[1]

    def _astar(self, a, b):
        s, g = self.free_cell(a), self.free_cell(b)
        if s is None or g is None:
            return None
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        openq, came, cost = [(h(s), 0.0, s)], {s: None}, {s: 0.0}
        while openq:
            _, c0, c = heapq.heappop(openq)
            if c == g:
                break
            if c0 > cost[c] + 1e-9:
                continue
            for di, dj, w in MOVES:
                n = (c[0] + di, c[1] + dj)
                if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny and self.free_grid[n]:
                    nc = c0 + w
                    if nc < cost.get(n, 1e18):
                        cost[n], came[n] = nc, c
                        heapq.heappush(openq, (nc + h(n), nc, n))
        if g not in came:
            return None
        cells, c = [], g
        while c is not None:
            cells.append(c)
            c = came[c]
        return [self.xy(c) for c in reversed(cells)]

    def plan(self, start, goal):
        """Shortest smoothed path start -> goal; points after start. None if unreachable."""
        a = tuple(start)
        b = self.snap(goal)
        if b is None:
            return None
        if self.segment_ok(start, b, slack=0.02):
            return [b]
        cells = self._astar(a, b)
        if cells is None:
            return None
        pts = [tuple(start)] + cells[1:-1] + [b]
        out, i = [], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_ok(pts[i], pts[j], slack=0.02):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def plan_via(self, start, points):
        """Chain plans through intermediate points; returns full polyline after start."""
        path, cur = [], tuple(start)
        for p in points:
            leg = self.plan(cur, p)
            if leg is None:
                return None
            path += leg
            cur = leg[-1]
        return path


def path_length(start, path):
    pts = [tuple(start)] + [tuple(p) for p in path]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
