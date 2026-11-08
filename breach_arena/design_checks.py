"""Static map design checks (no bots, no referee): sight lines, smoke placement, lure timing.

Idealised kinematics (accelerate then cruise, no turn braking) and a static observer who tags
after 1.2 s of continuous in-range sight (0.2 s reaction + 1.0 s aim). These numbers are design
estimates, not match outcomes; `python -m breach_arena matrix` runs the real referee.

Per map:
  no_smoke     exposure of every throw/staging run to the goal and stand-off guards without smoke
  smoke        per throw spot and guard: goal smoke, best guard-specific smoke, best single smoke
               against every guard at once ("universal")
  lure         feint distance, unseen relocation, commit distance of a hunter who turns back
               0.5 s after the dash starts
"""
import json
import math
from pathlib import Path

from .config import GameMap, load_rules
from .geometry import boxes, clear_ray, line_of_sight
from .nav import Planner, path_length

DT, TAG_WINDOW, WIN, LOSE = 0.02, 1.2, 0.6, 1.3


class Ctx:
    def __init__(self, m, rules):
        self.m, self.r = m, rules
        self.nav = Planner(m.covers, m.size, rules.planner.clearance_m, rules.planner.grid_m)
        self.obs = boxes(m.covers)
        self.goal = m.goal
        self.guards = {'on_point': m.goal, **{k: m.points[k] for k in m.names('standoff')}}
        sm = rules.smoke
        self.smoke_on, self.smoke_off = sm.cast_s + sm.fuse_s, sm.cast_s + sm.fuse_s + sm.duration_s
        self.eye, self.qh = rules.human.eye_height_m, rules.quadruped.postures.upright.height_m

    def route(self, start, via):
        pts = [self.m.resolve(v) for v in via] + [self.goal]
        return [tuple(start)] + (self.nav.plan_via(start, pts) or [])

    def profile(self, path, t0=0.0, v=None, a=None, stop_at_goal=True):
        up = self.r.quadruped.postures.upright
        v, a = v or up.speed, a or up.accel
        L = path_length(path[0], path[1:])
        out, s, vel, t = [], 0.0, 0.0, t0
        while s < L:
            p = point_at(path, s)
            out.append((t, p))
            if stop_at_goal and math.dist(p, self.goal) <= self.m.goal_radius:
                return out
            vel = min(v, vel + a * DT)
            s += vel * DT
            t += DT
        out.append((t, path[-1]))
        return out

    def exposure(self, samples, observer, smoke=None):
        longest = run = 0.0
        sm = self.r.smoke
        for t, p in samples:
            clouds = [dict(center=smoke, radius=sm.radius_m, height=sm.height_m)] if smoke and self.smoke_on <= t <= self.smoke_off else ()
            ok = math.dist(observer, p) <= self.r.tag.range_m and line_of_sight(observer, self.eye, p, self.qh, self.obs, clouds)
            run = run + DT if ok else 0.0
            longest = max(longest, run)
        return round(longest, 2)

    def landings(self, T, step=0.4):
        rng = self.r.smoke.throw_range_m
        n = int(rng / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                p = (T[0] + i * step, T[1] + j * step)
                if math.dist(p, T) <= rng and clear_ray((*T, 1.0), (*p, 1.0), self.obs):
                    yield p


def point_at(path, s):
    for a, b in zip(path, path[1:]):
        L = math.dist(a, b)
        if s <= L:
            u = s / L if L else 0.0
            return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        s -= L
    return path[-1]


def check_map(map_id, rules=None):
    rules = rules or load_rules()
    m = GameMap.load(map_id)
    c = Ctx(m, rules)
    plays = m.plays
    rep = dict(id=m.id, title=m.title, rules=rules.version, no_smoke={}, smoke={}, lure={})
    spots = {**{k: v['dash_via'] for k, v in plays.get('stagings', {}).items()},
             **{k: v['run_via'] for k, v in plays.get('throws', {}).items()}}
    for name, via in spots.items():
        s = c.profile(c.route(m.points[name], via))
        rep['no_smoke'][name] = dict(hidden_from=[g for g, gp in c.guards.items()
                                                  if not line_of_sight(gp, c.eye, m.points[name], c.qh, c.obs)],
                                     exposure={g: c.exposure(s, gp) for g, gp in c.guards.items()})
    for tname, T in plays.get('throws', {}).items():
        tp = m.points[tname]
        s = c.profile(c.route(tp, T['run_via']), t0=rules.smoke.cast_s)
        goal_ok = math.dist(tp, c.goal) <= rules.smoke.throw_range_m and clear_ray((*tp, 1.0), (*c.goal, 1.0), c.obs)
        row = dict(goal_throw_ok=goal_ok, vs={})
        cands = list(c.landings(tp))
        for g, gp in c.guards.items():
            best = min(cands, key=lambda p: (c.exposure(s, gp, p), math.dist(p, c.goal)))
            row['vs'][g] = dict(goal_smoke=c.exposure(s, gp, c.goal) if goal_ok else None,
                                best_smoke_at=[round(best[0], 1), round(best[1], 1)], best_smoke=c.exposure(s, gp, best))
        uni = min(cands, key=lambda p: (max(c.exposure(s, gp, p) for gp in c.guards.values()), math.dist(p, c.goal)))
        row['universal'] = dict(at=[round(uni[0], 1), round(uni[1], 1)],
                                worst=max(c.exposure(s, gp, uni) for gp in c.guards.values()))
        rep['smoke'][tname] = row
    hv = rules.human.speed
    for lname, L in plays.get('lures', {}).items():
        stg = L['staging']
        dash = c.profile(c.route(m.points[stg], plays['stagings'][stg]['dash_via']))
        chase = [c.goal] + (c.nav.plan_via(c.goal, [m.resolve(v) for v in L['chase']]) or [])
        Lc = path_length(chase[0], chase[1:])
        sight = {}
        d = 1.0
        while d <= Lc + 1e-9:
            hd = point_at(chase, d)
            back = [hd] + (c.nav.plan(hd, c.goal) or [])
            Lb = path_length(back[0], back[1:])
            run = longest = 0.0
            for t, q in dash:
                hp = point_at(back, min(max(0.0, t - 0.5) * hv, Lb))
                ok = math.dist(hp, q) <= rules.tag.range_m and line_of_sight(hp, c.eye, q, c.qh, c.obs)
                run = run + DT if ok else 0.0
                longest = max(longest, run)
            sight[round(d, 1)] = round(longest, 2)
            d += 0.5
        commit = None
        for dd in sorted(sight, reverse=True):
            if sight[dd] <= 1.0:
                commit = dd
            else:
                break
        rep['lure'][lname] = dict(feint_dist_goal=round(math.dist(m.points[L['feint']], c.goal), 1),
                                  feint_seen_from_goal=line_of_sight(c.goal, c.eye, m.points[L['feint']], c.qh, c.obs),
                                  dash_s=round(dash[-1][0], 2), chase_m=round(Lc, 1), commit_distance_m=commit,
                                  returning_hunter_sight_s=sight)
    rep['summary'] = dict(
        guard_beats_no_smoke_rush=all(v['exposure']['on_point'] >= LOSE for v in rep['no_smoke'].values()),
        spots_hidden_from_guards=all(len(v['hidden_from']) == len(c.guards) for v in rep['no_smoke'].values()),
        goal_smoke_beats_on_point=[t for t, r in rep['smoke'].items() if (r['vs']['on_point']['goal_smoke'] is not None
                                                                          and r['vs']['on_point']['goal_smoke'] <= WIN)],
        goal_smoke_fails_vs_standoff={g: [t for t, r in rep['smoke'].items() if r['goal_throw_ok'] and r['vs'][g]['goal_smoke'] >= LOSE]
                                      for g in c.guards if g != 'on_point'},
        universal_smoke_from=[t for t, r in rep['smoke'].items() if r['universal']['worst'] <= WIN],
        lure_commit_m={k: v['commit_distance_m'] for k, v in rep['lure'].items()})
    return rep, c


def draw(ax, rep, c):
    from matplotlib.patches import Circle, Rectangle
    m = c.m
    W, H = m.size
    ax.add_patch(Rectangle((-W / 2, -H / 2), W, H, fill=False, lw=1.2, ec='#444'))
    for w in m.covers:
        (cx, cy), (sx, sy, sz) = w['center'], w['size']
        low = sz < 1.0
        ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc='#d9a441' if low else '#9aa5b1',
                               ec='#7a5a12' if low else '#5f6b78', lw=.8, zorder=2))
    ax.add_patch(Circle(c.goal, m.goal_radius, fc='#3aa76d', alpha=.35, ec='#1f7a4c', lw=1.5, zorder=3))
    for tname, T in m.plays.get('throws', {}).items():
        path = c.route(m.points[tname], T['run_via'])
        ax.plot(*zip(*path), color='#c0392b', lw=.8, ls='--', zorder=4)
        u = rep['smoke'][tname]['universal']
        if u['worst'] <= WIN:
            ax.add_patch(Circle(u['at'], c.r.smoke.radius_m, fc='#bbbbbb', alpha=.2, ec='#888', ls='--', lw=.6, zorder=1))
    for pname, P in m.plays.get('stagings', {}).items():
        path = c.route(m.points[pname], P['dash_via'])
        ax.plot(*zip(*path), color='#8e44ad', lw=1.2, zorder=4)
    for k, kind in m.kinds.items():
        style = dict(throw=('^', '#c0392b'), staging=('o', '#8e44ad'), standoff=('s', '#2c3e91'),
                     feint=('*', '#e67e22'), door=('d', '#333333')).get(kind)
        if style:
            ax.plot(*m.points[k], marker=style[0], color=style[1], ms=7, ls='', zorder=5)
            ax.annotate(k, m.points[k], xytext=(4, 3), textcoords='offset points', fontsize=6.5, color=style[1])
    ax.plot(*m.spawns['quadruped']['xy'], 'o', color='#16a085', ms=8)
    ax.plot(*m.spawns['human']['xy'], 'D', color='#2c3e91', ms=6)
    ax.set_xlim(-W / 2 - .3, W / 2 + .3)
    ax.set_ylim(-H / 2 - .3, H / 2 + .3)
    ax.set_aspect('equal')
    ax.grid(alpha=.15)
    ax.set_title(f'{m.id} {m.title}', fontsize=10)


def run_checks(map_ids, out_dir, rules=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'PingFang SC'):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = f
            break
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(map_ids), 1, figsize=(9, 6.2 * len(map_ids)), squeeze=False)
    summary = {}
    for ax, mid in zip(axes[:, 0], map_ids):
        rep, c = check_map(mid, rules)
        (out / f'checks_{mid}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
        draw(ax, rep, c)
        summary[mid] = rep['summary']
    fig.tight_layout()
    fig.savefig(out / 'overview.png', dpi=120)
    return summary
