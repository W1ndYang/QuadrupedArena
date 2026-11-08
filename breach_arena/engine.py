"""Breach Arena referee: 50 Hz simultaneous snapshot, legal observations only.

Actor 0 is the quadruped (attacker), actor 1 the human (defender). Controllers
receive the dict built by ``observation`` and return a command dict:

    {"goal": [x, y] | None, "via": [[x, y], ...], "look": [x, y] | None,
     "fire": bool, "throw": [x, y] | None, "posture": "upright" | "low" | None,
     "mode": "short label for replay"}

Controllers never see the Match object, the opponent's hidden position or lock state.
"""
import copy
import math

from .geometry import bearing, boxes, clear_ray, line_of_sight, wrap
from .motion import Mover
from .nav import Planner

QUAD, HUMAN = 0, 1


class Match:
    def __init__(self, game_map, rules, quad_controller, human_controller, record_every=2):
        self.map, self.rules = game_map, rules
        self.planner = Planner(game_map.covers, game_map.size, rules.planner.clearance_m, rules.planner.grid_m)
        self.obstacles = boxes(game_map.covers)
        qs, hs = game_map.spawns['quadruped'], game_map.spawns['human']
        up = rules.quadruped.postures.upright
        self.actors = [Mover(qs['xy'], math.radians(qs['yaw_degrees']), up.speed, up.accel, rules.quadruped.yaw_rate),
                       Mover(hs['xy'], math.radians(hs['yaw_degrees']), rules.human.speed, rules.human.accel,
                             rules.human.yaw_rate)]
        self.controllers = [quad_controller, human_controller]
        self.posture, self.posture_ready_at = 'upright', 0.0
        self.smokes, self.inventory, self.cast = [], rules.smoke.count, None
        self.lock_age, self.lock_prev = [0.0, 0.0], [False, False]
        self.fire = [False, False]
        self.human_disabled_until, self.quad_cooldown_until = -1.0, -1.0
        self.memory = [None, None]
        self.detected = [False, False]
        self.first_detection = [None, None]
        self.route_key = [None, None]
        self.no_path_until = [-1.0, -1.0]
        self.notices = [[], []]
        self.events, self.trace = [], []
        self.modes = ['IDLE', 'IDLE']
        self.collision_stops = [0, 0]
        self.min_separation = float('inf')
        self.tick, self.t, self.result = 0, 0.0, None
        self.record_every = record_every
        for c in self.controllers:
            if hasattr(c, 'attach'):
                c.attach(game_map, rules)

    # ------------------------------------------------------------------ helpers
    def height(self, i):
        if i == HUMAN:
            return self.rules.human.eye_height_m
        return getattr(self.rules.quadruped.postures, self.posture).height_m

    def active_smokes(self, t=None):
        t = self.t if t is None else t
        return [s for s in self.smokes if s['active_at'] <= t < s['expires_at']]

    def sees(self, i):
        me, other = self.actors[i], self.actors[1 - i]
        half = (self.rules.quadruped if i == QUAD else self.rules.human).detect_half_fov_deg
        if half < 180 and abs(wrap(bearing(me.xy, other.xy) - me.yaw)) > math.radians(half) + 1e-9:
            return False
        return line_of_sight(me.xy, self.height(i), other.xy, self.height(1 - i), self.obstacles, self.active_smokes())

    def notify(self, i, kind, **data):
        self.notices[i].append(dict(t=round(self.t, 3), type=kind, **data))

    def log(self, kind, **data):
        self.events.append(dict(t=round(self.t, 3), type=kind, **data))

    def observation(self, i, visible):
        me, other = self.actors[i], self.actors[1 - i]
        det = None
        if visible:
            det = dict(xy=[round(float(v), 3) for v in other.xy], yaw=round(other.yaw, 4), t=round(self.t, 3))
            if i == QUAD:
                det['stunned'] = self.t < self.human_disabled_until
        ttl = (self.rules.quadruped if i == QUAD else self.rules.human).memory_s
        if det is not None:
            self.memory[i] = copy.deepcopy(det)
        elif self.memory[i] is not None and self.t - self.memory[i]['t'] > ttl:
            self.memory[i] = None
        own = dict(xy=[round(float(v), 3) for v in me.xy], yaw=round(me.yaw, 4), speed=round(me.v, 3))
        if i == QUAD:
            own.update(posture=self.posture, smoke_left=self.inventory, casting=self.cast is not None,
                       cooldown_left=round(max(0.0, self.quad_cooldown_until - self.t), 3))
        else:
            own.update(stunned_left=round(max(0.0, self.human_disabled_until - self.t), 3))
        notices, self.notices[i] = self.notices[i], []
        return dict(t=round(self.t, 3), remaining=round(self.rules.duration_s - self.t, 3), self=own,
                    opponent=det, memory=copy.deepcopy(self.memory[i]),
                    smokes=[dict(center=s['center'], radius=s['radius'], active_at=s['active_at'],
                                 expires_at=s['expires_at']) for s in self.smokes if s['expires_at'] > self.t],
                    notices=notices)

    def lock_valid(self, i, visible):
        me, other = self.actors[i], self.actors[1 - i]
        tag = self.rules.tag
        if not (visible and self.fire[i]):
            return False
        if i == QUAD:
            if not self.rules.quadruped.can_tag or self.t < self.quad_cooldown_until or self.cast is not None:
                return False
            if self.t < self.posture_ready_at or self.t < self.human_disabled_until:
                return False
        elif self.t < self.human_disabled_until:
            return False
        d = math.dist(me.xy, other.xy)
        err = abs(wrap(bearing(me.xy, other.xy) - me.yaw))
        return (1e-8 < d <= tag.range_m and err <= math.radians(min(tag.acquire_half_angle_deg, tag.tracking_error_deg)) + 1e-9
                and abs(me.v) <= tag.max_shooter_speed_mps + 1e-9)

    def throw_check(self, target):
        q = self.actors[QUAD]
        sm = self.rules.smoke
        if self.inventory <= 0:
            return 'no smoke left'
        if self.cast is not None:
            return 'already casting'
        hx, hy = self.map.size[0] / 2, self.map.size[1] / 2
        if not (-hx < target[0] < hx and -hy < target[1] < hy):
            return 'landing outside arena'
        if math.dist(q.xy, target) > sm.throw_range_m + 1e-9:
            return f'out of range ({math.dist(q.xy, target):.2f} m > {sm.throw_range_m} m)'
        if not clear_ray((q.xy[0], q.xy[1], sm.throw_height_m), (target[0], target[1], sm.throw_height_m), self.obstacles):
            return 'throw line blocked at 1 m height'
        return None

    # ------------------------------------------------------------------ commands
    def apply(self, i, cmd):
        me = self.actors[i]
        self.modes[i] = cmd.get('mode', self.modes[i])
        if i == HUMAN and self.t < self.human_disabled_until:
            me.stop()
            self.route_key[i] = None
            self.fire[i] = False
            return None
        if i == QUAD:
            want = cmd.get('posture')
            if want in ('upright', 'low') and want != self.posture:
                self.posture = want
                self.posture_ready_at = self.t + self.rules.quadruped.posture_switch_s
                p = getattr(self.rules.quadruped.postures, want)
                me.set_limits(p.speed, p.accel)
                self.log('posture', posture=want)
            if self.cast is None and cmd.get('throw') is not None:
                why = self.throw_check(cmd['throw'])
                if why is None:
                    self.inventory -= 1
                    self.cast = dict(target=[float(cmd['throw'][0]), float(cmd['throw'][1])], settled_at=None)
                    self.log('smoke_cast_started', target=self.cast['target'], at=[round(float(v), 3) for v in me.xy])
                    self.notify(QUAD, 'throw_accepted', target=self.cast['target'])
                else:
                    self.log('throw_rejected', target=list(cmd['throw']), reason=why)
                    self.notify(QUAD, 'throw_rejected', target=list(cmd['throw']), reason=why)
            if self.cast is not None:
                me.stop()
                self.route_key[i] = None
                self.fire[i] = False
                if abs(me.v) <= self.rules.tag.max_shooter_speed_mps:
                    if self.cast['settled_at'] is None:
                        self.cast['settled_at'] = self.t
                    if self.t - self.cast['settled_at'] >= self.rules.smoke.cast_s - 1e-9:
                        sm = self.rules.smoke
                        s = dict(center=self.cast['target'], radius=sm.radius_m, height=sm.height_m,
                                 active_at=round(self.t + sm.fuse_s, 4), expires_at=round(self.t + sm.fuse_s + sm.duration_s, 4))
                        self.smokes.append(s)
                        self.log('smoke_released', **s)
                        self.notify(QUAD, 'smoke_released', center=s['center'], active_at=s['active_at'])
                        self.cast = None
                return None
        self.fire[i] = bool(cmd.get('fire'))
        goal = cmd.get('goal')
        if goal is None:
            me.stop()
            self.route_key[i] = None
        else:
            via = [tuple(v) for v in cmd.get('via') or []]
            key = (tuple(round(float(g), 3) for g in goal), tuple(via))
            retry = self.no_path_until[i] if key == self.route_key[i] else -1.0
            if (key != self.route_key[i] or not me.path) and math.dist(me.xy, goal) > 0.05 and self.t >= retry:
                path = self.planner.plan_via(me.xy, via + [tuple(goal)])
                if path is None:
                    me.stop()
                    self.no_path_until[i] = self.t + 0.5
                    if key != self.route_key[i]:
                        self.notify(i, 'no_path', goal=list(goal))
                        self.log('no_path', actor=i, goal=list(goal), at=[round(float(v), 3) for v in me.xy])
                else:
                    me.navigate(path)
                    self.no_path_until[i] = -1.0
                self.route_key[i] = key
        look = cmd.get('look')
        return None if look is None else bearing(me.xy, look)

    # ------------------------------------------------------------------ loop
    def step(self):
        dt = self.rules.dt
        self.t = round(self.tick * dt, 6)
        vis = [self.sees(QUAD), self.sees(HUMAN)]
        for i in (QUAD, HUMAN):
            if vis[i] != self.detected[i]:
                self.log('detected' if vis[i] else 'lost', actor=i)
                self.detected[i] = vis[i]
                if vis[i] and self.first_detection[i] is None:
                    self.first_detection[i] = round(self.t, 3)
        valid = [self.lock_valid(i, vis[i]) for i in (QUAD, HUMAN)]
        if self.tick:
            for i in (QUAD, HUMAN):
                self.lock_age[i] = self.lock_age[i] + dt if valid[i] and self.lock_prev[i] else 0.0
        self.lock_prev = list(valid)
        hits = [valid[i] and self.lock_age[i] + 1e-9 >= self.rules.tag.seconds for i in (QUAD, HUMAN)]
        in_goal = math.dist(self.actors[QUAD].xy, self.map.goal) <= self.map.goal_radius + 1e-9
        outcome = None
        if in_goal and hits[HUMAN]:
            outcome = ('draw', 'goal_and_tag_same_tick')
        elif in_goal:
            outcome = ('quadruped', 'goal')
        elif hits[HUMAN]:
            outcome = ('human', 'tag')
            self.log('tag', shooter=HUMAN)
        elif self.t >= self.rules.duration_s - 1e-9:
            outcome = ('human', 'timeout')
        if outcome is None and hits[QUAD]:
            self.human_disabled_until = self.t + self.rules.human.stun_s
            self.quad_cooldown_until = self.t + self.rules.quadruped.tag_cooldown_s
            self.lock_age = [0.0, 0.0]
            self.lock_prev = [False, False]
            self.log('tag', shooter=QUAD, stun_until=round(self.human_disabled_until, 3))
            self.notify(QUAD, 'you_tagged_human', stun_s=self.rules.human.stun_s)
            self.notify(HUMAN, 'you_are_stunned', stun_s=self.rules.human.stun_s)
        if self.tick % self.record_every == 0 or outcome:
            self.record(vis)
        if outcome:
            self.result = dict(winner=outcome[0], reason=outcome[1], t=round(self.t, 3))
            self.log('finish', winner=outcome[0], reason=outcome[1])
            return False
        obs = [self.observation(i, vis[i]) for i in (QUAD, HUMAN)]
        cmds = [self.controllers[i].decide(obs[i]) for i in (QUAD, HUMAN)]
        looks = [self.apply(i, cmds[i]) for i in (QUAD, HUMAN)]
        before = [(a.xy.copy(), a.yaw, a.v, list(a.path)) for a in self.actors]
        slack = self.rules.planner.motion_slack_m
        for i, a in enumerate(self.actors):
            if i == HUMAN and self.t < self.human_disabled_until:
                a.step(dt, None)
                continue
            a.step(dt, looks[i])
            if not self.planner.segment_ok(before[i][0], a.xy, slack=slack, step=0.05):
                a.xy, a.v = before[i][0], 0.0
                a.stop()
                self.route_key[i] = None
                self.collision_stops[i] += 1
        sep = math.dist(self.actors[0].xy, self.actors[1].xy)
        if sep < self.rules.min_separation_m - 1e-9:
            for i, a in enumerate(self.actors):       # bodies block each other: undo, keep routes
                a.xy, a.v, a.path = before[i][0], 0.0, before[i][3]
                self.collision_stops[i] += 1
            sep = math.dist(self.actors[0].xy, self.actors[1].xy)
        self.min_separation = min(self.min_separation, sep)
        for s in self.smokes:
            if not s.get('_expired') and self.t >= s['expires_at']:
                s['_expired'] = True
                self.log('smoke_expired', center=s['center'])
        self.tick += 1
        return True

    def record(self, vis):
        self.trace.append(dict(
            t=round(self.t, 3),
            xy=[[round(float(v), 3) for v in a.xy] for a in self.actors],
            yaw=[round(a.yaw, 3) for a in self.actors],
            v=[round(a.v, 3) for a in self.actors],
            visible=list(vis), lock=[round(x, 3) for x in self.lock_age], fire=list(self.fire),
            posture=self.posture, mode=list(self.modes),
            casting=self.cast is not None, stunned=self.t < self.human_disabled_until,
            path=[[[round(float(v), 2) for v in p] for p in a.path[:6]] for a in self.actors]))

    def run(self, max_ticks=None, on_tick=None):
        n = 0
        while self.step():
            n += 1
            if on_tick is not None:
                on_tick(self)
            if max_ticks is not None and n >= max_ticks:
                break
        return self.result

    def summary(self):
        return dict(map=self.map.id, rules=self.rules.version, result=self.result,
                    smoke_used=self.rules.smoke.count - self.inventory,
                    first_detection=self.first_detection, min_separation_m=round(self.min_separation, 3),
                    collision_stops=self.collision_stops,
                    distance_m=[round(a.distance, 2) for a in self.actors])
