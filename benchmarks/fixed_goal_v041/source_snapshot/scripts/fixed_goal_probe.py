"""Isolated fixed-goal virtual-tag research prototype. CPU kinematics only.

Policies receive public map, own state and legal detections, never hidden truth.
No API calls, robot actuation, projectile physics or historical-rule mutations.
"""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from continuous_motion import ContinuousMover, rounded_route
from humanoid_motion import HumanoidMover
from tactical_logic import wrap
from trial_motion import Planner, clearance, path_length

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boxes(covers):
    return [(c['center'][0]-c['size'][0]/2, c['center'][0]+c['size'][0]/2,
             c['center'][1]-c['size'][1]/2, c['center'][1]+c['size'][1]/2,
             0., c['size'][2]) for c in covers]


def clear_ray(a, b, obstacles):
    """Closed 3D segment versus exact axis-aligned wall boxes."""
    for box in obstacles:
        lo, hi = 0., 1.
        for axis in range(3):
            origin = a[axis]
            delta = b[axis]-origin
            mn, mx = box[2*axis:2*axis+2]
            if abs(delta) < 1e-12:
                if origin < mn or origin > mx:
                    lo, hi = 2., -1.
                    break
            else:
                x, y = sorted(((mn-origin)/delta, (mx-origin)/delta))
                lo, hi = max(lo, x), min(hi, y)
        if lo <= hi:
            return False
    return True


def smoke_chord(a, b, center, radius, height=2.5):
    """Length of a 3D sight segment inside a vertical cylindrical smoke volume."""
    dx, dy, dz = (b[k]-a[k] for k in range(3))
    px, py = a[0]-center[0], a[1]-center[1]
    aa = dx*dx+dy*dy
    if aa < 1e-18:
        if px*px+py*py >= radius*radius:
            return 0.
        lo, hi = 0., 1.
    else:
        bb = 2*(px*dx+py*dy)
        cc = px*px+py*py-radius*radius
        disc = bb*bb-4*aa*cc
        if disc <= 0:
            return 0.
        lo, hi = max(0., (-bb-math.sqrt(disc))/(2*aa)), min(1., (-bb+math.sqrt(disc))/(2*aa))
    if abs(dz) < 1e-12:
        if not 0 <= a[2] <= height:
            return 0.
    else:
        x, y = sorted((-a[2]/dz, (height-a[2])/dz))
        lo, hi = max(lo, x), min(hi, y)
    return max(0., hi-lo)*math.sqrt(dx*dx+dy*dy+dz*dz)


def smoke_blocks(a, b, smokes, t, penetration=0.):
    chord = sum(smoke_chord(a, b, s['center'], s['radius'], s['height'])
                for s in smokes if s['active_at'] <= t < s['expires_at'])
    return chord > penetration+1e-9


def valid_lock(visible, distance, bearing_error, speed, omega, requested, cfg):
    # Acquisition cone and precise tracking error are separate constraints.
    # Finite actor turn rates still bound tracking; angular speed alone is not failure.
    return (visible and requested and 1e-8 < distance <= cfg['tag_range_m']
            and abs(bearing_error) <= math.radians(cfg['aim_half_angle_degrees'])+1e-9
            and abs(bearing_error) <= math.radians(cfg['aim_tracking_error_degrees'])+1e-9
            and abs(speed) <= cfg['aim_max_speed_mps']+1e-9
            )


class TagClock:
    """Both tags and goal are settled from one snapshot, on a fixed dt lattice.

    A new valid sample starts an episode at zero. Only consecutive valid samples
    accrue dt. Simultaneous opposing wins within a tick produce a draw, never
    an actor-list-order victory. Timing uncertainty is explicitly <= dt.
    """
    def __init__(self, seconds=1.):
        self.seconds = seconds
        self.age = [0., 0.]
        self.previous = [False, False]

    def step(self, dt, valid, goal=False, timeout=False):
        for i in range(2):
            self.age[i] = self.age[i]+dt if valid[i] and self.previous[i] else 0.
        hits = [valid[i] and self.age[i]+1e-9 >= self.seconds for i in range(2)]
        self.previous = list(valid)
        attacker_win = goal or hits[0]
        defender_win = hits[1]
        if attacker_win and defender_win:
            return 'draw', 'simultaneous', hits
        if attacker_win:
            return 'attacker', 'goal' if goal else 'tag', hits
        if defender_win:
            return 'defender', 'tag', hits
        if timeout:
            return 'defender', 'timeout', hits
        return None, None, hits


def memory_update(memory, detection, t, ttl):
    if detection is not None:
        return dict(position=list(detection['position']), yaw=detection['yaw'], last_seen=t)
    if memory is not None and t-memory['last_seen'] <= ttl:
        return copy.deepcopy(memory)
    return None


def sampled_min_separation(a0, a1, b0, b1):
    r = np.asarray(a0)-b0
    d = (np.asarray(a1)-a0)-(np.asarray(b1)-b0)
    dd = float(d@d)
    u = float(np.clip(-float(r@d)/dd, 0, 1)) if dd > 1e-16 else 0.
    return float(np.linalg.norm(r+u*d))


def make_config(base, goal):
    cfg = copy.deepcopy(base)
    cfg['collapse']['time'] = 1e9
    cfg['arena']['task_points'] = [dict(name='G', center=goal, radius=1.)]
    return cfg


DEFAULT_RULES = dict(
    version='fixed-goal-probe-v0.1', duration=60., dt=.02, goal_radius_m=1.,
    tag_seconds=1., tag_range_m=8., aim_half_angle_degrees=30.,
    aim_max_speed_mps=.15, aim_tracking_error_degrees=5., reaction_seconds=.2,
    human_half_fov_degrees=90., human_sensor_range_m=None,
    robot_half_fov_degrees=180., robot_sensor_range_m=None,
    memory_seconds=[4., 12.], min_actor_separation_m=.7,
    smoke_count=1, smoke_throw_range_m=6., smoke_radius_m=2.,
    smoke_duration_seconds=5., smoke_cast_seconds=.35, smoke_fuse_seconds=.4,
    smoke_height_m=2.5, smoke_penetration_m=0., scan_radps=2.4,
    collapse_enabled=False, win_condition='attacker_goal_or_tag; defender_tag_or_timeout',
    timing='50Hz simultaneous snapshot; opposing success in same tick is draw',
    sensor='3D representative point LOS; same smoke/wall test for both actors and tag',
    throw='stationary cast, bounded range, clear segment at z=1m; no ballistic simulation',
)


class PublicController:
    """Small deliberately transparent baseline; receives no simulator objects."""
    def __init__(self, kind, actor, route, home, patrol, rules):
        self.kind, self.actor, self.rules = kind, actor, rules
        self.route = [list(q) for q in route]
        self.home = list(home)
        self.patrol = [list(p) for p in patrol]
        self.patrol_index = 0
        self.first_visible = None

    def decide(self, packet):
        t, own, visible = packet['time'], packet['self'], packet['visible']
        if visible is None:
            self.first_visible = None
        elif self.first_visible is None:
            self.first_visible = t
        ready = visible is not None and t-self.first_visible >= self.rules['reaction_seconds']-1e-8
        aim_policy = self.actor == 1 or self.kind in ('contact_tag', 'wide_tag', 'smoke_tag')
        if aim_policy and ready and math.dist(own['position'], visible['position']) <= self.rules['tag_range_m']:
            return dict(mode='AIM', look=visible['position'])
        if self.actor == 0:
            if self.kind in ('smoke_rush', 'smoke_tag') and packet['smoke_available']:
                if math.dist(own['position'], packet['goal']) <= self.rules['smoke_throw_range_m']:
                    return dict(mode='THROW', target=packet['goal'])
            return dict(mode='ADVANCE', route=self.route)
        look = None
        memory = packet['memory']
        if memory and t-memory['last_seen'] < 1.:
            look = memory['position']
        if self.kind == 'patrol':
            if math.dist(own['position'], self.patrol[self.patrol_index]) < .15:
                self.patrol_index = (self.patrol_index+1) % len(self.patrol)
            return dict(mode='PATROL', target=self.patrol[self.patrol_index], look=look)
        return dict(mode='WATCH', look=look)


class Probe:
    def __init__(self, base, rules, goal, route, defender_position, defender_policy, attacker_policy,
                 yaw=0., patrol=None, posture='upright'):
        self.cfg = make_config(base, goal)
        self.rules = copy.deepcopy(rules)
        self.goal = list(goal)
        self.planner = Planner(self.cfg, self.cfg['arena']['covers'])
        self.obstacles = boxes(self.cfg['arena']['covers'])
        self.actor = [ContinuousMover(base['start_a'], 0., base['postures'][posture]['speed'],
                                     base['postures'][posture]['acceleration'], base['yaw_rate'][0]),
                      HumanoidMover(defender_position, yaw, base['max_speed'][1],
                                    base['acceleration'][1], base['yaw_rate'][1])]
        self.heights = [base['postures'][posture]['height'], base['representative_heights'][1]]
        self.memory = [None, None]
        self.smokes = []
        self.inventory = rules['smoke_count']
        self.cast = None
        self.clock = TagClock(rules['tag_seconds'])
        self.events, self.trace, self.decisions = [], [], []
        self.collision_stops = [0, 0]
        self.minimum_separation = float('inf')
        self.minimum_clearance = float('inf')
        self.detected = [False, False]
        self.first_detection = [None, None]
        self.last_mode = [None, None]
        self.move_goal = [None, None]
        self.progress = 0
        self.waypoints = [np.asarray(q, float) for q in route]
        self.patrol = patrol or [goal, defender_position]
        self.controllers = [PublicController(attacker_policy, 0, route, base['start_a'], [], rules),
                            PublicController(defender_policy, 1, [], defender_position, self.patrol, rules)]
        self.metadata = dict(goal=goal, goal_radius=rules['goal_radius_m'], defender_start=list(defender_position),
                             attacker_start=base['start_a'], defender_policy=defender_policy,
                             attacker_policy=attacker_policy, yaw=yaw, posture=posture,
                             route=[list(q) for q in route], rules=copy.deepcopy(rules))

    def observation(self, i, t):
        own, other = self.actor[i], self.actor[1-i]
        distance = math.dist(own.xy, other.xy)
        bearing = math.atan2(other.xy[1]-own.xy[1], other.xy[0]-own.xy[0])
        prefix = 'robot' if i == 0 else 'human'
        sensor_range = self.rules[prefix+'_sensor_range_m']
        allowed = ((sensor_range is None or distance <= sensor_range)
                   and abs(wrap(bearing-own.yaw)) <= math.radians(self.rules[prefix+'_half_fov_degrees'])+1e-9)
        if allowed:
            a, b = (*own.xy, self.heights[i]), (*other.xy, self.heights[1-i])
            allowed = clear_ray(a, b, self.obstacles) and not smoke_blocks(a, b, self.smokes, t, self.rules['smoke_penetration_m'])
        detection = dict(position=other.xy.tolist(), yaw=other.yaw) if allowed else None
        self.memory[i] = memory_update(self.memory[i], detection, t, self.rules['memory_seconds'][i])
        if allowed and self.first_detection[i] is None:
            self.first_detection[i] = t
        if allowed != self.detected[i]:
            self.events.append(dict(time=t, type='detected' if allowed else 'lost', actor=i))
            self.detected[i] = allowed
        return dict(time=t, self=dict(position=own.xy.tolist(), yaw=own.yaw, speed=own.v),
                    visible=detection, memory=copy.deepcopy(self.memory[i]), goal=self.goal,
                    smoke_available=i == 0 and self.inventory > 0 and self.cast is None)

    def route_to(self, i, target):
        mover = self.actor[i]
        key = tuple(target)
        if self.move_goal[i] == key and mover.route:
            return
        route = self.planner.route(mover.xy, target)
        if route is not None:
            if i == 0:
                route = rounded_route(mover.xy, route, self.planner)
            assert all(self.planner.segment(a, b) for a, b in zip([mover.xy]+route, route))
            mover.navigate(route)
            self.move_goal[i] = key

    def stop(self, i):
        if self.actor[i].route:
            self.actor[i].navigate([])
        self.move_goal[i] = None

    def throw_valid(self, target):
        return (self.inventory > 0 and self.planner.free(target)
                and math.dist(self.actor[0].xy, target) <= self.rules['smoke_throw_range_m']+1e-9
                and clear_ray((*self.actor[0].xy, 1.), (*target, 1.), self.obstacles))

    def execute(self, i, command, t):
        mover = self.actor[i]
        mode = command['mode']
        if i == 0 and self.cast is not None:
            self.stop(0)
            if abs(mover.v) <= self.rules['aim_max_speed_mps']:
                if self.cast['settled_at'] is None:
                    self.cast['settled_at'] = t
                if t-self.cast['settled_at'] >= self.rules['smoke_cast_seconds']-1e-9:
                    s = dict(center=self.cast['target'], radius=self.rules['smoke_radius_m'],
                             height=self.rules['smoke_height_m'], active_at=t+self.rules['smoke_fuse_seconds'])
                    s['expires_at'] = s['active_at']+self.rules['smoke_duration_seconds']
                    self.smokes.append(s)
                    self.events.append(dict(time=t, type='smoke_released', **s))
                    self.cast = None
            return None, False, 'THROW'
        if mode == 'THROW':
            if self.throw_valid(command['target']):
                self.inventory -= 1
                self.cast = dict(target=list(command['target']), settled_at=None)
                self.events.append(dict(time=t, type='smoke_cast_started', center=command['target']))
                self.stop(i)
                return None, False, 'THROW'
            mode = 'ADVANCE'  # blocked throw retains the inventory and keeps moving
        if mode == 'AIM':
            self.stop(i)
            q = command['look']
            return math.atan2(q[1]-mover.xy[1], q[0]-mover.xy[0]), True, 'AIM'
        if mode == 'ADVANCE':
            while self.progress < len(self.waypoints) and math.dist(mover.xy, self.waypoints[self.progress]) < .15:
                self.progress += 1
            self.route_to(i, self.waypoints[min(self.progress, len(self.waypoints)-1)])
            return None, False, 'ADVANCE'
        if mode == 'PATROL':
            self.route_to(i, command['target'])
        else:
            self.stop(i)
        q = command.get('look')
        look = math.atan2(q[1]-mover.xy[1], q[0]-mover.xy[0]) if q is not None else mover.yaw+self.rules['scan_radps']*self.rules['dt']
        return look, False, mode

    def run(self, identifier):
        dt = self.rules['dt']
        requested = [False, False]
        modes = ['IDLE', 'IDLE']
        winner = reason = None
        for tick in range(round(self.rules['duration']/dt)+1):
            t = round(tick*dt, 8)
            packets = [self.observation(i, t) for i in range(2)]
            valid = []
            for i, mover in enumerate(self.actor):
                other = self.actor[1-i]
                bearing = math.atan2(other.xy[1]-mover.xy[1], other.xy[0]-mover.xy[0])
                valid.append(valid_lock(packets[i]['visible'] is not None, math.dist(mover.xy, other.xy),
                                        wrap(bearing-mover.yaw), mover.v, mover.omega, requested[i], self.rules))
            inside = math.dist(self.actor[0].xy, self.goal) <= self.rules['goal_radius_m']+1e-9
            permitted = [valid[0] and t+1e-9 >= getattr(self.clock,'cooldown_until',[0,0])[0],
                         valid[1] and t+1e-9 >= getattr(self.clock,'disabled_until',0.)]
            winner, reason, hits = self.clock.step(0. if tick == 0 else dt, valid, inside, t >= self.rules['duration'])
            if any(hits):
                self.events.append(dict(time=t, type='tag', actors=[i for i, h in enumerate(hits) if h]))
            self.trace.append(dict(t=t, xy=[m.xy.tolist() for m in self.actor], yaw=[m.yaw for m in self.actor],
                                   speed=[m.v for m in self.actor], visible=[p['visible'] is not None for p in packets],
                                   aim=self.clock.age.copy(), modes=modes.copy(), smoke=copy.deepcopy(self.smokes),
                                   omega=[m.omega for m in self.actor],requested_fire=requested.copy(),
                                   aim_error_degrees=[math.degrees(wrap(math.atan2(self.actor[1-i].xy[1]-m.xy[1],self.actor[1-i].xy[0]-m.xy[0])-m.yaw)) for i,m in enumerate(self.actor)],
                                   lock_geometry_valid=valid.copy(),lock_valid=permitted))
            if winner is not None:
                self.events.append(dict(time=t, type='finish', winner=winner, reason=reason))
                break
            commands = [self.controllers[i].decide(packets[i]) for i in range(2)]
            looks = []
            for i, command in enumerate(commands):
                look, requested[i], mode = self.execute(i, command, t)
                looks.append(look)
                modes[i] = mode
                if mode != self.last_mode[i]:
                    self.events.append(dict(time=t, type='mode', actor=i, mode=mode))
                    self.last_mode[i] = mode
                if tick % 5 == 0:
                    self.decisions.append(dict(time=t, actor=i, public=packets[i], command=command, executed=mode))
            before = [copy.deepcopy(m.__dict__) for m in self.actor]
            for i, mover in enumerate(self.actor):
                mover.step(dt, looks[i])
                if not self.planner.segment(before[i]['xy'], mover.xy):
                    mover.__dict__ = before[i]
                    mover.v = 0.
                    self.collision_stops[i] += 1
            separation = sampled_min_separation(before[0]['xy'], self.actor[0].xy, before[1]['xy'], self.actor[1].xy)
            if separation < self.rules['min_actor_separation_m']-1e-9:
                for i, mover in enumerate(self.actor):
                    new_yaw, new_omega = mover.yaw, mover.omega
                    mover.__dict__ = before[i]
                    mover.yaw, mover.omega, mover.v = new_yaw, new_omega, 0.
                    self.collision_stops[i] += 1
                separation = math.dist(self.actor[0].xy, self.actor[1].xy)
            self.minimum_separation = min(self.minimum_separation, separation)
            for mover in self.actor:
                self.minimum_clearance = min(self.minimum_clearance, min(clearance(mover.xy, c) for c in self.planner.obstacles))
        summary = dict(id=identifier, winner=winner, reason=reason, seconds=t,
                       first_detection=self.first_detection, smoke_used=self.rules['smoke_count']-self.inventory,
                       minimum_separation_m=self.minimum_separation, minimum_wall_clearance_m=self.minimum_clearance,
                       collision_stop_ticks=self.collision_stops, distance_m=[m.distance for m in self.actor],
                       **{k: self.metadata[k] for k in ('attacker_policy', 'defender_policy', 'yaw', 'posture')})
        assert winner is not None
        assert self.minimum_separation >= self.rules['min_actor_separation_m']-1e-8
        assert self.minimum_clearance >= self.cfg['planner_clearance']-1e-8
        return dict(summary=summary, metadata=self.metadata, events=self.events, states=self.trace, decisions=self.decisions)


def save_match(folder, result):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identifier = result['summary']['id']
    with gzip.open(folder/(identifier+'.json.gz'), 'wt', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
    compact = {k: result[k] for k in ('summary', 'metadata', 'events')}
    compact['states'] = result['states'][::5]
    if compact['states'][-1]['t'] != result['states'][-1]['t']:
        compact['states'].append(result['states'][-1])
    dump(folder/(identifier+'.json'), compact)
    return compact
