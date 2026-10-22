"""Pure, deterministic presentation timeline; no learned policy or dynamics."""
import json
import math
from pathlib import Path
import numpy as np


def pose(path, t):
    path = np.asarray(path, dtype=float)
    i = min(max(np.searchsorted(path[:, 0], t, side="right") - 1, 0), len(path) - 2)
    a, b = path[i:i+2]
    u = np.clip((t-a[0]) / (b[0]-a[0]), 0, 1)
    xy = a[1:] + u * (b[1:]-a[1:])
    velocity = (b[1:]-a[1:]) / (b[0]-a[0])
    return xy, velocity


def segment_box(a, b, lo, hi):
    """Finite 3-D segment versus closed axis-aligned box (slab method)."""
    lower, upper = 0., 1.
    for k in range(3):
        d = b[k]-a[k]
        if abs(d) < 1e-10:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
        else:
            q, r = sorted(((lo[k]-a[k])/d, (hi[k]-a[k])/d))
            lower, upper = max(lower, q), min(upper, r)
            if lower > upper:
                return False
    return True


def visible(a, b, covers):
    for cover in covers:
        x, y = cover["center"]
        w, d, h = cover["size"]
        if segment_box(a, b, (x-w/2, y-d/2, 0), (x+w/2, y+d/2, h)):
            return False
    return True


def build(config, arena, dt=.02):
    rules = config["rules"]
    hp, scores, dwell, lock = [rules["hp"]]*2, [0., 0.], [0., 0.], [0., 0.]
    cooldown_until = -1.
    states, events = [], [{"time": 0., "type": "start"}]
    yaws = [0., math.pi]
    clearance, separation, max_speed = float("inf"), float("inf"), 0.
    for tick in range(round(config["match_seconds"]/dt)+1):
        t = round(tick*dt, 6)
        point_index = int(t >= rules["point_switch_seconds"])
        if tick and abs(t-rules["point_switch_seconds"]) < dt/2:
            events.append({"time": t, "type": "point_switch", "point": "P2"})
            dwell = [0., 0.]
        positions, speeds = [], []
        for i, name in enumerate(("A", "B")):
            xy, velocity = pose(config["paths"][name], t)
            speed = float(np.linalg.norm(velocity))
            if speed > .001:
                yaws[i] = math.atan2(velocity[1], velocity[0])
            positions.append(xy)
            speeds.append(speed)
            max_speed = max(max_speed, speed)
            for cover in arena["covers"]:
                delta = np.maximum(np.abs(xy-cover["center"])-np.asarray(cover["size"][:2])/2, 0)
                clearance = min(clearance, float(np.linalg.norm(delta)))
            assert abs(xy[0]) <= 5.65 and abs(xy[1]) <= 3.65, "route outside arena"
        separation = min(separation, float(np.linalg.norm(positions[0]-positions[1])))
        representatives = [np.r_[positions[0], .36], np.r_[positions[1], 1.05]]
        los = visible(*representatives, arena["covers"])
        requests = [any(start <= t < end and who == name for start, end, who in config["mark_windows"]) for name in ("A", "B")]
        for i in range(2):
            if requests[i]:
                delta = positions[1-i]-positions[i]
                yaws[i] = math.atan2(delta[1], delta[0])
        # Choreographed requests are intentional: this is NOT the draft's automatic referee.
        hits = []
        for i in range(2):
            delta = representatives[1-i]-representatives[i]
            in_range = np.linalg.norm(delta) <= rules["max_range"]
            elevation_ok = abs(math.atan2(delta[2], np.linalg.norm(delta[:2]))) <= math.pi/4
            if requests[i] and los and in_range and elevation_ok and t >= cooldown_until:
                lock[i] += dt
                if lock[i] + 1e-9 >= rules["lock_seconds"]:
                    hits.append(i)
            else:
                lock[i] = 0.
        if hits:
            for i in hits:
                hp[1-i] -= 1
                events.append({"time": t, "type": "mark", "attacker": "AB"[i], "target": "AB"[1-i], "hp": hp.copy(), "los": los})
            cooldown_until = t + rules["pause_seconds"]
            lock = [0., 0.]
            dwell = [0., 0.]
        target = arena["task_points"][point_index]
        inside = [np.linalg.norm(p-target["center"]) <= target["radius"] for p in positions]
        scoring = [False, False]
        for i in range(2):
            if inside[i] and not inside[1-i] and t >= cooldown_until and tick:
                dwell[i] += dt
                if dwell[i] > rules["dwell_seconds"] + 1e-9:
                    scores[i] += dt
                    scoring[i] = True
            else:
                dwell[i] = 0.
        states.append({"t": t, "xy": [p.tolist() for p in positions], "yaw": yaws.copy(), "speed": speeds,
                       "hp": hp.copy(), "score": [round(s, 2) for s in scores], "point": point_index,
                       "pause": max(0., cooldown_until-t), "scoring": scoring, "contested": all(inside),
                       "los": los, "lock": lock.copy()})
        assert min(hp) > 0, "This choreography must reach the P2 and timeout chapters"
    assert clearance >= .30, f"route too close to cover: {clearance}"
    assert separation >= .65, f"actors overlap: {separation}"
    assert max_speed <= .55, f"route too fast: {max_speed}"
    assert sum(e["type"] == "mark" for e in events) == 3, "Expected three valid scripted marks"
    winner = "A" if (scores[0], hp[0]) > (scores[1], hp[1]) else "B" if (scores[1], hp[1]) > (scores[0], hp[0]) else "draw"
    events.append({"time": config["match_seconds"], "type": "finish", "winner": winner, "scores": states[-1]["score"]})
    return states, events, {"min_center_to_cover_m": clearance, "min_actor_center_separation_m": separation,
                            "max_route_speed_m_s": max_speed, "winner": winner,
                            "scope": "Point/disc route checks only; not full-body dynamics or locomotion validation"}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    states, events, validation = build(json.loads((root/"configs/demo_v1.json").read_text()), json.loads((root/"configs/arena_v0.json").read_text()))
    print(json.dumps({"events": events, "validation": validation}, indent=2))
