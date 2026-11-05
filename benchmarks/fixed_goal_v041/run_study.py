"""Reproducible map screening and bounded deterministic virtual-tag probes."""
import argparse
import collections
import copy
import datetime
import json
import math
import shutil
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'source_snapshot/scripts' if (R/'source_snapshot/scripts/fixed_goal_probe.py').exists() else P/'scripts'))
import numpy as np
from fixed_goal_probe import DEFAULT_RULES, Probe, boxes, clear_ray, dump, make_config, read, save_match, sha
from trial_motion import Planner, clearance, path_length

CANDIDATES = [dict(id='W', label='West court', center=[-7., 1.4], via=[-1.2, 2.2]),
              dict(id='E', label='East court', center=[10.3, -1.], via=[5.9, 3.1]),
              dict(id='S', label='South court', center=[3.2, -4.2], via=[5.9, 3.1])]
ATTACKERS = ['direct_rush', 'smoke_rush', 'contact_tag', 'wide_tag', 'smoke_tag']
DEFENDERS = ['on_goal', 'cover_guard', 'patrol']
HEADINGS = [0., math.pi/2, math.pi, -math.pi/2]


def freeze():
    if (R/'protocol.json').exists():
        return read(R/'source_snapshot/configs/trial_a391_feedback_fix.json')
    baseline = {str(p.relative_to(P)): sha(p) for folder in ('scripts', 'configs') for p in sorted((P/folder).glob('*'))
                if p.suffix in ('.py', '.json') and p.name not in ('fixed_goal_probe.py', 'test_fixed_goal_probe.py')}
    dump(R/'baseline_manifest.json', baseline)
    for name in ('fixed_goal_probe.py', 'test_fixed_goal_probe.py', 'continuous_motion.py', 'humanoid_motion.py',
                 'tactical_logic.py', 'trial_motion.py', 'trial_rules.py'):
        dest = R/'source_snapshot/scripts'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(P/'scripts'/name, dest)
    dest = R/'source_snapshot/configs/trial_a391_feedback_fix.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(P/'configs/trial_a391_feedback_fix.json', dest)
    dump(R/'protocol.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          stage='exploratory design validation, not formal LLM comparison', rules=DEFAULT_RULES,
          candidates=CANDIDATES, attacker_policies=ATTACKERS, defender_policies=DEFENDERS,
          headings_rad=HEADINGS, base_matrix_matches=180,
          fixed='A3.9.1 initial wall geometry; no collapse; inherited kinematic movers and A*; no API calls',
          interventions='fixed 1m goal; deterministic 1s tag; both can be eliminated; shared stability requirements; smoke',
          guard_placement='goal and cover sentries start prepositioned for stress testing; patrol starts at old human spawn',
          screening='map-only sampled guard optimization, no hidden opponent trajectory supplied to policies',
          limitations=['Deterministic policy/heading combinations are not independent random trials.',
                       'This is a separate CPU prototype, not yet the production referee or native Isaac simulation.',
                       'Smoke is a nonphysical cylinder; reference movement does not establish dynamics feasibility.',
                       'Some movement routes are deliberately fixed baselines; no claimed optimal response.',
                       'Goal and tag resolution quantized to 20ms, opposing same-tick success is draw.']))
    return read(dest)


def concatenate(planner, start, destinations):
    route = []
    for goal in destinations:
        part = planner.route(start, goal)
        if part is None:
            return None
        if not all(planner.segment(a, b) for a, b in zip([start]+part, part)):
            raise RuntimeError('Invalid navigation reference')
        route.extend(part)
        start = goal
    return [list(p) for p in route]


def sample_route(start, route, goal, radius, spacing=.25):
    result = []
    distance = 0.
    for end in route:
        length = math.dist(start, end)
        for j in range(max(1, math.ceil(length/spacing))):
            u = j/max(1, math.ceil(length/spacing))
            q = [start[k]*(1-u)+end[k]*u for k in range(2)]
            if math.dist(q, goal) <= radius:
                return result
            result.append(dict(xy=q, distance=distance+length*u))
        distance += length
        start = end
    return result


def screen(base):
    results = []
    for target in CANDIDATES:
        goal = target['center']
        cfg = make_config(base, goal)
        planner = Planner(cfg, cfg['arena']['covers'])
        obstacle_boxes = boxes(cfg['arena']['covers'])
        gap = min(clearance(goal, c) for c in planner.obstacles)
        boundary = min(np.asarray(cfg['arena']['floor_size'][:2])/2-np.abs(goal))
        assert gap >= DEFAULT_RULES['goal_radius_m']+cfg['planner_clearance']-1e-9, (target['id'], gap)
        assert boundary >= DEFAULT_RULES['goal_radius_m']+.45
        routes = dict(direct=concatenate(planner, base['start_a'], [goal]),
                      wide=concatenate(planner, base['start_a'], [target['via'], goal]))
        assert all(routes.values())
        samples = {name: sample_route(base['start_a'], route, goal, DEFAULT_RULES['goal_radius_m']) for name, route in routes.items()}
        rows = []
        for n in sorted(planner.free_nodes):
            if n[0] % 4 or n[1] % 4:
                continue
            q = [float(n[0]*planner.grid), float(n[1]*planner.grid)]
            dg = math.dist(q, goal)
            near_wall = min(clearance(q, c) for c in planner.obstacles)
            if not 2.0 <= dg <= 7.8 or near_wall > 1.3:
                continue
            if not clear_ray((*q, 1.05), (*goal, .4), obstacle_boxes):
                continue
            if math.dist(q, base['start_a']) <= 8 and clear_ray((*q, 1.05), (*base['start_a'], .4), obstacle_boxes):
                continue
            windows = []
            for name, samples_ in samples.items():
                streak = longest = 0.
                count = 0
                last_d = 0.
                for item in samples_:
                    visible = math.dist(q, item['xy']) <= 8 and clear_ray((*q, 1.05), (*item['xy'], .4), obstacle_boxes)
                    step = item['distance']-last_d
                    streak = streak+step if visible else 0.
                    longest = max(longest, streak)
                    count += int(visible)
                    last_d = item['distance']
                windows.append(dict(route=name, longest_visible_m=longest,
                                    cruise_seconds=longest/base['postures']['upright']['speed'],
                                    visible_fraction=count/max(1, len(samples_))))
            row = dict(position=q, target_distance=dg, wall_distance=near_wall, windows=windows)
            row['screen_score'] = [sum(w['cruise_seconds'] >= 1. for w in windows),
                                   min(w['cruise_seconds'] for w in windows),
                                   sum(w['visible_fraction'] for w in windows), -near_wall]
            rows.append(row)
        rows.sort(key=lambda r: r['screen_score'], reverse=True)
        assert rows
        guard = rows[0]['position']
        # Patrol connects public-map viewpoints; no hidden agent state used.
        alternatives = [r for r in rows[1:] if math.dist(r['position'], guard) >= 3]
        second = alternatives[0]['position'] if alternatives else goal
        patrol = [guard, second]
        assert concatenate(planner, base['start_b'], patrol) is not None
        record = dict(**target, goal_radius=DEFAULT_RULES['goal_radius_m'], whole_circle_wall_clearance=gap-DEFAULT_RULES['goal_radius_m'],
                      free_nodes=len(planner.free_nodes), routes=routes,
                      route_lengths={k: path_length(base['start_a'], [np.asarray(q) for q in v]) for k, v in routes.items()},
                      sampled_guards=len(rows), selected_guard=rows[0], patrol=patrol, guard_screen=rows,
                      screen_limit='Unoriented static rays; cruise-distance proxy ignores acceleration, stopping, and detection reaction; not a win probability.')
        dump(R/'layouts'/(target['id']+'.json'), record)
        results.append(record)
        print('SCREEN', target['id'], 'guard', guard, 'routes', record['route_lengths'], flush=True)
    dump(R/'geometry_results.json', results)
    return results


def one(base, layout, attacker, defender, heading_index, rules=None, prefix='base', posture='upright'):
    rules = copy.deepcopy(rules or DEFAULT_RULES)
    goal = layout['center']
    position = goal if defender == 'on_goal' else layout['selected_guard']['position'] if defender == 'cover_guard' else base['start_b']
    destinations = [layout['via'], goal] if attacker == 'wide_tag' else [goal]
    identifier = f'{prefix}_{layout["id"]}_{attacker}_{defender}_{heading_index}_{posture}'
    probe = Probe(base, rules, goal, destinations, position, defender, attacker,
                  yaw=HEADINGS[heading_index], patrol=layout['patrol'], posture=posture)
    result = probe.run(identifier)
    result['summary']['layout'] = layout['id']
    result['summary']['condition'] = prefix
    return save_match(R/'matches', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['screen', 'pilot', 'base'], default='base')
    args = parser.parse_args()
    base = freeze()
    layouts = read(R/'geometry_results.json') if (R/'geometry_results.json').exists() else screen(base)
    if args.phase == 'screen':
        return
    results = []
    start = time.monotonic()
    for layout in layouts:
        for attacker in ATTACKERS:
            for defender in DEFENDERS:
                for h in range(len(HEADINGS)):
                    if args.phase == 'pilot' and (layout['id'] != 'S' or attacker not in ('direct_rush', 'smoke_rush', 'contact_tag') or defender != 'cover_guard' or h):
                        continue
                    identifier = f'base_{layout["id"]}_{attacker}_{defender}_{h}_upright'
                    file = R/'matches'/(identifier+'.json')
                    result = read(file) if file.exists() else one(base, layout, attacker, defender, h)
                    results.append(result['summary'])
                    dump(R/'progress.json', dict(completed=len(results), phase=args.phase, latest=result['summary'], wall_seconds=time.monotonic()-start))
                    print('MATCH', len(results), identifier, result['summary']['winner'], result['summary']['reason'], result['summary']['seconds'], flush=True)
    dump(R/(args.phase+'_results.json'), results)
    print('DONE', args.phase, len(results), 'wall_seconds', round(time.monotonic()-start, 2), flush=True)


if __name__ == '__main__':
    main()
