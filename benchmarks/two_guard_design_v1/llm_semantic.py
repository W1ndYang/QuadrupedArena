"""Versioned public map and strict LLM motion-hypothesis protocol.

Map vertices are spatial landmarks. Their edges are traversable geometry,
not a list of patrol routes; an LLM may compose any connected edge sequence.
"""
import hashlib
import json
import math

from two_node_geometry import add_two_node_geometry, make_config, Planner

SCHEMA_VERSION = 'guard-motion-hypotheses-v1'
RULES_VERSION = 'varied-guard-public-rules-v1'


def public_rules():
    base, rules = make_config()
    return dict(version=RULES_VERSION,
                general_behavior=(
                    'A guard follows a cyclic sequence of four private waypoints. '
                    'The four waypoints are sampled in distinct south-to-north '
                    'bands inside its region, then randomly ordered. It starts '
                    'on a random segment. At a waypoint it may change direction; '
                    'at one extreme endpoint it may dwell. When it continuously '
                    'sees the robot for the reaction delay and is within tag '
                    'range, it stops and aims at the robot. When not aiming it '
                    'patrols and turns its torso toward recent memory or scans.'),
                public_kinematics=dict(speed_range_mps=[.8, 1.5],
                    acceleration_mps2=base['acceleration'][1],
                    yaw_rate_radps=base['yaw_rate'][1],
                    reaction_seconds=rules['reaction_seconds'],
                    maximum_endpoint_dwell_seconds=4.,
                    tag_range_m=4.,tag_seconds=.6,
                    dt_seconds=rules['dt']),
                hidden_per_episode=(
                    'Exact four waypoint coordinates and order, initial segment '
                    'and fraction, exact speed and endpoint dwell are private. '
                    'Do not infer them from an episode seed or simulator state.'),
                observational_inference=(
                    'Position and movement are observed only at legal 10 Hz '
                    'visual samples; facing and patrol phase are not directly '
                    'available in the current structured sensor adapter. '
                    'Infer turns, next target and dwell as uncertain events.'))


def build_map():
    base, _ = make_config()
    add_two_node_geometry(base)
    planner = Planner(base, base['arena']['covers'])
    nodes = {}
    regions = {}
    for region, xs in (('first', (-3.9, -3., -2.1)),
                       ('second', (7.1, 8., 8.9))):
        ys = (-3.2, -1., 1., 3.2)
        members = []
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                node_id = f'{region}_x{ix}_y{iy}'
                if not planner.free((x, y)):
                    raise RuntimeError(f'Map landmark blocked: {node_id}')
                nodes[node_id] = dict(id=node_id, xy=[x, y],
                                      kind='guard_region_landmark', region=region)
                members.append(node_id)
        regions[region] = dict(id=region, kind='open_guard_patrol_envelope',
            polygon=[[xs[0], -3.35], [xs[-1], -3.35],
                     [xs[-1], 3.35], [xs[0], 3.35]],
            node_ids=members,
            entrances=[f'{region}_x0_y0', f'{region}_x0_y3'],
            exits=[f'{region}_x2_y0', f'{region}_x2_y3'])
    for name, xy in (('first_cover', (-8., 0.)),
                     ('first_south_gate', (-5., -1.6)),
                     ('first_north_gate', (-5., 1.6)),
                     ('second_cover', (3., 0.)),
                     ('second_south_gate', (6., -1.6)),
                     ('second_north_gate', (6., 1.6)),
                     ('goal', (10., 0.))):
        if not planner.free(xy):
            raise RuntimeError(f'Map landmark blocked: {name}')
        nodes[name] = dict(id=name, xy=list(xy), kind='robot_landmark',
                           region=None)
    edges = {}
    for region in regions:
        members = regions[region]['node_ids']
        for i, a in enumerate(members):
            for b in members[i+1:]:
                ax,ay=map(int,a.split('_x')[1].split('_y'))
                bx,by=map(int,b.split('_x')[1].split('_y'))
                if max(abs(ax-bx),abs(ay-by))==1 and planner.segment(
                        nodes[a]['xy'], nodes[b]['xy']):
                    for start,end in ((a,b),(b,a)):
                        edge_id=f'{start}__{end}'
                        edges[edge_id]=dict(id=edge_id,start=start,end=end,
                            path=[nodes[start]['xy'],nodes[end]['xy']],
                            length_m=round(math.dist(nodes[start]['xy'],
                                                     nodes[end]['xy']),6),
                            traversable_by=['guard'],bidirectional=False)
    for a, b in (('first_cover', 'first_south_gate'),
                 ('first_cover', 'first_north_gate'),
                 ('first_south_gate', 'second_cover'),
                 ('first_north_gate', 'second_cover'),
                 ('second_cover', 'second_south_gate'),
                 ('second_cover', 'second_north_gate'),
                 ('second_south_gate', 'goal'),
                 ('second_north_gate', 'goal')):
        route = planner.route(nodes[a]['xy'], nodes[b]['xy'])
        if route is None:
            raise RuntimeError(f'Robot landmark route blocked: {a}->{b}')
        path = [nodes[a]['xy'], *[list(map(float, p)) for p in route]]
        edges[f'{a}__{b}'] = dict(id=f'{a}__{b}', start=a, end=b,
            path=path,
            length_m=round(sum(math.dist(x, y) for x, y in zip(path, path[1:])), 6),
            traversable_by=['robot'], bidirectional=True)
    payload = dict(schema='arena-topology-v1', nodes=nodes, edges=edges,
                   regions=regions,
                   obstacles=[dict(name=c['name'], center=c['center'],
                                   size=c['size']) for c in base['arena']['covers']],
                   review=dict(status='geometry_verified_manual_semantics_pending',
                       note='All edges use the official map planner. Region labels '
                            'and entry/exit semantics require human review on Ubuntu.'))
    canonical = json.dumps({k:v for k,v in payload.items() if k!='review'},
                           sort_keys=True, separators=(',', ':'))
    payload['map_id'] = 'two-node-' + hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return payload


def _finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def validate_motion(raw, semantic_map, track_id, region, horizon):
    if not isinstance(raw, dict) or raw.get('schema') != SCHEMA_VERSION:
        raise ValueError('Motion response schema mismatch')
    if raw.get('map_id') != semantic_map['map_id'] or raw.get('track_id') != track_id:
        raise ValueError('Motion response map/track mismatch')
    hypotheses = raw.get('hypotheses')
    if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= 6:
        raise ValueError('Expected 1..6 motion hypotheses')
    ids, total = set(), 0.
    for h in hypotheses:
        if not isinstance(h, dict) or not isinstance(h.get('id'), str):
            raise ValueError('Invalid hypothesis ID')
        if h['id'] in ids:
            raise ValueError('Duplicate hypothesis ID')
        ids.add(h['id'])
        probability = _finite_number(h.get('probability'), 'probability')
        if not 0. <= probability <= 1.:
            raise ValueError('Probability outside [0,1]')
        total += probability
        state = h.get('state')
        if state not in ('MOVE', 'TURN', 'DWELL', 'REVERSE', 'UNKNOWN'):
            raise ValueError('Invalid motion state')
        edges = h.get('edge_ids')
        if not isinstance(edges, list) or len(edges) > 12:
            raise ValueError('Invalid edge sequence')
        target = h.get('next_target_id')
        if target not in semantic_map['nodes'] and target != region:
            raise ValueError('Unknown target ID')
        if target in semantic_map['nodes'] and semantic_map['nodes'][target]['region'] != region:
            raise ValueError('Target in wrong region')
        events = h.get('events')
        if not isinstance(events, list):
            raise ValueError('Events must be a list')
        if not isinstance(h.get('reason'), str) or not isinstance(h.get('uncertainty'), str):
            raise ValueError('Reason and uncertainty required')
        if state == 'UNKNOWN':
            if edges or events:
                raise ValueError('UNKNOWN must not assert a path or events')
            continue
        if not edges:
            raise ValueError('Known hypothesis has no path')
        previous = None
        for edge_id in edges:
            edge = semantic_map['edges'].get(edge_id)
            if edge is None or 'guard' not in edge['traversable_by']:
                raise ValueError(f'Unknown or non-guard edge {edge_id}')
            if semantic_map['nodes'][edge['start']]['region'] != region:
                raise ValueError('Hypothesis crosses guard region')
            if previous is not None and previous != edge['start']:
                raise ValueError('Disconnected directed edge sequence')
            previous = edge['end']
        if not events:
            raise ValueError('Known hypothesis needs events')
        for event in events:
            if not isinstance(event,dict):
                raise ValueError('Motion event must be an object')
            if event.get('type') not in ('MOVE', 'TURN', 'DWELL', 'REVERSE'):
                raise ValueError('Unknown motion event')
            ref = event.get('at_id')
            if ref is not None and ref not in semantic_map['nodes'] and ref not in semantic_map['edges']:
                raise ValueError('Event references unknown map ID')
        move_edges=[event.get('at_id') for event in events
                    if event['type']=='MOVE']
        if move_edges!=edges:
            raise ValueError('MOVE events must match edge sequence in order')
        speed = h.get('speed_range_mps')
        dwell = h.get('dwell_range_seconds')
        if (not isinstance(speed, list) or len(speed) != 2 or
                not isinstance(dwell, list) or len(dwell) != 2):
            raise ValueError('Speed/dwell ranges required')
        lo, hi = map(lambda v: _finite_number(v, 'speed'), speed)
        d0, d1 = map(lambda v: _finite_number(v, 'dwell'), dwell)
        if not (.5 <= lo <= hi <= 1.8 and 0. <= d0 <= d1 <= 4.):
            raise ValueError('Speed/dwell outside public bounds')
    if abs(total-1.) > .025:
        raise ValueError('Hypothesis probabilities must sum to 1')
    if horizon <= 0.:
        raise ValueError('Invalid horizon')
    return raw
