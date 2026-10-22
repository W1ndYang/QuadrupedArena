"""CPU-side design geometry. Explicit XYZ solids; not the legacy 2D planner.

Ramps are convex solids, containers are separate panels (never solid boxes),
and bridge underpass volume is deliberately empty. No match policy is run here.
"""
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_design(name='dockyard_v1.json'):
    path=Path(name)
    if not path.is_absolute():path=PROJECT/'configs'/path
    data=json.loads(path.read_text())
    if 'base_config' not in data:return data
    base=load_design(path.parent/data.pop('base_config'))
    for key,value in data.items():
        if isinstance(value,dict) and isinstance(base.get(key),dict):base[key].update(value)
        else:base[key]=value
    return base


def solids(design, fallen=False):
    result = []
    def box(name, center, size, material, role='obstacle'):
        result.append(dict(name=name, center=list(center), size=list(size), material=material, role=role, shape='box'))
    for w in design['walls']:
        center = [*w['center'], w['size'][2] / 2]
        size = w['size']
        if fallen and w['name'] == design['collapse']['wall']:
            center, size = design['collapse']['debris']['center'], design['collapse']['debris']['size']
        box(w['name'], center, size, w.get('material','concrete' if size[2] > 1 else 'pale'), 'cover')
    for f in design.get('facility_solids',[]):
        box(f['name'],f['center'],f['size'],f['material'],'service_building')
    for c in design['containers']:
        x, y = c['center']; length, width, h = c['length'], c['width'], c['height']
        def local(name, along, across, z, la, wi, hi, material=None):
            center = (x+along, y+across, z) if c['axis'] == 'x' else (x+across, y+along, z)
            size = (la, wi, hi) if c['axis'] == 'x' else (wi, la, hi)
            box(c['name']+'_'+name, center, size, material or c['material'], 'container_panel')
        local('floor', 0, 0, .015, length, width, .03, 'steel')
        local('roof', 0, 0, h-.06, length, width, .12)
        for side in [-1, 1]:
            local('side'+str(side+1), 0, side*(width/2-.07), h/2, length, .14, h)
            if not c['open_ends']:
                local('end'+str(side+1), side*(length/2-.07), 0, h/2, .14, width, h)
            else:
                local('header'+str(side+1), side*(length/2-.07), 0, h-.10, .14, width, .20)
    b = design['bridge']; cx, cy = b['center']; half = b['deck_length']/2; w = b['width']; h = b['height']; r = b['ramp_run']
    box('BridgeDeck', [cx,cy,h-b['deck_thickness']/2], [2*half,w,b['deck_thickness']], 'steel', 'bridge_deck')
    for side in [-1,1]:
        for sy in [-1,1]:
            box(f'Pier{side+1}{sy+1}', [cx+side*(half-.28),cy+sy*(w/2+.18),(h-.18)/2], [.32,.28,h-.18], 'dark', 'pier')
        low_x, high_x = (cx-half-r,cx-half) if side<0 else (cx+half+r,cx+half)
        points = [[low_x,cy-w/2,0],[low_x,cy+w/2,0],[high_x,cy-w/2,0],[high_x,cy+w/2,0],
                  [low_x,cy-w/2,.025],[low_x,cy+w/2,.025],[high_x,cy-w/2,h],[high_x,cy+w/2,h]]
        result.append(dict(name='RampWest' if side<0 else 'RampEast',shape='ramp',points=points,
            material='steel',role='ramp',low_x=low_x,high_x=high_x,y_min=cy-w/2,y_max=cy+w/2,height=h,toe=.025))
    return result


def ramp_height(s, x):
    return s['toe']+(s['height']-s['toe'])*(x-s['low_x'])/(s['high_x']-s['low_x'])


def occupied(point, geometry, clearance=0):
    p = np.asarray(point, float)
    for s in geometry:
        if s['shape']=='box':
            if np.all(np.abs(p-np.array(s['center'])) <= np.array(s['size'])/2+clearance):return s['name']
        else:
            lo,hi=sorted([s['low_x'],s['high_x']])
            if lo-clearance<=p[0]<=hi+clearance and s['y_min']-clearance<=p[1]<=s['y_max']+clearance:
                if -clearance<=p[2]<=ramp_height(s,min(hi,max(lo,p[0])))+clearance:return s['name']
    return None


def ground_route(design, geometry, start, goal, spacing=.25, radius=.42):
    """Read-only ground connectivity check with conservative XY obstacle margins.

No elevated routes, opponent state, or decision policy. Low toe areas may be
excluded conservatively; this is not a full-body navigation proof.
"""
    half=np.array(design['playable_size_m'])/2
    obstacles=[]
    for s in geometry:
        if s['shape']=='ramp':
            lo,hi=sorted([s['low_x'],s['high_x']]);obstacles.append((np.array([(lo+hi)/2,(s['y_min']+s['y_max'])/2]),np.array([(hi-lo)/2,(s['y_max']-s['y_min'])/2])))
        elif s['center'][2]-s['size'][2]/2 < 1.45 and s['center'][2]+s['size'][2]/2>.05:
            obstacles.append((np.array(s['center'][:2]),np.array(s['size'][:2])/2))
    def free_point(p):
        return np.all(np.abs(p)<half-radius) and all(np.linalg.norm(np.maximum(np.abs(p-c)-h,0))>=radius for c,h in obstacles)
    @lru_cache(maxsize=None)
    def free(node):return free_point(np.array(node)*spacing)
    def nearest(p):return tuple(int(round(v/spacing)) for v in p)
    s,g=nearest(start),nearest(goal)
    if not free(s) or not free(g):return None
    queue=[(0,s)];cost={s:0};parent={};seen=set()
    while queue:
        _,n=heapq.heappop(queue)
        if n in seen:continue
        if n==g:
            nodes=[n]
            while nodes[-1]!=s:nodes.append(parent[nodes[-1]])
            path=[(np.array(v)*spacing).tolist() for v in reversed(nodes)]
            return dict(path=path,length_m=cost[g]*spacing)
        seen.add(n)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nxt=(n[0]+dx,n[1]+dy)
            if not free(nxt):continue
            if dx and dy and (not free((n[0]+dx,n[1])) or not free((n[0],n[1]+dy))):continue
            nc=cost[n]+math.hypot(dx,dy)
            if nc<cost.get(nxt,float('inf')):
                cost[nxt]=nc;parent[nxt]=n;heapq.heappush(queue,(nc+math.dist(nxt,g),nxt))
    return None


def validate_design(design):
    before,after=solids(design),solids(design,True)
    tests=[]
    def check(ok,label):
        if not ok:raise AssertionError(label)
        tests.append(label)
    check(np.prod(design['playable_size_m'])==8*np.prod(design['previous_playable_size_m']),'actual playable footprint is eight times the old map')
    for c in design['containers']:
        if not c['open_ends']:continue
        x,y=c['center']
        for t in np.linspace(-c['length']/2-.3,c['length']/2+.3,45):
            for z in [.4,1.4]:
                p=[x+t,y,z] if c['axis']=='x' else [x,y+t,z]
                check(occupied(p,before) is None,f'{c["name"]} through-path clear at {t:.3f} m / {z} m height')
    b=design['bridge']
    for y in np.linspace(-2.3,2.3,31):
        for z in [.4,1.4]:check(occupied([0,y,z],before) is None,'ground underpass clear for both representative heights')
    ramps=[s for s in before if s['shape']=='ramp']
    approaches=[]
    for a in design.get('bridge_approaches',[]):
        lo=np.array([a['x_range'][0],a['y_range'][0],.05]);hi=np.array([a['x_range'][1],a['y_range'][1],2.0])
        for s in before:
            if s['shape']!='box':continue
            mn=np.array(s['center'])-np.array(s['size'])/2;mx=np.array(s['center'])+np.array(s['size'])/2
            check(not np.all(np.minimum(mx,hi)-np.maximum(mn,lo)>1e-6),a['name']+' clear of '+s['name'])
        approaches.append(dict(name=a['name'],length_m=float(hi[0]-lo[0]),width_m=float(hi[1]-lo[1]),clear_height_m=2.0))
    check(all(abs(ramp_height(s,s['high_x'])-b['height'])<1e-8 for s in ramps),'both ramp tops join deck exactly')
    check(all(abs(ramp_height(s,s['low_x'])-.025)<1e-8 for s in ramps),'ramp toes are 25 mm above floor')
    slope=math.degrees(math.atan2(b['height']-.025,b['ramp_run']))
    check(slope<20,'ramp geometry slope below 20 degrees; traction not validated')
    paths={}
    for phase,geometry in [('before',before),('after',after)]:
        pairs=[('A_to_P1',design['spawns']['A'],design['targets'][0]['center']),
               ('P1_to_P2',design['targets'][0]['center'],design['targets'][1]['center']),
               ('B_to_P2',design['spawns']['B'],design['targets'][1]['center'])]
        for name,start,goal in pairs:
            route=ground_route(design,geometry,start,goal);check(route is not None,phase+': '+name+' has a ground route')
            paths[phase+'_'+name]=route
        for key,a in zip(('A','B'),design.get('bridge_approaches',[])):
            route=ground_route(design,geometry,design['spawns'][key],a['entry'])
            check(route is not None,phase+': '+key+' can reach '+a['name'])
            paths[phase+'_'+key+'_to_'+a['name']]=route
    if design.get('facility_solids'):
        original={w['name']:w['size'][2] for w in load_design('dockyard_v1.json')['walls']}
        for w in design['walls']:check(w['size'][2]==original[w['name']],w['name']+' original cover height retained')
    check(design['collapse']['time']==20 and design['collapse']['angle_degrees']==90,'fixed collapse timing and angle preserved')
    check(len(design['targets'])==2 and all(p['radius']==.55 for p in design['targets']),'two rotating point locations retain 0.55 m radius')
    return dict(status='passed',checks=len(tests),details=tests,ramp_slope_degrees=slope,
        bridge_clear_height_m=b['height']-b['deck_thickness'],paths=paths,bridge_approaches=approaches,
        scope='Geometry / conservative ground connectivity only. No LLM match, no contact-dynamics validation.')
