"""Robot intent skills, not Codex instruction skills.

This module receives ONLY public observations and known map/rules. No Match,
renderer, opponent mover, RNG, referee events, or privileged sensor references.
Visual localization is an explicit RGB foot-pixel annotation plus calibration;
it is NOT an automatic detector or continuous target tracker.
"""
import copy
import math
import numpy as np
from adaptive_motion import RiskPlanner
from tactical_logic import block,wrap
from trial_motion import Planner,path_length,clearance
from trial_rules import covers_at

SKILLS=('MOVE_TO','TAKE_COVER','AIM_AT_TRACK','CAPTURE_POINT','OBSERVE')


def ground_point(frame,pixel):
    if not isinstance(pixel,list) or len(pixel)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pixel):raise ValueError('feet must be two finite pixels')
    u,v=pixel
    if not 0<=u<768 or not 0<=v<432:raise ValueError('Pixel outside image')
    a=math.radians(frame['camera_world_yaw_degrees']);p=math.radians(-5)
    forward=np.array([math.cos(a)*math.cos(p),math.sin(a)*math.cos(p),math.sin(p)])
    right=np.array([math.sin(a),-math.cos(a),0.]);up=np.cross(right,forward)
    ray=forward+(u-384)/384*right-(v-216)/384*up;eye=np.asarray(frame['camera_eye'])
    if ray[2]>=-.015:raise ValueError('Foot ray cannot intersect nearby ground')
    point=eye-eye[2]/ray[2]*ray
    if abs(point[0])>6.2 or abs(point[1])>4.2:raise ValueError('Estimated foot outside arena; do not annotate occluded feet')
    return point[:2].tolist()


class VisualMemory:
    def __init__(self,initial):self.track=dict(xy=list(initial),seen_at=0.,source='known_initial',base_uncertainty=.2)
    def update(self,annotation,packet):
        if annotation is None:return
        if not isinstance(annotation,dict) or set(annotation)!={'time','feet'}:raise ValueError('sighting requires time and feet only')
        if type(annotation['time']) not in (int,float) or not math.isfinite(annotation['time']):raise ValueError('Invalid sighting time')
        frame=next((f for f in packet['frames'] if f['simulation_seconds']==annotation['time']),None)
        if frame is None:raise ValueError('Sighting must reference an input frame')
        if annotation['time']<self.track['seen_at']:raise ValueError('Do not overwrite newer visual memory')
        self.track=dict(xy=ground_point(frame,annotation['feet']),seen_at=frame['simulation_seconds'],source='rgb_foot_pixel',
            base_uncertainty=.6,frame=frame['file'],feet=list(annotation['feet']))
    def belief(self,t):
        age=max(0.,t-self.track['seen_at'])
        if age>4.:return None
        return dict(**self.track,age=age,uncertainty_radius=self.track['base_uncertainty']+.65*age,
            aim_usable=self.track['source']=='rgb_foot_pixel' and age<=2.)
    def public(self,t):
        value=self.belief(t)
        return value if value is not None else dict(status='expired',last_seen_at=self.track['seen_at'],current_position='unknown')


class BeliefPlanner(RiskPlanner):
    """Known-map Dijkstra, uncertain sighting LOS cost, unknown enemy yaw.

Reuses only graph search/route smoothing from RiskPlanner, not its truth-facing
exposure implementation. Five possible threat positions; no enemy policy.
"""
    def __init__(self,base,belief,hp,mode):
        self.base=base;self.cfg=base.cfg;self.grid=base.grid;self.covers=base.covers
        self.enemy_alive=belief is not None;self.enemy=np.asarray(belief['xy'] if belief else [0,0])
        self.weight=(8. if mode=='safe' else 1.5)*(1.6 if hp<=1 else 1.)
        radius=belief['uncertainty_radius'] if belief else 0.
        self.samples=[] if belief is None else [self.enemy]+[self.enemy+radius*np.array(d) for d in ((1,0),(-1,0),(0,1),(0,-1))]
        self.samples=[p for p in self.samples if abs(p[0])<=6 and abs(p[1])<=4]
        self.nodes={n for n in base.free_nodes if self.dynamic_free(np.asarray(n)*self.grid)}
        self.risk={n:self.point_risk(np.asarray(n)*self.grid) for n in self.nodes}
        self.cost={};self.parent={};self.start=None;self.source=None
    def point_risk(self,p):
        if not self.samples:return 0.
        return sum(float(np.linalg.norm(p-e)<=self.cfg['range_m'] and block((*e,1.05),(*p,.4),self.covers) is None) for e in self.samples)/len(self.samples)
    def nearest(self,p):
        n=tuple(np.rint(np.asarray(p)/self.grid).astype(int))
        if n in self.nodes and np.linalg.norm(np.asarray(n)*self.grid-p)<1e-6:return n
        return super().nearest(p)


class SemanticSkills:
    def __init__(self,cfg):
        self.cfg=copy.deepcopy(cfg)
        # Remove opponent behavioral configuration; planner needs only two safety constants.
        self.cfg['policy']={k:cfg['policy'][k] for k in ('dynamic_avoid_radius','reverse_speed_limit','safety_stop_distance')}
        self.memory=VisualMemory(cfg['start_b']);self.status=dict(state='IDLE');self.action=None;self.intent=None
        self.generation=None;self.hp=None;self.goal=None;self.sequence=0

    def validate(self,request,packet):
        if not isinstance(request,dict) or set(request) not in ({'observation_id','skill','params'},{'observation_id','skill','params','reason'}):raise ValueError('Expected observation_id, skill, params and optional reason')
        if type(request['observation_id']) is not int or request['observation_id']!=packet['observation_id']:raise ValueError('Wrong observation_id')
        if request['skill'] not in SKILLS:raise ValueError('Unknown intent skill')
        if 'reason' in request and (not isinstance(request['reason'],str) or not 1<=len(request['reason'])<=100):raise ValueError('Use a short reason (1-100 characters)')
        params=request['params'];allowed={'sighting','risk'}
        if request['skill'] in ('MOVE_TO','CAPTURE_POINT'):allowed.add('target')
        if request['skill']=='TAKE_COVER':allowed.add('cover')
        if request['skill']=='OBSERVE':allowed.add('direction')
        if not isinstance(params,dict) or set(params)-allowed:raise ValueError('Unknown skill parameters; no manual paths or headings')
        if params.get('risk','safe') not in ('safe','fast'):raise ValueError('Unknown risk setting')
        if params.get('target','active') not in ['active']+[p['name'] for p in self.cfg['arena']['task_points']]:raise ValueError('Use a known target name')
        if 'cover' in params and params['cover'] not in [c['name'] for c in self.cfg['arena']['covers']]:raise ValueError('Unknown cover')
        if params.get('direction','scan') not in ('scan','threat','north','south','east','west'):raise ValueError('Unknown observation direction')
        candidate=copy.deepcopy(self.memory);candidate.update(params.get('sighting'),packet)
        return candidate

    def public(self,t):return dict(skill_status=copy.deepcopy(self.status),visual_memory=self.memory.public(t))

    def eta(self,own,route,reverse):
        if path_length(own['xy'],route)<.09:return 0.
        xy=np.asarray(own['xy']);yaw=own['yaw'];elapsed=max(0,self.cfg['max_speed'][0]-own['speed'])/self.cfg['acceleration'][0]
        for i,p in enumerate(route):
            d=np.asarray(p)-xy;bearing=math.atan2(d[1],d[0]);back=reverse and abs(wrap(bearing-yaw))>math.pi/2
            heading=bearing+(math.pi if back else 0.)
            if i==0 and (back or abs(wrap(heading-yaw))>.25):elapsed+=2*own['speed']/self.cfg['acceleration'][0]
            elapsed+=abs(wrap(heading-yaw))/self.cfg['yaw_rate'][0]+float(np.linalg.norm(d))/(self.cfg['policy']['reverse_speed_limit'] if back else self.cfg['max_speed'][0])
            xy=np.asarray(p);yaw=heading
        return elapsed

    def exposure_seconds(self,own,route,planner,reverse):
        """Conservative continuous LOS exposure estimate, not a hit probability."""
        xy=np.asarray(own['xy']);yaw=own['yaw'];current=0.;maximum=0.
        for dest in route:
            d=np.asarray(dest)-xy;distance=float(np.linalg.norm(d))
            if distance<1e-8:continue
            bearing=math.atan2(d[1],d[0]);back=reverse and abs(wrap(bearing-yaw))>math.pi/2
            heading=bearing+(math.pi if back else 0.)
            if planner.point_risk(xy)>.19:current+=abs(wrap(heading-yaw))/self.cfg['yaw_rate'][0]+own['speed']/self.cfg['acceleration'][0]
            maximum=max(maximum,current)
            speed=self.cfg['policy']['reverse_speed_limit'] if back else self.cfg['max_speed'][0]
            count=max(1,math.ceil(distance/.1))
            for u in np.linspace(1/count,1,count):
                if planner.point_risk(xy+u*d)>.19:current+=distance/count/speed
                else:current=0.
                maximum=max(maximum,current)
            xy=np.asarray(dest);yaw=heading
        return maximum

    def commit(self,request,packet):
        self.memory=self.validate(request,packet);self.intent=request['skill'];self.params=copy.deepcopy(request['params'])
        self.generation=packet['public_state']['target_generation'];self.hp=packet['self_state']['hp']
        self.sequence+=1
        return self.plan(packet)

    def plan(self,packet):
        t=packet['simulation_seconds'];own=packet['self_state'];skill=self.intent;params=self.params
        if skill=='CAPTURE_POINT' and self.generation!=packet['public_state']['target_generation']:
            self.status=dict(skill=skill,state='TARGET_EXPIRED');self.action=dict(skill='STOP',facing='keep',attack=False)
            return self.action,[],self.camera(packet),copy.deepcopy(self.status)
        self.base=Planner(self.cfg,covers_at(self.cfg,t));belief=self.memory.belief(t)
        p=BeliefPlanner(self.base,belief,own['hp'],params.get('risk','safe'));start=np.asarray(own['xy'])
        facing=own['yaw'] if belief is None else math.atan2(belief['xy'][1]-start[1],belief['xy'][0]-start[0])
        action=dict(skill='HOLD',facing=facing,attack=bool(belief and belief['aim_usable']))
        route=[];goal=None;state='RUNNING';eta=0.;cover_name=None;exposure=0.;rejected=0
        if skill in ('MOVE_TO','CAPTURE_POINT','TAKE_COVER'):
            p.solve(start);candidates=[]
            if skill in ('MOVE_TO','CAPTURE_POINT'):
                name=params.get('target','active');idx=packet['public_state']['active_target'] if name=='active' else next(i for i,x in enumerate(self.cfg['arena']['task_points']) if x['name']==name)
                point=self.cfg['arena']['task_points'][idx];center=np.asarray(point['center'])
                if skill=='CAPTURE_POINT' and idx!=packet['public_state']['active_target']:
                    self.status=dict(skill=skill,state='TARGET_INACTIVE');self.action=action;return action,[],self.camera(packet),self.status
                # Stationary safe-enough occupancy should not wander within the circle.
                if skill=='CAPTURE_POINT' and np.linalg.norm(start-center)<point['radius']-.04 and (belief is None or p.point_risk(start)<.3):
                    candidates=[(0.,start,None)]
                else:
                    points=[center]+[center+point['radius']*.65*np.array([math.cos(a),math.sin(a)]) for a in np.linspace(0,2*math.pi,8,endpoint=False)]
                    for g in points:
                        if not p.free(g):continue
                        n=p.nearest(g)
                        if n in p.cost:candidates.append((p.cost[n]+p.segment_cost(np.asarray(n)*p.grid,g)+8*p.point_risk(g),g,None))
            else:
                covers=[c for c in self.base.covers if 'cover' not in params or c['name']==params['cover']]
                # Reject a wall that will vanish before a normal approach can complete.
                covers=[c for c in covers if not(c['name']==self.cfg['collapse']['wall'] and 0<self.cfg['collapse']['time']-t<4)]
                for n,cost in p.cost.items():
                    g=np.asarray(n)*p.grid
                    nearby=[c for c in covers if .42-1e-8<=clearance(g,c)<.85]
                    if not nearby:continue
                    risk=p.point_risk(g)
                    if belief is not None and risk>.2:continue
                    c=max(nearby,key=lambda v:v['size'][2]);low_penalty=3 if c['size'][2]<1 else 0
                    candidates.append((cost+12*risk+low_penalty,g,c['name']))
            plans=[]
            for cost,g,cname in sorted(candidates,key=lambda v:v[0])[:8]:
                rr=p.route(g)
                if rr is None:continue
                # Both forward and permitted reverse considered using own yaw/speed only.
                times=[(self.eta(own,rr,r),r) for r in (False,True)];tt,reverse=min(times)
                transit=self.exposure_seconds(own,rr,p,reverse) if belief else None
                budget=.4 if own['hp']<=1 else .65 if own['hp']==2 else 1.
                if skill!='TAKE_COVER' and params.get('risk','safe')=='safe' and transit is not None and transit>budget:
                    rejected+=1;continue
                plans.append((cost+.7*tt,g,rr,reverse,tt,cname,transit))
            if plans:
                _,goal,route,reverse,eta,cover_name,exposure=min(plans,key=lambda v:v[0])
                if np.linalg.norm(goal-start)<.09:
                    route=[];state='HOLDING_POINT' if skill=='CAPTURE_POINT' else 'AT_DESTINATION'
                else:action.update(skill='MOVE',goal=goal.tolist(),reverse=reverse)
                if skill=='CAPTURE_POINT':action['target_generation']=self.generation
            else:state='NO_SAFE_COVER' if skill=='TAKE_COVER' else 'UNSAFE_ROUTE' if rejected else 'NO_ROUTE'
        elif skill=='AIM_AT_TRACK':
            if belief is None or not belief['aim_usable']:state='TARGET_STALE';action['attack']=False
        elif skill=='OBSERVE':
            action['attack']=False;direction=params.get('direction','scan')
            if direction=='scan':action['skill']='SCAN'
            elif direction!='threat':action['facing']={'east':0.,'north':math.pi/2,'west':math.pi,'south':-math.pi/2}[direction]
        self.goal=None if goal is None else goal.tolist();self.action=action
        self.status=dict(skill=skill,state=state,goal=self.goal,chosen_cover=cover_name,estimated_arrival_seconds=round(eta,3),
            estimated_max_continuous_exposure_seconds=exposure,rejected_route_candidates=rejected,
            estimated_destination_risk=None if goal is None or belief is None else p.point_risk(goal),
            threat_source='unknown' if belief is None else belief['source'],route=[v.tolist() for v in route],
            note='ETA includes turning and acceleration; heuristic, not a guarantee. Risk uses visual estimates, not opponent truth.')
        return action,route,self.camera(packet),copy.deepcopy(self.status)

    def camera(self,packet):
        belief=self.memory.belief(packet['simulation_seconds']);own=packet['self_state']
        explicit_scan=self.intent=='OBSERVE' and self.params.get('direction','scan')!='threat'
        if not explicit_scan and belief is not None and belief['source']=='rgb_foot_pixel':
            a=math.atan2(belief['xy'][1]-own['xy'][1],belief['xy'][0]-own['xy'][0])
            return dict(mode='hold',yaw_degrees=float(np.clip(math.degrees(wrap(a-own['yaw'])),-90,90)))
        return dict(mode='sweep',center_degrees=0,amplitude_degrees=60)

    def tick(self,packet):
        """50Hz own-state guards; no detection or 50Hz visual updates implied."""
        if self.action is None:return None
        own=packet['self_state'];t=packet['simulation_seconds'];belief=self.memory.belief(t)
        if self.intent=='CAPTURE_POINT' and packet['public_state']['target_generation']!=self.generation:
            self.status['state']='TARGET_EXPIRED';self.action=dict(skill='STOP',facing='keep',attack=False);return self.action,[]
        if own['hp']<self.hp:
            self.hp=own['hp'];self.status['last_hit_time']=t
            # Local interruption: never select a new scoring target for the LLM.
            if self.intent=='CAPTURE_POINT':
                self.intent='TAKE_COVER';self.params={'risk':'safe'}
                action,route,_,_=self.plan(packet)
                self.status.update(state='EMERGENCY_COVER' if route else 'EMERGENCY_'+self.status['state'],last_hit_time=t,interrupted_skill='CAPTURE_POINT')
                return action,route
        if self.status['state'] in ('TARGET_EXPIRED','INTERRUPTED_HIT'):return None
        self.action['attack']=self.intent!='OBSERVE' and bool(belief and belief['aim_usable'])
        if belief is not None and (self.intent!='OBSERVE' or self.params.get('direction')=='threat'):
            self.action['facing']=math.atan2(belief['xy'][1]-own['xy'][1],belief['xy'][0]-own['xy'][0])
        if self.intent=='AIM_AT_TRACK' and not self.action['attack']:self.status['state']='TARGET_STALE'
        if self.goal is not None and np.linalg.norm(np.asarray(own['xy'])-self.goal)<.09:self.status['state']='HOLDING_POINT' if self.intent=='CAPTURE_POINT' else 'AT_DESTINATION'
        if own['skill_status']=='safety_stop':self.status['state']='BLOCKED_SAFETY_STOP'
        return None
