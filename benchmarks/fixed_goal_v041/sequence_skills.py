"""Public-observation-only burst peeking and conditional skill sequences."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from ds_fixed_goal import IntentController,SkillProbe,WAYPOINTS,FIELDS,validate_output as legacy_validate,read,dump,digest
from peek_geometry import PeekCatalog,PEEK,preview,ray_open
from fixed_goal_probe import PublicController
from tactical_logic import wrap
from continuous_motion import rounded_route

FEATURES=dict(peek_enabled=True,sequence_enabled=True,event_driven=True,
              horizon_seconds=8.,max_nodes=5,max_cycles=3,guard_memory_seconds=8.,guard_hold_seconds=1.2)

def validate_plan_output(text,index):
    a=json.loads(text)
    if not isinstance(a,dict) or set(a)!={'observation_id','skill','params','assessment'} or type(a['observation_id']) is not int or a['observation_id']!=index:raise ValueError('Bad action envelope')
    if a['skill']!='PLAN':raise ValueError('Expected PLAN')
    p=a['params']
    if not isinstance(p,dict) or set(p)!={'steps','horizon_seconds'} or type(p['horizon_seconds']) not in (int,float) or not 4<=p['horizon_seconds']<=8:raise ValueError('Bad plan horizon')
    if not isinstance(p['steps'],list) or not 1<=len(p['steps'])<=5:raise ValueError('Plan needs 1..5 nodes')
    for node in p['steps']:
        if not isinstance(node,dict) or set(node)!={'skill','params','on_contact'} or node['on_contact'] not in ('continue','retreat'):raise ValueError('Bad conditional node')
        if node['skill']=='PEEK_BURST':
            q=node['params']
            if set(q)!={'candidate_id','max_cycles'} or not isinstance(q['candidate_id'],str) or not q['candidate_id'].startswith('PK') or type(q['max_cycles']) is not int or not 1<=q['max_cycles']<=3:raise ValueError('Bad peek node')
        elif node['skill']=='FOLLOW_ROUTE':
            q=node['params']
            if set(q)!={'waypoints','posture','fire'} or q['posture'] not in ('upright','low') or q['fire'] not in ('off','opportunistic') or not isinstance(q['waypoints'],list) or not 1<=len(q['waypoints'])<=4 or any(v not in WAYPOINTS for v in q['waypoints']):raise ValueError('Bad continuous route')
        elif node['skill'] in ('MOVE','TAKE_COVER','TAG_AND_ADVANCE','SMOKE_AND_MOVE'):
            legacy_validate(json.dumps(dict(observation_id=index,skill=node['skill'],params=node['params'],assessment=a['assessment'])),index)
        else:raise ValueError('Unsupported plan node')
    if not isinstance(a['assessment'],dict) or set(a['assessment'])!=set(FIELDS) or any(not isinstance(v,str) or not 1<=len(v)<=1600 for v in a['assessment'].values()):raise ValueError('Bad assessment')
    return a

class MemoryGuard(PublicController):
    """No hidden actor reference: observed position, self pose and static map only."""
    def __init__(self,old,planner,events):
        super().__init__(old.kind,old.actor,old.route,old.home,old.patrol,old.rules)
        self.planner=planner;self.events=events;self.last_seen=None;self.search_target=None;self.last_guard_state=None
    def decide(self,p):
        t=p['time'];seen=p['visible'];previous_seen=self.last_seen
        if seen is not None:
            if self.first_visible is None and previous_seen is not None and t-previous_seen<=3.:self.first_visible=t-self.rules['reaction_seconds']
            self.last_seen=t;self.search_target=None
            return super().decide(p)
        self.first_visible=None;memory=p['memory']
        if memory is None or t-memory['last_seen']>FEATURES['guard_memory_seconds']:return super().decide(p)
        age=t-memory['last_seen'];target=memory['position'];mode='WATCH_LAST_SEEN' if age<FEATURES['guard_hold_seconds'] else 'INVESTIGATE_LAST_SEEN'
        if mode!=self.last_guard_state:
            self.events.append(dict(time=t,type='guard_memory',state=mode,last_seen=memory['last_seen'],known_position=list(target)))
            self.last_guard_state=mode
        if mode=='WATCH_LAST_SEEN':return dict(mode='WATCH',look=target)
        if self.search_target is None:
            own=np.array(p['self']['position']);delta=own-np.array(target);norm=np.linalg.norm(delta)
            q=np.array(target)+delta/max(norm,1e-9)*min(1.2,norm)
            route=self.planner.route(own,q)
            self.search_target=q.tolist() if self.planner.free(q) and route is not None else list(target)
        return dict(mode='PATROL',target=self.search_target,look=target)

class SequenceController(IntentController):
    def __init__(self,decider,base,rules,folder,features=None):
        super().__init__(decider,base,rules,folder)
        self.features=dict(FEATURES,**(features or {}));self.catalog=PeekCatalog(base);self.steps=[];self.node_index=0
        self.plan_started=0.;self.horizon=8.;self.reason='initial';self.events=[];self.node_started=0.;self.node_phase='idle'
        self.peek=None;self.cycle=0;self.phase_started=0.;self.visible_started=None;self.max_exposure=0.;self.peek_samples=[]
        self.completed_bursts=[];self.emergency_target=None;self.last_visible=None;self.last_visible_at=None;self.trace_extra={}
        self.executed_skill='IDLE';self.last_model_time=-99.;self.pending=False;self.public_packets=[];self.last_goal=None

    def event(self,t,kind,**kw):self.events.append(dict(time=round(t,2),type=kind,**kw))

    def track(self,p):
        if p['visible'] is not None:return p['visible']['position'],p['time'],'current'
        if p['memory'] is not None:return p['memory']['position'],p['memory']['last_seen'],'memory'
        if p['time']==0:return self.base['start_b'],0.,'initial_only'
        return None,None,'unknown'

    def request(self,p):
        t=p['time'];threat,observed,source=self.track(p);options,smoke=self.options(p)
        candidates=self.catalog.candidates(p['self']['position'],threat,WAYPOINTS['GOAL'],observed,t) if self.features['peek_enabled'] else []
        samples=self.samples
        # Keep all transitions plus uniformly spaced samples; full legal data is logged separately.
        keep=set(range(0,len(samples),max(1,math.ceil(len(samples)/70))))
        if samples:keep.add(len(samples)-1)
        for i in range(1,len(samples)):
            if (samples[i][1] is None)!=(samples[i-1][1] is None):keep.update((i-1,i))
        recent=[samples[i] for i in sorted(keep)]
        velocity=None
        visible_samples=[v for v in samples if v[1] is not None]
        if len(visible_samples)>=2:
            a,b=visible_samples[0],visible_samples[-1];dt=b[0]-a[0]
            if dt>=.1:velocity=dict(estimate_xy_mps=[round((b[1][j]-a[1][j])/dt,3) for j in (0,1)],first_time=a[0],last_time=b[0],assumption='Average across legal samples; not a future-motion guarantee.')
        index=len(self.decisions)
        public=dict(observation_id=index,simulation_seconds=t,decision_trigger=self.reason,self_state=copy.deepcopy(p['self']),
            current_detection=copy.deepcopy(p['visible']),last_legal_memory=copy.deepcopy(p['memory']),track_source=source,
            sensor_samples_since_previous_decision=copy.deepcopy(recent),full_legal_sample_count=len(samples),motion_estimate=velocity,
            goal=dict(center=WAYPOINTS['GOAL'],radius=1.),remaining_seconds=round(60-t,2),route_options=options,smoke_suggestions=smoke,
            peek_candidates=candidates,recent_skill_events=copy.deepcopy(self.events[-14:]),recent_bursts=copy.deepcopy(self.completed_bursts[-3:]),
            model_call_behind_known_cover=bool(threat is not None and not ray_open(p['self']['position'],threat,self.catalog.walls)),
            cover_assessment_observation_age=None if observed is None else round(t-observed,3))
        self.public_packets.append(dict(time=t,samples=copy.deepcopy(samples)))
        action=self.decider(copy.deepcopy(public));validate_plan_output(json.dumps(action),index)
        if not self.features['sequence_enabled'] and (len(action['params']['steps'])>1 or any(n['skill']=='FOLLOW_ROUTE' for n in action['params']['steps'])):raise ValueError('Sequence feature disabled')
        for node in action['params']['steps']:
            if node['skill']=='PEEK_BURST' and node['params']['candidate_id'] not in self.catalog.registry:raise ValueError('Peek candidate not offered')
        self.steps=copy.deepcopy(action['params']['steps']);self.node_index=0;self.plan_started=t;self.last_model_time=t;self.horizon=action['params']['horizon_seconds'];self.pending=False
        self.node_phase='start';self.node_started=t;self.peek=None;self.emergency_target=None
        self.decisions.append(dict(**action,simulation_seconds=t,public_observation=public));self.samples=[]
        dump(self.folder/'decisions.json',self.decisions);dump(self.folder/'legal_samples.json',self.public_packets)
        self.event(t,'plan_accepted',decision=index,nodes=len(self.steps),horizon=self.horizon)

    def finish_node(self,t,reason):
        self.event(t,'node_completed',node=self.node_index,skill=self.executed_skill,reason=reason)
        self.node_index+=1;self.node_phase='start';self.node_started=t;self.peek=None;self.tag_done=False
        if self.node_index>=len(self.steps):self.pending=True;self.reason='plan_completed'

    def retreat(self,p,reason):
        threat,_,_=self.track(p)
        q=self.catalog.nearest_hidden(p['self']['position'],threat) if threat is not None else None
        if q is None:
            self.pending=True;self.reason='no_known_safe_retreat';self.event(p['time'],'interrupt',reason=reason,retreat_found=False)
            return None
        self.emergency_target=q;self.event(p['time'],'interrupt',reason=reason,retreat_found=True,target=q)
        self.node_phase='emergency';return dict(posture='upright',mode='ADVANCE',target=q)

    def peek_command(self,p,node):
        t=p['time'];own=p['self'];q=own['position'];seen=p['visible']
        if self.peek is None:
            self.peek=copy.deepcopy(self.catalog.registry[node['params']['candidate_id']]);self.cycle=0;self.node_phase='approach'
            self.peek_samples=[];self.visible_started=None;self.max_exposure=0.
            self.event(t,'peek_selected',candidate=self.peek['id'],home=self.peek['home'],out=self.peek['out'])
        home,out=self.peek['home'],self.peek['out'];yaw=math.atan2(out[1]-home[1],out[0]-home[0])
        if self.node_phase=='approach':
            if math.dist(q,home)>.025 or abs(own['speed'])>.03:return dict(posture='upright',mode='ADVANCE',target=home)
            self.node_phase='align';self.phase_started=t
        if self.node_phase=='align':
            if abs(wrap(own['yaw']-yaw))>.025 or abs(own['speed'])>.01 or own.get('transition_remaining',0)>1e-8:return dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(yaw))
            threat,at,source=self.track(p)
            # A stale prediction is explicitly marked; it is never a current truth update.
            if threat is None:threat=self.peek['expected_threat'];at=self.peek['observed_at'];source='stale_plan_hypothesis'
            if ray_open(q,threat,self.catalog.walls):
                self.event(t,'peek_rejected',reason='home_exposed_to_latest_legal_track',candidate=self.peek['id'])
                self.finish_node(t,'unsafe_home');return dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(own['yaw']))
            check=preview(q,out,threat,self.catalog.planner)
            if not check['feasible'] or check['max_exposure']>PEEK['max_exposure']:
                self.event(t,'peek_rejected',reason='motion_or_exposure_budget',candidate=self.peek['id']);self.finish_node(t,'preview_rejected')
                return dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(own['yaw']))
            self.peek['home']=list(q);self.peek['expected_threat']=list(threat);self.peek['observed_at']=at
            self.node_phase='out';self.phase_started=t;self.burst_started=t;self.visible_started=None;self.max_exposure=0.;self.peek_samples=[]
            self.event(t,'peek_burst_started',cycle=self.cycle+1,source=source,assumption_age=t-at,predicted_exposure=check['max_exposure'])
        if self.node_phase in ('out','hold','back'):
            possible=ray_open(q,self.peek['expected_threat'],self.catalog.walls) or seen is not None
            if possible:
                if self.visible_started is None:self.visible_started=t
                self.max_exposure=max(self.max_exposure,t-self.visible_started)
            else:self.visible_started=None
            if seen is not None:self.peek_samples.append(dict(time=t,position=seen['position'],yaw=seen['yaw']))
            # Abort observation immediately if the new legal sighting invalidates the return shield.
            invalid_home=seen is not None and ray_open(home,seen['position'],self.catalog.walls)
            if self.node_phase in ('out','hold') and (invalid_home or self.max_exposure>.40):
                self.event(t,'peek_early_return',reason='home_changed' if invalid_home else 'exposure_reserve')
                self.node_phase='back';self.phase_started=t
        if self.node_phase=='out':
            if math.dist(q,out)<1e-6 and abs(own['speed'])<.01:self.node_phase='hold';self.phase_started=t
            else:return dict(posture='upright',mode='PEEK_LOCAL',target=out)
        if self.node_phase=='hold':
            if t-self.phase_started<PEEK['hold']-1e-9:return dict(posture='upright',mode='PEEK_HOLD',look_degrees=math.degrees(yaw))
            self.node_phase='back';self.phase_started=t
        if self.node_phase=='back':
            if math.dist(q,self.peek['home'])>1e-6 or abs(own['speed'])>.01:return dict(posture='upright',mode='PEEK_LOCAL',target=self.peek['home'])
            threat,_,_=self.track(p);home_safe=threat is None or not ray_open(q,threat,self.catalog.walls)
            self.cycle+=1
            record=dict(start=self.burst_started,end=t,cycle=self.cycle,candidate=self.peek['id'],max_estimated_exposure=round(self.max_exposure,3),
                        samples=len(self.peek_samples),returned_home=True,home_safe_for_latest_legal_track=home_safe,
                        first_sample=None if not self.peek_samples else self.peek_samples[0],last_sample=None if not self.peek_samples else self.peek_samples[-1])
            self.completed_bursts.append(record);self.event(t,'peek_burst_completed',**{k:v for k,v in record.items() if k not in ('start','end')})
            if not home_safe:return self.retreat(p,'peek_anchor_invalidated') or dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(yaw))
            if self.cycle>=node['params']['max_cycles'] or (self.cycle>=2 and len(self.peek_samples)>=3):
                self.finish_node(t,'observation_complete');self.pending=True;self.reason='peek_completed_in_cover'
                return dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(yaw))
            self.node_phase='rest';self.phase_started=t
        if self.node_phase=='rest':
            if t-self.phase_started>=PEEK['hidden_pause']:self.node_phase='align'
            return dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(yaw))
        return dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(yaw))

    def decide(self,p):
        t=p['time'];seen=p['visible'];own=p['self'];q=own['position']
        self.samples.append([round(t,2),None if seen is None else [round(float(x),3) for x in seen['position']]+[round(seen['yaw'],4)]])
        if seen is None:self.first_visible=None
        elif self.first_visible is None:self.first_visible=t
        if self.emergency_target is not None:
            self.executed_skill='EMERGENCY_RETREAT'
            if math.dist(q,self.emergency_target)>.06 or abs(own['speed'])>.15:return self.record(t,dict(posture='upright',mode='ADVANCE',target=self.emergency_target))
            self.emergency_target=None;self.pending=True;self.reason='retreated_to_known_cover'
        critical=self.peek is not None and self.node_phase in ('out','hold','back')
        if self.steps and not self.features['event_driven'] and t-self.last_model_time>=2.-1e-8 and not critical:
            self.pending=True;self.reason='periodic_two_second_baseline'
        if self.steps and t-self.plan_started>=self.horizon-1e-8 and not critical:
            self.pending=True;self.reason='plan_horizon'
            threat,_,_=self.track(p)
            if threat is not None and ray_open(q,threat,self.catalog.walls):
                cmd=self.retreat(p,'horizon_in_exposed_position')
                if cmd is not None:return self.record(t,cmd)
        if not self.steps or self.pending:
            if self.decisions and t-self.last_model_time<.5:
                return self.record(t,dict(posture='upright',mode='WAIT_FIXED',look_degrees=math.degrees(own['yaw'])))
            self.request(p)
        node=self.steps[self.node_index];self.executed_skill=node['skill'];params=node['params']
        if self.executed_skill=='PEEK_BURST':return self.record(t,self.peek_command(p,node))
        posture=params['posture'];target=WAYPOINTS[params['waypoints'][-1] if self.executed_skill=='FOLLOW_ROUTE' else params['target']]
        if math.dist(q,target)<.14 and abs(own['speed'])<.25:
            self.finish_node(t,'target_reached');return self.record(t,dict(posture=posture,mode='WAIT_FIXED',look_degrees=math.degrees(own['yaw'])))
        if node['on_contact']=='retreat' and seen is not None and math.dist(q,seen['position'])<=8:
            cmd=self.retreat(p,'contact_guard');return self.record(t,cmd or dict(posture=posture,mode='WAIT_FIXED',look_degrees=math.degrees(own['yaw'])))
        ready=(seen is not None and self.first_visible is not None and t-self.first_visible>=self.rules['reaction_seconds']-1e-8 and math.dist(q,seen['position'])<=8 and own['tag_cooldown_remaining']<=1e-8 and seen.get('disabled_remaining',0)<=1e-8)
        if self.executed_skill=='TAG_AND_ADVANCE':
            if own['tag_cooldown_remaining']>1e-8:self.tag_done=True
            if ready and not self.tag_done:return self.record(t,dict(posture=posture,mode='AIM',look=seen['position']))
        if self.executed_skill in ('MOVE','FOLLOW_ROUTE') and params['fire']=='opportunistic' and ready:return self.record(t,dict(posture=posture,mode='AIM',look=seen['position']))
        if self.executed_skill=='SMOKE_AND_MOVE' and p['smoke_available']:return self.record(t,dict(posture=posture,mode='THROW',target=params['location'],destination=target))
        if self.executed_skill=='FOLLOW_ROUTE':return self.record(t,dict(posture=posture,mode='FOLLOW_PATH',targets=[WAYPOINTS[k] for k in params['waypoints']],route_key=[len(self.decisions),self.node_index]))
        self.node_phase='travel';return self.record(t,dict(posture=posture,mode='ADVANCE',target=target))

    def record(self,t,command):
        self.trace_extra[t]=dict(exec_skill=self.executed_skill,plan_node=self.node_index,skill_phase=self.node_phase,
                                peek_cycle=self.cycle,peek_exposure_estimate=round(self.max_exposure,3))
        return command

class SequenceProbe(SkillProbe):
    def __init__(self,base,rules,decider,folder,features=None):
        super().__init__(base,rules,decider,folder)
        self.controllers[0]=SequenceController(decider,self.cfg,rules,folder,features)
        self.controllers[1]=MemoryGuard(self.controllers[1],self.planner,self.events)
        self.metadata['architecture']='peek-sequences-v0.4';self.local_target=None;self.chain_key=None;self.chain_next=0;self.chain_paused=False
    def execute(self,i,command,t):
        if i==1:return super().execute(i,command,t)
        m=self.actor[0];mode=command['mode']
        if mode in ('PEEK_LOCAL','PEEK_HOLD'):
            m.reverse_allowed=True;m.reverse_limit=PEEK['reverse_speed'];m.vmax=PEEK['speed'];m.accel=PEEK['acceleration']
            if mode=='PEEK_HOLD':
                self.stop(0);self.local_target=None
                return math.radians(command['look_degrees']),False,'PEEK_HOLD'
            target=command['target'];key=tuple(target)
            if key!=self.local_target:
                if not self.planner.segment(m.xy,target):raise RuntimeError('Peek local path invalid')
                m.navigate([np.array(target)]);self.local_target=key;self.move_goal[0]=None
            return None,False,'PEEK_LOCAL'
        self.local_target=None;m.reverse_allowed=False
        if mode=='FOLLOW_PATH':
            # Build one smooth path through public map waypoints; do not stop at every hint.
            key=tuple(command['route_key']);targets=command['targets']
            if key!=self.chain_key:self.chain_key=key;self.chain_next=0;self.chain_paused=True
            while self.chain_next<len(targets)-1 and math.dist(m.xy,targets[self.chain_next])<.65:self.chain_next+=1
            if self.chain_paused or not m.route:
                route=[];at=m.xy.copy()
                for q in targets[self.chain_next:]:
                    part=self.planner.route(at,q)
                    if part is None:self.stop(0);self.feedback='NO_ROUTE_TO_SEQUENCE';return None,False,'NO_ROUTE'
                    route.extend(part);at=np.array(q)
                m.navigate(rounded_route(m.xy,route,self.planner));self.chain_paused=False;self.move_goal[0]=None
                self.feedback='CONTINUOUS_ROUTE_ACTIVE'
            # Preserve the normal pose transition handler without rebuilding this route.
            pose=command['posture']
            if pose!=self.pose_target:
                self.pose_from=self.heights[0];self.pose_start=t;self.pose_until=t+.4;self.pose_target=pose
                self.events.append(dict(time=t,type='posture_requested',posture=pose,until=self.pose_until))
            return None,False,'FOLLOW_PATH'
        self.chain_paused=True
        return super().execute(i,command,t)
    def run(self,identifier):
        result=super().run(identifier);controller=self.controllers[0]
        last=dict(exec_skill='IDLE',plan_node=0,skill_phase='idle',peek_cycle=0,peek_exposure_estimate=0.)
        for s in result['states']:
            if s['t'] in controller.trace_extra:last=controller.trace_extra[s['t']]
            s.update(last)
        result['skill_events']=controller.events;result['peek_bursts']=controller.completed_bursts
        result['summary'].update(peek_bursts=len(controller.completed_bursts),skill_event_count=len(controller.events),condition='peek_sequences_v04')
        dump(self.folder/'skill_events.json',controller.events);dump(self.folder/'peek_bursts.json',controller.completed_bursts)
        return result
