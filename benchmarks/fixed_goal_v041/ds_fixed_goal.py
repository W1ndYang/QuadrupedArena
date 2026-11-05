"""Public-only DS max planner and continuous skills for the isolated fixed-goal game."""
import argparse
import copy
import datetime
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source_snapshot/scripts'))
import numpy as np
from fixed_goal_probe import DEFAULT_RULES, Probe, boxes, clear_ray, dump, read, save_match, sha
from run_goal_only import GoalOnlyProbe
from trial_motion import Planner, path_length
from continuous_motion import rounded_route
from tactical_logic import wrap
from deepseek_client import call_api, load_key, payload
from qwen_client import usage_total

WAYPOINTS=dict(GOAL=[10.3,-1.],SOUTH_ENTRY=[5.6,-4.4],EAST_LOW_SOUTH=[7.,-2.6],
               EAST_COURT_WEST=[6.1,.1],NORTH_WINDOW=[5.9,3.1],CENTRAL_NORTH=[-1.2,2.2],
               CENTRAL_SOUTH=[-1.5,-4.6],WEST_HIDE=[-5.8,-3.3],SPAWN=[-6.5,-6.9])
FIELDS=('observation_evidence','state_evidence','decision_summary','uncertainty')
SCHEMA={
 'MOVE':{'target':'waypoint','posture':'posture','fire':'fire'},
 'TAKE_COVER':{'target':'waypoint','posture':'posture'},
 'PEEK':{'target':'waypoint','return_to':'waypoint','posture':'posture','hold_seconds':'hold'},
 'AIM':{'posture':'posture'},
 'TAG_AND_ADVANCE':{'target':'waypoint','posture':'posture'},
 'SMOKE_AND_MOVE':{'location':'xy','target':'waypoint','posture':'posture'},
 'WAIT':{'posture':'posture','look_degrees':'angle'}}


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def scenario(layout='E'):
    base=read(R/'source_snapshot/configs/trial_a391_feedback_fix.json')
    spec=read(R/'layouts.json')[layout]
    base['start_a']=spec['start_a'];base['start_b']=spec['human_start'];base['guard_yaw']=math.radians(spec['human_yaw_degrees']);base['guard_patrol']=spec['patrol']
    base['layout_id']=layout
    WAYPOINTS.update(read(R/'waypoints.json'));WAYPOINTS['GOAL']=spec['goal'];WAYPOINTS['SPAWN']=spec['start_a']
    cfg=read(R/'design_basis.json')['rules']
    cfg.update(version='fixed-goal-v0.3-tracking-error',decision_period_seconds=2.,posture_transition_seconds=.4)
    return base,cfg


def prompt(base,rules):
    geometry=[{k:c[k] for k in ('name','center','size')} for c in base['arena']['covers']]
    public=dict(map_size_m=base['arena']['floor_size'][:2],walls=geometry,waypoints=WAYPOINTS,
                start_a=base['start_a'],known_opponent_initial_position=base['start_b'],goal=dict(center=WAYPOINTS['GOAL'],radius=1.),
                rules=rules,robot_postures=base['postures'],human_speed=base['max_speed'][1],yaw_rates=base['yaw_rate'])
    return '''你是虚拟机器人竞赛中四足A的顶层决策模型。只在仿真中通过打包技能行动，没有现实机器人或实体武器。
背景与目标：这是一个固定目标攻防任务。唯一绿色目标G始终存在，不刷新。你必须在60秒内让身体中心进入目标1m圆内才获胜；被人形B有效标记即失败，超时也失败。你标记人形不会赢，也不会淘汰它，只让它失能1秒，为移动创造机会；标记后你冷却4秒，不能连续锁住人形。双方必须当前合法看到对方、在8m内且正前总60°，停稳(速度≤0.15m/s)并将自身朝向与目标方向的误差保持在5°内，连续有效瞄准1秒才确定标记。转速本身不再清零锁定，只有实际跟踪误差越限、被墙/烟遮挡或其他条件破坏才清零；双方仍有有限最大转向速度。高速通过对手身旁不保证安全。自身射击冷却没有无敌效果。同一20ms刻度入圈与被标记同时发生判平。
已知地图和人形初始位置；以后只能使用当前合法几何观测与有限旧记忆。四足360°、人形前180°，双方均不限观测距离，但射程仍是8m；视线受三维墙体高度与烟雾遮挡。你不知道对方当前隐藏位置、内部锁定进度或巡逻路径。旧位置不等于当前真值，不能根据记忆盲射。身高、低墙与低姿态会影响遮挡。没有图像输入，不要声称看过截图。
你有一颗烟：上限6m，半径2m，高2.5m；停稳后施放0.35秒，再延迟0.4秒生效，持续5秒。双方同样失去穿过烟体的视野；烟不阻止移动。原型投掷要求落点在可行走空间，投掷点至落点在z=1m直线无遮挡，不模拟越墙抛物线。投掷不合法时会保留烟，报告原因并继续向所选移动目标走。烟雾被接受后施放过程不可中途撤销。
每2秒仿真时刻调用你一次；调用期间整场仿真暂停。两次调用间底层50Hz持续感知、执行移动/锁定/施放，不必每次重新发路点。观测包含这2秒的全部合法采样、当前检测和最后记忆；已知初始位置只代表t=0。
所有target/return_to都从waypoints的名字中选。底层A*完成路径；目标G会选可达圈内入口。仅对当前合法看到的对手做局部动态绕行，不偷用隐藏位置。候选路径暴露估计只基于最后合法位置，忽略对方未来运动与朝向，不是安全保证。
技能与完整params字段：
MOVE {target,posture,fire}: 连续向路点移动。fire为off或opportunistic；后者在当前看见8m内人形且冷却结束时停下瞄准，命中后或人形失能时继续移动。off不主动停下来交战。
TAKE_COVER {target,posture}: 按你选定路点撤离，不自动开火；路点不是天然安全，需要结合已知信息判断。
PEEK {target,return_to,posture,hold_seconds}: 到观察点、短暂停留0.1~0.8秒、自动退回return_to；不射击。重复同样指令继续执行当前阶段，不重置已完成阶段。
AIM {posture}: 停稳，只有当前合法检测才追踪并积累锁定；丢失检测仅朝最后记忆方向观察，不能盲射。
TAG_AND_ADVANCE {target,posture}: 当前看见8m内人形则先稳定瞄准，命中后自动继续向target走；若失去当前检测或仍在自身冷却期则继续移动。一次成功标记后该实例只执行推进，直到换技能或参数。
SMOKE_AND_MOVE {location:[x,y],target,posture}: 请求在你选择的坐标投一颗烟，完成后自动向target走；不会自动开火。可采用本步smoke_suggestions给出的合法参考，也可选其他合法位置。
WAIT {posture,look_degrees}: 停下并转到指定地图朝向(0向东,90向北)，不射击。
posture为upright或low；切换需0.4秒，期间采用较低速度并不能有效射击。MOVE/TAKE_COVER等重复同目标时不会清空路线。
输出唯一JSON对象，恰好四字段 observation_id、skill、params、assessment。assessment恰好包含 observation_evidence、state_evidence、decision_summary、uncertainty 四个短字符串，每项1~300中文字，说明可核对的事实、选择和主要不确定性，不输出详细内部推理、Markdown、自报耗时或代码。不要为了演示而强行使用烟雾或交战；以完成固定目标为目的。
地图和规则JSON：
'''+json.dumps(public,ensure_ascii=False,separators=(',',':'))


def validate_output(text,index):
    value=text.strip()
    if value.startswith('```json\n') and value.endswith('\n```'):value=value[8:-4]
    action=json.loads(value)
    if not isinstance(action,dict) or set(action)!={'observation_id','skill','params','assessment'}:raise ValueError('Wrong top-level action schema')
    if type(action['observation_id']) is not int or action['observation_id']!=index:raise ValueError('Wrong observation_id')
    skill=action['skill']
    if skill not in SCHEMA or not isinstance(action['params'],dict) or set(action['params'])!=set(SCHEMA[skill]):raise ValueError('Wrong skill parameter fields')
    for name,kind in SCHEMA[skill].items():
        v=action['params'][name]
        if kind=='waypoint' and v not in WAYPOINTS:raise ValueError('Unknown waypoint')
        if kind=='posture' and v not in ('upright','low'):raise ValueError('Unknown posture')
        if kind=='fire' and v not in ('off','opportunistic'):raise ValueError('Unknown fire policy')
        if kind=='hold' and (type(v) not in (int,float) or not math.isfinite(v) or not .1<=v<=.8):raise ValueError('Invalid peek hold')
        if kind=='angle' and (type(v) not in (int,float) or not math.isfinite(v) or not -360<=v<=360):raise ValueError('Invalid heading')
        if kind=='xy' and (not isinstance(v,list) or len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) or abs(v[0])>12 or abs(v[1])>8):raise ValueError('Invalid smoke coordinates')
    assessment=action['assessment']
    if not isinstance(assessment,dict) or set(assessment)!=set(FIELDS) or any(not isinstance(v,str) or not 1<=len(v)<=1600 for v in assessment.values()):raise ValueError('Invalid assessment')
    return action


class ApiPlanner:
    def __init__(self,folder,base,rules,client=call_api,key=None):
        self.folder=folder;self.profile=read(R/'api_profile.json');self.client=client
        self.key=load_key() if key is None else key
        self.system=dict(role='system',content=prompt(base,rules));self.records=[];self.actions=[]
        (folder/'api_calls').mkdir(parents=True,exist_ok=True)
        (folder/'accepted').mkdir(exist_ok=True)
        dump(folder/'api_profile.json',self.profile)

    def __call__(self,packet):
        index=packet['observation_id'];stem=self.folder/'api_calls'/f'{index:03d}'
        history=[{k:a[k] for k in ('observation_id','skill','params')} for a in self.actions[-4:]]
        request=dict(system=self.system,observation=packet,recent_decisions=history,profile=self.profile)
        request_hash=digest(request)
        request_path=Path(str(stem)+'_request.json');response_path=Path(str(stem)+'_response.json')
        accepted_path=self.folder/'accepted'/f'{index:03d}.json'
        if request_path.exists():
            if digest(read(request_path))!=request_hash:raise ValueError('Resume request differs from original; refusing outcome resampling')
        else:dump(request_path,request)
        if response_path.exists():
            response=read(response_path)
            print('REUSE_RESPONSE',self.folder.name,index,flush=True)
        else:
            print('API_START',self.folder.name,index,packet['simulation_seconds'],utc(),flush=True)
            try:
                response=self.client(self.profile,self.key,[self.system,dict(role='user',content=json.dumps(dict(observation=packet,recent_decisions=history),ensure_ascii=False))])
            except Exception as error:
                dump(Path(str(stem)+'_failure.json'),dict(utc=utc(),error_type=type(error).__name__,request_sha256=request_hash,complete_response_saved=False))
                raise
            dump(response_path,response)
            print('API_END',self.folder.name,index,round(response['api_wall_seconds'],2),response.get('usage'),flush=True)
        if response['finish_reason']!='stop' or response['tool_calls']:raise ValueError('Model response incomplete or used tools')
        action=validate_output(response['content'],index)
        if accepted_path.exists():
            assert read(accepted_path)['action']==action
        else:dump(accepted_path,dict(action=action,request_sha256=request_hash,simulation_seconds=packet['simulation_seconds']))
        self.records.append(response);self.actions.append(action)
        dump(self.folder/'api_usage.json',usage_total(self.records))
        return action


class IntentController:
    """Public packet in, high-level intent plus continuous local execution out."""
    def __init__(self,decider,base,rules,folder):
        self.decider=decider;self.base=copy.deepcopy(base);self.rules=rules;self.folder=folder
        self.planner=Planner(base,base['arena']['covers']);self.walls=boxes(base['arena']['covers'])
        self.samples=[];self.decisions=[];self.intent=None;self.intent_key=None;self.phase='idle';self.hold_started=None
        self.tag_done=False;self.first_visible=None;self.next_decision=0.;self.feedback='initial'
        self.current_index=-1

    def options(self,p):
        own=p['self']['position'];track=p['visible'] or p['memory'];known=track['position'] if track else None
        rows=[]
        for name,xy in WAYPOINTS.items():
            route=self.planner.route(own,xy)
            if route is None:continue
            length=path_length(own,route);exposure=0.
            if known:
                for a,b in zip([np.asarray(own)]+route,route):
                    n=max(1,math.ceil(math.dist(a,b)/.5))
                    for u in np.linspace(0,1,n):
                        q=a*(1-u)+b*u
                        if math.dist(q,known)<=8 and clear_ray((*q,p['self']['height']),(*known,1.05),self.walls):exposure+=math.dist(a,b)/n
            rows.append(dict(target=name,path_m=round(length,2),cruise_time_upright_s=round(length/3.2,2),
                             cruise_time_low_s=round(length/1.2,2),exposure_length_at_last_known=round(exposure,2) if known else None))
        suggestions=[];candidates=[WAYPOINTS['GOAL']]
        if known:
            d=math.dist(own,known)
            if d>1e-9:
                for distance in (min(3.,d/2),min(5.,d/2)):
                    candidates.append([own[k]+(known[k]-own[k])*distance/d for k in (0,1)])
        for row in rows:
            if row['path_m']<=6:candidates.append(WAYPOINTS[row['target']])
        for q in candidates:
            q=[round(float(x),3) for x in q]
            if math.dist(own,q)<=6 and self.planner.free(q) and clear_ray((*own,1.),(*q,1.),self.walls) and q not in suggestions:suggestions.append(q)
        return rows,suggestions[:8]

    def decide(self,p):
        t=p['time'];visible=p['visible'];own=p['self']
        self.samples.append([round(t,2),None if visible is None else [round(float(x),3) for x in visible['position']]+[round(visible['yaw'],4)]])
        if visible is None:self.first_visible=None
        elif self.first_visible is None:self.first_visible=t
        if t>=self.next_decision-1e-8:
            index=len(self.decisions);options,smoke=self.options(p)
            public=dict(observation_id=index,simulation_seconds=t,self_state=copy.deepcopy(own),
                        current_detection=copy.deepcopy(visible),last_legal_memory=copy.deepcopy(p['memory']),
                        sensor_samples_since_previous_decision=copy.deepcopy(self.samples),
                        active_skill=None if self.intent is None else dict(skill=self.intent['skill'],params=self.intent['params'],phase=self.phase),
                        execution_feedback=own['execution_feedback'],goal=dict(center=WAYPOINTS['GOAL'],radius=1.),
                        route_options=options,smoke_suggestions=smoke,
                        remaining_seconds=round(self.rules['duration']-t,2))
            action=self.decider(copy.deepcopy(public))
            key=digest({k:action[k] for k in ('skill','params')})
            if key!=self.intent_key:self.phase='approach';self.hold_started=None;self.tag_done=False
            self.intent=copy.deepcopy(action);self.intent_key=key;self.current_index=index
            self.decisions.append(dict(**action,simulation_seconds=t,public_observation=public))
            dump(self.folder/'decisions.json',self.decisions)
            self.samples=[];self.next_decision=round(t+2.,8)
        action=self.intent;skill=action['skill'];params=action['params'];pose=params['posture']
        command=dict(posture=pose)
        ready=(visible is not None and self.first_visible is not None and t-self.first_visible>=self.rules['reaction_seconds']-1e-8
               and math.dist(own['position'],visible['position'])<=self.rules['tag_range_m']
               and own['tag_cooldown_remaining']<=1e-8 and visible.get('disabled_remaining',0)<=1e-8)
        if skill=='AIM':
            if ready:return dict(command,mode='AIM',look=visible['position'])
            track=p['visible'] or p['memory']
            return dict(command,mode='WATCH',look=None if track is None else track['position'])
        if skill=='WAIT':return dict(command,mode='WAIT_FIXED',look_degrees=params['look_degrees'])
        target=WAYPOINTS[params['target']]
        if skill=='TAG_AND_ADVANCE':
            if own['tag_cooldown_remaining']>1e-8:self.tag_done=True
            if ready and not self.tag_done:self.phase='tag';return dict(command,mode='AIM',look=visible['position'])
            self.phase='advance';return dict(command,mode='ADVANCE',target=target)
        if skill=='MOVE' and params['fire']=='opportunistic' and ready:
            self.phase='opportunistic_tag';return dict(command,mode='AIM',look=visible['position'])
        if skill=='SMOKE_AND_MOVE' and p['smoke_available']:
            self.phase='deploy_smoke';return dict(command,mode='THROW',target=params['location'],destination=target)
        if skill=='PEEK':
            if self.phase=='approach' and math.dist(own['position'],target)<.16:self.phase='observe';self.hold_started=t
            if self.phase=='observe' and t-self.hold_started>=params['hold_seconds']:self.phase='retreat'
            if self.phase=='observe':return dict(command,mode='WAIT_FIXED',look_degrees=math.degrees(own['yaw']))
            if self.phase=='retreat':target=WAYPOINTS[params['return_to']]
        return dict(command,mode='ADVANCE',target=target)


class ObservedOpponentPlanner(Planner):
    """Temporary navigation disk from a CURRENT legal detection only."""
    def __init__(self,cfg,covers,center):
        self.observed_center=np.asarray(center,float)
        super().__init__(cfg,covers)
    def free(self,p):
        return super().free(p) and math.dist(p,self.observed_center)>=.7-1e-10
    def segment(self,a,b):
        if not super().segment(a,b):return False
        a=np.asarray(a);b=np.asarray(b);d=b-a
        u=float(np.clip(np.dot(self.observed_center-a,d)/max(1e-12,np.dot(d,d)),0,1))
        return math.dist(a+u*d,self.observed_center)>=.7-1e-10


class SkillProbe(GoalOnlyProbe):
    def __init__(self,base,rules,decider,folder):
        super().__init__(base,rules,WAYPOINTS['GOAL'],[WAYPOINTS['GOAL']],base['start_b'],'patrol','ds_max',
                         yaw=base['guard_yaw'],patrol=base['guard_patrol'])
        self.controllers[0]=IntentController(decider,self.cfg,rules,folder)
        self.pose='upright';self.pose_target='upright';self.pose_start=0.;self.pose_until=0.;self.pose_from=.4
        self.pose_records={};self.feedback='initial';self.last_route_check=-99.;self.route_build_seconds=0.
        self.public_now=None;self.last_clock_time=0.;self.folder=folder

    def update_pose(self,t):
        if t>=self.pose_until:
            self.pose=self.pose_target;self.heights[0]=self.cfg['postures'][self.pose]['height']
            self.actor[0].vmax=self.cfg['postures'][self.pose]['speed'];self.actor[0].accel=self.cfg['postures'][self.pose]['acceleration']
        else:
            u=(t-self.pose_start)/(self.pose_until-self.pose_start)
            self.heights[0]=self.pose_from*(1-u)+self.cfg['postures'][self.pose_target]['height']*u
            self.actor[0].vmax=1.2;self.actor[0].accel=3.

    def observation(self,i,t):
        if i==0:self.last_clock_time=t;self.update_pose(t)
        p=super().observation(i,t)
        if i==0:
            p['self'].update(posture=self.pose,posture_target=self.pose_target,height=self.heights[0],
                transition_remaining=max(0.,self.pose_until-t),smoke_remaining=self.inventory,
                execution_feedback=self.feedback,collision_stop_ticks=self.collision_stops[0])
            self.public_now=copy.deepcopy(p)
            self.pose_records[t]=dict(posture=self.pose,posture_target=self.pose_target,representative_heights=self.heights.copy(),
                                     cooldown_remaining=p['self']['tag_cooldown_remaining'],smoke_remaining=self.inventory,
                                     human_disabled_remaining=max(0.,self.clock.disabled_until-t))
        return p

    def route_to(self,i,target):
        if i==1:return super().route_to(i,target)
        mover=self.actor[0];key=tuple(target);t=self.last_clock_time
        detection=self.public_now['visible'] if self.public_now else None
        obstruction=False
        if detection is not None and mover.route and t-self.last_route_check>=.5:
            q=np.asarray(detection['position']);a=mover.xy;b=mover.route[0];d=b-a
            u=float(np.clip(np.dot(q-a,d)/max(1e-12,np.dot(d,d)),0,1))
            obstruction=math.dist(a+u*d,q)<.85
        if self.move_goal[0]==key and mover.route and not obstruction:return
        started=time.perf_counter();planner=self.planner
        if detection is not None and math.dist(mover.xy,detection['position'])<6:
            planner=ObservedOpponentPlanner(self.cfg,self.cfg['arena']['covers'],detection['position'])
        goals=[target]
        if math.dist(target,self.goal)<1e-6:
            goals=[np.array(self.goal)+.88*np.array([math.cos(a),math.sin(a)]) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
            goals.sort(key=lambda q:math.dist(mover.xy,q))
        chosen=None
        for q in goals:
            if not planner.free(q):continue
            route=planner.route(mover.xy,q)
            if route is not None and all(planner.segment(a,b) for a,b in zip([mover.xy]+route,route)):
                chosen=rounded_route(mover.xy,route,planner);break
        self.route_build_seconds+=time.perf_counter()-started;self.last_route_check=t
        if chosen is None:
            self.stop(0);self.feedback='NO_ROUTE_TO_REQUESTED_TARGET';return
        mover.navigate(chosen);self.move_goal[0]=key;self.feedback='ROUTE_ACTIVE'

    def execute(self,i,command,t):
        if i==1:return super().execute(i,command,t)
        target_pose=command.get('posture',self.pose_target)
        if target_pose!=self.pose_target:
            self.pose_from=self.heights[0];self.pose_start=t;self.pose_until=t+.4;self.pose_target=target_pose
            self.events.append(dict(time=t,type='posture_requested',posture=target_pose,until=self.pose_until))
        mode=command['mode']
        if mode=='WAIT_FIXED' and self.cast is None:
            self.stop(0);return math.radians(command['look_degrees']),False,'WAIT'
        if mode=='ADVANCE':
            self.waypoints=[np.array(command['target'],float)];self.progress=0
        if mode=='THROW':
            self.waypoints=[np.array(command['destination'],float)];self.progress=0
            if not self.throw_valid(command['target']):self.feedback='SMOKE_NOT_CURRENTLY_LEGAL_CONTINUE_DESTINATION'
        look,fire,executed=super().execute(i,command,t)
        if t<self.pose_until-1e-9:fire=False
        return look,fire,executed

    def run(self,identifier):
        result=super().run(identifier)
        for s in result['states']:
            s.update(self.pose_records[s['t']])
            s['llm_decision']=sum(d['simulation_seconds']<=s['t'] for d in self.controllers[0].decisions)
        result['model_decisions']=self.controllers[0].decisions
        result['summary'].update(condition='tracking_v03',layout=self.cfg['layout_id'],model_decisions=len(result['model_decisions']),
                                 route_build_seconds=self.route_build_seconds,
                                 human_tags=sum(0 in e['actors'] for e in result['events'] if e['type']=='tag'))
        return result


def run_one(number,paid=False):
    base,rules=scenario();folder=R/f'run_{number}';folder.mkdir(exist_ok=True)
    dump(folder/'trial_config.json',dict(base=base,rules=rules,waypoints=WAYPOINTS))
    if not paid:raise ValueError('Explicit --execute-paid required')
    decider=ApiPlanner(folder,base,rules);probe=SkillProbe(base,rules,decider,folder)
    try:
        result=probe.run(f'ds_max_run_{number}')
        save_match(folder,result)
        dump(folder/'timeline.json',result['states']);dump(folder/'events.json',result['events'])
        summary=result['summary'];summary.update(usage=usage_total(decider.records),
            api_median_seconds=statistics.median(r['api_wall_seconds'] for r in decider.records),
            api_total_seconds=sum(r['api_wall_seconds'] for r in decider.records),
            response_models=sorted(set(r['response_model'] for r in decider.records)),status='complete')
        dump(folder/'result.json',summary);dump(folder/'process.json',dict(status='complete',utc=utc()))
        print('MATCH_COMPLETE',number,json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as error:
        dump(folder/'partial_timeline.json',probe.trace)
        dump(folder/'process.json',dict(status='interrupted',error_type=type(error).__name__,simulation_seconds=probe.trace[-1]['t'] if probe.trace else 0.,utc=utc()))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=int,choices=[1,2,3],required=True);p.add_argument('--execute-paid',action='store_true');a=p.parse_args()
    run_one(a.run,a.execute_paid)


if __name__=='__main__':main()
