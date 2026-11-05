"""Read-only checks of saved games, legal observations, API records and provenance."""
import collections
import datetime
import gzip
import json
import math
from pathlib import Path
import sys

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source_snapshot/scripts'))
from fixed_goal_probe import boxes,clear_ray,smoke_blocks,sha
from tactical_logic import wrap
from ds_fixed_goal import digest,scenario
from qwen_fixed_goal import CASES
from sequence_skills import validate_plan_output

read=lambda p:json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def visible(s,i,r,walls):
    j=1-i;p='robot' if i==0 else 'human';xy=s['xy'];h=s['representative_heights']
    angle=wrap(math.atan2(xy[j][1]-xy[i][1],xy[j][0]-xy[i][0])-s['yaw'][i])
    limit=r[p+'_sensor_range_m']
    return ((limit is None or math.dist(xy[i],xy[j])<=limit)
        and abs(angle)<=math.radians(r[p+'_half_fov_degrees'])+1e-9
        and clear_ray((*xy[i],h[i]),(*xy[j],h[j]),walls)
        and not smoke_blocks((*xy[i],h[i]),(*xy[j],h[j]),s['smoke'],s['t'],r['smoke_penetration_m']))

def run(number):
    scenario(CASES[number-1]['layout'])  # Load this layout's frozen waypoint vocabulary.
    folder=R/f'run_{number}';result=read(folder/'result.json')
    with gzip.open(folder/(result['id']+'.json.gz'),'rt') as f:data=json.load(f)
    states=data['states'];events=data['events'];rules=data['metadata']['rules'];dt=rules['dt']
    base=read(folder/'trial_config.json')['base'];walls=boxes(base['arena']['covers']);counts=collections.Counter()
    assert states==read(folder/'timeline.json') and events==read(folder/'events.json')
    assert states[0]['t']==0 and states[-1]['t']==result['seconds']
    assert all(abs(b['t']-a['t']-dt)<1e-8 for a,b in zip(states,states[1:]))
    def at(t):return states[round(t/dt)]
    def detection(packet,s,actor=0):
        if packet is None:assert not s['visible'][actor]
        else:
            assert s['visible'][actor]
            assert packet['position']==s['xy'][1-actor] and packet['yaw']==s['yaw'][1-actor]
        counts['public_detections']+=1
    def memory(packet,t,actor=0):
        if packet is not None:
            old=at(packet['last_seen'])
            assert old['visible'][actor] and old['xy'][1-actor]==packet['position']
            assert 0<=t-packet['last_seen']<=rules['memory_seconds'][actor]+1e-8
            counts['memory_checks']+=1
    for s in states:
        for i in (0,1):
            assert visible(s,i,rules,walls)==s['visible'][i],(number,s['t'],i)
            counts['visibility_checks']+=1
            angle=wrap(math.atan2(s['xy'][1-i][1]-s['xy'][i][1],s['xy'][1-i][0]-s['xy'][i][0])-s['yaw'][i])
            assert abs(math.degrees(angle)-s['aim_error_degrees'][i])<1e-8
            expected=(s['visible'][i] and s['requested_fire'][i] and 1e-8<math.dist(*s['xy'])<=rules['tag_range_m'] and abs(math.degrees(angle))<=min(rules['aim_half_angle_degrees'],rules['aim_tracking_error_degrees'])+1e-8 and abs(s['speed'][i])<=rules['aim_max_speed_mps']+1e-8)
            assert s['lock_geometry_valid'][i]==expected,(number,s['t'],i)
            allowed=expected and (s['cooldown_remaining']<=1e-8 if i==0 else s['human_disabled_remaining']<=1e-8)
            assert s['lock_valid'][i]==allowed,(number,s['t'],i,'permission')
            assert abs(s['omega'][i])<=base['yaw_rate'][i]+1e-7
            counts['tracking_checks']+=1
    for d in data['decisions']:
        p=d['public'];s=at(d['time']);i=d['actor']
        detection(p['visible'],s,i);memory(p['memory'],d['time'],i)
        assert p['self']['position']==s['xy'][i]
    actions=[];latencies=[];calls=[];reasoning_characters=[];normalizations=[]
    for d in read(folder/'decisions.json'):
        i=d['observation_id'];p=d['public_observation'];t=d['simulation_seconds'];s=at(t)
        req=read(folder/'api_calls'/f'{i:03d}_request.json');resp=read(folder/'api_calls'/f'{i:03d}_response.json');accepted=read(folder/'accepted'/f'{i:03d}.json')
        assert req['observation']==p and t==req['observation']['simulation_seconds']
        assert req['recent_decisions']==[{k:a[k] for k in ('observation_id','skill','params')} for a in actions[-4:]]
        assert req['profile']['thinking']=={'type':'enabled'} and req['profile']['reasoning_effort']=='low' and req['profile']['model']=='deepseek-flash'
        assert req['profile']['max_tokens']==65536 and 'response_format' not in req['profile'] and 'thinking_budget' not in req['profile']
        assert req['system']['content']+'\n'==(R/'prompts'/f"{CASES[number-1]['layout']}.txt").read_text()
        raw=json.loads(resp['content'].strip().removeprefix('```json\n').removesuffix('\n```'));normalization=accepted['normalization']
        if normalization:
            assert normalization['kind']=='ordered_assessment_array_to_named_object'
            from ds_fixed_goal import FIELDS
            raw['assessment']=dict(zip(FIELDS,raw['assessment']))
        action=validate_plan_output(json.dumps(raw),i)
        assert accepted['normalization']==normalization
        if normalization:normalizations.append(dict(step=i,record=normalization))
        reasoning_characters.append(resp['reasoning_characters_received'])
        if CASES[number-1]['mode']=='off':assert resp['reasoning_characters_received']==0
        assert action==accepted['action']=={k:d[k] for k in ('observation_id','skill','params','assessment')}
        assert accepted['request_sha256']==digest(req)
        assert resp['finish_reason']=='stop' and not resp['tool_calls'] and resp['reasoning_text_saved'] is False
        elapsed=(datetime.datetime.fromisoformat(resp['api_finished_utc'])-datetime.datetime.fromisoformat(resp['api_started_utc'])).total_seconds()
        assert abs(elapsed-resp['api_wall_seconds'])<.1
        detection(p['current_detection'],s);memory(p['last_legal_memory'],t)
        assert p['self_state']['position']==s['xy'][0]
        assert p['self_state']['tag_cooldown_remaining']==s['cooldown_remaining']
        samples=read(folder/'legal_samples.json')[i]['samples']
        assert len(samples)==p['full_legal_sample_count']
        compressed=p['sensor_samples_since_previous_decision']
        assert all(v in samples for v in compressed)
        for k in range(1,len(samples)):
            if (samples[k][1] is None)!=(samples[k-1][1] is None):assert samples[k] in compressed and samples[k-1] in compressed
        for st,v in samples:
            ss=at(st)
            if v is None:assert not ss['visible'][0]
            else:assert ss['visible'][0] and v==[round(float(x),3) for x in ss['xy'][1]]+[round(ss['yaw'][1],4)]
            counts['sensor_samples_checked']+=1
        actions.append(action);latencies.append(resp['api_wall_seconds'])
        calls.append(dict(index=i,time=t,skill=d['skill'],seconds=resp['api_wall_seconds'],usage=resp['usage'],started=resp['api_started_utc'],finished=resp['api_finished_utc']))
        counts['model_decisions']+=1
    tag_times=[]
    for e in events:
        if e['type']=='tag':
            end=round(e['time']/dt);begin=end-round(rules['tag_seconds']/dt);assert begin>=0
            if 0 in e['actors']:tag_times.append(e['time'])
            for i in e['actors']:
                for k in range(begin,end+1):
                    s=states[k];a,b=s['xy'][i],s['xy'][1-i]
                    assert s['visible'][i] and s['modes'][i]=='AIM'
                    assert math.dist(a,b)<=rules['tag_range_m']+1e-8
                    assert abs(wrap(math.atan2(b[1]-a[1],b[0]-a[0])-s['yaw'][i]))<=math.radians(rules['aim_half_angle_degrees'])+1e-8
                    assert abs(s['speed'][i])<=rules['aim_max_speed_mps']+1e-8
                    assert abs(s['aim_error_degrees'][i])<=rules['aim_tracking_error_degrees']+1e-7
                    if i==0:
                        assert all(not (p['time']<s['t']<p['until']-1e-8) for p in events if p['type']=='posture_requested')
                counts['tag_windows']+=1
        if e['type']=='smoke_released':
            s=at(e['time'])
            assert math.dist(s['xy'][0],e['center'])<=rules['smoke_throw_range_m']+1e-8
            assert clear_ray((*s['xy'][0],1.),(*e['center'],1.),walls)
            assert abs(e['expires_at']-e['active_at']-rules['smoke_duration_seconds'])<1e-8
            counts['smoke_releases']+=1
    assert all(b-a>=rules['tag_cooldown_seconds']-1e-8 for a,b in zip(tag_times,tag_times[1:]))
    for a,b in zip(states,states[1:]):
        if b['modes'][1]=='DISABLED':assert abs(wrap(b['yaw'][1]-a['yaw'][1]))<1e-8
    assert result['status']=='complete' and events[-1]['type']=='finish'
    assert result['winner']!='attacker' or result['reason']=='goal'
    if result['reason']=='goal':assert math.dist(states[-1]['xy'][0],data['metadata']['goal'])<=1.+1e-8
    assert result['minimum_wall_clearance_m']>=base['planner_clearance']-1e-8
    assert result['minimum_separation_m']>=rules['min_actor_separation_m']-1e-8
    assert sum(c['usage']['total_tokens'] for c in calls)==result['usage']['total_tokens']
    failures=list(folder.glob('api_calls/*_failure.json'))
    if failures:
        assert result['completion_mode']=='network_resumed' and result['failed_attempt_usage_unknown']
    else:assert result['completion_mode']=='unassisted'
    # Diagnose failed locks from saved physical states. Reasons are non-exclusive.
    invalid=collections.Counter();resets=[];turns=[]
    for k,s in enumerate(states):
        turn=0 if k==0 else abs(wrap(s['yaw'][1]-states[k-1]['yaw'][1]))/dt
        turns.append(turn);why=[];a,b=s['xy'][1],s['xy'][0]
        if not s['visible'][1]:why.append('not_visible')
        if math.dist(a,b)>rules['tag_range_m']:why.append('outside_range')
        if abs(wrap(math.atan2(b[1]-a[1],b[0]-a[0])-s['yaw'][1]))>math.radians(30)+1e-8:why.append('outside_aim_cone')
        if abs(s['speed'][1])>.15+1e-8:why.append('shooter_moving')
        if abs(s['aim_error_degrees'][1])>rules['aim_tracking_error_degrees']+1e-8:why.append('tracking_error')
        if s['modes'][1]!='AIM':why.append('not_aiming')
        if any(t<=s['t']<t+1.-1e-8 for t in tag_times):why.append('disabled')
        if any(e['type']=='tag' and 1 in e['actors'] and abs(e['time']-s['t'])<1e-8 for e in events):why.append('tag_fired')
        if s['modes'][1]=='AIM':invalid.update(why)
        if k and states[k-1]['aim'][1]>.02 and s['aim'][1]<1e-9:
            resets.append(dict(time=s['t'],previous_lock=states[k-1]['aim'][1],reasons=why))
    smoke_only=[]
    for i in (0,1):
        changed=[s['t'] for s in states[:-1] if not s['visible'][i] and visible(dict(s,smoke=[]),i,rules,walls)]
        smoke_only.append(dict(actor=i,seconds=round(len(changed)*dt,3),first=None if not changed else min(changed),last=None if not changed else max(changed)))
    analysis=dict(run=number,checks=dict(counts),calls=calls,tag_times=tag_times,
        response_reasoning_characters=reasoning_characters,normalizations=normalizations,
        human_max_lock=max([s['aim'][1] for s in states]+[rules['tag_seconds'] if any(e['type']=='tag' and 1 in e['actors'] for e in events) else 0]),human_lock_resets=resets,
        human_aim_invalid_seconds={k:round(v*dt,3) for k,v in invalid.items()},
        human_turn_radps=turns,
        human_fast_tracking_valid_seconds=round(sum(s['lock_valid'][1] and abs(s['omega'][1])>.8+1e-7 for s in states[:-1])*dt,3),
        first_valid_lock=[next((s['t'] for s in states if s['lock_valid'][i]),None) for i in (0,1)],
        human_visible_seconds=round(sum(s['visible'][1] for s in states[:-1])*dt,2),
        robot_visible_seconds=round(sum(s['visible'][0] for s in states[:-1])*dt,2),
        low_posture_seconds=round(sum(s['posture']=='low' for s in states[:-1])*dt,2),
        attack_moving_seconds=round(sum(abs(s['speed'][0])>.15 for s in states[:-1])*dt,2),
        skill_counts=dict(collections.Counter(d['skill'] for d in actions)),
        smoke_only_visibility_changes=smoke_only,
        smoke_diagnostic_note='Remove smoke from sight tests on the same saved positions and yaws. This is not a policy re-run or a causal outcome ablation.',
        note='Stored own cooldown/disabled fields are sampled before adjudication at that tick; aim counters are after adjudication. Event time is authoritative.')
    bursts=[]
    for burst in read(folder/'peek_bursts.json'):
        ss=[v for v in states if burst['start']<=v['t']<=burst['end']]
        longest=current=0.
        for v in ss:
            current=current+dt if v['visible'][1] and math.dist(*v['xy'])<=rules['tag_range_m'] else 0.
            longest=max(longest,current)
        bursts.append(dict(**burst,actual_human_visible_run_seconds=round(longest,3),max_human_lock=max(v['aim'][1] for v in ss)))
    violations=[]
    for prev,s in zip(states,states[1:]):
        if s['modes'][0]!='PEEK_LOCAL' or abs(s['speed'][0])<1e-6:continue
        dx=[(s['xy'][0][j]-prev['xy'][0][j])/dt for j in (0,1)]
        signed=dx[0]*math.cos(s['yaw'][0])+dx[1]*math.sin(s['yaw'][0])
        if signed*s['speed'][0]<-1e-8:violations.append(dict(time=s['t'],actual_longitudinal_velocity=signed,recorded_speed=s['speed'][0]))
    decisions=read(folder/'decisions.json')
    analysis.update(peek_checks=bursts,peek_reverse_sign_violations=violations,
        model_calls_behind_known_cover=sum(d['public_observation']['model_call_behind_known_cover'] for d in decisions),
        decision_intervals=[round(b['simulation_seconds']-a['simulation_seconds'],2) for a,b in zip(decisions,decisions[1:])],
        skill_event_counts=dict(collections.Counter(e['type'] for e in read(folder/'skill_events.json'))),
        limits='Cover and exposure estimates use legal current/old observations, not hidden truth. Motion sign violations are reported, not silently accepted as physically valid.')
    dump(folder/'analysis.json',analysis)
    return dict(run=number,status='passed_with_motion_exception' if violations else 'passed',checks=dict(counts),api_calls=len(actions),end=result['seconds'],motion_sign_violations=violations,metadata_assisted=False)

def main():
    manifest=read(R/'source_manifest.json')
    for name,value in manifest['files'].items():assert sha(R/name)==value,name
    results=[run(i) for i in (1,2,3)]
    assert all(not r['motion_sign_violations'] for r in results)
    checks=read(R/'layout_checks.json')
    for item in checks:
        assert not item['initial_los'] and item['initial_separation_m']>=10 and item['initial_goal_distance_m']>=10
        assert len(item['departures'])==2 and all(not d['first_two_seconds_human_tag'] for d in item['departures'])
    final=dict(status='passed',games=results,source_files_checked=len(manifest['files']),
        model='deepseek-flash',reasoning_effort='low',scene_checks=3,braking_patch='v0.4.1',
        limitations=['One sample per layout, not a stable win-rate estimate.',
        'Audit verifies records and visibility; documented executor limitations remain.'])
    dump(R/'audit.json',final);print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
