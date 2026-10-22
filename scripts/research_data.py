"""Read-only analysis of archived trials. New derivatives live in a report folder."""
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
import numpy as np

from arena_paths import workspace_root
BASE=Path(os.environ.get('ARENA_RESEARCH_RUNS',workspace_root()/'experiments/runs/quadruped-arena'))
OUT=Path(os.environ.get('ARENA_RESEARCH_OUTPUT',workspace_root()/'experiments/reports/quadruped-arena/research-comparison'))
SPECS=[
 ('O-A','全知 · 助手','Assistant / oracle','20261021_LLM首轮_2秒闭环_全知观测','oracle','#007F73'),
 ('O-Q','全知 · Qwen','Qwen / oracle','20261021-162029_Qwen3.6-27B_2秒闭环','oracle','#3979B5'),
 ('V-DH','视觉 · DS max（放宽额度）','DS max','20261022-095614-313039','vision','#A55E18'),
 ('V-DL','视觉 · DS low','DS low','20261021-203042-903738','vision','#DA9D32'),
 ('V-QH','视觉 · Qwen 思考开启','Qwen on','20261021-205454-693959','vision','#3979B5'),
 ('V-QL','视觉 · Qwen 思考关闭','Qwen off','20261021-210646-052984','vision','#78AADD'),
 ('V-A','视觉 · 助手','Assistant','20261021-211006-627504','vision','#007F73'),
]

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stats(values):
    if not values:return dict(n=0,mean=None,median=None,p95=None,min=None,max=None)
    a=np.array(values);return dict(n=len(values),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.sort(a)[math.ceil(.95*len(a))-1]),min=float(a.min()),max=float(a.max()))

def build():
    from semantic_skills import ground_point
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'data').mkdir(exist_ok=True)
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    sources=[];asset_manifest=[];runs=[];metrics=[];step_rows=[];annotations=[];episodes=[]
    def record(p):
        if p.exists():sources.append(dict(path=str(p),sha256=sha(p)))
    def copy_asset(src,relative,kind):
        dst=OUT/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists() or sha(src)!=sha(dst):shutil.copy2(src,dst)
        asset_manifest.append(dict(file=relative,source=str(src),sha256=sha(src),kind=kind))
        return relative
    def overhead(root,ident,times):
        result=[]
        old=read(root/'screenshots/manifest.json')
        if old:
            for frame in old['screenshots']:
                f=copy_asset(root/frame['file'],f'assets/{ident}/overhead_{frame["simulation_time"]:05.2f}.png','unchanged archived posthoc screenshot')
                result.append(dict(time=frame['simulation_time'],file=f,provenance='归档 Isaac 赛后截图；不是规划器输入'))
            return result
        native=read(OUT/'native_keyframes/manifest.json',[])
        entry=next((e for e in native if e['source']==str(root)),None)
        if entry:
            assert sha(root/'timeline.json')==entry['source_timeline_sha256']
            for frame in entry['frames']:
                f=copy_asset(Path(frame['file']),f'assets/{ident}/overhead_{frame["time"]:05.2f}.png','new native rendering of saved exact state')
                result.append(dict(time=frame['time'],file=f,provenance='本报告补渲染的 Isaac 俯视图；精确归档状态，不续跑，不是输入'))
            return result
        replay=read(root/'third_person_replay.json')
        if not replay:return result
        render=(root/replay['mp4']).parent.resolve()
        index=read(render/'frame_index.json');rr=read(render/'result.json')
        assert rr['replay']['source_sha256']['timeline.json']==sha(root/'timeline.json')
        wanted={}
        for t in times:
            match=[v for v in index['regular_frames'] if abs(v['simulation_time']-t)<1e-7]
            assert len(match)==1,(ident,t)
            wanted[match[0]['video_frame_index']]=t
        import imageio.v2 as imageio
        reader=imageio.get_reader(str(render/'demo.mp4'))
        try:
            for idx,pixels in enumerate(reader):
                if idx in wanted:
                    t=wanted[idx];f=f'assets/{ident}/overhead_{t:05.2f}.png'
                    dest=OUT/f;dest.parent.mkdir(exist_ok=True);imageio.imwrite(dest,pixels)
                    result.append(dict(time=t,file=f,video_frame_index=idx,provenance='Isaac MP4 精确帧解码，已计入片头/坍塌插入偏移；不是输入'))
                    asset_manifest.append(dict(file=f,source=str(render/'demo.mp4'),video_frame_index=idx,sha256=sha(dest),kind='exact video frame'))
                if idx>=max(wanted):break
        finally:reader.close()
        return result
    for ident,label,short,name,cohort,color in SPECS:
        root=BASE/name;timeline=read(root/'timeline.json');decisions=read(root/'decisions.json');events=read(root/'events.json')
        cfg=read(root/'trial_config.json');summary=read(root/'test_summary.json',read(root/'summary.json',{}))
        profile=read(root/'api_profile.json',{});protocol=read(root/'protocol.json',{})
        for f in ['timeline.json','decisions.json','events.json','trial_config.json','summary.json','test_summary.json','protocol.json','api_profile.json','planner_context.txt','review_addendum.json','post_run_review.json','visual_spotcheck.json','api_usage.json','annotation_audit.json','budget_rerun_comparison.json','third_person_replay.json']:
            record(root/f)
        for f in ['trial_config.json','protocol.json','api_profile.json','planner_context.txt','review_addendum.json','post_run_review.json','visual_spotcheck.json','annotation_audit.json','budget_rerun_comparison.json','api_usage.json']:
            if (root/f).exists():copy_asset(root/f,f'archive/{ident}/{f}','original configuration, prompt or review; no credentials')
        for p in (root/'source_snapshot/scripts').glob('*.py'):record(p)
        for p in (root/'api_calls').glob('*_response.json'):record(p)
        final=timeline[-1];end=final['t'];complete=bool(final['finished'])
        ts=np.array([s['t'] for s in timeline]);dt=np.diff(ts)
        xy=np.array([s['xy'][0] for s in timeline]);path=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
        visible=float(sum(dt[k]*bool(s.get('human_visible')) for k,s in enumerate(timeline[1:])))
        aiming=float(sum(dt[k]*(s['aim'][1]>0) for k,s in enumerate(timeline[1:])))
        stationary_seen=float(sum(dt[k]*(s['speed'][0]<.05 and s.get('human_visible',False)) for k,s in enumerate(timeline[1:])))
        scoring=float(sum(dt[k]*any(s['scoring']) for k,s in enumerate(timeline[1:])))
        waits=[d.get('decision_wait_seconds',d.get('decision_wall_seconds')) for d in decisions]
        api=[d.get('api',{}).get('api_wall_seconds',d.get('api_wall_seconds')) for d in decisions];api=[v for v in api if v is not None]
        compile_times=[d['skill_compile_seconds'] for d in decisions if 'skill_compile_seconds' in d]
        usage=read(root/'api_usage.json',summary.get('all_requests_usage'))
        if not usage and any(d.get('token_usage') for d in decisions):
            usage={k:sum(d['token_usage'].get(k,0) for d in decisions) for k in ['prompt_tokens','completion_tokens','reasoning_tokens','total_tokens','calls']}
        failed=read(root/f'api_calls/{len(decisions):03d}_response.json') if not complete else None
        fail_api=failed.get('api_wall_seconds') if failed else None
        accepted_api_stats=stats(api);all_api_stats=stats(api+([fail_api] if fail_api else []))
        selected=Counter(d.get('skill',d.get('action',{}).get('skill')) for d in decisions)
        rejected=sum(d.get('compiled_skill',{}).get('state')=='UNSAFE_ROUTE' for d in decisions)
        hit_a=[e for e in events if e['type']=='hit' and e['target']==0]
        hit_b=[e for e in events if e['type']=='hit' and e['target']==1]
        triggered=[e for e in events if e['type']=='target_triggered']
        claim_frames=set();all_claims=0
        for d in decisions:
            for e in d.get('assessment',{}).get('visual_evidence',[]):
                if e['opponent_visible'] is True:claim_frames.add(e['time']);all_claims+=1
        metric=dict(id=ident,label=label,cohort=cohort,completed=complete,score_observed=final['score'],
            final_score=final['score'] if complete else None,end_seconds=end,finish=final['finished'] or 'api_interrupted',
            hp_a=final['hp'][0],hp_b=final['hp'][1],decisions=len(decisions),calls=usage.get('calls') if usage else None,
            path_m=path,human_sees_a_seconds=visible,human_aim_seconds=aiming,stationary_seen_seconds=stationary_seen,
            scoring_seconds=scoring,score_per_observed_second=final['score']/end,hits_received=len(hit_a),hits_inflicted=len(hit_b),
            target_triggers=len(triggered),scoring_episodes=sum(e['type']=='scoring_started' for e in events),
            wait_mean=stats(waits)['mean'],wait_median=stats(waits)['median'],wait_p95=stats(waits)['p95'],
            api_mean=accepted_api_stats['mean'],api_all_mean=all_api_stats['mean'],api_median=accepted_api_stats['median'],api_p95=accepted_api_stats['p95'],
            compile_mean=stats(compile_times)['mean'],total_tokens=usage.get('total_tokens') if usage else None,
            reasoning_tokens=usage.get('reasoning_tokens') if usage else None,
            rejected_commands=rejected,unique_positive_frame_claims=len(claim_frames),positive_claims=all_claims,
            annotations=sum('sighting' in d.get('params',{}) for d in decisions),
            accepted_decisions_within_2s=sum(v<=2 for v in waits),
            score_at_16_8=next((s['score'] for s in timeline if abs(s['t']-16.8)<1e-7),None))
        metrics.append(metric)
        local_annotations=[];steps=[]
        over=overhead(root,ident,[d.get('simulation_seconds',d.get('simulation_start')) for d in decisions]+[end])
        for i,d in enumerate(decisions):
            o=read(root/f'observations/{i:03d}.json');rawpath=root/(f'actions/{i:03d}.json' if cohort=='vision' else f'requests/{i:03d}.json')
            raw=read(rawpath);record(rawpath);record(root/f'observations/{i:03d}.json')
            time=d.get('simulation_seconds',d.get('simulation_start'))
            row=dict(run=ident,decision=i,time=time,end=min(time+2,end),skill=d.get('skill',d.get('action',{}).get('skill')),
                wait_seconds=waits[i],api_seconds=d.get('api',{}).get('api_wall_seconds',d.get('api_wall_seconds')),
                compile_seconds=d.get('skill_compile_seconds'),tokens=d.get('tokens'),
                reasoning_tokens=d.get('api',{}).get('reasoning_tokens',d.get('token_usage',{}).get('reasoning_tokens')),
                first_output_seconds=d.get('api',{}).get('time_to_first_output_seconds'),
                first_content_seconds=d.get('api',{}).get('time_to_first_visible_content_seconds'),
                compiled_state=d.get('compiled_skill',{}).get('state'),risk=d.get('params',{}).get('risk','safe') if cohort=='vision' else None)
            step_rows.append(row)
            frames=[]
            if cohort=='vision':
                for frame in o['frames']:
                    f=copy_asset(root/frame['file'],f'assets/{ident}/{Path(frame["file"]).name}','original online RGB; unchanged bytes')
                    frames.append(dict(**frame,report_file=f))
            elif (root/f'observations/{i:03d}.png').exists():
                f=copy_asset(root/f'observations/{i:03d}.png',f'assets/{ident}/input_map_{i:03d}.png','original navigation schematic, not Isaac camera')
                frames=[dict(simulation_seconds=time,report_file=f)]
            sight=d.get('params',{}).get('sighting')
            if sight:
                frame=next(f for f in o['frames'] if f['simulation_seconds']==sight['time'])
                projected=ground_point(frame,sight['feet']);s=next(s for s in timeline if abs(s['t']-sight['time'])<1e-7)
                anno=dict(run=ident,decision=i,time=sight['time'],pixel_u=sight['feet'][0],pixel_v=sight['feet'][1],
                    error_m=math.dist(projected,s['xy'][1]),projected_x=projected[0],projected_y=projected[1],
                    truth_x=s['xy'][1][0],truth_y=s['xy'][1][1],file=next(f['report_file'] for f in frames if f['simulation_seconds']==sight['time']))
                annotations.append(anno);local_annotations.append(anno)
            after=next(s for s in timeline if abs(s['t']-row['end'])<1e-7)
            steps.append(dict(**row,raw=raw,frames=frames,observation=o,
                overhead=min(over,key=lambda x:abs(x['time']-time)) if over else None,
                outcome=dict(time=after['t'],score=after['score'],hp_a=after['hp'][0],finished=after['finished']),
                compiled_action=d.get('compiled_action'),compiled_skill=d.get('compiled_skill'),compiled_route=d.get('compiled_route'),
                api=d.get('api'),accepted_utc=d.get('decision_accepted_utc',d.get('accepted_utc')),
                observation_ready_utc=d.get('observation_ready_utc')))
        for e in triggered:
            invalid=next((v for v in events if v['type']=='target_invalidated' and v.get('generation')==e['generation']),None)
            astate=next(s for s in timeline if abs(s['t']-e['time'])<1e-7)
            gain=invalid['instance_score'] if invalid else final['score']-astate['score']
            episodes.append(dict(run=ident,generation=e['generation'],point=e['point']+1,start=e['time'],
                end=invalid['time'] if invalid else end,points=gain,reason=invalid['reason'] if invalid else 'observed_cutoff',closed=bool(invalid)))
        media={}
        for key,file in [('onboard','onboard.gif'),('third','demo.gif'),('video','demo.mp4')]:
            if (root/file).exists():media[key]=copy_asset(root/file,f'assets/{ident}/{file}','archived replay')
        replay=read(root/'third_person_replay.json')
        if replay:
            for key,src in [('third','gif'),('video','mp4')]:
                media[key]=copy_asset((root/replay[src]).resolve(),f'assets/{ident}/third_person.{"gif" if key=="third" else "mp4"}','archived native posthoc replay')
        review=read(root/'review_addendum.json',{})
        oldreview=read(root/'post_run_review.json',{})
        findings=review.get('findings',oldreview.get('findings',oldreview.get('summary',[])))
        runs.append(dict(id=ident,label=label,short=short,color=color,cohort=cohort,root=str(root),
            source_report=os.path.relpath(root/'report.html',OUT),metric=metric,steps=steps,overhead=over,annotations=local_annotations,
            events=events,timeline=timeline,config=cfg,protocol=protocol,profile=profile,usage=usage,
            accepted_api_stats=accepted_api_stats,all_api_stats=all_api_stats,wait_stats=stats(waits),
            selected_skills=dict(selected),findings=findings,media=media,failed_response=failed,
            memory_states=[read(root/f'observations/{i:03d}.json').get('skills',{}).get('visual_memory',{}) for i in range(len(decisions))] if cohort=='vision' else []))
    baseline=BASE/SPECS[0][3]/'baseline'
    bc=read(baseline/'comparison.json');bt=read(baseline/'timeline.json')
    for name in ['comparison.json','timeline.json','events.json','config.json']:record(baseline/name)
    history=[]
    for ident,label,name,note in [
        ('Q0','Qwen 全知接入失败','20261021-161447_Qwen3.6-27B_2秒闭环','0次动作，工具查询未收束；后续补背景与每步最多3次调用，不纳入主结果排名'),
        ('V0','助手视觉七字段首测','20261021-173528-369598','7.82分，30.62秒；不同技能接口，仅作演进记录'),
        ('S0','语义技能首测','20261021-190339-279787','4分，17.64秒；修复前技能，不与当前五轮混合'),
        ('DL0','DS low 无解释','20261021-193327-596027','8分，27.96秒；无assessment，4096预算/120秒，不是仅解释开关的消融'),
        ('DH0','DS max 首测断流','20261021-195343-236794','18秒系统中断，分数5.76；失败请求usage未知'),
        ('QF','Qwen 视觉围栏失败','20261021-204904-009504','2秒格式适配中断；额外2次请求，16,058 tokens，未隐藏失败'),
        ('DH16','DS max · 16K 额度中断（原主对照）','20261021-200759-757048','原 V-DH，现移入历史：28秒系统中断，10.56为非终局分数；14次有效决策/15次请求，共181,049 tokens。以独立完整重测替换主分析，不合并成绩或费用')]:
        root=BASE/name;s=read(root/'test_summary.json',read(root/'api_run_status.json',read(root/'result.json',{})))
        if ident=='Q0':
            responses=[read(p) for p in sorted((root/'api_calls').glob('*_response.json'))]
            s['known_total_tokens']=sum((x.get('usage') or {}).get('total_tokens',0) for x in responses)
        for f in ['test_summary.json','api_run_status.json','result.json','api_usage.json','api_profile.json','interruption.json']:
            record(root/f)
        historical=dict(id=ident,label=label,path=str(root),note=note,summary=s,
            source_report=os.path.relpath(root/'report.html',OUT) if (root/'report.html').exists() else None)
        if ident=='DH16':
            historical['failed_response']=read(root/'api_calls/014_response.json')
            historical['usage']=read(root/'api_usage.json',s.get('all_requests_usage'))
            historical['profile']=read(root/'api_profile.json')
            historical['endpoint']=read(root/'timeline.json')[-1]
            for f in ['timeline.json','decisions.json','api_calls/014_response.json']:record(root/f)
            for f in ['test_summary.json','api_usage.json','api_profile.json','api_calls/014_response.json']:
                if (root/f).exists():copy_asset(root/f,f'archive/{ident}/{f}','historical interrupted max; excluded from primary cohort')
        history.append(historical)
    names=['trial_config.json','planner_context.txt','source_snapshot/scripts/semantic_skills.py','source_snapshot/scripts/vision_core.py','source_snapshot/scripts/adaptive_motion.py']
    audit=[]
    for name in names:
        vals={r['id']:sha(Path(r['root'])/name) if (Path(r['root'])/name).exists() else None for r in runs}
        visual=[vals[r['id']] for r in runs if r['cohort']=='vision']
        audit.append(dict(file=name,sha256=vals,visual_identical=bool(all(visual)) and len(set(visual))==1))
    revision=dict(date='2026-10-22',operation='in-place replacement requested by user',primary_id='V-DH',
        previous_run='20261021-200759-757048',replacement_run='20261022-095614-313039',previous_history_id='DH16',
        independent_rerun=True,not_continuation=True,primary_completed=sum(r['metric']['completed'] for r in runs),
        accepted_decisions=len(step_rows),annotations=len(annotations),history_records=len(history),
        budget_caveat='New max uses max_tokens=393216, no total token budget or overall wall-clock cutoff; low retains 16384/180 s. Not an effort-only ablation.',
        original_runs_modified=False,new_model_calls=False,new_matches=False,new_rendering=False)
    dump(OUT/'data/revision.json',revision)
    manifest=dict(created='2026-10-22',revision=revision,analysis_only=True,no_model_calls=True,no_new_matches=True,
        sources=list({v['path']:v for v in sources}.values()),assets=list({v['file']:v for v in asset_manifest}.values()),shared_audit=audit)
    csvout(OUT/'data/metrics.csv',metrics);csvout(OUT/'data/decisions.csv',step_rows)
    csvout(OUT/'data/annotation_errors.csv',annotations);csvout(OUT/'data/target_episodes.csv',episodes)
    dump(OUT/'data/source_manifest.json',manifest)
    data=dict(runs=runs,baseline=dict(summary=bc,timeline=bt,label='规则/A*参考',id='BT',color='#777777'),history=history,
        metrics=metrics,annotations=annotations,episodes=episodes,audit=audit,revision=revision)
    dump(OUT/'data/analysis.json',data)
    return data

if __name__=='__main__':
    data=build();print(json.dumps(data['metrics'],ensure_ascii=False,indent=2))
