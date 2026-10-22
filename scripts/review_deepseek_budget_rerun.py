"""Post-terminal audit for the max-effort/platform-output-limit rerun."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from llm_trial import dump
from vision_report import build

def review(root):
    read=lambda p:json.loads(p.read_text())
    result=read(root/'result.json')
    assert result['status']=='passed' and result['finish'], 'Post-terminal analysis only'
    profile=read(root/'api_profile.json');assert profile['reasoning_effort']=='max' and profile['max_tokens']==393216
    prior=root.parent/profile['comparison_run']
    shared={}
    for name in ['planner_context.txt','trial_config.json',*(f'source_snapshot/scripts/{n}.py' for n in ('semantic_skills','vision_core','trial_rules','adaptive_motion'))]:
        a=hashlib.sha256((root/name).read_bytes()).hexdigest();b=hashlib.sha256((prior/name).read_bytes()).hexdigest()
        assert a==b,name
        shared[name]=dict(identical=True,sha256=a)
    decisions=read(root/'decisions.json');states=read(root/'timeline.json');events=read(root/'events.json');usage=read(root/'api_usage.json')
    responses=[read(p) for p in sorted((root/'api_calls').glob('*_response.json'))]
    assert all(r['finish_reason']=='stop' for r in responses)
    assert len(responses)==len(decisions)==usage['calls']
    assert not list((root/'api_calls').glob('*_failure.json'))
    completions=[r['usage']['completion_tokens'] for r in responses]
    over=[dict(decision=i,time=decisions[i]['simulation_seconds'],completion_tokens=n,
        reasoning_tokens=responses[i]['usage'].get('completion_tokens_details',{}).get('reasoning_tokens'),
        elapsed_seconds=responses[i]['api_wall_seconds']) for i,n in enumerate(completions) if n>16384]
    times=[r['api_wall_seconds'] for r in responses]
    old_last=read(prior/'timeline.json')[-1]
    old_responses=[read(p) for p in sorted((prior/'api_calls').glob('*_response.json'))]
    old_decisions=read(prior/'decisions.json')
    value=dict(written_after_match=True,not_planner_input=True,
        scope='Fresh single episode; no model retries, no strategy repair, no continuation or outcome selection',
        previous=dict(run=prior.name,completed=bool(old_last['finished']),score_observed=old_last['score'],
            final_score=None,observed_seconds=old_last['t'],hp=old_last['hp'],accepted_decisions=len(old_decisions),
            api_requests=len(old_responses),max_tokens=16384,api_mean_accepted=statistics.mean(d['api']['api_wall_seconds'] for d in old_decisions)),
        current=dict(run=root.name,completed=True,final_score=result['score'],observed_seconds=result['simulation_seconds'],finish=result['finish'],
            hp=result['final_hp'],accepted_decisions=len(decisions),api_requests=len(responses),provider_max_tokens=393216,
            experiment_total_token_budget=None,decision_wall_deadline=None,launcher_wall_deadline=None,
            api_mean=statistics.mean(times),api_median=statistics.median(times),api_max=max(times),api_total=sum(times),usage=usage,
            max_observed_completion_tokens=max(completions),requests_exceeding_old_cap=over),
        shared_files=shared,
        caveats=['Provider limit remains finite; this is not infinite output.',
            'Old 10.56 is an interrupted score, not a completed outcome. No superiority claim from a single rerun.',
            'Output and time limits both changed; service/model sampling can also vary despite identical prompt and seed.',
            'No API retries or human-modified actions; no hidden reasoning text saved.'])
    dump(root/'budget_rerun_comparison.json',value)
    findings=[f"本次放宽输出额度重测自然终局：{result['score']:g}分、{result['simulation_seconds']:g}秒，HP {result['final_hp']}，终止原因 {result['finish']}。不是在旧局28秒处继续执行。",
        f"{len(responses)}次API请求全部finish_reason=stop，无重试、无失败；其中{len(over)}次completion超过旧16384限制，最高{max(completions):,} tokens。已解决本轮的低额度截断问题，不代表云接口永不失败。",
        '本轮不是无限输出：没有额外实验总token预算，单次使用官方最大393216（384K）；未传参数反而默认128K。客户端决策和进程总时长截止已取消，保留600秒网络无数据超时。',
        '与旧max的背景提示词、规则、地图、种子、运动与视觉代码哈希一致；详见budget_rerun_comparison.json。旧max在28秒中断，10.56不是终局成绩；不能按这个分数直接排名。']
    hits=[(e['time'],e['attacker'],e['target']) for e in events if e['type']=='hit']
    rejected=[d['simulation_seconds'] for d in decisions if d.get('compiled_skill',{}).get('state')=='UNSAFE_ROUTE']
    findings.append(f'赛后事件核验：命中(time,attacker,target)={hits}；UNSAFE_ROUTE出现时刻={rejected}。角色0为四足A，1为人形B。')
    findings.append('占点轨迹为P1→P2→P1→P2：前三个实例各4分，第四个实例在39.26秒圈内受击结束、得到1.22分。20次意图中17次CAPTURE_POINT、3次MOVE_TO；没有命中人形。')
    if (root/'annotation_audit.json').exists():
        annotations=read(root/'annotation_audit.json')['annotations']
        value['posthoc_annotations']=annotations;dump(root/'budget_rerun_comparison.json',value)
        findings.append('图像复核仍暴露定位问题：9.5秒人形双脚可见，提交[389,315]却落在前景地面，身体中心投影偏差约3.417米；28秒近景[160,340]落在可见鞋部，偏差约0.091米。二者是赛后选定标注核验，不代表全局感知准确率，不能用本轮更高得分掩盖第一处错误。')
    dump(root/'review_addendum.json',dict(written_after_match=True,not_planner_input=True,findings=findings,
        next_steps=['本轮作为独立复测留档；原科研报告不自动覆盖。后续可在固定输出/超时协议下开展多种子比较。'],
        limits=['视觉组decision_wait_seconds不含后续技能编译；UTC就绪到接受与API起止时间均独立保留。不能把它们当成纯思考耗时。',
            '单个新样本不能证明输出预算放宽导致得分变化；没有更改原始决策、未来试跑或失败后重选结果。']))
    build(root)
    print(json.dumps(value['current'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);review(p.parse_args().run)
