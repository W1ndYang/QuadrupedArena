"""Outcome-driven report for API trials; never reuse first-assistant conclusions."""
import csv
import html
import json
from pathlib import Path
from llm_report import audit,read,write_json,score_chart,describe_event
from report_screenshots import extract_screenshots
from report_ui import build_ui


def build_api_report(directory,render_dir=None):
    summary,entries,outcomes,states,events=audit(directory)
    if render_dir is None:
        old=read(directory/'summary.json') if (directory/'summary.json').exists() else {}
        value=old.get('native_render',{}).get('directory')
        if value:render_dir=Path(value)
    with (directory/'decision_timing.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['decision','sim_start_s','sim_end_s','skill','decision_wall_s','execution_wall_s','api_wall_s','tokens','prompt_tokens','completion_tokens','reasoning_tokens','token_status'])
        for i,(e,o) in enumerate(zip(entries,outcomes)):
            u=e.get('token_usage',{})
            writer.writerow([i+1,e['simulation_start'],o['simulation_end'],e['action']['skill'],e['decision_wall_seconds'],o['execution_wall_seconds'],e.get('api_wall_seconds'),e.get('tokens'),u.get('prompt_tokens'),u.get('completion_tokens'),u.get('reasoning_tokens'),e.get('token_status')])
    if not render_dir:
        summary['report_status']='awaiting_native_render';write_json(directory/'summary.json',summary)
        return summary
    result=read(render_dir/'result.json')
    if result['status']!='passed' or result['replay']['source_sha256']!=summary['replay']['source_sha256']:raise ValueError('Native replay does not match this API trial')
    for name in ('demo.gif','demo.mp4','poster.png','intro.png'):
        link=directory/name;target=(render_dir/name).resolve()
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve()!=target:raise ValueError('Refusing artifact overwrite')
        else:link.symlink_to(target)
    summary['native_render']=dict(directory=str(render_dir),frames=result['frames'],duration_seconds=result['duration_seconds'],wall_seconds=result['wall_seconds'],renderer=result['renderer'],trajectory_hashes_verified=True)
    shots=extract_screenshots(directory,render_dir,summary,[e['simulation_start'] for e in entries]+[states[-1]['t']])
    model=summary['planner'];last=states[-1];latency=summary['latency_seconds']
    intro=[f"规划模型：{model}。完成 {len(entries)} 次决策，比赛 {last['t']:.2f} 秒，得分 {last['score']:.2f}，终局 HP {last['hp']}，结束原因 {last['finished']}。",
        f"每两秒调用固定技能；决策端到端等待合计 {latency['total']:.2f}s，平均 {latency['mean']:.2f}s；API 网络请求合计 {summary.get('api_wall_seconds',0):.2f}s。两者都不是纯芯片推理时间。",
        f"接口报告的总 token：{summary['token_usage'] if summary['token_usage'] is not None else '不完整/不可用'}。输入、输出、reasoning tokens（若提供）及原始 usage 保存在每步记录和 api_calls；reasoning tokens 是输出 token 的子集，不能重复相加。",
        '当前状态 JSON、当前导航示意图及已知人形策略交给模型。几何工具仅检查当前可达性、路径及视线，不允许未来试跑。报告中的 Isaac 截图来自事后回放，不是原始视觉输入。',
        '原规则、速度限制、种子和技能不变；提示词、历史组织和图像实际消费方式与上一轮交互式助手不同，不是严格只换模型的统计对照。',
        'CPU 裁判与运动学技能闭环，Isaac 原生回放；没有接触动力学、真实步态或端侧实时性验证。']
    phases=[(f"{e['time']:.2f} 秒",describe_event(e)) for e in events if describe_event(e)]
    limits=['这是单次固定种子的探索结果，不能外推模型通用优劣。','没有自动重试或由助手代写策略；非法输出会停止本轮并保留记录。','人形仍无听觉或受击方向响应，这一策略弱点未为本轮改变。','API 返回的模型标识、usage 与请求耗时已记录；未提供的值保持 null，不估算。']
    if (directory/'post_run_review.json').exists():
        review=read(directory/'post_run_review.json')
        intro.extend('评估者事后复盘：'+p for p in review.get('summary',[]))
        limits.extend(review.get('limitations',[]))
        summary['post_run_review']='post_run_review.json'
        summary['deterministic_audit']=review.get('deterministic_audit')
    skills=[('MOVE','固定几何 A* 跟随；可倒退/限速，不自动做战术决策。'),('HOLD','减速停止并保持指定朝向。'),('AIM','停止并按原转速上限跟踪敌人朝向。'),('SCAN','固定速率扫描。'),('STOP','停止当前路径。')]
    (directory/'score_timeline.svg').write_text(score_chart(states))
    build_ui(directory,summary,entries,outcomes,intro,phases,limits,skills,shots)
    md=['# '+model+' 比赛记录','',*intro,'','[选项卡报告](report.html) · [GIF](demo.gif) · [MP4](demo.mp4) · [耗时 CSV](decision_timing.csv)','']
    sections=[]
    for i,entry in enumerate(entries):
        request=read(directory/'requests'/f'{i:03}.json');text=json.dumps(request,ensure_ascii=False,indent=2)
        title=f"第 {i+1} 步：{entry['simulation_start']} → {outcomes[i]['simulation_end']} 秒"
        md.extend(['## '+title,'',f'![场景截图](screenshots/{i:03}.png)','',f'```json\n{text}\n```',''])
        sections.append(f'<section><h2>{title}</h2><img width="600" src="screenshots/{i:03}.png"><pre>{html.escape(text)}</pre></section>')
    (directory/'试验报告.md').write_text('\n'.join(md)+'\n')
    (directory/'report_detail.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(model)+'</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}pre{white-space:pre-wrap}img{max-width:100%}section{margin:30px 0}</style><a href="report.html">返回选项卡报告</a>'+''.join('<p>'+html.escape(p)+'</p>' for p in intro)+''.join(sections))
    summary['report_status']='complete';summary['report_viewer']=dict(version=2,screenshot_count=len(shots['screenshots']))
    write_json(directory/'summary.json',summary)
    return summary
