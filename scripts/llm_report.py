"""Audit and presentation for an already finished external-planner trial.

This script does not load private checkpoints, execute a policy or advance time.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from replay_io import load_replay


def read(path):
    return json.loads(path.read_text())


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def audit(directory):
    cfg,states,events,validation,replay=load_replay(directory)
    entries=read(directory/'decisions.json');outcomes=[]
    for i,entry in enumerate(entries):
        commit=read(directory/'decisions'/f'{i:03}.json')
        request=read(directory/'requests'/f'{i:03}.json')
        obs=read(directory/'observations'/f'{i:03}.json')
        outcome=read(directory/'decisions'/f'{i:03}_outcome.json')
        assert entry==commit and all(commit[k]==v for k,v in request.items())
        assert entry['observation_id']==obs['observation_id']==i
        assert abs(entry['simulation_start']-obs['state']['t'])<1e-8
        assert abs(outcome['simulation_end']-min(entry['simulation_start']+2,states[-1]['t']))<1e-8
        assert entry['decision_wall_seconds']>=0
        assert entry['tokens'] is None or type(entry['tokens']) is int and entry['tokens']>=0
        assert outcome['state']==states[round(outcome['simulation_end']/cfg['dt'])]
        assert obs['state']==states[round(entry['simulation_start']/cfg['dt'])]
        if i:assert abs(entry['simulation_start']-outcomes[-1]['simulation_end'])<1e-8
        outcomes.append(outcome)
    assert outcomes[-1]['state']==states[-1]
    times=[e['decision_wall_seconds'] for e in entries]
    summary=dict(status='passed',decisions=len(entries),simulation_seconds=states[-1]['t'],
        score=states[-1]['score'],hp=states[-1]['hp'],finish=states[-1]['finished'],seed=cfg['seed'],
        latency_seconds=dict(total=sum(times),mean=statistics.mean(times),median=statistics.median(times),
            p95_nearest_rank=sorted(times)[math.ceil(.95*len(times))-1],minimum=min(times),maximum=max(times)),
        skill_execution_wall_seconds=sum(o['execution_wall_seconds'] for o in outcomes),
        token_usage=None,token_status='unavailable; not zero; no estimate',
        latency_scope='observation ready through accepted action; includes tool calls, analysis, commentary and context processing; not pure inference',
        commit_audit='request equals pre-execution commit; observation/outcome match full timeline; 2s cadence; no missing decision',
        replay=replay,validation=validation)
    protocol=read(directory/'protocol.json') if (directory/'protocol.json').exists() else {}
    summary['planner']=protocol.get('planner','external assistant')
    if protocol.get('controller_type')=='external_api':
        summary['protocol_version']=protocol.get('protocol_version','qwen-r1')
        summary['prompt_sha256']=protocol.get('prompt_sha256')
        summary['latency_scope']='observation appended through accepted action; API waits, tool execution and record I/O; not pure inference'
    token_values=[e.get('tokens') for e in entries]
    summary['token_usage']=sum(token_values) if all(type(v) is int for v in token_values) else None
    summary['token_status']='provider_reported' if summary['token_usage'] is not None else 'unavailable_or_partial; not zero'
    if any('api_wall_seconds' in e for e in entries):
        summary['api_wall_seconds']=sum(e.get('api_wall_seconds',0) for e in entries)
        summary['api_call_count']=sum(e.get('api_call_count',0) for e in entries)
    if (directory/'baseline/comparison.json').exists():
        b=read(directory/'baseline/comparison.json');bc=read(directory/'baseline/config.json')
        assert bc==cfg
        summary['reference_baseline']=dict(score=b['last']['score'],survival_seconds=b['last']['t'],hp=b['last']['hp'],
            finish=b['last']['finished'],config_equal=True,
            caveat='Single seed only; richer LLM observations and different action interfaces/control cadence. Same seed does not align realized random hits after trajectories diverge.')
    return summary,entries,outcomes,states,events


def describe_event(e):
    kind=e['type'];t=e['time']
    if kind=='hit':return f"{t:.2f}s {'A → B' if e['attacker']==0 else 'B → A'} 命中"
    if kind=='target_triggered':return f"{t:.2f}s P{e['point']+1} 触发，{e['expires_at']:.2f}s 到期"
    if kind=='target_invalidated':return f"{t:.2f}s P{e['point']+1} 失效（{e['reason']}）"
    if kind=='collapse':return f'{t:.2f}s 固定墙体坍塌'
    if kind=='eliminated':return f"{t:.2f}s {'A' if e['actor']==0 else 'B'} 淘汰"
    if kind=='finish':return f"{t:.2f}s 结束（{e['reason']}）"
    if kind=='skill_cancelled':return f"{t:.2f}s 技能取消（{e['reason']}）"
    return None


def score_chart(states,baseline=None):
    paths=[]
    ceiling=max(12,math.ceil(max(s['score'] for s in states+(baseline or []))/4)*4)
    scale=156/ceiling
    for series,color,label in ((states,'#38bdf8','LLM'),(baseline,'#fb923c','旧策略')):
        if not series:continue
        pts=' '.join(f"{60+s['t']*17:.1f},{230-s['score']*scale:.1f}" for s in series[::5]+series[-1:])
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts}"/>')
    grid=''.join(f'<line x1="60" y1="{230-v*scale}" x2="740" y2="{230-v*scale}" stroke="#304253"/><text x="25" y="{234-v*scale}">{v}</text>' for v in range(0,ceiling+1,4))
    labels=''.join(f'<text x="{52+t*17}" y="256">{t}s</text>' for t in (0,10,20,30,40))
    legend='<text x="140" y="26" fill="#fb923c">旧策略（终止后不外推）</text>' if baseline else ''
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 285" role="img" aria-label="累计得分随比赛时间变化"><rect width="800" height="285" rx="12" fill="#132333"/><g fill="#dce9f2" font-size="14">'+grid+labels+'<text x="60" y="26" fill="#38bdf8">LLM</text>'+legend+''.join(paths)+'</g></svg>'


def build(directory,render_dir=None):
    if (directory/'protocol.json').exists() and read(directory/'protocol.json').get('controller_type')=='external_api':
        from api_trial_report import build_api_report
        return build_api_report(directory,render_dir)
    summary,entries,outcomes,states,events=audit(directory)
    # Keep the original long-form page available when upgrading the viewer.
    previous=directory/'report.html';backup=directory/'report_v1.html'
    if previous.exists() and not backup.exists():backup.write_bytes(previous.read_bytes())
    if render_dir is None and (directory/'summary.json').exists():
        saved=read(directory/'summary.json').get('native_render',{})
        if saved.get('directory'):render_dir=Path(saved['directory'])
    screenshots=None
    # Replay links are attached only after the renderer reports success and the
    # source hashes agree. A failed/partial video is not a deliverable.
    if render_dir:
        result=read(render_dir/'result.json')
        assert result['status']=='passed'
        assert result['replay']['source_sha256']==summary['replay']['source_sha256']
        for name in ('demo.gif','demo.mp4','poster.png','intro.png'):
            link=directory/name;target=(render_dir/name).resolve()
            if link.is_symlink():assert link.resolve()==target
            elif link.exists():raise ValueError('Refusing to replace existing artifact '+str(link))
            else:link.symlink_to(target)
        summary['native_render']=dict(directory=str(render_dir),frames=result['frames'],
            duration_seconds=result['duration_seconds'],wall_seconds=result['wall_seconds'],renderer=result['renderer'],
            trajectory_hashes_verified=True)
        from report_screenshots import extract_screenshots
        screenshots=extract_screenshots(directory,render_dir,summary,
            [entry['simulation_start'] for entry in entries]+[outcomes[-1]['simulation_end']])
        summary['report_viewer']=dict(version=2,screenshot_count=len(screenshots['screenshots']),
            screenshot_manifest='screenshots/manifest.json',default_tab='decisions',json_content='original request, observation, outcome and commitment')
    write_json(directory/'summary.json',summary)
    with (directory/'decision_timing.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['decision','sim_start_s','sim_end_s','skill','decision_wall_s','execution_wall_s','tokens','token_status'])
        for i,(e,o) in enumerate(zip(entries,outcomes)):
            writer.writerow([i+1,e['simulation_start'],o['simulation_end'],e['action']['skill'],e['decision_wall_seconds'],o['execution_wall_seconds'],'','unavailable'])
    latency=summary['latency_seconds'];baseline=summary.get('reference_baseline')
    intro=[
        '本轮完成了真实的“观测 → 外部大模型选择技能 → 执行 2 秒 → 再观测”闭环。未训练，未用预先编写的高层动作序列代替模型决策；未回滚挑选成功结果。',
        f"比赛 {summary['simulation_seconds']:.0f} 秒，共 {len(entries)} 次决策，四足 {summary['score']:.2f} 分，终局 HP：A={summary['hp'][0]}、B={summary['hp'][1]}。P1 合计 8 分，P2 合计 4 分。",
        f"决策端到端等待合计 {latency['total']:.2f} 秒（{latency['total']/60:.2f} 分钟），平均 {latency['mean']:.2f} 秒/步，中位数 {latency['median']:.2f} 秒，P95（最近秩）{latency['p95_nearest_rank']:.2f} 秒，最长 {latency['maximum']:.2f} 秒。",
        '耗时口径：从本步观测文件准备完成，到动作校验并接受；包含分析、工具调用、界面交互、上下文处理等等待。它不是纯模型推理时间，也不能直接用来推算端侧芯片速度。环境搭建、报告编写、后续渲染均不在这些决策窗口内。',
        'Token：当前会话工具未提供逐次 usage，全部明确记为 null / unavailable；不是 0，也未按文本长度伪造估算。',
        '本轮使用当前助手进行规划，服务端精确模型版本未暴露。每步主要读取结构化真值状态，同时生成可按需查看的顶视示意图；不是基于 Isaac 相机图像的视觉闭环。后续 Isaac 图片仅用于忠实重放和展示。',
        '边界：CPU 裁判和运动学代理承担闭环状态推进，Isaac 负责原生模型与工业场景回放；本轮没有在 Isaac 接触动力学中驱动真实步态，也没有端侧推理、传感器噪声或网络实时性验证。',
    ]
    if baseline:intro.append(f"同配置同种子的旧风险绕行策略参考：{baseline['score']:.2f} 分，{baseline['survival_seconds']:.2f} 秒淘汰，终局 HP {baseline['hp']}。本轮多 4 分（50%），但仅一个种子，不能视为统计优势。模型获得更完整的对手状态/策略知识，动作接口和控制频率也不同；共同随机种子不代表轨迹分叉后的命中事件逐次相同。")
    observations=[
        ('0–6 秒：拿满首点','在墙后完成驻留；临近首次 5 秒失效时提前启动转移，实际拿到 4 分。'),
        ('6–14 秒：暴露转移与一次失血','穿越开阔区时未能完全避开人形，8.74 秒损失 1 HP；随后利用已发生命中的冷却窗口前进，10.20 秒切断视线，12.20 秒触发 P2。'),
        ('16–24 秒：提前离开即将失效的点','16 秒限速慢移，保住剩余计分同时开始转移；17.20 秒 P2 满 4 分到期。20 秒墙体坍塌，四足已经离开原北侧驻留区域；23.32 秒再次触发 P1。'),
        ('24–30 秒：驻留中攻击、再倒退绕墙','25.06 秒四足首次命中尚未发现它的人形；28.32 秒 P1 再拿满 4 分，总分 12。28.60/28.66 秒双方互相命中，四足撤离过程中降至 1 HP。'),
        ('32–36 秒：一次值得批评的转移','32 秒向 P2 倒退前进，于 33.68 秒重新暴露，34 秒被迫回撤。这个来回没有新增得分，风险评估仍不够好；34.90 秒回撤途中击败人形属于本种子实际随机结果，不是撤离必然安全。'),
        ('36–40 秒：没有赶上最后占点','淘汰人形不直接加分。四足转身、绕过坍塌障碍后未及进入 P2 完成驻留，40 秒以 12 分结束。'),
    ]
    limitations=[
        '可以支持的结论：模型能在暂停机制下，结合剩余驻留时间、目标刷新、墙体遮挡、对手状态和冷却，逐步组合固定技能，完成一轮可审计闭环。',
        '暂不能支持的结论：普遍显著强于决策树、视觉自主理解已经完成、真实机器人运动可行、27B 或某款端侧设备能实时达到相同效果。',
        '人形没有听觉和受击方向响应：25.06 秒被打中后仍可按巡逻逻辑行动。本轮在未知自身已暴露的人形身上取得攻击优势；下一轮应把它作为公开的对手设计问题，不能包装成普遍智能优势。',
        '两秒只重规划一次，窗口内只执行技能和几何安全检查，没有自动战术躲避。这会造成受击后无法立刻改计划；下一阶段可比较“2 秒常规决策 + 紧急事件打断”，但需对基线同等开放。',
        '后续应在相同观测与相同技能接口下，对多个事先固定种子比较 LLM、规则策略及端侧候选模型；分别统计得分、存活、暴露、无效往返、非法动作、推理耗时和真实 API usage。本轮不自动开展这些新试验。',
    ]
    skills=[('MOVE','给定 goal / via，几何 A* 生成路线；可限制速度或允许倒退。行进中身体朝向受移动器约束，并非独立炮塔。'),
        ('HOLD','清空运动路径、减速停止并保持指定朝向；规则继续计时和计分。'),
        ('AIM','停止移动，按转向上限持续朝向当前人形；几何与随机命中由原裁判负责。'),
        ('SCAN','停止移动并按固定速率转向；本次未使用。'),('STOP','停止路径执行；安全检查或目标代次失效也可触发停止，本次未主动调用。')]
    md=['# LLM 首轮闭环试验报告','',*[p+'\n' for p in intro],
        '## 架构与输入边界','','`当前状态 → 暂停比赛计时 → 模型提交技能 JSON → 校验并存档 → 执行 2 s → 下一状态`','',
        '模型知道地图、双方位置/朝向、血量、冷却、瞄准累计、当前对手状态/记忆/已规划路径、目标刷新及固定坍塌规则；不提供 RNG 状态、随机命中阈值、未来实际轨迹或未来试跑工具。技能层没有旧的 QuadrupedPolicy 风险策略，也不会自行选择占点或撤离。','',
        '| 技能 | 固定能力 |','| --- | --- |',*[f'| {k} | {v} |' for k,v in skills],'',
        'MOVE 可选 target_generation：绑定当前代次时，目标刷新后取消旧指令；未绑定允许提前向下一点移动。所有速度、加速度、转向、清障距离沿用旧配置。首次进圈 5 秒失效、驻留 1 秒后计分，20 秒固定坍塌。','',
        '## 得分过程','','![累计得分](score_timeline.svg)','',
        *[f'### {title}\n\n{body}\n' for title,body in observations],
        '## 结论与局限','',*[f'- {p}' for p in limitations],'',
        '## 决策耗时与实际结果','','| 步 | 仿真区间 | 技能 | 端到端等待/s | 执行计算/s | 结果分数 | A/B HP | Token |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    cards=[];rows=[]
    for i,(entry,outcome) in enumerate(zip(entries,outcomes)):
        before=read(directory/'observations'/f'{i:03}.json')['state'];after=outcome['state']
        start,end=entry['simulation_start'],outcome['simulation_end'];skill=entry['action']['skill']
        row=f"| {i+1:02} | {start:g}–{end:g}s | {skill} | {entry['decision_wall_seconds']:.2f} | {outcome['execution_wall_seconds']:.4f} | {after['score']:.2f} | {after['hp'][0]}/{after['hp'][1]} | N/A |"
        md.append(row)
        rows.append(f'<tr><td><a href="decision_cards/{i+1:02}.html">{i+1:02}</a></td><td>{start:g}–{end:g}s</td><td>{skill}</td><td>{entry["decision_wall_seconds"]:.2f}</td><td>{after["score"]:.2f}</td><td>{after["hp"][0]}/{after["hp"][1]}</td><td>N/A</td></tr>')
        ev=[describe_event(e) for e in events if start<e['time']<=end];ev=[e for e in ev if e]
        actual=f"得分 {before['score']:.2f} → {after['score']:.2f}；HP {before['hp']} → {after['hp']}；技能状态 {after['skill_status']}。"+('；'.join(ev) if ev else '本窗口无命中、失效或坍塌等关键事件。')
        title=f'决策 {i+1:02}｜{start:g} → {end:g} 秒｜{skill}'
        items=[('观测依据',entry['evidence']),('选择摘要',entry['choice_summary']),('备选与取舍',entry['alternatives_summary']),('提交时预期',entry['expected_result']),('执行后核对',actual)]
        details=''.join(f'<p><strong>{k}：</strong>{html.escape(v)}</p>' for k,v in items)
        command=json.dumps(entry['action'],ensure_ascii=False,indent=2)
        image_dir='screenshots' if screenshots else 'observations'
        width,height=(960,960) if screenshots else (1000,760)
        figures=f'<div class="pair"><figure><img loading="lazy" width="{width}" height="{height}" src="{image_dir}/{i:03}.png"><figcaption>决策前状态 · {start:g}s</figcaption></figure><figure><img loading="lazy" width="{width}" height="{height}" src="{image_dir}/{i+1:03}.png"><figcaption>执行后状态 · {end:g}s（非预测图）</figcaption></figure></div>'
        full_request=json.dumps(read(directory/'requests'/f'{i:03}.json'),ensure_ascii=False,indent=2)
        image_note='Isaac 原生回放截图；事后展示，不是原始规划视觉输入。' if screenshots else '程序绘制观测示意图。'
        cards.append(f'<section id="decision-{i+1}"><h2>{title}</h2><p class="meta">端到端等待 {entry["decision_wall_seconds"]:.2f}s · 技能执行计算 {outcome["execution_wall_seconds"]:.4f}s · token 未提供</p>{figures}<p class="meta">{image_note}</p>{details}<h3>原始输出 JSON（执行前存档）</h3><pre>{html.escape(full_request)}</pre></section>')
        entry['_report_detail']=(title,items,command)
    md.extend(['','## 每步决策说明','','以下是基于可核验状态的决策摘要与结果核对，不是内部推理逐字记录。观测图中的路线为此前已执行指令的剩余路径；本步新提交路线保存在 decisions 中的 compiled_route。',''])
    for i,entry in enumerate(entries):
        title,items,command=entry.pop('_report_detail')
        image_dir='screenshots' if screenshots else 'observations'
        md.extend([f'### {title}','',f'![决策前]({image_dir}/{i:03}.png)','',*[f'**{k}：** {v}\n' for k,v in items],f"本步端到端等待：{entry['decision_wall_seconds']:.2f}s；Token：不可取得。",'',f'```json\n{command}\n```',''])
    native=''
    if 'native_render' in summary:
        native='<section><h2>Isaac 原生工业场景回放</h2><video controls preload="metadata" poster="poster.png" src="demo.mp4"></video><p><a href="demo.gif">打开 GIF</a> · <a href="demo.mp4">打开 MP4</a>。比赛 40 秒，额外片头、片尾与坍塌展示镜头不增加比赛计时。图像仅重放已提交轨迹。</p></section>'
        md[2:2]=['[GIF 演示](demo.gif) · [MP4 演示](demo.mp4) · [浏览器图文报告](report.html) · [逐步耗时 CSV](decision_timing.csv)','', '![Isaac 工业场景终局](poster.png)','']
    baseline_states=read(directory/'baseline/timeline.json') if baseline else None
    (directory/'score_timeline.svg').write_text(score_chart(states,baseline_states))
    (directory/'试验报告.md').write_text('\n'.join(md)+'\n')
    content=''.join(f'<p>{html.escape(p)}</p>' for p in intro)
    phases=''.join(f'<h3>{html.escape(t)}</h3><p>{html.escape(b)}</p>' for t,b in observations)
    skill_html=''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in skills)
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM 首轮闭环试验</title><style>
body{margin:0;background:#0c1722;color:#e3edf5;font:16px/1.75 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:32px}h2{font-size:24px;color:#72d4ff}h3{font-size:19px}section{background:#142533;border:1px solid #294253;border-radius:12px;margin:24px 0;padding:24px}a{color:#7ed9ff}p{max-width:1050px}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #344c5d;padding:9px}pre{white-space:pre-wrap;background:#0c1924;padding:15px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}img,video{width:100%;height:auto;border-radius:8px}figcaption,.meta{color:#afc7d8;font-size:14px}.notice{border-left:4px solid #fb923c;padding-left:18px}svg{max-width:850px}details{margin-top:12px}@media(max-width:800px){main{padding:12px}.pair{grid-template-columns:1fr}section{padding:14px}table{font-size:12px}}@media print{body{background:white;color:#111}section{background:white;break-inside:avoid}h2{color:#124f70}a{color:#125d83}video{display:none}}
</style><main><h1>LLM 首轮闭环试验</h1><p>2026-10-21 · R0 v0.4 · 全知当前状态 · 每 2 秒技能调用 · 单次不回滚</p>'''
    report+=f'<section><h2>结果与测量口径</h2>{content}<p><a href="decision_timing.csv">下载逐步耗时 CSV</a> · <a href="summary.json">审计摘要 JSON</a> · <a href="试验报告.md">Markdown 报告</a></p></section>'+native
    report+='<section><h2>控制分层</h2><pre>当前观测 → 暂停比赛 → 模型提交技能 → 校验与存档 → 执行 2 秒 → 新观测</pre><p>输入是当前真值、规则与已知对手策略；不含未来命中阈值或未来实际轨迹。技能层为固定执行器，没有自主战术选择。人形策略不变。</p><table>'+skill_html+'</table><p>本轮实际使用 MOVE、HOLD、AIM。MOVE 行进朝向由路径跟随器控制；facing 参数不是独立炮塔。可绑定目标代次，刷新时取消过期指令。</p></section>'
    report+='<section><h2>比赛过程</h2><img src="score_timeline.svg" alt="累计得分曲线">'+phases+'</section>'
    report+='<section class="notice"><h2>如何解读本轮结果</h2>'+''.join(f'<p>{html.escape(p)}</p>' for p in limitations)+'</section>'
    report+='<section><h2>每步耗时与结果</h2><table><tr><th>决策</th><th>比赛区间</th><th>技能</th><th>等待/s</th><th>累计分</th><th>A/B HP</th><th>Token</th></tr>'+''.join(rows)+'</table><p>下文记录可核验的决策摘要、执行前预期和执行后结果。Isaac 截图为事后回放展示，不是该步规划输入；图中路线是上一指令的剩余路线。本步新路线见 decisions/NNN.json 的 compiled_route。</p></section>'
    report+=''.join(cards)+'</main></html>'
    (directory/'report_detail.html').write_text(report)
    if screenshots:
        from report_ui import build_ui
        build_ui(directory,summary,entries,outcomes,intro,observations,limitations,skills,screenshots)
    else:(directory/'report.html').write_text(report)
    # Separate cards remain convenient to open, screenshot or share without
    # navigating a very long illustrated document.
    card_dir=directory/'decision_cards';card_dir.mkdir(exist_ok=True)
    for i,card in enumerate(cards):
        navigation='<a href="../report.html">返回总报告</a>'
        if i:navigation+=f' · <a href="{i:02}.html">上一步</a>'
        if i+1<len(cards):navigation+=f' · <a href="{i+2:02}.html">下一步</a>'
        page=report.split('<main>')[0]+'<main><p>'+navigation+'</p>'+card.replace('src="observations/','src="../observations/').replace('src="screenshots/','src="../screenshots/')+'</main></html>'
        (card_dir/f'{i+1:02}.html').write_text(page)
    audit_files=[directory/'protocol.json',directory/'trial_config.json',directory/'timeline.json',directory/'events.json',directory/'decisions.json',directory/'validation.json',
        *sorted((directory/'decisions').glob('*.json')),*sorted((directory/'requests').glob('*.json')),*sorted((directory/'observations').glob('*.json'))]
    write_json(directory/'audit_sha256.json',{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in audit_files})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--render-dir',type=Path)
    args=parser.parse_args();print(json.dumps(build(args.directory,args.render_dir),ensure_ascii=False,indent=2))
