"""Local report: recorded decisions, native Isaac images and measured outcomes."""
import collections,html,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle

R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
dump=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
esc=html.escape

def plots(cases,checks):
    out=R/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.2),constrained_layout=True)
    for ax,c,ch in zip(axes,cases,checks):
        base=read(R/f"run_{c['run']}/trial_config.json")['base']
        for w in base['arena']['covers']:
            x,y=w['center'];sx,sy,h=w['size'];ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,fc='#e0e6e8' if h>1 else '#b9c1b8',ec='#7b8b91',lw=.7))
        for route in ch['departures']:
            ps=[ch['start_a']]+route['path'];ax.plot(*zip(*ps),color='#539799',lw=3,alpha=.6)
        states=read(R/f"run_{c['run']}/timeline.json")
        for i,color in ((0,'#007ba5'),(1,'#c05d26')):
            ps=[s['xy'][i] for s in states];ax.plot(*zip(*ps),color=color,lw=1.3,label='Quadruped A' if i==0 else 'Humanoid B')
            ax.scatter(*ps[0],color=color,s=38,marker='o',zorder=5);ax.scatter(*ps[-1],color=color,s=35,marker='x',zorder=5)
        ax.add_patch(Circle(ch['goal'],1,fc='#e2eee5',ec='#327c50',lw=1.5));ax.text(*ch['goal'],'G',ha='center',va='center',color='#28613d')
        ax.set(xlim=(-12,12),ylim=(-8,8),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f"{c['layout']}  |  A-B {ch['initial_separation_m']:.1f} m")
        ax.grid(alpha=.12);ax.legend(loc='lower left',fontsize=7)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'layouts_trajectories.{ext}',dpi=200)
    plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,5),constrained_layout=True)
    for c,ch,color in zip(cases,checks,['#007ba5','#c05d26','#6e5a97']):
        states=read(R/f"run_{c['run']}/timeline.json");decs=read(R/f"run_{c['run']}/decisions.json")
        ts=[s['t'] for s in states];distance=[math.dist(s['xy'][0],ch['goal']) for s in states]
        axs[0].plot(ts,distance,color=color,label=c['layout'])
        axs[1].plot(ts,[s['aim'][1] for s in states],color=color,label=c['layout'])
        for d in decs:
            s=states[round(d['simulation_seconds']/.02)];axs[0].plot(s['t'],math.dist(s['xy'][0],ch['goal']),'.',color=color,ms=5)
    axs[0].axhline(1,ls='--',lw=.8,color='#83918b');axs[0].set(ylabel='Goal distance (m)')
    axs[1].axhline(1,ls='--',lw=.8,color='#83918b');axs[1].set(ylabel='Defender lock (s)',xlabel='Simulation time (s)',ylim=(-.03,1.1))
    for ax in axs:ax.legend();ax.grid(alpha=.15)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'progress.{ext}',dpi=200)
    plt.close(fig)

def main():
    cases=read(R/'cases.json');checks=read(R/'layout_checks.json');reviews=read(R/'review.json') if (R/'review.json').exists() else {}
    plots(cases,checks);data=[]
    for c,ch in zip(cases,checks):
        f=R/f"run_{c['run']}";result=read(f/'result.json');ds=read(f/'decisions.json');events=read(f/'skill_events.json');states=read(f/'timeline.json');steps=[]
        for i,d in enumerate(ds):
            t=d['simulation_seconds'];end=ds[i+1]['simulation_seconds'] if i+1<len(ds) else result['seconds']
            ss=[s for s in states if t<=s['t']<=end];resp=read(f/f'api_calls/{i:03d}_response.json')
            steps.append(dict(index=i,start=t,end=end,api_seconds=resp['api_wall_seconds'],usage=resp['usage'],
                api_started_utc=resp['api_started_utc'],api_finished_utc=resp['api_finished_utc'],
                start_goal_m=math.dist(ss[0]['xy'][0],ch['goal']),end_goal_m=math.dist(ss[-1]['xy'][0],ch['goal']),
                traveled_m=sum(math.dist(a['xy'][0],b['xy'][0]) for a,b in zip(ss,ss[1:])),
                max_human_lock=max(s['aim'][1] for s in ss),events=[e for e in events if t<=e['time']<end],
                image=f"run_{c['run']}/native/steps/{i:03d}.png"))
        data.append(dict(case=c,check=ch,result=result,decisions=ds,steps=steps,review=reviews.get(c['layout'],{}),analysis=read(f/'analysis.json')))
    dump(R/'comparison.json',data)
    table=''.join(f"<tr><td>{v['case']['layout']}</td><td>{'到点获胜' if v['result']['winner']=='attacker' else '被标记' if v['result']['reason']=='tag' else '超时'}</td><td>{v['result']['seconds']:.2f} s</td><td>{len(v['decisions'])}</td><td>{v['result']['peek_bursts']}</td><td>{v['result']['api_median_seconds']:.1f} s</td><td>{v['result']['usage']['total_tokens']:,}</td></tr>" for v in data)
    findings=''.join(f"<div class='finding'><b>{esc(v['case']['layout'])} · {esc(v['review'].get('headline','按逐步记录审阅'))}</b><p>{esc(v['review'].get('summary',''))}</p><p class='note'>可取之处：{esc(v['review'].get('model_strength',''))}<br>解释限制：{esc(v['review'].get('limitation',''))}</p><div class='links'><a href='run_{v['case']['run']}/decisions.json'>逐步记录</a> · <a href='run_{v['case']['run']}/analysis.json'>轨迹统计</a></div></div>" for v in data)
    buttons=''.join(f"<button data-run='{i}' onclick='selectRun({i})'>{esc(c['layout'])} · {esc(c['label'])}</button>" for i,c in enumerate(cases))
    geometry=''
    for ch in checks:
        lengths=' / '.join(format(e['initially_occluded_prefix_m'],'.2f') for e in ch['departures'])
        geometry+=f"<tr><td>{ch['layout']}</td><td>{ch['start_a']}</td><td>{ch['start_b']}</td><td>{ch['goal']}</td><td>{ch['initial_separation_m']:.2f} m</td><td>{lengths} m</td></tr>"
    blob=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>遮挡出生 · DS LOW 三场检查</title><link rel="stylesheet" href="report_style.css">
<header>QuadrupedArena / Fixed goal v0.4.1 / 2026-11-04</header><nav><button data-tab="overview" onclick="showTab('overview')">总览与结果</button><button data-tab="layouts" onclick="showTab('layouts')">出生与布局检查</button><button data-tab="replay" onclick="showTab('replay')">Isaac 回放</button><button data-tab="steps" onclick="showTab('steps')">逐步决策</button><button data-tab="protocol" onclick="showTab('protocol')">设置与留档</button></nav><main>
<section id="overview" class="panel"><div class="lead"><div class="eyebrow">SHELTERED STARTS / THREE NEW LAYOUTS</div><h1>有遮挡、有出路，再检查模型决策</h1><p>三组新出生点、对手位置与固定目标。保留 DS LOW、合法几何感知及原有人形策略，检验获得信息后能否推进，不用“开局贴脸被击中”替代策略评估。</p></div><article><h2>本轮观测结果</h2><div class="tablewrap"><table><thead><tr><th>布局</th><th>终局</th><th>仿真时长</th><th>决策数</th><th>完成探身</th><th>调用中位数</th><th>已知 tokens</th></tr></thead><tbody>TABLE</tbody></table></div><p class="note">每布局仅一局，属于探索性样本，不能据此估计稳定胜率或作模型排名。token 含推理 token；失败请求如无用量回执不计为零。API 耗时是外部请求总耗时，含服务端排队、推理与传输，非纯思考时长。</p>FINDINGS</article><article><h2>实际路线与起步余地</h2><img class="figure" src="figures/layouts_trajectories.png"><p class="note">蓝线为四足，橙线为人形；圆点为出生、叉号为终局。浅青粗线是提前检查的两条出路，不是下发给模型的路线。图中完整人形轨迹只供事后审阅。</p><div class="links"><a href="figures/layouts_trajectories.svg">SVG</a> · <a href="figures/layouts_trajectories.pdf">PDF</a></div></article><article><h2>推进与人形锁定</h2><img class="figure" src="figures/progress.png"><p class="note">上图小圆点为模型决策时刻；直线距离上升也可能是合理绕行。下图锁定计时在命中时可能清零，最终命中以 events.json 为准。</p></article></section>
<section id="layouts" class="panel"><div class="sectionhead"><h1>先排除明显不合理的开局</h1></div><article><p>地图建筑保持不变。三个出生点均被实体墙遮挡，且相对人形的初始距离大于 10 m；目标不贴出生点。将人形巡逻路径以不超过 5 cm 的间隔采样，所有采样位置对出生点的视线均被墙阻挡。</p><div class="tablewrap"><table><tr><th>布局</th><th>四足 A</th><th>人形 B</th><th>目标 G</th><th>A–B 距离</th><th>两条遮挡首段</th></tr>GEOMETRY</table></div><p>每组另用实际 50 Hz 执行器沿两条出路各试走 2 秒，六次起步检查均未被命中。检查只用于证明存在可利用的起步余地，不是无敌保护，也不保证模型选择任意路线都安全。</p><p class="note">三组布局在首次付费调用前冻结；没有根据正式对局输赢重新换点。</p><a href="layout_checks.json">几何与起步检查原始数据</a></article></section>
<section id="replay" class="panel"><div class="sectionhead"><h1>第三视角回放</h1><div class="stepbar runbar">BUTTONS</div></div><article><h2 id="replay-title"></h2><img id="gif" class="replay"><p class="note">CPU 运动学裁判执行；Isaac Sim 原生 PathTracing 按保存轨迹渲染。绿色目标、蓝色四足、橙色人形。模型没有看这些全场图片，思考期间仿真暂停。</p><a id="giflink">单独打开 GIF</a></article></section>
<section id="steps" class="panel"><div class="sectionhead"><h1>逐步决策与实际执行</h1><div class="stepbar runbar">BUTTONS</div><div id="stepbar" class="stepbar"></div></div><article><div id="stepmeta" class="stepmeta"></div><div class="decision-grid"><div><img id="shot" class="decision-shot"><div id="facts" class="facts"></div><details><summary>本步实际执行事件</summary><pre id="events"></pre></details></div><div><pre id="action"></pre><details><summary>模型当时收到的合法公共观测</summary><pre id="packet"></pre></details></div></div><div class="links"><a id="requestlink">原始请求</a> · <a id="responselink">原始返回</a></div></article></section>
<section id="protocol" class="panel"><div class="sectionhead"><h1>可复核设置</h1></div><article><p>DeepSeek 官方 deepseek-flash，thinking enabled，reasoning_effort=low；每次最大响应 65,536 token，没有额外累计 token 预算。仅发送最终动作历史，不把前轮 assessment 或完整推理追加到提示词；每轮最多保留最近四次动作。</p><p>规则延续：60 秒内进入唯一目标半径 1 m 获胜；被人形有效标记即失败。双方视距不限，四足 360°、人形 180°，墙与烟挡视线。射程 8 m，射手停稳且实际方向误差不超过 5°，连续合法锁定 1 秒才命中。四足标记人形仅造成 1 秒失能，冷却 4 秒；一颗烟。</p><p>技能和接触守卫语义保持原样：retreat 遇到可见且 8 m 内人形先撤回，会优先于机会射击。FOLLOW_ROUTE 连接地图路点；PEEK_BURST 在掩体短探后返回。事件触发模型调用，期间整体时停。</p><p>本轮版本差异：启用已验证的 v0.4.1 探身提前撤回刹车修复；仍保留原始航向数值、路线执行与人形记忆搜索策略。更换双方起点、目标与巡逻段，SPAWN 路点同步为实际出生点。未把既往失败诊断、测试结果或人工策略建议交给模型。</p><p>运动修复与场景同时变更，因此这轮是新场景能力检查，不是相对上一局的单变量性能实验。每组一局不足以区分模型随机性与布局差异。</p><div class="links"><a href="preregistration.json">原实验预注册</a> · <a href="source_manifest.json">发布版源码哈希</a> · <a href="layouts.json">布局</a> · <a href="api_profile.json">API 设置</a> · <a href="audit.json">轨迹与观测审计</a> · <a href="comparison.json">对比数据</a> · <a href="README.md">复现说明</a></div></article></section>
</main><footer>记录模型输出与实际行为；不把事后解释当作模型的内部思考。</footer><script>
const data=DATA;let runIndex=0,stepIndex=0;
function showTab(id){if(!document.getElementById(id))id='overview';document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===id));document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===id));history.replaceState(null,'','#'+id)}
function selectRun(i){runIndex=i;document.querySelectorAll('[data-run]').forEach(b=>b.classList.toggle('active',+b.dataset.run===i));const r=data[i],f='run_'+r.case.run;document.getElementById('gif').src=f+'/demo.gif';document.getElementById('giflink').href=f+'/demo.gif';document.getElementById('replay-title').textContent=r.case.layout+' · '+r.case.label;const bar=document.getElementById('stepbar');bar.replaceChildren();r.decisions.forEach((d,j)=>{const b=document.createElement('button');b.textContent=(j+1)+' · '+d.simulation_seconds.toFixed(2)+'s';b.dataset.step=j;b.onclick=()=>selectStep(j);bar.append(b)});selectStep(0)}
function selectStep(j){stepIndex=j;const r=data[runIndex],d=r.decisions[j],s=r.steps[j],folder='run_'+r.case.run;document.querySelectorAll('[data-step]').forEach(b=>b.classList.toggle('active',+b.dataset.step===j));document.getElementById('shot').src=s.image;document.getElementById('action').textContent=JSON.stringify(Object.fromEntries(['observation_id','skill','params','assessment'].map(k=>[k,d[k]])),null,2);document.getElementById('packet').textContent=JSON.stringify(d.public_observation,null,2);document.getElementById('events').textContent=JSON.stringify(s.events,null,2);document.getElementById('stepmeta').textContent=`${r.case.layout} / 决策 ${j+1} / 仿真 ${s.start.toFixed(2)}–${s.end.toFixed(2)} s / API ${s.api_seconds.toFixed(2)} s / ${s.usage.total_tokens.toLocaleString()} tokens`;document.getElementById('facts').textContent=`实际移动 ${s.traveled_m.toFixed(2)} m；目标直线距离 ${s.start_goal_m.toFixed(2)} → ${s.end_goal_m.toFixed(2)} m；本区间人形最大记录锁定 ${s.max_human_lock.toFixed(2)} s。API UTC：${s.api_started_utc} → ${s.api_finished_utc}。`;const prefix=folder+'/api_calls/'+String(j).padStart(3,'0');document.getElementById('requestlink').href=prefix+'_request.json';document.getElementById('responselink').href=prefix+'_response.json'}
selectRun(0);showTab(location.hash.slice(1)||'overview');
</script></html>'''
    for a,b in [('TABLE',table),('FINDINGS',findings),('GEOMETRY',geometry),('BUTTONS',buttons),('DATA',blob)]:doc=doc.replace(a,b)
    (R/'index.html').write_text(doc)
    print('REPORT_BUILT',sum(len(v['decisions']) for v in data),'decisions')

if __name__=='__main__':main()
