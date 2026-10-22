"""Build an offline, evidence-linked HTML research report from archived data."""
import html
import json
import shutil
from pathlib import Path
from research_data import OUT

def esc(x):return html.escape(str(x),quote=True)
def fmt(x,n=2):return '—' if x is None else f'{x:,.{n}f}'
def table(head,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def picture(file,caption,marker=None):
    overlay=''
    if marker:overlay=f'<svg class="pixel-overlay" viewBox="0 0 768 432" aria-label="模型提交的脚点标注"><circle cx="{marker[0]}" cy="{marker[1]}" r="9" fill="none" stroke="white" stroke-width="4"/><circle cx="{marker[0]}" cy="{marker[1]}" r="9" fill="none" stroke="#d23d65" stroke-width="2"/><path d="M {marker[0]-15} {marker[1]} h30 M{marker[0]} {marker[1]-15} v30" stroke="#d23d65" stroke-width="2"/></svg>'
    return f'<figure class="photo"><a class="image-box" href="{esc(file)}" target="_blank"><img loading="lazy" src="{esc(file)}" alt="{esc(caption)}">{overlay}</a><figcaption>{caption}</figcaption></figure>'
def figure(name,num,title,caption):
    return f'<figure class="science-figure"><div class="figure-top"><span>FIGURE {num:02}</span><div>'+''.join(f'<a download href="figures/{name}.{ext}">{label}</a>' for ext,label in [('svg','SVG'),('pdf','PDF'),('png','PNG · 600 dpi'),('json','源数据')])+f'</div></div><a href="figures/{name}.svg" target="_blank"><img loading="lazy" src="figures/{name}.svg" alt="图 {num}：{title}"></a><figcaption><strong>图 {num}｜{title}。</strong>{caption}</figcaption></figure>'

def build():
    d=json.loads((OUT/'data/analysis.json').read_text());runs=d['runs'];by={r['id']:r for r in runs};v=runs[2:]
    decision_count=sum(len(r['steps']) for r in runs)
    historical_max=next(h for h in d['history'] if h['id']=='DH16')
    def shot(ident,time,kind='input',mark=False):
        r=by[ident]
        if kind=='over':
            f=min(r['overhead'],key=lambda f:abs(f['time']-time));return picture(f['file'],f'{ident} · t={f["time"]:.2f} s · 赛后 Isaac 俯视图，不是决策输入。')
        f=next(f for s in r['steps'] for f in s['frames'] if abs(f['simulation_seconds']-time)<1e-6)
        a=next((a for a in r['annotations'] if abs(a['time']-time)<1e-6),None)
        return picture(f['report_file'],f'{ident} · t={time:.2f} s · 原始机载 RGB'+('；粉色十字为模型实际提交的脚点。' if mark and a else '。'),[a['pixel_u'],a['pixel_v']] if mark and a else None)
    def raw(ident,step):return '<details><summary>查看当时原始结构化决策 JSON</summary><pre>'+esc(json.dumps(by[ident]['steps'][step]['raw'],ensure_ascii=False,indent=2))+'</pre></details>'
    def excerpt(ident,step):
        s=by[ident]['steps'][step];r=s['raw'];return '<blockquote>'+esc(r.get('assessment',{}).get('decision_summary',r.get('choice_summary','')))+f'<cite>{ident} · decision {step:03d} · t={s["time"]:.2f} s；原样摘录</cite></blockquote>'
    rows=[]
    for r in runs:
        m=r['metric'];rows.append([f'<a href="#explorer" data-jump="{r["id"]}:0"><span class="dot" style="--c:{r["color"]}"></span>{r["id"]}</a>',esc(r['label']),f'{m["score_observed"]:g}'+(' <span class="badge warn">中断值</span>' if not m['completed'] else ''),fmt(m['end_seconds']),f'{m["hp_a"]} / {m["hp_b"]}',str(m['decisions']),str(m['calls'] or '未记录'),('时间到' if m['finish']=='time_limit' else '四足清零' if m['completed'] else 'API 无可执行输出')])
    scoreboard=table(['编号','条件','得分','时长 / s','HP A / B','有效决策','API 请求','观察终点'],rows)
    latency=table(['条件','决策等待均值 / 中位数 / P95（s）','API 均值（s）','技能编译均值（s）','tokens 总数','其中 reasoning'],[[r['id'],f'{fmt(r["metric"]["wait_mean"])} / {fmt(r["metric"]["wait_median"])} / {fmt(r["metric"]["wait_p95"])}',fmt(r['metric']['api_mean']),fmt(r['metric']['compile_mean'],3),fmt(r['metric']['total_tokens'],0),fmt(r['metric']['reasoning_tokens'],0)] for r in runs])
    mechanism=table(['条件','走行距离 / m','B 看见 A / s','其中 A 静止 / s','命中对手 / 受击','触发点实例数','路线拒绝','脚点标注'],[[r['id'],fmt(r['metric']['path_m']),fmt(r['metric']['human_sees_a_seconds']),fmt(r['metric']['stationary_seen_seconds']),f'{r["metric"]["hits_inflicted"]} / {r["metric"]["hits_received"]}',str(r['metric']['target_triggers']),str(r['metric']['rejected_commands']),str(r['metric']['annotations'])] for r in v])
    registry=table(['编号','实验目录','纳入方式'],[[r['id'],f'<a href="{esc(r["source_report"])}">{esc(Path(r["root"]).name)}</a>','主分析'+('；系统中断，单独标记' if not r['metric']['completed'] else '')] for r in runs]+[['BT','<code>O-A/baseline</code>','规则/A* 上下文参考，不是同接口的模型对照']]+[[h['id'],f'<a href="{esc(h["source_report"])}">{esc(Path(h["path"]).name)}</a>' if h['source_report'] else esc(Path(h['path']).name),esc(h['note'])] for h in d['history']])
    audit=table(['核验对象','最近五轮','SHA-256（共同值）'],[[esc(a['file']),'字节一致' if a['visual_identical'] else '不一致/缺失',f'<code>{next((x for x in a["sha256"].values() if x),"—") if a["file"]=="trial_config.json" else a["sha256"]["V-A"]}</code>'] for a in d['audit']])
    q=by['O-Q'];p=1-__import__('math').exp(-1.5*(2+.5*(__import__('math').exp(-2.02/.5)-__import__('math').exp(-.02/.5))))
    sections=[]
    sections.append(f'''<section id="overview"><div class="section-kicker">01 / EXECUTIVE SUMMARY</div><h2>首先，这是一组机制诊断，<br>还不是模型能力排行榜。</h2>
<p class="lead">现有证据支持“LLM 可调用技能完成占点博弈闭环”；但不同模型的差异，同时经过规则理解、视觉定位、技能参数、执行约束和接口可靠性放大。最终分数只是这条链路的出口。</p>
<div class="finding-grid"><article><span class="finding-no">01</span><h3>全知也会误用规则</h3><p>助手 12 分、Qwen 10.4 分；Qwen 在 t=28–30 s 的记录暴露了对持续瞄准风险与冷却归属的误解。掌握状态不等于正确预测后果。</p></article><article><span class="finding-no">02</span><h3>视觉描述不等于可用定位</h3><p>视觉 Qwen 两档各有 3 个偏差较大的脚点；新 DS max 提交 2 个，投影偏差约 3.417 m / 0.091 m。高分不代表视觉定位始终准确。</p></article><article><span class="finding-no">03</span><h3>占点推进与收益相一致</h3><p>新 DS max 得 13.22 分、助手 12 分，均未命中对手；Qwen 关闭思考命中对手 2 次，却仍只有 4 分。这里只描述本局机制，不证明攻击没有价值。</p></article><article><span class="finding-no">04</span><h3>额度也是实验条件</h3><p>DS max 放宽额度后 20 次调用全部返回动作，其中 2 次输出超过旧 16K 上限；API 均值 43.48 s。旧中断移入历史，不能把新旧差异只归因于模型能力。</p></article></div>
<div class="notice"><strong>原位更新 · 2026-10-22</strong> V-DH 已替换为完整重测 <code>20261022-095614-313039</code>；原 <code>20261021-200759-757048</code> 以 DH16 保留在历史账本与失败案例中。新局从头运行，不是接续 28 s；主表、图表、截图、逐步 JSON 和下载数据均采用新局。<a href="data/revision.json">查看替换记录</a>。</div>
<div class="notice"><strong>阅读边界</strong> 每个主条件仅 1 局；7 个 LLM 条件均有完整终局，另列 1 个规则参考与 7 个历史适配/演进记录。新的 DS max 放宽了输出额度及总等待时限，与 low 并非纯思考档位消融。不做显著性检验、总体胜率、误差棒或统计排名。助手保留项目工程上下文，并非与 API 模型完全等价的盲测。</div>
{scoreboard}<p class="small">A = 四足；B = 人形。时长均为仿真时钟，不是等待模型的真实墙钟。无复活；人形死亡不会使比赛立即结束。模型名称和参数采用测试当天保存的 profile，不用今天的在线名称替换历史记录。</p>
{figure('fig01_outcomes',1,'分组结果与累计得分轨迹','a–b：每个点代表一局完整终点，非均值；新 DS max 为 13.22 分、39.26 s，旧中断不混入主图。c–d：0.02 s 状态轨迹；虚线为 20 s 坍塌。曲线只画到真实终局，不补齐至 40 s。规则/A*（8 分、21.86 s）使用更连续的几何控制，不能据此宣称 LLM 在相同输入/动作预算下优于决策树。重叠轨迹可能遮蔽颜色，数值见上表和下载数据。')}
</section>''')
    sections.append(f'''<section id="design"><div class="section-kicker">02 / EXPERIMENTAL DESIGN</div><h2>两代实验，两个不同的问题</h2>
<div class="protocol-grid"><article><span class="tag">COHORT O · 全知</span><h3>状态已知时，能否作出好决策？</h3><p>O-A / O-Q 获得双方当前状态、地图和规则，以及更丰富的对手策略信息。原始图像输入是导航示意图，不是实时 Isaac 机载照片。动作使用 MOVE / HOLD / AIM 等低层封装；Qwen 可多轮查询工具，携带较长会话历史。</p></article><article><span class="tag">COHORT V · 非全知 + SKILL</span><h3>原始画面能否变成有效动作？</h3><p>最近五轮知道地图、规则、对手初始位置，但没有持续对手真值。每 2 s 决策一次，接收间隔 0.5 s 的 Isaac 原始 RGB 帧、相机姿态和自身/公开状态；中间帧同样是证据。模型只选择语义技能并提供参数。</p></article></div>
<div class="pipeline" aria-label="实验闭环"><div><small>实际进程</small><b>Python 规则 + 运动学</b><span>推进 2 s，记录状态</span></div><i>→</i><div><small>视觉传感</small><b>Isaac 按姿态渲染</b><span>0.5 s / 帧，相机扫描</span></div><i>→</i><div><small>冻结仿真时间</small><b>LLM → 结构化技能</b><span>公开证据与简要解释</span></div><i>→</i><div><small>执行器</small><b>路径 / 风险 / 朝向编译</b><span>检查后执行，进入下一轮</span></div></div>
<p><strong>不是纯二维看图，也不是 Isaac 刚体动力学比赛：</strong>运动与裁判由 Python 运动学/几何规则驱动，Isaac 根据同一状态产生在线机载图像；第三视角回放是赛后渲染。接触力、足地作用、轮胎摩擦、真实跌倒、坍塌碎块动力学均没有通过本实验验证。坍塌在规则中是 20 s 的几何瞬时切换，回放另有动画展示。</p>
<h3>共同规则与已核验边界</h3>
{table(['项目','存档设置与解释'],[['赛制','40 s 上限，A/B 各 3 HP；四足 HP 归零即终止。只占点计分，击中不直接加分。'],['占点','目标半径 0.55 m；第一次进入开始 5 s 寿命；连续驻留 1 s 后以 1 分/s 计分。到期或点内受击使本实例失效并轮换；P1/P2 交替刷新。单次理想实例可得约 4 分。'],['命中','射程 8 m，正前总计 60°（±30°），代表点连线无墙体遮挡。连续有效瞄准增长条件命中风险；打中后只有射手冷却 2 s。'],['人形','仅几何可见观测 + 最近位置记忆 + 巡逻/搜索，不是全知追踪器。感知半角 60°（总计 120°）与开火半角 30°不同；人形搜索/巡逻扫描为 2.4 rad/s。'],['运动','A/B 最高平移速度 1.1 / 0.65 m/s；最高转速 2.5 / 3.0 rad/s。仍有速度、加速度、规划碰撞约束，但没有接触动力学。'],['场地','12 × 8 m 任务地面，固定墙体；High3 在 t=20 s 沿固定方向倒下。外观院落更大，不等于任务边界扩大。'],['相机','768 × 432，水平 FOV 90°；眼高 0.6 m、前偏置 0.36 m、俯角 5°；云台 ±90°，最大转速 120°/s。'],['视觉记忆','需要模型提交 sighting.time + feet 像素；平地投影得到估计位置。规划记忆最长 4 s，瞄准使用更严格的 2 s 新鲜度；初始位置不是可开火视觉锁定。'],['语义技能','MOVE_TO / CAPTURE_POINT / TAKE_COVER / AIM_AT_TRACK / OBSERVE。risk=safe/fast 是模型选项；拒绝危险路径可能返回 HOLD，而不是自动避弹。'],['共享随机性','规则 seed=2026102107；命中按每个角色独立随机流，在有效瞄准段抽取指数阈值。路径改变会改变瞄准段和随机数消耗，固定 seed 不等于固定命中时刻。']])}
<div class="notice">配置文件残留的 <code>policy.quadruped_knowledge</code> 描述属于原规则控制器，不能单凭这一字符串判定视觉组泄漏真值。实际边界以 <code>observations/*.json</code> 和保存的 planner context 为准；视觉决策输入没有当前 B 坐标、朝向、HP 或策略内部状态。赛后轨迹与误差计算使用真值，明确与当时输入分开。</div>
<h3>“最高 / 最低”并不是统一参数</h3>
{table(['条件','思考配置','预算 / 超时','额外差异'],[['O-Q','enable_thinking=true；thinking_budget=4096','max_tokens=8192；180 s','完整会话与工具查询，最多 3 次调用/决策；16 步共 22 次请求'],['V-DH（新完整重测）','thinking enabled；reasoning_effort=max','max_tokens=393216；无累计 token 上限；无总墙钟截止；600 s 网络无数据超时','采用测试当天平台最大输出额度，包含 reasoning；20 次调用均 stop，零重试'],['V-DL','thinking enabled；reasoning_effort=low','max_tokens=16384；180 s 总截止','与新 max 不止 effort 不同；旧同额度 max 已归历史 DH16'],['V-QH / V-QL','思考开启（省略 thinking_budget）/ 关闭','max_tokens=16384；600 s','存档将“max/min”记为实验标签，不发送等价的 DS effort 参数；Qwen 回答预算语义与 DS 不同'],['V-A / O-A','当前助手交互式规划','无可比 API effort / tokens 计量','记录外部等待时间；工程背景与历史熟悉度不可消除']])}
<p>“不加实验 token 上限”不等于服务端无限输出。新 max 使用平台最大 393,216 tokens（384K），而非省略参数；具体值以本轮 <a href="archive/V-DH/api_profile.json">profile</a> 和 <a href="https://api-docs.deepseek.com/api/create-chat-completion/">DeepSeek 官方接口文档</a>为据。max / low 的额度、超时与运行日期不同，无法把收益或时延差异单独归因于思考强度。</p>
<p>最近五轮均保留最近 4 次历史决策，要求四字段 JSON（observation_id、skill、params、assessment）；API 模型无自动策略兜底、无自动重试。Qwen 允许剥离单一 JSON 代码围栏，这是格式适配，不是替模型修改动作。不同厂商图像 detail/tokenizer 与输入编码仍可能不同。</p>
</section>''')
    sections.append(f'''<section id="oracle"><div class="section-kicker">03 / ORACLE COHORT</div><h2>全知组：状态完整，<br>但规则推演仍是关键。</h2>
<p>O-A 得到 12 分并坚持到 40 s，剩余 1 HP、人形 0 HP；O-Q 得到 10.4 分，在 31.64 s 四足清零、人形剩余 1 HP。两者均触发三轮占点。O-Q 的 API token 总量 565,031，主要来自长上下文与工具循环，不能与视觉组直接解释成“模型思考效率”的差异。</p>
<div class="photo-grid">{shot('O-A',28,'over')}{shot('O-Q',30,'over')}</div>
<article class="case"><span class="tag">CASE O-Q / t = 28–31.64 s</span><h3>当前瞄准值小，不代表未来两秒安全</h3>
{excerpt('O-Q',14)}<p>当时 B 的连续瞄准时间约 0.02 s。若未来 2 s 一直满足射程、角度与视线条件，规则给出的条件命中概率约为 <strong>{p:.1%}</strong>，不是 0.02 s 这一瞬间的风险值。此计算只是反事实风险示例，不等于声称实际随后两秒视线必然不变。</p>
{excerpt('O-Q',15)}<p>t=30 s 的解释又把“击中对手”当成会迫使对手冷却；真实规则是<strong>只有开火命中的一方冷却</strong>。日志中 A 在 30.06 s 击中 B，但 B 仍在 31.64 s 击中并淘汰 A。该错误可以由原始决策与事件交叉核验，不必猜测未保存的内部推理。</p>{raw('O-Q',14)}{raw('O-Q',15)}</article>
<h3>从本组能说什么，不能说什么</h3><p>可以说：这局助手更成功地完成了规则下的占点与攻击分配，Qwen 出现可定位的规则误用。不能说：助手显著优于 Qwen，或 LLM 显著优于规则/A*。基线 8 分、21.86 s 使用连续几何响应；与 2 s LLM 决策、工具查询权限并非严格匹配。本组也不是检验视觉能力的实验。</p>
</section>''')
    sections.append(f'''<section id="vision"><div class="section-kicker">04 / PARTIAL OBSERVABILITY</div><h2>非全知组：高分路径的共同点，<br>是持续转移，而不是持续对射。</h2>
<p>新 DS max 依次取得 P1 / P2 / P1 / P2 的 4 + 4 + 4 + 1.22 分，共 13.22 分，在 39.26 s 四足清零、人形剩 3 HP；视觉助手取得三个完整点实例共 12 分，在 32.94 s 被淘汰。DS low 与 Qwen 两档均只完成首个点实例，之后没有触发 P2，最后得分都为 4。新 max 的终局更高，但这里只是一局不同预算条件下的观察，不能宣称稳定排名。</p>
{mechanism}
<p class="small">暴露时间是赛后真值中“人形几何传感器看到 A”的持续时间，并非 A 的相机看到 B，也不是开火锥内时间；静止子集定义为 A 速度 &lt;0.05 m/s。积分按 0.02 s 记录右端状态计算。不同局长度不同，不能只按这些累计量排名。命中可能由技能编译后的自动攻击产生，不等于显式选择了 AIM。</p>
{figure('fig03_trajectories',3,'相同地图、不同路径与受击位置','a–e：五局 A（彩色实线）和 B（灰色虚线）的赛后真值轨迹，坐标、比例尺一致；方形为起点，圆点为终点，红色 × 为 A 受击；路径数字为 8/16/24/32 s 时刻。斜线区域表示 High3 坍塌后的地面占用，不表示整局墙体已倒下。f：图例。该图是分析可视化，不替代下方原始 Isaac 机载证据。')}
{figure('fig04_skill_execution',4,'意图、执行与受击的时间关系','a：每个色块是提交给技能层的命令，× 为 UNSAFE_ROUTE，实际执行退化为 HOLD；不是自动执行了躲藏。b：第一次进入后触发的点寿命窗口，标签为点名与该实例实际得分；新 DS max 的第四个点在受击时以 1.22 分结束。c：人形看到 A 的灰色时段、其中 A 静止的彩色子集，以及受击事件。虚线为固定 20 s 坍塌；各行终止线表示真实终局，横轴完整覆盖至 40 s。')}
<article class="case"><span class="tag">V-DH / COMPLETE RERUN</span><h3>持续推进到第四个点，但仍未摆脱追击</h3><div class="photo-grid">{shot('V-DH',28,'over')}{shot('V-DH',38,'over')}</div><p>20 步选择 17 次 CAPTURE_POINT、3 次 MOVE_TO，没有 UNSAFE_ROUTE。28 s 的解释提出提前去 P2、放弃 P1 尾分；但实际转向和执行期间仍累计完该点 4 分，所以不能将意图文字直接当成执行效果。四足在 9.14 / 30.84 / 39.26 s 受击，最后一击终止第四个点计分；模型在剩 1 HP 时仍优先争取增量得分，没有实现全程安全避敌。</p>{excerpt('V-DH',14)}{raw('V-DH',14)}{excerpt('V-DH',16)}{raw('V-DH',16)}</article>
<div class="two-col"><article><h3>任务对齐：击中多，不一定得分高</h3><p>V-QL 命中 B 两次，但首点后反复 AIM_AT_TRACK，最终 4 分；V-A 没有命中 B，却取得 12 分。这不是证明攻击无用，而是说明本规则中攻击应服务于移位与占点，不能用“命中数”代替任务收益。</p></article><article><h3>一个共同观察窗口</h3><p>截至所有五局都覆盖的 16.80 s，得分分别为：DS max 6.56、DS low 4、Qwen 开启 4、Qwen 关闭 4、助手 7.14。它减少终止时长差异，但仍是单局描述，且窗口由最短已观察局决定，不是预注册的主终点。</p></article></div>
<article class="case"><span class="tag">EXECUTION GAP</span><h3>“请求移动”与“已经逃脱”是两件事</h3><p>DS low 在 t=10/16/18 s 出现 3 次路径拒绝；Qwen 开启在 t=10/14/16/24 s 出现 4 次。技能层没有返回替代逃生动作，而是 HOLD。模型即使在文字里提出规避，仍可能停在受压位置。这里同时存在上层决策、视觉估计和下层失败语义的问题，不能全归咎于某个模型。</p>{excerpt('V-DL',8)}{raw('V-DL',8)}</article>
</section>''')
    sections.append(f'''<section id="perception"><div class="section-kicker">05 / VISUAL EVIDENCE</div><h2>把“看到了”拆开检验：<br>描述、脚点、记忆、执行。</h2>
<p>保存的 assessment 让“是否利用画面”有了可检查线索，但仅凭语言描述不能证明定位正确或内部推理充分。这里用原始 RGB、提交脚点、平地投影和执行结果建立可复核链条。原图不做生成式重绘，十字标记是独立 SVG 覆盖层。</p>
{figure('fig05_perception',5,'视觉陈述与可执行定位的差距','a：去重后的正向可见帧声明数与脚点提交数；它们不是同一分母，不能算作召回率。重复引用同一帧只记一次正向帧声明。b：全部 13 条脚点经已知相机外参/平地模型投影，与同帧 B 身体中心坐标的距离；对数轴。包含标点、标定、地平面及步态偏差，不是纯像素标注误差，也不是感知准确率。c：赛后总暴露与静止暴露子集。d：共同 16.80 s 窗口得分。所有点来自单次轨迹，不能把脚点或决策当成独立比赛重复。')}
<article class="case"><span class="tag">CASE 1 / DS max · NEW RUN</span><h3>两次脚点：一次落在前景，一次贴近可见脚部</h3><div class="photo-grid">{shot('V-DH',9.5,mark=True)}{shot('V-DH',28,mark=True)}</div><p>新 V-DH 有 {by['V-DH']['metric']['unique_positive_frame_claims']} 个不同帧的正向可见声明，提交 2 次 sighting。9.5 s 的 [389,315] 落在前景地面；人工复核可见人形脚部纵坐标约 225–230，投影与身体中心偏差 3.417 m。28 s 的 [160,340] 接近画面左侧可见鞋部，投影偏差约 0.091 m。这替代了旧局“零脚点”的分析，不能据两个样本估计总体感知准确率。</p>{excerpt('V-DH',5)}{raw('V-DH',5)}{raw('V-DH',14)}<p class="small"><a href="archive/V-DH/visual_spotcheck.json">逐图复核记录</a> · <a href="archive/V-DH/annotation_audit.json">投影误差计算</a>；两者均为赛后材料，不回传规划器。</p></article>
<article class="case"><span class="tag">CASE 2 / Qwen on vs off</span><h3>文字识别人形，脚点却落在地面或墙上</h3><div class="photo-grid">{shot('V-QH',9.5,mark=True)}{shot('V-QL',11.5,mark=True)}</div><p>Qwen 开启思考在 9.5 s 提交 [450,350]，与真实人形脚部约 [417,230] 不符；地面投影与身体中心偏差 3.545 m。关闭思考在 11.5 s 提交 [384,290]，实际脚部约 [444,243]，投影偏差 1.861 m。上述“约”的像素是复核读图，不是建立了完整人工标注集。</p><div class="photo-grid">{shot('V-QH',21.5,mark=True)}{shot('V-QH',22,'over')}</div><p>21.5 s 的 [320,240] 落在倒塌墙附近，真实人形在画面右侧；投影偏差 2.177 m。当前脚点接口没有目标检测置信度或重投影一致性门控，这会使错误定位进一步影响射击朝向、风险评估和路线拒绝。</p>{raw('V-QH',5)}{raw('V-QL',6)}{raw('V-QH',11)}</article>
<article class="case"><span class="tag">CASE 3 / DS low</span><h3>定位较准，也仍可能无法完成第二个点</h3><div class="photo-grid">{shot('V-DL',9.5,mark=True)}{shot('V-DL',18,mark=True)}</div><p>三次已提交脚点的投影偏差约 0.090、0.060、0.157 m，说明这局的定位不是主要失败证据；但 t=12 s 看见脚部未及时更新，以及后续路线拒绝、停留受击仍然存在。比起“看对了就能赢”，更准确的结论是：感知、记忆新鲜度、技能失败处理和任务推进缺一不可。</p>{raw('V-DL',6)}{raw('V-DL',9)}</article>
<article class="case"><span class="tag">CASE 4 / Assistant</span><h3>中间帧提供信息，但没有消除背后盲区</h3><div class="photo-grid">{shot('V-A',7.5,mark=True)}{shot('V-A',8)}</div><p>t=8 s 决策引用的是 7.5 s 中间帧脚点 [419,220]，而非只盯最后一帧；与身体中心的投影偏差约 0.043 m，t=10 s 第二次约 0.027 m。t=6 s 提前离开首点，8/10 s 继续向 High3 北侧推进；18 s 离开即将坍塌的区域。26/28 s 在点内 OBSERVE 时仍被动累计分数，说明技能名不直接等同计分状态。</p><div class="photo-grid">{shot('V-A',26,'over')}{shot('V-A',30)}</div><p>最后仍在 29.90 s 和 32.94 s 受击。第三轮点结束后的人形追击与后方视野不足，是这局没有活到时间上限的重要失效情景；不能因为得分较高就忽略闭环的脆弱性。</p>{excerpt('V-A',4)}{raw('V-A',4)}{raw('V-A',13)}{raw('V-A',15)}</article>
</section>''')
    sections.append(f'''<section id="resources"><div class="section-kicker">06 / LATENCY & RELIABILITY</div><h2>速度要测端到端，<br>可靠性也必须计入成本。</h2>
{latency}<p class="small">外部等待沿用原始计时字段：视觉组 decision_wait_seconds 从观测就绪计至动作验证后、技能编译前；全知旧接口 decision_wall_seconds 包含编译/检查开销，计时边界并不完全相同。完整就绪到接受还需计入技能编译；保存了 accepted UTC 的视觉轮次可据此核验。API 时间是调用开始至结束，含网络、排队、推理与生成，不是纯“思考耗时”。O-Q 每步 API 时间为该步多次请求之和。P95 使用 nearest-rank；小样本不代表总体尾延迟。助手 tokens 缺失，不记成零。此表只计主分析新局，DH16 的历史费用在下方案例单列。</p>
{figure('fig02_latency_tokens',2,'逐决策墙钟延迟、首输出与资源账本','a：所有被接受决策的外部等待，短竖线为该局中位数，红色虚线是 2 s 控制周期；不是独立重复试验的误差范围。b：首流输出与完整 API 响应；首输出可能是 reasoning token，不代表动作 JSON 已可执行；新 DS max 最长 147.33 s，均成功返回。c：主分析 provider usage 累加，reasoning 是 completion 的子集，不重复相加；V-QL 没有 reasoning_tokens 字段，显示的其余 completion 不证明其内部计算为零。d：技能编译耗时，未包括 Isaac 渲染。助手无 API token 计量，留空而不估算。')}
<div class="two-col"><article><h3>时停闭环 ≠ 端侧实时</h3><p>{decision_count} 次被接受的主分析决策中，没有一次外部等待 ≤2 s。视觉技能编译均值约 {min(r['metric']['compile_mean'] for r in v):.3f}–{max(r['metric']['compile_mean'] for r in v):.3f} s，真正的大头仍在高层调用。当前时停使机器人不会在模型等待时暴露，这证明的是离线/暂停式规划可行性，不证明 5070、Spark 或真实机器人实时部署可行。</p></article><article><h3>“首 token 很快”不是“动作很快”</h3><p>流式第一段内容可能很早出现，但完整结构化动作需等待后续输出。Qwen 关闭思考的 API 均值 9.09 s，约为开启的 1/5.23；这仅是这两局服务响应的描述比值，不能剥离网络、上下文长度和服务负载影响。</p></article></div>
<article class="case"><span class="tag">V-DH / NO EXPERIMENT TOKEN CAP</span><h3>放宽额度后：20 次完整动作，代价仍是长等待</h3><p>新局 API 均值 43.48 s、中位 36.15 s、总计 869.58 s；总 255,034 tokens，其中 prompt 98,692、completion 156,342，reasoning 147,451 已包含在 completion 中。零失败、零重试，20 次均 finish_reason=stop。decision 003（t=6 s）输出 17,520 tokens，decision 014（t=28 s）输出 26,319 tokens，均超过旧 16,384 上限。后者等待 147.33 s，其中约 145.29 s 才开始可见内容。</p><p>这直接说明新局有两次完成所需输出超过旧额度；但它是从头独立运行，不是给原失败请求续写。不能声称两局分差全由额度造成，更不能由单局正常结束推断长期无故障。<a href="archive/V-DH/budget_rerun_comparison.json">查看新旧参数与原始记录对照</a>。</p></article>
<article class="case"><span class="tag">HISTORICAL FAILURE / DH16 · EXCLUDED FROM PRIMARY</span><h3>原 16K max：第 15 次有输出流，却没有动作</h3><p>这段属于被替换的旧局，不是上方新 V-DH。旧局前 14 次成功 API 均值 35.37 s；含失败的 15 次均值 38.97 s。失败请求耗时 89.42 s，finish_reason=length，content 为空，completion_tokens=reasoning_tokens=16,384。该次总 tokens 为 21,430；整局含失败共 181,049 tokens，完整保留在历史成本中，不与新局 255,034 合并。旧局停在 28 s、10.56 分、HP [2,3]，不是终局成绩。</p><p>旧失败仍是系统可靠性证据，没有因本次替换而删除。存档未保存 reasoning 文本，只保存计数和简要 assessment，不补写隐藏推理。<a href="archive/DH16/api_calls/014_response.json">原始失败响应下载</a>。</p><details><summary>旧局失败响应原始记录</summary><pre>{esc(json.dumps(historical_max['failed_response'],ensure_ascii=False,indent=2))}</pre></details></article>
{figure('fig06_mechanisms',6,'持续瞄准风险与输出额度复核','a：由已知规则解析计算，λ(a)=1.5[1−exp(−a/τ)]，持续有效瞄准至 t 的累计命中概率为 1−exp{−1.5[t−τ(1−exp(−t/τ))]}；τA=1 s、τB=0.5 s。曲线不是实验拟合，也不含遮挡中断、冷却或运动。b：新 DS max 20 次请求的 completion tokens（含 reasoning）；请求编号从 1 开始，分别对应 decision 000–019。虚线是旧局额度，不是新局上限；第 4 / 15 次超过旧额度且成功 stop。不能从此推断相同输入在旧额度下必然产生完全相同的输出轨迹。')}
<p>token 总量也不宜直接当作金钱或能耗：厂商 tokenizer、图像编码与缓存计量不同；旧 Qwen 保留长历史并调用工具，视觉组只保留 4 次决策。这里不给出未经账单验证的费用或本地功耗换算。</p>
</section>''')
    sections.append(f'''<section id="explorer"><div class="section-kicker">07 / FULL DECISION ATLAS</div><h2>逐步核验：原始图片 × 原始决策</h2><p>包含全部 {decision_count} 次已接受决策，新 V-DH 为 20 步；不仅是上方选取的案例。默认显示最后一张输入帧；点击缩略图查看中间帧。俯视画面供赛后解释，明确标注真实时间；没有精确对应帧时采用最邻近已渲染状态，不伪装成同一时刻。</p>
<div class="viewer"><div class="viewer-toolbar"><label>实验条件<select id="run-select"></select></label><label>决策点<select id="step-select"></select></label><button id="previous" aria-label="上一步">← 上一步</button><button id="next">下一步 →</button></div><div id="step-summary" class="step-summary"></div>
<div class="viewer-grid"><div><h4 id="input-title">决策时输入</h4><div id="frame-display" class="frame-display"></div><div id="frame-thumbnails" class="thumbnails"></div><div id="frame-caption" class="small"></div><label class="overlay-control"><input type="checkbox" id="marker-toggle" checked> 显示该步实际提交的脚点（若与所选帧匹配）</label><h4>赛后第三视角</h4><div id="overhead-display"></div></div><div class="records"><div role="tablist" class="tabs"><button role="tab" data-tab="decision" aria-selected="true">原始决策 JSON</button><button role="tab" data-tab="compiled" aria-selected="false">技能执行</button><button role="tab" data-tab="observation" aria-selected="false">公开观测</button><button role="tab" data-tab="timing" aria-selected="false">计时 / usage</button></div><div id="record-note" class="small"></div><pre id="record-json" tabindex="0"></pre><div id="step-outcome" class="notice"></div></div></div>
<details class="replay"><summary>展开本局 GIF / 视频与赛后关键帧</summary><div id="replay-content"></div></details></div>
<p class="small">“思考输出”在这里指当时留存的 evidence / choice_summary / assessment 等可审计简要解释，不是模型隐藏推理链。第三视角使用全局真值，不能反向算作非全知规划器当时可用的信息。</p></section>''')
    sections.append(f'''<section id="limitations"><div class="section-kicker">08 / INTERPRETATION & NEXT EXPERIMENTS</div><h2>下一步，要把耦合问题拆开。</h2>
{table(['结论层级','本次可支持的表述','当前不支持的表述'],[['直接观察','具体得分、受击、路线、等待、原始脚点、JSON 缺失等可由日志复核。','“某模型总体最强”“高思考一定更好/更差”。'],['机制解释','规则误用、错误脚点、HOLD 退化与不推进任务在记录上形成一致链条。','在没有受控消融时，把全部得分差异归因于某一个因素。'],['泛化','同一地图/规则/种子的可行性与失效样本。','对新地图、不同人形策略、真机、端侧芯片或动力学扰动的成功率。'],['感知',f'{len(d["annotations"])} 条提交脚点的投影偏差；案例图可人工复核。','检测 precision/recall、全帧漏检率、遮挡鲁棒性总体结论。'],['工程可靠性','新 DS max 20 次均正常返回；旧 DH16 预算截断、历史断流与格式失败仍保留。','用单局完赛估计稳定故障率；隐藏失败或把事后放宽预算视为严格消融。']])}
<h3>最重要的混杂因素</h3><p>每条件 n=1；固定 seed 但轨迹不同导致瞄准事件和随机数消耗不同；助手保留历史背景、API 模型从提示词接入；全知组与视觉组的技能接口、工具权限和历史长度不同；厂商 effort、token 预算和图像处理语义不同。尤其新 DS max 同时放宽输出额度、取消总等待截止，并在次日重跑，不能当作与 low 仅差一个 effort 参数的对照；滚动模型别名也不保证日期间服务实现完全相同。以事后重测替换中断主项会改变纳入机制，因此本报告显式保留旧局、旧费用与替换记录，不把两个 max 当成同配置重复或拼接轨迹。逐步等待与脚点数据在同一局内高度相关；像素投影真值比较含标定/步态误差。所有这些限制都先于“谁比谁强”的推断。</p>
<h3>建议的分阶段验证矩阵（本报告未执行）</h3>
{table(['阶段','只改变什么','测量与验收'],[['A · 冻结图片回放','同一组 RGB、相机姿态、输出 schema，替换模型/思考模式；关闭运动反馈。','人工标注可见性与脚部位置；按场景/序列分组统计检测、定位、漏更新和合法 JSON。先解决“看图→坐标”。'],['B · 固定状态技能决策','统一全知输入或统一外部感知轨迹，保持同一语义 SKILL 与相同背景提示。','测规则测验、占点推进、拒绝后的重规划、技能参数与动作延迟；消除视觉能力差异。'],['C · 同接口闭环配对','地图、对手、种子列表、episode 上限、预算和失败处理预先固定；随机化条件运行顺序。','例如先规划 20 个配对种子作为探索批次，再据变异确定正式样本量；报告每局散点/配对差值及 episode 层面的区间，而非把决策当重复。'],['D · 单因素消融','脚点一致性门控；UNSAFE_ROUTE 逃生备选；后视感知；任务优先级提示；解释开关逐一切换。','每次只改一项并冻结其他参数。上一代“无解释 DS low”同时改变预算/超时，不可作该消融结论。'],['E · 异步与端侧','在相同策略下取消时停、引入过期动作与状态漂移，再在目标硬件实际部署。','测端到端 P50/P95、超时/缺动作率、热/功耗、内存、任务收益及碰撞。随后才加入 Isaac 接触动力学或真机安全测试。']])}
<p><strong>优先工程假设：</strong>先增加可观测的技能失败返回与重规划路径，给脚点更新加几何一致性检查，并提供前后视信息；再比较更大的模型或更高思考预算。这是基于本次失效样本的测试优先级建议，不是已验证的性能提升。</p>
</section>''')
    sections.append(f'''<section id="provenance"><div class="section-kicker">09 / METHODS, PROVENANCE & DOWNLOADS</div><h2>每一个数，都应能回到原记录。</h2>
<h3>纳入目录与历史失败账本</h3>{registry}<p class="small">上表中的原报告链接依赖本机原实验目录；其余本报告图表、图片、日志浏览和下载在离线包内自包含。历史接入失败、断流与无解释版本只作演进账本，不混入主表的效能统计。</p>
<h3>共享代码审计</h3>{audit}<p>最近五轮的规则、背景说明、语义技能、视觉控制与自适应运动脚本哈希一致。该事实证明这些文件相同，不证明所有厂商适配器、模型内部状态、网络环境和助手上下文相同。配置哈希在全部七个主条件中一致。所有源文件哈希与图像来源见 manifest。</p>
<h3>计算口径</h3><p>从原始 timeline（步长 0.02 s）重算终点分数、HP、路径弧长、可见/静止暴露、占点窗口和共同 16.8 s 分数；从 decisions 重算逐步等待、API 总时长和技能命令；tokens 优先采用 provider usage。主分析现为 7 个完整终局、{decision_count} 个已接受决策；旧 DH16 的 14 个动作与失败请求只留历史，不计入主分析均值、分布和 token 总量。其 10.56 分仅为中断观察值，无终局成绩。缺失字段为 null / 空白，而非零。逐决策 API 时间包含工具多次调用时求和；历史失败请求虽不计作动作，费用仍在独立账本保留。</p>
<p>原始机载 PNG 按字节复制，不调色、不裁掉不利证据。旧全知俯视图沿用存档截图；新 DS max、Qwen 两档与助手视觉组从已有 Isaac 视频按 frame_index 精确映射解码，计入片头/坍塌插播偏移，并校验源 timeline SHA-256。DS low 仍用上一版为保存状态补渲染的 6 帧；旧 max 的 7 帧留在 native_keyframes 历史资源中，不再作为新局证据。本次原位更新没有新渲染、没有推进比赛或调用决策模型。新 max 的展示视频长约 47.42 s，含片头/片尾及坍塌展示，不等于比赛 39.26 s；报告统一以仿真时钟分析。</p>
<div class="downloads"><a href="data/metrics.csv" download>主结果 CSV</a><a href="data/decisions.csv" download>{decision_count} 步决策指标 CSV</a><a href="data/annotation_errors.csv" download>脚点偏差 CSV</a><a href="data/target_episodes.csv" download>占点实例 CSV</a><a href="data/analysis.json" download>完整派生数据 JSON</a><a href="data/source_manifest.json" download>来源与 SHA-256</a><a href="data/revision.json" download>本次替换记录</a><a href="data/validation.json" download>验证结果</a><a href="code/research_data.py" download>分析脚本</a><a href="code/research_figures.py" download>制图脚本</a><a href="code/research_report.py" download>报告脚本</a></div>
<h3>科研制图说明与参考</h3><p>采用多 panel 叙事、统一坐标/色彩、独立图注、单局原始点、文字可编辑 SVG、矢量 PDF 和 600 dpi PNG。借鉴 Nature / Science 科研图的表达方式，不使用期刊标识，也不声称通过其投稿规范审核。可核验的具体设计参考来自 Nature 官方指南；Science 主刊作者指南本次访问受限，未据此宣称满足其专属格式。</p>
<ol class="references"><li><a href="https://research-figure-guide.nature.com/">Nature Research Figure Guide</a>：清晰、可读、可访问的科研图设计。</li><li><a href="https://www.nature.com/documents/ncomms-formatting-instructions.pdf">Nature Communications formatting instructions</a>：多 panel 标识、独立图注、符号定义与图形准备。</li><li><a href="https://www.nature.com/documents/natrev-artworkguide_PS.pdf">Nature Reviews artwork guide</a>：逻辑组织、统一视觉语言。</li><li><a href="https://www.nature.com/articles/492180a">Nature: Know when your numbers are significant</a>：重复、误差与统计表达的边界。</li><li><a href="https://www.science.org/content/page/instructions-preparing-initial-manuscript">Science initial manuscript instructions</a>：本次服务器拒绝访问，列作来源入口，不据此转述未核验条款。</li></ol>
<p class="small">整理及原位更新日期：2026-10-22（Asia/Shanghai）。六个主条件来自 2026-10-21，新 DS max 来自 2026-10-22；本次只是分析已完成测试，没有新比赛、API 模型请求或渲染。旧原始实验文件不改写。</p></section>''')

    viewer=[]
    for r in runs:
        viewer.append({k:r[k] for k in ['id','label','cohort','metric','steps','overhead','media','profile']})
    nav=[('overview','结论'),('design','设计'),('oracle','全知'),('vision','非全知'),('perception','视觉证据'),('resources','时延资源'),('explorer','逐步核验'),('limitations','下一步'),('provenance','数据方法')]
    template=Path(__file__).with_name('research_template.html').read_text()
    output=template.replace('<!--NAV-->', ''.join(f'<a href="#{k}">{v}</a>' for k,v in nav)).replace('<!--CONTENT-->','\n'.join(sections)).replace('/*DATA*/',json.dumps(viewer,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    output=output.replace('<!--HERO-->',shot('V-A',16,'over')).replace('<!--DECISIONS-->',str(decision_count))
    (OUT/'index.html').write_text(output)
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['research_data.py','research_keyframes.py','research_figures.py','research_report.py','research_template.html','research_validate.py','check_research_report.mjs','semantic_skills.py','adaptive_motion.py','tactical_logic.py','trial_motion.py','trial_rules.py']:
        source=Path(__file__).with_name(name)
        if source.resolve()!=(code/name).resolve():shutil.copy2(source,code/name)
    print(f'Report: {OUT}/index.html; {sum(len(r["steps"]) for r in runs)} accepted decisions.')

if __name__=='__main__':build()
