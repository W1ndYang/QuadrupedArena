"""Prepare/probe/run one API-planned R0 trial; no autonomous retries or fallback."""
import argparse
import base64
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time
import numpy as np

from adaptive_motion import HumanPolicy,load_config
from llm_trial import Match,PROJECT,dump,save,export,observation_packet
from qwen_client import APIError,KEY_PATH,call_api,endpoint,load_key,setup_key,usage_total
from tactical_logic import block
from trial_motion import path_length

CONFIG=PROJECT/'configs/qwen_trial.json'
from arena_paths import workspace_root
RUNS=workspace_root()/'experiments/runs/quadruped-arena'
PREP=RUNS/'20261021_Qwen3.6-27B_测试准备'
PROTOCOL_VERSION='qwen-r2-context-budget'
BACKGROUND='''【项目背景与本轮职责】
我们在 Isaac 场景中研究“基座大语言模型做顶层意图规划，调用打包好的代码技能执行运动”的机器人自主决策架构，未来希望在有限端侧算力上部署。本轮先以云端 Qwen 验证规划闭环，不训练模型，也不测试真实机器人。你不是来解释如何搭建比赛，而是正在控制一场比赛中的 A，需要逐步提交可执行指令。
这是虚拟占点对抗：A 是你控制的轮足四足机器人，B 是固定决策树控制的人形施压对手。唯一成绩是 A 的累计占点分。攻击没有直接分数，但会改变后续安全占点机会；仅躲避存活但不占点也不会得分。由你自主权衡路线、遮挡、驻留收益、攻击、撤离和时间，不提供预设获胜动作序列。
当前试验是 R0 简化模型：CPU 裁判和运动学执行器推进状态，Isaac 用于赛后原生渲染。没有真实接触动力学、腿部步态、弹道或沙石摩擦建模。视觉装饰边界不等于合法运动边界，以规则 JSON 的 arena 和 covers 为准。
你没有项目历史对话，也不应假设知道以前试验的策略或成绩。此 system 消息、当前观测、当前导航图和本轮已有记录是你的完整任务上下文。全局地图、当前 B 的真实位置/朝向/状态及其策略对你可见；B 并不全知，只按自己的视野、遮挡、记忆和搜索逻辑行动。你知道 B 在哪里，不等于 B 已发现 A。不能把位置全知理解为能预知随机命中。

【交互流程】
收到 observation_id=n 的观测时，比赛已暂停。你可直接提交一个技能，或先查询当前几何。工具返回 role=tool 的结果就是已完成查询的答案，不是让你原样再次调用的指令；相同静态问题无需重复查询。调用工具不会移动机器人，也不会改变比赛时钟。每个决策最多两轮工具，第三次模型请求将禁用工具并必须给最终动作。
最终 content 中的六字段 JSON 才会被执行：校验通过→存档→运行最多 2 秒→返回新 observation_id。工具调用本身不是运动命令。下一次决策用新技能替换旧技能；远目标可能需要多个周期继续选择 MOVE，没有必须在单个周期到达的要求。到达目标后不自动选下一占点。两秒执行窗口内没有助手代你重新规划，受击或敌人移动的风险仍然存在。比赛内 2 秒与 API 等待的现实秒数不同，等待不消耗比赛时间。

【观测字段和坐标】
所有双元素演员数组按 [A, B] 排序；xy 是平面米坐标，yaw 是弧度，0 朝 +x，正角向 +y。图像坐标不是可直接提交的世界坐标。state.t 是比赛秒数，hp 是生命值，score 是 A 总分，aim 是连续有效瞄准秒数，cooldown 是剩余冷却秒数。
selected/active_target 与目标索引、eligible、target_generation、target_remaining、驻留状态应结合 task_points 和完整状态读取；P1/P2 的显示编号与零起始索引区分。target_remaining=null 表示尚未触发倒计时，不表示已失效。目标坐标固定，当前有效目标循环刷新，不是两个点同时计分。
geometry.representative_point_los 只表示当前两代表点间是否无遮挡；aim_geometry 分别给出双方当前射击几何条件。human_visible 是 B 是否当前看到 A，human_memory 是 B 最后所见位置，不是你自己的传感器。human_policy.route 是 B 当前计划而不是保证不变的未来轨迹。recent_events 是已发生事件，不是预测。
probe_geometry 返回从 A 当前坐标到候选点的可达性、路径长度和路线；current_los_to_b 只检查候选终点对 B 当前所在位置的视线，不证明沿路安全，也不预测 B 移动或未来命中。reachable=true 不表示值得选择；path_length=0 可表示已经在该坐标，不是工具失败。

【输出约束与评估】
你的目标是在剩余比赛时间内最大化最终得分，同时考虑 A 淘汰导致无法继续得分的风险。预期只是预测，下一次应依据真实新观测调整。解释字段请给简明、可核验的依据与取舍摘要，不需要内部思维链全文。不要请求用户确认、不要给搭建教程、不要输出整场脚本、不要假称已移动或已命中。只选择白名单技能，裁判负责执行与随机判定。
'''
PROMPT='''你是虚拟机器人占点比赛中的四足 A 的高层规划器。目标是最大化占点得分，不是最大化攻击次数。仅模拟，不操作真实机器人。
你每次选择一个固定技能，裁判执行最多 2 秒再返回观测；你生成回复或调用几何工具时，全场状态与计时冻结。
输入包括当前完整状态、环境地图、对手策略源码及规则。不得假设未来随机命中结果，不得要求修改规则、速度上限或对手策略。
可选工具 probe_geometry 仅检查当前几何路径/距离/视线，不试跑未来。至多使用两轮工具，再提交最终动作。
固定技能 MOVE/HOLD/AIM/SCAN/STOP。MOVE 必须提供 goal=[x,y]，可选 via（最多7个中间点）、reverse、speed_limit。
MOVE 使用几何 A*，不会自动识别战术风险或躲避；facing 为 keep、enemy 或弧度数值。路径跟随中的机身朝向受运动器约束，facing 不是独立炮塔；到达后才按指定朝向。reverse 允许倒退但不保证全程对准敌人，倒退上限0.8m/s。
HOLD 减速停止，保持指定朝向；AIM 停止并按转速上限朝向当前敌人；SCAN 停止并固定缓慢转向；STOP 停止路径。除 MOVE 外不得提供 goal/via。
可选 attack 布尔值，默认 true；瞄准有效后按原随机裁判自动攻击。命中仅触发命中方冷却，受击方仍可还击；命中不直接加分。
可选 target_generation 绑定当前目标代次，刷新后取消旧技能；省略允许跨刷新执行。几何安全失败只停止，不会自动替你选别的策略。
初始双方3 HP；四足清零终局，人形清零不终局。40秒上限。单点轮换，首次进入后5秒失效；连续驻留1秒才开始每秒1分，离圈重置驻留但不重置5秒截止时间。圈内被击中提前使当前点失效并刷新下一点；历史得分保留。
有效瞄准要求正前总计60度、距离不超过8米、代表点之间无遮挡；连续有效瞄准越久条件命中率越高，人形增长更快，失效时重新累积。20秒一次固定坍塌，导航一直保留坍塌扫掠区域。具体数值与实际状态以输入规则和观测为准。
命中强度为 lambda(a)=lambda_max*(1-exp(-a/tau_i))，a 是连续有效瞄准时间；它是每秒条件强度，不是一个固定必中的瞄准秒数。小时间窗命中概率约为 1-exp(-lambda(a)*dt)，实际裁判按累计强度积分采样。命中减对方 1 HP，命中方冷却 2 秒；不中不触发冷却。双方可同一步互相命中，A 淘汰即结束，不复活；B 淘汰后 A 仍可继续占点到时间上限。每个目标代次只有第一次进入触发五秒期限，失效后按 P1→P2→P1 循环刷新新代次。
必须直接输出一个 JSON 对象（不要 Markdown、代码围栏或其他前后文字），恰好包含以下字段：
{"observation_id":当前整数编号,"action":{"skill":"HOLD","facing":"keep","attack":true},"evidence":"当前可核验事实","choice_summary":"简明决策依据","alternatives_summary":"备选及取舍摘要","expected_result":"接下来2秒的预期与风险"}
上述 action 只是格式示例，不是策略推荐。说明使用中文摘要，不要求展开内部推理全文。只能提交白名单动作，不得输出或执行任意代码。
'''
TOOLS=[dict(type='function',function=dict(name='probe_geometry',description='当前时刻的几何可达性、路线长度、路线及对人形视线；不推进比赛，不预测敌人。',
    parameters=dict(type='object',properties=dict(points=dict(type='array',minItems=1,maxItems=8,items=dict(type='array',minItems=2,maxItems=2,items=dict(type='number')))),required=['points'],additionalProperties=False)))]


def settings(path=CONFIG):
    config=json.loads(Path(path).read_text());endpoint(config)
    if config['decision_period_seconds']!=2 or config['automatic_retries']!=0 or config['automatic_strategy_fallback']:
        raise ValueError('First API trial requires 2s cadence, no retries and no strategy fallback')
    if config['model']!='qwen3.6-27b':raise ValueError('This trial is pinned to qwen3.6-27b; model changes require a separate profile')
    if config['max_calls_per_decision']!=3 or config['max_calls_per_match']!=60:raise ValueError('Unexpected API call budget')
    return config


def game_config(config):
    cfg=load_config(PROJECT/'configs'/config['rules_config']);cfg['seed']=config['match_seed'];return cfg


def system_message(cfg):
    rules={k:v for k,v in cfg.items() if k not in ('seed','policy','fps','resolution','render_spp','intro_seconds','outro_seconds','playback_speed')}
    rules['human_parameters']={k:v for k,v in cfg['policy'].items() if k.startswith('human_')}
    return dict(role='system',content=BACKGROUND+'\n【比赛规则与技能接口】\n'+PROMPT+'\n规则与地图：\n'+json.dumps(rules,ensure_ascii=False)+'\n人形固定策略（只作规则说明，不可修改）：\n'+inspect.getsource(HumanPolicy))


def request_context(messages,index,attempt,config):
    remaining=config['max_calls_per_decision']-1-attempt
    note=f'\n【当前调用预算】observation_id={index}；本步第 {attempt+1} 次模型请求；剩余工具轮数 {remaining}。'
    note+=('工具已禁用。基于已获得的观测与工具结果，现在必须提交六字段最终动作 JSON。' if remaining==0 else '信息足够即可直接提交最终动作；不要重复已完成的相同几何查询。')
    current=[dict(messages[0],content=messages[0]['content']+note),*messages[1:]]
    return current,dict(config,tool_choice='none' if remaining==0 else 'auto')


def observation_message(packet,include_map):
    text=json.dumps(packet['observation'],ensure_ascii=False)
    if not include_map:return dict(role='user',content=text)
    encoded=base64.b64encode(Path(packet['image']).read_bytes()).decode()
    return dict(role='user',content=[dict(type='text',text=text+'\n附件是当前状态的程序绘制导航示意图，不是原生相机图像。'),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+encoded))])


def probe(match,points):
    if not isinstance(points,list) or not 1<=len(points)<=8:raise ValueError('points must contain 1..8 positions')
    result=[]
    for point in points:
        p=np.asarray(point,dtype=float)
        if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Invalid geometric point')
        route=match.planner.route(match.a.xy,p)
        result.append(dict(point=p.tolist(),reachable=route is not None,
            path_length=None if route is None else path_length(match.a.xy,route),
            route=None if route is None else [v.tolist() for v in route],
            current_los_to_b=block((*p,.4),(*match.b.xy,1.05),match.planner.covers) is None))
    return result


def parse_action(content,match):
    # No eval, regex salvage, hidden action edits or strategy repair.
    request=json.loads(content)
    fields={'observation_id','action','evidence','choice_summary','alternatives_summary','expected_result'}
    if not isinstance(request,dict) or set(request)!=fields:raise ValueError('Response must contain exactly the six required fields')
    if type(request['observation_id']) is not int or request['observation_id']!=len(match.decisions):raise ValueError('Stale/wrong observation_id')
    for field in fields-{'observation_id','action'}:
        if not isinstance(request[field],str) or not request[field].strip():raise ValueError('Missing explanation '+field)
    return request,match.compile(request['action'])


def exclusive(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def export_api_match(match,directory):
    export(match,directory)
    validation=json.loads((directory/'validation.json').read_text())
    values=[entry.get('tokens') for entry in match.decisions]
    complete=bool(values) and all(type(v) is int for v in values)
    validation['token_usage']=sum(values) if complete else None
    validation['token_usage_status']='provider_reported' if complete else 'not_yet_reported_or_partial'
    dump(directory/'validation.json',validation)


def prepare(config,directory):
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'observations').mkdir();cfg=game_config(config);match=Match(cfg)
    packet=observation_packet(match,directory)
    dump(directory/'api_profile.json',config);dump(directory/'trial_config.json',cfg)
    dump(directory/'sample_messages.json',[system_message(cfg),observation_message(packet,config['include_observation_map'])])
    dump(directory/'tool_schema.json',TOOLS)
    dump(directory/'preparation.json',dict(status='prepared_not_run',api_requests_sent=0,simulation_seconds=0,
        credentials_saved_here=False,api_key_path=str(KEY_PATH),network_probe_required=True,
        max_calls_per_match=config['max_calls_per_match'],max_output_tokens_per_call=config['max_tokens'],
        input='current state JSON + program-drawn current map; no historical replay screenshots',
        reference='same rules/seed/skills; provider prompt/history and image consumption differ from interactive assistant'))
    print('测试准备完成（无 API 请求、无比赛执行）：'+str(directory))


def network_probe(config,key):
    # Explicit paid smoke test, separate from the competition and its seed.
    small=dict(config,max_tokens=1024,thinking_budget=256)
    response=call_api(small,key,[dict(role='user',content='仅输出 JSON：{"ok":true}。不要添加其他文字。')])
    if response['finish_reason']!='stop' or json.loads(response['content'])!={'ok':True}:raise ValueError('Probe response was not the expected complete JSON')
    directory=RUNS/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'_Qwen_API连通检查');directory.mkdir()
    dump(directory/'probe.json',response)
    print(json.dumps(dict(status='passed',requested_model=config['model'],response_model=response['response_model'],
        api_wall_seconds=response['api_wall_seconds'],usage=response['usage'],record=str(directory)),ensure_ascii=False))


def run(config,directory,key,client=call_api):
    directory.mkdir(parents=True,exist_ok=False)
    for name in ('observations','requests','decisions','api_calls','source_snapshot'):(directory/name).mkdir()
    cfg=game_config(config);match=Match(cfg);messages=[system_message(cfg)];total_calls=0
    (directory/'planner_context.txt').write_text(messages[0]['content'])
    for folder in ('scripts','configs'):
        target=directory/'source_snapshot'/folder;target.mkdir()
        for file in (PROJECT/folder).glob('*.py' if folder=='scripts' else '*.json'):shutil.copyfile(file,target/file.name)
    dump(directory/'api_profile.json',config);dump(directory/'protocol.json',dict(planner=config['model'],controller_type='external_api',
        decision_period_sim_seconds=2,seed=cfg['seed'],rules='v0.4',first_run_only=True,no_rewinds=True,
        protocol_version=PROTOCOL_VERSION,tool_budget_enforcement='auto, auto, none; current budget appended to system context',
        prompt_sha256=hashlib.sha256(messages[0]['content'].encode()).hexdigest(),
        automatic_retries=0,automatic_strategy_fallback=False,request_budget=config['max_calls_per_match'],
        token_status='provider usage if returned; null otherwise',input_map=config['include_observation_map'],
        timing='end-to-end decision wait plus per-call network wall time/first-output time; not pure inference',
        limitations=['Full current state and known HumanPolicy; no future random state',
            'Original assistant prompt/history not an identical API scaffold; not a controlled model-only comparison',
            'Current map images may be consumed differently; native screenshots are post-hoc only']))
    save(match,directory);export_api_match(match,directory);packet=observation_packet(match,directory)
    try:
        while not match.ref.finished:
            index=len(match.decisions);started=time.perf_counter();messages.append(observation_message(packet,config['include_observation_map']));records=[]
            for attempt in range(config['max_calls_per_decision']):
                if total_calls>=config['max_calls_per_match']:raise ValueError('Match API request budget exhausted')
                stem=f'{index:03}_{attempt:02}';total_calls+=1
                current,call_config=request_context(messages,index,attempt,config)
                exclusive(directory/'api_calls'/(stem+'_request.json'),dict(messages=current,tools=TOOLS,tool_choice=call_config['tool_choice'],profile=call_config))
                response=client(call_config,key,current,TOOLS);records.append(response)
                exclusive(directory/'api_calls'/(stem+'_response.json'),response)
                print(json.dumps(dict(api_call=stem,finish=response['finish_reason'],tool_choice=call_config['tool_choice'],api_seconds=response['api_wall_seconds']),ensure_ascii=False),flush=True)
                if response['tool_calls']:
                    if response['finish_reason']!='tool_calls':raise ValueError('Unexpected incomplete tool response')
                    if attempt>=config['max_calls_per_decision']-1:raise ValueError('Geometry tool call budget exhausted')
                    messages.append(dict(role='assistant',content=response['content'] or None,tool_calls=response['tool_calls']))
                    for call in response['tool_calls']:
                        if call['function']['name']!='probe_geometry':raise ValueError('Unknown tool requested')
                        args=json.loads(call['function']['arguments'])
                        if set(args)!={'points'}:raise ValueError('Unexpected geometry tool arguments')
                        answer=probe(match,args['points'])
                        messages.append(dict(role='tool',tool_call_id=call['id'],content=json.dumps(answer,ensure_ascii=False)))
                    exclusive(directory/'api_calls'/(stem+'_tool_result.json'),messages[-len(response['tool_calls']):])
                    continue
                if response['finish_reason']!='stop':raise ValueError('Incomplete/truncated model output; simulation remains paused')
                request,route=parse_action(response['content'],match)
                break
            else:raise ValueError('No final action within call budget')
            usage=usage_total(records)
            entry=dict(**request,decision_wall_seconds=time.perf_counter()-started,simulation_start=match.t,
                observation_ready_utc=packet['ready']['ready_utc'],accepted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                compiled_route=[p.tolist() for p in route],tokens=usage['total_tokens'],token_status='provider_reported' if usage['usage_complete'] else 'unavailable_or_partial',
                token_usage=usage,api_wall_seconds=sum(r['api_wall_seconds'] for r in records),api_call_count=len(records),
                response_models=[r['response_model'] for r in records])
            exclusive(directory/'requests'/f'{index:03}.json',request)
            exclusive(directory/'decisions'/f'{index:03}.json',entry)
            match.decisions.append(entry);match.apply(request['action'],route)
            execution=time.perf_counter();match.advance(2);elapsed=time.perf_counter()-execution
            save(match,directory);export_api_match(match,directory)
            exclusive(directory/'decisions'/f'{index:03}_outcome.json',dict(simulation_end=match.t,execution_wall_seconds=elapsed,state=match.states[-1]))
            messages.append(dict(role='assistant',content=response['content']))
            packet=observation_packet(match,directory)
            print(json.dumps(dict(step=index+1,t=match.t,score=match.ref.score,hp=match.ref.hp,
                wait_seconds=entry['decision_wall_seconds'],tokens=entry['tokens']),ensure_ascii=False),flush=True)
    except Exception as error:
        # Do not print arbitrary provider responses, prompts or credential-bearing exceptions.
        safe=str(error) if isinstance(error,APIError) else 'Request/skill validation or execution failed: '+type(error).__name__
        dump(directory/'api_run_status.json',dict(status='stopped_requires_review',simulation_seconds=match.t,
            committed_decisions=len(match.decisions),api_calls=total_calls,error=safe,
            automatic_resume=False,original_checkpoint_preserved=True))
        raise APIError(safe+'; inspect saved records; do not restart blindly') from None
    dump(directory/'api_run_status.json',dict(status='complete',simulation_seconds=match.t,score=match.ref.score,hp=match.ref.hp,api_calls=total_calls))
    print('比赛完成，轨迹已保存；原生回放命令：bash run.sh --demo --industrial --headless --replay-dir '+str(directory))
    return directory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['setup','status','prepare','probe','run'])
    parser.add_argument('--replace',action='store_true');parser.add_argument('--directory',type=Path)
    parser.add_argument('--profile',type=Path,default=CONFIG);args=parser.parse_args();config=settings(args.profile)
    if args.command=='setup':setup_key(args.replace)
    elif args.command=='status':print(json.dumps(dict(provider=config['provider'],base_url=config['base_url'],model=config['model'],
        key_configured=KEY_PATH.is_file(),key_path=str(KEY_PATH),api_requests_sent=0),ensure_ascii=False,indent=2))
    elif args.command=='prepare':prepare(config,args.directory or PREP)
    elif args.command=='probe':network_probe(config,load_key())
    else:
        key=load_key();directory=args.directory or RUNS/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'_Qwen3.6-27B_2秒闭环')
        run(config,directory,key)


if __name__=='__main__':
    try:main()
    except (APIError,ValueError,FileExistsError,FileNotFoundError) as error:
        # A missing key is normal during setup. Never echo key file contents.
        print('未执行或已停止：'+(str(error) if not isinstance(error,FileNotFoundError) else '缺少配置或 API Key，请先运行 bash qwen.sh setup'))
        raise SystemExit(1)
