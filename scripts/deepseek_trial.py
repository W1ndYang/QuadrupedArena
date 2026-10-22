"""Public-observation planner adapter; no opponent truth, no simulation access.

setup/status/prepare are offline. probe and run are explicit paid operations.
"""
import argparse
import base64
import copy
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from deepseek_client import APIError,KEY_PATH,call_api,endpoint,load_key,payload,save_key,usage_total

PROJECT=Path(__file__).resolve().parents[1]
from arena_paths import workspace_root
RUNS=workspace_root()/'experiments/runs/quadruped-arena'
CONFIG=PROJECT/'configs/deepseek_trial.json'
PROMPT='''你是虚拟机器人占点比赛中轮足四足A的顶层规划器，不控制真实机器人。本项目验证“大模型选择意图、代码技能执行运动”。无需训练，也无需解释搭建方法。目标是最大化A累计占点分，不是攻击次数或仅存活。
这是Python运动学/裁判＋Isaac在线机载RGB，不运行接触动力学。比赛每2秒暂停一次等待你的决策，等待时间不计比赛时间。每0.5秒采样；第一步一张图，之后通常是上一边界图加4张新图。逐张结合时间和相机方向判断，不能只看末帧。
你预先知道地图、规则、对手初始位置；不知道对手当前真实位置、朝向、生命值、策略内部状态或随机数。只能使用当前公开观测、机载图片和过去的合法记忆。未看见不代表不存在；记忆不是实时位置。B是倾向施压攻击的人形，不提供其策略源码或当前路线。不要假设知道历史试验结果。
世界xy单位米，yaw单位弧度，0朝+x（东），正转朝+y（北）。图片768×432，左上角像素[0,0]，u向右v向下。每图文字标注其时间、机身和云台朝向。相机不是武器，转相机不能替代机身瞄准。
双方初始3HP。A生命清零立即终局；B生命清零不终局、不复活。只有占点得分、击中不加分。单点依次P1/P2循环刷新：首次进入触发5秒期限，驻留1秒后每秒1分；离圈重置驻留但不重置期限；圈内受击使该点立即失效并刷新下一代；累计分保留。active_target是零起始索引，target_remaining=null表示未触发，不是失效。到期后不要继续守旧点。
射击要求距离≤8米、正前总计60°、代表点连线无遮挡，连续有效瞄准的条件命中强度随时间增长，人形增长更快。不是固定瞄准时长必定命中；仅命中方冷却2秒。20秒固定墙体坍塌。具体参数以规则JSON为准。
你只需选择并输出可执行决策，一个JSON对象，恰好三字段：observation_id、skill、params。不要输出reason、理由、分析、解释、总结、计时、思维链、Markdown、工具调用或任意代码。记录和计时全部由外部程序完成。
五个技能：
MOVE_TO：移动到目标，params.target为active/P1/P2，默认active；不自动选择未来目标。
CAPTURE_POINT：移动并驻留当前有效目标；target同上；失效停止，受击由底层触发局部撤掩体。
TAKE_COVER：底层选择掩体及路线；可选params.cover为地图掩体名，省略让算法选。
AIM_AT_TRACK：停下对准最后视觉估计，不是穿墙追踪；需要新鲜视觉记忆才允许主动攻击。
OBSERVE：停下观察；params.direction为scan/threat/north/south/east/west，默认scan。方向是机身世界朝向，云台自动扫描。
所有技能允许params.risk=safe或fast（默认safe）。safe可能拒绝候选危险路线并返回UNSAFE_ROUTE，不保证全局安全；应根据下一次结果重新选择。路线、倒退、速度、掩体站位由代码计算，禁止输出goal/via/数值朝向等底层字段。技能输出只代表请求，下一次观测才是执行结果。
任一技能可附params.sighting={"time":当前输入帧的精确时间,"feet":[u,v]}：仅在图中确实看见人形接地脚点时标注。脚被墙遮挡时省略，不猜测像素或传入世界坐标。底层按相机标定与平地假设估计位置，记忆4秒过期、误差随时间增长；只有≤2秒视觉估计可使能射击。已知初始坐标不使能攻击。标注不是成熟连续跟踪。
短格式示例（只是格式，不是推荐动作）：{"observation_id":0,"skill":"OBSERVE","params":{"direction":"north"}}
你每次只有一次API请求，没有自动重试或替你修复策略。根据本步观测直接给出三字段决策JSON。'''


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def settings(path=None):
    source=Path(path or os.environ.get('ARENA_DEEPSEEK_CONFIG',CONFIG))
    c=json.loads(source.read_text());payload(c,[])
    for name in ('wall_timeout_seconds','launch_timeout_seconds'):
        value=c.get(name)
        if value is not None and (type(value) not in (int,float) or value<=0):raise ValueError('Invalid '+name)
    if type(c['timeout_seconds']) not in (int,float) or c['timeout_seconds']<=0:raise ValueError('Invalid network timeout')
    if c['automatic_retries']!=0 or c['automatic_strategy_fallback'] or c['max_calls_per_match']!=20:
        raise ValueError('Unexpected request budget or fallback')
    if (c['decision_period_seconds'],c['frame_interval_seconds'])!=(2,.5):raise ValueError('Unexpected cadence')
    if c['rules_config']!='trial_rotating.json' or c['match_seed']!=2026102107:raise ValueError('Unexpected rules/seed')
    return c


def system_message(cfg,config=None):
    # Public static rules only. Exclude opponent policy, RNG seed and rendering config.
    keys=('arena','collapse','duration','dt','initial_hp','range_m','aim_half_angle_degrees','tau','lambda_max',
          'cooldown_seconds','hold_seconds','score_rate','representative_heights','target_rotation','max_speed','acceleration','yaw_rate')
    rules={k:cfg[k] for k in keys if k in cfg}
    prompt=PROMPT
    if config and config.get('explain_decisions'):
        prompt=prompt.replace('你只需选择并输出可执行决策，一个JSON对象，恰好三字段：observation_id、skill、params。不要输出reason、理由、分析、解释、总结、计时、思维链、Markdown、工具调用或任意代码。记录和计时全部由外部程序完成。',
            '输出一个JSON对象，恰好四字段：observation_id、skill、params、assessment。assessment是供人核验的简明依据，不是详细内部思维链。不要Markdown、工具调用、任意代码或自报耗时。')
        prompt=prompt.replace('短格式示例（只是格式，不是推荐动作）：{"observation_id":0,"skill":"OBSERVE","params":{"direction":"north"}}',
            '''assessment恰好包含visual_evidence、state_evidence、decision_summary、uncertainty四字段。
visual_evidence为数组，对本步每一张输入图片按时间各写一条：{"time":该帧时间,"opponent_visible":true或false或null,"observation":"具体画面事实，最多200字符"}。不确定用null。描述可核对的墙体、通道、目标圈、人形局部或遮挡，不能只复述地图坐标。看不清就明确说明，不能为了证明看图编造目标。
state_evidence为公开状态依据，最多300字符。decision_summary为技能选择与主要取舍摘要，最多300字符。uncertainty为当前未知与风险，最多300字符。将图片事实、状态信息和推断分开，不输出详细思维链。
短格式示例（仅示意结构，不是策略推荐）：{"observation_id":0,"skill":"OBSERVE","params":{"direction":"north"},"assessment":{"visual_evidence":[{"time":0,"opponent_visible":null,"observation":"画面被近处墙面遮挡，不能确认墙后是否有人形。"}],"state_evidence":"自身HP3，尚未得分。","decision_summary":"先观察侧向通道，代价是推迟入点。","uncertainty":"未看到不代表不存在。"}}''')
        prompt=prompt.replace('根据本步观测直接给出三字段决策JSON。','根据本步观测给出决策和简短可核验依据的四字段JSON。')
    return dict(role='system',content=prompt+'\n公开规则和地图：\n'+json.dumps(rules,ensure_ascii=False))


def motion_request(request):
    return {k:request[k] for k in ('observation_id','skill','params')}


def parse_output(content,allow_single_json_fence=False):
    # Transport formatting only. Never extract prose, repair JSON, or edit an action.
    if allow_single_json_fence:
        lines=content.strip().splitlines()
        if len(lines)>=3 and lines[0] in ('```json','```') and lines[-1]=='```':
            content='\n'.join(lines[1:-1])
            if '```' in content:raise ValueError('Multiple/nested code fences are not accepted')
    return json.loads(content)


def validate_model_output(request,packet,config,validator):
    expected={'observation_id','skill','params'}|({'assessment'} if config.get('explain_decisions') else set())
    if not isinstance(request,dict) or set(request)!=expected:raise ValueError('Unexpected model output fields')
    if 'assessment' in request:
        a=request['assessment']
        if not isinstance(a,dict) or set(a)!={'visual_evidence','state_evidence','decision_summary','uncertainty'}:raise ValueError('Invalid assessment fields')
        for key in ('state_evidence','decision_summary','uncertainty'):
            if not isinstance(a[key],str) or not 1<=len(a[key])<=300:raise ValueError('Assessment summary must contain 1..300 characters')
        evidence=a['visual_evidence'];times=[f['simulation_seconds'] for f in packet['frames']]
        if not isinstance(evidence,list) or len(evidence)!=len(times):raise ValueError('Describe every input frame once')
        for item,t in zip(evidence,times):
            if not isinstance(item,dict) or set(item)!={'time','opponent_visible','observation'}:raise ValueError('Invalid visual evidence fields')
            if type(item['time']) not in (int,float) or item['time']!=t:raise ValueError('Evidence must cite the input frame time')
            if item['opponent_visible'] is not None and type(item['opponent_visible']) is not bool:raise ValueError('Visibility must be true/false/null')
            if not isinstance(item['observation'],str) or not 1<=len(item['observation'])<=200:raise ValueError('Visual description must contain 1..200 characters')
    validator(motion_request(request),packet)


def public_observation(packet):
    expected={'observation_id','simulation_seconds','self_state','public_state','frames','skills'}
    if packet['observation_id']==0:expected.add('opponent_initial_xy')
    # perception is an explanatory public string from the renderer.
    expected.add('perception')
    if set(packet)!=expected:raise ValueError('Observation fields do not match public whitelist')
    if set(packet['self_state'])!={'xy','yaw','hp','speed','cooldown','skill_status'}:raise ValueError('Invalid self-state fields')
    if set(packet['public_state'])!={'score','active_target','target_generation','target_remaining','dwell','finished'}:raise ValueError('Invalid public-state fields')
    allowed={'file','simulation_seconds','own_xy','body_yaw_degrees','camera_pan_degrees','camera_world_yaw_degrees','camera_eye','horizontal_fov_degrees'}
    if not 1<=len(packet['frames'])<=5:raise ValueError('Expected 1..5 current frames')
    for f in packet['frames']:
        if set(f)!=allowed or f['simulation_seconds']>packet['simulation_seconds']:raise ValueError('Invalid/future frame metadata')
    return copy.deepcopy(packet)


def observation_message(packet,root,config,history):
    p=public_observation(packet)
    blocks=[dict(type='text',text=json.dumps(dict(observation=p,recent_decisions=history[-config['history_decisions']:]),ensure_ascii=False))]
    manifests=[];root=Path(root).resolve()
    for f in p['frames']:
        path=(root/f['file']).resolve()
        if path.parent!=root/'frames' or path.suffix!='.png':raise ValueError('Frame path must stay in run/frames')
        content=path.read_bytes()
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Expected PNG frame')
        blocks.append(dict(type='text',text=json.dumps(f,ensure_ascii=False)))
        blocks.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(content).decode(),detail=config['image_detail'])))
        manifests.append(dict(file=f['file'],sha256=hashlib.sha256(content).hexdigest(),bytes=len(content)))
    return dict(role='user',content=blocks),manifests


class Planner:
    def __init__(self,config,cfg,key,root,client=call_api):
        self.config=config;self.system=system_message(cfg,config);self.key=key;self.root=Path(root);self.client=client
        self.calls=0;self.history=[];self.records=[]
        (self.root/'api_calls').mkdir(exist_ok=False)
        dump(self.root/'api_profile.json',config)
        (self.root/'planner_context.txt').write_text(self.system['content'])

    def decide(self,packet,validator):
        if self.calls>=self.config['max_calls_per_match']:raise ValueError('API request budget exhausted')
        index=packet['observation_id']
        if index!=self.calls:raise ValueError('Unexpected observation sequence; no automatic retry')
        started=time.perf_counter();message,frames=observation_message(packet,self.root,self.config,self.history)
        messages=[self.system,message];stem=self.root/'api_calls'/f'{index:03d}'
        # Exact images remain in frames/; manifest avoids duplicating base64 in the report.
        dump(str(stem)+'_request.json',dict(profile=self.config,system=self.system,observation=packet,
            recent_decisions=self.history[-self.config['history_decisions']:],images=frames,
            message_build_seconds=time.perf_counter()-started))
        self.calls+=1
        call_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();call_start=time.perf_counter()
        try:
            response=self.client(self.config,self.key,messages);self.records.append(response)
            dump(str(stem)+'_response.json',response)
            if response['finish_reason']!='stop' or response['tool_calls']:raise ValueError('Incomplete/non-JSON action response')
            request=parse_output(response['content'],self.config.get('allow_single_json_fence',False))
            validate_model_output(request,packet,self.config,validator)
        except Exception as error:
            dump(str(stem)+'_failure.json',dict(status='failed_no_action',error_type=type(error).__name__,automatic_retry=False,
                call_start_utc=call_start_utc,call_failed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                call_until_failure_seconds=time.perf_counter()-call_start))
            raise
        self.history.append(copy.deepcopy(motion_request(request)));usage=usage_total([response])
        timing={k:response[k] for k in ('api_wall_seconds','api_started_utc','api_finished_utc','time_to_first_output_seconds','time_to_first_visible_content_seconds','response_model','response_id')}
        timing.update(usage=response['usage'],tokens=usage['total_tokens'],reasoning_tokens=usage['reasoning_tokens'],
            planner_wall_seconds=time.perf_counter()-started,reasoning_effort=self.config['reasoning_effort'],reasoning_text_saved=False)
        return request,timing


def main():
    parser=argparse.ArgumentParser(description='DeepSeek official Flash adapter; effort from config; setup/status/prepare do not call API')
    sub=parser.add_subparsers(dest='command',required=True)
    setup=sub.add_parser('setup',help='Hidden local API key input');setup.add_argument('--replace',action='store_true')
    sub.add_parser('status');sub.add_parser('prepare',help='Offline prompt and request preparation')
    sub.add_parser('probe',help='One paid JSON connectivity request at configured effort')
    sub.add_parser('run',help='Paid full partial-observation Isaac match; max 20 requests')
    args=parser.parse_args();config=settings()
    if args.command=='setup':
        if not sys.stdin.isatty():raise ValueError('Run bash deepseek.sh setup in your local terminal; hidden input requires a TTY')
        save_key(getpass.getpass('DeepSeek 官方 API Key（输入隐藏，粘贴后回车）：'),path=KEY_PATH,replace=args.replace)
        print('密钥已保存到本机独立私密文件（0600）；未发送API请求。');return
    if args.command=='status':
        configured=KEY_PATH.exists()
        if configured:load_key()
        print(json.dumps(dict(configured=configured,key_path=str(KEY_PATH),profile=config,api_requests_sent=0),ensure_ascii=False,indent=2));return
    if args.command=='prepare':
        from adaptive_motion import load_config
        directory=RUNS/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'_DeepSeek_Flash_准备')
        directory.mkdir(parents=True);cfg=load_config(PROJECT/'configs'/config['rules_config'])
        dump(directory/'api_profile.json',config);dump(directory/'sample_request.json',payload(config,[system_message(cfg,config),dict(role='user',content='本步公开观测与1到5张Isaac机载RGB将在运行时加入。')]))
        dump(directory/'preparation.json',dict(status='prepared_not_connected',api_requests_sent=0,credentials_included=False,
            docs=['https://api-docs.deepseek.com/','https://api-docs.deepseek.com/guides/thinking_mode/','https://api-docs.deepseek.com/guides/vision/'],
            input='Public state + onboard RGB; no opponent current truth; bounded four-decision text history',
            latest_alias_not_immutable=True,ready_for_probe_after_key=True))
        print(str(directory));return
    key=load_key()
    if args.command=='probe':
        r=call_api(config,key,[dict(role='user',content='仅输出JSON对象：{"ok":true}，不要其他文字。')])
        directory=RUNS/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'_DeepSeek_API检查');directory.mkdir(parents=True);dump(directory/'response.json',r)
        if r['finish_reason']!='stop' or json.loads(r['content'])!={'ok':True}:raise ValueError('Probe did not return complete expected JSON')
        print(json.dumps(dict(status='passed',record=str(directory),model=r['response_model'],api_seconds=r['api_wall_seconds'],usage=r['usage']),ensure_ascii=False));return
    raise SystemExit(subprocess.call(['bash','run.sh','--vision-trial','--semantic-skills','--deepseek','--headless'],cwd=PROJECT))


if __name__=='__main__':
    try:main()
    except (APIError,ValueError,OSError) as error:
        # No raw HTTP exception, key or provider body in the console.
        print('DeepSeek配置/执行未完成：'+str(error) if isinstance(error,APIError) else 'DeepSeek配置/执行未完成：'+type(error).__name__+'；请检查密钥是否已配置、文件权限及本地参数。',file=sys.stderr)
        raise SystemExit(1)
