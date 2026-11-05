"""Official Qwen adapter for event-driven, public-only sequence decisions."""
import argparse
import copy
import json
import statistics
from pathlib import Path
import qwen_fixed_goal as client
from ds_fixed_goal import R,scenario,prompt as old_prompt,WAYPOINTS,dump,utc,save_match
from sequence_skills import SequenceProbe,FEATURES,validate_plan_output
from qwen_client import usage_total

def prompt(base,rules):
    old=old_prompt(base,rules);head=old.split('每2秒仿真时刻')[0];map_json=old.split('地图和规则JSON：\n')[1]
    return head+'''\n本轮架构：先在必要时获取信息，再在掩体后规划4~8秒的条件式技能序列。底层50Hz连续执行，不再每2秒强制调用模型。计划完成、PEEK_BURST完成或到达计划时限后触发新决策；探出期间不会中断技能等待模型。调用期间仿真仍整体暂停，这不是实时端侧时延测试。
重要规则澄清：速度≤0.15m/s约束开枪的一方，绝不约束被射击目标！四足高速运动也能被人形击中。自身冷却也没有无敌效果。地图0°向东，90°向北，−90°向南。opportunistic只要合法可见、射程、反应和冷却允许就会主动停下瞄准，不会等到人形准备攻击才触发。
对手会记住最近合法看见的四足位置：失联后短时监视，再靠近最后位置，不会因你回到掩体就忘记你。对手仍无法知道你的隐藏真位置。重复探头不是无限安全。
公开输入提供地图路点、可行探身候选、合法观测摘要、运动趋势估计、最近执行事件和技能返回结果。候选的暴露预测基于当时合法敌位或已知初始位置并假设其暂时不动；接近候选本身也可能暴露。所有预测都不是安全保证。没有探身候选时可以先转移到合适掩体。旧敌位不能当成当前真值，路线末端需要重新确认。
当信息不足且有交战风险时，优先考虑低成本探身；已有足够信息、能够直接完成目标时可跳过。不要为了表演强行使用某个技能。你只决定阶段、路线与条件，底层负责连续轨迹和快速撤回。可用单节点计划先移动到掩体或观察；拿到信息后再给出较长路线。PEEK_BURST完成会停在掩体并重新调用你，因此宜放在计划最后。
输出唯一JSON，四字段 observation_id、skill、params、assessment；skill固定为PLAN。
params恰好包含 horizon_seconds（4~8）和steps（1~5个节点）。每节点恰好包含skill、params、on_contact。
on_contact为continue或retreat。retreat表示当前合法发现8m内人形时立即中止路线，底层用已知敌位选择附近遮挡位置撤回，然后重新调用你；continue表示按该技能继续执行，射击仍须合法。没有安全撤退点时会报告失败，不能保证免伤。探身自身始终执行快速撤回。
节点技能及完整参数：
PEEK_BURST {candidate_id,max_cycles}: candidate_id来自本次peek_candidates，max_cycles为1~3。自动到候选home、在遮挡内停稳对齐、直立短探并沿原路倒退；最多两次成功探身有足够有效样本后提前结束，或达到次数上限结束。探身过程不射击，最新观测使原掩体失效时会提前返回/撤退。完整往返包含加减速，预测暴露≤0.70s才可发起；这并非对未知移动对手的绝对保证。无需你每次控制探出和返回。
FOLLOW_ROUTE {waypoints,posture,fire}: waypoints为1~4个公开地图路点名称，底层A*连接并平滑成一条连续路线，不在中间提示点停下；中间路点允许0.65m通过容差。posture为upright/low，fire为off/opportunistic。它不会读取隐藏对手位置；未预见接触可用on_contact=retreat中止。选择路线时结合已知敌位、遮挡和信息年龄，不要只选最短路。
MOVE {target,posture,fire}: 移动到单一路点。fire=off保持推进，opportunistic遇到合法可射击的人形会主动停下瞄准，命中后恢复推进。
TAKE_COVER {target,posture}: 到你选择的路点，不开火；路点不天然安全。
TAG_AND_ADVANCE {target,posture}: 当前有合法射击机会则先停稳标记一次，命中后向target前进；失去当前检测或自身处于冷却则继续走。不是每次都能抢到先手。
SMOKE_AND_MOVE {location:[x,y],target,posture}: 按已有规则投烟，再向target移动，不主动开火。烟遮挡双方；投掷不可行时保留烟并报告，不会自动寻找新投点。
所有target/waypoints只使用公开地图路点名。不要生成未知坐标路点或虚构技能。assessment恰好包含 observation_evidence、state_evidence、decision_summary、uncertainty 四个短字符串，每项不超过300中文字，说明可核对事实、选择及不确定性。不输出完整内部推理、Markdown或自报用时。
地图和规则JSON：
'''+map_json

client.prompt=prompt
client.validate_output=validate_plan_output

def run_one(number,paid=False):
    if not paid:raise ValueError('--execute-paid required')
    case=client.CASES[number-1];base,rules=scenario(case['layout']);folder=R/f'run_{number}';folder.mkdir(exist_ok=True)
    rules.update(version='fixed-goal-v0.4-peek-sequences',decision_period_seconds=None,decision_policy='event_driven_plan_or_peek_completion')
    dump(folder/'case.json',case);dump(folder/'trial_config.json',dict(base=base,rules=rules,waypoints=copy.deepcopy(WAYPOINTS),features=FEATURES))
    (folder/'system_prompt.txt').write_text(prompt(base,rules)+'\n')
    decider=client.QwenPlanner(folder,base,rules,case['mode']);probe=SequenceProbe(base,rules,decider,folder)
    probe.metadata['attacker_policy']='qwen_thinking_'+case['mode']
    try:
        result=probe.run(f'qwen_v04_{case["layout"]}_{case["mode"]}');save_match(folder,result)
        dump(folder/'timeline.json',result['states']);dump(folder/'events.json',result['events'])
        summary=result['summary'];summary.update(thinking_mode=case['mode'],pair=case['pair'],run=number,status='complete',
            usage=usage_total(decider.records),api_total_seconds=sum(r['api_wall_seconds'] for r in decider.records),
            api_median_seconds=statistics.median(r['api_wall_seconds'] for r in decider.records),response_models=sorted({r['response_model'] for r in decider.records}))
        dump(folder/'result.json',summary);dump(folder/'process.json',dict(status='complete',utc=utc()))
        print('MATCH_COMPLETE',number,json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as e:
        dump(folder/'partial_timeline.json',probe.trace);dump(folder/'skill_events.json',probe.controllers[0].events)
        dump(folder/'process.json',dict(status='interrupted',error_type=type(e).__name__,simulation_seconds=probe.trace[-1]['t'] if probe.trace else 0,utc=utc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int,choices=range(1,7),required=True);p.add_argument('--execute-paid',action='store_true');a=p.parse_args();run_one(a.run,a.execute_paid)
