"""Posthoc review of the FIRST semantic-skill trial; never planner context."""
import json
from llm_trial import dump


def review_for(root,result,decisions,observations):
    old=root.parent/'20261021-173528-369598'
    previous=json.loads((old/'result.json').read_text());old_decisions=json.loads((old/'decisions.json').read_text())
    new_sizes=[len(json.dumps(json.loads((root/'actions'/f'{i:03d}.json').read_text()),ensure_ascii=False,separators=(',',':'))) for i in range(len(decisions))]
    old_sizes=[len(json.dumps(json.loads((old/'actions'/f'{i:03d}.json').read_text()),ensure_ascii=False,separators=(',',':'))) for i in range(len(old_decisions))]
    mean=result['decision_wait_seconds']/len(decisions);old_mean=previous['decision_wait_seconds']/len(old_decisions)
    timing=dict(previous_mean_wait_seconds=old_mean,current_mean_wait_seconds=mean,
        current_mean_skill_compile_seconds=result['skill_compile_seconds']/len(decisions),
        previous_mean_action_characters=sum(old_sizes)/len(old_sizes),current_mean_action_characters=sum(new_sizes)/len(new_sizes),
        current_longest_reason_characters=max(len(d['reason']) for d in decisions),
        current_used_skills=sorted(set(d['skill'] for d in decisions)),tokens=None,
        caveat='Different trajectories and sample counts; same development session, not controlled inference benchmark. Characters are not tokens. Old wait includes validation/path compilation, new compilation is separate.')
    dump(root/'comparison.json',timing)
    return dict(written_after_match=True,not_planner_input=True,
        conclusion='五个意图技能及短JSON接口已实跑；等待时间没有明显降低，首轮比赛表现下降。原始失败保留，修复后的技能另做工程测试。',
        findings=[
            f"五个技能均实际调用。顶层只提交observation_id、skill、params和一句reason；平均动作JSON由{timing['previous_mean_action_characters']:.0f}字符缩至{timing['current_mean_action_characters']:.0f}字符，字符数不是token数。",
            f"当前平均等待{mean:.2f}秒，上一轮{old_mean:.2f}秒；底层规划平均{timing['current_mean_skill_compile_seconds']:.3f}秒。不能据此声称明显加速；两轮轨迹不同，也不是纯推理时间比较。",
            '首轮9次决策，37张真实在线机载图，17.64秒淘汰，4分；没有回滚、没有改写失败动作，也没有接入DeepSeek。',
            '8秒和14秒只下达TAKE_COVER及截图中的脚底像素，底层自行选择High1西侧、倒退和具体路线。视觉估计不读取对手真值。',
            '10秒AIM_AT_TRACK按视觉估计转身；随后视线被墙挡住，没有自动穿墙瞄准或补齐隐藏坐标。它是限时最后目击点对准，不是成熟的连续视觉跟踪。',
            '发现并保留的实现错误：4秒OBSERVE north的朝向被tick中的旧目标记忆覆盖，实际未完成指定转身；赛后已修复，增加方向和云台优先级回归测试。',
            '规划不足：只加权空间暴露成本仍选择了危险转移，目标点风险低不等于途中安全；受击仅停车会留在暴露区。赛后增加连续暴露时长门限和占点受击后的局部撤掩体，不声称已提高完整比赛成绩。',
            '修复后的OBSERVE单独进行了Isaac两秒工程复测；首轮比赛使用的原代码保存在本目录source_snapshot，修复不追溯修改它。'],
        next_steps=[
            '本轮只完成用户确认的技能封装与短输出测试；没有配置Flash、没有调用任何付费模型API。',
            '当前交互会话的端到端等待仍很长。若继续测试速度，应使用专用自动请求循环和固定输入，分别计时图像准备、请求往返、完整动作返回及底层规划。',
            '修复版需另做完整闭环试验和多种子对照；本轮4分不能用于宣称技能增强有效，也不能将旧7.82分与本轮视作严格公平排名。'],
        limits=[
            'Python运动学＋Isaac在线RGB，无接触动力学、无训练；规则、种子、2秒决策和0.5秒采样保持不变。',
            '相较旧版额外改变了执行接口、风险路径、程序化相机和射击使能：只有不超过2秒的视觉位置估计才允许主动射击。不是只改变输出长度的单变量试验。',
            '没有新增自动视觉检测器。模型在当前输入帧中标注可见脚底像素，由相机标定与平地假设换算位置；像素误差、遮挡脚底和地面假设都会影响估计。',
            '视觉记忆最多4秒，不确定半径随时间增加；瞄准估计最多2秒。未知不等于安全；当前五点位置假设只是启发式，不是完整概率滤波。',
            '底层只接收公开自状态、规则地图、视觉估计，未接入对手当前坐标、朝向、HP、瞄准时间、随机数或内部策略。',
            '同一开发会话继承了旧工程背景，不是全新上下文严格盲测；原始图像不是实拍照片。',
            '50Hz执行和自状态检查不等于50Hz视觉感知；截图仍是2Hz，模型只在2秒边界更新视觉标注。',
            '报告将在线一句依据、程序编译的路线和赛后复盘分开呈现；详细复盘不是当时的思考原文。',
            '计时是观测就绪至提交，包括看图、工具及交互开销。原始机制无法进一步拆出纯推理耗时。Token不可获取，记录null。',
            '修复后的危险路径门限只检查候选路线，不保证一定找到全局安全路径；若候选不满足约束，返回UNSAFE_ROUTE交给顶层重新选择。'])
