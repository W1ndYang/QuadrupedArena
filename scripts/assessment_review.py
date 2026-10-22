"""Generic post-match statistics for the shared four-field assessment protocol."""
import json
import statistics
from collections import Counter
from llm_trial import dump


def review_for(root,result,decisions,protocol):
    api=result['mode']=='qwen_semantic_trial'
    waits=[d['decision_wait_seconds'] for d in decisions]
    times=[d['api']['api_wall_seconds'] for d in decisions] if api else []
    usage=json.loads((root/'api_usage.json').read_text()) if api else None
    summary=dict(written_after_match=True,not_planner_input=True,model=protocol['planner'],reasoning_effort=protocol['reasoning_effort'],
        score=result['score'],simulation_seconds=result['simulation_seconds'],finish=result['finish'],final_hp=result['final_hp'],
        decisions=len(decisions),frames=result['frames'],decision_wait_mean_seconds=statistics.mean(waits),
        decision_wait_median_seconds=statistics.median(waits),decision_wait_max_seconds=max(waits),
        api_mean_seconds=statistics.mean(times) if times else None,api_median_seconds=statistics.median(times) if times else None,
        api_min_seconds=min(times) if times else None,api_max_seconds=max(times) if times else None,
        skill_compile_mean_seconds=result['skill_compile_seconds']/len(decisions),
        selected_skills=dict(Counter(d['skill'] for d in decisions)),visual_annotations=sum('sighting' in d['params'] for d in decisions),
        usage=usage,tokens=result['tokens'],automatic_retries=0,raw_decision_fields=protocol['intent_schema'],pure_thinking_seconds=None,
        timing_caveat='API elapsed includes transport, queue, compute and output. Assistant wait includes image/tool handling; not directly comparable to API time.')
    dump(root/'test_summary.json',summary)
    findings=[f"本轮{len(decisions)}次决策、{result['frames']}帧，得分{result['score']:.2f}，比赛{result['simulation_seconds']:.2f}秒；结束原因{result['finish']}；终局HP {result['final_hp']}（A、B）。",
        f"平均观测就绪至动作接受{statistics.mean(waits):.2f}秒；底层规划平均{summary['skill_compile_mean_seconds']:.3f}秒。",
        f"技能选择：{summary['selected_skills']}；脚点标注{summary['visual_annotations']}次。",
        '每步保留逐帧事实、公开状态、决策摘要和不确定性；解释是待核验自述，不等同于实际执行或内部思维链。']
    limits=['Python运动学裁判＋Isaac在线RGB，不运行接触动力学；决策每2秒、图像每0.5秒，等待期间比赛暂停。',
        '仅预知地图规则与对手初始位置；当前对手真值不进入观测。脚点估计不是自动跟踪器。',
        '无重试、无自动模型回退、无人工动作修复；单个种子和单轮闭环不能证明模型普遍优劣。',
        '同一代码技能1.1，未修复上轮暴露的UNSAFE_ROUTE原地停留行为，以免混淆本轮对照。']
    if api:
        findings.append(f"API平均{statistics.mean(times):.2f}秒、中位{statistics.median(times):.2f}秒、最慢{max(times):.2f}秒；总Token {usage['total_tokens']}，推理Token {usage['reasoning_tokens']}（未返回则为空，不估算）。")
        profile=json.loads((root/'api_profile.json').read_text())
        limits.extend([profile['effort_semantics'],
            'Qwen只接收同一背景提示词、当前1～5张图及最近4个动作，不接收旧Qwen全知字段、对手策略源码或助手工程对话。',
            '输入原始PNG未改变；服务端图像编码与DeepSeek不同。Qwen两轮同temperature=0.6、回答预算16384和600秒请求截止，思考开关不同。'])
    else:
        limits.extend(['本轮是当前助手交互会话，继承了工程讨论和历史结果，不是全新上下文盲测；不读取本轮尚未结束的对手真值。',
            '每步外部计时从观测就绪至动作接受，含看图、工具和交互开销；无法分离纯推理时间，也没有逐步Token遥测，故Token为null。'])
    return dict(written_after_match=True,not_planner_input=True,conclusion=protocol['planner']+' / '+protocol['reasoning_effort'],
        findings=findings,next_steps=['核对截图中的视觉事实与实际编译动作，结合多种子复测；本页不将解释流畅视为策略有效。'],limits=limits)
