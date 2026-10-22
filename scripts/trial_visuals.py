"""Presentation-only captions. Match state always comes from the R0 referee."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arena_fonts import load_font
FONTS={s:load_font(s) for s in (15,17,19,22,27)}
WHITE="#e9f2fb";CYAN="#49c9ff";ORANGE="#ffb360"
EYE=np.array([-9.,8.,12.]);TARGET=np.array([-.2,.15,.3]);FOCAL=30.;APERTURE=20.955


def project_point(point,width,height):
    forward=TARGET-EYE;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward);d=np.asarray(point)-EYE;z=d@forward
    return (width/2+(d@right)/z*FOCAL/APERTURE*width,height/2-(d@up)/z*FOCAL/APERTURE*width)


def panel(raw,s,events,cfg,collapse=None,intro=False,outro=False):
    w,h=raw.size;image=Image.new("RGB",(w,h+240),"#101d2d");image.paste(raw,(0,104));d=ImageDraw.Draw(image)
    adaptive='a_state' in s
    llm='llm_decision' in s
    partial_vision=cfg.get('_presentation_partial_vision',False)
    decision_count=cfg.get('_presentation_decisions',math.ceil(cfg['duration']/2))
    rotating=cfg.get('target_rotation',{}).get('enabled',False)
    planner=cfg.get('_presentation_planner')
    effort=cfg.get('_presentation_effort','unknown')
    vision_heading=(f"DeepSeek Flash · {effort}｜非全知视觉决策回放" if planner=='deepseek-flash'
        else f"Qwen3.6-27B · {'思考开启' if effort in ('max','maximum_default_budget') else '思考关闭' if effort in ('min','thinking_off') else effort}｜非全知回放" if planner=='qwen3.6-27b'
        else "助手决策｜非全知视觉回放" if planner=='current_assistant_session'
        else "LLM｜非全知视觉决策回放")
    heading=vision_heading if partial_vision else "LLM 首轮｜2 秒闭环规划 × 固定运动技能" if llm else "R0 规则 v0.4｜限时轮换占点 × 快速搜索" if rotating else "R0 策略改进｜有限感知 × 风险绕行" if adaptive else "R0 规则首轮试验｜轮足占点 × 人形施压"
    d.text((20,8),heading,font=FONTS[22],fill=WHITE)
    d.text((w-230,12),"Isaac Sim · 原生场景渲染",font=FONTS[17],fill="#aec3d8")
    d.text((20,44),f"轮足 A  HP {s['hp'][0]}/3    得分 {s['score']:05.2f}",font=FONTS[22],fill=CYAN)
    d.text((w-325,44),f"人形 B  HP {s['hp'][1]}/3    不计分",font=FONTS[22],fill=ORANGE)
    d.text((405,47),f"{s['t']:05.2f} / {cfg['duration']:.0f} s",font=FONTS[19],fill=WHITE)
    for i,x in enumerate((20,610)):
        status=f"命中冷却 {s['cooldown'][i]:.2f}s" if s['cooldown'][i]>0 else f"连续瞄准 {s['aim'][i]:.2f}s · λ {s['rates'][i]:.2f}/s"
        d.text((x,77),status,font=FONTS[15],fill=CYAN if i==0 else ORANGE)
    for i,p in enumerate(cfg['arena']['task_points']):
        if rotating and not s['eligible'][i]:continue
        x,y=project_point((*p['center'],.05),w,h);y+=104
        text=f"P{i+1} " + ("失效" if not s['eligible'][i] else "+1 分/s" if s['scoring'][i] else "可占领")
        if rotating:text=f"P{i+1} " + ('待触发' if s['target_remaining'] is None else f"{s['target_remaining']:.1f}s")
        color="#ff7070" if not s['eligible'][i] else "#a2ffc0"
        d.rounded_rectangle((x-50,y-38,x+76,y-12),radius=5,fill="#142b32")
        d.text((x-43,y-36),text,font=FONTS[15],fill=color)
    for i in range(2):
        x,y=project_point((*s['xy'][i],1 if i==0 else 1.65),w,h);y+=104
        label=("A" if i==0 else "B")+(" 淘汰" if s['hp'][i]==0 else "")
        d.text((x-10,y-12),label,font=FONTS[22],fill=CYAN if i==0 else ORANGE,stroke_width=2,stroke_fill="#142b32")
    title="接近目标｜已知地图 A*，人形主动寻找射击位置"
    detail="同圈不阻止得分；只有实际命中才扣血，并使所在目标永久失效。"
    if adaptive:
        human={'PATROL':'巡逻搜索','PATROL_SCAN':'原地扫视','VISIBLE_AIM':'发现目标·瞄准','VISIBLE_COOLDOWN':'发现目标·自身冷却','SEARCH_LAST_SEEN':'追查最后位置','SEARCH_SCAN':'最后位置附近扫视','ELIMINATED':'已淘汰'}[s['human_state']]
        d.rounded_rectangle((18,114,430,148),radius=6,fill='#142b32')
        memory='' if s['human_memory_age'] is None or s['human_visible'] else f" · 记忆 {s['human_memory_age']:.1f}s"
        d.text((28,119),'人形：'+human+memory,font=FONTS[17],fill=ORANGE)
        title={'RISK_ROUTE':'风险绕行｜距离与暴露共同决定路径','EVADE':'主动避险｜先朝掩体移动，切断连续瞄准','HOLD':'掩体保护下驻留｜安全时扫视，不隔墙持续盯人','WINDOW_ADVANCE':'利用冷却窗口｜根据已发生的命中事件转移','WAIT_SCAN':'暂时无可行通路｜等待并重新规划','ADVANCE':'接近目标'}[s['a_state']]
        detail='人形仅接收视野内无遮挡的四足位置；失去目标后使用最后观测，不读取实时隐藏坐标。'
    if rotating:
        remaining='首次进入后倒计时' if s['target_remaining'] is None else f"剩余 {s['target_remaining']:.2f}s"
        detail=f"第 {s['target_generation']} 次刷新 · P{s['active_target']+1} · {remaining}；超时/圈内受击后切换下一点。"
    if llm:
        title=f"决策 {s['llm_decision']:02d}/{decision_count} · {s['llm_skill']}｜每 2 秒观测后下发，思考时比赛暂停"
    if not adaptive and any(s['scoring']):title="驻留达标，持续计分｜在掩体后争取占点时间"
    if not adaptive and any(not v for v in s['eligible']) and not any(s['inside']):title="原目标已失效｜转移至剩余目标，历史得分保留"
    recent=[e for e in events if e['type']=='hit' and 0<=s['t']-e['time']<1.5]
    if recent:
        e=recent[-1];title=f"{'人形 B → 轮足 A' if e['attacker']==1 else '轮足 A → 人形 B'} 命中｜仅命中方冷却 2 秒"
        detail="受击方若存活且不在自身冷却中，仍可立即还击；击中本身不加分。"
    if cfg['collapse']['time']-2<=s['t']<cfg['collapse']['time']:
        title="场地变化预告｜蓝色高墙即将在第 20 秒向场内倾倒"
    if s['collapsed'] and s['t']-cfg['collapse']['time']<3:
        title="高墙变为低矮障碍｜导航更新，遮挡关系重新计算"
        detail="R0：固定墙、固定倾倒方向和终态；倒塌区域由规划器提前避让。"
    if collapse is not None:
        title="坍塌事件慢镜头｜固定轴旋转 90°，形成局部障碍"
        detail="这是规则瞬时切换的展示镜头，不增加比赛时间；并非刚体倒塌物理仿真。"
    if intro:
        title="首轮参数试验｜两处目标始终开放，不按时间切换"
        detail="双方 3 HP；驻留 1s 后 +1分/s；正前总计 60° 且无遮挡才有效瞄准。"
        if adaptive:
            title='策略改进对照｜比赛规则与随机种子不变'
            detail='人形：120° 视野搜索，60° 射击；四足：保留位置真值，增加风险绕行与主动避险。'
        if rotating:
            title='单点轮换｜首次进圈后 5 秒失效，P1 → P2 → P1 循环'
            detail='离圈不重置倒计时；驻留 1 秒后计分；圈内受击提前失效；人形加快转身与搜索。'
        if llm:
            title='真实逐步闭环记录｜全知当前状态，未读取未来命中随机数'
            detail=f'{decision_count} 次外部决策调用 MOVE / HOLD / AIM 等固定技能；原规则、原限速、原人形策略。'
        if partial_vision:
            title='原比赛：仅初始敌方位置已知，随后依据机载多帧与公开状态决策'
            detail=f'{decision_count} 次决策；本片是赛后第三视角展示，HUD与全场信息未提供给模型。'
    if outro:
        reason={"quadruped_eliminated":"轮足生命清零","all_targets_invalid":"所有目标失效","time_limit":"达到时间上限"}[s['finished']]
        title=f"本轮结束：{reason}｜轮足得分 {s['score']:.2f}"
        detail=f"P1 {s['target_scores'][0]:.2f} 分 / P2 {s['target_scores'][1]:.2f} 分；这是基线结果，不代表规则已平衡。"
        if llm:detail=f"P1 {s['target_scores'][0]:.2f} 分 / P2 {s['target_scores'][1]:.2f} 分；单次探索结果，不代表已验证优于决策树。"
    y=h+115;d.text((20,y),title,font=FONTS[22],fill=WHITE)
    d.text((20,y+36),detail,font=FONTS[17],fill="#bbd0e2")
    footer="非全知机载决策 · 本片为赛后全场回放 · 运动学姿态 · 不重跑策略，不调用模型API" if partial_vision else "无训练 · LLM 顶层 + 固定 A* 技能 · 运动学姿态 · 真实轨迹重放；图像不用于视觉闭环" if llm else "无训练 · 脚本/A* 决策 · 运动学姿态，非接触动力学验证 · 字幕与裁判辅助线为后期/场景叠加"
    d.text((20,y+65),footer,font=FONTS[15],fill="#93aabd")
    legend='青线：轮足规划路径   小橙圈：人形最后观测位置   大扇形：120° 感知方向示意   小扇形：60° 射击' if adaptive else "绿圈：有效目标   红圈：本回合失效   扇形：正前 60°   红线：被墙阻挡的视线   黄线：有效瞄准"
    d.text((20,y+90),legend,font=FONTS[15],fill="#93aabd")
    return image
