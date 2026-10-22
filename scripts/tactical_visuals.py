"""Presentation overlays only; no access to the agent's decision inputs."""
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

TITLES={"TAKE_COVER":"01  发现对手 · 快速进掩体", "HIDDEN":"02  藏在墙后 · 视线被遮挡",
    "PEEK":"03  探出墙角 · 重新观察", "AIM":"03  朝向对手 · 持续锁定",
    "WITHDRAW":"04  标记后撤回 · 不停留对射", "WATCH":"05  对手绕侧 · 扫描重新发现",
    "RELOCATE":"06  侧面威胁出现 · 更换掩体", "NEW_COVER":"07  新掩体阻断视线",
    "TO_POINT":"08  快速转移 · 进入任务区", "CAPTURE":"09  占点与虚拟标记 · 结算"}

@lru_cache(None)
def font(size):
    from arena_fonts import load_font
    return load_font(size)

def panel(rgb,state,events,intro=False,outro=False):
    w,h=rgb.size
    im=Image.new("RGB",(w,h+224),"#101d2d"); im.paste(rgb,(0,96))
    d=ImageDraw.Draw(im); white="#eef5ff"; blue="#49c9ff"; orange="#ffb360"
    def text(x,y,s,size=18,color=white): d.text((x,y),s,font=font(size),fill=color)
    text(22,9,"Go2-W 轮足｜掩体机动决策 Demo",24)
    text(w-240,14,"Isaac Sim 原生渲染",18,"#aec3d8")
    text(22,53,f"轮足 A  HP {state['hp'][0]}/3   积分 {state['score'][0]:04.1f}",20,blue)
    text(390,55,f"{state['t']:04.1f} s",20)
    text(595,53,f"人形 B  HP {state['hp'][1]}/3   积分 {state['score'][1]:04.1f}",20,orange)
    y=h+108; title=TITLES[state["state"]]; reason=state["reason"]
    if intro:
        title="躲藏 → 探出标记 → 撤回 → 应对绕侧 → 换位争夺"
        reason="可见观测触发规则决策；轮足加减速与轮子旋转均由脚本驱动。"
    if outro:
        winner=events[-1]["winner"]
        title=f"回合结束｜{'轮足 A 胜出' if winner=='A' else '人形 B 胜出' if winner=='B' else '平局'}（编排展示）"
        reason="对手生命归零，提前结束。" if min(state["hp"])<=0 else "展示回合到时，按积分与生命结算。"
        reason+=f"  积分 {state['score'][0]:.1f} : {state['score'][1]:.1f}；生命 {state['hp'][0]} : {state['hp'][1]}。"
    text(22,y,title,23); text(22,y+36,reason,18,"#bdd2e4")
    text(22,y+68,"规则策略 + 预设通道 / 无训练 / 运动学动画，非接触动力学验证 / 展示赛制",16,"#8da5bd")
    d.rectangle((22,y+103,w-22,y+108),fill="#2c4155")
    d.rectangle((22,y+103,22+(w-44)*state["t"]/60,y+108),fill=blue)
    d.rounded_rectangle((18,110,505,205),radius=9,fill="#132539")
    observed=state["observation_a"] is not None
    text(30,117,f"A 观测：{'当前可见' if observed else '不可见，仅保留旧观测'}   速度 {state['speed'][0]:.2f} m/s",17,blue)
    text(30,146,f"裁判视线：{'畅通' if state['los'] else '被掩体挡住，禁止穿墙标记'}",17)
    text(30,175,f"锁定 A {state['lock'][0]:.2f}/0.80s   B {state['lock'][1]:.2f}/0.80s",17)
    active=f"P{state['point']+1} 激活"
    if any(state["scoring"]): active+=" · "+("A" if state["scoring"][0] else "B")+" 占点计分"
    if state["pause"]>0: active=f"命中冷却 {state['pause']:.1f}s · 仍可移动"
    d.rounded_rectangle((w-325,110,w-18,148),radius=8,fill="#132539")
    text(w-313,117,active,17)
    recent=[e for e in events if e["type"]=="mark" and 0<=state["t"]-e["time"]<2]
    if recent:
        e=recent[-1]; d.rounded_rectangle((w-325,156,w-18,194),radius=8,fill="#62402b")
        text(w-313,163,f"虚拟标记 {e['attacker']} → {e['target']}  HP -1",17,"#fff0d6")
    if state["state"]=="WATCH" and not observed:
        text(24,h+60,"A 不读取墙后实时位置；等待重新看见绕侧对手",17)
    return im

def shot(state):
    if state["t"]<1.2:
        return (-6.7,-5.3,2.8),(-4.1,-2.6,.35),"wheel_detail"
    if state["state"] in ("TAKE_COVER","HIDDEN","PEEK","AIM","WITHDRAW","WATCH"):
        return (-7.8,-8.8,7.8),(-2.4,-1.2,.3),"cover"
    return (10.,-13.,13.),(-.5,-.5,.3),"overview"
