"""Self-contained local review page; native Isaac images stay as linked files."""
import html
import json
from pathlib import Path


def build_report(run, design, result):
    run=Path(run); esc=html.escape;v=result['geometry_validation']
    names={k:p['label'] for k,p in design['cameras'].items()}
    names['collapse_overview']='09 · 坍塌后鸟瞰'
    names['collapse_detail']='10 · 坍塌后近景'
    notes={
        'overview':'32 × 24 米可用区域，中央高架路线与地面绕行形成纵深。机器人以真实资产尺寸作为比例参照。',
        'reverse':'反向观察中央桥、南侧封闭货箱与外围路线。并非 CS / CF 地图复刻。',
        'bridge':'双坡道连接 2.2 米高桥面；约 17.7° 坡度。黄色护栏、抗滑条和桥下支撑均为三维几何。',
        'tunnel':'相机位于 T01 内部，向另一端出口观察。侧壁、顶板、底板分别建模，内部没有封死的碰撞盒。',
        'underpass':'桥下净高 2.02 米，南北向可以从桥下穿行。这里只验证了几何净空，不代表已验证机器人动力学通行。',
        'robot_eye':'约 0.62 米视高观察 P1 接近路线，检查低视角下的掩体辨识与空间尺度。不是比赛中的一次模型决策截图。',
        'north_lane':'北侧贯通集装箱与高低墙组合；箱内路径和箱外绕行提供不同的暴露方式。',
        'top':'俯视核对布局。P1 / P2 的圆环同时画出仅为位置审阅；正式比赛依然只有一个活动点。',
        'collapse_overview':'同一面 High3 墙按原定 90° 角倒下后的静态状态；保持赛制 t=20 秒触发设定。',
        'collapse_detail':'墙体与其碰撞盒一起旋转，形成 2.4 × 2.2 × 0.4 米的局部低障碍。没有新增随机碎石。',
    }
    cards=[];buttons=[]
    for i,(key,label) in enumerate(names.items()):
        if key not in result['outputs']:continue
        buttons.append(f'<button role="tab" aria-selected="{str(i==0).lower()}" onclick="showView(\'{key}\',this)">{esc(label)}</button>')
        cards.append(f'<figure id="{key}" class="view {"active" if i==0 else ""}"><a href="{key}.png" target="_blank"><img src="{key}.png" alt="{esc(label)}" {"" if i==0 else "loading=lazy"}></a><figcaption><b>{esc(label)}</b><span>{esc(notes.get(key,"原生 Isaac 相机截图。"))}</span><a href="{key}.png" download>下载原图 ↗</a></figcaption></figure>')
    route_rows=''.join(f'<tr><td>{esc(k.replace("before_", "坍塌前 · ").replace("after_", "坍塌后 · "))}</td><td>{p["length_m"]:.2f} m</td><td>{p["length_m"]/1.1:.2f} s</td></tr>' for k,p in v['paths'].items())
    refs=''.join(f'<li><a href="{esc(r["url"])}" target="_blank">{esc(r["title"])}</a><small>{esc(r["use"])}</small></li>' for r in design['references'])
    scene=Path(result['scene_directory'])
    tour=''
    if (run/'dockyard_tour.mp4').exists():
        tour='''<section id="tour"><div class="section-head"><p class="eyebrow">02 / CAMERA TOUR</p><h2>绕场镜头演示</h2><p>相机绕场巡览，非比赛回放。所有帧都由 Isaac 渲染；GIF / MP4 仅做编码。</p></div><video controls muted loop playsinline preload="metadata" poster="overview.png" src="dockyard_tour.mp4"></video><p><a class="pill" href="dockyard_tour.gif">打开 GIF</a> <a class="pill" href="dockyard_tour.mp4" download>下载 MP4</a></p></section>'''
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DOCK 07 · 新地图审阅</title><style>
:root{color-scheme:dark;--bg:#111719;--panel:#192124;--text:#e9e8df;--muted:#a4b2b5;--line:#344245;--gold:#dfb265;--cyan:#86c4c4}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}a{color:var(--gold);text-decoration:none}a:hover{text-decoration:underline}.wrap{max-width:1320px;margin:auto;padding:0 32px}header{padding:52px 0 28px;border-bottom:1px solid var(--line)}.eyebrow{font-size:12px;letter-spacing:3px;color:var(--gold);font-weight:700;margin:0 0 10px}h1{font-size:clamp(32px,5vw,65px);line-height:1.12;letter-spacing:-2px;margin:12px 0 18px}h1 span{font-weight:400;color:var(--muted);font-size:.48em;letter-spacing:0;display:block;margin-top:16px}h2{font-size:29px;line-height:1.35;margin:8px 0 14px}h3{font-size:20px;margin:0 0 9px}p{margin:8px 0 16px}.lead{color:var(--muted);max-width:900px}.tag{display:inline-block;font-size:12px;border:1px solid var(--line);padding:4px 11px;border-radius:20px;color:var(--cyan);margin:0 5px 6px 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);margin:26px 0;gap:12px}.metric{padding:16px 20px;border-left:2px solid var(--gold);background:var(--panel)}.metric b{font-size:28px;font-variant-numeric:tabular-nums;display:block}.metric span{color:var(--muted);font-size:13px}.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}button{background:transparent;border:1px solid var(--line);border-radius:5px;color:var(--muted);padding:9px 13px;font:inherit;font-size:13px;cursor:pointer}button[aria-selected=true]{background:var(--gold);color:#192124;border-color:var(--gold)}.view{display:none;margin:0;border:1px solid var(--line);background:var(--panel)}.view.active{display:block}img{display:block;width:100%;height:auto}figcaption{padding:20px 24px}figcaption b{display:block;font-size:18px}figcaption span{display:block;color:var(--muted);margin:7px 0}figcaption a{font-size:13px}section{padding:40px 0;border-bottom:1px solid var(--line)}.section-head{max-width:950px}.section-head p:not(.eyebrow){color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{padding:24px;background:var(--panel);border:1px solid var(--line)}.card p,.card li{color:var(--muted)}.notice{padding:18px 22px;border-left:3px solid var(--gold);background:#27251e;margin:22px 0;color:#d7d2bf}table{width:100%;border-collapse:collapse;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:10px;text-align:left}th{color:var(--muted);font-weight:500}td:last-child{font-variant-numeric:tabular-nums}.pill{display:inline-block;border:1px solid var(--line);padding:7px 15px;border-radius:4px}video{width:100%;background:#000;display:block}small{display:block;color:var(--muted)}li{margin:8px 0}code{font-size:12px;overflow-wrap:anywhere}footer{padding:30px 0 50px;color:var(--muted);font-size:12px}details{margin-top:16px}summary{cursor:pointer;color:var(--gold)}@media(max-width:800px){.wrap{padding:0 16px}.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}header{padding-top:30px}section{padding:28px 0}figcaption{padding:15px}h1{letter-spacing:-1px}}
</style></head><body><div class="wrap"><header><p class="eyebrow">ISAAC SIM / ENVIRONMENT DESIGN REVIEW / V1</p><h1>DOCK 07<span>集装箱维修货场 · 大地图审阅版</span></h1><p class="lead">沿用原比赛机制，把舞台扩展成有高低差、洞内遮蔽和多路径绕行的工业货场。本轮交付地图、原生相机视图和绕场演示，等待审阅后再接入比赛控制器。</p><div><span class="tag">原创布局 · 非游戏地图复刻</span><span class="tag">Isaac RTX Path Tracing</span><span class="tag">PBR 颜色 / 法线 / 粗糙度</span><span class="tag">未调用决策模型</span></div><div class="metrics"><div class="metric"><b>32 × 24 m</b><span>768 m² · 原可用面积的 8 倍</span></div><div class="metric"><b>8 + 6</b><span>高墙 / 低掩体</span></div><div class="metric"><b>3 / 5</b><span>贯通集装箱 / 全部箱体</span></div><div class="metric"><b>2.20 m</b><span>桥面高度 · 双向坡道</span></div></div></header>
<section><div class="section-head"><p class="eyebrow">01 / NATIVE ISAAC CAMERAS</p><h2>先看地图</h2><p>切换视角检查布局、通道和材质；点击图片打开全尺寸原图。画面中的机器人是静态比例参照，不是正在进行的比赛。</p></div><nav class="tabs" role="tablist">__BUTTONS__</nav>__CARDS__</section>
__TOUR__
<section><p class="eyebrow">03 / DESIGN INTENT</p><h2>空间不是单纯放大，路线有不同代价</h2><div class="grid"><article class="card"><h3>中路：桥上暴露，桥下遮顶</h3><p>中央东西向桥面高 2.20 米，桥宽 2.40 米；两侧各 6.80 米坡道，坡度约 17.74°。南北向保留桥下通路，净高 2.02 米。同一个 XY 区域因此有上下两层，后续可以产生高度相关的观察与射击差异。</p></article><article class="card"><h3>侧路：进入掩体，也承担出口风险</h3><p>西侧 T01、东侧 T02、北侧 T03 为两端开口集装箱，外宽 2.60 米、外高 2.60 米。内壁净宽约 2.32 米，门头下净高约 2.37 米。洞内有遮顶和侧向遮蔽，但两个洞口仍暴露，并非无敌避难所。</p></article><article class="card"><h3>高低掩体：让路线与机体高度有关</h3><p>8 面高墙约 2.0–2.2 米，6 个低掩体约 0.6–0.8 米。低掩体对四足和人形的遮挡效果不应被默认当成一致；接比赛时必须按实际射击参考点和三维视线处理。两只封闭货箱提供完整绕行障碍。</p></article><article class="card"><h3>可读性与工业风格</h3><p>用青蓝 / 铁锈红 / 土黄区分货箱，黄色护栏和边线提示结构边缘。混凝土模板缝、金属波纹、接缝、排水沟、维修库和围栏形成尺度参照。贴图来自 CC0 材质库；不复制 CS / CF 的商业资产。</p></article></div></section>
<section><p class="eyebrow">04 / RULES & READINESS</p><h2>规则冻结，工程接入边界明确</h2><div class="grid"><article class="card"><h3>本轮不改比赛规则</h3><p>已逐字节保留最近完整测试的规则快照。仍保留 40 秒上限、双方 3 HP、占点驻留与连续得分、首次触发后 5 秒轮换、受击中断、正前总计 60°、8 米射程、连续瞄准命中机制、仅命中方冷却等既有设置。</p><p>保留固定墙在 t=20 秒、固定 90° 坍塌的设定。预览展示坍塌前后两个静态状态，没有随机破碎，也没有运行一次正式比赛。</p><a href="rules_reference.json">查看原规则快照</a> · <a href="map_design.json">查看地图参数</a></article><article class="card"><h3>完成地图 ≠ 完成三维比赛适配</h3><p>旧控制器使用二维障碍与导航。新地图必须接入上下层导航、坡道参考高度 / 姿态、三维视线遮挡和相机位姿，才能公平地跑同一套赛制。本轮没有把桥上桥下错误地压成一个不可通行平面，也没有宣称已完成适配。</p><p>已建立结构碰撞并检查通道净空 / 地面连通性；尚未测试足端接触、牵引、跌落和整机动力学。桥栏有碰撞，细小波纹、标牌等美术细节没有独立碰撞。</p></article></div><div class="notice">尺寸取舍：保持原四足速度 1.10 m/s，P1→P2 的保守地面路线约 20.40 m，对应约 18.55 s 纯行进时间，尚未加入躲避、瞄准和转向。因此新图会明显增加转点成本，并可能降低原 40 秒赛制内的刷新次数；没有为得到好看结果而暗改速度或时长。</div><details><summary>展开几何验证与路线长度</summary><p>__CHECKS__ 项断言通过。地面连通检查使用 0.25 m 网格、0.42 m 膨胀半径和 1.45 m 代表高度；不使用桥上路线，非全身运动可行性证明。起终点吸附到网格，因此下表是设计检查值，不是实测比赛耗时。人形路径一栏也统一用四足速度作尺度参照。</p><table><thead><tr><th>地面路线</th><th>网格路线长度</th><th>按 1.10 m/s 的行进估算</th></tr></thead><tbody>__ROUTES__</tbody></table><p><a href="geometry_validation.json">完整验证记录</a> · <a href="art_manifest.json">碰撞 / 材质 / 渲染清单</a></p></details></section>
<section><p class="eyebrow">05 / REVIEW CHECKLIST & FILES</p><h2>这一版建议你重点审阅</h2><ol><li>32 × 24 米是否达到你希望的“更大”，还是希望中央交战区更紧凑？</li><li>贯通箱体、坡道桥、桥下路线的数量和摆放是否合适？</li><li>工业货场的色调、光照和贴图质感是否符合演示方向？</li><li>是否保留目前较高的转点成本，再让规划模型面对这项挑战？</li></ol><p><a class="pill" href="__BEFORE__">打开 / 下载坍塌前 USD</a> <a class="pill" href="__AFTER__">打开 / 下载坍塌后 USD</a> <a class="pill" href="result.json">构建结果</a></p><p><small>USD 是本机可复用场景，含绝对材质与机器人资产路径；迁移其他电脑时需一起打包并重定向依赖。网页可本机离线打开，不需要外部脚本或 CDN。</small></p><details><summary>设计参考与材质许可</summary><ul>__REFS__</ul><p>借鉴的是可读性、掩体路径和集装箱通道这些通用原则；布局、结构与标识为本项目原创。材质来源与 SHA-256 在 art_manifest.json 记录。</p></details></section>
<footer>__RUN__ · __ENGINE__ · __RES__ · 本轮无训练 / 无模型调用 / 无比赛成绩。<br>原生图片与相机巡览均来自本次 USD 场景；HTML 只负责展示。</footer></div><script>function showView(id,button){document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('[role=tab]').forEach(x=>x.setAttribute('aria-selected',String(x===button)));}</script></body></html>'''
    for token,value in {'BUTTONS':''.join(buttons),'CARDS':''.join(cards),'TOUR':tour,'CHECKS':str(v['checks']),
        'ROUTES':route_rows,'BEFORE':str(scene/'dockyard_before.usda'),'AFTER':str(scene/'dockyard_after.usda'),
        'REFS':refs,'RUN':esc(run.name),'ENGINE':'Isaac Sim RTX Path Tracing',
        'RES':' × '.join(map(str,result['validation']['resolution']))+' px'}.items():page=page.replace('__'+token+'__',value)
    (run/'review.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    import sys
    run=Path(sys.argv[1]);build_report(run,json.loads((run/'map_design.json').read_text()),json.loads((run/'result.json').read_text()))
