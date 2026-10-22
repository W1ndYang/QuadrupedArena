# 四足掩体挑战赛 · Isaac 原型

> 历史记录归档：以下路径、命令和相对链接针对原作者的工作站，相关实验数据未随 Git 分发。
> 新机安装、当前规则和入口以仓库根目录 README 及 docs/INSTALL.md、docs/RULES.md 为准。
> 原记录中含有不同阶段的信息权限和规则，不应将所有段落理解为同一个实验协议。

独立赛事工程。沿用本机已经部署并验证过的 Isaac Sim 5.1.0 / Isaac Lab v2.3.2，
Python 3.11.15，PyTorch 2.7.0+cu128。

## DOCK 07 大地图审阅版（2026-10-22）

[地图审阅网页：10 个原生相机视角 + 绕场 GIF/MP4](/home/yang/workspace/experiments/runs/quadruped-arena/20261022-153300-036600/review.html)。
新场地为原创集装箱维修货场，实际可用范围 32×24 米（原来的 8 倍），包含 8 面高墙、6 个低掩体、
3 个双端贯通集装箱、2 个封闭货箱、2.2 米高双坡道桥和桥下通路。混凝土 / 沥青复用本机 CC0 PBR，
新增 ambientCG Metal032 金属贴图，所有结构与截图均由本机 Isaac/USD 构建渲染，不使用 AI 生图。

本轮仅构建地图、检查几何和出图，**没有运行比赛或调用模型 API**。旧赛制配置未修改，
最近完整比赛的原始规则按字节保存为 `rules_reference.json`。固定墙在 20 秒 / 90° 倒下的机制保留，
交付坍塌前后两个 USD 快照。网页的两个点圈仅表示候选位置，不意味着同时激活两个点。
旧控制器是二维导航；本地图接比赛前仍需分层导航、坡道高度/姿态和三维视线适配。
碰撞结构已建立，但没有验证足端接触或整机动力学通行。

保守地面路径检查：P1→P2 约 20.40 米，按原四足 1.10 m/s 约 18.55 秒纯行进；在原 40 秒赛制下会增加转点压力。
地图参数为 [configs/dockyard_v1.json](configs/dockyard_v1.json)，结构/美术/出图入口为 `dockyard_geometry.py`、
`dockyard_art.py`、`dockyard_scene.py`。每次运行独立保存 source_snapshot、材质哈希、规则快照和验证记录。
USD 位于 `workspace/simulators/scenes/quadruped-arena/<运行时间>/dockyard_before.usda` 及 `dockyard_after.usda`，
引用本机绝对资产路径，跨机使用须打包并重定向依赖。旧实验文件和旧场景不覆盖。

```bash
bash run.sh --dockyard --headless                    # 高清截图 + 9 秒相机巡览
bash run.sh --dockyard --preview --no-tour --headless # 快速审阅截图
bash run.sh --dockyard --no-tour --interactive       # 出图后保留 Isaac 窗口
/home/yang/miniforge3/envs/isaaclab-go2/bin/python -m unittest discover -s scripts -p test_dockyard_geometry.py -v
```

本轮 7 项新增测试、344 项几何断言；全套 192 项离线测试通过。坡道约 17.74°，桥下净高 2.02 米；
贯通箱体独立侧板 / 顶板 / 底板 / 门头碰撞，不用实心碰撞盒替代内部空间。

## DeepSeek max 放宽输出额度重测（2026-10-22）

[本轮截图、解释、机载及第三视角报告](/home/yang/workspace/experiments/runs/quadruped-arena/20261022-095614-313039/report.html)。本轮正常终局：**13.22 分、39.26 秒四足淘汰、终局 HP [0,3]**；20 次 API 均 `stop`，零重试。平均 API 43.48 秒，最长 147.33 秒；总 255,034 tokens，其中 reasoning 147,451。两次输出分别 17,520 和 26,319 tokens，均超过旧 16,384 额度但正常完成。

用户要求不加实验 token 消耗上限：官方 API 没有真正无限值，故使用平台最大 `max_tokens=393216`（384K，包含推理；不传则 max 档默认128K），不设置实验累计 token 预算；取消客户端 API 和启动器总墙钟截止，保留 600 秒网络无数据超时。独立配置 [deepseek_max_platform_limit.json](configs/deepseek_max_platform_limit.json)，原 low 默认配置不变。依据：[DeepSeek 官方 Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)。

背景提示词、规则/地图/种子、语义技能与运动视觉核心代码与旧 max 哈希一致。旧 max 在 28 秒中断，10.56 分不是终局成绩；新一局不是续跑，也不是单种子统计优越性证明。按用户后续要求，科研对比报告已原位替换主项 V-DH 为本轮，旧中断以 DH16 保留历史；本轮对照记于 `budget_rerun_comparison.json`。两次脚点中一次明显偏离，赛后误差约3.417米/0.091米，详见 `annotation_audit.json` 与追加复盘。运行前183项、最终185项离线测试通过。

复现（会产生官方 API 费用，最多20次决策，单次输出仅受平台最大额度约束）：

```bash
ARENA_DEEPSEEK_CONFIG=/home/yang/workspace/projects/quadruped-arena/configs/deepseek_max_platform_limit.json bash run.sh --vision-trial --semantic-skills --deepseek --headless
```

## 全知 / 非全知科研对比报告（2026-10-22）

[打开完整 HTML 报告](/home/yang/workspace/experiments/reports/quadruped-arena/20261022_全知与非全知_科研对比/index.html)：两代协议、7 个完整终局 LLM 条件、规则/A*参考及7个历史记录；全部 105 个已接受决策的原始截图/JSON 浏览器，6 组 SVG/PDF/600 dpi PNG 科研图，以及 CSV、来源哈希、方法与局限。DS max 主项现为 `20261022-095614-313039`，13.22分、39.26秒、20步；截图、完整回放、统计和离线包同步更新。此次只分析已有记录，没有新比赛、API调用或渲染；保留原报告备份及 `data/revision.json` 替换记录。

每条件仅一局，不能作为统计排名。新 DS max 使用393216输出额度、无总墙钟截止，low仍16384/180秒，不是纯思考档位消融。旧 max 的28秒中断、10.56已观察分及181,049 tokens成本移入DH16历史，不合并到新局统计。助手具有历史工程上下文；全知与视觉组信息/技能接口不同。所有这些差异在报告中单列。复现分析使用 `scripts/research_data.py`、`scripts/research_figures.py`、`scripts/research_report.py`；验证入口 `scripts/research_validate.py` 和 `scripts/check_research_report.mjs`。

## 最新试验：LLM 每 2 秒闭环规划

**2026-10-21 三轮连续非全知视觉测试完成**：同地图、种子 `2026102107`、规则、语义技能 1.1、2 秒决策 / 0.5 秒机载采样，保留四字段 JSON 与逐帧依据。进入 [三轮统一对照页](/home/yang/workspace/experiments/runs/quadruped-arena/20261021_三轮对照_Qwen思考开关_助手/index.html)，切换查看原始截图、决策解释、耗时、机载与第三视角 GIF。裁判与运动仍为 Python 运动学，Isaac 在线提供机载 RGB；第三视角为终局后保存轨迹重渲染，不重新调用模型或重跑策略。

| 决策层 | 得分 | 淘汰时刻 | 决策数 | 平均 API 耗时 | 总 Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B 开启思考、默认最大预算 | 4 | 24.88 秒 | 13 | 47.55 秒 | 132,070 |
| Qwen3.6-27B 关闭思考 | 4 | 16.80 秒 | 9 | 9.09 秒 | 55,712 |
| 当前助手会话 | 12 | 32.94 秒 | 17 | 不可获取 | 不可获取 |

对应运行：`20261021-205454-693959`、`20261021-210646-052984`、`20261021-211006-627504`。助手平均端到端等待 48.07 秒、中位 33.57 秒，包含工具、读图和交互，不能直接与云 API 推理时长比较；每步开始/接受时间和 CSV 均保留。Qwen 最高/最低不是原生 `reasoning_effort` 档位，而是 `enable_thinking=true` 且不传 `thinking_budget` / `enable_thinking=false`；两轮均回答上限 16384、temperature=0.6、600 秒截止，最多 20 次请求，零自动重试。依据：[百炼官方接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)、[深度思考说明](https://help.aliyun.com/zh/model-studio/deep-thinking)。配置 `configs/qwen_vision_trial.json`；独立于旧全知 `qwen.sh run`，Key 不进入项目或快照。

Qwen 两轮各 3 次脚点坐标均明显偏离脚部，具体截图坐标记于各轮 `review_addendum.json`；仅有可见性描述和成功命中不代表定位正确。助手完成 P1→P2→P1 三次各 4 分计分；使用中间帧标注两次、两次圈内 OBSERVE，终局仍因转移期间未切断追击射线而淘汰。助手保留工程背景及历史结果，仅当前比赛按观测白名单决策，不是同上下文盲测。单种子单次结果不足以证明稳定模型排名。

另外完整保留一次接入失败 `20261021-204904-009504`：第 2 次 Qwen 请求返回代码围栏包装的完整 JSON，旧解析器拒绝，比赛止于 2 秒。修复仅接受单个完整围栏、启用 JSON 请求格式后从头重新执行正式轮；原始响应与失败不覆盖、不手工改策略。额外 2 次调用、16,058 tokens 单列，三轮正式 Qwen 总量 187,782，加接入失败共 203,840 tokens；助手用量未知。180 项离线测试通过。

复现入口（前两条会发送图片至百炼官方并产生 API 费用；第三条等待外部逐步提交动作）：

```bash
bash run.sh --vision-trial --semantic-skills --qwen-vision max --headless
bash run.sh --vision-trial --semantic-skills --qwen-vision min --headless
bash run.sh --vision-trial --semantic-skills --assessment --headless
```

**DeepSeek low＋逐帧依据复测已完整结束**：`20261021-203042-903738`。相对上一轮max＋assessment，仅将API思考档位改回low；同一系统提示词、地图、种子、技能、16384输出预算和180秒超时。10次决策、39帧，18.84秒四足淘汰，4分，人形剩2HP；未到20秒坍塌。API平均23.67秒、中位19.61秒、最慢64.97秒；总87,968 tokens，其中推理36,186。相比上轮max成功请求平均35.37秒，本次均值低约33.1%，但单次闭环轨迹不同，不能声称稳定提速或策略更优。

[low解释模式报告：截图＋依据＋JSON＋复盘](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-203042-903738/report.html) · [机载GIF](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-203042-903738/onboard.gif) · [截图抽查](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-203042-903738/visual_spotcheck.json)。8次占点、1次躲藏、1次移动，3次脚点标注；11条人形可见断言涉及10张不同图片，逐张核对存在人形，但未全面核验所有否定帧或描述细节。10.70秒底层依据新鲜视觉记忆自动命中，不是高层调用AIM技能。12秒看到脚点却未刷新记忆；16/18秒路线均被UNSAFE_ROUTE拒绝后停留，18.84秒淘汰。暴露了视觉更新、拒绝后避险及文字意图与底层自动攻击语义之间的缺口。本轮未修改策略补救、未重试；175项离线测试通过。赛后附加分析独立保存于review_addendum.json，重新生成报告可保留，原始动作与快照不改写。当前默认保留low＋assessment。

**DeepSeek max＋逐帧依据复测**：`20261021-200759-757048`。按用户要求将输出改为 observation_id/skill/params/assessment，assessment分别记录每帧可见事实、公开状态、简短取舍和未知；不要求内部思维链。仍max、16384输出预算、180秒超时、同地图/种子/技能，不自动重试；历史仍只传最近4条动作，不累积解释。14次决策执行到28秒，暂时10.56分、2HP；第15次请求89.42秒后返回length，16384输出tokens全为reasoning、content为空，故中止而非完赛。已完成请求平均35.37秒；全部15次共181,049 tokens，其中推理102,157。首次加入中断后导出轨迹，不再丢失已执行过程；真值仍不提供给规划器。

[解释模式报告：截图＋逐帧依据＋JSON](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-200759-757048/report.html) · [人工截图抽查记录](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-200759-757048/visual_spotcheck.json)。4次“人形可见”断言均有截图支持；7.5/9.5秒脚部可见却未提交sighting，20.5/22.5秒脚点遮挡说明了仅脚点定位的局限。另发现10秒坍塌窗口等规则误述。12次占点、2次提前转移，无主动观察/躲藏/瞄准、无脚点标注。不能由解释自述证明内部推理质量，也不能将28秒暂时得分当作最终成绩。

**DeepSeek Flash max 首测中断（不是完赛）**：`20261021-195343-236794`，仅改变思考档至max，并将输出预算4096→16384、超时120→180秒；地图、提示词、种子、图片、技能与三字段决策不变。9次决策已执行至18秒，暂时5.76分、四足2HP；第10次请求等待147.09秒后HTTP流发生IncompleteRead，无完整返回，没有重试。不能据此判定模型淘汰或比较最终得分；本地日志无法区分服务端或网络断流原因。已完成9次API均值36.93秒、中位9.16秒、最长87.20秒；已知61,962 tokens，其中推理20,892；失败请求usage未知且可能计费，不能视为0。9次均选择CAPTURE_POINT，无视觉脚底标注。

[max中断报告：截图＋已执行决策＋耗时](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-195343-236794/report.html) · [中断统计](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-195343-236794/test_summary.json)。该次配置快照保留max；旧low运行快照、报告和回放不变。原始失败result.json保留。中断前机载GIF只覆盖0～18秒；未导出完整裁判轨迹，不能提供该次完整第三视角回放。

**DeepSeek Flash low 首次完整非全知测试已完成**：`20261021-193327-596027`，14次API调用、57张在线Isaac机载图，8分、27.96秒四足淘汰，人形剩3HP。每步严格三字段决策，无理由、无自动重试。API平均7.405秒/中位6.457秒/范围2.236～21.546秒，端到端决策等待平均7.425秒、底层编译0.232秒；仍不满足2秒实时控制。总78,663 tokens（输入64,899，输出13,764，其中推理13,386）。14次均选择CAPTURE_POINT、没有提交视觉目标脚底标注；不能据此声称已有主动观察/躲藏/反击博弈能力。技能层仍执行自身受击保护。原始结果不回滚，官方版本为滚动别名，当前响应model为deepseek-flash。

[本轮截图＋决策JSON＋外部计时报告](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-193327-596027/report.html) · [机载GIF](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-193327-596027/onboard.gif) · [统计JSON](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-193327-596027/test_summary.json)。与助手上一轮59.1秒、4分相比仅能作描述性比较：自动API、短上下文和修复版技能等同时变化，不是模型单变量对照。

已补齐同一场比赛的[第三视角GIF](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-194156-510827/demo.gif)和[MP4](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-194156-510827/demo.mp4)，并加入原报告“第三视角回放”选项卡。Isaac工业场景路径追踪，12fps，约36秒（含片头/片尾/坍塌展示镜头）；没有重新运行比赛或调用API，全场HUD仅供赛后展示，非模型观测。输出目录result.json包含原轨迹SHA256。

**DeepSeek 官方 Flash 接入已配置，文字连通检查通过（2026-10-21）**：配置文件 `configs/deepseek_trial.json`，官方地址 `https://api.deepseek.com`，模型 `deepseek-flash`，文档当前对应 DeepSeek-V4.1-Flash-1017。思考保持enabled；首次用low，测试max后，当前按用户要求改回low并保留简短依据输出。low不是关闭思考，max为最高档；运行记录、网页和回放字幕均读取该次协议的档位。别名可能随服务更新，运行时记录响应model和配置日期。依据：[官方模型](https://api-docs.deepseek.com/)、[思考档位](https://api-docs.deepseek.com/guides/thinking_mode/)、[图片接口](https://api-docs.deepseek.com/guides/vision/)。

在本机终端运行 `bash deepseek.sh setup` 隐藏输入官方API Key，保存在 `/home/yang/.config/quadruped-arena/deepseek.key`（0600、与百炼分开），不写入工程/运行快照/报告。`status`只检查配置，不显示密钥；`prepare`离线导出背景提示词和配置，均不联网。`probe`明确发送一次付费JSON连通请求；`run`启动付费完整Isaac在线RGB比赛，最多20次模型请求，输出预算/超时读取配置（当前16384 tokens/180秒，首次low为4096/120）；不重试、不自动修复或回退策略。已完成一次真实文字probe：`20261021-193212-669855_DeepSeek_API检查`，总耗时1.3128秒、输入67/输出25/总92 tokens，其中推理19 tokens。此数值不是多帧视觉或比赛决策性能；试验结果见上。

DeepSeek输入为公开规则地图、仅初始敌方位置、当前1～5张原始768×432机载图、自状态/公开计分状态/视觉记忆、最近4条短动作；**不沿用Qwen全知提示词，不给当前敌方真值或策略源码**。截图和上述公开数据会发送至DeepSeek官方云API。保持2秒决策、0.5秒采样、修复后五技能接口1.1；比赛等待API时暂停。每步只请求一次JSON：执行字段为observation_id、skill、params，当前explain_decisions=true时另加assessment，记录逐帧视觉事实、公开状态依据、决策摘要和不确定性；不要求内部思维链或自报耗时。可标注图中可见脚底像素供代码估计位置。图片detail为original（避免坐标标注因低清缩放混淆），低档设置针对思考强度。

外部程序自动记录原始动作、请求图片哈希、API开始/结束UTC时间戳（api_started_utc/api_finished_utc）、单调时钟总时长、首输出/首正文时间、观测至接受耗时、底层规划时间和提供方usage（输入/输出/总量及可用的reasoning tokens）；不保存内部思维链。API总时长含网络和服务端排队/计算/输出，不是纯思考时长；去掉解释不保证内部思考同比减少，仍用low档。结束后生成同款截图＋JSON选项卡报告/GIF/CSV。因上下文长度、自动调用方式、技能修复等差异，不能把与旧助手结果的差异全归因于模型。`python -m unittest discover -s scripts -p 'test_deepseek_trial.py'`为离线测试，不调用API。

2026-10-21 **五个语义技能＋短指令首测**：`20261021-190339-279787`。接口为 `observation_id / skill / params / reason`，理由最多100字符；`MOVE_TO / TAKE_COVER / AIM_AT_TRACK / CAPTURE_POINT / OBSERVE` 已全部实跑。底层负责候选路径、掩体、转向与加减速估计、倒退、云台和自身状态保护；视觉位置来自助手对当前截图的脚底像素标注，不是自动检测器或隐藏坐标。保留2秒高层周期、0.5秒Isaac在线采样、Python运动学裁判，没有接入DeepSeek或付费API。

首轮9次决策、37帧、17.64秒淘汰、4分。平均端到端等待59.10秒（上一轮62.88秒），底层规划0.349秒；**不能声称明显提速或策略提升**，轨迹与执行接口不相同，不是严格单变量对照。Token不可获取。原始失败与当时源码保留；赛后修复OBSERVE方向被记忆覆盖、危险转移不拒绝和占点受击只停车的问题。157项测试通过，另在 `20261021-191611-741975` 完成2秒Isaac朝向工程复测（机身90°）；修复版尚未重跑完整比赛。

[技能首测：截图＋短JSON＋复盘](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-190339-279787/report.html) · [机载GIF](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-190339-279787/onboard.gif)。正式启动：`bash run.sh --vision-trial --semantic-skills --headless`，仍需外部逐步提交动作，并非自动助手。工程预检添加 `--vision-smoke --smoke-skill OBSERVE`。仅以公开观测008重检，修复后对原末步占点指令返回 `UNSAFE_ROUTE`；这是局部检查，不是反事实获胜证明。详见运行目录 `postfix_validation.json`。

2026-10-21 新增 **运动学＋Isaac在线多帧非全知视觉** 正式比赛：`20261021-173528-369598`，16次助手决策、30.62秒、63张原生机载RGB、7.82分，四足淘汰。每0.5秒采样，默认每步上一边界帧＋4张新帧；云台可扫描/保持，±90°限位、120°/秒限速，扫描消耗比赛时间。运动学/裁判仍在Python，Isaac不运行接触动力学。对手动态真值不进入决策观测，禁用`AIM`和`facing: enemy`；图像是本次动作前在线渲染，不是事后回放截图。

[多帧截图＋JSON报告](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-173528-369598/report.html) · [机载GIF](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-173528-369598/onboard.gif)。`bash run.sh --vision-trial --vision-smoke --headless`做2秒工程预检；去掉`--vision-smoke`启动正式暂停接口，等待外部逐步提交`actions/NNN.json`（不是自动策略）。观测在`observations/NNN.json`，七字段示例见该次动作目录。每步最多等待1200秒、整体上限5400秒；超时不伪造后续决策。`scripts/vision_report.py RUN_DIR`在终局后生成审计与报告。等待合计1006.14秒，含读图和工具开销；token不可获取。本轮是同一开发会话的接口隔离验证，不是全新上下文严格盲测；失败和原始记录均保留，不能由单轮宣称策略优越。

Qwen3.6-27B（百炼北京）已完成第二轮真实 API 闭环：16 次决策，31.64 秒淘汰、10.4 分，565031 tokens。首轮工具循环失败记录保留，第二轮补全背景并强制限制工具轮数；不是端侧实时性验证。`bash qwen.sh setup` 隐藏输入 Key，`probe` 检查连通性，`run` 启动新比赛（可能计费）。Key 与项目/报告隔离。详见 [配置与实验协议](docs/QWEN_API_TRIAL.md)。

Isaac 物理接入第一阶段（2026-10-21）：`bash run.sh --embodied-probe --headless` 做实际 Go2-W 关节驱动和机载 RGB 验收。17 秒基础测试已通过；另在 `20261021-172006-618380` 完成助手读取机载图片后逐步提交的 3×2 秒 MOVE/TURN/STOP 联调，记录见该运行目录的 `test_summary.json`。这是静态视觉标靶工程测试，**不是完整非全知比赛**；尚无动态人形、射击占点、坍塌闭环。低速转向仍有误差，物理资产有辅助链节质量/惯量警告，需要进一步验证。旧比赛路径仍是 CPU 运动学裁判＋Isaac 事后回放，不能与新物理验收混称。

已完成首轮全知当前状态的外部 LLM 技能调用：20 次决策、40 秒比赛，12 分，四足剩 1 HP。原规则与人形策略不变，决策期间比赛暂停；没有训练或未来试跑。每步观测、动作、结果、端到端耗时及 token 不可用说明均已存档。

[图文报告](/home/yang/workspace/experiments/runs/quadruped-arena/20261021_LLM首轮_2秒闭环_全知观测/report.html) · [试验说明与复现接口](docs/LLM_FIRST_TRIAL.md)。同种子旧策略参考为 8 分、21.86 秒淘汰；单次且观测/控制接口不完全相等，不构成通用优越性证明。闭环是 CPU 裁判 + 运动学技能，Isaac 承担原生工业场景回放，并非已接入接触动力学。

`bash run.sh --demo --replay-dir <已完成比赛目录> --industrial --headless` 可把保存轨迹在工业场景重渲染，不重算比赛。输出源轨迹 SHA256 供核验。

## 最新规则：5 秒轮换占点 + 人形快速转向

本轮已归档到 [20261021_规则v0.4_5秒轮换占点_人形快速转向](/home/yang/workspace/experiments/runs/quadruped-arena/20261021_规则v0.4_5秒轮换占点_人形快速转向/)，入口为 [测试记录](/home/yang/workspace/experiments/runs/quadruped-arena/20261021_规则v0.4_5秒轮换占点_人形快速转向/测试记录.md)，包含 GIF、结果和后续新场景重渲染待办。

`bash run.sh --demo --rotating-targets --headless`：每次只激活一个点，四足首次进圈后 5 秒失效；超时或圈内受击后依 P1 → P2 → P1 循环刷新。人形转向上限改为 3.0 rad/s，搜索扫视 2.4 rad/s，仍受视野与遮挡约束。旧命令/配置保留复现。详见 [v0.4 规则、参数、首轮观察](docs/rules_v0_3/ROTATING_TARGETS_V04.md)。

## 最新视觉场景：工业战术训练场

`bash run.sh --art --headless`：真实 PBR 材质、混凝土高低掩体、围墙铁网、控制室、工业设备与标线，输出四张原生 Isaac 高清渲染及坍塌前后 USD。视觉围合扩至 18×14 m，原 12×8 m 比赛核心区和规则保持不变。独立美术分支，不自动替换旧比赛 GIF。

详见 [美术设计、素材许可、打开与复现](docs/ART_DIRECTION_V1.md)。可用 `bash run.sh --art --preview --interactive` 打开窗口查看。

## 最新策略：有限感知人形 + 风险绕行轮足

`bash run.sh --demo --rules-v3 --adaptive --headless`：人形只通过 120° 理想视野和遮挡检查发现四足，丢失后搜索最后观测位置，不再获得隐藏坐标。四足保留 R0 人形位置/朝向信息，增加暴露风险路径代价、主动避险、倒退、冷却窗口转移和安全时扫视。射击仍为总计 60°，赛规和地图沿用首轮。

详见 [策略说明、配对对照与局限](docs/rules_v0_3/POLICY_UPDATE.md)、[新增策略配置](configs/trial_adaptive.json)。66 项自动测试通过；5 个配对种子只有探索意义，不能代替统计验收。新策略更会保命，但 P2 转移偏保守，不保证每个种子得分更高。未训练，未验证接触动力学。

## 2026-10-21 R0 首轮规则试验

新版方向为“人形攻击施压、评价四足占点得分”，采用独立攻击冷却、圈内受击后该点整局失效、多个固定目标点，以及一次固定墙体坍塌。详见 [规则规格 v0.3](docs/rules_v0_3/RULES.md)、[配置草案](docs/rules_v0_3/rules_draft.json) 和 [验收案例](docs/rules_v0_3/ACCEPTANCE_CASES.md)。

新入口 `bash run.sh --demo --rules-v3 --headless` 已接入独立裁判、平地高低墙、固定倾倒事件、已知地图 A* 轮足基线和攻击优先人形决策树。使用 [首轮参数配置](configs/trial_v3.json)，详见 [参数、复现方式、实测结果与限制](docs/rules_v0_3/FIRST_TRIAL.md)。首轮 40 秒上限，第 20 秒倾倒；双方 3 HP，驻留 1 秒后每秒 1 分，命中方独立冷却 2 秒。

原 `rules_draft.json` 仍是设计草案，不能直接传入 Demo。自动测试 38 项通过，不等同于设计文档 64 项案例全部验收。仍为无训练、运动学演示；不声称平衡性或接触动力学已验证。下文保留历史场景与旧 Demo 入口，其行为不自动升级为新规则。

## 目录与环境

| 用途 | 路径（相对 `/home/yang/workspace`） |
| --- | --- |
| 本项目代码、配置、说明 | `projects/quadruped-arena/` |
| Isaac Lab 源码（现有，共用） | `projects/unitree/IsaacLab/` |
| 场地 USD，每次运行单独生成 | `simulators/scenes/quadruped-arena/<运行时间>/` |
| 可选 Go2 转换资产 | `simulators/assets/quadruped-arena/go2/` |
| 日志、结果 JSON、渲染图 | `experiments/runs/quadruped-arena/<运行时间>/` |
| 后续模型权重、训练检查点 | `models/pretrained/`、`models/checkpoints/` |
| 环境与实测记录 | `notes/setup/QUADRUPED_ARENA_2026-10-19.md` |
| 原始交接材料 | `quadruped-sim-handoff/` |

解释器沿用 `/home/yang/miniforge3/envs/isaaclab-go2/bin/python`，与系统 Python 隔离。
本工程当前与原 Go2 工程共用依赖，不是一个新建的独立 Conda 环境。
启动器仅设置子进程环境。未来需要改变依赖版本时，再克隆专用环境，避免影响既有工程。
`ARENA_PYTHON` 可以指定另一套兼容解释器。原有 IsaacLab 与 Unitree 仓库含用户本地修改，保持原状。

Linux 活跃工程存于 ext4 的 workspace；`/mnt/data` 的 NTFS 分区只作跨 Windows/Linux 资料交换。

## 启动

```bash
cd /home/yang/workspace/projects/quadruped-arena

# 空场地，GPU 物理与地面接触验证
bash run.sh --headless --empty

# 高低掩体、出生点、任务点；地面/高墙/低墙落球碰撞验证
bash run.sh --headless

# 640×480 裁判视角渲染
bash run.sh --headless --capture

# 新增：起伏地面 + 固定碎石（含全景与碎石近景）
bash run.sh --headless --terrain rough --capture

# Isaac 原生高画质单张：1280×960、路径追踪、PBR 材质，隐藏红球/测试块
bash run.sh --headless --terrain rough --robot go2 --beauty

# 分别查看起伏和碎石，或回到原平地
bash run.sh --headless --terrain undulating --capture
bash run.sh --headless --terrain gravel --capture
bash run.sh --headless --terrain flat --capture

# 可选：用本机 Go2 资产验证加载与固定关节目标站立
bash run.sh --headless --robot go2 --steps 600 --capture

# 桌面交互查看场地，关闭窗口结束
bash run.sh --interactive
```

在 Codex 中运行需要宿主 GPU 设备访问权限；沙箱内 `nvidia-smi` 可能失败。
默认有限运行 240 步、2 秒仿真时间。详细输出写入每次运行的 `runtime.log`，
检查结果写入 `result.json`，渲染写入 `overview.png`；只有退出码为 0 且结果通过才算完整运行成功。
`process.json` 汇总进程结果；`environment.json` 和源码快照用于复现，有限运行超时为300秒。
导出的场地不含验证用红球及可选机器人，可供后续工程复用。

## 当前范围

`configs/arena_v0.json` 为可修改样场：12×8 米、4 个高掩体与 4 个低掩体，
Z 向上、米为单位、场地中心为原点；出生位置与掩体为中心旋转对称布局。
这是一张新建工程样场，不是对交接 HTML 图纸的精确复刻，也不是已冻结赛场。
圆盘仅表示任务点和出生点，不参与碰撞；场地边缘尚未实现越界判罚或保护层。

Go2 是现有资产的可选工程样例，不代表已选定参赛机型。其固定关节 PD 验证不包含行走策略。
站立检查要求整个运行中机身高度≥0.18米，最终机身向上方向的Z分量≥0.8、
水平漂移≤0.3米且关节数据有限；这些只是本样例的工程检查值。
旧 50 迭代模型仅为训练链路验证产物，不能当作成熟步态。
人形、行走/转向策略、机载相机、三维可见信息过滤、虚拟标记、计分与复位仍待后续实现。

本机 16GB RAM、RTX 4050 Laptop 6GB VRAM 低于官方最低配置，当前仅以实测小场景为能力边界。
官方文档已把 5.1 标记为不再支持；沿用它是为复用本机已有兼容组合，升级应另建环境验证。

## 本轮实测（2026-10-19）

- 空场景GPU物理与正常退出通过：`experiments/runs/quadruped-arena/20261019-170827-771805`。
- 场地、Go2站立、离屏渲染与正常退出通过：`experiments/runs/quadruped-arena/20261019-171203-418125`。
- 600步、5秒仿真；三个落球接触检查通过；Go2最低机身高度0.265m、水平漂移0.046m。
- 结果目录含`result.json`、`process.json`、`overview.png`、依赖版本与源码快照。
- Go2头部两个无visual链接有导入告警，详见环境报告；尚未验收GUI交互及传感器模型。

来源：
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
- https://github.com/isaac-sim/IsaacLab/tree/v2.3.2
- 本机 `notes/setup/UNITREE_ISAACLAB_2026-08-04.md`

## 起伏与碎石配置

`configs/terrain_v1.json` 控制路段位置、尺寸、高度、颜色、石块数量及摩擦。
`--terrain` 默认 `flat`，保留原样场；`--empty` 会忽略全部地形选项。
`rough` 同时启用两类地形，`undulating` / `gravel` 分别启用一种。

- 起伏：左右各一块2×3米路段，网格间距0.1米，高度0–0.12米；边缘与地面平接。
- 碎石：两块1.2×1.8米路段，各36块不重叠的固定石块；石高4–11厘米，底床厚1.5厘米。
- 随机种子20261019固定布局，两个碎石区互为180度旋转副本。
- 保留出生点、任务点及掩体基础附近平地。最高点含床层为12.5厘米。
- 石块为静态凸形网格；不会滚动、沉陷或被推走。路面为硬质支撑，不是松沙模型。

外观和物理分别配置，但本工程把它们对应起来：

| 层 | 作用 | 本轮实现 |
| --- | --- | --- |
| 可见网格与视觉材质 | 决定相机看见的形状、颜色、阴影 | 土色起伏网格、灰褐石块；没有使用照片贴图 |
| 碰撞形状 | 决定物体落在哪里、脚能否穿过表面 | 起伏用三角网格；凸形石块用相同外形的凸包 |
| 物理材质 | 决定接触时的摩擦、回弹 | 静摩擦0.85、动摩擦0.7、恢复系数0；未按真实土石标定 |

只把一张沙石照片贴在平板上不会增加碰撞起伏；法线/凹凸贴图通常也只改变着色。
真实建模的起伏是否影响物理，还取决于有没有给它配置对应碰撞体。
本轮同一份高度和石块形状数据生成可见与碰撞几何，用PhysX射线与动态落物核对。
落球测试山丘中央的小平台，平底落块测试石块平顶，以减少球滚落造成的歧义。
这些测试不代表机器人已经能够通过整段粗糙路面；Go2仍只在平整出生区做站立验证。

`result.json`包含地形种子、配置哈希、石块数、碰撞检查；`terrain_config.json`保存配置快照。
`probe_trajectories.json`记录落物轨迹，`terrain_detail.png`为近景。
场景输出名称为`arena_flat.usda`、`arena_rough.usda`等，每次运行独立保存。

## 原生高画质出图

`--beauty` 隐含 `--capture`，只生成一张 `overview.png`，不生成近景。
物理验证完成后切换到 Isaac Sim 的 `PathTracing`：1280×960、目标256采样/像素、
每次更新8采样、6次光线反弹和OptiX降噪；渲染期间不推进物理步。
`scripts/beauty.py` 调整日光/环境光，生成固定种子的程序化颜色与法线贴图，
各表面使用不同颜色、纹理尺度和粗糙度。这是近似材质，非实景扫描土石材质。
不调用AI生图，不改变地形几何、碰撞和摩擦。`/World/Probes` 出图时不可见，验证仍执行。
额外导出 `arena_beauty.usda`，保留视觉材质、灯光和相机，贴图放在同目录 `textures/`；
当前引用绝对路径，跨机器搬迁需收集并重新定位依赖。`result.json` 的 `beauty` 字段记录渲染配置与贴图来源。

## 无训练比赛流程 Demo

```bash
cd /home/yang/workspace/projects/quadruped-arena
bash run.sh --headless --demo
# 先检查 0、35、61、74、120 秒的五个关键画面
bash run.sh --headless --demo --preview
# 不启动 Isaac 的逻辑测试
/home/yang/miniforge3/envs/isaaclab-go2/bin/python -m unittest discover -s scripts -p test_demo_timeline.py -v
```

`configs/demo_v1.json` 给定双方路线、时序、帧率与标记请求窗口，`--demo-config PATH`可换配置。

新增 Go2-W 掩体机动版：`bash run.sh --headless --demo --tactical`，可加 `--preview` 检查十个关键帧。它使用可见观测驱动的规则状态机与预设通道，展示藏身、探出、撤回和应对绕侧；仍是运动学动画，不是训练或接触动力学验证。配置为 `configs/demo_tactical_v2.json`，详见 [DEMO_V2_GUIDE.md](DEMO_V2_GUIDE.md)。
复用20261019-175157-223904的现有粗糙场地；可用`--scene PATH`指定兼容的同布局场地。
旧场地USD不保存覆盖，Demo的场景快照、程序化贴图另放新时间戳场景目录。

演示把120秒比赛以4倍速呈现，加3秒片头与5秒结尾，共38.1秒，10fps。
画面是Isaac Sim原生PathTracing离屏渲染（960×720，目标32采样/像素），
上下中文HUD由Pillow后期叠加，MP4为960×944，GIF缩为800×786。
由当前环境已有的imageio-ffmpeg导出H.264 MP4及循环GIF；无需安装依赖或训练。
输出位于`experiments/runs/quadruped-arena/<时间>/`：

- `demo.mp4`：演示投屏首选。
- `demo.gif`：循环预览。
- `poster.png`、`intro.png`、`frame_*.png`：结算封面、片头与关键帧。
- `events.json`、`timeline.json`、`demo_config.json`：编排事件、50Hz状态与配置快照。
- `result.json`、`process.json`：渲染/编码检查、帧数、时长和退出状态。
- `source_snapshot/`：包括项目脚本和JSON配置，可复现。

流程：出生点就位→绕行与起伏路段→P1占点/争夺→虚拟标记和3秒暂停→60秒激活P2→转移争夺→120秒结算。
四足使用既有Go2转换资产，人形使用本机Unitree G1 23自由度URDF/STL外观；机型只是演示样例。
蓝色为四足A，橙色为人形B，绿色圈为激活点，黄色连线为锁定/标记示意，细线为已经走过的路线。

### 展示边界（必须随演示说明）

这是脚本编排，不是已训练的自主比赛系统，也不是物理步态测试。
双方根位姿沿预设路径插值，肢体使用URDF前向运动学和周期关节动画；人形没有真实遥操作。
没有运行接触动力学、步态控制器、感知策略或自主导航。
地形跟随只近似调整模型高度，不证明机器人能通过沙石地面。
路径中心到掩体、两台模型中心间距有几何检查，但不能代替全身动态外廓与安全验收。

占点积分按0.5秒驻留、独占条件、任务点切换和命中后3秒暂停计算。
标记请求在编排窗口开启，再检查8米范围、竖直角度、固定代表点连线与三维掩体盒体遮挡、0.8秒锁定。
为展示流程而设置请求窗口，不等于草案中“任何时刻满足条件就自动标记”的正式裁判。
代表点高度和遮挡检查尚未接入实际模型姿态、地形/全身遮挡、边界不确定性或丢测处理。
显示的输赢与得分仅属于本次预设编排，不用于能力比较，草案规则仍未冻结。
Demo运行超时单独为1200秒；原静态/物理验证仍保持300秒。

2026-10-19实测：组合模式600步，7项动态接触与4条射线检查、Go2平地站立及两张渲染图通过。
结果：`experiments/runs/quadruped-arena/20261019-172355-574122`。
flat、undulating、gravel各240步的模式回归也通过，详细记录见环境报告。
