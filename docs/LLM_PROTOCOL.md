# LLM、SKILL 与记录协议

## 1. 非全知闭环

```text
Python 裁判 / 运动学（dt=0.02 s）
  → 每 0.5 s 把位姿交给 Isaac，生成机载 RGB
  → 每 2 s 形成公开观测，暂停模拟时钟
  → LLM 返回一个受限 JSON 意图
  → 校验器 + SemanticSkills 编译路线、朝向、观察命令
  → 执行下一窗口并持续检查局部约束
```

渲染、网络请求、模型思考所用现实时间不计入比赛 40 秒。此设置测的是决策质量，不是实时端侧闭环性能。Python 是比赛状态权威；Isaac 在线生成当前视图。相机首帧为 t=0，后续通常包含上一窗口末帧和 t+0.5 / 1.0 / 1.5 / 2.0 四张新帧；提前终局时少于五张。

## 2. 每轮实际输入

源码定位：`deepseek_trial.system_message`、`public_observation`、`observation_message`；Qwen 视觉版本复用该公开协议。

- 固定 system：任务背景、规则、坐标和相机说明、五种技能、已知地图与坍塌、输出约束。
- 当前 user：自身位置/朝向/HP/速度/冷却、得分、活动目标及 generation、剩余目标时间、技能状态、合法视觉记忆、当前 1–5 张 RGB 和对应时间/标定。
- 最近至多 4 条紧凑的历史动作：不追加旧 assessment、完整图片历史或内部推理全文。
- 不提供当前 B 的真实位置、yaw、HP、路线、策略内部变量或裁判随机阈值。B 的初始位置已知，不意味着它后续一直停在那。

因而每次请求不是无限增长的聊天记录。后期耗时增加还可能来自当前情境复杂度、输出推理量、图像 Token、缓存/服务排队等，不能仅用“比赛步数变大”推断历史上下文累积。

## 3. 五个意图技能

| skill | 主要参数 | 语义 |
| --- | --- | --- |
| `MOVE_TO` | `target=active/P1/P2` | 移动到指定已知目标；不自动做后续策略选择 |
| `CAPTURE_POINT` | `target=active/P1/P2` | 移动并驻留当前有效目标，失效后停止 |
| `TAKE_COVER` | 可选 `cover=掩体名` | 根据地图和合法威胁记忆选掩体位置并规划 |
| `AIM_AT_TRACK` | 可选新 sighting | 停下对准仍新鲜的视觉估计，不是穿墙真值跟踪 |
| `OBSERVE` | `direction=scan/threat/north/south/east/west` | 停下观察，云台扫描；不自动攻击 |

通用 `risk=safe/fast` 改变路线风险偏好，**不是修改机器人最大速度**。`safe` 不保证安全；可能拒绝路线并返回 `UNSAFE_ROUTE`，需要下一轮决策处理。

通用 `sighting={"time":帧时间,"feet":[u,v]}` 只有实际看见人形接地脚点时才可提交。图片分辨率 768×432，左上为 [0,0]。脚被遮挡时应省略而非猜测。底层结合相机标定、地面 z=0 假设反投影；记忆在 4 秒后过期，≤2 秒的新鲜 RGB 估计才允许主动攻击。已知初始位置本身不使能射击。

底层技能是有约束的研究原型，非成熟视觉检测 / 跟踪系统。相机云台运动不同于机身射击朝向；转相机不等于瞄准。

## 4. 决策输出样例

带摘要的统一实验输出恰好为以下四个顶层字段。`assessment` 是可核验的观察和决策依据摘要，不是完整内部思维链。下面仅示意格式，不能当作任意状态下的正确策略。

```json
{
  "observation_id": 0,
  "skill": "OBSERVE",
  "params": {"direction": "north", "risk": "safe"},
  "assessment": {
    "visual_evidence": [
      {"time": 0, "opponent_visible": null, "observation": "近处墙体遮挡，无法确认墙后是否有人形。"}
    ],
    "state_evidence": "自身 HP=3，尚未得分。",
    "decision_summary": "先观察侧向通道，再依据执行结果决定移动。",
    "uncertainty": "未看到对手并不表示通道安全。"
  }
}
```

`visual_evidence` 必须引用本轮实际输入的时间；看见对手但脚不可见时，可以在摘要说明，不应伪造 sighting。不得加入任意代码、数值目标世界坐标、逐点路径或未知字段。

## 5. 接入自己的规划器

不必接入特定商业 API。启动外部决策模式：

```bash
bash run.sh --vision-trial --semantic-skills --assessment --headless
```

终端打印 run 目录后，由另一个进程执行：

1. 读取 `pending.json` 的 `observation_id`，以及 `observations/NNN.json` 和其中引用的机载 PNG。
2. 只把公开观测和 `planner_context.txt` 送给规划器；不读取中间私有状态或事后真值。
3. 输出上述四字段 JSON 到 `actions/NNN.json`，编号用三位零填充，例如 `000.json`。
4. 先写同目录临时文件，再原子重命名；主进程看到文件即开始读取，避免读取半个 JSON。
5. 等待新的 `pending.json`；终局 `finished=true` 后停止提交。

单步外部动作等待上限为 1200 秒；格式错误会停止，不隐式修复策略。不要用旧 observation_id 重复提交，不要提前填满未来动作。无 `--assessment` 的语义模式允许简短三字段动作及可选短 reason，但不是本轮模型比较的统一四字段形式。

不要加载来自陌生人的 `.pkl` / `.pickle`；早期暂停式 CPU 工具的 pickle 是可信本地检查点，不是公开数据交换格式。

## 6. API 配置与费用

密钥只能通过本地终端交互录入：

```bash
bash deepseek.sh setup
bash qwen.sh setup
```

保存位置默认 `~/.config/quadruped-arena/{deepseek,dashscope}.key`，专用目录 0700、文件 0600。不要提交、贴到 issue、放在命令行或分享截图里。替换已有密钥需显式 `setup --replace`。不用真实密钥也可以运行所有 CPU 单元测试；API 测试使用模拟响应，不会联网扣费。

DeepSeek 默认 `configs/deepseek_trial.json`，强度 low，输出摘要。最高强度：

```bash
export ARENA_DEEPSEEK_CONFIG="$PWD/configs/deepseek_max_platform_limit.json"
bash deepseek.sh prepare    # 离线
bash deepseek.sh run        # 付费，可能长时间运行
```

该 profile 没有额外实验总 Token 停止预算，但仍有供应商的单次输出上限（记录值 393216）和 600 秒网络空闲超时；不是数学意义的无限。API 标识和上限是实验时快照，应在开跑前对照官方文档确认。不应把意外失败自动切成其它模型或降低强度。

Qwen 视觉的 `max` 表示开启 thinking、未额外指定 thinking_budget；`min` 表示关闭 thinking。它们是实验标签，**不是 Qwen API 的同名 reasoning_effort 档位**，也不能与 DeepSeek max/low 视为等量推理计算。`configs/qwen_vision_trial.json` 保存实际参数。`qwen.sh run` 使用不同的早期全知接口，应单独分析。

供应商模型调用最多 20 次，正好覆盖 40 秒 / 2 秒；这是请求次数上限，不是总 Token 上限。视觉 API 模式不自动重试、无隐藏备用策略。网络/JSON/供应商错误会保留中断记录，停止而不是伪造完成结果。

## 7. 输出文件与计时口径

| 文件 / 目录 | 含义 |
| --- | --- |
| `protocol.json` / `api_profile.json` | 信息权限、周期、模型和思考档位 |
| `observations/`、`frames/`、`frame_index.json` | 每步公开输入与真实机载图 |
| `actions/`、`decisions/`、`decisions.json` | 原始决策、摘要、接受时间及编译技能 |
| `api_calls/` | 调用输入摘要、输出/usage/时长等；以实际运行产物为准 |
| `skill_execution.json` | 底层技能状态和地图重规划记录 |
| `trial_config.json`、`timeline.json`、`events.json` | 完整配置与事后裁判 / 轨迹；不能回灌给当前非全知规划器 |
| `validation.json`、`result.json` | 最终得分、终局及验证摘要 |
| `interruption.json` | 未完成比赛，不能当成完整 40 秒样本 |
| `report.html`（视觉试验）等报告 | 截图、决策摘要、时间线和回放展示 |

以文件实际存在为准：不同历史模式报告文件名有所区别。`vision_report.py --help` 等查看对应报告 CLI。

- `decision_wait_seconds`：公开观测就绪到动作接受，包含模型、工具/传输和人为等待，不是纯推理时间。
- `api_wall_seconds`：外部计时的网络调用开始到结束，包含服务排队、图像处理、生成和传输。
- 首输出 / 首内容延迟与技能编译时间独立记录，不能混为同一指标。
- Token 使用供应商返回的 usage；reasoning_tokens 通常是 completion 的子集，不重复相加。缺失数据保留 null，不估算冒充实测。
- 外部交互式助手不一定有逐步 Token 统计，而且可能继承项目对话背景；比较报告必须披露这一不对称条件。
