# Qwen3.6-27B · 百炼北京 API 测试接入

当前状态（2026-10-21）：真实连通检查通过。首次正式请求在第 0 步重复查询工具后停止，失败记录保留。用户确认补全背景及工具预算协议后，第二轮已完成真实闭环：16 次决策、22 次 API 请求，31.64 秒四足淘汰、10.4 分、终局 HP 0/1。没有自动重试或替模型决策；129 项离线测试通过。

## 第二轮协议：补全背景与强制结束工具阶段

用户确认后于 2026-10-21 启动独立第二轮，协议版本 `qwen-r2-context-budget`。第一轮失败目录不修改。新目录为 `20261021-162029_Qwen3.6-27B_2秒闭环`，结果以目录内 `api_run_status.json` 为准。

- system 上下文新增项目目标、A/B 角色、得分与存活的关系、全知信息的不对称性、暂停执行流程、坐标与状态字段语义、几何工具的能力边界；不提供上一轮动作或成绩。
- 明确查询工具不是运动执行，工具返回已是查询答案，重复静态查询不提供新信息。
- 每次请求附上当前 observation_id 与剩余工具轮数。前两轮 `tool_choice=auto`，第三轮 `tool_choice=none`，必须输出六字段最终动作 JSON。
- 比赛规则、种子、技能、模型参数和总请求预算不变；仍不自动重试、不自动修复动作。
- 每轮保存 `planner_context.txt`，协议记录提示词 SHA256，逐请求记录实际工具选择模式。补充传输与预算回归测试后共 129 项离线测试通过。
- 官方参数依据：[百炼 Function Calling](https://help.aliyun.com/en/model-studio/qwen-function-calling)，思考模式支持 `auto` 和 `none`。

### 第二轮实际结果

- 决策端到端等待合计 750.985 秒，平均 46.937 秒，中位数 28.032 秒，最长 165.621 秒；不是纯推理时间或端侧速度。
- API usage：输入 526562、输出 38469、合计 565031 tokens；reasoning 34399 包含于输出。长历史重复输入计入实际 token 总量。
- 第 11、16 步实际使用 `tool_choice=none` 后提交合法动作。原规则、种子与技能不变。
- 赛后按保存源码、配置、动作进行确定性核对，1583 个时间线状态完全一致。
- 28 秒时低估持续瞄准的未来两秒风险；30 秒时把命中方冷却误解为受击方冷却。具体原输出和评估者事后分析见本轮 `requests/014.json`、`requests/015.json`、`post_run_review.json`。
- 完整比赛背景已提供，但仍存在规则应用错误；不能仅凭本轮对模型普遍能力下结论，也不能视为端侧实时性验证。

## 输入 Key

在本机终端执行：

```bash
cd /home/yang/workspace/projects/quadruped-arena
bash qwen.sh setup
```

粘贴 Key 后回车，输入不会显示。Key 保存在 `/home/yang/.config/quadruped-arena/dashscope.key`，文件权限 0600、专用父目录 0700；不放在项目、实验、源码快照或 HTML 中。不要把 Key 发到聊天，也不要放在命令行参数或截图里。重复配置必须显式使用 `setup --replace`。

```bash
bash qwen.sh status  # 不发请求，只显示是否已保存 Key，不显示 Key
```

默认 Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`，模型 ID：`qwen3.6-27b`。也可在非敏感配置 `configs/qwen_trial.json` 中使用控制台给出的北京工作空间专属地址：`https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。只接受官方北京 HTTPS 主机，拒绝重定向和 URL 中的密码/query。

## 分开进行连通检查与正式比赛

保存 Key 本身不调用 API。下面两步会调用百炼，可能计费，需明确执行：

```bash
bash qwen.sh probe  # 一个短请求：检查身份、模型权限、流式回复与 usage
bash qwen.sh run    # 正式新一轮；最多 20 次决策，每步最多 3 个 API 请求
```

连通检查不推进比赛，不与正式种子混用。正式比赛使用独立的新目录，旧助手试验保持不变；目录已存在时拒绝覆盖。不自动重试，也不自动换模型、改策略、放宽规则或修复非法动作。遇到 HTTP/限流/截断/非法 JSON/不可达目标等问题，停止并保留当前记录，等待人工检查。请求中断也可能已计费，不能把缺少 usage 当作零消耗。

当前版本刻意不自动恢复失败的 API 轮次；检查保存的 `api_run_status.json`、最后请求/响应后再决定如何继续，不能盲目重跑挑选好结果。

## 首轮冻结参数

- 规则 v0.4，40 秒上限，2 秒仿真周期，思考和几何工具期间暂停比赛。
- 种子 2026102107，与上一轮相同；双方 HP、命中过程、占点、坍塌、人形策略、运动限制均不变。
- `enable_thinking=true`、`thinking_budget=4096`、`max_tokens=8192`、`temperature=0.6`；这些是首轮配置，不是已验证的最优值。
- 最多每步 3 个 API 请求（至多两轮几何工具 + 最终动作），全场最多 60 个请求；单请求 180 秒 socket 超时。无自动重试，费用不以本地秒数估算。
- 输入当前结构化真值与当前程序绘制导航图，附上规则及原 `HumanPolicy` 源码；不提供上轮实际动作、命中随机阈值、RNG 状态或未来轨迹。
- 历史上下文保留本轮观测、工具结果及最终输出；不保存或传回 reasoning_content 正文。
- 可调用同类 `probe_geometry` 检查当前几何路径、路径长度和当前视线；它不推进比赛、不试跑未来。
- 输出原六字段 JSON：observation_id / action / evidence / choice_summary / alternatives_summary / expected_result。原固定 MOVE/HOLD/AIM/SCAN/STOP 执行器，不包含助手代决策。

与上一轮尽量保持任务和权限一致，但不能称为严格只换模型：上一轮为交互式助手，提示词、历史组织和图片实际读取方式并不是相同的 API scaffold。API 延迟也不能直接与上一轮包含工具/交互/上下文处理的等待时间当作纯推理速度比较。

## 日志与报告

原 observations / requests / decisions / timeline / events 结构保留；额外的 `api_calls/` 保存无鉴权头的请求消息、最终文本/工具调用、响应模型 ID、响应 ID、finish_reason、usage 及耗时。API Key 不在这些记录里；reasoning_content 正文也不保存。

记录分为：

1. 每步端到端决策等待时间（可能包含多次几何工具/API 请求）。
2. 每个 API 请求的网络墙钟时间、首次任意输出时间、首次可见正文时间；都不是纯模型推理耗时。
3. prompt_tokens、completion_tokens、total_tokens；若返回 reasoning_tokens 也单列，它属于 completion_tokens 的子集，不重复相加。缺失字段保留 null。

结束后使用保存轨迹原生重渲染，再生成同款四选项卡报告：

```bash
bash run.sh --demo --industrial --headless --replay-dir <本次Qwen运行目录>
/home/yang/miniforge3/envs/isaaclab-go2/bin/python scripts/llm_report.py <本次Qwen运行目录> --render-dir <本次原生渲染目录>
```

截图是本轮轨迹的事后展示，不是旧助手轨迹，也不是原始模型相机输入。报告按本轮实际终局、步数和 usage 生成，不复制上一轮的 12 分结论。提前终局会通过原生视频的 `frame_index.json` 精确映射决策截图。

## 已生成的无密钥准备包

`/home/yang/workspace/experiments/runs/quadruped-arena/20261021_Qwen3.6-27B_测试准备/`

包含冻结比赛配置、公开 API 参数、初始观测、示例消息、工具 schema；状态为 prepared_not_run。目录内不存在 Key，不代表 Qwen 已执行。

## 官方依据（2026-10-21 核对）

- [Qwen 官方模型卡：Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)：包含百炼模型 ID、接口示例及 enable_thinking 参数差异。
- [百炼 Chat Completions 接口](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)：北京地域端点、流式 include_usage、token 细分等；旧 dashscope 域名仍可用。

真实账号的模型开通权限、余额、限流和接口参数兼容性，以输入 Key 后的小请求检查为准；未连接成功前不声称已通过真实 API 验证。
