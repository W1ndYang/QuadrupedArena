# 高频 Skill 强化学习分支：代码导览

本分支 `exp/rl-skill-10hz` 从 `51a3063`（合成视觉模块、Skill 接入和 0.5 秒观测完成）建立。此前的有限状态 Oracle 研究保存在 `archive/oracle-finite-state`；它的采集脚本、44 项对比动作目录、R2-L 抽象和优化器没有并入本分支。`main` 未改动；两个新分支已发布到远端，原远端实验分支名已退役。

**本分支尚未实现强化学习环境或 10 Hz Skill 决策。** 当前比赛仍是 Python 运动学和裁判推进，Isaac Sim 负责在线相机 RGB 与场景渲染；`SemanticSkills` 把高层意图编译为底层动作。

## 直接复用的主链

| 职责 | 代码 | 用于新方向时的边界 |
|---|---|---|
| 比赛规则、对手和运动学 | `trial_rules.py`、`llm_trial.py`、`trial_motion.py`、`adaptive_motion.py`、`tactical_logic.py`、`configs/trial_rotating.json` | 可作为 CPU 训练环境的规则来源；不要把真实对手内部状态无意放进策略观测 |
| 公开观测和相机 | `vision_core.py`、`partial_observation.py`、`vision_trial.py`、`scene.py`、`arena_art.py` | `public_packet` 已区分己方／公开字段与隐藏对手真值；在线 RGB 仍走 Isaac 场景 |
| Skill 接口和执行 | `semantic_skills.py` | 保留 `validate → commit → tick` 契约、风险路线和视觉记忆；策略输出应走正式 Skill 请求，而非直接改裁判状态 |
| 启动、素材与复现 | `run.sh`、`launch.py`、`prepare_assets.py`、`arena_paths.py`、`replay_io.py`、`doctor.py` | 现有工作区与环境快照规则可复用，训练产物继续放 `.arena-workspace/` |
| 回归检查 | `test_trial_rules.py`、`test_vision_core.py`、`test_semantic_skills.py`、`test_synthetic_vision_mvp.py` 等 CPU 测试 | 改变决策频率前先保持规则、Skill 与公开观测行为可复核 |

## 视觉模块如何使用

`synthetic_vision_mvp.py` 和 `synthetic_sighting.py` 值得保留。它们与 `formal_geometry_diagnostic.py`、`geometry_interface_probe.py`、`isaacsim_minimal_geometry_test.py` 形成几何投影、遮挡查询、候选足点和 Skill 视觉记忆的验证链。可用于诊断、监督标签或受控训练输入。

这套 synthetic sighting 来自仿真几何真值，**不是从 RGB 自动识别目标的检测器**。既有运行记录将 `same_frame_verified` 标为 false、同步集成标为 blocked。若未来训练策略直接读取真值状态，应在数据契约中明确它属于可用观测还是仅供训练期的特权信息；不能把 synthetic label 默认为部署时可由相机获得的输入。

## 保留为参考、暂不接入训练环

- `deepseek_trial.py`、`qwen_vision.py`、`llm_trial.py` 中的 LLM 交互层以及报告生成脚本仍服务现有演示和复现，不必为了 RL 删除；新训练环不依赖在线 LLM 或历史 HTML 报告。
- `formal_geometry_diagnostic.py`、`geometry_interface_probe.py` 和最小几何实验是视觉调试工具，只有在修改相机、分割、深度或遮挡实现时才运行。
- 旧 Oracle 的 `oracle_state_v1.py`、`oracle_pilot_runtime.py`、`finite_state_optimizer.py` 等在归档分支中完整保留。它们的短窗口采集、可执行性过滤与成对评价方法可作研究参考，但 R2-L 离散状态和 44 项动作目录不是新策略输入的默认设计。

## 建立 10 Hz 环境前必须重新定义的接口

当前 `vision_trial.py` 每 **100 个 `dt` 微步**做一次约 2 秒的高层决策，每 25 微步采一帧约 0.5 秒 RGB；`SemanticSkills.tick` 已在每个微步运行。10 Hz 高层 Skill 选择意味着约每 5 个 `dt` 微步可做一次策略决策。只改循环次数会让 `commit` 反复重规划、覆盖尚未完成的 Skill，也不会自动得到 10 Hz 图像。新环境适配层应明确以下契约后再实现：

1. **动作。** 五类正式 Skill 与参数如何编码；增加“继续当前 Skill”还是每次重发请求；不合法或受阻动作如何掩码、记录与计入奖励。
2. **观测。** 当前自身运动状态、HP、公开目标／得分、已知地图、Skill 状态、相机帧与视觉记忆的时间戳；训练期特权状态与策略实际可见信息分开记录。
3. **时钟。** 物理／裁判微步、10 Hz 策略时钟和实际 RGB 更新时钟分别配置；旧 0.5 秒 RGB 不应被误报为 10 Hz 视觉。
4. **奖励与终止。** 以正式规则的得分、生命和 40 秒终局为基准，若加密集奖励则单独标识；验证仍用独立比赛 seed 的官方终局分数。
5. **运行结构。** 训练环境封装独立于 `vision_trial.py` 的 LLM／演示循环，先复用 CPU 裁判与 Skill，再接 Isaac RGB，避免在已有演示入口中叠加另一套主循环。

这些是后续实现接口的待决项，本次整理没有替用户预设状态可见性、算法或奖励形状。

## 分支与资料

- `archive/oracle-finite-state`：停研的 Oracle 研究历史与代码；不要在这里直接做 10 Hz RL 开发。
- `exp/rl-skill-10hz`：本分支，以视觉／Skill 基础为起点。
- `main`：现有稳定入口。本次没有合并或改写它。
- `docs/codex_context/`、`docs/codex_report/` 及 `.arena-workspace/` 保持原忽略规则；研究报告继续写到 `docs/codex_report/`，运行包不要提交源码 Git。

远端 `origin/archive/oracle-finite-state` 保留 Oracle 研究 HEAD，`origin/exp/rl-skill-10hz` 跟踪当前开发分支。确认两个新远端引用后，已按用户授权删除旧名称 `origin/exp/isaacsim-minimal-geometry`；删除时使用了精确提交租约，避免覆盖并发更新。
