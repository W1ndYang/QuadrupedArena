# 固定目标攻防 v0.4.1：有遮挡出生、探身与技能序列

这是 **2026-11-04 已完成的三场 DS LOW 测试**和可接续的游戏实现，2026-11-05 整理为可移植发布包。采用 A3.9.1 的 24 × 16 m 地图，关闭坍塌；固定一个目标，到点即赢。它与仓库最初发布的“循环刷新点、累计得分”赛制相互独立。

**运行方式：CPU 运动学裁判与技能执行，Isaac Sim 仅负责三维回放；模型输入是合法几何观测，不是 RGB。** 不包括步态训练或真实机器人控制。

- [完整比赛规则、感知与人形设置](RULES.md)
- [模型协议、技能及已知问题](MODEL_PROTOCOL.md)
- [离线 HTML 报告](index.html)：下载仓库后用浏览器打开；GitHub 文件页不会执行 HTML。
- [SE GIF](run_1/demo.gif) · [EW GIF](run_2/demo.gif) · [WC GIF](run_3/demo.gif)

## 不用 API 复现这三局

在仓库根目录，使用 Python 3.11：

```bash
python -m pip install -r requirements-cpu.txt
python benchmarks/fixed_goal_v041/benchmark.py audit
python benchmarks/fixed_goal_v041/benchmark.py replay \
  --output .arena-workspace/fixed-goal-replay
```

`replay` 真正重新执行 50 Hz 裁判和技能，不只是播放 JSON。它逐步比较模型收到的公共观测，再复用对应已保存动作，最后要求**所有轨迹状态及裁判事件完全相同**。输入一旦不同立即停止，防止把旧动作硬套进新场景后宣称模型复现成功。不加载密钥、不调用网络、不需要 GPU。输出目录必须为空，防止覆盖旧记录。

| 布局 | 四足出生 | 人形出生 | 目标 | 已存结果 | 模型调用 |
| --- | --- | --- | --- | --- | --- |
| SE | (-5.8, -1.0) | (6.2, -2.7) | (10.1, -0.8) | 11.60 s 到点 | 3 |
| EW | (5.8, -0.1) | (-6.3, -2.8) | (-10.1, 1.1) | 6.86 s 被标记 | 2 |
| WC | (-9.8, 1.5) | (5.8, -3.2) | (1.8, -4.7) | 14.66 s 被标记 | 4 |

三种出生位置都被实体墙遮挡、双方初始距离大于 10 m、到目标的距离大于 10 m。每组检查了两条出路，并用执行器各试走前 2 s；详见 [layout_checks.json](layout_checks.json)。这只证明存在起步空间，不保证模型任意选路都安全。每布局仅一局，不能据此估计稳定胜率。

## 准备和运行自己的模型实验

```bash
# 复制运行代码和配置到新目录，完成不收费的出生检查及源码冻结。
python benchmarks/fixed_goal_v041/benchmark.py prepare \
  --output .arena-workspace/my-fixed-goal

# 在自己的交互终端输入官方 DeepSeek 密钥；存到仓库外，权限 0600。
bash deepseek.sh setup

# 以下命令会调用付费 API；--run 1 可只运行一局。
python .arena-workspace/my-fixed-goal/run_batch.py --execute-paid
```

默认沿用本次实验快照 `deepseek-flash`、`thinking=enabled`、`reasoning_effort=low`、`max_tokens=65536`。这是发布时的实验配置，不保证供应商以后仍提供相同别名或响应；遇到不支持的模型时应新建配置版本并记录，不应悄悄替换。调用使用官方 `https://api.deepseek.com`，密钥从 `${XDG_CONFIG_HOME:-~/.config}/quadruped-arena/deepseek.key` 读取，不写入请求记录。

若修改布局，在**首次模型调用之前**编辑新目录的 `layouts.json`（含 `start_a`、`human_start`、`goal`、`patrol` 和两条 `escape_points`），再运行该目录的 `prepare.py` 重新检查和冻结。不要修改随仓库附带的 `run_1`～`run_3`，也不要在已产生 API 请求的目录重新预注册。

没有自动重试、策略兜底或对已接受动作重新采样。网络中断时保留原请求/错误记录，复用已完成返回；重启相同命令只允许请求哈希完全相同。失败请求没有 usage 回执时，消耗未知，不能记为零。修改了环境或策略必须另开实验。

## 用 Isaac 重新渲染

先按仓库 [安装说明](../../docs/INSTALL.md) 安装参考版本 Isaac Sim 5.1.0 / Isaac Lab v2.3.2、Unitree ROS 资产与 ambientCG 贴图。生成资产通过 `ARENA_WORKSPACE` 定位，Unitree 和 Isaac Lab 可分别用 `UNITREE_ROS_DIR`、`ISAACLAB_PATH` 指定。

```bash
export ARENA_WORKSPACE=/absolute/path/to/arena-workspace
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
export UNITREE_ROS_DIR=/absolute/path/to/unitree_ros
# 仅在接受 NVIDIA 条款后设置：
export OMNI_KIT_ACCEPT_EULA=YES

python scripts/prepare_assets.py
# 渲染前面离线 replay 生成的目录，避免覆盖随仓库发布的原生图像。
python benchmarks/fixed_goal_v041/render_native.py --headless \
  --data-dir .arena-workspace/fixed-goal-replay
```

渲染器读取完成的 `timeline.json`、`events.json`、`result.json`、`decisions.json` 和配置；不运行策略、不调用模型。`--only-run 1` 可只渲染一局。图像是 12 fps 原生 PathTracing 回放，烟的可视化是圆柱范围内的半透明几何，不是流体模拟。CPU 路径不需要安装 Isaac。

## 文件与版本边界

| 内容 | 入口 |
| --- | --- |
| 地图尺寸、墙高、速度和基础参数 | `source_snapshot/configs/trial_a391_feedback_fix.json` |
| 本轮实际生效的合并配置 | `run_*/trial_config.json` |
| 新布局、地图路点、规则覆盖 | `layouts.json`、`waypoints.json`、`design_basis.json` |
| 裁判、视线与命中 | `source_snapshot/scripts/fixed_goal_probe.py`、`run_goal_only.py` |
| 人形记忆、PLAN、短探和路线执行 | `sequence_skills.py` |
| 先刹车再反向的 v0.4.1 修复 | `peek_braking_fix.py` |
| A*、连续运动、人形移动 | `source_snapshot/scripts/{trial_motion,continuous_motion,humanoid_motion}.py` |
| 官方 API、短说明校验与保存 | `run_batch.py`、`qwen_fixed_goal.py`、`source_snapshot/scripts/deepseek_client.py` |
| 完整系统提示词 | `prompts/{SE,EW,WC}.txt` |
| 每步观测、响应、接受动作与外部用时 | `run_*/api_calls/`、`run_*/accepted/` |
| 轨迹、事件、截图、GIF | `run_*/` |
| 发布可移植性变更和原实验审计 | `provenance/` |

`qwen_fixed_goal.py` 的历史文件名只表示存储与校验适配器来源；本次实际客户端和模型均为 DeepSeek。原始实验预注册保存在 `preregistration.json`；发布路径调整后的代码哈希使用 `source_manifest.json`，完整文件哈希使用 `publication_manifest.json`，不可把两套哈希混为一谈。

为保证规则可追溯，基础 JSON 保留了一些旧赛制字段。**本轮的实际裁判是 `GoalOnlyProbe` 与固定目标规则覆盖**，不是旧的 HP、轮换点、坍塌字段。旧 `run_study.py` / `v04_model.py` 的独立命令不是本包支持入口，只保留其运行依赖；使用上面的 `benchmark.py` / `run_batch.py`。

本包约 30 MB，明确收录三场必要复现样例，包括失败。没有纳入 261 MB 可重新生成的 USD 缓存、NVIDIA 软件、Unitree 原模型、PBR 原图、私密凭据或其他历史实验。代码遵循仓库 MIT；渲染中使用的第三方素材见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
