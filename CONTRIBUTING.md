# 接续开发与贡献

## 建议流程

1. 从 `main` 建立自己的分支；先运行 CPU 测试和基线，保留结果。
2. 修改规则时新增配置版本，明确基于哪个 base_config；不要在旧实验目录中改 `trial_config.json`。
3. 修改地图时同步检查渲染几何、导航/LOS 几何、出生点、目标、坍塌后状态及预留扫掠区。
4. 新增模型仅负责公开观测 → 受限 JSON；不能访问 Match / Referee / RNG / 对方真值来“修复”策略。
5. 新技能应首先添加 CPU schema / 安全 / 权限隔离测试，再做不调用 API 的 Isaac smoke，最后才做明确付费实验。
6. 提交代码、配置、必要的小型样例和说明，不提交 API Key、安装环境、机器人转换缓存、海量帧、完整实验目录或模型权重。

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
python scripts/cpu_demo.py
git diff --check
git status --short
```

## 核心源码导航

| 内容 | 文件 |
| --- | --- |
| 概率命中、计分、目标轮换和终局 | `scripts/trial_rules.py` |
| 几何寻路 / 运动学 / 障碍检查 | `trial_motion.py`、`tactical_logic.py` |
| 人形有限观察策略 / 风险基线 | `adaptive_motion.py` |
| 可暂停比赛状态 | `llm_trial.py` |
| 公开观测和相机控制 | `vision_core.py`、`vision_trial.py` |
| 意图技能和合法视觉记忆 | `semantic_skills.py` |
| DeepSeek 提示词和调用 | `deepseek_trial.py`、`deepseek_client.py` |
| Qwen 视觉适配 | `qwen_vision.py`、`qwen_client.py` |
| 同轨迹重新渲染 | `replay_io.py`、`trial_demo.py` |
| 场景外观和大地图 | `arena_art.py`、`dockyard_*.py` |
| 多模型统计与报告 | `research_*.py`；需要完整原始实验数据 |

表中未带目录的文件也位于 `scripts/`。

## 可复核的实验记录

每次至少记录：源码 commit、完整合并配置、seed、观察权限、提示词版本、模型请求和实际返回标识、思考档位、单次/总预算、失败与重试策略、技能版本、逐步观测/动作、API 时长、Token 缺失情况、终局原因和得分。

改变一个因素的实验才能清楚归因。多模型比较中，不要混用全知 / 非全知、旧 / 新规则、不同技能栈、短截断 / 完整比赛。报告应展示所有失败运行，且区分平台预算耗尽、API 中断和机器人被淘汰。

## 发布检查

- `git diff --check` 无空白错误；所有 JSON 可解析。
- CPU CI 不依赖本机绝对路径、私有素材或真实 Key。
- 下载脚本拒绝未校验文件，正文包含第三方来源。
- 说明验证的是哪种模式；未做完整安装/动力学/大地图比赛验证时明确写出。
- 不从陌生来源加载 pickle，不执行 LLM 输出的任意代码。
- 若要分享完整视频或数据集，使用独立 Release / 数据集存储，并给出哈希；不要强制写入 Git 历史。

## 固定目标 v0.4.1 发布样例

`benchmarks/fixed_goal_v041/` 是本次明确纳入的三局可复核样例（约 30 MB），包括完整失败轨迹、九步决策及必要的 GIF/截图。它不等于默认允许提交后续所有实验目录。生成的新实验仍放在 `.arena-workspace/`；大型批量结果继续优先使用 Release 或数据集存储。

新固定目标赛制从 `benchmark.py` 的 `audit`、`replay`、`prepare` 入口开始，不把该目录的独立裁判与根目录历史累计占点裁判交叉导入。改动行为后应新增样例和协议版本，不覆盖已发布三局。

## Breach Arena（`breach_arena/`）

这是一个独立游戏：有自己的规则版本、地图格式和决策协议，不与根目录或 `benchmarks/` 下的代码交叉导入（`breach_arena/tests/test_isolation.py` 会检查）。改动后至少运行：

```bash
python -m unittest discover -s breach_arena/tests -t .
python -m breach_arena matrix --map BA-1
python -m breach_arena matrix --map BA-2
```

改动规则时，同时更新 `rules.json` 的版本号和 `RULES.md`；改动地图时，同步更新 `cycle_checks`，并重新生成 `maps/checks/` 快照（`python -m breach_arena checks --out breach_arena/maps/checks`）。
