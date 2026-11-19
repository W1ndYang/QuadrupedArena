# LLM 决策研究的 Ubuntu 运行说明

## 当前批次

通过 Git 同步本阶段提交，在 Ubuntu 仓库根目录、已经配置 `DEEPSEEK_API_KEY` 的终端运行。

```bash
python3 benchmarks/two_guard_design_v1/llm_study_online.py \
  --direct-model M1 \
  --output .arena-workspace/experiments/llm-study/ubuntu-pilot-v1
```

本批用预先固定的 16 组种子完成 80 局，比较 D0、原版 D1、同输入 D1、D2 和 D3。模型统一为 DeepSeek Flash 非思考模式。每组的五种方法在同一台机器使用同一对守卫种子、地图、规则和正式裁判。仿真在决策期间暂停。

脚本会保存各局日志、回放结果与配对统计，结束时自动校验并打印 `upload_this`。回传该路径指向的一个 `.tar.gz` 即可。已有输出目录用于保留历史运行，再执行时使用新目录。

若旧仓库含此前同步包留下的代码，可在 Git 同步后新建工作树运行，保留原目录中的历史结果。冻结种子与运行配置均在 Git 中。

## 后续留出

初步配对包回传后，使用 `llm_study_power.py` 测算独立留出规模。工具默认以 15 个百分点的绝对胜率差和 80% 功效计算。确定目标后用 `--new-holdout-plan` 生成并冻结计划。正式留出入口加 `--phase holdout --plan <冻结种子计划.json>`。入口会核对模型、方法、配置指纹和决策代码指纹。

## 验证和回放

```bash
python3 -m unittest discover -s benchmarks/two_guard_design_v1 -p 'test_llm*.py' -q
python3 benchmarks/two_guard_design_v1/llm_study_online.py --verify <完整回合回传包>
python3 benchmarks/two_guard_design_v1/llm_study_offline.py --verify <历史离线回传包>
```

完整回合入口接受 `--fixture-for-tests`，可使用预设响应做零 API 的工程检查。历史离线分析继续读取原来冻结的 `observations.json` 和 `evaluation.json`。Ubuntu 当前配对批次可直接使用 Git 中的配置启动。

### 已冻结的独立留出方案

初步批次已回传并完成分析。Git 中的 `config/llm_study_holdout_seeds.json` 固定了 360 组新种子、DeepSeek 配置和决策版本。按初步批次估计，六批全部运行约需 11.7 小时。可通过 Git 同步后，在同一个干净工作树上按顺序运行六个批次。每批 60 组，批次输出目录各自独立。

```bash
python3 benchmarks/two_guard_design_v1/llm_study_online.py \
  --phase holdout \
  --plan benchmarks/two_guard_design_v1/config/llm_study_holdout_seeds.json \
  --direct-model M1 \
  --batch 1/6 \
  --output .arena-workspace/experiments/llm-study/ubuntu-holdout-v1-batch-01
```

随后使用 `--batch 2/6` 至 `--batch 6/6`，输出目录分别以 `batch-02` 至 `batch-06` 结尾。回传各批最后一行 `upload_this` 指向的归档。本地用 `llm_study_holdout.py` 校验并合并，只有所有冻结种子齐全且互不重叠时才生成配对统计。
