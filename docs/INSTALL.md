# 安装、目录与环境配置

## 1. 安装层次

只验证规则 / 基线：Python 3.11 + `requirements-cpu.txt` 即可，不需要 GPU、Isaac 或任何 API Key。

需要 Isaac RGB / USD / 回放：在独立 Python 3.11 环境中安装固定 Isaac 组合、NVIDIA 驱动、Unitree URDF 和 PBR 素材。下面是原工程已使用版本的可迁移安装步骤；完整联网重装尚未在第二台机器上验收，因此请同时执行 `pip check` 和逐层 smoke test。

参考开发机为 Ubuntu 22.04.5、RTX 4050 Laptop 6 GB、驱动 580.159.03。能够跑小规模 headless 演示，不代表 6 GB 满足官方推荐要求或能支持多环境训练。正式部署前核对 [Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。首次编译 shader / 导入模型可能显著慢于后续启动。

| 组件 | 固定参考版本 |
| --- | --- |
| Python | 3.11；原机 3.11.15 |
| Isaac Sim | 5.1.0，分发包版本显示 5.1.0.0 |
| Isaac Lab | v2.3.2 / `37ddf626871758333d6ed89cf64ad702aef127d0` |
| PyTorch | 2.7.0 + cu128 |
| Unitree ROS 模型 | `f3772ce54c56ef2d34c6aee8100bc768896c7d19` |
| CPU 数据与图形依赖 | [requirements-cpu.txt](../requirements-cpu.txt) |

## 2. 工作目录

所有命令均在克隆的仓库根目录执行，激活你自己的环境。下面用 `ARENA_DATA_DIR` 作示例变量，不修改系统 `HOME`。

```bash
export ARENA_PROJECT_DIR="$PWD"
export ARENA_DATA_DIR="$PWD/.arena-workspace"
export ARENA_WORKSPACE="$ARENA_DATA_DIR"
export ISAACLAB_PATH="$ARENA_DATA_DIR/projects/unitree/IsaacLab"
export UNITREE_ROS_DIR="$ARENA_DATA_DIR/projects/unitree/unitree_ros"
mkdir -p "$ARENA_DATA_DIR/projects/unitree"
```

若已有集中式 workspace，将 `ARENA_WORKSPACE` 改成绝对路径并指定已有依赖的路径即可，脚本不会强制覆盖这些环境变量。

| 变量 | 默认值 / 用途 |
| --- | --- |
| `ARENA_WORKSPACE` | `<仓库>/.arena-workspace`；统一容纳缓存、场景和实验 |
| `ARENA_PYTHON` | `python`；包装脚本使用的解释器，可指定绝对路径 |
| `ISAACLAB_PATH` | `<workspace>/projects/unitree/IsaacLab` |
| `UNITREE_ROS_DIR` | `<workspace>/projects/unitree/unitree_ros` |
| `UNITREE_RL_LAB_PATH` | 仅历史物理探针相关；比赛主线不需要 |
| `XDG_CONFIG_HOME` | 默认 `~/.config`；凭据在其 `quadruped-arena/` 子目录 |
| `ARENA_DEEPSEEK_CONFIG` | 可选；DeepSeek 配置 JSON 的绝对路径 |
| `OMNI_KIT_ACCEPT_EULA` | 只有你已阅读并接受 NVIDIA 条款后才设 `YES` |

请不要把 `ARENA_WORKSPACE` 设成仓库根目录；尽管产物目录已忽略，独立目录更便于归档和迁移。

## 3. 准备独立 Isaac 环境

推荐从新的 Conda / Miniforge 环境开始（不要与 CPU `.venv` 同时激活）：

```bash
conda create -n quadruped-arena python=3.11 pip
conda activate quadruped-arena
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
python -m pip install wheel==0.45.1 setuptools==75.8.0
python -m pip install flatdict==4.0.1 --no-build-isolation
```

NVIDIA 安装包及扩展缓存很大，需要稳定网络和充足磁盘。详见 [Isaac Sim 5.1 安装文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)。这里不下载/分发安装包，也不自动代你接受许可。

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
git -C "$ISAACLAB_PATH" checkout 37ddf626871758333d6ed89cf64ad702aef127d0

# 必要的固定版本兼容性补丁；仅适用于上述 commit 的干净检出。
git -C "$ISAACLAB_PATH" apply --check "$ARENA_PROJECT_DIR/patches/isaaclab-v2.3.2-starlette.patch"
git -C "$ISAACLAB_PATH" apply "$ARENA_PROJECT_DIR/patches/isaaclab-v2.3.2-starlette.patch"
python -m pip install -c constraints-isaac.txt -e "$ISAACLAB_PATH/source/isaaclab"
python -m pip install -r requirements-cpu.txt -c constraints-isaac.txt
python -m pip check
```

补丁原因：固定版本 Isaac Lab 的 `starlette==0.49.1` 与 Isaac Sim 5.1 的 FastAPI 约束冲突；原工程使用 `0.45.3`。`flatdict` 构建依赖旧 setuptools 提供的 `pkg_resources`。不要在未知版本上强行应用此补丁，也不要用 `--no-deps` 掩盖未解决依赖。若已复用原机安装，补丁很可能已经存在，不要重复应用。

比赛主线只使用 `isaaclab` 核心和本仓库脚本，不需要安装完整 PPO/RSL-RL 训练栈。若自行扩展训练，请另行管理 `isaaclab_tasks`、`isaaclab_rl`、策略权重和训练依赖。

Linux 渲染还需要可用的 Vulkan / NVIDIA 驱动。文字绘制使用系统字体，Ubuntu 可安装 `fonts-dejavu-core`、`fonts-noto-cjk`。不要以 root 身份运行比赛。

## 4. Unitree 模型和 PBR 素材

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git "$UNITREE_ROS_DIR"
git -C "$UNITREE_ROS_DIR" checkout f3772ce54c56ef2d34c6aee8100bc768896c7d19
python scripts/prepare_assets.py
python scripts/prepare_assets.py --check
```

这里仅使用仓库中的 URDF 和 mesh；**不要求安装 ROS、Gazebo 或连接真实机器人**。Go2-W / Go2 的 USD 缓存会在首次 Isaac 使用时从 URDF 生成；G1 的显示网格直接从固定 URDF 引用的模型构建。生成的 USD 引用本机路径，应在新机重新生成，不能把旧 USD 当独立可移植资产。

PBR 来源为 ambientCG 的 `Concrete023 2K`、`Concrete034 2K`、`Asphalt012 1K`、`Metal032 2K`，共十二张 Color / NormalGL / Roughness JPG，约 34 MB（ZIP 下载包含额外文件，可能更大）。脚本仅安装白名单成员并校验原始 SHA-256，不会偷偷替换修改过的文件。任何哈希不符均停止，需人工核对源素材是否变更。

若不能联网，可从 [素材清单](../assets/pbr_manifest.json) 的官方地址下载四个 ZIP 到同一目录：

```bash
python scripts/prepare_assets.py --archive-dir /absolute/path/to/pbr-zips
```

也支持 `<目录>/<素材ID>/<原始ZIP名>` 的原机缓存布局。未下载资产时 CPU 测试会跳过一项贴图集成检查；下载后必须运行 `--check`，不能仅以跳过测试宣称素材准备成功。

## 5. 分层验收

```bash
python scripts/doctor.py
python -m unittest discover -s scripts -p 'test_*.py' -v
python scripts/cpu_demo.py
python scripts/doctor.py --isaac

# 阅读 NVIDIA 条款后：
export OMNI_KIT_ACCEPT_EULA=YES
bash run.sh --demo --rotating-targets --industrial --headless --preview
bash run.sh --vision-trial --semantic-skills --vision-smoke --headless
```

只有两项 GPU 预检成功后才建议启用付费模型。`doctor --isaac` 只检查安装元数据、GPU 可访问性、URDF 文件和贴图哈希；它不能替代真实启动验证。

## 6. 常见问题

- `No module named isaaclab`：检查 `which python`、`ARENA_PYTHON` 和 editable 安装，不要只切换终端名字。
- GPU 不可访问 / Vulkan 错误：在有设备访问权限的宿主机执行，检查驱动、容器 GPU 挂载；CPU 测试不受影响。
- `Missing PBR input`：确认运行与下载素材时 `ARENA_WORKSPACE` 相同，运行 `prepare_assets.py --check`。
- URDF 找不到：检查 `UNITREE_ROS_DIR/robots/...`，不要把变量设成 `robots` 子目录。
- 首次运行长时间没有图片：查看启动打印的 `runtime.log`，可能在编译 shader / 转换 URDF。不要无意重复启动多个 GPU 进程。
- API 报模型不存在/额度/参数错误：先停，核对官方渠道与当前供应商文档。配置中的模型别名不是不可变权重；不要自动换模型后继续声称是同一实验。
- LLM 未返回可验证 JSON：程序停止并保留失败记录，没有隐藏策略补救；可能已经产生费用。不要盲目无限重试。
- 报告找不到历史 run：源码仓库不附历史实验数据，见 [复现边界](REPRODUCIBILITY.md)。
