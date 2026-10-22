# 工业战术训练场 · 视觉升级 v1

本版是实际 Isaac Sim / USD 三维场景，不是 AI 生成图片，也不是对旧渲染做后期换皮。
方向是写实工业训练场：深色硬质地面、灰色预制混凝土、低饱和蓝灰设备、黄色安全标识。
这是第一版美术场景，不声称已达到成品 3A 游戏的资产与打磨水平。

## 边界与规则

- 原比赛核心区仍是 **12 × 8 m**，掩体位置、碰撞盒尺寸、占点坐标、射击与得分规则未改。
- 视觉围合扩展到 **18 × 14 m**，外围为维护通道、铁网、围墙、入口、控制室和工业背景建筑；不是新增合法得分/导航区域。
- 保留平整硬质地面，没有恢复沙地或散落碎石。
- High3 的固定轴、固定角度坍塌保留。墙体标牌、色带、细节随父节点移动，提供坍塌前后两份场景。
- 视觉倒角内缩，仍使用原始矩形碰撞体；毫米级装饰不参与 R0 质点射线判定。外围装饰暂不配置碰撞，不可直接当作扩展后的可训练地图。
- 这是独立美术分支：旧 `--demo --rules-v3 --adaptive` 不会被自动替换；本次机器人是静态比例参照，不是新一轮比赛结果。

## 已完成的内容

1. 混凝土墙接入颜色、OpenGL 法线、粗糙度贴图；薄倒角、模板拼缝、固定点、墙体编号和底部警示色带。
2. 沥青核心区、混凝土检修步道、伸缩缝、排水沟、边界标线和平面占点圈。
3. 分层边界：近处围墙与出入口，两侧铁网，后部控制室、服务门、设备柜和灯杆。
4. 后景工业建筑，减少低机位看到无限空地的感觉。
5. 原生路径追踪，太阳光与天空补光，三个保存到 USD 的观察机位。

贴图只改变材质着色：法线不是实际起伏，未开启位移，也不改变抓地力。核心碰撞体保留静摩擦 0.8、动摩擦 0.7。

## 素材来源与目录

以下素材来自 ambientCG，使用其 [CC0 1.0 许可](https://docs.ambientcg.com/license/)：

| 素材 | 用途 | 分辨率 |
| --- | --- | --- |
| [Concrete034](https://ambientcg.com/view?id=Concrete034) | 主要混凝土墙、步道 | 2K JPG |
| [Concrete023](https://ambientcg.com/view?id=Concrete023) | 外围风化墙面、警示色带底纹 | 2K JPG |
| [Asphalt012](https://ambientcg.com/view?id=Asphalt012) | 沥青地面 | 1K JPG |

本机缓存：`/home/yang/workspace/simulators/assets/quadruped-arena/pbr/`。
每次运行的 `art_manifest.json` 记录原始下载地址、实际使用的贴图路径和 SHA-256；原下载压缩包保留。
场内标牌为本项目程序排版生成；不是外部商标贴图。机器人沿用既有本机 Unitree 资产，来源许可不因本次贴图许可而改变。

考虑 RTX 4050 Laptop 6 GB 显存，主材质使用 1K/2K，没有盲目导入 8K 材质包或高多边形完整游戏地图。

## 复现与查看

```bash
cd /home/yang/workspace/projects/quadruped-arena

# 高清：四张 1600×1000，目标 256 spp，原生路径追踪
bash run.sh --art --headless

# 快速预览：1120×700，目标 64 spp
bash run.sh --art --preview --headless

# 打开 Isaac 窗口，渲染后保留场景供手动浏览；关闭窗口退出
bash run.sh --art --preview --interactive
```

交互入口代码已加入；自动验收使用 headless，不声称已人工操作过交互窗口。

输出目录：

- 图像、日志、素材校验、代码快照：`workspace/experiments/runs/quadruped-arena/<运行时间>/`。
- USD 与标牌：`workspace/simulators/scenes/quadruped-arena/<运行时间>/`。
- `courtyard_before.usda`：坍塌前；`courtyard_after.usda`：坍塌后。
- 原生图片：`overview.png`、`courtyard.png`、`cover_detail.png`、`collapse_overview.png`。

在 Isaac 中 File → Open 打开 USD，选择 `/World/Cameras/overview`、`courtyard` 或 `cover_detail`。
若手动打开后显示效果与截图不同，请切换渲染模式为 RTX Path Tracing；采样/降噪是启动脚本设置，不保证随 USD 文件自动切换。
USD 引用当前机器上的绝对贴图和机器人资产路径。移交到另一台电脑时，需要携带这些资源并重定位路径，不能只发送单个 USD。

## 验证与后续

2026-10-21 高清实测：`20261021-130649-233619`。进程退出码 0，`process.json` 为 `passed: true`，四张图片均人工查看；70 项 CPU 自动测试通过。

- [主视角](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-130649-233619/courtyard.png)
- [全景](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-130649-233619/overview.png)
- [墙体近景](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-130649-233619/cover_detail.png)
- [坍塌后](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-130649-233619/collapse_overview.png)
- [坍塌前场景](/home/yang/workspace/simulators/scenes/quadruped-arena/20261021-130649-233619/courtyard_before.usda)
- [运行结果与素材清单](/home/yang/workspace/experiments/runs/quadruped-arena/20261021-130649-233619/result.json)

日志仍有沿用 Go2-W 导入资产的跨引用材质警告及部分无效视觉引用警告；已绑定本地材质，四张渲染中机器人可见。本轮未重做厂商 URDF 资产，不把这些既有警告描述为已消除。

CPU 回归覆盖原规则、策略及新增美术网格；原生场景中检查原碰撞盒、装饰无碰撞，以及倾倒前后碰撞体的世界坐标与尺寸。
这不是机器人接触动力学、扩展导航图或比赛平衡性验收。

若后续决定把实际比赛区也扩大，需一起调整导航边界、出生点、任务点、射程/感知范围及回合时长；不能只把地面放大。
若将美术接入训练，可保留轻量碰撞层，并按需开关高质量材质、铁网与灯光；本次没有测训练吞吐或实时帧率。
