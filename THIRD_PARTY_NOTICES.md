# 第三方软件与资产说明

仓库根目录 MIT 仅覆盖本项目原创代码与文档，不改变第三方软件、资产、服务和字体的许可证。

| 依赖 / 资产 | 来源 | 处理方式 |
| --- | --- | --- |
| NVIDIA Isaac Sim 5.1 | [官方安装文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html) | 不随仓库分发；用户自行安装并接受 NVIDIA 对应条款 |
| Isaac Lab v2.3.2 | [官方仓库](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2) | 不整体分发；遵循其 LICENSE；附一个明确标注的版本兼容补丁 |
| Unitree Go2 / Go2-W / G1 URDF 和 mesh | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | 从固定 commit 获取；上游仓库 BSD-3-Clause，资产相关通知也以该提交为准；不将机器人资产改称 MIT |
| PBR 贴图 | [ambientCG](https://ambientcg.com/)、[许可说明](https://docs.ambientcg.com/license/) | CC0-1.0；保存出处和 SHA-256；不将原始贴图塞入 Git |
| Python 包 / PyTorch | 各上游项目 | 通过包管理器安装；保留各自许可义务 |
| DejaVu / Noto 字体 | 系统字体包 | 不在仓库分发字体文件；用户自行安装 |
| DeepSeek / 阿里云百炼 | 供应商官方 API | 服务条款和费用独立；仓库不提供密钥、额度或模型权重 |

`docs/images/dockyard-overview.png` 是原工程在 Isaac Sim 中生成的场景截图（DOCK 07 v1，非 AI 生成）；`match-preview.png` 是发布副本原生比赛预检的截图。场景主要结构由本项目代码生成，渲染使用上述 ambientCG 表面材质和 Unitree 角色。截图是说明材料，不等于获得第三方原始资产的额外授权。

PBR 单项来源与文件哈希见 `assets/pbr_manifest.json`。贴图变更需同步审核清单；不能只改 URL 让程序默默下载另一份素材。

兼容性补丁 `patches/isaaclab-v2.3.2-starlette.patch` 仅修改上游安装依赖的版本约束。其上下文来自指定 Isaac Lab 版本；使用者应保留上游的许可证和版权通知。
