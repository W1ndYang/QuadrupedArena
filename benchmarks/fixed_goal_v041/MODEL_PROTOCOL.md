# 模型、技能和记录协议

顶层模型只输出结构化意图，不能运行代码或修改裁判。模型调用时整个仿真暂停；这是决策有效性实验，不是已满足实时端侧延迟的系统。

## 输入

系统提示词给出地图墙盒、路点、速度、姿态、规则、双方初始位置以及技能语义。每次用户消息提供当前自身状态、剩余时间、当前合法对手检测、有限旧记忆、上次决策以来的合法观测摘要、路线选项、烟点建议、探身候选、近期执行事件及技能返回。

底层每 20 ms 感知；模型输入保留约 70 个均匀抽样点及所有可见性切换边缘，完整合法序列另存 `legal_samples.json`。无全场图片或隐藏对手位置。路线暴露预测基于最后合法敌位，忽略未来运动与朝向，不是安全保证。

最多附带最近四个 `observation_id` / `skill` / `params` 动作，**不带前轮 assessment 或完整推理**。公开朝向仍是弧度数值，可能超出一周，解释时应归一化；本包保留原协议以复现原实验。

## 输出

```json
{
  "observation_id": 0,
  "skill": "PLAN",
  "params": {
    "horizon_seconds": 6,
    "steps": [{
      "skill": "FOLLOW_ROUTE",
      "params": {"waypoints": ["CENTRAL_NORTH", "NORTH_WINDOW"], "posture": "upright", "fire": "off"},
      "on_contact": "retreat"
    }]
  },
  "assessment": {
    "observation_evidence": "当前合法观测与信息年龄。",
    "state_evidence": "自身位置、任务与资源状态。",
    "decision_summary": "选择哪条路线和执行条件。",
    "uncertainty": "尚未知的信息及主要风险。"
  }
}
```

上例只说明格式，不是可直接用于任意场景的策略。计划 1～5 个节点，时限 4～8 s。计划完成、时限达到、短探结束或紧急撤退完成时触发新决策，不再每 2 s 强行打断。

## 技能

| 技能 | 参数 | 行为 |
| --- | --- | --- |
| `FOLLOW_ROUTE` | 1～4 个命名路点、posture、fire | A* 连接并平滑为连续路线，中间路点容差 0.65 m |
| `MOVE` | target、posture、fire | 前往单一路点 |
| `TAKE_COVER` | target、posture | 不开火移动到指定路点；路点不天然安全 |
| `PEEK_BURST` | candidate_id、max_cycles | 到候选 home，短探再倒退返回，最多 1～3 次 |
| `TAG_AND_ADVANCE` | target、posture | 有当前合法机会则先标记一次，再推进 |
| `SMOKE_AND_MOVE` | location、target、posture | 按投掷规则施烟后移动；不主动开火 |

`posture` 为 `upright` 或 `low`；`fire` 为 `off` 或 `opportunistic`。后者在满足合法交战条件时会主动停下瞄准，可能拖慢任务推进。

`on_contact=retreat` 表示看见 8 m 内人形时中断并撤向基于已知敌位选择的附近遮挡位置；`continue` 表示继续该技能。普通移动的接触守卫优先于机会射击，所以 `retreat + opportunistic` 不代表先打再退。当前实现对 `PEEK_BURST` 提前分流，接近 home 阶段没有执行这个通用守卫；这是已知缺陷，不应把参数说明当成已经完整实现的保证。

短探一般单程约 0.30 m、速度/倒退上限 1.6 m/s、加速度 6 m/s²、外侧停留 0.04 s、遮挡内间隔 0.18 s。预测连续暴露不超过 0.70 s 才发起。实际新观测使 home 失效或探出估计暴露过长时提前返回。v0.4.1 专门修复提前返回时尚未停稳就反向的问题；没有同时修改策略或拓展技能。

## 记录与计量

- `api_calls/*_request.json`：系统提示、公共输入、近期动作和无密钥 API 配置。
- `*_response.json`：最终 JSON 文本、服务端 usage、模型标识、外部开始/结束 UTC、请求总耗时；仅记录收到的推理字符数，不保存完整推理。
- `accepted/*.json`：校验后动作、请求哈希及对应仿真时刻。
- `decisions.json` / `skill_events.json`：模型指令及真实执行中断/返回，必须结合查看。
- `timeline.json` / `events.json`：全状态轨迹及裁判结果，仅供审阅，不回传模型。

API 总耗时包含排队、生成和网络，不能视作纯思考时间。总 token 已含 reasoning token，不能再相加。本次 9 个完整调用共 89,885 个已知 token，没有失败请求；每个布局仅一局，不能推导稳定成功率。

更换供应商或修改提示词、技能守卫、模型档位、观测能力、布局后，需要建立新实验目录和来源记录。不要在旧动作与新公共输入不一致时强行继续“重放”。
