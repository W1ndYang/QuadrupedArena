# 四足决策协议 `breach-protocol-v1`

顶层决策器（大模型或规则基线）只输出一个技能序列，不能运行代码，也不能读取裁判状态。底层技能每 20 ms 闭环执行，只用四足自己的合法观测。

## 何时请求决策

`agent.QuadAgent` 在以下时刻发出请求，请求原因写在 `trigger` 字段里：

- 开局；
- 计划执行完毕；
- 技能失败或被打断，例如接触撤退、投掷被拒；
- 上一次返回不合法。

同一时间最多只有一个请求在途。等待期间，当前技能继续执行；如果没有可执行的技能，四足就原地等待，人在 9 m 内时会自动撤到隐蔽点。新计划到达后，立即替换剩余的计划。

## 时序

| `--timing` | 行为 | 适用 |
| --- | --- | --- |
| `virtual`（无头默认） | 请求在后台线程执行，比赛不按墙钟节流；但请求在途时，比赛时间不能超过请求发出后经过的墙钟时间。所以计划生效时，比赛时间正好过去了实测延迟那么久。没有请求在途时全速推进 | 用真实延迟做批量实验 |
| `realtime`（真人默认） | 后台线程执行，比赛按墙钟推进；计划在哪一刻到达就在哪一刻生效 | 真人对战 |
| `fixed:S` | 计划在请求发出 S 秒（比赛时间）后生效，与墙钟无关 | 可复现测试 |
| `pause` | 阻塞调用，比赛等待返回 | 只用于和旧的“调用时暂停”研究对照 |

每次调用都会记录：发出时刻 `sent_t`、生效时刻 `applied_t`、比赛内延迟 `latency_s`、墙钟用时 `wall_s` 和 token 用量。

## 系统提示词

用 `python -m breach_arena prompt --map BA-1` 查看。内容包括：

- 胜负规则、速度、感知、烟雾参数；
- “比赛不暂停”的说明；
- 技能表和输出格式；
- 公开地图：墙盒、目标、出生点、命名点位，以及可选的打法路线。

可选部分由两个开关控制：

- `--hints none|routes|full`：`routes`（默认）给出投烟点到目标的路线、接近路线、埋伏点突袭路线和诱敌转移路线；`full` 再加上每个投烟点的烟落点提示（目标烟、路径烟）；`none` 只给墙和点位。
- `--no-strategy`：去掉“守点吃烟、出击吃迂回、目标烟只挡贴点”这段策略提示，用来测试模型能否自己推断出对策。

提示词里**不包含**矩阵用例、人方机器人参数和规则基线的偏好，测试会检查这一点。

## 每次请求的用户消息（只含合法信息）

```json
{"observation_id": 3, "trigger": "plan finished (HOLD: timer)", "t": 5.54, "remaining_s": 54.46,
 "me": {"xy": [-3.0, 3.9], "yaw": 0.1, "speed": 0.0, "posture": "upright", "smoke_left": 1, "casting": false, "cooldown_left": 0},
 "human_visible_now": {"xy": [7.0, 0.05], "yaw": 3.1, "t": 5.54, "stunned": false},
 "human_last_known": {"xy": [7.0, 0.05], "yaw": 3.1, "t": 5.54},
 "sightings": [{"t": 3.1, "human": [7.0, 0.1], "me": [-3.0, 3.9]}],
 "smokes": [], "current_skill": null, "queued": [],
 "recent_skills": [{"skill": "MOVE", "status": "done", "reason": "arrived", "ended": 3.04}],
 "earlier_rounds": [{"round": 1, "result": {"winner": "quadruped", "reason": "goal", "t": 12.9}, "skills": [], "human_sightings": []}]}
```

- 只有在裁判判定四足此刻能看到人时，`human_visible_now` 才有值。`human_last_known` 最多保留 4 s。`sightings` 是按 0.25 s 抽样的合法目击记录，最多 12 条。
- `earlier_rounds` 用于多局对战，里面只有上一局的结果、技能执行情况和合法目击记录，没有人方的真实轨迹。

## 返回

```json
{"observation_id": 3,
 "plan": [{"skill": "SMOKE_RUSH", "params": {"throw_from": "T_N", "smoke_at": [8.6, 2.2],
           "approach_via": ["N_LANE_W", "N_LANE_E"], "run_via": ["N_DOOR_OUT", "N_DOOR"]}}],
 "assessment": {"human_style": "hold", "evidence": "…", "decision_summary": "…", "risk": "…"}}
```

- `observation_id` 必须与请求一致。计划为 1～5 步；点位可以写地图中的命名点，也可以写 `[x, y]`。
- 返回不合法时（不是 JSON、技能未知、点位不存在、参数越界），会记录错误，不会执行任何猜测出来的动作，2 s 后带着新的观测重新请求。网络或接口错误同样记录，本次调用不重试，也是 2 s 后再发下一次请求。

## 技能

| 技能 | 参数 | 行为 |
| --- | --- | --- |
| `MOVE` | `to`、`via`（≤6）、`on_contact`=`retreat`/`continue` | 沿路径前往。`retreat`：人在 9 m 内连续可见 0.3 s，且前方路线仍然暴露，就中止并撤到隐蔽点（4 m 内立即撤）；高速穿过门洞视线不会触发 |
| `HOLD` | `seconds` 0–10、`on_contact`、`until`=`contact` | 原地停留；人可见时朝向人 |
| `SMOKE_RUSH` | `throw_from`、`smoke_at`、`approach_via`、`run_via` | 前往投烟点（接近阶段遇接触会撤）→ 站定投烟 → 沿 `run_via` 冲向目标（不回头）。投掷被拒时以失败返回，烟保留 |
| `FEINT` | `at`、`approach_via`、`show_s`、`leave_inside_m`、`staging`、`relocate_via` | 到对方可见但打不到的位置露面，人逼近或时间到就隐蔽转移到埋伏点 |
| `DASH` | `via` | 不放烟，直接冲点 |
| `TAG` | `max_s` | 站定锁定可见的人，命中让人失能 1 s（不算胜利） |
| `RETREAT` | — | 撤到离当前位置较近、且人从最后合法位置看不到的命名点 |

## 记录

每局的输出目录里有：

- `roundN_api/system_prompt.txt`；
- `roundN_api/NNN_request.json`：精简后的观测包，加上不含密钥的客户端配置；
- `roundN_api/NNN_response.json`：原始返回文本、解析后的计划、校验错误、延迟和 token；
- `roundN.replay.json.gz` 里的 `quadruped.log`：每次请求、生效和技能结束的事件。

服务商密钥只从环境变量或 `--key-file` 读取，不会写进任何文件。
