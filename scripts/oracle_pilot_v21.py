"""Read-only v2 bundle reanalysis and independent four-window replay validation.

analyze_bundle never launches Isaac. replay_validate only launches exact formal
prefixes; it has no candidate action, policy lookup, or matrix path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

from arena_paths import PROJECT
from oracle_pilot import (DEFAULT_CONFIG, atomic_json, digest, read, resolve_config,
                          run_task, task_base)
from oracle_pilot_discovery import (DISCOVERY_SCHEMA, analyze_records, boundary_evidence,
                                    near_miss_markdown, reference_diff,
                                    strict_replay_reference_digest)

ANALYSIS_VERSION = "oracle_pilot_v2_1_angle_reanalysis"
DEFAULT_ANALYSIS = PROJECT / "docs/codex_report/oracle_pilot_v21_corrected_analysis"
DEFAULT_REPLAY = PROJECT / "docs/codex_report/oracle_pilot_v21_replay_validation"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_bundle(path):
    """Validate the entire original ZIP before interpreting any of its files."""
    archive = zipfile.ZipFile(path)
    try:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Input ZIP has duplicate names or CRC errors")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Input ZIP contains an unsafe member path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("discovery_schema") != DISCOVERY_SCHEMA:
            raise ValueError("Input is not an action-prefix discovery v2 bundle")
        entries = manifest["files"]
        if manifest["file_count"] != len(entries):
            raise ValueError("Input manifest count mismatch")
        expected = {}
        for item in entries:
            member = item["path"]
            data = archive.read(member)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Input manifest checksum mismatch: {member}")
            expected[member] = item["sha256"]
        checksum_lines = archive.read("SHA256SUMS").decode().splitlines()
        checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
        if checksums != expected:
            raise ValueError("Input SHA256SUMS disagrees with manifest")
        return archive, manifest
    except Exception:
        archive.close()
        raise


def package_artifacts(output, filename, metadata):
    output = Path(output)
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name not in
             (filename, filename + ".tmp", "manifest.json", "SHA256SUMS")]
    entries = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                "sha256": sha256_file(p)} for p in files]
    atomic_json(output / "manifest.json", {"file_count": len(entries), "files": entries, **metadata})
    (output / "SHA256SUMS").write_text("".join(f"{x['sha256']}  {x['path']}\n" for x in entries))
    temp = output / (filename + ".tmp")
    archive = output / filename
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for path in files + [output / "manifest.json", output / "SHA256SUMS"]:
            z.write(path, path.relative_to(output))
    with zipfile.ZipFile(temp) as z:
        for item in entries:
            data = z.read(item["path"])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Output archive manifest mismatch")
        if z.testzip() is not None:
            raise ValueError("Output archive CRC failure")
    temp.replace(archive)
    return archive


def archive_relative_paths(value):
    """Replace temporary extraction paths with immutable original ZIP members."""
    if isinstance(value, list):
        return [archive_relative_paths(x) for x in value]
    if isinstance(value, dict):
        result = {k: archive_relative_paths(v) for k, v in value.items()}
        if "sample_id" in result and "source_run" in result and "target_step" in result:
            result["full_diagnostic_file"] = (f"raw/{result['source_run']}/diagnostics/"
                                              f"reference_{result['target_step']:03d}.json")
        return result
    return value


def pairwise_group_details(group, snapshots):
    candidates = group["candidate_samples"]
    pairs = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            a, b = snapshots[left["sample_id"]], snapshots[right["sample_id"]]
            pairs.append({"left": left["sample_id"], "right": right["sample_id"],
                          "own_position_distance_m": round(math.dist(a["a"]["xy"], b["a"]["xy"]), 6),
                          "difference": reference_diff(a, b)})
    return pairs


def corrected_report(samples, near, candidates, original, input_hash):
    lines = ["# Oracle Pilot v2.1 修正后离线分析", "",
             f"- 原始 ZIP SHA-256：`{input_hash}`",
             f"- 原始发现源码指纹：`{samples['source_sha256']}`",
             f"- 分析版本：`{ANALYSIS_VERSION}`",
             f"- 条件：场景 `{samples['scene']}`，规则 `{samples['rules_config']}`，"
             f"对手 `{samples['opponent_policy']}`，比赛 seed `{samples['game_seed']}`，"
             f"合成视觉 seed `{samples['synthetic_seed']}`。",
             f"- 原始报告合格组：{original['qualifying_group_count']}；角度修正后：{samples['qualifying_group_count']}。",
             f"- 前缀：{samples['completed_prefixes']}/{samples['planned_prefixes']} 完成；边界访问 {samples['boundary_visits']}；不同规范 ID {samples['unique_state_ids']}。",
             "", "## 摘要语义", "",
             "`replay_reference_digest` 对原始 R1 角度表示保持严格 1e-6 量化；"
             "`meaningful_state_digest` 对 a.yaw/b.yaw 按 2π 周期归一化后量化。"
             "仅相机/视觉记忆、时间/日志路径元数据或 2π 表示差异不算不同真实动力学状态。", "",
             "## 合格精确碰撞组", "",
             "| 完整 R2-L state ID | 边界 | 全部候选 | 推荐样本前缀 |", "|---|---:|---:|---|"]
    for group in candidates["groups"]:
        selected = next(g for g in samples["groups"] if g["state_id"] == group["state_id"])
        lines.append(f"| `{group['state_id']}` | {group['boundary_time_seconds']} 秒 | "
                     f"{len(group['candidate_samples'])} | " + ", ".join(f"`{x['prefix_id']}`" for x in selected["samples"]) + " |")
    lines += ["", "### 候选样本实际状态", "",
              "| ID 尾部 | 前缀 | 边界 | 自身位置 (m) | 官方分数 | 当前目标驻留 (s) | 是否推荐 |",
              "|---|---|---:|---|---:|---:|---|"]
    for group in candidates["groups"]:
        selected_ids = {x["sample_id"] for x in next(g for g in samples["groups"] if g["state_id"] == group["state_id"])["samples"]}
        for item in group["candidate_samples"]:
            xy = "(" + ", ".join(f"{value:.4f}" for value in item["own_xy"]) + ")"
            lines.append(f"| `{group['state_id'][-12:]}` | `{item['prefix_id']}` | "
                         f"{item['target_step'] * 2} 秒 | `{xy}` | {item['official_score']:.2f} | "
                         f"{item['active_dwell_seconds']:.2f} | {'是' if item['sample_id'] in selected_ids else '否'} |")
    lines += ["", "### 全部候选的完整前缀与摘要", ""]
    for group in candidates["groups"]:
        lines.append(f"**状态 `{group['state_id']}`**")
        lines.append("")
        for item in group["candidate_samples"]:
            lines.append(f"- `{item['sample_id']}`：前缀 `{item['prefix_id']}`，"
                         f"边界 {item['target_step'] * 2} 秒；"
                         f"replay digest `{item['replay_reference_digest']}`；"
                         f"meaningful digest `{item['meaningful_state_digest']}`。")
            lines.append("  完整动作前缀：" + " → ".join(f"`{x}`" for x in item["full_prefix_action_ids"]) + "。")
        lines.append("")
    lines += ["### 候选间位置距离与真实差异", ""]
    for group in candidates["groups"]:
        lines.append(f"- 状态 `{group['state_id'][-12:]}`：")
        names = {x["sample_id"]: x["prefix_id"] for x in group["candidate_samples"]}
        for pair in group["pairwise_differences"]:
            values = pair["difference"]["values"]
            fields = "; ".join(f"`{field}` {value['left']} / {value['right']}"
                               for field, value in values.items() if field not in ("tick", "t", "profile_id"))
            lines.append(f"  - `{names[pair['left']]}` / `{names[pair['right']]}`："
                         f"自身位置距离 {pair['own_position_distance_m']} m；{fields or '仅元数据差异'}。")
    lines += ["", "8 秒组用首个前缀作锚点，再依自身位置的最小距离选择跨度较大的样本；"
              "原先空间距离很近的 `apv2_02` 候选仍完整保留在 candidate_samples_all.json。"
              "6 秒组只有三个不同真实状态，全部保留；约 0.02 的得分/驻留差异只是实际记录，不代表已证明动作价值差异。", "",
              "## 被剔除的精确 ID 假碰撞", ""]
    for item in near["nonqualifying_exact_matches"]:
        if item["reason"] == "periodic_angle_alias":
            lines.append(f"- `{item['state_id']}`：仅原始角度表示不同，"
                         f"`{item['raw_angle_values']}`；周期归一化后动力学摘要相同。")
    lines += ["", f"near-miss 不同 ID 候选对：{near['near_pair_count']}；"
              "只用于诊断，不能作为样本或策略输入。", "",
              "## 下一关口", "",
              "四条唯一前缀须在 Ubuntu 新进程逐边界重放。当前只是旧 ZIP 的离线重分析，"
              "没有运行 Isaac，也没有生成动作价值矩阵、策略表或 policy_eval。", ""]
    return "\n".join(lines)


def analyze_bundle(input_zip, output, config):
    input_zip = Path(input_zip).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not input_zip.is_file():
        raise FileNotFoundError(input_zip)
    if output == input_zip.parent or input_zip in output.parents:
        raise ValueError("Corrected analysis must be separate from the read-only input ZIP")
    started = time.monotonic()
    input_hash = sha256_file(input_zip)
    archive, manifest = checked_bundle(input_zip)
    try:
        original = json.loads(archive.read("samples.json"))
        old_config = json.loads(archive.read("resolved_config.json"))
        library = json.loads(archive.read("prefix_library.json"))
        if (original.get("discovery_schema") != DISCOVERY_SCHEMA or
                original.get("source_sha256") != old_config.get("source_sha256") or
                library.get("hash") != config["prefix_library"]["hash"] or
                original.get("prefix_library_hash") != library["hash"]):
            raise ValueError("Original source, schema, or formal prefix library mismatch")
        for key in ("scene", "rules_config", "opponent_policy", "game_seed", "discovery_synthetic_seed"):
            if old_config[key] != config[key]:
                raise ValueError(f"Experiment condition changed: {key}")
        records = [json.loads(archive.read(name)) for name in archive.namelist()
                   if name.startswith("tasks/") and name.endswith(".json")]
        records = [x for x in records if x["task"].get("mode") == "discover"]
        plans = {p["prefix_id"]: p for p in library["prefixes"]}
        records.sort(key=lambda x: int(x["task"]["prefix_id"].split("_")[1]))
        if len(records) != original["attempted_prefixes"]:
            raise ValueError("Discovery task count differs from original samples.json")
        snapshots = {}
        boundaries_by_prefix = {}
        with tempfile.TemporaryDirectory(prefix="oracle_pilot_v21_") as tmp:
            temp_root = Path(tmp)
            for record in records:
                key = record["task_key"]
                prefix_id = record["task"]["prefix_id"]
                if prefix_id not in plans:
                    raise ValueError("Unknown prefix in original bundle")
                raw = temp_root / "raw" / key
                for filename in ("boundaries.json", "decisions.json", "synthetic_samples.json"):
                    member = f"raw/{key}/{filename}"
                    raw.mkdir(parents=True, exist_ok=True)
                    (raw / filename).write_bytes(archive.read(member))
                boundaries = json.loads((raw / "boundaries.json").read_text())
                boundaries_by_prefix[prefix_id] = boundaries
                for step, boundary in enumerate(boundaries):
                    member = f"raw/{key}/{boundary['full_diagnostic_file']}"
                    data = archive.read(member)
                    destination = raw / boundary["full_diagnostic_file"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    snapshot = json.loads(data)["snapshot"]
                    computed = strict_replay_reference_digest(snapshot)
                    if computed != boundary["replay_reference_digest"]:
                        raise ValueError(f"Original replay digest mismatch: {member}")
                    state = boundary["limited"]
                    if (not state["state_id"] or
                            state["state_id"].split(":")[-1] != hashlib.sha256(
                                state["canonical_state"].encode()).hexdigest()):
                        raise ValueError(f"Original canonical state hash mismatch: {member}")
                    snapshots[key + f"_{step:02d}"] = snapshot
                record["raw_dir"] = str(raw)
            analysis_config = {**config, "source_sha256": original["source_sha256"],
                               "discovery_batch": 1}
            samples, near = analyze_records(records, library, analysis_config)
            candidate_groups = []
            for group in samples["all_qualifying_groups"]:
                members = []
                for item in group["candidate_samples"]:
                    snap = snapshots[item["sample_id"]]
                    target = snap["active_target"]
                    members.append({**archive_relative_paths(item),
                                    "own_xy": snap["a"]["xy"],
                                    "official_score": snap["score"],
                                    "active_dwell_seconds": snap["dwell"][target]})
                candidate_groups.append({"state_id": group["state_id"],
                                         "canonical_state": group["canonical_state"],
                                         "category": group["category"],
                                         "boundary_time_seconds": members[0]["target_step"] * 2,
                                         "candidate_samples": members,
                                         "pairwise_differences": pairwise_group_details(group, snapshots)})
            samples = archive_relative_paths(samples)
            samples["analysis_version"] = ANALYSIS_VERSION
            samples["input_zip_sha256"] = input_hash
            samples["original_discovery_source_sha256"] = original["source_sha256"]
            samples["analysis_source_sha256"] = config["source_sha256"]
            near["analysis_version"] = ANALYSIS_VERSION
            near["input_zip_sha256"] = input_hash
            candidate_all = {"analysis_version": ANALYSIS_VERSION,
                             "input_zip_sha256": input_hash, "groups": candidate_groups}
            output.mkdir(parents=True, exist_ok=True)
            prior = output / "input_manifest.json"
            if prior.exists() and read(prior).get("input_zip_sha256") != input_hash:
                raise ValueError("Output directory already belongs to another input ZIP")
            atomic_json(output / "samples_corrected.json", samples)
            atomic_json(output / "near_miss_report_corrected.json", near)
            (output / "near_miss_report_corrected.md").write_text(near_miss_markdown(near))
            atomic_json(output / "candidate_samples_all.json", candidate_all)
            selected_groups = [g for g in samples["groups"] if len(g["samples"]) >= 3]
            selected_samples = [{"sample_id": x["sample_id"], "prefix_id": x["prefix_id"],
                                 "target_step": x["target_step"], "state_id": g["state_id"],
                                 "canonical_state": g["canonical_state"]}
                                for g in selected_groups for x in g["samples"]]
            prefixes = sorted({x["prefix_id"] for x in selected_samples})
            replay_prefixes = []
            for prefix_id in prefixes:
                plan = plans[prefix_id]
                source = boundaries_by_prefix[prefix_id]
                source_key = next(x["task_key"] for x in records if x["task"]["prefix_id"] == prefix_id)
                expected = []
                for step, boundary in enumerate(source):
                    item = boundary_evidence(boundary)
                    item["collapsed"] = snapshots[source_key + f"_{step:02d}"]["collapsed"]
                    expected.append(item)
                replay_prefixes.append({"prefix_id": prefix_id, "prefix_action_ids": plan["prefix_action_ids"],
                                        "prefix_actions": plan["actions"], "source_task_key": source_key,
                                        "expected_boundaries": expected})
            corrected_sha = sha256_file(output / "samples_corrected.json")
            replay_plan = {"analysis_version": ANALYSIS_VERSION, "input_zip_sha256": input_hash,
                           "original_source_sha256": original["source_sha256"],
                           "analysis_source_sha256": config["source_sha256"],
                           "corrected_samples_sha256": corrected_sha,
                           "prefix_library_hash": library["hash"],
                           "scene": original["scene"], "rules_config": original["rules_config"],
                           "opponent_policy": original["opponent_policy"],
                           "game_seed": original["game_seed"],
                           "synthetic_seed": original["synthetic_seed"],
                           "selected_samples": selected_samples, "prefixes": replay_prefixes}
            atomic_json(output / "replay_plan.json", replay_plan)
            (output / "analysis_report_corrected.md").write_text(corrected_report(
                samples, near, candidate_all, original, input_hash))
            input_manifest = {"analysis_version": ANALYSIS_VERSION,
                              "input_zip_path": str(input_zip), "input_zip_sha256": input_hash,
                              "input_manifest_sha256": hashlib.sha256(archive.read("manifest.json")).hexdigest(),
                              "original_source_sha256": original["source_sha256"],
                              "analysis_source_sha256": config["source_sha256"],
                              "analysis_script_sha256": sha256_file(Path(__file__)),
                              "prefix_library_hash": library["hash"],
                              "original_claimed_groups": original["qualifying_group_count"],
                              "corrected_qualifying_groups": samples["qualifying_group_count"],
                              "elapsed_wall_seconds": round(time.monotonic() - started, 3)}
            atomic_json(output / "input_manifest.json", input_manifest)
        result_zip = package_artifacts(output, "corrected_analysis_v21_bundle.zip",
                                       {"analysis_version": ANALYSIS_VERSION,
                                        "input_zip_sha256": input_hash,
                                        "original_source_sha256": original["source_sha256"],
                                        "analysis_source_sha256": config["source_sha256"]})
        return {"groups": samples["qualifying_group_count"], "selected_three_state_groups": len(selected_groups),
                "replay_prefixes": prefixes, "output": str(output), "zip": str(result_zip),
                "elapsed_wall_seconds": input_manifest["elapsed_wall_seconds"]}
    finally:
        archive.close()


def replay_report(summary):
    lines = ["# Oracle Pilot v2.1 完整前缀独立重放", "",
             f"- 结果：`{summary['status']}`；通过前缀 {summary['passed_prefixes']}/{summary['planned_prefixes']}；"
             f"覆盖拟用样本 {summary['covered_samples']}/{summary['planned_samples']}。",
             f"- 原始 ZIP SHA-256：`{summary['input_zip_sha256']}`",
             f"- 修正样本 SHA-256：`{summary['corrected_samples_sha256']}`",
             f"- 本次执行源码指纹：`{summary['replay_source_sha256']}`",
             f"- 已运行任务墙钟合计：{summary['total_task_wall_seconds']} 秒。", "",
             "每条前缀是全新 Isaac 进程；只运行四个正式动作窗口，不提交候选动作。"
             "0、2、4、6、8 秒逐边界核对规范 ID、canonical state、原始参考摘要、"
             "官方分数、Skill、VisualMemory、finished 和 collapsed。", "",
             "| 前缀 | 状态 | 匹配边界 | 最早差异 | 墙钟秒 | task key |",
             "|---|---|---:|---|---:|---|"]
    for item in summary["prefix_results"]:
        mismatch = item.get("first_mismatch") or {}
        difference = (f"{mismatch.get('boundary_step')}：{', '.join(mismatch.get('fields', []))}"
                      if mismatch else "无" if item["status"] == "passed" else "未取得差异 JSON")
        lines.append(f"| `{item['prefix_id']}` | {item['status']} | {item['matched_boundaries']} | "
                     f"{difference} | {item.get('wall_seconds')} | `{item.get('task_key')}` |")
    lines += ["", "失败任务的 launcher stdout/stderr、runtime.log、边界、动作、诊断和差异 JSON"
              "均保留在 `raw/TASK_KEY/` 与 `tasks/TASK_KEY.json`。"
              "`pilot_actions` 仍关闭；本报告不表示已运行 44 项动作矩阵或策略评估。", ""]
    return "\n".join(lines)


def replay_validate(analysis_dir, output, config, prefix_ids=None, retry_failed=False):
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    plan = read(analysis_dir / "replay_plan.json")
    samples_path = analysis_dir / "samples_corrected.json"
    samples = read(samples_path)
    if (plan["analysis_version"] != ANALYSIS_VERSION or
            sha256_file(samples_path) != plan["corrected_samples_sha256"] or
            plan["analysis_source_sha256"] != config["source_sha256"] or
            samples["analysis_source_sha256"] != config["source_sha256"] or
            samples["input_zip_sha256"] != plan["input_zip_sha256"] or
            samples["qualifying_group_count"] < 1 or
            plan["prefix_library_hash"] != config["prefix_library"]["hash"] or
            plan["game_seed"] != config["game_seed"] or
            plan["synthetic_seed"] != config["discovery_synthetic_seed"] or
            plan["original_source_sha256"] != samples["original_discovery_source_sha256"]):
        raise ValueError("Corrected analysis/replay plan is missing or inconsistent")
    selected = [{"sample_id": item["sample_id"], "prefix_id": item["prefix_id"],
                 "target_step": item["target_step"], "state_id": group["state_id"],
                 "canonical_state": group["canonical_state"]}
                for group in samples["groups"] if len(group["samples"]) >= 3
                for item in group["samples"]]
    library = {item["prefix_id"]: item for item in config["prefix_library"]["prefixes"]}
    if (len(selected) != 6 or selected != plan["selected_samples"] or
            len(plan["prefixes"]) != 4 or
            {item["prefix_id"] for item in selected} !=
            {item["prefix_id"] for item in plan["prefixes"]}):
        raise ValueError("Replay plan must cover exactly the six selected boundaries on four unique prefixes")
    for item in plan["prefixes"]:
        source = library.get(item["prefix_id"])
        if (source is None or item["prefix_action_ids"] != source["prefix_action_ids"] or
                item["prefix_actions"] != source["actions"] or
                len(item["expected_boundaries"]) != 5 or
                any(not all(field in boundary for field in
                            ("state_id", "canonical_state", "replay_reference_digest", "score",
                             "skill_status", "visual_memory", "finished", "collapsed"))
                    for boundary in item["expected_boundaries"])):
            raise ValueError(f"Replay prefix plan is incomplete or changed: {item['prefix_id']}")
    planned = {x["prefix_id"]: x for x in plan["prefixes"]}
    requested = list(planned) if prefix_ids is None else prefix_ids
    if not requested or len(requested) != len(set(requested)) or set(requested) - set(planned):
        raise ValueError("Replay prefix IDs must be unique IDs from corrected replay_plan.json")
    if output == analysis_dir or analysis_dir in output.parents:
        raise ValueError("Replay output must be separate from corrected analysis")
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "replay_validation.json"
    if previous.exists() and (read(previous).get("input_zip_sha256") != plan["input_zip_sha256"] or
                              read(previous).get("corrected_samples_sha256") != plan["corrected_samples_sha256"]):
        raise ValueError("Replay output already belongs to a different corrected analysis")
    records = {}
    for prefix_id, item in planned.items():
        task = task_base(config, "replay_validate", plan["synthetic_seed"])
        task.update(prefix_id=prefix_id, prefix_action_ids=item["prefix_action_ids"],
                    prefix_actions=item["prefix_actions"], max_steps=4,
                    expected_boundaries=item["expected_boundaries"],
                    analysis_version=ANALYSIS_VERSION,
                    original_discovery_source_sha256=plan["original_source_sha256"],
                    input_zip_sha256=plan["input_zip_sha256"],
                    corrected_samples_sha256=plan["corrected_samples_sha256"])
        key = digest(task)[:20]
        if prefix_id in requested:
            records[prefix_id] = run_task(output, task, retry_failed=retry_failed)
        elif (output / "tasks" / f"{key}.json").exists():
            records[prefix_id] = read(output / "tasks" / f"{key}.json")
    prefix_results = []
    for prefix_id in planned:
        record = records.get(prefix_id)
        outcome = (record or {}).get("result") or {}
        matched = 5 if record and record["status"] == "passed" and outcome.get("stop_reason") == "planned_boundary" and outcome.get("boundaries") == 5 else (
            outcome.get("replay_mismatch_detail", {}).get("boundary_step", 0) if record else 0)
        status = "passed" if matched == 5 else "failed" if record else "pending"
        prefix_results.append({"prefix_id": prefix_id, "status": status,
                               "matched_boundaries": matched,
                               "first_mismatch": outcome.get("replay_mismatch_detail"),
                               "stop_reason": outcome.get("stop_reason"),
                               "task_key": record.get("task_key") if record else None,
                               "wall_seconds": record.get("wall_seconds") if record else None})
    passed = {x["prefix_id"] for x in prefix_results if x["status"] == "passed"}
    coverage = [{**item, "verified": item["prefix_id"] in passed}
                for item in plan["selected_samples"]]
    summary = {"analysis_version": ANALYSIS_VERSION,
               "input_zip_sha256": plan["input_zip_sha256"],
               "original_source_sha256": plan["original_source_sha256"],
               "corrected_samples_sha256": plan["corrected_samples_sha256"],
               "replay_source_sha256": config["source_sha256"],
               "planned_prefixes": len(planned), "passed_prefixes": len(passed),
               "total_task_wall_seconds": round(sum(x["wall_seconds"] or 0 for x in prefix_results), 3),
               "planned_samples": len(coverage),
               "covered_samples": sum(x["verified"] for x in coverage),
               "sample_coverage": coverage, "prefix_results": prefix_results,
               "status": "passed" if len(passed) == len(planned) and all(x["verified"] for x in coverage) else
               "failed" if any(x["status"] == "failed" for x in prefix_results) else "incomplete",
               "same_frame_verified": False, "render_sync_quality": "blocked"}
    atomic_json(output / "replay_validation.json", summary)
    (output / "replay_validation.md").write_text(replay_report(summary))
    archive = package_artifacts(output, "replay_validation_bundle.zip",
                                {"analysis_version": ANALYSIS_VERSION,
                                 "input_zip_sha256": plan["input_zip_sha256"],
                                 "corrected_samples_sha256": plan["corrected_samples_sha256"],
                                 "replay_source_sha256": config["source_sha256"],
                                 "same_frame_verified": False, "render_sync_quality": "blocked"})
    return {"status": summary["status"], "passed_prefixes": len(passed),
            "covered_samples": summary["covered_samples"], "output": str(output), "zip": str(archive)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    analyze = sub.add_parser("analyze_bundle", help="Read-only offline ZIP reanalysis; does not launch Isaac")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS)
    replay = sub.add_parser("replay_validate", help="Fresh-process full-prefix replay; no candidate actions")
    replay.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    replay.add_argument("--output", type=Path, default=DEFAULT_REPLAY)
    replay.add_argument("--prefix-ids", help="Comma-separated subset from replay_plan.json; omitted means all four")
    replay.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    config = resolve_config(DEFAULT_CONFIG)
    if args.mode == "analyze_bundle":
        result = analyze_bundle(args.input, args.output, config)
    else:
        ids = args.prefix_ids.split(",") if args.prefix_ids else None
        result = replay_validate(args.analysis_dir, args.output, config, ids, args.retry_failed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.mode == "analyze_bundle" or result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
