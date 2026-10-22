#!/usr/bin/env python3
"""Compare saved 1 FPS, event-refinement and fixed-interval six-episode runs.

Only reads saved artifacts; never calls a model. Example:
  python scripts/report_interval_probe.py --project /path/to/split_cloth
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from report_event_probe import (
    META_FIELDS, TOKEN_FIELDS, directory, finite_number, key, manifest, md_table,
    read_object, rounded, run_path, snapshot as event_snapshot, token_summary,
)


DEFAULT_BASELINE = "qwen3_8_max_paired6_20261022"
DEFAULT_PREVIOUS = "event_probe_max_paired6_20261022"
DEFAULT_CANDIDATE = "interval_nodes_max_paired6_20261022"
DISCLAIMER = (
    "本报告对照同一批六条既有轨迹的三种工程实现。自动accepted、节点一致性和边界变化均不是准确率。"
    "没有独立逐帧人工真值，不计算accuracy，也不能从边界更早或更接近反馈推断结果更正确。"
    "反馈稳定平台是固定候选节点估计，尚不是已校准的首次物理完全松爪帧。"
    "GPT审计属于模型复核，仍非人工金标。"
)


def read_array(path: Path, warnings: list[str], required: bool = False) -> list:
    if not path.is_file():
        if required:
            warnings.append(f"missing: {path.name}")
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is not list:
            raise ValueError("expected an array")
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        warnings.append(f"unreadable {path.name}: {type(exc).__name__}")
        return []


def _free_frame_fields(value: Any, path: str = "model_output") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for name, item in value.items():
            child = f"{path}.{name}"
            if "frame" in str(name).lower() or "timestamp" in str(name).lower():
                errors.append(child)
            errors.extend(_free_frame_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_free_frame_fields(item, f"{path}[{index}]"))
    return errors


def fixed_node_checks(annotation: dict, raw: dict, nodes: list, intervals: list) -> dict:
    """Check the algorithm contract independently of visual correctness."""
    failures = []
    forbidden = _free_frame_fields(raw)
    if forbidden:
        failures.append("model output contains forbidden frame/timestamp fields")
    node_index = {node["node_id"]: node for node in nodes
                  if type(node) is dict and type(node.get("node_id")) is str}
    selected = raw.get("stage_end_nodes", [])
    selected = selected if type(selected) is list else []
    proposed = annotation.get("boundaries", [])
    proposed = proposed if type(proposed) is list else []
    details = []
    frames = []
    for index, item in enumerate(selected):
        if type(item) is not dict:
            failures.append(f"stage_end_nodes[{index}] is not an object")
            continue
        stage, node_id = item.get("stage_id"), item.get("node_id")
        node = node_index.get(node_id) if type(node_id) is str else None
        expected = node.get("source_frame_id") if node else None
        matching = [boundary for boundary in proposed if type(boundary) is dict
                    and boundary.get("stage_id", boundary.get("from_stage")) == stage]
        checks = []
        if node_id is not None and node is None:
            checks.append("selected_node_missing")
        if node is not None:
            frames.append(expected)
            if node.get("eligible_stage_end") is not True:
                checks.append("selected_node_not_eligible")
            participants = item.get("participant_arms", [])
            opened = node.get("open_arms", [])
            if not participants or type(participants) is not list or type(opened) is not list or any(arm not in opened for arm in participants):
                checks.append("participant_not_open")
            if len(matching) != 1:
                checks.append("expected_one_boundary_for_selected_node")
        elif matching:
            checks.append("boundary_without_selected_node")
        for boundary in matching:
            if boundary.get("node_id") != node_id:
                checks.append("boundary_node_id_changed")
            for field in ("end_frame", "coarse_end_frame"):
                if field in boundary and boundary[field] != expected:
                    checks.append(f"{field}_differs_from_fixed_node")
            if boundary.get("next_stage_start_frame") != expected + 1 if type(expected) is int else True:
                checks.append("next_stage_start_is_not_node_plus_one")
        failures.extend(f"stage {stage}: {check}" for check in checks)
        details.append({"stage_id": stage, "node_id": node_id, "fixed_source_frame_id": expected,
                        "participant_arms": item.get("participant_arms"),
                        "evidence_interval_ids": item.get("evidence_interval_ids"),
                        "emitted_boundaries": matching, "failures": checks})
    for boundary in proposed:
        if type(boundary) is not dict:
            failures.append("emitted boundary is not an object")
        elif not any(detail["stage_id"] == boundary.get("stage_id", boundary.get("from_stage")) for detail in details):
            failures.append("emitted boundary has no model-selected stage slot")
    valid_frames = all(type(frame) is int for frame in frames)
    if not valid_frames or any(left >= right for left, right in zip(frames, frames[1:])):
        failures.append("selected node frames are not strictly increasing")
    accepted = annotation.get("status") == "accepted"
    if accepted and (len(selected) != 3 or len(frames) != 3 or annotation.get("boundary_frames") != frames):
        failures.append("accepted boundary_frames must equal the three selected fixed nodes")
    if not accepted and annotation.get("boundary_frames", []) != []:
        failures.append("nonaccepted sample must not emit accepted boundary_frames")
    if annotation.get("training_ready") is not False:
        failures.append("experimental annotation must explicitly set training_ready=false")
    cursor = 0
    for interval in intervals:
        if type(interval) is not dict:
            failures.append("interval is not an object")
            continue
        start, end = interval.get("start_frame"), interval.get("end_frame")
        supplied = interval.get("sample_frame_ids", [])
        if type(start) is not int or type(end) is not int or start != cursor or end < start:
            failures.append("intervals do not form a consecutive closed partition")
        else:
            cursor = end + 1
        if type(supplied) is not list or start not in supplied or end not in supplied:
            failures.append("interval endpoints absent from supplied sample frames")
    if cursor != annotation.get("frame_count"):
        failures.append("intervals do not cover complete source video")
    mapping_tokens = ("boundary_node_id_changed", "differs_from_fixed_node", "next_stage_start_is_not_node_plus_one",
                      "boundary_without_selected_node", "expected_one_boundary_for_selected_node", "emitted boundary has no")
    mapping_passed = not forbidden and not any(any(token in failure for token in mapping_tokens) for failure in failures)
    label_gate_passed = (not failures if accepted else annotation.get("boundary_frames", []) == []) and annotation.get("training_ready") is False
    return {"passed": not failures, "failures": list(dict.fromkeys(failures)),
            "fixed_node_mapping_passed": mapping_passed, "label_gate_passed": label_gate_passed,
            "forbidden_model_frame_fields": forbidden, "stage_details": details,
            "interpretation": "Engineering constraints only; agreement does not prove semantic accuracy or physical full release."}


def snapshot(run: Path, row: dict | None, require_nodes: bool = False) -> dict | None:
    if row is None:
        return None
    result = event_snapshot(run, row)
    warnings = result["warnings"]
    path = directory(run, row)
    annotation = read_object(path / "annotation.json", warnings)
    node_artifact = read_object(path / "gripper_nodes.json", warnings, required=require_nodes)
    nodes = node_artifact.get("nodes", [])
    nodes = nodes if type(nodes) is list else []
    intervals = read_array(path / "intervals.json", warnings, required=require_nodes)
    response = read_object(path / "selection/response.json", warnings)
    raw = read_object(path / "selection/model_output.json", warnings)
    if not raw and type(annotation.get("model_output")) is dict:
        raw = annotation["model_output"]
    if not raw and type(response.get("content")) is str:
        try:
            parsed = json.loads(response["content"])
            raw = parsed if type(parsed) is dict else {}
        except ValueError:
            warnings.append("selection response content is not JSON")
    audit = read_object(path / "gpt_interval_audit.json", warnings)
    result.update({
        "annotation_path": str(path / "annotation.json"),
        "node_count": len(nodes) if node_artifact else None,
        "eligible_node_count": sum(node.get("eligible_stage_end") is True for node in nodes if type(node) is dict) if node_artifact else None,
        "alignment_verified": node_artifact.get("alignment_verified", result.get("alignment_verified")),
        "node_calibration": node_artifact.get("calibration"), "node_diagnostics": node_artifact.get("diagnostics"),
        "gripper_nodes": nodes, "interval_count": len(intervals) if intervals else None, "intervals": intervals,
        "selected_stage_end_nodes": raw.get("stage_end_nodes"),
        "interval_observations": raw.get("interval_observations"),
        "selection_model_output": raw,
        "selection_response": {field: response.get(field) for field in
                               ("model", "request_id", "finish_reason", "image_count", "elapsed_s", "usage")},
        "selection_attempts": read_array(path / "selection/attempts.json", warnings),
        "stage4_completion": annotation.get("stage4_completion"),
        "gpt_interval_audit": audit or None,
        "hard_constraints": fixed_node_checks(annotation, raw, nodes, intervals) if require_nodes else None,
    })
    return result


def _summary(rows: list[dict | None]) -> dict:
    existing = [row for row in rows if row is not None]
    return {"episodes_in_manifest": len(existing), "status_counts": dict(Counter(row["status"] for row in existing)),
            "automatic_accepted_not_accuracy": sum(row["status"] == "accepted" for row in existing),
            "model_ids": sorted({row["model_id"] for row in existing if type(row["model_id"]) is str}),
            "execution_error_count": sum(bool(row["execution_error"]) for row in existing),
            "tokens": token_summary(existing)}


def report(project: Path, baseline_run: Path, previous_run: Path, candidate_run: Path) -> dict:
    manifests = {name: manifest(run) for name, run in
                 (("baseline", baseline_run), ("previous", previous_run), ("candidate", candidate_run))}
    indices = {name: {key(row): row for row in rows} for name, rows in manifests.items()}
    canonical = [key(row) for row in manifests["baseline"]]
    missing = {name: sorted(set(canonical) - set(rows)) for name, rows in indices.items()}
    extra = {name: sorted(set(rows) - set(canonical)) for name, rows in indices.items()}
    keys = canonical + sorted((set(indices["previous"]) | set(indices["candidate"])) - set(canonical))
    episodes = []
    for episode_key in keys:
        source_rows = {name: rows.get(episode_key) for name, rows in indices.items()}
        snapshots = {name: snapshot(run, source_rows[name], require_nodes=name == "candidate")
                     for name, run in (("baseline", baseline_run), ("previous", previous_run), ("candidate", candidate_run))}
        conflicts = {name: [field for field in META_FIELDS if source_rows["baseline"] and source_rows[name]
                            and field in source_rows["baseline"] and field in source_rows[name]
                            and source_rows["baseline"][field] != source_rows[name][field]]
                     for name in ("previous", "candidate")}
        model_ids = {item["model_id"] for item in snapshots.values() if item and type(item["model_id"]) is str}
        episodes.append({"key": episode_key, "metadata_conflicts": conflicts,
                         "model_identity_conflict": len(model_ids) > 1, **snapshots})
    paired = [row for row in episodes if row["key"] in canonical]
    summaries = {name: _summary([row[name] for row in paired]) for name in ("baseline", "previous", "candidate")}
    candidate_tokens = summaries["candidate"]["tokens"]["known_usage"]["total_tokens"]
    ratios = {}
    for name in ("baseline", "previous"):
        denominator = summaries[name]["tokens"]["known_usage"]["total_tokens"]
        ratios[f"candidate_over_{name}"] = rounded(candidate_tokens / denominator) if finite_number(candidate_tokens) and finite_number(denominator) and denominator > 0 else None
    checks = [(row["key"], row["candidate"]["hard_constraints"]) for row in paired if row["candidate"]]
    return {
        "report_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(), "project": str(project),
        "baseline_run_id": baseline_run.name, "previous_run_id": previous_run.name, "candidate_run_id": candidate_run.name,
        "disclaimer": DISCLAIMER, "is_human_ground_truth": False, "acceptance_is_accuracy": False,
        "expected_fixed_episode_count": 6, "baseline_episode_count": len(canonical),
        "fixed_six_sample_contract_passed": len(canonical) == 6 and not any(missing.values()) and not any(extra.values()),
        "missing_keys": missing, "extra_keys": extra, "paired_summaries": summaries,
        "known_total_token_ratios": ratios,
        "token_pairing_complete": all(summary["tokens"]["episodes_with_known_usage"].get("total_tokens", 0) == len(canonical) for summary in summaries.values()),
        "hard_constraints_passed_episodes": sum(check["passed"] for _, check in checks),
        "hard_constraints_checked_episodes": len(checks),
        "fixed_node_mapping_passed_episodes": sum(check["fixed_node_mapping_passed"] for _, check in checks),
        "label_gate_passed_episodes": sum(check["label_gate_passed"] for _, check in checks),
        "hard_constraint_failure_keys": [episode_key for episode_key, check in checks if not check["passed"]],
        "model_identity_conflict_keys": [row["key"] for row in episodes if row["model_identity_conflict"]],
        "accuracy_conclusion": "无独立逐帧人工真值，本报告不计算准确率、不宣称准确率提升。",
        "episodes": episodes,
    }


def markdown(data: dict) -> str:
    names = ("baseline", "previous", "candidate")
    labels = {"baseline": "整段1 FPS", "previous": "事件候选＋连续帧精修", "candidate": "固定节点＋区间1 FPS"}
    summaries = data["paired_summaries"]
    lines = ["# 固定夹爪节点与区间拼图：六条三版本实测", "", data["disclaimer"], "",
             f"固定六条匹配检查：`{data['fixed_six_sample_contract_passed']}`。", ""]
    lines.extend([f"- {labels[name]}：`{data[name + '_run_id']}`" for name in names])
    lines.extend(["", "## 处理状态与已知用量", ""])
    metrics = [
        ["配对manifest条数", *[summaries[name]["episodes_in_manifest"] for name in names]],
        ["模型", *[summaries[name]["model_ids"] for name in names]],
        ["状态计数", *[summaries[name]["status_counts"] for name in names]],
        ["自动接受（非准确率）", *[summaries[name]["automatic_accepted_not_accuracy"] for name in names]],
        ["执行错误", *[summaries[name]["execution_error_count"] for name in names]],
        ["独立保存响应", *[summaries[name]["tokens"]["unique_responses"] for name in names]],
        ["已记录尝试", *[summaries[name]["tokens"]["attempts_recorded"] for name in names]],
    ]
    metrics.extend([[f"已知 {field}", *[summaries[name]["tokens"]["known_usage"][field] for name in names]] for field in TOKEN_FIELDS])
    lines.extend(md_table(["指标", *[labels[name] for name in names]], metrics))
    ratios = data["known_total_token_ratios"]
    lines.extend(["", f"新区间方案 / 整段1 FPS基线：**{ratios['candidate_over_baseline']} 倍**；新区间方案 / 上轮事件精修：**{ratios['candidate_over_previous']} 倍**。", "",
                  "已知token按保存的响应和重试归档去重累计，缺少响应时才回退annotation聚合值。无用量失败请求成本未知。token倍率不是计费价格倍率。",
                  f"六条token配对完整：`{data['token_pairing_complete']}`。", "", "## 六条逐项对照", ""])
    comparison_rows = []
    for row in data["episodes"]:
        candidate = row["candidate"] or {}
        comparison_rows.append([row["key"], *[f"{(row[name] or {}).get('status', 'missing')} / {(row[name] or {}).get('boundary_frames')}" for name in names],
                                f"{candidate.get('node_count')} / {candidate.get('eligible_node_count')} / {candidate.get('interval_count')}",
                                ((candidate.get("usage") or {}).get("known_usage") or {}).get("total_tokens")])
    lines.extend(md_table(["Episode", *[labels[name] + " 状态/边界" for name in names], "节点/eligible/区间", "新方案已知tokens"], comparison_rows))
    lines.extend(["", "## 固定节点及帧归属约束", "",
                  f"原始节点选择回答的资格/帧归属检查：{data['hard_constraints_passed_episodes']}/{data['hard_constraints_checked_episodes']} 条通过。这里的失败包含已被程序拦截的回答矛盾，不等同于程序放行错误。", "",
                  f"固定节点帧号映射：{data['fixed_node_mapping_passed_episodes']}/{data['hard_constraints_checked_episodes']} 条通过；实验标签放行保护：{data['label_gate_passed_episodes']}/{data['hard_constraints_checked_episodes']} 条通过。映射通过不代表选对阶段。", "",
                  "5.2采用闭区间：选中节点422则422仍属于结束阶段，423开始下一阶段。第4阶段完成仅取已提供的区间末帧证据，不称为首次完成帧。", ""])
    selected_rows = []
    for row in data["episodes"]:
        candidate = row["candidate"] or {}
        for item in (candidate.get("hard_constraints") or {}).get("stage_details", []):
            selected_rows.append([row["key"], item["stage_id"], item["node_id"], item["fixed_source_frame_id"],
                                  item["participant_arms"], item["evidence_interval_ids"], item["failures"]])
    lines.extend(md_table(["Episode", "阶段", "选中node_id", "固定原始帧", "参与手", "证据区间", "约束错误"], selected_rows))
    lines.extend(["", "## 拒判、执行错误与节点诊断", ""])
    lines.extend(md_table(["Episode", "新状态", "复核原因", "alignment", "硬约束错误"],
                         [[row["key"], (row["candidate"] or {}).get("status"), (row["candidate"] or {}).get("review_reasons"),
                           (row["candidate"] or {}).get("alignment_verified"), ((row["candidate"] or {}).get("hard_constraints") or {}).get("failures")]
                          for row in data["episodes"]]))
    audits = [(row["key"], name, field, row[name][field]) for row in data["episodes"] for name in names
              if row[name] for field in ("gpt_event_audit", "gpt_interval_audit") if row[name].get(field)]
    if audits:
        lines.extend(["", "## 已保存的后续GPT审计（非人工真值）", "",
                      "审计不覆盖原模型输出，也不据此计算accuracy。完整审计对象保留在JSON内。", ""])
        for episode_key, name, field, audit in audits:
            lines.extend([f"### {episode_key} · {labels[name]}", "",
                          f"产物：`{field}`；状态：`{audit.get('status', 'unspecified')}`；审计者：`{audit.get('reviewer', 'unspecified')}`。", ""])
            findings = audit.get("findings", [])
            findings = findings if isinstance(findings, list) else [findings]
            lines.extend(md_table(["阶段", "类型", "发现", "源帧证据"],
                                 [[item.get("stage_id"), item.get("kind"), item.get("description"), item.get("evidence_source_frames")]
                                  if isinstance(item, dict) else [None, None, item, None] for item in findings]))
            lines.append("")
    warnings = [[row["key"], name, row[name]["warnings"]] for row in data["episodes"] for name in names if row[name] and row[name]["warnings"]]
    conflicts = [row for row in data["episodes"] if row["model_identity_conflict"] or any(row["metadata_conflicts"].values())]
    if warnings or conflicts or any(data["missing_keys"].values()) or any(data["extra_keys"].values()):
        lines.extend(["", "## 输入与产物问题", "",
                      f"缺失key：`{json.dumps(data['missing_keys'], ensure_ascii=False)}`；额外key：`{json.dumps(data['extra_keys'], ensure_ascii=False)}`。", ""])
        if warnings:
            lines.extend(md_table(["Episode", "版本", "问题"], warnings))
        if conflicts:
            lines.extend(["", *md_table(["Episode", "模型身份冲突", "元数据冲突"], [[row["key"], row["model_identity_conflict"], row["metadata_conflicts"]] for row in conflicts])])
    lines.extend(["", data["accuracy_conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--previous", default=DEFAULT_PREVIOUS)
    parser.add_argument("--run", default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    baseline_run, previous_run, candidate_run = [run_path(project, value) for value in (args.baseline, args.previous, args.run)]
    if len({baseline_run, previous_run, candidate_run}) != 3:
        parser.error("baseline, previous and candidate runs must differ")
    data = report(project, baseline_run, previous_run, candidate_run)
    output = candidate_run / "reports"
    output.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output / "interval_probe.json", output / "interval_probe.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(data), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "markdown": str(md_path),
                      "fixed_six_sample_contract_passed": data["fixed_six_sample_contract_passed"],
                      "hard_constraint_failure_keys": data["hard_constraint_failure_keys"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
