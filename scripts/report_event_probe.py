#!/usr/bin/env python3
"""Read saved probe artifacts and write a paired engineering report; no model calls.

Example:
  python scripts/report_event_probe.py --project . \
    --run event_probe_max_paired6_20261022 --baseline qwen3_8_max_paired6_20261022

A GPT montage review is retained only as a coarse model reference. Neither
boundary changes, event agreement nor automatic acceptance is an accuracy metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DISCLAIMER = (
    "这是相同样本的工程可行性实测。自动接受数、与夹爪事件的一致性及边界变化均不等于标注准确率。"
    "已有GPT记录来自1FPS网格，只作粗语义参考，不是逐帧人工真值，不用于证明首次完全松爪帧的准确率。"
    "候选事件来自实测state，仍依赖开合阈值、时间同步、动作语义及视觉可见性。"
    "六条既有样本不是独立随机测试集；结果不能外推至全部数据或另一代机器人。"
)
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
META_FIELDS = ("source_fps", "frame_count", "camera_key", "source_video_path", "source_video_sha256")


def read_object(path: Path, warnings: list[str], required: bool = False) -> dict:
    if not path.is_file():
        if required:
            warnings.append(f"missing: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        warnings.append(f"unreadable {path.name}: {type(exc).__name__}")
        return {}


def key(row: dict) -> str:
    return row.get("key") or f"{row['dataset_id']}__{row['episode_id']}"


def manifest(run: Path) -> list[dict]:
    path = run / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate manifest keys in {path}")
    return rows


def directory(run: Path, row: dict) -> Path:
    return run / "episodes" / row["dataset_id"] / row["episode_id"]


def finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def frame(value: Any) -> bool:
    return type(value) is int and value >= 0


def three_frames(value: Any, frame_count: Any) -> bool:
    return (
        isinstance(value, list) and len(value) == 3 and all(frame(item) for item in value)
        and value[0] < value[1] < value[2]
        and (not frame(frame_count) or value[-1] < frame_count)
    )


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def usage_snapshot(path: Path, annotation: dict, warnings: list[str]) -> dict:
    """Sum actual provider responses once, including archived retry responses.

    The annotation's aggregate usage is a fallback only. Missing usage and failed
    requests remain explicit; an absent token count never establishes zero cost.
    """
    totals = Counter()
    known_fields = Counter()
    seen = set()
    stages: dict[str, dict] = {}
    response_count = 0
    duplicate_count = 0
    without_total = 0
    response_paths = sorted(
        p for p in path.rglob("*.json")
        if p.name == "response.json" or (p.parent.name == "invalid_responses" and p.name.startswith("response_") and not p.name.endswith(".metadata.json"))
    ) if path.is_dir() else []
    for response_path in response_paths:
        response = read_object(response_path, warnings)
        if not response:
            continue
        request_id = response.get("request_id")
        identity = "id:" + request_id if isinstance(request_id, str) and request_id else "sha:" + hashlib.sha256(response_path.read_bytes()).hexdigest()
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        response_count += 1
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        stage_path = response_path.parent.parent if response_path.parent.name == "invalid_responses" else response_path.parent
        stage = str(stage_path.relative_to(path))
        group = stages.setdefault(stage, {"responses": 0, "known_total_tokens": 0, "responses_without_total_tokens": 0})
        group["responses"] += 1
        for field in TOKEN_FIELDS:
            value = usage.get(field)
            if finite_number(value) and value >= 0:
                totals[field] += value
                known_fields[field] += 1
        if finite_number(usage.get("total_tokens")) and usage["total_tokens"] >= 0:
            group["known_total_tokens"] += usage["total_tokens"]
        else:
            group["responses_without_total_tokens"] += 1
            without_total += 1
    source = "provider_response_artifacts"
    if not response_count:
        usage = annotation.get("usage") if isinstance(annotation.get("usage"), dict) else {}
        for field in TOKEN_FIELDS:
            value = usage.get(field)
            if finite_number(value) and value >= 0:
                totals[field] += value
                known_fields[field] += 1
        source = "annotation_aggregate_fallback" if known_fields else "unavailable"
    attempt_count = 0
    attempt_statuses = Counter()
    for attempts_path in sorted(path.rglob("attempts.json")) if path.is_dir() else []:
        try:
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            if not isinstance(attempts, list):
                raise ValueError("expected a list")
            attempt_count += len(attempts)
            attempt_statuses.update(str(item.get("status", "unknown")) for item in attempts if isinstance(item, dict))
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append(f"unreadable attempts.json: {type(exc).__name__}")
    return {
        "source": source, "response_artifact_files": len(response_paths),
        "unique_responses": response_count, "duplicate_response_files_ignored": duplicate_count,
        "responses_without_total_tokens": without_total,
        "known_usage": {field: totals[field] if known_fields[field] else None for field in TOKEN_FIELDS},
        "usage_complete_for_saved_responses": bool(response_count) and without_total == 0,
        "attempts_recorded": attempt_count, "attempt_status_counts": dict(attempt_statuses),
        "stages": stages,
        "cost_note": "仅汇总已保存响应中的已知token；无用量的失败请求成本未知。未进行价格换算。",
    }


def event_frame(event: dict) -> int | None:
    for field in ("source_frame_id", "frame_index", "frame", "source_frame_index", "end_frame", "release_frame", "first_open_frame"):
        if frame(event.get(field)):
            return event[field]
    return None


def event_identity(event: dict) -> Any:
    return event.get("event_id", event.get("id"))


def nearest(events: list[dict], target: Any, fps: Any) -> dict | None:
    if not frame(target):
        return None
    valid = [event for event in events if event_frame(event) is not None]
    if not valid:
        return None
    chosen = min(valid, key=lambda event: abs(event_frame(event) - target))
    delta = event_frame(chosen) - target
    return {"event_id": event_identity(chosen), "event_frame": event_frame(chosen),
            "event_minus_target_frames": delta,
            "event_minus_target_seconds": rounded(delta / fps) if finite_number(fps) and fps > 0 else None}


def snapshot(run: Path, row: dict) -> dict:
    warnings: list[str] = []
    path = directory(run, row)
    annotation = read_object(path / "annotation.json", warnings, required=True)
    candidates = read_object(path / "event_candidates.json", warnings)
    refinement = read_object(path / "refinement.json", warnings)
    review = read_object(path / "gpt_review.json", warnings)
    event_audit = read_object(path / "gpt_event_audit.json", warnings)
    events = candidates.get("events")
    events = [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    boundaries = refinement.get("boundaries")
    boundaries = [item for item in boundaries if isinstance(item, dict)] if isinstance(boundaries, list) else []
    return {
        "status": annotation.get("status", "missing"), "execution_error": annotation.get("execution_error"),
        "model_id": annotation.get("model_id"), "boundary_frames": annotation.get("boundary_frames"),
        "boundary_frames_valid": three_frames(annotation.get("boundary_frames"), row.get("frame_count")),
        "stage4_completion_frame": annotation.get("stage4_completion_frame"),
        "review_reasons": annotation.get("review_reasons", []),
        "candidate_count": len(events), "candidate_artifact_present": bool(candidates),
        "candidate_status": candidates.get("status"), "candidate_calibration": candidates.get("calibration"),
        "alignment_verified": candidates.get("alignment_verified", (candidates.get("calibration") or {}).get("alignment_verified")),
        "candidate_diagnostics": candidates.get("diagnostics", candidates.get("diagnostic")),
        "candidate_events": events, "refinement_status": refinement.get("status"),
        "refinement_boundaries": boundaries,
        "gpt_coarse_reference": {"status": review.get("status"), "applicable": review.get("applicable"),
            "boundary_frames": review.get("boundary_frames"), "is_frame_ground_truth": False} if review else None,
        "gpt_event_audit": event_audit or None,
        "usage": usage_snapshot(path, annotation, warnings), "warnings": warnings,
    }


def token_summary(rows: list[dict]) -> dict:
    result = {field: 0 for field in TOKEN_FIELDS}
    known_episodes = Counter()
    for row in rows:
        for field in TOKEN_FIELDS:
            value = row["usage"]["known_usage"][field]
            if value is not None:
                result[field] += value
                known_episodes[field] += 1
    return {
        "known_usage": {field: result[field] if known_episodes[field] else None for field in TOKEN_FIELDS},
        "episodes_with_known_usage": dict(known_episodes),
        "unique_responses": sum(row["usage"]["unique_responses"] for row in rows),
        "attempts_recorded": sum(row["usage"]["attempts_recorded"] for row in rows),
        "responses_without_total_tokens": sum(row["usage"]["responses_without_total_tokens"] for row in rows),
    }


def summary(rows: list[dict]) -> dict:
    return {"episodes": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
        "accepted": sum(row["status"] == "accepted" for row in rows),
        "model_ids": sorted({row["model_id"] for row in rows if isinstance(row["model_id"], str)}),
        "execution_errors": sum(bool(row["execution_error"]) for row in rows),
        "episodes_with_warnings": sum(bool(row["warnings"]) for row in rows), "tokens": token_summary(rows)}


def report(project: Path, candidate_run: Path, baseline_run: Path) -> dict:
    candidate_rows, baseline_rows = manifest(candidate_run), manifest(baseline_run)
    baseline_index = {key(row): row for row in baseline_rows}
    rows, baseline_paired, candidate_paired = [], [], []
    for row in candidate_rows:
        row_key = key(row)
        baseline_manifest = baseline_index.get(row_key)
        baseline = snapshot(baseline_run, baseline_manifest) if baseline_manifest else None
        candidate = snapshot(candidate_run, row)
        model_conflict = bool(baseline and baseline["model_id"] and candidate["model_id"] and baseline["model_id"] != candidate["model_id"])
        conflicts = [field for field in META_FIELDS if baseline_manifest and field in row and field in baseline_manifest and row[field] != baseline_manifest[field]]
        fps = row.get("source_fps")
        both_valid = bool(baseline and baseline["boundary_frames_valid"] and candidate["boundary_frames_valid"] and not conflicts and not model_conflict)
        deltas = [after - before for before, after in zip(baseline["boundary_frames"], candidate["boundary_frames"])] if both_valid else None
        refinement_by_stage = {item.get("from_stage"): item for item in candidate["refinement_boundaries"]}
        details = []
        for stage in (1, 2, 3):
            before = baseline["boundary_frames"][stage - 1] if baseline and baseline["boundary_frames_valid"] else None
            after = candidate["boundary_frames"][stage - 1] if candidate["boundary_frames_valid"] else None
            local = refinement_by_stage.get(stage, {})
            selected = local.get("selected_event_ids", [])
            selected = selected if isinstance(selected, list) else [selected]
            details.append({"from_stage": stage, "baseline_frame": before, "candidate_frame": after,
                "candidate_proposed_frame": local.get("end_frame"),
                "candidate_minus_baseline_frames": deltas[stage - 1] if deltas else None,
                "nearest_raw_event_to_baseline": nearest(candidate["candidate_events"], before, fps),
                "nearest_raw_event_to_candidate": nearest(candidate["candidate_events"], after, fps),
                "refinement": local,
                "selected_events": [event for event in candidate["candidate_events"] if event_identity(event) in selected]})
        if baseline:
            baseline_paired.append(baseline)
            candidate_paired.append(candidate)
        rows.append({"key": row_key, "matched": baseline is not None, "source_fps": fps,
            "metadata_conflicts": conflicts, "model_identity_conflict": model_conflict, "baseline": baseline, "candidate": candidate,
            "both_accepted_with_valid_boundaries": bool(both_valid and baseline["status"] == candidate["status"] == "accepted"),
            "candidate_minus_baseline_frames": deltas,
            "candidate_minus_baseline_seconds": [rounded(delta / fps) for delta in deltas] if deltas and finite_number(fps) and fps > 0 else None,
            "boundaries": details})
    comparable = [row for row in rows if row["both_accepted_with_valid_boundaries"]]
    boundary_details = [detail for row in rows for detail in row["boundaries"]]
    baseline_summary, candidate_summary = summary(baseline_paired), summary(candidate_paired)
    baseline_tokens = baseline_summary["tokens"]["known_usage"]["total_tokens"]
    candidate_tokens = candidate_summary["tokens"]["known_usage"]["total_tokens"]
    audit_rows = [row for row in rows if row["candidate"]["gpt_event_audit"]]
    rejected_audits = [row for row in audit_rows if row["candidate"]["gpt_event_audit"].get("status") == "reject_for_training"]
    same_accepted_deltas = [value for row in comparable for value in row["candidate_minus_baseline_seconds"] or []]
    return {
        "report_version": 2, "updated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": baseline_run.name, "candidate_run_id": candidate_run.name,
        "project": str(project), "disclaimer": DISCLAIMER,
        "is_human_ground_truth": False, "acceptance_is_accuracy": False,
        "comparison_kind": "independent_pipeline_with_fixed_model_and_samples",
        "manifest_candidate_episodes": len(candidate_rows), "manifest_baseline_episodes": len(baseline_rows),
        "matched_episodes": len(baseline_paired),
        "model_identity_conflict_keys": [row["key"] for row in rows if row["model_identity_conflict"]],
        "missing_baseline_keys": [row["key"] for row in rows if not row["matched"]],
        "baseline_paired": baseline_summary, "candidate_paired": candidate_summary,
        "known_total_token_ratio_candidate_over_baseline": rounded(candidate_tokens / baseline_tokens)
            if finite_number(baseline_tokens) and baseline_tokens > 0 and finite_number(candidate_tokens) else None,
        "newly_accepted_keys": [row["key"] for row in rows if row["baseline"]
            and row["baseline"]["status"] != "accepted" and row["candidate"]["status"] == "accepted"],
        "no_longer_accepted_keys": [row["key"] for row in rows if row["baseline"]
            and row["baseline"]["status"] == "accepted" and row["candidate"]["status"] != "accepted"],
        "both_accepted_boundary_change_seconds_range": [min(same_accepted_deltas), max(same_accepted_deltas)]
            if same_accepted_deltas else None,
        "gpt_event_audited_episodes": len(audit_rows),
        "gpt_event_audit_reject_for_training_episodes": len(rejected_audits),
        "gpt_event_audit_status_counts": dict(Counter(
            str(row["candidate"]["gpt_event_audit"].get("status", "unspecified")) for row in audit_rows)),
        "both_accepted_boundary_change_episodes": len(comparable),
        "both_accepted_changed_boundaries": sum(delta != 0 for row in comparable for delta in row["candidate_minus_baseline_frames"]),
        "both_accepted_compared_boundaries": len(comparable) * 3,
        "candidate_boundaries_with_local_artifacts": sum(bool(item["refinement"]) for item in boundary_details),
        "candidate_visual_confirmed_boundaries": sum(item["refinement"].get("visual_confirmed") is True for item in boundary_details),
        "accuracy_conclusion": "未建立独立逐帧人工真值，无法计算精确边界准确率或确认准确率提升。",
        "training_assessment": "本轮复核发现原模型自报确认无法排除的错误，当前实现不适合直接批量生成训练标注。自动accepted不代表成功，需先修正阶段语义选择与证据一致性。"
            if rejected_audits else "当前仅为工程实测，尚无独立逐帧人工真值支持直接批量生成训练标注。",
        "episodes": rows,
    }


def cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *[
        "| " + " | ".join(cell(item) for item in row) + " |" for row in rows]]


def markdown(data: dict) -> str:
    baseline, candidate = data["baseline_paired"], data["candidate_paired"]
    lines = [f"# 夹爪事件候选与连续帧实测：{data['candidate_run_id']}", "", DISCLAIMER, "",
        f"对照：`{data['baseline_run_id']}` → `{data['candidate_run_id']}`；按key匹配 {data['matched_episodes']}/{data['manifest_candidate_episodes']} 条。", "",
        "## 同一批样本的处理与用量", ""]
    table = [["模型身份", baseline["model_ids"], candidate["model_ids"]], ["轨迹数", baseline["episodes"], candidate["episodes"]],
        ["自动接受（不是准确率）", baseline["accepted"], candidate["accepted"]],
        ["状态计数", baseline["status_counts"], candidate["status_counts"]],
        ["执行错误", baseline["execution_errors"], candidate["execution_errors"]],
        ["已保存的独立响应", baseline["tokens"]["unique_responses"], candidate["tokens"]["unique_responses"]],
        ["已记录尝试", baseline["tokens"]["attempts_recorded"], candidate["tokens"]["attempts_recorded"]]]
    for field in TOKEN_FIELDS:
        table.append([f"已知 {field}", baseline["tokens"]["known_usage"][field], candidate["tokens"]["known_usage"][field]])
    table.append(["有已知total_tokens的轨迹", baseline["tokens"]["episodes_with_known_usage"].get("total_tokens", 0), candidate["tokens"]["episodes_with_known_usage"].get("total_tokens", 0)])
    if data["model_identity_conflict_keys"]:
        lines.extend(["模型身份不同，不能将边界变化单独归因于事件/精修方法：" + ", ".join(data["model_identity_conflict_keys"]), ""])
    lines.extend(md_table(["指标", "1FPS基线", "事件候选＋连续帧"], table))
    lines.extend(["", "用量按实际response及重试归档去重累计；annotation聚合用量仅在无响应产物时回退使用。失败请求无用量时成本未知。新流程独立重选阶段并进行精修，其总用量含所有已保存selection/refinement响应。", "",
        f"相同配对样本的已知total_tokens倍率：{data['known_total_token_ratio_candidate_over_baseline']}。计费金额还受缓存和输入/输出单价影响，token倍率不是价格倍率。", "",
        "## 本轮工程发现", "",
        f"新增自动接受：{', '.join(data['newly_accepted_keys']) or '无'}；从接受转为非接受：{', '.join(data['no_longer_accepted_keys']) or '无'}。这只说明状态变化，不能认定前者修正或后者退化。", "",
        f"双方accepted交集中新−旧边界时间差范围：{data['both_accepted_boundary_change_seconds_range']} 秒。大幅提前应检查动作语义，不能直接解释为修正了1FPS量化。", ""])
    execution_rows = [row for row in data["episodes"] if row["candidate"]["execution_error"]]
    if execution_rows:
        lines.extend(md_table(["执行错误Episode", "类型", "原因"], [[row["key"], row["candidate"]["execution_error"],
            row["candidate"]["review_reasons"]] for row in execution_rows]))
        lines.extend(["", "接口图片审核等执行错误与模型语义拒判分开统计；原始失败记录保留，没有把未完成请求计作正确拒判。", ""])
    lines.extend(["## 全部样本逐条对照", ""])
    lines.extend(md_table(["Episode", "基线状态/边界", "新状态/边界", "新−旧（帧）", "候选数", "新已知total_tokens"], [
        [row["key"], f"{(row['baseline'] or {}).get('status', 'missing')} / {(row['baseline'] or {}).get('boundary_frames')}",
         f"{row['candidate']['status']} / {row['candidate']['boundary_frames']}", row["candidate_minus_baseline_frames"],
         row["candidate"]["candidate_count"], row["candidate"]["usage"]["known_usage"]["total_tokens"]] for row in data["episodes"]]))
    lines.extend(["", "## 事件时间对齐与校准", ""])
    lines.extend(md_table(["Episode", "候选数", "alignment_verified", "诊断"], [[row["key"],
        row["candidate"]["candidate_count"], row["candidate"]["alignment_verified"], row["candidate"]["candidate_diagnostics"]] for row in data["episodes"]]))
    lines.extend(["", "阈值与校准完整记录保存在JSON。alignment_verified为false时，夹爪候选不能作为可靠的原始视频帧锚点。", "", f"双方accepted且元数据一致的边界变化覆盖 {data['both_accepted_boundary_change_episodes']} 条、{data['both_accepted_compared_boundaries']} 个边界，其中 {data['both_accepted_changed_boundaries']} 个改变。变化幅度不代表误差减少。", "",
        "## 连续帧与候选证据", "",
        f"留存局部精修记录 {data['candidate_boundaries_with_local_artifacts']} 个；模型自报视觉确认 {data['candidate_visual_confirmed_boundaries']} 个。这些计数不构成人工验证。", ""])
    evidence_rows = []
    for row in data["episodes"]:
        for detail in row["boundaries"]:
            local = detail["refinement"]
            if not local:
                continue
            near = detail["nearest_raw_event_to_baseline"]
            evidence_rows.append([row["key"], detail["from_stage"], detail["baseline_frame"], detail["candidate_proposed_frame"],
                local.get("selected_event_ids"), local.get("visual_confirmed"),
                f"{near['event_id']} ({near['event_minus_target_frames']:+d})" if near else None,
                local.get("evidence_text", local.get("review_reasons"))])
    lines.extend(md_table(["Episode", "阶段", "旧帧", "新提议帧（含待复核）", "选中event", "模型自报视觉确认", "距旧帧最近event（帧差）", "证据/复核原因"], evidence_rows))
    lines.extend(["", "最近候选仅是诊断信息，没有用它回填结果；最近松爪可能属于调整、重抓或下一阶段。先判定阶段语义，再查看参与夹爪的连续帧。", "",
        "## 已有GPT粗参考", "", "下表仅保留既有1FPS视觉记录，不将新原始帧与粗采样帧的距离称为逐帧误差或准确率。", ""])
    lines.extend(md_table(["Episode", "参考状态", "GPT粗候选帧"], [[row["key"],
        ((row["baseline"] or {}).get("gpt_coarse_reference") or {}).get("status"),
        ((row["baseline"] or {}).get("gpt_coarse_reference") or {}).get("boundary_frames")] for row in data["episodes"]]))
    audit_rows = [row for row in data["episodes"] if row["candidate"]["gpt_event_audit"]]
    if audit_rows:
        lines.extend(["", "## 本轮GPT产物复核（非人工金标）", "",
            "下列复核独立于原模型自报字段保存，保留原始模型输出。本次复核已知模型结果，不是盲审。GPT复核仍不是人工逐帧真值，不计算准确率。完整审计对象见本报告JSON。", ""])
        for row in audit_rows:
            audit = row["candidate"]["gpt_event_audit"]
            lines.extend([f"### {row['key']}", ""])
            lines.extend([f"复核者：{audit.get('reviewer', 'unspecified')}；范围：{audit.get('review_scope', 'unspecified')}；状态：{audit.get('status', 'unspecified')}。", ""])
            findings = audit.get("findings", [])
            findings = findings if isinstance(findings, list) else [findings]
            finding_rows = [[item.get("stage_id"), item.get("kind"), item.get("description"), item.get("evidence_source_frames")]
                if isinstance(item, dict) else [None, None, item, None] for item in findings]
            lines.extend(md_table(["阶段", "问题类型", "复核发现", "源帧证据"], finding_rows))
            lines.append("")
    warnings = [(row["key"], row["metadata_conflicts"], (row["baseline"] or {}).get("warnings", []), row["candidate"]["warnings"]) for row in data["episodes"]
        if row["metadata_conflicts"] or (row["baseline"] or {}).get("warnings") or row["candidate"]["warnings"]]
    if warnings:
        lines.extend(["", "## 输入与产物问题", ""])
        lines.extend(md_table(["Episode", "元数据冲突", "基线问题", "新流程问题"], [list(item) for item in warnings]))
    lines.extend(["", data["training_assessment"], "", data["accuracy_conclusion"], ""])
    return "\n".join(lines)


def run_path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / "runs" / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    candidate_run, baseline_run = run_path(project, args.run), run_path(project, args.baseline)
    if candidate_run == baseline_run:
        parser.error("--run and --baseline must differ")
    data = report(project, candidate_run, baseline_run)
    output = candidate_run / "reports"
    output.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output / "event_probe.json", output / "event_probe.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(data), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "markdown": str(markdown_path), "matched": data["matched_episodes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
