#!/usr/bin/env python3
"""Read saved text-criteria/any-open artifacts and compare a fixed six-run pair.

Works during preparation before response.json or annotation.json exists. No
model calls; only writes reports inside the new candidate run. Example:
  python3 scripts/report_text_criteria_probe.py --project /path/to/split_cloth
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

from report_event_probe import (
    META_FIELDS, TOKEN_FIELDS, directory, finite_number, key, manifest, md_table,
    read_object, rounded, run_path, token_summary,
)
from report_interval_probe import fixed_node_checks, read_array, snapshot as interval_snapshot


DEFAULT_PREVIOUS = "interval_nodes_max_paired6_20261022"
DEFAULT_CANDIDATE = "text_criteria_anyopen_max_paired6_20261022"
DISCLAIMER = (
    "同六条既有轨迹、同一模型的工程对照。本轮同时改变三项：参考图片改为GPT离线文字判据、"
    "开闭节点改为任一手opening-only节点、启用全区间时序一致性校验。无法将结果变化归因于其中某一项。"
    "自动状态、节点映射一致性、时序校验和GPT复核均不等于准确率；没有独立逐帧人工真值，不宣称准确率提升。"
    "节点仍是实测反馈平台估计，不是首次物理完全松爪帧的人工金标。"
)


def _manifest(run: Path, warnings: list[str]) -> list[dict]:
    if not (run / "manifest.jsonl").is_file():
        warnings.append(f"manifest not yet available: {run.name}")
        return []
    try:
        return manifest(run)
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        warnings.append(f"unreadable manifest {run.name}: {type(exc).__name__}")
        return []


def _labels(path: Path) -> dict:
    """Inspect this pipeline's 1-D int64 .npy without a NumPy dependency."""
    if not path.is_file():
        return {"present": False, "all_minus_one": None, "non_unknown_frames": None}
    try:
        with path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise ValueError("invalid NPY magic")
            version = tuple(stream.read(2))
            size = 2 if version == (1, 0) else 4 if version in {(2, 0), (3, 0)} else 0
            if not size:
                raise ValueError("unsupported NPY version")
            header_size_bytes = stream.read(size)
            if len(header_size_bytes) != size:
                raise ValueError("truncated NPY header")
            header_size = int.from_bytes(header_size_bytes, "little")
            if header_size > 65536:
                raise ValueError("unexpectedly large NPY header")
            header = ast.literal_eval(stream.read(header_size).decode("utf-8" if version == (3, 0) else "latin1"))
            shape, dtype = header.get("shape"), header.get("descr")
            if type(shape) is not tuple or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 0:
                raise ValueError("expected one-dimensional labels")
            if dtype not in {"<i8", ">i8", "=i8"} or header.get("fortran_order") is not False:
                raise ValueError("expected int64 C-order labels")
            data = stream.read()
            if len(data) != shape[0] * 8:
                raise ValueError("label byte count does not match shape")
            values = Counter(value[0] for value in struct.iter_unpack(dtype[0] + "q", data))
            return {"present": True, "frame_count": shape[0], "dtype": dtype,
                    "value_counts": dict(values), "all_minus_one": values.get(-1, 0) == shape[0],
                    "non_unknown_frames": shape[0] - values.get(-1, 0)}
    except (OSError, ValueError, TypeError, SyntaxError, AttributeError, struct.error) as exc:
        return {"present": True, "all_minus_one": None, "non_unknown_frames": None,
                "inspection_error": type(exc).__name__ + ": " + str(exc)[:160]}


def _observation_order(raw: dict, intervals: list[dict]) -> dict:
    expected = [item.get("interval_id") for item in intervals if type(item) is dict]
    observations = raw.get("interval_observations")
    if type(observations) is not list:
        return {"checked": False, "complete": None, "chronological": None,
                "expected_count": len(expected), "observed_count": None}
    actual = [item.get("interval_id") if type(item) is dict else None for item in observations]
    valid_ids = all(type(item) is str for item in actual)
    complete = valid_ids and len(actual) == len(expected) and len(set(actual)) == len(actual) and set(actual) == set(expected)
    return {"checked": True, "complete": complete, "chronological": actual == expected,
            "expected_count": len(expected), "observed_count": len(actual),
            "expected_interval_ids": expected, "observed_interval_ids": actual}


def _reference_inventory(path: Path, payload: dict, request: dict, warnings: list[str]) -> dict:
    pages = read_array(path / "interval_pages.json", warnings)
    target_paths = [str(Path(page["image_path"]) if Path(page["image_path"]).is_absolute() else path / page["image_path"])
                    for page in pages if type(page) is dict and type(page.get("image_path")) is str]
    files = payload.get("image_files")
    files = files if type(files) is list else []
    sent_paths = [str(Path(item["path"]) if Path(item["path"]).is_absolute() else path / item["path"])
                  for item in files if type(item) is dict and type(item.get("path")) is str]
    images = 0
    for message in request.get("messages", []):
        if type(message) is dict and type(message.get("content")) is list:
            images += sum(type(part) is dict and part.get("type") == "image_url" for part in message["content"])
    checked = bool(payload) and bool(request) and bool(pages)
    extra_paths = list((Counter(sent_paths) - Counter(target_paths)).elements())
    target_only = Counter(sent_paths) == Counter(target_paths) if checked else None
    declared_count = payload.get("reference_image_count")
    verified_zero = (declared_count == 0 and target_only and images == len(target_paths)
                     and request.get("image_count") == images) if checked else None
    return {"checked": checked, "reference_mode": payload.get("reference_mode"),
            "declared_reference_image_count": declared_count,
            "reference_images_zero_verified": bool(verified_zero) if checked else None,
            "request_image_count": images if request else None, "target_interval_page_count": len(pages) if pages else None,
            "inventory_contains_only_target_pages": target_only, "non_target_image_paths": extra_paths,
            "text_criteria_sha256": payload.get("text_criteria_sha256")}


def snapshot(run: Path, row: dict | None, run_metadata: dict) -> dict | None:
    if row is None:
        return None
    result = interval_snapshot(run, row, require_nodes=False)
    warnings = result["warnings"]
    path = directory(run, row)
    annotation = read_object(path / "annotation.json", warnings)
    payload = read_object(path / "payload_inventory.json", warnings)
    request = read_object(path / "selection/request.json", warnings)
    raw = result["selection_model_output"]
    response = result["selection_response"]
    has_response = (path / "selection/response.json").is_file()
    if annotation:
        phase, status = "annotated", annotation.get("status", "unknown")
    elif has_response:
        phase, status = "response_saved_annotation_pending", "annotation_pending"
    elif payload and request:
        phase, status = "prepared_for_selection", "prepared_for_selection"
    else:
        phase, status = "preparation_incomplete", "preparation_incomplete"
    if not annotation:
        # A missing annotation is expected during prepare-only, not a failed run.
        warnings[:] = [warning for warning in warnings if warning != "missing: annotation.json"]
    config = run_metadata.get("config", {})
    config = config if type(config) is dict else {}
    provider = config.get("provider", {})
    provider = provider if type(provider) is dict else {}
    model_id = annotation.get("model_id") or response.get("model") or provider.get("model_id")
    nodes, intervals = result["gripper_nodes"], result["intervals"]
    changes = [change for node in nodes if type(node) is dict for change in node.get("changes", []) if type(change) is dict]
    node_artifact_present = (path / "gripper_nodes.json").is_file()
    every_node_has_changes = all(type(node) is dict and type(node.get("changes")) is list and bool(node["changes"])
                                 and all(type(change) is dict for change in node["changes"]) for node in nodes)
    only_open = every_node_has_changes and all(change.get("kind") == "open" for change in changes)
    node_config = config.get("nodes", {})
    node_config = node_config if type(node_config) is dict else {}
    temporal = annotation.get("temporal_validation", {})
    temporal = temporal if type(temporal) is dict else {}
    stage_slots = raw.get("stage_end_nodes")
    mapping = fixed_node_checks(annotation, raw, nodes, intervals) if annotation and type(stage_slots) is list else None
    node_index = {node["node_id"]: node for node in nodes if type(node) is dict and type(node.get("node_id")) is str}
    raw_proposals = []
    for item in stage_slots if type(stage_slots) is list else []:
        if type(item) is not dict:
            continue
        node_id = item.get("node_id")
        node = node_index.get(node_id) if type(node_id) is str else None
        checked = next((detail for detail in mapping["stage_details"] if detail["stage_id"] == item.get("stage_id")), None) if mapping else None
        raw_proposals.append({"stage_id": item.get("stage_id"), "node_id": node_id,
                              "fixed_source_frame_id": node.get("source_frame_id") if node else None,
                              "participant_arms": item.get("participant_arms"),
                              "evidence_interval_ids": item.get("evidence_interval_ids"),
                              "evidence_text": item.get("evidence_text"),
                              "mapping_checked_against_annotation": checked is not None,
                              "failures": checked["failures"] if checked else []})
    result.update({
        "status": status, "artifact_phase": phase, "model_id": model_id,
        "annotation_present": bool(annotation), "response_present": has_response,
        "request_present": bool(request), "request_fingerprint": request.get("input_fingerprint"),
        "reference_inventory": _reference_inventory(path, payload, request, warnings),
        "node_policy": payload.get("node_policy", node_config.get("node_policy", "all_transitions")),
        "opening_changes_only_verified": only_open if node_artifact_present else None,
        "change_kind_counts": dict(Counter(str(change.get("kind", "unknown")) for change in changes)),
        "single_arm_open_node_count": sum(len(node.get("open_arms", [])) == 1 for node in nodes if type(node) is dict),
        "all_nodes_eligible": all(node.get("eligible_stage_end") is True for node in nodes if type(node) is dict) if node_artifact_present else None,
        "observation_order": _observation_order(raw, intervals),
        "temporal_validation": temporal or None,
        "temporal_summary": {"enabled": temporal.get("enabled", config.get("temporal_order", False)),
                             "status": temporal.get("status", "not_recorded"),
                             "findings": temporal.get("findings", []),
                             "deferred_stage_ids": temporal.get("deferred_stage_ids", [])},
        "hard_constraints": mapping,
        "raw_node_proposals": raw_proposals,
        "training_ready": annotation.get("training_ready") if annotation else None,
        "saved_labels": _labels(path / "stage.npy"),
        "gpt_text_criteria_audit": read_object(path / "gpt_text_criteria_audit.json", warnings) or None,
    })
    return result


def _summary(rows: list[dict | None]) -> dict:
    rows = [row for row in rows if row is not None]
    annotated = [row for row in rows if row["annotation_present"]]
    mapping = [row["hard_constraints"] for row in rows if row["hard_constraints"] is not None]
    ordering = [row["observation_order"] for row in rows if row["observation_order"]["checked"]]
    references = [row["reference_inventory"] for row in rows if row["reference_inventory"]["checked"]]
    return {
        "episodes_in_manifest": len(rows), "annotated_episodes": len(annotated),
        "response_present_episodes": sum(row["response_present"] for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "artifact_phase_counts": dict(Counter(row["artifact_phase"] for row in rows)),
        "model_ids": sorted({row["model_id"] for row in rows if type(row["model_id"]) is str}),
        "execution_errors": sum(bool(row["execution_error"]) for row in rows),
        "tokens": token_summary(rows),
        "fixed_node_mapping_checked_episodes": len(mapping),
        "fixed_node_mapping_passed_episodes": sum(item["fixed_node_mapping_passed"] for item in mapping),
        "observation_order_checked_episodes": len(ordering),
        "observations_complete_and_ordered_episodes": sum(item["complete"] and item["chronological"] for item in ordering),
        "reference_inventory_checked_episodes": len(references),
        "zero_reference_images_verified_episodes": sum(item["reference_images_zero_verified"] is True for item in references),
        "opening_changes_only_verified_episodes": sum(row["opening_changes_only_verified"] is True for row in rows),
        "temporal_status_counts": dict(Counter(row["temporal_summary"]["status"] for row in rows)),
        "training_ready_false_episodes": sum(row["training_ready"] is False for row in annotated),
        "training_ready_true_episodes": sum(row["training_ready"] is True for row in annotated),
        "saved_label_arrays": sum(row["saved_labels"]["present"] for row in rows),
        "saved_all_minus_one_label_arrays": sum(row["saved_labels"]["all_minus_one"] is True for row in rows),
        "gpt_visual_audit_status_counts": dict(Counter(str(audit.get("status", "unspecified")) for row in rows
            if (audit := row.get("gpt_text_criteria_audit") or row.get("gpt_interval_audit")))),
    }


def report(project: Path, previous_run: Path, candidate_run: Path) -> dict:
    warnings = []
    previous_manifest, candidate_manifest = _manifest(previous_run, warnings), _manifest(candidate_run, warnings)
    previous_index = {key(row): row for row in previous_manifest}
    candidate_index = {key(row): row for row in candidate_manifest}
    metadata = {"previous": read_object(previous_run / "run.json", warnings),
                "candidate": read_object(candidate_run / "run.json", warnings)}
    canonical = list(previous_index)
    keys = canonical + sorted(set(candidate_index) - set(previous_index))
    episodes = []
    for episode_key in keys:
        previous_row, candidate_row = previous_index.get(episode_key), candidate_index.get(episode_key)
        previous = snapshot(previous_run, previous_row, metadata["previous"])
        candidate = snapshot(candidate_run, candidate_row, metadata["candidate"])
        conflicts = [field for field in META_FIELDS if previous_row and candidate_row
                     and field in previous_row and field in candidate_row and previous_row[field] != candidate_row[field]]
        model_conflict = bool(previous and candidate and previous["model_id"] and candidate["model_id"]
                              and previous["model_id"] != candidate["model_id"])
        episodes.append({"key": episode_key, "metadata_conflicts": conflicts, "model_identity_conflict": model_conflict,
                         "previous": previous, "candidate": candidate})
    paired = [row for row in episodes if row["key"] in previous_index]
    summaries = {name: _summary([row[name] for row in paired]) for name in ("previous", "candidate")}
    before = summaries["previous"]["tokens"]["known_usage"]["total_tokens"]
    after = summaries["candidate"]["tokens"]["known_usage"]["total_tokens"]
    missing, extra = sorted(set(previous_index) - set(candidate_index)), sorted(set(candidate_index) - set(previous_index))
    return {
        "report_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(), "project": str(project),
        "previous_run_id": previous_run.name, "candidate_run_id": candidate_run.name,
        "disclaimer": DISCLAIMER, "is_human_ground_truth": False, "acceptance_is_accuracy": False,
        "changed_factors": ["reference_images_to_gpt_text_criteria", "all_transitions_to_any_arm_opening_nodes", "enable_full_interval_temporal_validation"],
        "causal_attribution_supported": False,
        "fixed_six_sample_contract_passed": len(canonical) == 6 and not missing and not extra,
        "missing_candidate_keys": missing, "extra_candidate_keys": extra,
        "model_identity_conflict_keys": [row["key"] for row in episodes if row["model_identity_conflict"]],
        "metadata_conflict_keys": [row["key"] for row in episodes if row["metadata_conflicts"]],
        "paired_summaries": summaries,
        "known_total_token_ratio_candidate_over_previous": rounded(after / before) if finite_number(after) and finite_number(before) and before > 0 else None,
        "token_pairing_complete": all(summary["tokens"]["episodes_with_known_usage"].get("total_tokens", 0) == len(canonical)
                                      for summary in summaries.values()) and bool(canonical),
        "candidate_all_six_annotated_and_training_not_released": summaries["candidate"]["annotated_episodes"] == 6
            and summaries["candidate"]["training_ready_false_episodes"] == 6,
        "run_contracts": {name: {field: value.get(field) for field in
                                ("reference_mode", "stage_criteria_sha256", "spec_version", "algorithm_hash")}
                          for name, value in metadata.items()},
        "warnings": warnings, "episodes": episodes,
        "accuracy_conclusion": "未建立独立逐帧人工真值，不能计算或宣称标注准确率提升。",
    }


def markdown(data: dict) -> str:
    names = ("previous", "candidate")
    summaries = data["paired_summaries"]
    lines = ["# 文字阶段判据＋任一手张开节点：六条工程对照", "", data["disclaimer"], "",
             f"上一版：`{data['previous_run_id']}`；本轮：`{data['candidate_run_id']}`。", "",
             f"固定六条匹配：`{data['fixed_six_sample_contract_passed']}`；六条已完成且training_ready=false：`{data['candidate_all_six_annotated_and_training_not_released']}`。", "",
             "## 请求、状态与已知用量", ""]
    metrics = [["manifest条数", *[summaries[name]["episodes_in_manifest"] for name in names]],
               ["已生成annotation", *[summaries[name]["annotated_episodes"] for name in names]],
               ["模型身份", *[summaries[name]["model_ids"] for name in names]],
               ["状态（不是准确率）", *[summaries[name]["status_counts"] for name in names]],
               ["阶段产物状态", *[summaries[name]["artifact_phase_counts"] for name in names]],
               ["执行错误", *[summaries[name]["execution_errors"] for name in names]],
               ["独立保存响应", *[summaries[name]["tokens"]["unique_responses"] for name in names]],
               ["已记录请求尝试", *[summaries[name]["tokens"]["attempts_recorded"] for name in names]]]
    metrics.extend([[field, *[summaries[name]["tokens"]["known_usage"][field] for name in names]] for field in TOKEN_FIELDS])
    lines.extend(md_table(["指标", "图片参考＋开闭节点", "文字判据＋opening-only＋时序gate"], metrics))
    lines.extend(["", f"本轮 / 上版已知total_tokens：`{data['known_total_token_ratio_candidate_over_previous']}`；token配对完整：`{data['token_pairing_complete']}`。", "",
                  "实际保存响应及重试归档按request_id或内容去重；未知用量不填0。prepared_for_selection表示尚未请求，annotation_pending表示响应已保存但尚未落标注。token比例不等于计费金额比例。", "",
                  "## 逐条产物与候选", ""])
    lines.extend(md_table(["Episode", "上版状态/候选帧", "本轮状态/候选帧", "节点/区间/图", "参考图", "仅open changes", "观察全序", "时序检查"],
        [[row["key"], f"{(row['previous'] or {}).get('status')} / {[item.get('fixed_source_frame_id') for item in (row['previous'] or {}).get('raw_node_proposals', [])]}",
          f"{(row['candidate'] or {}).get('status')} / {[item.get('fixed_source_frame_id') for item in (row['candidate'] or {}).get('raw_node_proposals', [])]}",
          f"{(row['candidate'] or {}).get('node_count')} / {(row['candidate'] or {}).get('interval_count')} / {((row['candidate'] or {}).get('reference_inventory') or {}).get('request_image_count')}",
          ((row["candidate"] or {}).get("reference_inventory") or {}).get("declared_reference_image_count"),
          (row["candidate"] or {}).get("opening_changes_only_verified"),
          ((row["candidate"] or {}).get("observation_order") or {}).get("chronological"),
          ((row["candidate"] or {}).get("temporal_summary") or {}).get("status")]
         for row in data["episodes"]]))
    lines.extend(["", "## 原始节点提议与固定原始帧映射", "",
                  "候选均来自原始模型回答；待复核时仍保留，不用上版结果回填或移动节点。映射通过不证明所选阶段正确。", ""])
    proposal_rows = []
    for row in data["episodes"]:
        for name in names:
            item = row[name] or {}
            for detail in item.get("raw_node_proposals", []):
                proposal_rows.append([row["key"], name, detail["stage_id"], detail["node_id"], detail["fixed_source_frame_id"],
                                      detail["participant_arms"], detail["evidence_interval_ids"],
                                      detail["failures"] if detail["mapping_checked_against_annotation"] else "待annotation，尚未核对"])
    lines.extend(md_table(["Episode", "版本", "阶段", "原始node_id", "固定帧", "参与手", "证据区间", "约束问题"], proposal_rows))
    candidate_summary = summaries["candidate"]
    lines.extend(["", "## 独立输入与时序核对", "",
        f"参考图为0独立核对：{candidate_summary['zero_reference_images_verified_episodes']}/{candidate_summary['reference_inventory_checked_episodes']}；"
        f"只含open changes：{candidate_summary['opening_changes_only_verified_episodes']}/{candidate_summary['episodes_in_manifest']}；"
        f"完整且时序一致的区间观察：{candidate_summary['observations_complete_and_ordered_episodes']}/{candidate_summary['observation_order_checked_episodes']}；"
        f"固定节点映射：{candidate_summary['fixed_node_mapping_passed_episodes']}/{candidate_summary['fixed_node_mapping_checked_episodes']}。", "",
        "参考图核对同时比较保存request图片数、实际TARGET区间页清单和payload图片清单。时序gate只判断模型陈述是否一致，不能验证其看图解释。", ""])
    lines.extend(md_table(["Episode", "时序发现", "总体复核原因", "训练就绪", "保存标签全-1"],
        [[row["key"], ((row["candidate"] or {}).get("temporal_summary") or {}).get("findings"),
          (row["candidate"] or {}).get("review_reasons"), (row["candidate"] or {}).get("training_ready"),
          ((row["candidate"] or {}).get("saved_labels") or {}).get("all_minus_one")]
         for row in data["episodes"]]))
    lines.extend(["", f"本轮training_ready=false：{candidate_summary['training_ready_false_episodes']}/{candidate_summary['annotated_episodes']}；"
                  f"已保存标签全为-1：{candidate_summary['saved_all_minus_one_label_arrays']}/{candidate_summary['saved_label_arrays']}。未完成请求时不预判最终状态。", ""])
    audits = [(row["key"], name, field, row[name][field]) for row in data["episodes"] for name in names if row[name]
              for field in ("gpt_text_criteria_audit", "gpt_interval_audit") if row[name].get(field)]
    if audits:
        lines.extend(["## 后续视觉审计（非人工真值）", "",
                      "审计独立保存，不覆盖原模型回答，也不把自动accepted或审计结论折算成accuracy。GPT视觉复核仍需人工确认。", ""])
        for episode_key, name, field, audit in audits:
            lines.extend([f"### {episode_key} · {name}", "",
                          f"产物：`{field}`；状态：`{audit.get('status', 'unspecified')}`；审计者：`{audit.get('reviewer', 'unspecified')}`。", ""])
            findings = audit.get("findings", [])
            findings = findings if type(findings) is list else [findings]
            lines.extend(md_table(["阶段", "类型", "发现", "源帧证据"],
                                 [[item.get("stage_id"), item.get("kind"), item.get("description"), item.get("evidence_source_frames")]
                                  if type(item) is dict else [None, None, item, None] for item in findings]))
            lines.append("")
    issues = [[row["key"], name, row[name]["warnings"]] for row in data["episodes"] for name in names if row[name] and row[name]["warnings"]]
    if data["warnings"] or issues or data["model_identity_conflict_keys"] or data["metadata_conflict_keys"]:
        lines.extend(["## 输入与产物问题", "", *[f"- {warning}" for warning in data["warnings"]], "",
                      f"模型冲突：`{data['model_identity_conflict_keys']}`；元数据冲突：`{data['metadata_conflict_keys']}`。", ""])
        lines.extend(md_table(["Episode", "版本", "问题"], issues))
    lines.extend(["", data["accuracy_conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--previous", default=DEFAULT_PREVIOUS)
    parser.add_argument("--run", default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    previous_run, candidate_run = [run_path(project, value) for value in (args.previous, args.run)]
    if previous_run == candidate_run:
        parser.error("previous and candidate runs must differ; old reports are never overwritten")
    data = report(project, previous_run, candidate_run)
    output = candidate_run / "reports"
    output.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output / "text_criteria_probe.json", output / "text_criteria_probe.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(data), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "markdown": str(md_path),
                      "candidate_status_counts": data["paired_summaries"]["candidate"]["status_counts"],
                      "fixed_six_sample_contract_passed": data["fixed_six_sample_contract_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
