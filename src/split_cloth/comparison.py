"""Read-only, key-aligned comparison of two model annotation runs.

Only the candidate's two comparison reports are written. Acceptance is an
execution/semantic gate, never an accuracy estimate. Existing GPT comparisons
are reported with their coverage; comparable errors on the same GPT reference
are additionally restricted to a shared accepted intersection.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from .util import atomic_bytes, episode_dir, now, read_json, read_jsonl, write_json


DISCLAIMER = (
    "本报告按candidate manifest的key对齐相同样本，主计数只覆盖成功匹配的配对轨迹。"
    "自动接受数/通过率不等于标注准确率；执行错误与语义待复核分开统计。"
    "GPT视觉参考不是独立人工金标，各模型GPT可比指标必须连同覆盖数阅读。"
    "只有双方accepted、元数据一致且使用同一GPT参考的交集才进入共同误差对照。"
    "候选模型审核更多、接受更多或拒判更多均不能单独证明优劣，本报告不排名。"
)


def _key(manifest: dict) -> str:
    value = manifest.get("key")
    if isinstance(value, str) and value:
        return value
    return f"{manifest['dataset_id']}__{manifest['episode_id']}"


def _manifest(run: Path) -> tuple[list[dict], dict[str, dict]]:
    if not (run / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"Manifest does not exist: {run / 'manifest.jsonl'}")
    rows = read_jsonl(run / "manifest.jsonl")
    by_key = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Manifest entries must be JSON objects")
        key = _key(row)
        if key in by_key:
            raise ValueError(f"Duplicate manifest key: {key}")
        by_key[key] = row
    return rows, by_key


def _object(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value, None
    except (OSError, ValueError, UnicodeError) as exc:
        return None, f"{path.name}: {type(exc).__name__}"


def _errors(value: Any) -> list[float] | None:
    if type(value) is not list or len(value) != 3:
        return None
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        return None
    return [float(item) for item in value]


def _frames(value: Any) -> bool:
    return type(value) is list and len(value) == 3 and all(type(item) is int for item in value) and 0 <= value[0] < value[1] < value[2]


def _category(annotation: dict, artifact_error: str | None) -> str:
    if artifact_error:
        return "artifact_read_error"
    if annotation.get("execution_error"):
        return "execution_error"
    status = annotation.get("status", "missing")
    if status == "needs_review":
        return "semantic_needs_review"
    return status if status in {"accepted", "out_of_scope", "unreadable", "pending", "missing"} else "other"


def _snapshot(run: Path, manifest: dict, fallback_model: str | None) -> dict:
    directory = episode_dir(run, manifest)
    annotation, annotation_error = _object(directory / "annotation.json")
    review, review_error = _object(directory / "gpt_review.json")
    annotation = annotation or {}
    comparison = (review or {}).get("comparison")
    comparison = deepcopy(comparison) if isinstance(comparison, dict) else None
    warnings = [value for value in (annotation_error, review_error) if value]
    errors = _errors((comparison or {}).get("signed_errors_seconds"))
    comparable = bool(
        not annotation_error and not review_error and not annotation.get("execution_error")
        and annotation.get("status") == "accepted"
        and (review or {}).get("reviewer") == "gpt_visual"
        and (review or {}).get("applicable") is True
        and (comparison or {}).get("accepted_comparable") is True
        and _frames(annotation.get("boundary_frames"))
        and _frames((review or {}).get("boundary_frames"))
        and errors is not None
    )
    # A comparison may have been cached before an annotation or review changed.
    # Preserve it in the report for inspection, but never score stale evidence.
    if comparison:
        expected = {
            "qwen_status": annotation.get("status", "missing"),
            "qwen_boundary_frames": annotation.get("boundary_frames"),
            "gpt_boundary_frames": (review or {}).get("boundary_frames"),
            "source_fps": manifest.get("source_fps"),
        }
        for field, value in expected.items():
            if field in comparison and comparison[field] != value:
                warnings.append(f"stale GPT comparison: {field} differs from current input")
                comparable = False
    return {
        "status": annotation.get("status", "unreadable_annotation" if annotation_error else "missing"),
        "category": _category(annotation, annotation_error),
        "model_id": annotation.get("model_id") or fallback_model,
        "boundary_frames": deepcopy(annotation.get("boundary_frames")),
        "stage4_completion_frame": annotation.get("stage4_completion_frame"),
        "execution_error": annotation.get("execution_error"),
        "review_reasons": deepcopy(annotation.get("review_reasons", [])),
        "gpt_review_present": review is not None,
        "gpt_review_status": (review or {}).get("status"),
        "gpt_boundary_frames": deepcopy((review or {}).get("boundary_frames")),
        "gpt_comparison": comparison,
        "gpt_metric_eligible": comparable,
        "signed_errors_seconds": errors if comparable else None,
        "mae_seconds": round(sum(map(abs, errors)) / 3, 6) if comparable else None,
        "artifact_warnings": warnings,
    }


def _summary(rows: list[dict]) -> dict:
    statuses = Counter(row["status"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    comparisons = Counter((row["gpt_comparison"] or {}).get("status", "missing_comparison") for row in rows if row["gpt_review_present"])
    eligible = [row for row in rows if row["gpt_metric_eligible"]]
    errors = [value for row in eligible for value in row["signed_errors_seconds"]]
    return {
        "total": len(rows),
        "model_ids": sorted({row["model_id"] for row in rows if isinstance(row["model_id"], str)}),
        "status_counts": dict(sorted(statuses.items())),
        "category_counts": dict(sorted(categories.items())),
        "accepted": categories["accepted"],
        "semantic_needs_review": categories["semantic_needs_review"],
        "execution_errors": categories["execution_error"],
        "artifact_read_errors": categories["artifact_read_error"],
        "gpt_reviewed": sum(row["gpt_review_present"] for row in rows),
        "gpt_comparison_counts": dict(sorted(comparisons.items())),
        "gpt_accepted_comparable_episodes": len(eligible),
        "gpt_compared_boundaries": len(errors),
        "gpt_mae_seconds": round(sum(map(abs, errors)) / len(errors), 6) if errors else None,
        "gpt_fully_within_1s": sum(all(abs(value) <= 1 for value in row["signed_errors_seconds"]) for row in eligible),
        "gpt_boundary_disagreements": sum(any(abs(value) > 1 for value in row["signed_errors_seconds"]) for row in eligible),
        "gpt_scope_disagreements": comparisons["disagreement"],
    }


def _conflicts(baseline: dict, candidate: dict) -> list[str]:
    fields = ("source_fps", "frame_count", "camera_key", "source_video_path", "source_video_sha256")
    return [field for field in fields if field in baseline and field in candidate and baseline[field] != candidate[field]]


def _common_reference(baseline: dict, candidate: dict, conflicts: list[str]) -> tuple[bool, str]:
    if conflicts:
        return False, "manifest metadata differ: " + ", ".join(conflicts)
    if not baseline["gpt_metric_eligible"] or not candidate["gpt_metric_eligible"]:
        return False, "both models must be accepted with a valid existing GPT comparison"
    if baseline["gpt_boundary_frames"] != candidate["gpt_boundary_frames"]:
        return False, "GPT reference boundaries differ between runs"
    return True, "same matched sample and same three GPT reference boundaries"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_errors(row: dict | None) -> str:
    if not row:
        return "无匹配样本"
    comparison = row["gpt_comparison"]
    status = comparison.get("status", "unknown") if comparison else "未比较"
    errors = row["signed_errors_seconds"]
    return status + (" / " + ", ".join(f"{value:+.3f}" for value in errors) + " 秒" if errors is not None else " / 不计边界指标")


def _markdown(report: dict) -> str:
    b, c, matching, paired = report["baseline"], report["candidate"], report["matching"], report["paired"]
    lines = [f"# 模型配对对照：{report['baseline_run_id']} → {report['candidate_run_id']}", "", DISCLAIMER, "",
             f"基线模型：{', '.join(b['model_ids']) or '未记录'}；候选模型：{', '.join(c['model_ids']) or '未记录'}。", "",
             f"candidate清单 {matching['candidate_manifest_episodes']} 条，与baseline按key匹配 {matching['matched_episodes']} 条；未匹配 {len(matching['missing_baseline_keys'])} 条。", "",
             f"baseline完整清单 {matching['baseline_manifest_episodes']} 条的计数另保存在JSON的full_manifest；下表的两列均为相同的配对样本。", "",
             "## 同一批样本的处理计数", "", "| 指标 | baseline | candidate |", "|---|---:|---:|"]
    labels = (("total", "配对轨迹"), ("accepted", "自动接受"), ("semantic_needs_review", "语义待复核（无执行错误）"),
              ("execution_errors", "执行错误"), ("artifact_read_errors", "结果文件读取错误"), ("gpt_reviewed", "已有GPT复核记录"),
              ("gpt_accepted_comparable_episodes", "各自accepted且GPT可比轨迹"), ("gpt_boundary_disagreements", "各自与GPT边界差异超过1秒"),
              ("gpt_scope_disagreements", "各自与GPT适用范围分歧"))
    for key, label in labels:
        lines.append(f"| {label} | {b[key]} | {c[key]} |")
    for status in sorted(set(b["status_counts"]) | set(c["status_counts"])):
        lines.append(f"| 原始状态 `{_cell(status)}` | {b['status_counts'].get(status, 0)} | {c['status_counts'].get(status, 0)} |")
    lines.extend(["", f"自动接受数变化：{paired['accepted_delta']:+d}；新增接受 {len(paired['newly_accepted_keys'])} 条，转为非接受 {len(paired['no_longer_accepted_keys'])} 条。此变化不能解释为准确率变化。", "",
                  "## GPT粗参考误差与分歧", "",
                  f"各自可比覆盖：baseline {b['gpt_accepted_comparable_episodes']} 条 / candidate {c['gpt_accepted_comparable_episodes']} 条。各自MAE：{b['gpt_mae_seconds'] if b['gpt_mae_seconds'] is not None else '无'} / {c['gpt_mae_seconds'] if c['gpt_mae_seconds'] is not None else '无'} 秒；覆盖不同，不能直接据此排名。", "",
                  f"双方accepted且使用同一GPT参考的交集：{paired['gpt_common_reference_episodes']} 条，{paired['gpt_common_compared_boundaries']} 个边界。", "",
                  "| 共同参考交集指标 | baseline | candidate |", "|---|---:|---:|",
                  f"| MAE（秒） | {paired['baseline_common_mae_seconds'] if paired['baseline_common_mae_seconds'] is not None else '无可比样本'} | {paired['candidate_common_mae_seconds'] if paired['candidate_common_mae_seconds'] is not None else '无可比样本'} |",
                  f"| 三个边界均在±1秒内的轨迹 | {paired['baseline_common_fully_within_1s']} | {paired['candidate_common_fully_within_1s']} |", "",
                  "正误差表示Qwen边界晚于GPT粗参考。上述指标只是与GPT模型参考的一致性，不是人工验证准确率。", "",
                  "## 按key对齐的逐条结果", "",
                  "| Episode | baseline状态 / 类别 | candidate状态 / 类别 | 接受变化 | baseline GPT比较 / 有符号误差 | candidate GPT比较 / 有符号误差 | 共同参考交集 |",
                  "|---|---|---|---|---|---|---|"])
    changes = {"became_accepted": "新增接受", "lost_accepted": "转为非接受", "both_accepted": "双方接受", "neither_accepted": "双方非接受", "unmatched": "无基线匹配"}
    for row in report["episodes"]:
        base, cand = row["baseline"], row["candidate"]
        values = [row["key"], f"{base['status']} / {base['category']}" if base else "无匹配样本", f"{cand['status']} / {cand['category']}",
                  changes[row["acceptance_change"]], _format_errors(base), _format_errors(cand), "是" if row["gpt_common_reference"] else "否"]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        if row["metadata_conflicts"]:
            lines.append(f"\n- `{_cell(row['key'])}` 元数据冲突：{', '.join(row['metadata_conflicts'])}；排除共同误差指标。\n")
    lines.extend(["", "原annotation、gpt_review及GPT盲审状态未修改。若需解释某条差异，请结合JSON中的reason、review_reasons、artifact_warnings与原始网格图检查。", ""])
    return "\n".join(lines)


def compare_runs(baseline: Path, candidate: Path) -> dict:
    """Write model_comparison.json/.md under candidate/reports, preserving inputs."""
    baseline, candidate = Path(baseline), Path(candidate)
    if baseline.resolve() == candidate.resolve():
        raise ValueError("Baseline and candidate must be different runs")
    baseline_rows, baseline_index = _manifest(baseline)
    candidate_rows, _ = _manifest(candidate)
    baseline_meta, _ = _object(baseline / "run.json")
    candidate_meta, _ = _object(candidate / "run.json")
    baseline_model = ((baseline_meta or {}).get("config") or {}).get("provider", {}).get("model_id")
    candidate_model = ((candidate_meta or {}).get("config") or {}).get("provider", {}).get("model_id")
    baseline_all = {_key(row): _snapshot(baseline, row, baseline_model) for row in baseline_rows}
    candidate_all = {_key(row): _snapshot(candidate, row, candidate_model) for row in candidate_rows}
    paired_baseline, paired_candidate, rows, missing, metadata_conflicts = [], [], [], [], []
    new_accepts, lost_accepts, both_accepts = [], [], []
    common_baseline, common_candidate = [], []
    for manifest in candidate_rows:
        key = _key(manifest)
        c = candidate_all[key]
        b = baseline_all.get(key)
        conflicts = _conflicts(baseline_index[key], manifest) if b is not None else []
        if b is None:
            missing.append(key)
            common, common_reason, change = False, "candidate key is absent from baseline manifest", "unmatched"
        else:
            paired_baseline.append(b)
            paired_candidate.append(c)
            common, common_reason = _common_reference(b, c, conflicts)
            b_accepted, c_accepted = b["category"] == "accepted", c["category"] == "accepted"
            change = "both_accepted" if b_accepted and c_accepted else "became_accepted" if c_accepted else "lost_accepted" if b_accepted else "neither_accepted"
            if change == "both_accepted":
                both_accepts.append(key)
            elif change == "became_accepted":
                new_accepts.append(key)
            elif change == "lost_accepted":
                lost_accepts.append(key)
            if common:
                common_baseline.append(b)
                common_candidate.append(c)
        if conflicts:
            metadata_conflicts.append({"key": key, "fields": conflicts})
        rows.append({"key": key, "matched": b is not None, "metadata_conflicts": conflicts, "baseline": b, "candidate": c,
                     "acceptance_change": change, "gpt_common_reference": common, "gpt_common_reference_reason": common_reason})
    b_summary, c_summary = _summary(paired_baseline), _summary(paired_candidate)
    b_common, c_common = _summary(common_baseline), _summary(common_candidate)
    report = {
        "baseline_run_id": baseline.name, "candidate_run_id": candidate.name, "updated_at": now(),
        "comparison_scope": "matched_candidate_manifest_keys", "disclaimer": DISCLAIMER,
        "is_human_ground_truth": False, "acceptance_is_accuracy": False, "ranking_performed": False,
        "matching": {"baseline_manifest_episodes": len(baseline_rows), "candidate_manifest_episodes": len(candidate_rows),
                     "matched_episodes": len(paired_baseline), "missing_baseline_keys": missing, "metadata_conflicts": metadata_conflicts},
        "baseline": b_summary, "candidate": c_summary,
        "full_manifest": {"baseline": _summary(list(baseline_all.values())), "candidate": _summary(list(candidate_all.values()))},
        "paired": {"accepted_delta": c_summary["accepted"] - b_summary["accepted"], "newly_accepted_keys": new_accepts,
                   "no_longer_accepted_keys": lost_accepts, "both_accepted_keys": both_accepts,
                   "gpt_common_reference_episodes": len(common_baseline), "gpt_common_compared_boundaries": b_common["gpt_compared_boundaries"],
                   "baseline_common_mae_seconds": b_common["gpt_mae_seconds"], "candidate_common_mae_seconds": c_common["gpt_mae_seconds"],
                   "baseline_common_fully_within_1s": b_common["gpt_fully_within_1s"], "candidate_common_fully_within_1s": c_common["gpt_fully_within_1s"]},
        "episodes": rows,
    }
    write_json(candidate / "reports/model_comparison.json", report)
    atomic_bytes(candidate / "reports/model_comparison.md", _markdown(report).encode())
    return report
