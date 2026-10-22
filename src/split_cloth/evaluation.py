"""Compare Qwen results with sampled GPT visual reviews, never human gold.

Only automatically accepted Qwen annotations enter boundary error metrics.
An abstained model's candidate boundaries are retained for inspection but are
not credited as correct predictions.  GPT uncertainty cannot establish either
a correct or an incorrect boundary.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from .util import atomic_bytes, episode_dir, now, read_json, read_jsonl, write_json


DISCLAIMER = (
    "本报告只比较抽样轨迹的Qwen输出与GPT自身视觉盲审。"
    "GPT视觉判断不是独立人工金标准，不能据此报告100条数据的准确率。"
    "边界指标仅覆盖accepted且GPT能提供三个粗边界的轨迹；"
    "被拒判轨迹的候选边界不计入准确性指标，阶段4画外放置质量也未获验证。"
)


def _valid_frames(value: Any, frame_count: Any) -> bool:
    if type(value) is not list or len(value) != 3:
        return False
    if any(type(frame) is not int for frame in value):
        return False
    if not 0 <= value[0] < value[1] < value[2]:
        return False
    if type(frame_count) is int and frame_count > 0:
        return value[2] < frame_count - 1
    return False


def _valid_fps(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return value > 0 and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _compare(review: dict, annotation: dict, manifest: dict) -> dict:
    gpt_frames = review.get("boundary_frames")
    qwen_frames = annotation.get("boundary_frames")
    fps = manifest.get("source_fps")
    frame_count = manifest.get("frame_count")
    gpt_comparable = review.get("applicable") is True and _valid_frames(gpt_frames, frame_count)
    qwen_status = annotation.get("status", "missing")
    gpt_out_of_scope = review.get("status") == "out_of_scope" or review.get("applicable") is False
    qwen_out_of_scope = qwen_status == "out_of_scope"
    result = {
        "status": "unknown",
        "comparison_scope": "three_coarse_stage_boundaries_only",
        "gpt_status": review.get("status"),
        "gpt_task_type": review.get("task_type"),
        "gpt_applicable": review.get("applicable"),
        "qwen_status": qwen_status,
        "qwen_task_type": annotation.get("task_type"),
        "qwen_boundary_frames": deepcopy(qwen_frames),
        "gpt_boundary_frames": deepcopy(gpt_frames),
        "source_fps": fps,
        "accepted_comparable": False,
        "signed_error_convention": "(qwen_frame - gpt_frame) / source_fps; positive means Qwen is later",
        "signed_errors_seconds": None,
        "mae_seconds": None,
        "within_1s": None,
        "all_within_1s": None,
        "is_human_ground_truth": False,
    }
    if review.get("reviewer") != "gpt_visual":
        result["reason"] = "reviewer is not gpt_visual; no GPT boundary comparison performed"
    elif gpt_out_of_scope and qwen_out_of_scope:
        result.update(status="scope_agreement", reason="Both reviewers reject the four-stage folding specification")
    elif gpt_out_of_scope != qwen_out_of_scope:
        result.update(status="disagreement", reason="GPT and Qwen disagree on applicability to the four-stage folding specification")
    elif qwen_status == "accepted" and gpt_comparable:
        if not _valid_frames(qwen_frames, frame_count):
            result["reason"] = "Accepted Qwen annotation has invalid boundary frame types, order, or range"
        elif not _valid_fps(fps):
            result["reason"] = "Manifest source_fps is missing or invalid; seconds cannot be calculated"
        else:
            errors = [(qwen - gpt) / fps for qwen, gpt in zip(qwen_frames, gpt_frames)]
            within = [abs(error) <= 1.0 for error in errors]
            result.update(
                status="pass" if all(within) else "needs_review",
                accepted_comparable=True,
                signed_errors_seconds=[round(error, 6) for error in errors],
                mae_seconds=round(sum(map(abs, errors)) / 3, 6),
                within_1s=within,
                all_within_1s=all(within),
                reason="All three coarse boundaries are within ±1 second" if all(within)
                else "At least one accepted coarse boundary differs from the GPT review by more than one second",
            )
    elif qwen_status == "needs_review":
        if gpt_comparable:
            result.update(status="model_abstained", reason="GPT supplied three boundaries, but Qwen abstained; candidate boundaries are not scored")
        else:
            result.update(status="appropriate_abstention", reason="GPT could not supply three certain applicable boundaries, and Qwen also requested review")
    elif qwen_status == "accepted":
        result["reason"] = "GPT review is uncertain or has null/invalid boundaries; accepted Qwen boundaries cannot be independently scored"
    else:
        result["reason"] = "No comparable accepted annotation or conclusive scope decision is available"
    return result


def evaluate_gpt(run: Path) -> dict:
    """Write per-review comparisons and JSON/Markdown sample reports.

    Existing review status, reviewer, summary, issues, and provenance remain
    unchanged. Only a ``comparison`` object is added or replaced. A malformed
    review is reported as unknown and its original file is left untouched.
    """
    run = Path(run)
    manifests = read_jsonl(run / "manifest.jsonl")
    rows: list[dict] = []
    status_counts: dict[str, int] = {}
    accepted_errors: list[float] = []
    within_boundary_count = 0
    for manifest in manifests:
        output = episode_dir(run, manifest)
        review_path = output / "gpt_review.json"
        if not review_path.exists():
            continue
        row = {
            "key": manifest.get("key", f"{manifest['dataset_id']}__{manifest['episode_id']}"),
            "dataset_id": manifest["dataset_id"],
            "episode_id": manifest["episode_id"],
            "split": manifest.get("split"),
        }
        try:
            review = read_json(review_path)
            if type(review) is not dict:
                raise ValueError("Review is not a JSON object")
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            comparison = {
                "status": "unknown", "accepted_comparable": False,
                "reason": f"Review could not be read: {type(exc).__name__}",
                "is_human_ground_truth": False,
            }
            row["gpt_review"] = None
        else:
            original_review = deepcopy(review)
            original_review.pop("comparison", None)
            # Do not accumulate a nested copy of earlier evaluation results.
            row["gpt_review"] = original_review
            try:
                annotation = read_json(output / "annotation.json", {})
                annotation = annotation if type(annotation) is dict else {}
            except (json.JSONDecodeError, UnicodeError):
                annotation = {}
            comparison = _compare(original_review, annotation, manifest)
            review["comparison"] = comparison
            write_json(review_path, review)
        row["comparison"] = comparison
        rows.append(row)
        status = comparison["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if comparison.get("accepted_comparable"):
            accepted_errors.extend(comparison["signed_errors_seconds"])
            within_boundary_count += sum(comparison["within_1s"])

    comparable = len(accepted_errors) // 3
    boundary_disagreements = status_counts.get("needs_review", 0)
    scope_disagreements = status_counts.get("disagreement", 0)
    report = {
        "run_id": run.name,
        "updated_at": now(),
        "review_source": "gpt_visual",
        "review_scope": "sampled_independent_montage_blind_review",
        "is_human_ground_truth": False,
        "can_report_100_episode_accuracy": False,
        "disclaimer": DISCLAIMER,
        "total_manifest_episodes": len(manifests),
        "reviewed": len(rows),
        "not_reviewed": len(manifests) - len(rows),
        "accepted_comparable_episodes": comparable,
        "fully_within_1s": status_counts.get("pass", 0),
        "compared_boundaries": len(accepted_errors),
        "boundaries_within_1s": within_boundary_count,
        "mae_seconds": round(sum(map(abs, accepted_errors)) / len(accepted_errors), 6) if accepted_errors else None,
        "scope_agreements": status_counts.get("scope_agreement", 0),
        "appropriate_abstentions": status_counts.get("appropriate_abstention", 0),
        "model_abstentions": status_counts.get("model_abstained", 0),
        "disagreements": boundary_disagreements + scope_disagreements,
        "boundary_disagreements": boundary_disagreements,
        "scope_disagreements": scope_disagreements,
        "unknowns": status_counts.get("unknown", 0),
        "status_counts": status_counts,
        "episodes": rows,
    }
    write_json(run / "reports/gpt_validation.json", report)
    lines = [
        f"# {run.name} — GPT抽样视觉比较", "", DISCLAIMER, "",
        f"抽样已有GPT审核记录 {report['reviewed']} 条，manifest共 {len(manifests)} 条，未审 {report['not_reviewed']} 条。", "",
        "## 原始计数", "",
        f"- 可比较的accepted轨迹：{comparable}",
        f"- 三个边界全部在±1秒内：{report['fully_within_1s']}",
        f"- 比较边界数：{report['compared_boundaries']}；±1秒内边界数：{within_boundary_count}",
        f"- 可比较边界MAE（秒）：{report['mae_seconds'] if report['mae_seconds'] is not None else '无可比样本'}",
        f"- 适用范围一致拒绝：{report['scope_agreements']}",
        f"- GPT不确定且模型拒判：{report['appropriate_abstentions']}",
        f"- GPT给出三界但模型拒判：{report['model_abstentions']}",
        f"- 差异：{report['disagreements']}（边界超过1秒 {boundary_disagreements}；适用范围不同 {scope_disagreements}）",
        f"- 无法比较：{report['unknowns']}", "",
        "## 逐条比较", "",
        "| Episode | GPT原状态 | Qwen状态 | 比较结果 | Qwen−GPT边界误差（秒） |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        comparison = row["comparison"]
        original_status = (row.get("gpt_review") or {}).get("status", "unreadable")
        errors = comparison.get("signed_errors_seconds")
        error_text = ", ".join(f"{value:+.3f}" for value in errors) if errors else "不计入边界指标"
        values = [row["key"], original_status, comparison.get("qwen_status", "unknown"), comparison["status"], error_text]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    lines.extend(["", "正误差表示Qwen边界晚于GPT候选；±1秒包括恰好1秒。原始GPT状态与问题记录保留，比较结果单独写入comparison。", ""])
    atomic_bytes(run / "reports/gpt_validation.md", "\n".join(lines).encode())
    return report
