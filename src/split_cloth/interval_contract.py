"""Validate interval semantics and map fixed gripper node IDs to stage ends.

The model never supplies frame indices.  In specification 5.2, the selected
opening node belongs to the ending stage, and the next stage starts one frame
later.  Feedback nodes remain estimates, not human ground truth.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .boundaries import TASK_TYPES


SPEC_VERSION = "gripper_interval_v1"
ACTIVITIES = {"prepare", "first_fold", "second_fold", "cross_fold", "remove", "other", "unknown"}
_OUTPUT_KEYS = {
    "task_type", "applicable", "all_stages_observed", "needs_review", "review_reasons",
    "interval_observations", "stage_end_nodes", "stage4_completion",
}
_OBSERVATION_KEYS = {"interval_id", "activity", "operation_complete", "evidence_text"}
_END_KEYS = {"stage_id", "node_id", "participant_arms", "evidence_interval_ids", "evidence_text"}
_COMPLETION_KEYS = {"observed", "interval_id", "evidence_text"}


def _exact_keys(value: dict, expected: set[str], path: str, errors: list[str]) -> None:
    missing, extra = expected - value.keys(), value.keys() - expected
    if missing:
        errors.append(f"{path}: missing fields {sorted(missing)}")
    if extra:
        errors.append(f"{path}: unexpected fields {sorted(map(repr, extra))}")


def _id(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _nullable_id(value: Any) -> bool:
    return value is None or _id(value)


def _unique_ids(value: Any) -> bool:
    return type(value) is list and all(_id(item) for item in value) and len(value) == len(set(value))


def validate_interval_selection(raw: Any) -> list[str]:
    """Validate the exact node-only response schema without coercion.

    Null nodes and empty evidence arrays are valid representations of an
    unresolved decision; semantic validation subsequently keeps labels at -1.
    In particular, extra frame/timestamp fields are never silently ignored.
    """
    errors: list[str] = []
    if type(raw) is not dict:
        return ["model_output: expected object"]
    _exact_keys(raw, _OUTPUT_KEYS, "model_output", errors)
    if type(raw.get("task_type")) is not str or raw.get("task_type") not in TASK_TYPES:
        errors.append("model_output.task_type: invalid task type")
    if raw.get("applicable") is not None and type(raw.get("applicable")) is not bool:
        errors.append("model_output.applicable: expected boolean or null")
    for name in ("all_stages_observed", "needs_review"):
        if type(raw.get(name)) is not bool:
            errors.append(f"model_output.{name}: expected boolean")
    reasons = raw.get("review_reasons")
    if type(reasons) is not list or any(type(item) is not str for item in reasons):
        errors.append("model_output.review_reasons: expected list of strings")
    observations = raw.get("interval_observations")
    if type(observations) is not list:
        errors.append("model_output.interval_observations: expected list")
    else:
        for index, item in enumerate(observations):
            path = f"model_output.interval_observations[{index}]"
            if type(item) is not dict:
                errors.append(f"{path}: expected object")
                continue
            _exact_keys(item, _OBSERVATION_KEYS, path, errors)
            if not _id(item.get("interval_id")):
                errors.append(f"{path}.interval_id: expected nonempty string")
            if type(item.get("activity")) is not str or item.get("activity") not in ACTIVITIES:
                errors.append(f"{path}.activity: invalid activity")
            if type(item.get("operation_complete")) is not bool:
                errors.append(f"{path}.operation_complete: expected boolean")
            if type(item.get("evidence_text")) is not str:
                errors.append(f"{path}.evidence_text: expected string")
    ends = raw.get("stage_end_nodes")
    if type(ends) is not list or len(ends) != 3:
        errors.append("model_output.stage_end_nodes: expected exactly three stage ends")
    if type(ends) is list:
        for index, item in enumerate(ends):
            path = f"model_output.stage_end_nodes[{index}]"
            if type(item) is not dict:
                errors.append(f"{path}: expected object")
                continue
            _exact_keys(item, _END_KEYS, path, errors)
            if type(item.get("stage_id")) is not int or item.get("stage_id") != index + 1:
                errors.append(f"{path}.stage_id: expected integer {index + 1}")
            if not _nullable_id(item.get("node_id")):
                errors.append(f"{path}.node_id: expected nonempty string or null")
            arms = item.get("participant_arms")
            if not _unique_ids(arms) or any(arm not in {"left", "right"} for arm in arms):
                errors.append(f"{path}.participant_arms: expected unique left/right arm list")
            if not _unique_ids(item.get("evidence_interval_ids")):
                errors.append(f"{path}.evidence_interval_ids: expected unique interval ID list")
            if type(item.get("evidence_text")) is not str:
                errors.append(f"{path}.evidence_text: expected string")
    completion = raw.get("stage4_completion")
    path = "model_output.stage4_completion"
    if type(completion) is not dict:
        errors.append(f"{path}: expected object")
    else:
        _exact_keys(completion, _COMPLETION_KEYS, path, errors)
        if type(completion.get("observed")) is not bool:
            errors.append(f"{path}.observed: expected boolean")
        if not _nullable_id(completion.get("interval_id")):
            errors.append(f"{path}.interval_id: expected nonempty string or null")
        if type(completion.get("evidence_text")) is not str:
            errors.append(f"{path}.evidence_text: expected string")
    return errors


def _provided_frames(interval: dict) -> list[int]:
    """Read the sampled frames actually made available for an interval."""
    values = interval.get("sample_frame_ids")
    if values is None:
        samples = interval.get("samples")
        values = [sample.get("source_frame_id") for sample in samples if type(sample) is dict] if type(samples) is list else []
    return values if type(values) is list else []


def _input_indices(nodes: Any, intervals: Any, frame_count: int, errors: list[str]):
    by_node: dict[str, dict] = {}
    by_interval: dict[str, dict] = {}
    if type(nodes) is not list:
        errors.append("nodes: expected list")
        nodes = []
    previous_frame = -1
    for index, node in enumerate(nodes):
        path = f"nodes[{index}]"
        if type(node) is not dict:
            errors.append(f"{path}: expected object")
            continue
        node_id, frame = node.get("node_id"), node.get("source_frame_id")
        valid = True
        if not _id(node_id) or node_id in by_node:
            errors.append(f"{path}.node_id: expected unique nonempty string")
            valid = False
        if type(frame) is not int or not 0 <= frame < frame_count:
            errors.append(f"{path}.source_frame_id: outside video or not integer")
            valid = False
        elif frame <= previous_frame:
            errors.append("nodes: source_frame_id must be strictly increasing")
        else:
            previous_frame = frame
        arms = node.get("open_arms")
        if not _unique_ids(arms) or any(arm not in {"left", "right"} for arm in arms):
            errors.append(f"{path}.open_arms: expected unique left/right arm list")
        if type(node.get("eligible_stage_end")) is not bool:
            errors.append(f"{path}.eligible_stage_end: expected boolean")
        if valid:
            by_node[node_id] = node
    if type(intervals) is not list or not intervals:
        errors.append("intervals: expected nonempty list")
        intervals = []
    cursor = 0
    for index, interval in enumerate(intervals):
        path = f"intervals[{index}]"
        if type(interval) is not dict:
            errors.append(f"{path}: expected object")
            continue
        interval_id = interval.get("interval_id")
        start, end = interval.get("start_frame"), interval.get("end_frame")
        valid_id = _id(interval_id) and interval_id not in by_interval
        if not valid_id:
            errors.append(f"{path}.interval_id: expected unique nonempty string")
        valid_bounds = type(start) is int and type(end) is int and 0 <= start <= end < frame_count
        if not valid_bounds:
            errors.append(f"{path}: invalid closed interval bounds")
        else:
            if start != cursor:
                errors.append(f"{path}: intervals must cover every frame once in order")
            cursor = end + 1
        if valid_id and valid_bounds:
            by_interval[interval_id] = interval
        end_node_id = interval.get("right_node_id")
        end_node = by_node.get(end_node_id) if type(end_node_id) is str else None
        if end_node_id is not None and (end_node is None or end_node["source_frame_id"] != end):
            errors.append(f"{path}.right_node_id: must resolve to the interval end")
        frames = _provided_frames(interval)
        if not frames or any(type(frame) is not int for frame in frames):
            errors.append(f"{path}: missing supplied sample_frame_ids")
        elif valid_bounds:
            if any(not start <= frame <= end for frame in frames):
                errors.append(f"{path}: supplied frames must lie within interval")
            if frames != sorted(set(frames)):
                errors.append(f"{path}: supplied frames must be strictly increasing")
            if start not in frames or end not in frames:
                errors.append(f"{path}: first and last interval frames must be supplied")
    if cursor != frame_count:
        errors.append("intervals: must cover the complete video")
    return by_node, by_interval


def _temporal_validation(response, by_node, by_interval, by_observation, frame_count, applicable):
    """Check model claims against all intervals, without inventing boundaries.

    Preparation, adjustment and uncertain observations never advance the stage.
    An opening alone does not advance it either. Known folding activities must
    agree with the selected stage ranges; uncertain boundaries defer checks
    whose limits cannot be established. This checks claim consistency, not the
    truth of a visual interpretation.
    """
    findings = []
    interval_ids = list(by_interval)
    observations = response.get("interval_observations")
    observations = observations if type(observations) is list else []
    observed_ids = [item.get("interval_id") if type(item) is dict else None for item in observations]
    order_valid = observed_ids == interval_ids
    if not order_valid:
        findings.append("temporal_order: interval observations must follow the complete supplied interval order")
    summary = {
        "enabled": True, "status": "passed", "observation_order_valid": order_valid,
        "checked_interval_ids": interval_ids, "stage_coverage": [],
        "observed_activity_sequence": [], "deferred_stage_ids": [], "findings": findings,
        "interpretation": "Temporal consistency of model claims only; not verified visual accuracy. Unresolved node limits are deferred, never inferred.",
    }
    if not applicable:
        summary["status"] = "needs_review" if findings else "not_applicable"
        return summary
    ends = response.get("stage_end_nodes")
    ends = ends if type(ends) is list else []
    end_frames = []
    for index in range(3):
        item = ends[index] if index < len(ends) and type(ends[index]) is dict else {}
        node_id = item.get("node_id")
        node = by_node.get(node_id) if type(node_id) is str else None
        end_frames.append(node["source_frame_id"] if node is not None else None)
    activities = ("first_fold", "second_fold", "cross_fold", "remove")
    ranks = {activity: index + 1 for index, activity in enumerate(activities)}
    ranges = {}
    for stage, activity in enumerate(activities, start=1):
        lower = 0 if stage == 1 else (end_frames[stage - 2] + 1 if end_frames[stage - 2] is not None else None)
        upper = frame_count - 1 if stage == 4 else end_frames[stage - 1]
        fully_known = lower is not None and upper is not None and lower <= upper
        ranges[stage] = (lower, upper)
        members = [interval for interval in by_interval.values() if fully_known
                   and lower <= interval["start_frame"] <= interval["end_frame"] <= upper]
        completed = [interval["interval_id"] for interval in members
                     if by_observation.get(interval["interval_id"], {}).get("activity") == activity
                     and by_observation.get(interval["interval_id"], {}).get("operation_complete") is True
                     and str(by_observation.get(interval["interval_id"], {}).get("evidence_text", "")).strip()]
        summary["stage_coverage"].append({
            "stage_id": stage, "expected_activity": activity,
            "start_frame": lower, "end_frame": upper, "range_fully_known": fully_known,
            "interval_ids": [interval["interval_id"] for interval in members],
            "completed_activity_interval_ids": completed,
        })
        if not fully_known:
            summary["deferred_stage_ids"].append(stage)
        elif not completed:
            findings.append(f"temporal_order: stage {stage} requires completed {activity} evidence within its selected range")
    highest_rank = 0
    highest_interval_id = None
    for interval_id, interval in by_interval.items():
        observation = by_observation.get(interval_id, {})
        activity = observation.get("activity")
        if type(activity) is not str or activity not in ranks:
            continue
        stage = ranks[activity]
        summary["observed_activity_sequence"].append({
            "interval_id": interval_id, "activity": activity,
            "operation_complete": observation.get("operation_complete") is True,
        })
        if stage < highest_rank:
            findings.append(f"temporal_order: {interval_id} {activity} regresses after stage {highest_rank} activity in {highest_interval_id}")
        if stage > highest_rank:
            highest_rank, highest_interval_id = stage, interval_id
        lower, upper = ranges[stage]
        if ((lower is not None and interval["start_frame"] < lower)
                or (upper is not None and interval["end_frame"] > upper)):
            findings.append(f"temporal_order: {interval_id} {activity} lies outside selected stage {stage} range")
    summary["status"] = "needs_review" if findings else ("deferred" if summary["deferred_stage_ids"] else "passed")
    return summary


def normalize_interval_selection(raw: Any, nodes: Any, intervals: Any, manifest: Any, *, temporal_order: bool = False) -> dict:
    """Return safe labels, retaining unresolved candidates for review.

    Three selected fixed nodes must be increasing eligible opening nodes. Each
    stage needs evidence of its own folding operation and of completion in the
    interval ending at the chosen node. Stage 4 completion refers to the end of
    an observed interval; it does not claim a precise first completion frame.
    Optional temporal checks evaluate every interval's claims, including those
    omitted from boundary evidence. The default retains legacy acceptance.
    """
    errors = validate_interval_selection(raw)
    if type(manifest) is not dict:
        errors.append("manifest: expected object")
        manifest = {}
    frame_count = manifest.get("frame_count", manifest.get("num_frames"))
    if type(frame_count) is not int or frame_count <= 0:
        errors.append("manifest.frame_count: expected positive integer")
        frame_count = 0
    by_node, by_interval = _input_indices(nodes, intervals, frame_count, errors)
    response = raw if type(raw) is dict else {}
    task_type, applicable = response.get("task_type", "unknown"), response.get("applicable")
    out_of_scope = (type(task_type) is str and task_type in TASK_TYPES - {"fold_cloth", "unknown"}) or applicable is False
    if task_type == "unknown":
        out_of_scope = False
    check_semantics = not out_of_scope
    observations = response.get("interval_observations")
    observations = observations if type(observations) is list else []
    by_observation: dict[str, dict] = {}
    for item in observations:
        if type(item) is not dict or not _id(item.get("interval_id")):
            continue
        interval_id = item["interval_id"]
        if interval_id not in by_interval:
            errors.append(f"interval_observations: unknown interval ID {interval_id!r}")
        if interval_id in by_observation:
            errors.append(f"interval_observations: duplicate interval ID {interval_id!r}")
        by_observation[interval_id] = item
    if set(by_observation) != set(by_interval) or len(observations) != len(by_interval):
        errors.append("interval_observations: each supplied interval requires exactly one observation")
    if check_semantics:
        if task_type != "fold_cloth":
            errors.append("task_type: fold_cloth must be confirmed for automatic acceptance")
        if applicable is not True:
            errors.append("applicable: must be true for automatic acceptance")
        if response.get("all_stages_observed") is not True:
            errors.append("all_stages_observed: all four stages must be observed")
        if response.get("needs_review") is True:
            errors.append("model_requested_review")
        if response.get("review_reasons"):
            errors.append("model_supplied_review_reasons")
    ends = response.get("stage_end_nodes")
    ends = ends if type(ends) is list else []
    boundaries = []
    previous_frame = -1
    selected_ids = []
    expected_activities = ("first_fold", "second_fold", "cross_fold")
    for index, item in enumerate(ends[:3]):
        if type(item) is not dict:
            continue
        path = f"stage_end_nodes[{index}]"
        node_id = item.get("node_id")
        node = by_node.get(node_id) if type(node_id) is str else None
        evidence_ids = item.get("evidence_interval_ids")
        evidence_ids = evidence_ids if type(evidence_ids) is list else []
        evidence_intervals = [by_interval[item] for item in evidence_ids if type(item) is str and item in by_interval]
        arms = item.get("participant_arms")
        arms = arms if type(arms) is list else []
        if check_semantics:
            if node is None:
                errors.append(f"{path}.node_id: unresolved or unknown fixed node {node_id!r}")
            else:
                frame = node["source_frame_id"]
                if node.get("eligible_stage_end") is not True or not node.get("open_arms"):
                    errors.append(f"{path}: selected node must be an eligible opening node")
                if frame <= previous_frame or node_id in selected_ids:
                    errors.append(f"{path}: selected nodes must be strictly increasing and unique")
                if frame >= frame_count - 1:
                    errors.append(f"{path}: selected end must leave a frame for the next stage")
                open_arms = node.get("open_arms")
                open_arms = open_arms if type(open_arms) is list else []
                if not arms or any(arm not in open_arms for arm in arms):
                    errors.append(f"{path}.participant_arms: all participants must be open at the selected node")
                if not evidence_ids or len(evidence_intervals) != len(evidence_ids):
                    errors.append(f"{path}: evidence must reference known supplied intervals")
                if any(interval["start_frame"] <= previous_frame or interval["end_frame"] > frame for interval in evidence_intervals):
                    errors.append(f"{path}: evidence must belong to this stage before its selected end")
                terminal = [interval for interval in evidence_intervals
                            if interval.get("right_node_id") == node_id and interval["end_frame"] == frame]
                if not terminal:
                    errors.append(f"{path}: evidence must include the interval ending at the selected node")
                elif not any(by_observation.get(interval["interval_id"], {}).get("operation_complete") is True
                             and str(by_observation.get(interval["interval_id"], {}).get("evidence_text", "")).strip()
                             for interval in terminal):
                    errors.append(f"{path}: terminal interval must confirm completed operation with evidence")
                if not any(by_observation.get(interval["interval_id"], {}).get("activity") == expected_activities[index]
                           and str(by_observation.get(interval["interval_id"], {}).get("evidence_text", "")).strip()
                           for interval in evidence_intervals):
                    errors.append(f"{path}: {expected_activities[index]} evidence is required")
                if not str(item.get("evidence_text", "")).strip():
                    errors.append(f"{path}.evidence_text: completion and release evidence is required")
        if node is not None:
            frame = node["source_frame_id"]
            boundaries.append({
                "from_stage": index + 1, "to_stage": index + 2, "stage_id": index + 1,
                "node_id": node_id, "coarse_end_frame": frame, "end_frame": frame,
                "next_stage_start_frame": frame + 1,
                "participant_arms": deepcopy(arms), "evidence_interval_ids": deepcopy(evidence_ids),
                "evidence_text": item.get("evidence_text", ""),
                "precision": "fixed_gripper_node_selection", "node_is_human_ground_truth": False,
            })
            previous_frame = frame
            selected_ids.append(node_id)
    completion_raw = response.get("stage4_completion")
    completion_raw = completion_raw if type(completion_raw) is dict else {}
    completion_id = completion_raw.get("interval_id")
    completion_interval = by_interval.get(completion_id) if type(completion_id) is str else None
    completion_observation = by_observation.get(completion_id, {}) if type(completion_id) is str else {}
    completion_frame = None
    if completion_raw.get("observed") is True and completion_interval is not None:
        end = completion_interval.get("end_frame")
        if type(end) is int and end in _provided_frames(completion_interval):
            completion_frame = end
    if check_semantics:
        if completion_raw.get("observed") is not True:
            errors.append("stage4_completion.observed: final stage completion must be observed independently")
        if completion_interval is None or completion_frame is None:
            errors.append("stage4_completion.interval_id: must identify an interval with its ending frame supplied")
        elif len(boundaries) != 3 or completion_interval["start_frame"] <= boundaries[2]["end_frame"]:
            errors.append("stage4_completion: interval must start after the third stage end")
        if completion_observation.get("activity") != "remove" or completion_observation.get("operation_complete") is not True:
            errors.append("stage4_completion: completed remove activity evidence is required")
        if not str(completion_raw.get("evidence_text", "")).strip() or not str(completion_observation.get("evidence_text", "")).strip():
            errors.append("stage4_completion: nonempty visual evidence is required")
    temporal_validation = {
        "enabled": False, "status": "disabled", "findings": [],
        "interpretation": "Legacy normalization; optional temporal consistency checks were not enabled.",
    }
    if type(temporal_order) is not bool:
        errors.append("temporal_order: expected boolean configuration")
        temporal_validation["status"] = "invalid_configuration"
    elif temporal_order:
        temporal_validation = _temporal_validation(response, by_node, by_interval, by_observation, frame_count, check_semantics)
        errors.extend(temporal_validation["findings"])
    errors = list(dict.fromkeys(errors))
    status = "needs_review" if errors else ("out_of_scope" if out_of_scope else "accepted")
    candidate_frames = [boundary["end_frame"] for boundary in boundaries]
    accepted_frames = candidate_frames if status == "accepted" else []
    if status == "accepted":
        starts, ends = [0] + [frame + 1 for frame in accepted_frames], accepted_frames + [frame_count - 1]
        segments = [{"stage": stage, "start_frame": start, "end_frame": end}
                    for stage, (start, end) in enumerate(zip(starts, ends), start=1)]
    else:
        segments = [{"stage": -1, "start_frame": 0, "end_frame": frame_count - 1}] if frame_count else []
    raw_reasons = response.get("review_reasons")
    raw_reasons = raw_reasons if type(raw_reasons) is list else []
    return {
        "schema_version": "1.0", "spec_version": SPEC_VERSION,
        "precision": "fixed_gripper_node_selection", "training_ready": False,
        "precision_note": "Fixed measured-feedback node estimates; not human ground truth or visually verified first fully-open frames.",
        "interval_convention": "closed; selected end belongs to previous stage; next starts at end + 1",
        "frame_count": frame_count, "status": status,
        "task_type": task_type if type(task_type) is str and task_type in TASK_TYPES else "unknown",
        "applicable": applicable if type(applicable) is bool else None,
        "all_stages_observed": response.get("all_stages_observed") is True,
        "needs_review": status == "needs_review",
        "review_reasons": list(dict.fromkeys([reason for reason in raw_reasons if type(reason) is str] + errors)),
        "machine_review_reasons": errors,
        "out_of_scope_reason": "task type or applicability is outside the four-stage cloth specification" if status == "out_of_scope" else None,
        "interval_observations": deepcopy(observations),
        "temporal_validation": temporal_validation,
        "boundaries": boundaries, "candidate_boundary_frames": candidate_frames,
        "boundary_frames": accepted_frames, "segments": segments,
        "stage4_completion": {
            "observed": completion_raw.get("observed") is True,
            "interval_id": completion_id if type(completion_id) is str else None,
            "evidence_text": completion_raw.get("evidence_text", ""),
            "source_frame_id": completion_frame,
            "precision": "observed_interval_end_not_first_completion_frame",
        },
        "stage4_completion_frame": completion_frame,
        "model_output": deepcopy(raw),
    }
