"""Strict VLM output validation and inclusive coarse stage boundaries.

The model chooses *sample IDs*, never source frame numbers.  This module is
the sole conversion point from those IDs to original video frame indices.
Every interval in the public contract is closed: [start_frame, end_frame].
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SPEC_VERSION = "coarse_v1.1_expert"
TASK_TYPES = {"fold_cloth", "fold_pants", "hang_cloth", "mixed", "other", "unknown"}
_OUTPUT_KEYS = {
    "task_type", "applicable", "all_stages_observed", "needs_review",
    "review_reasons", "stage4_completion", "boundary_proposals",
}
_PROPOSAL_KEYS = {
    "from_stage", "to_stage", "last_not_complete_sample_id",
    "first_complete_sample_id", "evidence_text",
}
_COMPLETION_KEYS = {"observed", "completion_sample_id", "evidence_text"}


def _exact_keys(value: dict, expected: set[str], path: str, errors: list[str]) -> None:
    missing = expected.difference(value)
    extra = set(value).difference(expected)
    if missing:
        errors.append(f"{path}: missing fields {sorted(missing)}")
    if extra:
        # repr avoids an exception when a malformed caller supplies mixed keys.
        errors.append(f"{path}: unexpected fields {sorted(map(repr, extra))}")


def _nullable_id(value: Any) -> bool:
    return value is None or (type(value) is str and bool(value.strip()))


def validate_model_output(model_output: Any) -> list[str]:
    """Return JSON-schema errors; this function never coerces values.

    Both boundary IDs may be null when evidence is insufficient.  Such an
    output is schema-valid, but cannot be automatically accepted.
    """
    errors: list[str] = []
    if type(model_output) is not dict:
        return ["model_output: expected object"]
    _exact_keys(model_output, _OUTPUT_KEYS, "model_output", errors)
    task_type = model_output.get("task_type")
    if type(task_type) is not str or task_type not in TASK_TYPES:
        errors.append("model_output.task_type: invalid task type")
    applicable = model_output.get("applicable")
    if applicable is not None and type(applicable) is not bool:
        errors.append("model_output.applicable: expected boolean or null")
    for name in ("all_stages_observed", "needs_review"):
        if type(model_output.get(name)) is not bool:
            errors.append(f"model_output.{name}: expected boolean")
    reasons = model_output.get("review_reasons")
    if type(reasons) is not list or any(type(item) is not str for item in reasons):
        errors.append("model_output.review_reasons: expected list of strings")
    completion = model_output.get("stage4_completion")
    if type(completion) is not dict:
        errors.append("model_output.stage4_completion: expected object")
    else:
        path = "model_output.stage4_completion"
        _exact_keys(completion, _COMPLETION_KEYS, path, errors)
        if type(completion.get("observed")) is not bool:
            errors.append(f"{path}.observed: expected boolean")
        if not _nullable_id(completion.get("completion_sample_id")):
            errors.append(f"{path}.completion_sample_id: expected nonempty string or null")
        if type(completion.get("evidence_text")) is not str:
            errors.append(f"{path}.evidence_text: expected string")
    proposals = model_output.get("boundary_proposals")
    if type(proposals) is not list or len(proposals) != 3:
        errors.append("model_output.boundary_proposals: expected exactly three proposals")
    if type(proposals) is list:
        for index, proposal in enumerate(proposals):
            path = f"model_output.boundary_proposals[{index}]"
            if type(proposal) is not dict:
                errors.append(f"{path}: expected object")
                continue
            _exact_keys(proposal, _PROPOSAL_KEYS, path, errors)
            for name, expected in (("from_stage", index + 1), ("to_stage", index + 2)):
                value = proposal.get(name)
                if type(value) is not int or value != expected:
                    errors.append(f"{path}.{name}: expected integer {expected}")
            for name in ("last_not_complete_sample_id", "first_complete_sample_id"):
                if not _nullable_id(proposal.get(name)):
                    errors.append(f"{path}.{name}: expected nonempty string or null")
            if type(proposal.get("evidence_text")) is not str:
                errors.append(f"{path}.evidence_text: expected string")
    return errors


def validate_review_frames(
    frame_count: Any, boundary_frames: Any, stage4_completion_frame: Any,
) -> list[str]:
    """Validate three manual/automatic inclusive ends and a completion frame.

    Returns an empty list on success, suitable for a web API's 422 details.
    ``bool`` is deliberately not accepted as an integer.
    """
    if type(frame_count) is not int or frame_count <= 0:
        return ["frame_count: expected positive integer"]
    errors: list[str] = []
    valid_boundaries = type(boundary_frames) is list and len(boundary_frames) == 3
    if not valid_boundaries:
        errors.append("boundary_frames: expected exactly three integers")
    elif any(type(frame) is not int for frame in boundary_frames):
        errors.append("boundary_frames: each boundary must be an integer")
        valid_boundaries = False
    if valid_boundaries:
        if any(frame < 0 or frame >= frame_count - 1 for frame in boundary_frames):
            errors.append("boundary_frames: each boundary must be in [0, frame_count - 2]")
        if not boundary_frames[0] < boundary_frames[1] < boundary_frames[2]:
            errors.append("boundary_frames: boundaries must be strictly increasing")
    valid_completion = type(stage4_completion_frame) is int
    if not valid_completion:
        errors.append("stage4_completion_frame: expected integer")
    elif not 0 <= stage4_completion_frame < frame_count:
        errors.append("stage4_completion_frame: outside original video")
    if valid_boundaries and valid_completion and stage4_completion_frame <= boundary_frames[2]:
        errors.append("stage4_completion_frame: must be after the third boundary")
    return errors


def _sample_index(samples: Any, frame_count: int, errors: list[str]) -> dict[str, dict]:
    if type(samples) is not list or not samples:
        errors.append("samples: expected nonempty list")
        return {}
    by_id: dict[str, dict] = {}
    by_seq: dict[int, dict] = {}
    for index, sample in enumerate(samples):
        path = f"samples[{index}]"
        if type(sample) is not dict:
            errors.append(f"{path}: expected object")
            continue
        sample_id = sample.get("sample_id")
        seq = sample.get("sample_seq")
        frame = sample.get("source_frame_id")
        timestamp = sample.get("source_timestamp_s")
        valid = True
        if type(sample_id) is not str or not sample_id.strip():
            errors.append(f"{path}.sample_id: expected nonempty string")
            valid = False
        elif sample_id in by_id:
            errors.append(f"{path}.sample_id: duplicate ID {sample_id!r}")
            valid = False
        if type(seq) is not int or seq < 0:
            errors.append(f"{path}.sample_seq: expected nonnegative integer")
            valid = False
        elif seq in by_seq:
            errors.append(f"{path}.sample_seq: duplicate sequence number {seq}")
            valid = False
        if type(frame) is not int or not 0 <= frame < frame_count:
            errors.append(f"{path}.source_frame_id: outside original video or not integer")
            valid = False
        if type(timestamp) not in (int, float) or not 0 <= timestamp < float("inf"):
            errors.append(f"{path}.source_timestamp_s: expected finite nonnegative number")
            valid = False
        if valid:
            by_id[sample_id] = sample
            by_seq[seq] = sample
    ordered = [by_seq[seq] for seq in sorted(by_seq)]
    for previous, current in zip(ordered, ordered[1:]):
        if current["sample_seq"] != previous["sample_seq"] + 1:
            errors.append("samples: sample_seq must be consecutive")
        if current["source_frame_id"] <= previous["source_frame_id"]:
            errors.append("samples: source_frame_id must increase with sample_seq")
        if current["source_timestamp_s"] <= previous["source_timestamp_s"]:
            errors.append("samples: source_timestamp_s must increase with sample_seq")
    return by_id


def normalize_annotation(model_output: dict, samples: list, manifest: dict) -> dict:
    """Turn one VLM response into a safe, reviewable annotation.

    Invalid or unresolved model output returns ``needs_review`` with a single
    stage=-1 segment; labels are never guessed. Known unsupported tasks return
    ``out_of_scope``. ``frame_count`` (alias ``num_frames``) is the only required
    manifest field.  Identity/provenance metadata can be added by the caller.
    """
    errors = validate_model_output(model_output)
    if type(manifest) is not dict:
        errors.append("manifest: expected object")
        manifest = {}
    frame_count = manifest.get("frame_count", manifest.get("num_frames"))
    if type(frame_count) is not int or frame_count <= 0:
        errors.append("manifest.frame_count: expected positive integer")
        frame_count = 0
    indexed = _sample_index(samples, frame_count, errors)
    response = model_output if type(model_output) is dict else {}
    task_type = response.get("task_type", "unknown")
    applicable = response.get("applicable")
    out_of_scope = (
        type(task_type) is str and task_type in TASK_TYPES - {"fold_cloth", "unknown"}
    ) or applicable is False
    # A task-type guess of unknown is never a definitive scope rejection.
    if task_type == "unknown":
        out_of_scope = False
    check_semantics = not out_of_scope
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

    boundaries: list[dict] = []
    proposals = response.get("boundary_proposals")
    proposals = proposals if type(proposals) is list else []
    for index, proposal in enumerate(proposals[:3]):
        if type(proposal) is not dict:
            continue
        left_id = proposal.get("last_not_complete_sample_id")
        right_id = proposal.get("first_complete_sample_id")
        left = indexed.get(left_id) if type(left_id) is str else None
        right = indexed.get(right_id) if type(right_id) is str else None
        if check_semantics:
            for side, sample_id, sample in (("last_not_complete", left_id, left), ("first_complete", right_id, right)):
                if sample is None:
                    errors.append(f"boundary[{index + 1}].{side}_sample_id: unresolved or unknown ID {sample_id!r}")
            if left and right and right["sample_seq"] != left["sample_seq"] + 1:
                errors.append(f"boundary[{index + 1}]: L/R samples must be adjacent in sample_seq")
        if left and right:
            first_frame = left["source_frame_id"] + 1
            end_frame = right["source_frame_id"]
            boundaries.append({
                "from_stage": index + 1,
                "to_stage": index + 2,
                "last_not_complete_sample_id": left_id,
                "first_complete_sample_id": right_id,
                "last_not_complete_source_frame_id": left["source_frame_id"],
                "first_complete_source_frame_id": end_frame,
                "last_not_complete_timestamp_s": left["source_timestamp_s"],
                "first_complete_timestamp_s": right["source_timestamp_s"],
                "coarse_end_frame": end_frame,
                "next_stage_start_frame": end_frame + 1,
                "estimated_boundary_interval_frames": [first_frame, end_frame],
                "evidence_text": proposal.get("evidence_text", ""),
            })

    completion_raw = response.get("stage4_completion")
    completion_raw = completion_raw if type(completion_raw) is dict else {}
    completion_id = completion_raw.get("completion_sample_id")
    completion_sample = indexed.get(completion_id) if type(completion_id) is str else None
    completion_frame = completion_sample["source_frame_id"] if completion_sample else None
    completion = {
        "observed": completion_raw.get("observed") is True,
        "completion_sample_id": completion_id if type(completion_id) is str else None,
        "evidence_text": completion_raw.get("evidence_text", ""),
        "source_frame_id": completion_frame,
        "source_timestamp_s": completion_sample["source_timestamp_s"] if completion_sample else None,
    }
    candidate_frames = [boundary["coarse_end_frame"] for boundary in boundaries]
    if check_semantics:
        if completion_raw.get("observed") is not True:
            errors.append("stage4_completion.observed: final stage completion must be observed")
        if completion_sample is None:
            errors.append(f"stage4_completion.completion_sample_id: unresolved or unknown ID {completion_id!r}")
        errors.extend(validate_review_frames(frame_count, candidate_frames, completion_frame))
        if len(boundaries) == 3 and completion_sample:
            right_id = boundaries[2]["first_complete_sample_id"]
            if completion_sample["sample_seq"] <= indexed[right_id]["sample_seq"]:
                errors.append("stage4_completion: completion sample must be after third boundary sample")
    errors = list(dict.fromkeys(errors))
    status = "needs_review" if errors else ("out_of_scope" if out_of_scope else "accepted")
    accepted_frames = candidate_frames if status == "accepted" else []
    if status == "accepted":
        starts = [0] + [frame + 1 for frame in accepted_frames]
        ends = accepted_frames + [frame_count - 1]
        segments = [
            {"stage": stage, "start_frame": start, "end_frame": end}
            for stage, (start, end) in enumerate(zip(starts, ends), start=1)
        ]
    else:
        segments = [{"stage": -1, "start_frame": 0, "end_frame": frame_count - 1}] if frame_count else []
    raw_reasons = response.get("review_reasons")
    raw_reasons = raw_reasons if type(raw_reasons) is list else []
    review_reasons = [reason for reason in raw_reasons if type(reason) is str]
    return {
        "schema_version": "1.0",
        "spec_version": SPEC_VERSION,
        "precision": "sampled_1fps",
        "frame_count": frame_count,
        "status": status,
        "task_type": task_type if type(task_type) is str and task_type in TASK_TYPES else "unknown",
        "applicable": applicable if type(applicable) is bool else None,
        "all_stages_observed": response.get("all_stages_observed") is True,
        "needs_review": status == "needs_review",
        "review_reasons": list(dict.fromkeys(review_reasons + errors)),
        "machine_review_reasons": errors,
        "out_of_scope_reason": "task type or applicability is outside the four-stage cloth specification" if status == "out_of_scope" else None,
        "stage4_completion": completion,
        "stage4_completion_frame": completion_frame,
        "boundaries": boundaries,
        "boundary_frames": accepted_frames,
        "segments": segments,
        "model_output": deepcopy(model_output),
    }


def expand_labels(annotation: dict):
    """Return int64 labels for every original frame (requires NumPy).

    Only accepted annotations produce stage labels.  Unknown/review/scope
    results always remain -1, regardless of any retained boundary proposals.
    Invalid accepted segments raise ValueError rather than silently losing
    frames.  NumPy is imported lazily so contract validation is stdlib-only.
    """
    import numpy as np

    frame_count = annotation.get("frame_count")
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("annotation.frame_count must be a nonnegative integer")
    labels = np.full(frame_count, -1, dtype=np.int64)
    if annotation.get("status") != "accepted":
        return labels
    segments = annotation.get("segments")
    if type(segments) is not list or len(segments) != 4:
        raise ValueError("accepted annotation must contain exactly four segments")
    cursor = 0
    for expected_stage, segment in enumerate(segments, start=1):
        if type(segment) is not dict:
            raise ValueError("segment must be an object")
        stage = segment.get("stage")
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        if any(type(value) is not int for value in (stage, start, end)):
            raise ValueError("stage and segment frame indices must be integers")
        if stage != expected_stage or start != cursor or not start <= end < frame_count:
            raise ValueError("segments must cover the video once in stage order")
        labels[start:end + 1] = stage
        cursor = end + 1
    if cursor != frame_count:
        raise ValueError("segments do not cover all original frames")
    return labels
