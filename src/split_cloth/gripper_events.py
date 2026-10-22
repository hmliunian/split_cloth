"""Conservative, episode-local *candidate* gripper opening events.

Feedback in state[6] and state[13] is used, never the commanded action.
An open-threshold crossing is only a proposal for visual review: neither a
physical release nor completion of a clothing stage follows from this signal.
All values and thresholds remain in the dataset's raw units.
"""
from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path


ARM_STATE_INDICES = {"left": 6, "right": 13}
DEFAULT_CANDIDATE_LEVELS = ((.5, .25), (.75, .45), (.9, .65))


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _validate_axis(timestamps, frame_indices, source_timestamps, tolerance):
    times = [_finite_float(v, "timestamps") for v in timestamps]
    indices = []
    for value in frame_indices:
        number = _finite_float(value, "frame_indices")
        if number < 0 or not number.is_integer():
            raise ValueError("frame_indices must contain nonnegative integers")
        indices.append(int(number))
    if len(times) != len(indices):
        raise ValueError("timestamps and frame_indices must have the same length")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing within one episode")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("frame_indices must be strictly increasing within one episode")
    alignment = {"alignment_verified": False, "row_count": len(times),
                 "mapping": "explicit_frame_index", "timestamp_units": "seconds"}
    if source_timestamps is None:
        return times, indices, alignment
    source = [_finite_float(v, "source_timestamps") for v in source_timestamps]
    if not source or any(b <= a for a, b in zip(source, source[1:])):
        raise ValueError("source_timestamps must be a nonempty, strictly increasing video PTS axis")
    if indices and indices[-1] >= len(source):
        raise ValueError("parquet frame_index is outside the decoded video PTS axis")
    # A small fraction of a frame, not half a frame: a one-frame shift must fail.
    if tolerance is None:
        intervals = [b - a for a, b in zip(source, source[1:])]
        tolerance = max(1e-4, min(intervals) * .05) if intervals else 1e-4
    tolerance = _finite_float(tolerance, "timestamp_tolerance_s")
    if tolerance < 0:
        raise ValueError("timestamp_tolerance_s must be nonnegative")
    errors = [times[row] - source[index] for row, index in enumerate(indices)]
    mismatches = [{"row": row, "source_frame_id": indices[row],
                   "parquet_timestamp_s": times[row],
                   "video_timestamp_s": source[indices[row]], "error_s": error}
                  for row, error in enumerate(errors) if abs(error) > tolerance]
    if mismatches:
        raise ValueError(f"Parquet/video PTS alignment failed ({len(mismatches)} rows; "
                         f"tolerance={tolerance:g}s): {mismatches[:3]}")
    alignment.update({"alignment_verified": True, "source_frame_count": len(source),
                      "timestamp_tolerance_s": tolerance,
                      "max_abs_timestamp_error_s": max(map(abs, errors), default=0.0),
                      "median_signed_timestamp_error_s": _quantile(errors, .5) if errors else None})
    return times, indices, alignment


def detect_gripper_events(
    states: Sequence,
    timestamps: Sequence,
    frame_indices: Sequence,
    *,
    source_timestamps: Sequence | None = None,
    timestamp_tolerance_s: float | None = None,
    stable_samples: int = 2,
    stable_duration_s: float = .05,
    open_fraction: float = .9,
    rearm_fraction: float = .65,
    quantile_low: float = .05,
    quantile_high: float = .95,
    min_dynamic_range: float = 1e-6,
    min_dynamic_fraction: float = .01,
    opening_directions: Mapping[str, str] | None = None,
):
    """Find every supported opening candidate in a single episode.

    ``frame_indices`` are mandatory original frame IDs, not row numbers.
    ``source_timestamps`` must contain every decoded video PTS in presentation
    order, starting at frame 0. Supplying it verifies each row's explicit ID and
    timestamp, and rejects shifted or out-of-range axes. Without it output is
    explicitly unverified and must not be used for precise frame annotation.

    Each arm uses robust episode quantiles, hysteresis, and consecutive stable
    open feedback. The candidate is backdated to the *first* high-threshold
    sample; the later confirmation frame is kept separately. Initial open
    states, nonfinite/missing rows, and gaps cannot create an opening event.
    The default higher-is-open direction is an explicit configurable assumption
    requiring visual validation, not a fact inferred from endpoint values.
    """
    if isinstance(stable_samples, bool) or int(stable_samples) != stable_samples or stable_samples < 1:
        raise ValueError("stable_samples must be a positive integer")
    stable_samples = int(stable_samples)
    stable_duration_s = _finite_float(stable_duration_s, "stable_duration_s")
    if stable_duration_s < 0:
        raise ValueError("stable_duration_s must be nonnegative")
    if not 0 <= rearm_fraction < open_fraction <= 1:
        raise ValueError("require 0 <= rearm_fraction < open_fraction <= 1")
    if not 0 <= quantile_low < quantile_high <= 1:
        raise ValueError("require 0 <= quantile_low < quantile_high <= 1")
    min_dynamic_range = _finite_float(min_dynamic_range, "min_dynamic_range")
    min_dynamic_fraction = _finite_float(min_dynamic_fraction, "min_dynamic_fraction")
    if min_dynamic_range < 0 or min_dynamic_fraction < 0:
        raise ValueError("dynamic range minima must be nonnegative")
    directions = {"left": "higher", "right": "higher"}
    if opening_directions:
        if set(opening_directions) - set(ARM_STATE_INDICES):
            raise ValueError("opening_directions keys must be left and/or right")
        directions.update(opening_directions)
    if any(direction not in {"higher", "lower"} for direction in directions.values()):
        raise ValueError("opening direction must be higher or lower")
    times, indices, alignment = _validate_axis(
        timestamps, frame_indices, source_timestamps, timestamp_tolerance_s)
    if len(states) != len(times):
        raise ValueError("states, timestamps and frame_indices must have equal lengths")

    diagnostics = []
    if not alignment["alignment_verified"]:
        diagnostics.append({"code": "unverified_video_alignment",
                            "message": "No decoded source PTS supplied; row-to-video mapping is unverified."})
    valid_states = []
    invalid_rows = []
    for row_index, row in enumerate(states):
        try:
            if row is None or len(row) != 14:
                raise ValueError("state must have 14 components")
            values = [_finite_float(v, "state") for v in row]
        except (TypeError, ValueError, OverflowError):
            valid_states.append(None)
            invalid_rows.append(row_index)
        else:
            valid_states.append(values)
    if invalid_rows:
        diagnostics.append({"code": "missing_or_nonfinite_state", "count": len(invalid_rows),
                            "first_rows": invalid_rows[:20],
                            "message": "Invalid 14D feedback rows break event continuity; no action fallback."})
    gap_rows = [row for row in range(1, len(indices)) if indices[row] != indices[row - 1] + 1]
    if gap_rows:
        diagnostics.append({"code": "nonconsecutive_source_frames", "count": len(gap_rows),
                            "first_rows": gap_rows[:20],
                            "message": "Frame gaps break opening and confirmation continuity."})
    calibration = {**alignment, "state_dimensions": 14,
                   "signal_source": "measured_state_only", "raw_units": "unspecified",
                   "quantiles": [quantile_low, quantile_high],
                   "stable_samples": stable_samples, "stable_duration_s": stable_duration_s,
                   "open_fraction": open_fraction, "rearm_fraction": rearm_fraction,
                   "min_dynamic_range_raw": min_dynamic_range,
                   "min_dynamic_fraction": min_dynamic_fraction,
                   "valid_state_rows": len(states) - len(invalid_rows),
                   "invalid_state_rows": len(invalid_rows), "arms": {},
                   "interpretation": "feedback_opening_candidates_require_visual_release_and_stage_confirmation"}
    events = []
    for arm, state_index in ARM_STATE_INDICES.items():
        values = [row[state_index] if row is not None else None for row in valid_states]
        finite_values = [v for v in values if v is not None]
        arm_calibration = {"state_index": state_index, "opening_direction": directions[arm],
                           "direction_is_assumed": True, "finite_count": len(finite_values),
                           "status": "missing_feedback"}
        calibration["arms"][arm] = arm_calibration
        if not finite_values:
            diagnostics.append({"code": "missing_feedback", "arm": arm})
            continue
        low = _quantile(finite_values, quantile_low)
        high = _quantile(finite_values, quantile_high)
        dynamic_range = high - low
        required_range = max(min_dynamic_range, min_dynamic_fraction * max(abs(low), abs(high)))
        arm_calibration.update({"observed_min_raw": min(finite_values),
                                "observed_max_raw": max(finite_values),
                                "quantile_low_raw": low, "quantile_high_raw": high,
                                "dynamic_range_raw": dynamic_range,
                                "required_dynamic_range_raw": required_range})
        if len(finite_values) < max(3, stable_samples) or dynamic_range <= required_range:
            arm_calibration["status"] = "insufficient_dynamic_range"
            diagnostics.append({"code": "insufficient_dynamic_range", "arm": arm,
                                "message": "Cannot calibrate open/closed endpoints from this episode."})
            continue
        higher = directions[arm] == "higher"
        raw_threshold = lambda fraction: low + dynamic_range * (fraction if higher else 1 - fraction)
        arm_calibration.update({"status": "calibrated", "open_threshold_raw": raw_threshold(open_fraction),
                                "rearm_threshold_raw": raw_threshold(rearm_fraction)})
        armed = False
        candidate_rows = []
        for row, value in enumerate(values):
            if value is None or (row and indices[row] != indices[row - 1] + 1):
                armed = False
                candidate_rows = []
            if value is None:
                continue
            score = (value - low) / dynamic_range
            if not higher:
                score = 1 - score
            if score <= rearm_fraction:
                armed = True
                candidate_rows = []
            elif armed and score >= open_fraction:
                candidate_rows.append(row)
                onset = candidate_rows[0]
                if (len(candidate_rows) >= stable_samples and
                        times[row] - times[onset] + 1e-9 >= stable_duration_s):
                    previous = onset - 1
                    evidence = {"state_index": state_index, "raw_onset_value": values[onset],
                                "raw_confirmation_values": [values[i] for i in candidate_rows],
                                "confirmation_source_frame_ids": [indices[i] for i in candidate_rows],
                                "open_threshold_raw": arm_calibration["open_threshold_raw"],
                                "rearm_threshold_raw": arm_calibration["rearm_threshold_raw"],
                                "previous_source_frame_id": indices[previous] if previous >= 0 else None,
                                "previous_raw_value": values[previous] if previous >= 0 else None,
                                "stability_duration_s": times[row] - times[onset]}
                    events.append({"event_id": f"gripper_{arm}_{indices[onset]:06d}",
                                   "arm": arm, "source_frame_id": indices[onset],
                                   "timestamp_s": times[onset],
                                   "confirmed_at_source_frame_id": indices[row],
                                   "confirmed_at_timestamp_s": times[row],
                                   "kind": "feedback_open_threshold_candidate", "evidence": evidence})
                    armed = False
                    candidate_rows = []
            else:
                candidate_rows = []
    events.sort(key=lambda event: (event["source_frame_id"], event["arm"]))
    return {"alignment_verified": alignment["alignment_verified"], "calibration": calibration,
            "events": events, "diagnostics": diagnostics}


def _read_measured_parquet(path, state_column, timestamp_column, frame_index_column):
    import pyarrow.parquet as pq

    path = Path(path)
    names = pq.read_schema(path).names
    if state_column is None:
        candidates = [name for name in ("observation.state", "state") if name in names]
        if len(candidates) != 1:
            raise ValueError("Expected one measured state column; pass state_column explicitly")
        state_column = candidates[0]
    if state_column not in {"observation.state", "state"}:
        raise ValueError("Only measured observation.state/state may be used, never actions")
    columns = [state_column, timestamp_column, frame_index_column]
    missing = [name for name in columns if name not in names]
    if missing:
        raise ValueError(f"Missing required measured-state/timing columns: {missing}")
    table = pq.read_table(path, columns=columns).to_pydict()
    return table, state_column


def detect_gripper_events_from_parquet(path, *, source_timestamps=None,
                                      state_column=None, timestamp_column="timestamp",
                                      frame_index_column="frame_index", **options):
    """Read only measured state and explicit timing/index columns from Parquet.

    Auto-detection accepts exactly one of ``observation.state`` and ``state``;
    action/command columns are deliberately never read or substituted.
    """
    table, state_column = _read_measured_parquet(path, state_column, timestamp_column, frame_index_column)
    result = detect_gripper_events(table[state_column], table[timestamp_column],
                                  table[frame_index_column], source_timestamps=source_timestamps,
                                  **options)
    result["calibration"]["parquet_path"] = str(path)
    result["calibration"]["state_column"] = state_column
    return result


def detect_gripper_event_candidates(
    states, timestamps, frame_indices, *, source_timestamps=None,
    levels=DEFAULT_CANDIDATE_LEVELS, grouping_window_s=.25, **base_options,
):
    """Increase recall with several raw feedback thresholds, never physical truth.

    Each ``(open_fraction, rearm_fraction)`` level runs the existing strict
    feedback detector. Same-arm crossings within ``grouping_window_s`` of the
    group's *first* crossing form one proposal, anchored at the highest level
    actually observed in that group. Grouping is not transitive. A repeated
    crossing of the same level begins an independent group, preserving rapid
    regrip cycles rather than silently swallowing their separate events.

    Lower plateaus can be real openings even when a later opening sets a much
    higher episode quantile. Therefore all independent groups are retained,
    including groups with only a low threshold. Vision must confirm whether
    the participating gripper released and whether the operation completed.
    Base ``open_fraction``/``rearm_fraction`` options are superseded by levels;
    every other calibration, stability and axis option is passed unchanged.
    """
    window = _finite_float(grouping_window_s, "grouping_window_s")
    if window < 0:
        raise ValueError("grouping_window_s must be nonnegative")
    parsed_levels = []
    try:
        for level in levels:
            if len(level) != 2:
                raise ValueError("Each candidate level must be (open_fraction, rearm_fraction)")
            opening = _finite_float(level[0], "candidate open fraction")
            rearm = _finite_float(level[1], "candidate rearm fraction")
            if not 0 <= rearm < opening <= 1:
                raise ValueError("Each candidate level requires 0 <= rearm < open <= 1")
            parsed_levels.append((opening, rearm))
    except TypeError as exc:
        raise ValueError("levels must be a nonempty sequence of threshold pairs") from exc
    if not parsed_levels or len(set(parsed_levels)) != len(parsed_levels):
        raise ValueError("Candidate levels must be nonempty and unique")
    if len({opening for opening, _ in parsed_levels}) != len(parsed_levels):
        raise ValueError("Candidate open fractions must be unique")
    parsed_levels.sort()
    options = dict(base_options)
    options.pop("open_fraction", None)
    options.pop("rearm_fraction", None)
    results = []
    components = []
    for level_index, (opening, rearm) in enumerate(parsed_levels):
        result = detect_gripper_events(
            states, timestamps, frame_indices, source_timestamps=source_timestamps,
            open_fraction=opening, rearm_fraction=rearm, **options)
        results.append(result)
        for event in result["events"]:
            components.append({**event, "level_index": level_index,
                               "open_fraction": opening, "rearm_fraction": rearm})
    primary = results[-1]
    calibration = deepcopy(primary["calibration"])
    calibration["ensemble"] = {
        "method": "multiple_raw_feedback_thresholds",
        "levels": [{"open_fraction": opening, "rearm_fraction": rearm,
                    "event_count": len(result["events"])}
                   for (opening, rearm), result in zip(parsed_levels, results)],
        "primary_open_fraction": parsed_levels[-1][0],
        "grouping_window_s": window,
        "grouping_reference": "first_crossing_not_transitive",
        "repeated_same_level_starts_independent_group": True,
        "anchor": "highest_observed_open_fraction_within_group",
        "level_calibrations": [{"open_fraction": opening,
                                 "arms": result["calibration"]["arms"]}
                                for (opening, _), result in zip(parsed_levels, results)],
    }
    calibration["interpretation"] = "multithreshold_candidates_not_physical_release_or_stage_completion"
    groups = []
    for arm in ARM_STATE_INDICES:
        arm_components = sorted((event for event in components if event["arm"] == arm),
                                key=lambda event: (event["timestamp_s"], event["level_index"]))
        group = []
        for event in arm_components:
            if group and (event["timestamp_s"] - group[0]["timestamp_s"] > window + 1e-9
                          or event["level_index"] in {item["level_index"] for item in group}):
                groups.append(group)
                group = []
            group.append(event)
        if group:
            groups.append(group)
    events = []
    for group in groups:
        anchor = max(group, key=lambda event: event["level_index"])
        first = group[0]
        event = {key: deepcopy(value) for key, value in anchor.items()
                 if key not in {"level_index", "open_fraction", "rearm_fraction"}}
        event.update({
            "event_id": f"gripper_{first['arm']}_{first['source_frame_id']:06d}",
            "kind": "multithreshold_feedback_opening_candidate",
            "anchor_level": anchor["open_fraction"],
            "levels_detected": sorted(item["open_fraction"] for item in group),
            "candidate_group_start_source_frame_id": first["source_frame_id"],
            "candidate_group_start_timestamp_s": first["timestamp_s"],
            "level_evidence": [deepcopy(item) for item in group],
        })
        events.append(event)
    events.sort(key=lambda event: (event["source_frame_id"], event["arm"], event["event_id"]))
    diagnostics = deepcopy(primary["diagnostics"])
    diagnostics.append({"code": "multithreshold_feedback_is_not_physical_truth",
                        "message": "All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."})
    return {"alignment_verified": primary["alignment_verified"], "calibration": calibration,
            "events": events, "diagnostics": diagnostics}


def detect_gripper_event_candidates_from_parquet(
    path, *, source_timestamps=None, state_column=None, timestamp_column="timestamp",
    frame_index_column="frame_index", levels=DEFAULT_CANDIDATE_LEVELS,
    grouping_window_s=.25, **base_options,
):
    """Read measured feedback once and return all multi-threshold candidates."""
    table, state_column = _read_measured_parquet(path, state_column, timestamp_column, frame_index_column)
    result = detect_gripper_event_candidates(
        table[state_column], table[timestamp_column], table[frame_index_column],
        source_timestamps=source_timestamps, levels=levels,
        grouping_window_s=grouping_window_s, **base_options)
    result["calibration"]["parquet_path"] = str(path)
    result["calibration"]["state_column"] = state_column
    return result
