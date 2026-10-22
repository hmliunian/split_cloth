"""Fixed interval boundaries estimated from measured gripper feedback.

Section 5.2 calls for the first *fully released* frame, not the start of an
opening command. These nodes therefore backdate a confirmed feedback plateau
to its first frame. An episode-relative plateau is still only an estimate of
physical release: the raw evidence and that limitation travel with each node.
"""
from __future__ import annotations

from copy import deepcopy

from .gripper_events import (
    ARM_STATE_INDICES,
    DEFAULT_CANDIDATE_LEVELS,
    _finite_float,
    _read_measured_parquet,
    _validate_axis,
    detect_gripper_event_candidates,
)


DEFAULT_NODE_LEVELS = ((.25, .1),) + DEFAULT_CANDIDATE_LEVELS + ((.98, .9),)


def _positive_count(value, name):
    if isinstance(value, bool) or int(value) != value or value < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    return int(value)


def _initial_status(values, times, indices, calibration, samples, duration, tolerance):
    """Only stable, calibrated episode endpoints establish an initial state."""
    if calibration["status"] != "calibrated" or len(values) < samples:
        return None
    stop = samples - 1
    while stop < len(values) and times[stop] - times[0] + 1e-9 < duration:
        stop += 1
    if stop >= len(values):
        return None
    window = values[:stop + 1]
    if max(window) - min(window) > tolerance + 1e-12:
        return None
    raw = sum(window) / len(window)
    score = (raw - calibration["quantile_low_raw"]) / calibration["dynamic_range_raw"]
    if calibration["opening_direction"] == "lower":
        score = 1 - score
    kind = "open" if score >= .9 else "close" if score <= .1 else None
    return {"kind": kind, "raw_plateau_value": raw,
            "confirmed_at_source_frame_id": indices[stop]} if kind else None


def _settle_candidate(candidate, values, times, indices, *, direction, tolerance,
                      reversal_tolerance, minimum_excursion,
                      samples, duration, search_window, cutoff):
    """Confirm a plateau while allowing bounded actuator overshoot/settling.

    Plateau noise and motion reversal use separate tolerances. A small retreat
    after an overshoot is normal settling, but the final plateau must retain a
    directional excursion from the earliest crossing's preceding measurement.
    An opposite crossing supplies an additional hard limit on the search.
    """
    index_to_row = {frame: row for row, frame in enumerate(indices)}
    onset = index_to_row[candidate["source_frame_id"]]
    first_crossing = index_to_row[candidate["candidate_group_start_source_frame_id"]]
    reference_row = max(0, first_crossing - 1)
    reference_value = values[reference_row]
    extreme = values[onset]
    limit_time = times[onset] + search_window
    for start in range(onset, len(values)):
        if times[start] > limit_time + 1e-9 or (cutoff is not None and indices[start] >= cutoff):
            break
        extreme = max(extreme, values[start]) if direction == "higher" else min(extreme, values[start])
        retreat = extreme - values[start] if direction == "higher" else values[start] - extreme
        if retreat > reversal_tolerance + 1e-12:
            break
        stop = start + samples - 1
        while stop < len(values) and times[stop] - times[start] + 1e-9 < duration:
            stop += 1
        if stop >= len(values) or times[stop] > limit_time + 1e-9:
            break
        if cutoff is not None and indices[stop] >= cutoff:
            break
        window = values[start:stop + 1]
        if max(window) - min(window) > tolerance + 1e-12:
            continue
        retained_excursion = (min(window) - reference_value if direction == "higher"
                              else reference_value - max(window))
        if retained_excursion + 1e-12 < minimum_excursion:
            continue
        return {"source_frame_id": indices[start], "timestamp_s": times[start],
                "confirmed_at_source_frame_id": indices[stop],
                "confirmed_at_timestamp_s": times[stop],
                "raw_plateau_value": sum(window) / len(window),
                "raw_plateau_values": window,
                "plateau_source_frame_ids": indices[start:stop + 1],
                "excursion_reference_source_frame_id": indices[reference_row],
                "excursion_reference_raw_value": reference_value,
                "retained_directional_excursion_raw": retained_excursion,
                "retreat_from_extreme_raw": retreat,
                "stability_duration_s": times[stop] - times[start]}
    return None


def detect_gripper_nodes(
    states, timestamps, frame_indices, *, source_timestamps=None,
    opening_directions=None, candidate_levels=DEFAULT_NODE_LEVELS,
    grouping_window_s=.25, settled_samples=3, settled_duration_s=.10,
    settled_tolerance_fraction=.02, settled_search_window_s=2.0,
    settled_reversal_fraction=.10, settled_min_excursion_fraction=.10,
    **detector_options,
):
    """Return immutable open/close nodes for interval-based stage selection.

    A source PTS axis enforces complete one-row-per-video-frame feedback. Missing
    states, frame gaps and shifted timestamps are rejected, never interpolated.
    Without the source axis the result is explicitly unverified. Open nodes are
    *eligible* stage-end candidates; closing nodes split context intervals only.
    The caller must additionally verify the stage operation and participating
    arms, and must not let the model replace a node's original frame ID.
    """
    samples = _positive_count(settled_samples, "settled_samples")
    duration = _finite_float(settled_duration_s, "settled_duration_s")
    fraction = _finite_float(settled_tolerance_fraction, "settled_tolerance_fraction")
    reversal_fraction = _finite_float(settled_reversal_fraction, "settled_reversal_fraction")
    excursion_fraction = _finite_float(settled_min_excursion_fraction, "settled_min_excursion_fraction")
    search_window = _finite_float(settled_search_window_s, "settled_search_window_s")
    if duration < 0 or not 0 <= fraction < .25 or search_window <= 0:
        raise ValueError("require settled_duration_s >= 0, 0 <= settled_tolerance_fraction < .25, "
                         "and settled_search_window_s > 0")
    if search_window < duration:
        raise ValueError("settled_search_window_s must cover settled_duration_s")
    if not 0 <= reversal_fraction <= 1 or not 0 < excursion_fraction <= 1:
        raise ValueError("require 0 <= settled_reversal_fraction <= 1 and "
                         "0 < settled_min_excursion_fraction <= 1")
    times, indices, alignment = _validate_axis(
        timestamps, frame_indices, source_timestamps,
        detector_options.get("timestamp_tolerance_s"))
    if len(states) != len(times) or not times:
        raise ValueError("states, timestamps and frame_indices must have the same nonzero length")
    if any(b != a + 1 for a, b in zip(indices, indices[1:])):
        raise ValueError("Node detection requires consecutive source frame indices")
    if source_timestamps is not None and indices != list(range(len(source_timestamps))):
        raise ValueError("Node detection requires measured state for every source video frame")
    measured = []
    for row, state in enumerate(states):
        try:
            if state is None or len(state) != 14:
                raise ValueError("expected 14 components")
            measured.append([_finite_float(value, "state") for value in state])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Node detection requires finite 14D measured state at row {row}") from exc

    directions = {"left": "higher", "right": "higher"}
    if opening_directions:
        directions.update(opening_directions)
    # The existing detector validates direction names, raw scales and timing.
    opening = detect_gripper_event_candidates(
        measured, times, indices, source_timestamps=source_timestamps,
        opening_directions=directions, levels=candidate_levels,
        grouping_window_s=grouping_window_s, **detector_options)
    reverse = {arm: "lower" if value == "higher" else "higher"
               for arm, value in directions.items()}
    closing = detect_gripper_event_candidates(
        measured, times, indices, source_timestamps=source_timestamps,
        opening_directions=reverse, levels=candidate_levels,
        grouping_window_s=grouping_window_s, **detector_options)
    calibration = deepcopy(opening["calibration"])
    calibration.update({
        "method": "multithreshold_crossings_then_local_settled_plateaus",
        "interpretation": "feedback_settled_estimate",
        "physical_full_open_calibrated": False,
        "settled_samples": samples, "settled_duration_s": duration,
        "settled_tolerance_fraction": fraction, "settled_search_window_s": search_window,
        "settled_reversal_fraction": reversal_fraction,
        "settled_min_excursion_fraction": excursion_fraction,
        "reversal_tolerance": "max(reversal_fraction * dynamic_range, 3 * plateau_tolerance_raw)",
        "boundary_frame": "first_frame_of_confirmed_plateau_not_confirmation_frame",
        "same_frame_arm_merge_only": True,
        "initial_states": {},
    })
    candidates = [{**deepcopy(item), "transition_kind": kind}
                  for kind, result in (("open", opening), ("close", closing))
                  for item in result["events"]]
    candidates.sort(key=lambda item: (item["source_frame_id"], item["arm"], item["transition_kind"]))
    diagnostics = deepcopy(opening["diagnostics"])
    diagnostics.append({"code": "settled_feedback_is_not_physical_full_open",
                        "message": "A stable raw-feedback plateau estimates release timing; "
                                   "episode quantiles and assumed directions do not calibrate physical full opening."})
    settled_changes = []
    values_by_arm = {}
    tolerances = {}
    for arm, state_index in ARM_STATE_INDICES.items():
        arm_calibration = calibration["arms"][arm]
        values = [state[state_index] for state in measured]
        values_by_arm[arm] = values
        tolerance = arm_calibration.get("dynamic_range_raw", 0) * fraction
        reversal_tolerance = max(arm_calibration.get("dynamic_range_raw", 0) * reversal_fraction,
                                 3 * tolerance)
        minimum_excursion = arm_calibration.get("dynamic_range_raw", 0) * excursion_fraction
        tolerances[arm] = tolerance
        initial = _initial_status(values, times, indices, arm_calibration, samples, duration, tolerance)
        calibration["initial_states"][arm] = initial
        arm_candidates = [item for item in candidates if item["arm"] == arm]
        accepted = []
        for candidate in arm_candidates:
            kind = candidate["transition_kind"]
            opposite = [item["candidate_group_start_source_frame_id"] for item in arm_candidates
                        if item["transition_kind"] != kind
                        and item["candidate_group_start_source_frame_id"] > candidate["source_frame_id"]]
            cutoff = min(opposite, default=None)
            direction = directions[arm] if kind == "open" else reverse[arm]
            plateau = _settle_candidate(
                candidate, values, times, indices, direction=direction,
                tolerance=tolerance, reversal_tolerance=reversal_tolerance,
                minimum_excursion=minimum_excursion, samples=samples, duration=duration,
                search_window=search_window, cutoff=cutoff)
            if plateau is None:
                diagnostics.append({"code": "crossing_without_settled_plateau", "arm": arm,
                                    "kind": kind, "candidate_event_id": candidate["event_id"],
                                    "crossing_source_frame_id": candidate["source_frame_id"]})
                continue
            # Multi-threshold groups can converge on the same plateau.
            duplicate = next((item for item in accepted
                              if item["kind"] == kind and item["source_frame_id"] == plateau["source_frame_id"]), None)
            if duplicate:
                duplicate["evidence"]["crossing_candidates"].append(candidate)
                continue
            change = {"arm": arm, "kind": kind, **plateau,
                      "evidence": {"interpretation": "feedback_settled_estimate",
                                   "state_index": state_index, "opening_direction": directions[arm],
                                   "direction_is_assumed": True,
                                   "raw_units": "unspecified", "plateau_tolerance_raw": tolerance,
                                   "reversal_tolerance_raw": reversal_tolerance,
                                   "minimum_directional_excursion_raw": minimum_excursion,
                                   "physical_full_open_calibrated": False,
                                   "crossing_candidates": [candidate]}}
            accepted.append(change)
        settled_changes.extend(accepted)
    settled_changes.sort(key=lambda item: (item["source_frame_id"], item["arm"], item["kind"]))
    grouped = {}
    for change in settled_changes:
        grouped.setdefault(change["source_frame_id"], []).append(change)
    nodes = []
    for frame, changes in sorted(grouped.items()):
        row = frame - indices[0]
        arm_states = {}
        for arm in ARM_STATE_INDICES:
            previous = [item for item in settled_changes
                        if item["arm"] == arm and item["source_frame_id"] <= frame]
            state = previous[-1] if previous else calibration["initial_states"][arm]
            within_plateau = (state is not None and
                              abs(values_by_arm[arm][row] - state["raw_plateau_value"])
                              <= tolerances[arm] + 1e-12)
            arm_states[arm] = ("estimated_open" if state["kind"] == "open" else "estimated_closed") if within_plateau else "unknown"
        nodes.append({
            "node_id": f"a{len(nodes) + 1:04d}", "source_frame_id": frame,
            "timestamp_s": times[row], "changes": changes,
            "arm_states": arm_states,
            "open_arms": [arm for arm, state in arm_states.items() if state == "estimated_open"],
            "closed_arms": [arm for arm, state in arm_states.items() if state == "estimated_closed"],
            "unknown_arms": [arm for arm, state in arm_states.items() if state == "unknown"],
            "eligible_stage_end": any(item["kind"] == "open" for item in changes),
            "evidence": {"interpretation": "feedback_settled_estimate",
                         "physical_full_open_calibrated": False,
                         "alignment_verified": alignment["alignment_verified"],
                         "frame_id_fixed_by_feedback": True},
        })
    return {"nodes": nodes, "calibration": calibration, "diagnostics": diagnostics,
            "alignment_verified": alignment["alignment_verified"]}


def prepare_gripper_nodes(parquet_path, source_timestamps, *, state_column=None,
                          timestamp_column="timestamp", frame_index_column="frame_index", **options):
    """Read measured state once, verify the entire video axis and build nodes."""
    if source_timestamps is None:
        raise ValueError("prepare_gripper_nodes requires decoded source_timestamps")
    table, measured_column = _read_measured_parquet(
        parquet_path, state_column, timestamp_column, frame_index_column)
    result = detect_gripper_nodes(
        table[measured_column], table[timestamp_column], table[frame_index_column],
        source_timestamps=source_timestamps, **options)
    result["calibration"].update({"parquet_path": str(parquet_path), "state_column": measured_column})
    return result
