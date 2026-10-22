import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from split_cloth.gripper_events import (
    detect_gripper_event_candidates, detect_gripper_event_candidates_from_parquet,
    detect_gripper_events, detect_gripper_events_from_parquet,
)


def episode(left, right=None, frame_indices=None):
    right = [0.0] * len(left) if right is None else right
    states = []
    for lvalue, rvalue in zip(left, right):
        row = [0.0] * 14
        row[6], row[13] = lvalue, rvalue
        states.append(row)
    indices = list(range(len(left))) if frame_indices is None else frame_indices
    times = [index / 20 for index in indices]
    source = [index / 20 for index in range(max(indices, default=0) + 1)]
    return states, times, indices, source


def detect(left, right=None, frame_indices=None, **options):
    states, times, indices, source = episode(left, right, frame_indices)
    return detect_gripper_events(states, times, indices, source_timestamps=source, **options)


class TestGripperEvents(unittest.TestCase):
    def test_confirmation_backdates_to_first_open_frame(self):
        result = detect([0] * 5 + [1] * 4)
        event = result["events"][0]
        self.assertEqual(event["source_frame_id"], 5)
        self.assertEqual(event["confirmed_at_source_frame_id"], 6)
        self.assertEqual(event["timestamp_s"], .25)
        self.assertTrue(result["alignment_verified"])
        json.dumps(result, allow_nan=False)

    def test_all_openings_retained_and_arms_independent(self):
        values = ([0] * 4 + [1] * 4) * 5
        result = detect(values, values)
        self.assertEqual(len(result["events"]), 10)
        self.assertEqual(len({event["event_id"] for event in result["events"]}), 10)
        self.assertEqual({event["arm"] for event in result["events"]}, {"left", "right"})

    def test_initial_open_state_does_not_create_release(self):
        result = detect([1] * 5 + [0] * 5 + [1] * 5)
        self.assertEqual([event["source_frame_id"] for event in result["events"]], [10])

    def test_single_frame_spike_and_threshold_chatter_do_not_duplicate(self):
        result = detect([0] * 6 + [1, .8, 1, 1, .8, 1, 1] + [0] * 6)
        self.assertEqual([event["source_frame_id"] for event in result["events"]], [8])

    def test_duration_confirmation_preserves_onset(self):
        result = detect([0] * 5 + [1] * 6, stable_duration_s=.15)
        self.assertEqual(result["events"][0]["source_frame_id"], 5)
        self.assertEqual(result["events"][0]["confirmed_at_source_frame_id"], 8)

    def test_raw_scale_and_opening_direction_are_explicit(self):
        small = detect([0] * 5 + [.07] * 5)
        large = detect([0] * 5 + [68] * 5)
        self.assertEqual(small["events"][0]["source_frame_id"], large["events"][0]["source_frame_id"])
        self.assertAlmostEqual(small["calibration"]["arms"]["left"]["open_threshold_raw"], .063)
        self.assertEqual(large["calibration"]["raw_units"], "unspecified")
        reverse = detect([1] * 5 + [0] * 5, opening_directions={"left": "lower"})
        self.assertEqual(reverse["events"][0]["source_frame_id"], 5)

    def test_constant_and_tiny_relative_range_are_diagnostic(self):
        result = detect([68] * 5 + [68.01] * 5)
        self.assertEqual(result["events"], [])
        self.assertEqual(result["calibration"]["arms"]["left"]["status"], "insufficient_dynamic_range")

    def test_nonfinite_and_missing_feedback_break_continuity(self):
        states, times, indices, source = episode([0] * 5 + [1, 1, 1, 1] + [0] * 4 + [1] * 4)
        states[6][6] = float("nan")
        states[2] = None
        result = detect_gripper_events(states, times, indices, source_timestamps=source)
        self.assertEqual([event["source_frame_id"] for event in result["events"]], [13])
        self.assertEqual(result["calibration"]["invalid_state_rows"], 2)
        json.dumps(result, allow_nan=False)

    def test_missing_all_feedback_has_no_action_fallback(self):
        result = detect_gripper_events([None] * 5, [i / 20 for i in range(5)], list(range(5)))
        self.assertEqual(result["events"], [])
        self.assertFalse(result["alignment_verified"])
        self.assertEqual(result["calibration"]["arms"]["left"]["status"], "missing_feedback")

    def test_uses_explicit_frame_indices_instead_of_row_numbers(self):
        result = detect([0] * 5 + [1] * 4, frame_indices=list(range(100, 109)))
        self.assertEqual(result["events"][0]["source_frame_id"], 105)
        self.assertEqual(result["events"][0]["timestamp_s"], 5.25)

    def test_frame_gap_cannot_supply_stability_or_crossing(self):
        result = detect([0] * 5 + [1] * 4, frame_indices=[0, 1, 2, 3, 4, 6, 7, 8, 9])
        self.assertEqual(result["events"], [])
        self.assertIn("nonconsecutive_source_frames", [d["code"] for d in result["diagnostics"]])

    def test_one_frame_timestamp_shift_fails_strict_alignment(self):
        states, times, indices, source = episode([0] * 5 + [1] * 4)
        with self.assertRaisesRegex(ValueError, "PTS alignment failed"):
            detect_gripper_events(states, [time + .05 for time in times], indices, source_timestamps=source)

    def test_invalid_axis_values_are_rejected(self):
        states, times, indices, source = episode([0] * 5 + [1] * 4)
        invalid_cases = [([float("nan")] + times[1:], indices),
                         (times, [0, 0] + indices[2:]),
                         (times, [0.5] + indices[1:]),
                         (times, indices[:-1] + [99])]
        for bad_times, bad_indices in invalid_cases:
            with self.subTest(indices=bad_indices), self.assertRaises(ValueError):
                detect_gripper_events(states, bad_times, bad_indices, source_timestamps=source)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed locally")
    def test_parquet_reads_state_not_action(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        states, times, indices, source = episode([0] * 5 + [1] * 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.parquet"
            pq.write_table(pa.table({"observation.state": states, "action": [[0] * 14] * len(states),
                                     "timestamp": times, "frame_index": indices}), path)
            result = detect_gripper_events_from_parquet(path, source_timestamps=source)
            self.assertEqual(result["events"][0]["source_frame_id"], 5)
            with self.assertRaisesRegex(ValueError, "never actions"):
                detect_gripper_events_from_parquet(path, state_column="action", source_timestamps=source)


class TestMultiThresholdCandidates(unittest.TestCase):
    def candidates(self, left, right=None, frame_indices=None, **options):
        states, times, indices, source = episode(left, right, frame_indices)
        return detect_gripper_event_candidates(states, times, indices, source_timestamps=source, **options)

    def test_partial_plateau_below_global_ninety_percent_is_retained(self):
        values = [0] * 5 + [.63] * 5 + [0] * 5 + [.71] * 5
        self.assertEqual([event["source_frame_id"] for event in detect(values)["events"]], [15])
        result = self.candidates(values)
        self.assertEqual([event["source_frame_id"] for event in result["events"]], [5, 15])
        self.assertEqual([event["anchor_level"] for event in result["events"]], [.75, .9])
        self.assertEqual(result["events"][0]["levels_detected"], [.5, .75])
        self.assertEqual(result["calibration"]["open_fraction"], .9)
        self.assertEqual(len(result["calibration"]["ensemble"]["levels"]), 3)
        json.dumps(result, allow_nan=False)

    def test_two_arm_realistic_partial_plateaus_have_independent_candidates(self):
        left = [0] * 5 + [.06315] * 5 + [0] * 5 + [.07059] * 5
        right = [0] * 5 + [.03623] * 5 + [0] * 5 + [.0618] * 5
        result = self.candidates(left, right)
        first = {event["arm"]: event for event in result["events"] if event["source_frame_id"] == 5}
        self.assertEqual(set(first), {"left", "right"})
        self.assertEqual(first["left"]["anchor_level"], .75)
        self.assertEqual(first["right"]["anchor_level"], .5)

    def test_candidate_frames_and_levels_are_scale_invariant(self):
        values = [0] * 5 + [.63] * 5 + [0] * 5 + [.71] * 5
        signatures = []
        for scale in (.001, 1, 1000):
            result = self.candidates([value * scale for value in values])
            signatures.append([(event["event_id"], event["source_frame_id"], event["anchor_level"])
                               for event in result["events"]])
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_highest_level_anchor_keeps_all_raw_crossing_evidence(self):
        result = self.candidates([0] * 5 + [.55] * 2 + [.8] * 2 + [1] * 6)
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["candidate_group_start_source_frame_id"], 5)
        self.assertEqual(event["source_frame_id"], 9)
        self.assertEqual(event["confirmed_at_source_frame_id"], 10)
        self.assertEqual(event["anchor_level"], .9)
        self.assertEqual([item["source_frame_id"] for item in event["level_evidence"]], [5, 7, 9])
        self.assertTrue(all(item["evidence"]["raw_confirmation_values"] for item in event["level_evidence"]))

    def test_group_window_is_measured_from_first_crossing_not_transitively(self):
        result = self.candidates([0] * 5 + [.55] * 4 + [.8] * 4 + [1] * 6)
        self.assertEqual([event["source_frame_id"] for event in result["events"]], [9, 13])
        self.assertEqual([event["candidate_group_start_source_frame_id"] for event in result["events"]], [5, 13])
        self.assertEqual([event["levels_detected"] for event in result["events"]], [[.5, .75], [.9]])

    def test_rapid_independent_cycles_are_not_swallowed_by_grouping(self):
        result = self.candidates([0] * 4 + [1] * 2 + [0] * 2 + [1] * 5)
        self.assertEqual([event["source_frame_id"] for event in result["events"]], [4, 8])
        self.assertEqual(len({event["event_id"] for event in result["events"]}), 2)
        self.assertEqual([len(event["level_evidence"]) for event in result["events"]], [3, 3])

    def test_base_threshold_options_are_explicitly_superseded(self):
        result = self.candidates([0] * 5 + [1] * 5, open_fraction=.99, rearm_fraction=.98)
        self.assertEqual(result["events"][0]["levels_detected"], [.5, .75, .9])

    def test_strict_axis_and_missing_feedback_rules_remain_in_force(self):
        states, times, indices, source = episode([0] * 5 + [1] * 5)
        with self.assertRaisesRegex(ValueError, "PTS alignment failed"):
            detect_gripper_event_candidates(states, [time + .05 for time in times], indices, source_timestamps=source)
        missing = detect_gripper_event_candidates([None] * 10, times, indices, source_timestamps=source)
        self.assertEqual(missing["events"], [])
        self.assertEqual(missing["calibration"]["invalid_state_rows"], 10)
        json.dumps(missing, allow_nan=False)

    def test_invalid_level_options_are_rejected(self):
        for levels in ([], [(.9, .9)], [(.5, .25), (.5, .25)], [(.5, .25), (.5, .3)], [(.5,)], None):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                self.candidates([0] * 5 + [1] * 5, levels=levels)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed locally")
    def test_parquet_multithreshold_entrypoint_uses_measured_feedback(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        states, times, indices, source = episode([0] * 5 + [.63] * 5 + [0] * 5 + [.71] * 5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.parquet"
            pq.write_table(pa.table({"state": states, "action": [[0] * 14] * len(states),
                                     "timestamp": times, "frame_index": indices}), path)
            result = detect_gripper_event_candidates_from_parquet(path, source_timestamps=source,
                                                                  open_fraction=.9, rearm_fraction=.65)
            self.assertEqual([event["source_frame_id"] for event in result["events"]], [5, 15])
            self.assertEqual(result["calibration"]["state_column"], "state")


if __name__ == "__main__":
    unittest.main()
