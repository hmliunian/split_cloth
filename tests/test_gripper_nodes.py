import json
import unittest
from unittest.mock import patch

from split_cloth.gripper_nodes import _settle_candidate, detect_gripper_nodes, prepare_gripper_nodes


def episode(left, right=None):
    right = [0] * len(left) if right is None else right
    states = []
    for left_value, right_value in zip(left, right):
        row = [0.] * 14
        row[6], row[13] = left_value, right_value
        states.append(row)
    indices = list(range(len(states)))
    times = [index / 20 for index in indices]
    return states, times, indices


def detect(left, right=None, **options):
    states, times, indices = episode(left, right)
    return detect_gripper_nodes(states, times, indices, source_timestamps=times, **options)


class TestGripperNodes(unittest.TestCase):
    def test_section_52_uses_first_settled_frame_not_motion_or_confirmation(self):
        result = detect([0] * 420 + [.6, .9] + [1] * 40)
        node = next(node for node in result["nodes"] if node["eligible_stage_end"])
        self.assertEqual(node["source_frame_id"], 422)
        change = node["changes"][0]
        self.assertEqual(change["confirmed_at_source_frame_id"], 424)
        self.assertEqual(change["evidence"]["crossing_candidates"][0]["candidate_group_start_source_frame_id"], 420)
        self.assertEqual(node["open_arms"], ["left"])
        self.assertTrue(result["alignment_verified"])
        self.assertEqual(node["evidence"]["interpretation"], "feedback_settled_estimate")
        self.assertFalse(node["evidence"]["physical_full_open_calibrated"])
        json.dumps(result, allow_nan=False)

    def test_closings_split_intervals_but_are_ineligible_as_stage_ends(self):
        result = detect([0] * 6 + [1] * 6 + [0] * 6)
        self.assertEqual([item["source_frame_id"] for item in result["nodes"]], [6, 12])
        self.assertEqual([item["node_id"] for item in result["nodes"]], ["a0001", "a0002"])
        self.assertEqual([item["eligible_stage_end"] for item in result["nodes"]], [True, False])
        self.assertEqual(result["nodes"][1]["changes"][0]["kind"], "close")
        self.assertEqual(result["nodes"][1]["closed_arms"], ["left"])

    def test_partial_open_plateau_and_its_closing_remain_candidates(self):
        result = detect([0] * 6 + [.3] * 6 + [0] * 6 + [1] * 6)
        self.assertEqual([item["source_frame_id"] for item in result["nodes"]], [6, 12, 18])
        self.assertEqual([item["eligible_stage_end"] for item in result["nodes"]], [True, False, True])
        change = result["nodes"][0]["changes"][0]
        self.assertAlmostEqual(change["raw_plateau_value"], .3)
        self.assertEqual(change["evidence"]["crossing_candidates"][0]["anchor_level"], .25)

    def test_opening_direction_and_raw_scale_are_explicit(self):
        values = [0] * 6 + [1] * 6 + [0] * 6
        regular = detect(values)
        scaled = detect([value * .07 for value in values])
        inverse = detect([68 * (1 - value) for value in values], opening_directions={"left": "lower"})
        for result in (regular, scaled, inverse):
            self.assertEqual([item["source_frame_id"] for item in result["nodes"]], [6, 12])
            self.assertEqual(result["nodes"][0]["changes"][0]["kind"], "open")
        self.assertEqual(inverse["nodes"][0]["changes"][0]["evidence"]["opening_direction"], "lower")
        self.assertAlmostEqual(scaled["nodes"][0]["changes"][0]["raw_plateau_value"], .07)

    def test_same_frame_arm_changes_merge_but_nearby_frames_do_not(self):
        left = [0] * 6 + [1] * 12
        same = detect(left, left)["nodes"]
        self.assertEqual(len(same), 1)
        self.assertEqual(len(same[0]["changes"]), 2)
        self.assertEqual(same[0]["open_arms"], ["left", "right"])
        delayed = detect(left, [0] * 8 + [1] * 10)["nodes"]
        self.assertEqual([item["source_frame_id"] for item in delayed], [6, 8])
        self.assertEqual(delayed[0]["arm_states"]["right"], "estimated_closed")
        self.assertEqual(delayed[1]["open_arms"], ["left", "right"])

    def test_initial_open_does_not_create_opening_event(self):
        result = detect([1] * 6 + [0] * 6 + [1] * 6)
        self.assertEqual([node["source_frame_id"] for node in result["nodes"]], [6, 12])
        self.assertEqual(result["calibration"]["initial_states"]["left"]["kind"], "open")
        self.assertFalse(result["nodes"][0]["eligible_stage_end"])

    def test_constant_arm_remains_unknown_instead_of_inventing_calibration(self):
        result = detect([0] * 6 + [1] * 6, [68] * 12)
        self.assertEqual(result["nodes"][0]["unknown_arms"], ["right"])
        self.assertEqual(result["calibration"]["arms"]["right"]["status"], "insufficient_dynamic_range")

    def test_arm_in_motion_is_unknown_even_after_an_earlier_opening(self):
        left = [0] * 6 + [1] * 6 + [.8, .6, .4, .2] + [0] * 6
        right = [0] * 14 + [1] * 8
        result = detect(left, right)
        right_open = next(node for node in result["nodes"] if node["source_frame_id"] == 14)
        self.assertEqual(right_open["arm_states"]["left"], "unknown")
        self.assertEqual(right_open["open_arms"], ["right"])

    def test_short_spike_or_ramp_without_hold_has_no_settled_node(self):
        result = detect([0] * 8 + [.3, .6, 1, .6, .3] + [0] * 8)
        self.assertFalse(any(node["eligible_stage_end"] for node in result["nodes"]))

    def test_opening_overshoot_then_lower_stable_open_plateau_is_retained(self):
        values = [0] * 8 + [.4, .8, 1, 1] + [.91] * 8 + [0] * 8
        result = detect(values)
        opening = next(node for node in result["nodes"] if node["eligible_stage_end"])
        self.assertEqual(opening["source_frame_id"], 12)
        self.assertEqual(opening["changes"][0]["confirmed_at_source_frame_id"], 14)
        change = opening["changes"][0]
        self.assertAlmostEqual(change["retreat_from_extreme_raw"], .09)
        self.assertGreater(change["retained_directional_excursion_raw"], .8)
        self.assertEqual(opening["arm_states"]["left"], "estimated_open")
        dynamic_range = result["calibration"]["arms"]["left"]["dynamic_range_raw"]
        self.assertAlmostEqual(change["evidence"]["reversal_tolerance_raw"], .1 * dynamic_range)

    def test_closing_overshoot_then_stable_closed_plateau_is_symmetric(self):
        values = [1] * 8 + [.6, .2, 0, 0] + [.09] * 8 + [1] * 8
        result = detect(values)
        closing = next(node for node in result["nodes"] if not node["eligible_stage_end"])
        self.assertEqual(closing["source_frame_id"], 12)
        self.assertEqual(closing["arm_states"]["left"], "estimated_closed")

    def test_excursion_guard_rejects_closed_plateau_even_with_large_reversal_budget(self):
        values = [0] * 5 + [1, 1] + [0] * 8
        indices = list(range(len(values)))
        candidate = {"source_frame_id": 5, "candidate_group_start_source_frame_id": 5}
        plateau = _settle_candidate(
            candidate, values, [index / 20 for index in indices], indices,
            direction="higher", tolerance=.02, reversal_tolerance=1,
            minimum_excursion=.1, samples=3, duration=.1, search_window=2, cutoff=None)
        self.assertIsNone(plateau)

    def test_opposite_crossing_cutoff_remains_hard_after_relaxing_reversal(self):
        result = detect([0] * 8 + [1] * 2 + [.1] * 8,
                        settled_reversal_fraction=1, settled_min_excursion_fraction=.01)
        self.assertFalse(any(node["eligible_stage_end"] for node in result["nodes"]))

    def test_no_synthetic_node_at_episode_start_or_end(self):
        result = detect([0] * 6 + [1] * 6)
        self.assertEqual([node["source_frame_id"] for node in result["nodes"]], [6])

    def test_invalid_full_state_or_video_axis_fails_without_interpolation(self):
        states, times, indices = episode([0] * 6 + [1] * 6)
        with self.assertRaisesRegex(ValueError, "PTS alignment failed"):
            detect_gripper_nodes(states, [time + .05 for time in times], indices, source_timestamps=times)
        with self.assertRaisesRegex(ValueError, "every source video frame"):
            detect_gripper_nodes(states[1:], times[1:], indices[1:], source_timestamps=times)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            detect_gripper_nodes(states[:5] + states[6:], times[:5] + times[6:], indices[:5] + indices[6:], source_timestamps=times)
        for invalid in (None, [0] * 13, [float("nan")] + [0] * 13, [0] * 13 + [float("inf")]):
            bad_states = list(states)
            bad_states[4] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite 14D"):
                detect_gripper_nodes(bad_states, times, indices, source_timestamps=times)

    def test_unverified_pure_detector_is_marked_and_prepare_requires_source_pts(self):
        states, times, indices = episode([0] * 6 + [1] * 6)
        result = detect_gripper_nodes(states, times, indices)
        self.assertFalse(result["alignment_verified"])
        self.assertFalse(result["nodes"][0]["evidence"]["alignment_verified"])
        with self.assertRaisesRegex(ValueError, "requires decoded"):
            prepare_gripper_nodes("unused.parquet", None)

    def test_unverified_excerpt_preserves_explicit_source_frame_ids(self):
        states, times, indices = episode([0] * 6 + [1] * 6)
        result = detect_gripper_nodes(states, [time + 5 for time in times],
                                      [index + 100 for index in indices])
        self.assertEqual(result["nodes"][0]["source_frame_id"], 106)
        self.assertEqual(result["nodes"][0]["changes"][0]["confirmed_at_source_frame_id"], 108)
        self.assertEqual(result["calibration"]["initial_states"]["left"]["confirmed_at_source_frame_id"], 102)

    def test_prepare_reads_measured_feedback_and_preserves_axis_and_evidence(self):
        states, times, indices = episode([0] * 6 + [1] * 6)
        table = {"observation.state": states, "timestamp": times, "frame_index": indices}
        with patch("split_cloth.gripper_nodes._read_measured_parquet", return_value=(table, "observation.state")) as reader:
            result = prepare_gripper_nodes("episode.parquet", times)
        reader.assert_called_once_with("episode.parquet", None, "timestamp", "frame_index")
        self.assertEqual(result["calibration"]["state_column"], "observation.state")
        self.assertEqual(result["calibration"]["signal_source"], "measured_state_only")
        self.assertEqual(result["nodes"][0]["source_frame_id"], 6)

    def test_invalid_settling_options_are_rejected(self):
        for options in ({"settled_samples": True}, {"settled_samples": 1},
                        {"settled_duration_s": -.1}, {"settled_tolerance_fraction": .25},
                        {"settled_search_window_s": 0},
                        {"settled_reversal_fraction": -.01},
                        {"settled_reversal_fraction": 1.01},
                        {"settled_min_excursion_fraction": 0},
                        {"settled_min_excursion_fraction": float("nan")},
                        {"settled_search_window_s": .05, "settled_duration_s": .1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                detect([0] * 6 + [1] * 6, **options)


if __name__ == "__main__":
    unittest.main()
