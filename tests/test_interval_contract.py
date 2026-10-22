"""Fixed-node selection, complete interval evidence, and spec 5.2 ownership."""
from copy import deepcopy
import unittest

from split_cloth.interval_contract import normalize_interval_selection, validate_interval_selection


def fixture():
    nodes = [{"node_id": f"a{index}", "source_frame_id": frame,
              "eligible_stage_end": True, "open_arms": ["left", "right"]}
             for index, frame in enumerate((100, 422, 500, 600, 800))]
    intervals = []
    start = 0
    for index, end in enumerate((100, 422, 500, 600, 800, 999)):
        intervals.append({"interval_id": f"i{index}", "start_frame": start, "end_frame": end,
                          "right_node_id": f"a{index}" if index < 5 else None,
                          "sample_frame_ids": sorted(set([start, end] + list(range(start, end + 1, 20))))})
        start = end + 1
    observations = [
        {"interval_id": f"i{index}", "activity": activity,
         "operation_complete": index != 2, "evidence_text": f"Visible {activity} sequence"}
        for index, activity in enumerate(("prepare", "first_fold", "second_fold", "second_fold", "cross_fold", "remove"))
    ]
    raw = {
        "task_type": "fold_cloth", "applicable": True, "all_stages_observed": True,
        "needs_review": False, "review_reasons": [], "interval_observations": observations,
        "stage_end_nodes": [
            {"stage_id": stage, "node_id": node_id, "participant_arms": ["left", "right"],
             "evidence_interval_ids": evidence, "evidence_text": "Fold completed and participating grippers open"}
            for stage, node_id, evidence in ((1, "a1", ["i0", "i1"]), (2, "a3", ["i2", "i3"]), (3, "a4", ["i4"]))
        ],
        "stage4_completion": {"observed": True, "interval_id": "i5", "evidence_text": "Folded cloth has been removed"},
    }
    return raw, nodes, intervals, {"frame_count": 1000, "source_fps": 20}


class IntervalContractTests(unittest.TestCase):
    def normalize(self, raw=None, nodes=None, intervals=None, manifest=None):
        defaults = fixture()
        return normalize_interval_selection(*[default if supplied is None else supplied
                                              for supplied, default in zip((raw, nodes, intervals, manifest), defaults)])

    def assert_review(self, annotation, reason=None):
        self.assertEqual(annotation["status"], "needs_review", annotation["review_reasons"])
        self.assertEqual(annotation["boundary_frames"], [])
        self.assertEqual(annotation["segments"], [{"stage": -1, "start_frame": 0, "end_frame": 999}])
        if reason:
            self.assertTrue(any(reason in value for value in annotation["review_reasons"]), annotation["review_reasons"])

    def test_node_422_ends_stage_1_and_423_starts_stage_2(self):
        annotation = self.normalize()
        self.assertEqual(annotation["status"], "accepted", annotation["review_reasons"])
        self.assertEqual(annotation["boundary_frames"], [422, 600, 800])
        def stage_at(frame):
            return next(item["stage"] for item in annotation["segments"]
                        if item["start_frame"] <= frame <= item["end_frame"])
        self.assertEqual([stage_at(frame) for frame in (421, 422, 423)], [1, 1, 2])
        self.assertFalse(annotation["training_ready"])
        self.assertEqual(annotation["precision"], "fixed_gripper_node_selection")
        self.assertFalse(annotation["boundaries"][0]["node_is_human_ground_truth"])

    def test_exact_schema_rejects_model_frame_fields_at_every_level(self):
        for location in ("root", "observation", "end", "completion"):
            with self.subTest(location=location):
                raw, *_ = fixture()
                target = {"root": raw, "observation": raw["interval_observations"][0],
                          "end": raw["stage_end_nodes"][0], "completion": raw["stage4_completion"]}[location]
                target["end_frame"] = 422
                self.assert_review(self.normalize(raw=raw), "unexpected fields")

    def test_exact_schema_rejects_missing_fields_and_bool_stage_ids(self):
        raw, *_ = fixture()
        raw["stage_end_nodes"][0]["stage_id"] = True
        del raw["stage4_completion"]["interval_id"]
        self.assertTrue(any("expected integer 1" in error for error in validate_interval_selection(raw)))
        self.assert_review(self.normalize(raw=raw), "missing fields")

    def test_unresolved_node_retains_candidates_but_never_labels(self):
        raw, *_ = fixture()
        raw["stage_end_nodes"][1]["node_id"] = None
        annotation = self.normalize(raw=raw)
        self.assert_review(annotation, "unresolved")
        self.assertEqual(annotation["candidate_boundary_frames"], [422, 800])
        self.assertIsNone(annotation["model_output"]["stage_end_nodes"][1]["node_id"])

    def test_unknown_node_is_rejected(self):
        raw, *_ = fixture()
        raw["stage_end_nodes"][0]["node_id"] = "not_supplied"
        self.assert_review(self.normalize(raw=raw), "unknown fixed node")

    def test_nodes_must_be_unique_and_increasing(self):
        for selected in ("a1", "a0"):
            with self.subTest(selected=selected):
                raw, *_ = fixture()
                raw["stage_end_nodes"][1]["node_id"] = selected
                self.assert_review(self.normalize(raw=raw), "strictly increasing")

    def test_close_or_ineligible_node_cannot_end_stage(self):
        raw, nodes, *_ = fixture()
        nodes[1]["eligible_stage_end"] = False
        self.assert_review(self.normalize(raw=raw, nodes=nodes), "eligible opening node")

    def test_participating_arms_must_be_open_and_nonempty(self):
        for arms in ([], ["left", "right"]):
            with self.subTest(arms=arms):
                raw, nodes, *_ = fixture()
                nodes[1]["open_arms"] = ["left"]
                raw["stage_end_nodes"][0]["participant_arms"] = arms
                self.assert_review(self.normalize(raw=raw, nodes=nodes), "all participants must be open")

    def test_cannot_skip_terminal_evidence_after_earlier_fold(self):
        raw, *_ = fixture()
        raw["stage_end_nodes"][1]["evidence_interval_ids"] = ["i2"]
        self.assert_review(self.normalize(raw=raw), "interval ending at the selected node")

    def test_terminal_interval_must_confirm_operation_completion(self):
        raw, *_ = fixture()
        raw["interval_observations"][3]["operation_complete"] = False
        self.assert_review(self.normalize(raw=raw), "terminal interval must confirm")

    def test_prepare_is_not_first_fold(self):
        raw, *_ = fixture()
        raw["interval_observations"][1]["activity"] = "prepare"
        self.assert_review(self.normalize(raw=raw), "first_fold evidence is required")

    def test_each_stage_requires_its_specific_folding_operation(self):
        for index, required in ((2, "second_fold"), (4, "cross_fold")):
            with self.subTest(required=required):
                raw, *_ = fixture()
                raw["interval_observations"][index]["activity"] = "other"
                if index == 2:
                    raw["interval_observations"][3]["activity"] = "other"
                self.assert_review(self.normalize(raw=raw), f"{required} evidence is required")

    def test_evidence_cannot_come_from_previous_or_later_stage(self):
        for interval_id in ("i1", "i4"):
            with self.subTest(interval_id=interval_id):
                raw, *_ = fixture()
                raw["stage_end_nodes"][1]["evidence_interval_ids"].append(interval_id)
                self.assert_review(self.normalize(raw=raw), "evidence must belong to this stage")

    def test_each_interval_requires_exactly_one_observation(self):
        for mutation in ("missing", "duplicate", "unknown"):
            with self.subTest(mutation=mutation):
                raw, *_ = fixture()
                if mutation == "missing":
                    raw["interval_observations"].pop(0)
                elif mutation == "duplicate":
                    raw["interval_observations"].append(deepcopy(raw["interval_observations"][0]))
                else:
                    raw["interval_observations"][0]["interval_id"] = "invented"
                self.assert_review(self.normalize(raw=raw), "exactly one observation")

    def test_all_stages_and_final_completion_are_independently_required(self):
        for observed, all_stages, reason in ((False, True, "independently"), (True, False, "all four stages")):
            with self.subTest(observed=observed, all_stages=all_stages):
                raw, *_ = fixture()
                raw["stage4_completion"]["observed"] = observed
                raw["all_stages_observed"] = all_stages
                self.assert_review(self.normalize(raw=raw), reason)

    def test_final_completion_uses_only_supplied_interval_end(self):
        annotation = self.normalize()
        self.assertEqual(annotation["stage4_completion_frame"], 999)
        self.assertEqual(annotation["stage4_completion"]["precision"], "observed_interval_end_not_first_completion_frame")

    def test_final_completion_cannot_precede_stage_4(self):
        raw, *_ = fixture()
        raw["stage4_completion"]["interval_id"] = "i4"
        self.assert_review(self.normalize(raw=raw), "start after the third stage end")

    def test_final_completion_requires_remove_evidence(self):
        raw, *_ = fixture()
        raw["interval_observations"][5]["activity"] = "other"
        self.assert_review(self.normalize(raw=raw), "completed remove activity")

    def test_missing_supplied_end_frame_cannot_become_completion_frame(self):
        _, _, intervals, _ = fixture()
        intervals[-1]["sample_frame_ids"].remove(999)
        annotation = self.normalize(intervals=intervals)
        self.assert_review(annotation, "ending frame supplied")
        self.assertIsNone(annotation["stage4_completion_frame"])

    def test_interval_split_cannot_overlap_boundary_frame(self):
        _, _, intervals, _ = fixture()
        intervals[2]["start_frame"] = 422
        intervals[2]["sample_frame_ids"].insert(0, 422)
        self.assert_review(self.normalize(intervals=intervals), "cover every frame once")

    def test_clear_nonapplicable_is_out_of_scope(self):
        raw, *_ = fixture()
        raw.update(task_type="hang_cloth", applicable=False, all_stages_observed=False)
        for item in raw["stage_end_nodes"]:
            item.update(node_id=None, participant_arms=[], evidence_interval_ids=[], evidence_text="")
        raw["stage4_completion"].update(observed=False, interval_id=None, evidence_text="")
        annotation = self.normalize(raw=raw)
        self.assertEqual(annotation["status"], "out_of_scope", annotation["review_reasons"])
        self.assertEqual(annotation["boundary_frames"], [])
        self.assertEqual(annotation["segments"][0]["stage"], -1)

    def test_unknown_is_review_even_with_applicable_false(self):
        raw, *_ = fixture()
        raw.update(task_type="unknown", applicable=False)
        self.assert_review(self.normalize(raw=raw), "fold_cloth must be confirmed")

    def test_model_requested_review_or_reasons_keep_unknown_labels(self):
        for change in ({"needs_review": True}, {"review_reasons": ["occluded gripper"]}):
            with self.subTest(change=change):
                raw, *_ = fixture()
                raw.update(change)
                self.assert_review(self.normalize(raw=raw), "model_")

    def test_blank_visual_evidence_is_not_accepted(self):
        raw, *_ = fixture()
        raw["stage_end_nodes"][0]["evidence_text"] = "  "
        self.assert_review(self.normalize(raw=raw), "completion and release evidence")

    def test_invalid_response_shapes_return_review_without_raising(self):
        for raw in (None, [], "{}", {"task_type": []}, {"stage_end_nodes": [None, 5, []]}):
            with self.subTest(raw=raw):
                annotation = normalize_interval_selection(raw, *fixture()[1:])
                self.assert_review(annotation)

    def test_invalid_input_bounds_return_review_without_raising(self):
        raw, nodes, intervals, manifest = fixture()
        intervals[0]["start_frame"] = None
        annotation = normalize_interval_selection(raw, nodes, intervals, manifest)
        self.assert_review(annotation, "invalid closed interval bounds")

    def test_input_and_response_are_not_mutated(self):
        inputs = fixture()
        original = deepcopy(inputs)
        normalize_interval_selection(*inputs)
        self.assertEqual(inputs, original)


if __name__ == "__main__":
    unittest.main()
