"""The annotation contract must never turn uncertainty into stage labels."""

from copy import deepcopy
import importlib.util
import unittest

from split_cloth.boundaries import (
    expand_labels,
    normalize_annotation,
    validate_model_output,
    validate_review_frames,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture():
    # Include the real 422/423 example and sparse sampling intervals.
    frames = [0, 402, 422, 442, 600, 620, 640, 800, 820, 840, 900]
    samples = [
        {
            "sample_id": f"s{index:04d}",
            "sample_seq": index,
            "source_frame_id": frame,
            "source_timestamp_s": frame / 20,
        }
        for index, frame in enumerate(frames)
    ]
    output = {
        "task_type": "fold_cloth",
        "applicable": True,
        "all_stages_observed": True,
        "needs_review": False,
        "review_reasons": [],
        "stage4_completion": {
            "observed": True,
            "completion_sample_id": "s0010",
            "evidence_text": "Cloth folded and grippers released.",
        },
        "boundary_proposals": [
            {
                "from_stage": stage,
                "to_stage": stage + 1,
                "last_not_complete_sample_id": f"s{left:04d}",
                "first_complete_sample_id": f"s{right:04d}",
                "evidence_text": "Fold completed and active gripper released.",
            }
            for stage, left, right in ((1, 1, 2), (2, 4, 5), (3, 7, 8))
        ],
    }
    return output, samples, {"frame_count": 1000, "fps": 20}


class NormalizeAnnotationTests(unittest.TestCase):
    def normalize(self, output=None, samples=None, manifest=None):
        default_output, default_samples, default_manifest = fixture()
        return normalize_annotation(
            default_output if output is None else output,
            default_samples if samples is None else samples,
            default_manifest if manifest is None else manifest,
        )

    def assert_review(self, result):
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["boundary_frames"], [])
        self.assertTrue(result["machine_review_reasons"])
        if result["frame_count"]:
            self.assertEqual(result["segments"], [{
                "stage": -1, "start_frame": 0,
                "end_frame": result["frame_count"] - 1,
            }])

    def test_422_belongs_to_previous_stage_423_to_next(self):
        result = self.normalize()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["boundary_frames"], [422, 620, 820])
        self.assertEqual(result["segments"][0], {"stage": 1, "start_frame": 0, "end_frame": 422})
        self.assertEqual(result["segments"][1]["start_frame"], 423)
        self.assertEqual(result["boundaries"][0]["next_stage_start_frame"], 423)
        self.assertEqual(result["stage4_completion_frame"], 900)
        self.assertEqual(result["segments"][-1]["end_frame"], 999)
        self.assertEqual(result["spec_version"], "coarse_v1.1_expert")
        self.assertEqual(result["precision"], "sampled_1fps")

    def test_uncertainty_interval_is_inclusive_and_excludes_left(self):
        result = self.normalize()
        boundary = result["boundaries"][0]
        self.assertEqual(boundary["estimated_boundary_interval_frames"], [403, 422])
        lower, upper = boundary["estimated_boundary_interval_frames"]
        self.assertEqual(upper - lower + 1, 20)
        self.assertEqual(boundary["last_not_complete_timestamp_s"], 20.1)
        self.assertEqual(boundary["first_complete_timestamp_s"], 21.1)
        self.assertEqual(sum(item["end_frame"] - item["start_frame"] + 1 for item in result["segments"]), 1000)

    def test_unknown_is_review_even_if_applicable_false(self):
        output, _, _ = fixture()
        output["task_type"] = "unknown"
        output["applicable"] = False
        self.assert_review(self.normalize(output))

    def test_known_nonfold_tasks_and_inapplicability_are_out_of_scope(self):
        for task in ("fold_pants", "hang_cloth", "mixed", "other"):
            output, _, _ = fixture()
            output["task_type"] = task
            output["all_stages_observed"] = False
            output["stage4_completion"]["observed"] = False
            output["stage4_completion"]["completion_sample_id"] = None
            for proposal in output["boundary_proposals"]:
                proposal["last_not_complete_sample_id"] = None
                proposal["first_complete_sample_id"] = None
            result = self.normalize(output)
            self.assertEqual(result["status"], "out_of_scope", task)
            self.assertFalse(result["needs_review"])
            self.assertEqual(result["boundary_frames"], [])
        output, _, _ = fixture()
        output["applicable"] = False
        self.assertEqual(self.normalize(output)["status"], "out_of_scope")

    def test_fabricated_sample_id_is_never_accepted(self):
        output, _, _ = fixture()
        output["boundary_proposals"][0]["first_complete_sample_id"] = "invented"
        result = self.normalize(output)
        self.assert_review(result)
        self.assertTrue(any("unknown ID" in item for item in result["machine_review_reasons"]))
        self.assertEqual(result["model_output"], output)

    def test_nonadjacent_samples_are_rejected(self):
        output, _, _ = fixture()
        output["boundary_proposals"][0]["last_not_complete_sample_id"] = "s0000"
        result = self.normalize(output)
        self.assert_review(result)
        self.assertTrue(any("adjacent" in item for item in result["machine_review_reasons"]))

    def test_duplicate_sample_ids_and_seq_are_rejected(self):
        _, samples, _ = fixture()
        samples.append(deepcopy(samples[1]))
        result = self.normalize(samples=samples)
        self.assert_review(result)
        self.assertTrue(any("duplicate ID" in item for item in result["machine_review_reasons"]))
        self.assertTrue(any("duplicate sequence" in item for item in result["machine_review_reasons"]))

    def test_bool_cannot_stand_in_for_stage_frame_or_sequence_integer(self):
        output, samples, _ = fixture()
        output["boundary_proposals"][0]["from_stage"] = True
        self.assert_review(self.normalize(output))
        samples[0]["sample_seq"] = False
        self.assert_review(self.normalize(samples=samples))
        self.assertTrue(validate_review_frames(100, [True, 20, 30], 40))
        self.assertTrue(validate_review_frames(True, [10, 20, 30], 40))
        self.assertTrue(validate_review_frames(100, [10, 20, 30], True))

    def test_strict_model_types(self):
        for field, invalid in (("applicable", 1), ("all_stages_observed", 1), ("needs_review", 0), ("review_reasons", "none")):
            output, _, _ = fixture()
            output[field] = invalid
            self.assertTrue(validate_model_output(output), field)
            self.assert_review(self.normalize(output))

    def test_extra_fields_at_each_level_are_rejected(self):
        for location in ("top", "completion", "proposal"):
            output, _, _ = fixture()
            target = {"top": output, "completion": output["stage4_completion"], "proposal": output["boundary_proposals"][0]}[location]
            target["guessed_confidence"] = 0.9
            self.assert_review(self.normalize(output))

    def test_final_stage_start_is_not_completion(self):
        output, _, _ = fixture()
        output["stage4_completion"]["observed"] = False
        self.assert_review(self.normalize(output))
        output["stage4_completion"]["observed"] = True
        output["stage4_completion"]["completion_sample_id"] = "s0008"
        self.assert_review(self.normalize(output))

    def test_reversed_stage_boundaries_are_rejected(self):
        output, _, _ = fixture()
        first, second = output["boundary_proposals"][:2]
        for field in ("last_not_complete_sample_id", "first_complete_sample_id"):
            first[field], second[field] = second[field], first[field]
        self.assert_review(self.normalize(output))

    def test_null_or_missing_boundaries_remain_unknown(self):
        output, _, _ = fixture()
        output["boundary_proposals"][1]["first_complete_sample_id"] = None
        self.assertEqual(validate_model_output(output), [])
        self.assert_review(self.normalize(output))
        output["boundary_proposals"].pop()
        self.assert_review(self.normalize(output))

    def test_model_review_cannot_be_ignored(self):
        output, _, _ = fixture()
        output["needs_review"] = True
        output["review_reasons"] = ["occluded gripper"]
        result = self.normalize(output)
        self.assert_review(result)
        self.assertIn("occluded gripper", result["review_reasons"])

    def test_malformed_responses_return_review_instead_of_raising(self):
        for malformed in (None, [], "not JSON object", {"task_type": []}, {"boundary_proposals": [False, None, []]}):
            self.assert_review(normalize_annotation(malformed, fixture()[1], fixture()[2]))

    def test_bad_manifest_does_not_crash_pipeline(self):
        output, samples, _ = fixture()
        for manifest in (None, {}, {"frame_count": True}, {"frame_count": -1}):
            result = normalize_annotation(output, samples, manifest)
            self.assert_review(result)
            self.assertEqual(result["frame_count"], 0)

    def test_num_frames_manifest_alias_and_arbitrary_input_order(self):
        output, samples, _ = fixture()
        result = normalize_annotation(output, list(reversed(samples)), {"num_frames": 1000})
        self.assertEqual(result["status"], "accepted")

    def test_manual_frame_validation(self):
        self.assertEqual(validate_review_frames(1000, [422, 620, 820], 900), [])
        for boundaries, completion in (([422, 620, 999], 999), ([422, 422, 820], 900), ([422, 620, 820], 820), ([422, 620, 820], 1000), ([-1, 620, 820], 900)):
            self.assertTrue(validate_review_frames(1000, boundaries, completion))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is not installed in this runtime")
    def test_expansion_int64_shape_off_by_one_and_unknown(self):
        import numpy as np
        result = self.normalize()
        labels = expand_labels(result)
        self.assertEqual(labels.shape, (1000,))
        self.assertEqual(labels.dtype, np.dtype("int64"))
        self.assertEqual(labels[422], 1)
        self.assertEqual(labels[423], 2)
        self.assertEqual(labels[820], 3)
        self.assertEqual(labels[821], 4)
        self.assertEqual(labels[999], 4)
        output, _, _ = fixture()
        output["task_type"] = "unknown"
        self.assertTrue(np.all(expand_labels(self.normalize(output)) == -1))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is not installed in this runtime")
    def test_expansion_rejects_gap_in_accepted_annotation(self):
        annotation = self.normalize()
        annotation["segments"][1]["start_frame"] += 1
        with self.assertRaises(ValueError):
            expand_labels(annotation)


if __name__ == "__main__":
    unittest.main()
