"""Metrics must separate accepted predictions, abstentions, and GPT uncertainty."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from split_cloth.evaluation import evaluate_gpt
from split_cloth.util import episode_dir, read_json, write_json, write_jsonl


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "test-run"
        self.manifest = {
            "key": "fold_cloth_v3__episode_000123", "dataset_id": "fold_cloth_v3",
            "episode_id": "episode_000123", "source_fps": 20, "frame_count": 1000,
            "split": "heldout",
        }
        write_jsonl(self.run / "manifest.jsonl", [self.manifest])
        self.output = episode_dir(self.run, self.manifest)
        self.review = {
            "reviewer": "gpt_visual", "status": "reference_only",
            "task_type": "fold_cloth", "applicable": True,
            "boundary_frames": [100, 300, 500], "stage4_completion_frame": 800,
            "summary": "Independent visual coarse reference, not human gold.",
            "issues": ["Off-screen placement cannot be checked"],
            "precision": "sampled_1fps", "is_human_ground_truth": False,
            "review_scope": "independent_montage_blind_review",
        }
        self.annotation = {
            "status": "accepted", "task_type": "fold_cloth",
            "boundary_frames": [100, 300, 500], "stage4_completion_frame": 800,
        }

    def evaluate(self, review=None, annotation=None):
        write_json(self.output / "gpt_review.json", self.review if review is None else review)
        write_json(self.output / "annotation.json", self.annotation if annotation is None else annotation)
        return evaluate_gpt(self.run)

    def comparison(self, report):
        return report["episodes"][0]["comparison"]

    def test_accepted_good_boundaries_inclusive_one_second_tolerance(self):
        self.annotation["boundary_frames"] = [120, 280, 500]
        report = self.evaluate()
        comparison = self.comparison(report)
        self.assertEqual(comparison["status"], "pass")
        self.assertEqual(comparison["signed_errors_seconds"], [1.0, -1.0, 0.0])
        self.assertEqual(comparison["within_1s"], [True, True, True])
        self.assertEqual(report["reviewed"], 1)
        self.assertEqual(report["accepted_comparable_episodes"], 1)
        self.assertEqual(report["fully_within_1s"], 1)
        self.assertAlmostEqual(report["mae_seconds"], 2 / 3, places=6)
        self.assertEqual(report["boundaries_within_1s"], 3)

    def test_large_error_records_signed_errors_and_needs_review(self):
        self.annotation["boundary_frames"] = [121, 260, 560]
        report = self.evaluate()
        comparison = self.comparison(report)
        self.assertEqual(comparison["status"], "needs_review")
        self.assertEqual(comparison["signed_errors_seconds"], [1.05, -2.0, 3.0])
        self.assertEqual(report["fully_within_1s"], 0)
        self.assertEqual(report["boundary_disagreements"], 1)
        self.assertEqual(report["disagreements"], 1)
        self.assertAlmostEqual(report["mae_seconds"], 6.05 / 3, places=6)

    def test_scope_agreement_is_not_boundary_accuracy(self):
        self.review.update(status="out_of_scope", task_type="hang_cloth", applicable=False, boundary_frames=[None, None, None])
        self.annotation.update(status="out_of_scope", task_type="hang_cloth", boundary_frames=[])
        report = self.evaluate()
        self.assertEqual(self.comparison(report)["status"], "scope_agreement")
        self.assertEqual(report["scope_agreements"], 1)
        self.assertEqual(report["accepted_comparable_episodes"], 0)
        self.assertIsNone(report["mae_seconds"])

    def test_scope_disagreement_is_explicit(self):
        self.review.update(status="out_of_scope", task_type="hang_cloth", applicable=False, boundary_frames=[None, None, None])
        report = self.evaluate()
        self.assertEqual(self.comparison(report)["status"], "disagreement")
        self.assertEqual(report["scope_disagreements"], 1)
        self.assertEqual(report["accepted_comparable_episodes"], 0)

    def test_gpt_null_and_model_rejection_are_appropriate_abstention(self):
        self.review.update(status="needs_review", boundary_frames=[100, 300, None])
        self.annotation.update(status="needs_review", boundary_frames=[100, 300, 500])
        report = self.evaluate()
        self.assertEqual(self.comparison(report)["status"], "appropriate_abstention")
        self.assertEqual(report["appropriate_abstentions"], 1)
        self.assertEqual(report["accepted_comparable_episodes"], 0)
        self.assertEqual(report["compared_boundaries"], 0)
        self.assertIsNone(report["mae_seconds"])

    def test_rejected_candidate_boundaries_are_never_credited_as_correct(self):
        self.annotation["status"] = "needs_review"
        report = self.evaluate()
        self.assertEqual(self.comparison(report)["status"], "model_abstained")
        self.assertEqual(report["model_abstentions"], 1)
        self.assertEqual(report["accepted_comparable_episodes"], 0)
        self.assertEqual(report["fully_within_1s"], 0)
        self.assertIsNone(self.comparison(report)["signed_errors_seconds"])

    def test_no_review_file_is_not_counted_as_reviewed(self):
        write_json(self.output / "annotation.json", self.annotation)
        report = evaluate_gpt(self.run)
        self.assertEqual(report["reviewed"], 0)
        self.assertEqual(report["not_reviewed"], 1)
        self.assertEqual(report["episodes"], [])
        self.assertIsNone(report["mae_seconds"])

    def test_bool_is_not_an_integer_boundary_for_either_reviewer(self):
        self.review["boundary_frames"] = [True, 300, 500]
        report = self.evaluate()
        self.assertEqual(report["accepted_comparable_episodes"], 0)
        self.assertEqual(self.comparison(report)["status"], "unknown")
        self.review["boundary_frames"] = [100, 300, 500]
        self.annotation["boundary_frames"] = [True, 300, 500]
        report = self.evaluate()
        self.assertEqual(report["accepted_comparable_episodes"], 0)
        self.assertEqual(self.comparison(report)["status"], "unknown")

    def test_fps_comes_from_manifest_and_invalid_bool_is_not_a_rate(self):
        self.manifest["source_fps"] = 10
        write_jsonl(self.run / "manifest.jsonl", [self.manifest])
        self.annotation["boundary_frames"] = [120, 300, 500]
        report = self.evaluate()
        self.assertEqual(self.comparison(report)["signed_errors_seconds"], [2.0, 0.0, 0.0])
        self.manifest["source_fps"] = True
        write_jsonl(self.run / "manifest.jsonl", [self.manifest])
        report = self.evaluate()
        self.assertEqual(report["accepted_comparable_episodes"], 0)
        self.assertEqual(report["unknowns"], 1)

    def test_raw_review_status_and_provenance_are_preserved_on_repeated_runs(self):
        original = deepcopy(self.review)
        first = self.evaluate()
        second = evaluate_gpt(self.run)
        saved = read_json(self.output / "gpt_review.json")
        self.assertEqual({key: value for key, value in saved.items() if key != "comparison"}, original)
        self.assertEqual(saved["status"], "reference_only")
        self.assertEqual(saved["reviewer"], "gpt_visual")
        self.assertFalse(saved["is_human_ground_truth"])
        self.assertEqual(first["episodes"][0]["gpt_review"], original)
        self.assertEqual(second["episodes"][0]["gpt_review"], original)
        self.assertEqual(read_json(self.run / "reports/gpt_validation.json")["reviewed"], 1)
        markdown = (self.run / "reports/gpt_validation.md").read_text()
        self.assertIn("不能据此报告100条数据的准确率", markdown)
        self.assertFalse(second["can_report_100_episode_accuracy"])

    def test_uncertain_gpt_and_accepted_qwen_remain_unscored(self):
        self.review.update(status="needs_review", boundary_frames=[100, None, 500])
        report = self.evaluate()
        self.assertEqual(self.comparison(report)["status"], "unknown")
        self.assertEqual(report["unknowns"], 1)
        self.assertEqual(report["accepted_comparable_episodes"], 0)

    def test_multiple_reviews_aggregate_raw_counts_and_only_accepted_errors(self):
        self.evaluate()
        second = {**self.manifest, "key": "fold_cloth_v3__episode_000124", "episode_id": "episode_000124"}
        third = {**self.manifest, "key": "fold_cloth_v3__episode_000125", "episode_id": "episode_000125"}
        write_jsonl(self.run / "manifest.jsonl", [self.manifest, second, third])
        write_json(episode_dir(self.run, second) / "gpt_review.json", self.review)
        write_json(episode_dir(self.run, second) / "annotation.json", {**self.annotation, "boundary_frames": [140, 300, 500]})
        write_json(episode_dir(self.run, third) / "gpt_review.json", self.review)
        write_json(episode_dir(self.run, third) / "annotation.json", {**self.annotation, "status": "needs_review"})
        report = evaluate_gpt(self.run)
        self.assertEqual(report["reviewed"], 3)
        self.assertEqual(report["accepted_comparable_episodes"], 2)
        self.assertEqual(report["compared_boundaries"], 6)
        self.assertEqual(report["fully_within_1s"], 1)
        self.assertEqual(report["model_abstentions"], 1)
        self.assertAlmostEqual(report["mae_seconds"], 2 / 6, places=6)


if __name__ == "__main__":
    unittest.main()
