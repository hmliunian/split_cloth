"""Paired comparison must not equate acceptance or GPT coverage with accuracy."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from split_cloth.comparison import compare_runs
from split_cloth.util import episode_dir, read_json, write_json, write_jsonl


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.baseline = Path(self.temporary.name) / "baseline"
        self.candidate = Path(self.temporary.name) / "candidate"

    def manifest(self, index):
        return {"key": f"fold_cloth_v3__episode_{index:06d}", "dataset_id": "fold_cloth_v3",
                "episode_id": f"episode_{index:06d}", "source_fps": 20, "frame_count": 1000, "camera_key": "base_0_rgb"}

    def write_manifest(self, run, rows):
        write_jsonl(run / "manifest.jsonl", rows)

    def annotation(self, run, row, status="accepted", frames=None, **extra):
        value = {"status": status, "boundary_frames": frames or [100, 300, 500], "model_id": run.name, **extra}
        write_json(episode_dir(run, row) / "annotation.json", value)
        return value

    def review(self, run, row, errors=(0, 0, 0), status="pass", frames=None):
        annotation = read_json(episode_dir(run, row) / "annotation.json")
        reference = frames or [100, 300, 500]
        value = {"reviewer": "gpt_visual", "status": "reference_only", "applicable": True, "boundary_frames": reference,
                 "summary": "Independent coarse visual reference", "is_human_ground_truth": False,
                 "comparison": {"status": status, "accepted_comparable": errors is not None,
                                "qwen_status": annotation["status"], "qwen_boundary_frames": annotation["boundary_frames"],
                                "gpt_boundary_frames": reference, "source_fps": row["source_fps"],
                                "signed_errors_seconds": list(errors) if errors is not None else None}}
        write_json(episode_dir(run, row) / "gpt_review.json", value)
        return value

    def test_key_alignment_uses_candidate_order_and_same_denominator(self):
        one, two, extra = [self.manifest(i) for i in (1, 2, 3)]
        self.write_manifest(self.baseline, [one, two, extra])
        self.write_manifest(self.candidate, [two, one])
        for row in [one, two, extra]:
            self.annotation(self.baseline, row, status="needs_review")
        self.annotation(self.candidate, one)
        self.annotation(self.candidate, two, status="needs_review")
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual([row["key"] for row in report["episodes"]], [two["key"], one["key"]])
        self.assertEqual(report["baseline"]["total"], 2)
        self.assertEqual(report["candidate"]["total"], 2)
        self.assertEqual(report["full_manifest"]["baseline"]["total"], 3)
        self.assertEqual(report["paired"]["accepted_delta"], 1)
        self.assertEqual(report["paired"]["newly_accepted_keys"], [one["key"]])

    def test_semantic_review_execution_errors_and_missing_files_are_separate(self):
        rows = [self.manifest(i) for i in (1, 2, 3)]
        self.write_manifest(self.baseline, rows)
        self.write_manifest(self.candidate, rows)
        for run in [self.baseline, self.candidate]:
            self.annotation(run, rows[0], status="needs_review", review_reasons=["occluded"])
            self.annotation(run, rows[1], status="needs_review", execution_error="TimeoutError")
        report = compare_runs(self.baseline, self.candidate)
        for side in ["baseline", "candidate"]:
            self.assertEqual(report[side]["semantic_needs_review"], 1)
            self.assertEqual(report[side]["execution_errors"], 1)
            self.assertEqual(report[side]["category_counts"]["missing"], 1)

    def test_extra_candidate_gpt_coverage_cannot_create_shared_accuracy_claim(self):
        row = self.manifest(1)
        for run in [self.baseline, self.candidate]:
            self.write_manifest(run, [row])
            self.annotation(run, row)
        self.review(self.candidate, row)
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["baseline"]["gpt_accepted_comparable_episodes"], 0)
        self.assertEqual(report["candidate"]["gpt_accepted_comparable_episodes"], 1)
        self.assertEqual(report["paired"]["gpt_common_reference_episodes"], 0)
        self.assertIsNone(report["paired"]["candidate_common_mae_seconds"])
        self.assertFalse(report["acceptance_is_accuracy"])
        self.assertFalse(report["ranking_performed"])
        self.assertFalse(report["is_human_ground_truth"])
        markdown = (self.candidate / "reports/model_comparison.md").read_text()
        self.assertIn("通过率不等于标注准确率", markdown)
        self.assertIn("覆盖不同，不能直接据此排名", markdown)

    def test_same_reference_intersection_reports_signed_errors_and_disagreement(self):
        row = self.manifest(1)
        for run in [self.baseline, self.candidate]:
            self.write_manifest(run, [row])
        self.annotation(self.baseline, row, frames=[160, 380, 580])
        self.annotation(self.candidate, row, frames=[100, 320, 500])
        self.review(self.baseline, row, errors=[3, 4, 4], status="needs_review")
        self.review(self.candidate, row, errors=[0, 1, 0])
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["paired"]["gpt_common_reference_episodes"], 1)
        self.assertEqual(report["paired"]["gpt_common_compared_boundaries"], 3)
        self.assertAlmostEqual(report["paired"]["baseline_common_mae_seconds"], 11 / 3, places=6)
        self.assertAlmostEqual(report["paired"]["candidate_common_mae_seconds"], 1 / 3, places=6)
        self.assertEqual(report["baseline"]["gpt_boundary_disagreements"], 1)
        self.assertEqual(report["candidate"]["gpt_boundary_disagreements"], 0)
        self.assertEqual(report["episodes"][0]["baseline"]["signed_errors_seconds"], [3, 4, 4])

    def test_scope_disagreement_and_model_abstention_remain_unscored(self):
        row = self.manifest(1)
        for run in [self.baseline, self.candidate]:
            self.write_manifest(run, [row])
        self.annotation(self.baseline, row, status="out_of_scope")
        self.annotation(self.candidate, row, status="needs_review")
        self.review(self.baseline, row, errors=None, status="disagreement")
        self.review(self.candidate, row, errors=None, status="model_abstained")
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["baseline"]["gpt_scope_disagreements"], 1)
        self.assertEqual(report["candidate"]["gpt_comparison_counts"]["model_abstained"], 1)
        self.assertEqual(report["candidate"]["gpt_compared_boundaries"], 0)
        self.assertIsNone(report["candidate"]["gpt_mae_seconds"])

    def test_unmatched_candidates_are_explicit_and_excluded_from_paired_counts(self):
        row, missing = self.manifest(1), self.manifest(2)
        self.write_manifest(self.baseline, [row])
        self.write_manifest(self.candidate, [row, missing])
        self.annotation(self.baseline, row)
        self.annotation(self.candidate, row)
        self.annotation(self.candidate, missing)
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["matching"]["missing_baseline_keys"], [missing["key"]])
        self.assertEqual(report["candidate"]["total"], 1)
        self.assertEqual(report["full_manifest"]["candidate"]["total"], 2)
        self.assertEqual(report["episodes"][1]["acceptance_change"], "unmatched")
        self.assertIsNone(report["episodes"][1]["baseline"])

    def test_metadata_or_reference_changes_exclude_common_errors(self):
        row = self.manifest(1)
        changed = {**row, "source_fps": 10}
        self.write_manifest(self.baseline, [row])
        self.write_manifest(self.candidate, [changed])
        self.annotation(self.baseline, row)
        self.annotation(self.candidate, changed)
        self.review(self.baseline, row)
        self.review(self.candidate, changed)
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["episodes"][0]["metadata_conflicts"], ["source_fps"])
        self.assertEqual(report["paired"]["gpt_common_reference_episodes"], 0)
        self.write_manifest(self.candidate, [row])
        self.review(self.candidate, row, frames=[120, 300, 500])
        report = compare_runs(self.baseline, self.candidate)
        self.assertFalse(report["episodes"][0]["gpt_common_reference"])
        self.assertIn("reference boundaries differ", report["episodes"][0]["gpt_common_reference_reason"])

    def test_stale_comparison_and_malformed_artifact_are_not_scored(self):
        row = self.manifest(1)
        for run in [self.baseline, self.candidate]:
            self.write_manifest(run, [row])
            self.annotation(run, row)
            self.review(run, row)
        self.annotation(self.candidate, row, frames=[120, 300, 500])
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["candidate"]["gpt_accepted_comparable_episodes"], 0)
        self.assertIn("stale GPT comparison", report["episodes"][0]["candidate"]["artifact_warnings"][0])
        (episode_dir(self.candidate, row) / "annotation.json").write_text("{broken")
        report = compare_runs(self.baseline, self.candidate)
        self.assertEqual(report["candidate"]["artifact_read_errors"], 1)
        self.assertEqual(report["candidate"]["execution_errors"], 0)
        self.assertEqual(report["candidate"]["semantic_needs_review"], 0)

    def test_reports_are_only_writes_and_original_review_is_preserved(self):
        row = self.manifest(1)
        for run in [self.baseline, self.candidate]:
            self.write_manifest(run, [row])
            self.annotation(run, row)
            self.review(run, row)
        original = {path: path.read_bytes() for run in [self.baseline, self.candidate] for path in run.rglob("*") if path.is_file()}
        first = compare_runs(self.baseline, self.candidate)
        second = compare_runs(self.baseline, self.candidate)
        self.assertEqual(first["episodes"], second["episodes"])
        for path, value in original.items():
            self.assertEqual(path.read_bytes(), value)
        self.assertFalse((self.baseline / "reports").exists())
        self.assertEqual(sorted(path.name for path in (self.candidate / "reports").iterdir()), ["model_comparison.json", "model_comparison.md"])

    def test_duplicate_keys_fail_instead_of_silently_overwriting_alignment(self):
        row = self.manifest(1)
        self.write_manifest(self.baseline, [row])
        self.write_manifest(self.candidate, [row, deepcopy(row)])
        with self.assertRaisesRegex(ValueError, "Duplicate manifest key"):
            compare_runs(self.baseline, self.candidate)


if __name__ == "__main__":
    unittest.main()
