"""Contract checks for frame claims, label acceptance and isolated API caches."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from split_cloth.event_probe import (
    _call, _one, collect_usage, make_annotation, validate_refinement, validate_selection,
)
from split_cloth.boundaries import expand_labels
from split_cloth.util import episode_dir, file_hash, write_json, write_jsonl


def selection():
    return {
        "task_type": "fold_cloth", "applicable": True, "all_stages_observed": True,
        "needs_review": False, "review_reasons": [],
        "boundaries": [{"from_stage": i, "candidate_event_ids": [f"e{i}"],
                        "participant_arms": ["left"], "evidence_text": "Completed and released"} for i in range(1, 4)],
        "stage4_completion": {"observed": True, "completion_sample_id": "s4", "evidence_text": "Removed"},
    }


def refinement(end=22):
    return {"end_frame": end, "previous_not_complete_frame": end - 1,
            "operation_complete": True, "participating_grippers_released": True,
            "first_complete_confirmed": True, "next_frame_consistent": True,
            "evidence_text": "f-1 closed, f released, f+1 withdraws", "review_reasons": []}


class EventProbeContracts(unittest.TestCase):
    def setUp(self):
        self.events = [{"event_id": f"e{i}", "arm": "left", "source_frame_id": i * 20} for i in range(1, 4)]
        self.samples = [{"sample_id": "s4", "source_frame_id": 80}]

    def test_event_reference_must_exist_and_participate(self):
        good = selection()
        validate_selection(good, self.events, self.samples)
        good["boundaries"][0]["participant_arms"] = ["right"]
        with self.assertRaisesRegex(ValueError, "participant"):
            validate_selection(good, self.events, self.samples)

    def test_dense_model_cannot_invent_unseen_frame(self):
        with self.assertRaisesRegex(ValueError, "not supplied"):
            validate_refinement(refinement(422), list(range(420, 422)))

    def test_first_frame_claim_requires_both_neighbors(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_refinement(refinement(22), [21, 22])
        validate_refinement(refinement(22), [21, 22, 23])

    def test_422_stays_previous_stage_and_423_begins_next(self):
        rows = [{**refinement(f), "visual_confirmed": True} for f in (422, 500, 600)]
        a = make_annotation({"frame_count": 800}, selection(), rows, [{"sample_id": "s4", "source_frame_id": 700}])
        self.assertEqual(a["status"], "accepted")
        labels = expand_labels(a)
        self.assertEqual((labels[421], labels[422], labels[423]), (1, 1, 2))
        self.assertFalse(a["training_ready"])

    def test_unconfirmed_frame_never_creates_training_labels(self):
        rows = [{**refinement(f), "visual_confirmed": f != 40} for f in (20, 40, 60)]
        a = make_annotation({"frame_count": 100}, selection(), rows, self.samples)
        self.assertEqual(a["status"], "needs_review")
        self.assertTrue((expand_labels(a) == -1).all())

    def test_scope_negative_cannot_be_forced_into_four_stages(self):
        s = selection()
        s.update(task_type="hang_cloth", applicable=False)
        a = make_annotation({"frame_count": 100}, s, [], self.samples)
        self.assertEqual(a["status"], "out_of_scope")
        self.assertTrue((expand_labels(a) == -1).all())

    def test_unknown_task_is_reviewed_instead_of_definitively_excluded(self):
        s = selection()
        s.update(task_type="unknown", applicable=False)
        a = make_annotation({"frame_count": 100}, s, [], self.samples)
        self.assertEqual(a["status"], "needs_review")

    def test_request_cache_rejects_changed_images_or_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            response = {"content": "{}", "usage": {}}
            with patch("split_cloth.event_probe.request_annotation", return_value=response) as provider:
                messages = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
                _call(directory, Path(directory) / "selection", messages, {"model_id": "test"})
                messages[0]["content"][0]["text"] = "changed"
                with self.assertRaisesRegex(ValueError, "Changed request"):
                    _call(directory, Path(directory) / "selection", messages, {"model_id": "test"})
                self.assertEqual(provider.call_count, 1)

    def test_token_accounting_includes_paid_invalid_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            for relative in ("selection/response.json", "refine_stage_1/invalid_responses/response_x.json"):
                write_json(Path(directory) / relative, {"content": "{}", "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}})
            write_json(Path(directory) / "selection/attempts.json", [{}, {}])
            totals = collect_usage(directory)
            self.assertEqual(totals["usage"]["total_tokens"], 26)
            self.assertEqual(totals["attempt_count"], 2)

    def test_prepare_only_writes_payload_inventory_without_provider_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, run = root / "baseline", root / "probe"
            video, parquet, wrist, reference = [root / name for name in ("video", "parquet", "wrist", "reference.png")]
            for path in (video, parquet, wrist, reference):
                path.write_bytes(b"source-fixture")
            manifest = {"key": "dataset__episode_000001", "dataset_id": "dataset", "episode_id": "episode_000001",
                        "episode_index": 1, "frame_count": 100, "source_fps": 20,
                        "source_video_path": str(video), "parquet_path": str(parquet)}
            source = episode_dir(baseline, manifest)
            write_json(source / "input_manifest.json", {"source_video_sha256": file_hash(video), "input_hash": "context"})
            write_json(source / "pages.json", [])
            write_jsonl(source / "samples.jsonl", [])
            write_json(run / "expert.json", {"directory": "reference_expert"})
            write_json(run / "reference_expert/pages.json", [])
            write_json(run / "standard_catalog.json", {"stages": [{"reference_image": str(reference)}]})
            candidates = {"alignment_verified": True, "events": [], "diagnostics": [],
                          "calibration": {"row_count": 100, "invalid_state_rows": 0}}
            with patch("split_cloth.event_probe._camera_path", return_value=wrist), \
                 patch("split_cloth.event_probe.decode_video_axis", return_value=[i / 20 for i in range(100)]), \
                 patch("split_cloth.event_probe.detect_gripper_event_candidates_from_parquet", return_value=candidates), \
                 patch("split_cloth.event_probe.extract_source_frames", return_value=[]), \
                 patch("split_cloth.event_probe.render_frame_pages", return_value=[]), \
                 patch("split_cloth.event_probe._selection_messages", return_value=[{"role": "user", "content": [{"type": "text", "text": "prepared"}]}]), \
                 patch("split_cloth.event_probe.request_annotation", side_effect=AssertionError("Provider access forbidden during preparation")) as provider:
                result = _one(root, run, baseline, manifest, {"events": {"detector": {}}, "provider": {"model_id": "test", "base_url": "https://provider.invalid"}}, "code", prepare_only=True)
            self.assertEqual(result["status"], "prepared_for_selection")
            self.assertEqual(provider.call_count, 0)
            self.assertTrue((episode_dir(run, manifest) / "payload_inventory.json").exists())


if __name__ == "__main__":
    unittest.main()
