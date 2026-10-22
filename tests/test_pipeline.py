"""Offline pipeline regression tests: no model requests or real videos.

These exercise the riskier resume paths in addition to a successful contract
round trip. Runtime dependencies are installed in the server environment.
"""

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("numpy", "av", "PIL", "httpx"))
if HAS_RUNTIME:
    from split_cloth.pipeline import annotate_run, prepare_run
    from split_cloth.sampling import prepare_episode
    from split_cloth.util import episode_dir, object_hash, read_json, write_json, write_jsonl


def model_output():
    return {
        "task_type": "fold_cloth", "applicable": True,
        "all_stages_observed": True, "needs_review": False,
        "review_reasons": [],
        "stage4_completion": {
            "observed": True, "completion_sample_id": "s000004",
            "evidence_text": "Fold completed and both grippers open.",
        },
        "boundary_proposals": [
            {
                "from_stage": i + 1, "to_stage": i + 2,
                "last_not_complete_sample_id": f"s{i:06d}",
                "first_complete_sample_id": f"s{i + 1:06d}",
                "evidence_text": "The current fold has completed and released.",
            }
            for i in range(3)
        ],
    }


@unittest.skipUnless(HAS_RUNTIME, "Requires the installed server runtime (NumPy, PyAV, Pillow, httpx)")
class PipelineResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.run = self.project / "runs" / "test"
        self.manifest = {
            "key": "fold_cloth_v3__episode_000001", "dataset_id": "fold_cloth_v3",
            "episode_id": "episode_000001", "episode_index": 1,
            "frame_count": 100, "source_fps": 20, "split": "heldout",
        }
        self.output = episode_dir(self.run, self.manifest)
        self.output.mkdir(parents=True)
        self.prepared = {**self.manifest, "input_hash": "input-v1", "page_count": 1}
        self.config = {
            "spec_version": "coarse_v1.1_expert",
            "provider": {"model_id": "qwen-test", "max_inflight": 1},
        }
        write_jsonl(self.run / "manifest.jsonl", [self.manifest])
        write_json(self.run / "expert.json", {"validated": True, "key": "expert", "boundary_frames": [20, 40, 60]})
        write_json(self.run / "standard_catalog.json", {"stages": [{"stage_id": i} for i in range(1, 5)]})
        write_json(self.output / "input_manifest.json", self.prepared)
        write_jsonl(self.output / "samples.jsonl", [
            {"sample_id": f"s{i:06d}", "sample_seq": i,
             "source_frame_id": frame, "source_timestamp_s": frame / 20}
            for i, frame in enumerate((0, 20, 40, 60, 80, 99))
        ])
        self.response = {
            "content": json.dumps(model_output()), "model": "qwen-test",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            "elapsed_s": .01, "finish_reason": "stop",
        }
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("split_cloth.pipeline.build_messages", return_value=[
            {"role": "system", "content": "test prompt"},
            {"role": "user", "content": [{"type": "text", "text": "test target"}]},
        ]))
        self.request = self.stack.enter_context(patch(
            "split_cloth.pipeline.request_annotation", return_value=self.response,
        ))

    def annotate(self):
        return annotate_run(self.project, self.run, self.config)

    def test_accepted_round_trip_preserves_original_frame_mapping(self):
        import numpy as np
        summary = self.annotate()
        annotation = read_json(self.output / "annotation.json")
        labels = np.load(self.output / "stage.npy")
        self.assertEqual(summary["counts"], {"accepted": 1})
        self.assertEqual(annotation["boundary_frames"], [20, 40, 60])
        self.assertEqual(annotation["stage4_completion_frame"], 80)
        self.assertEqual((labels[20], labels[21], labels[60], labels[61], labels[99]), (1, 2, 3, 4, 4))
        self.annotate()
        self.assertEqual(self.request.call_count, 1, "An unchanged successful request should resume without billing twice")

    def test_model_uncertainty_never_produces_positive_labels(self):
        import numpy as np
        output = model_output()
        output["task_type"] = "unknown"
        self.request.return_value = {**self.response, "content": json.dumps(output)}
        summary = self.annotate()
        self.assertEqual(summary["counts"], {"needs_review": 1})
        self.assertTrue(np.all(np.load(self.output / "stage.npy") == -1))

    def test_transient_execution_failure_is_retried_on_resume(self):
        self.request.side_effect = [RuntimeError("provider_http_503"), self.response]
        first = self.annotate()
        self.assertEqual(first["counts"], {"needs_review": 1})
        second = self.annotate()
        self.assertEqual(self.request.call_count, 2, "An execution failure must not become a permanent completed cache entry")
        self.assertEqual(second["counts"], {"accepted": 1})

    def test_changed_input_fingerprint_cannot_report_old_accepted_labels(self):
        self.annotate()
        changed = deepcopy(self.prepared)
        changed["input_hash"] = "input-v2"
        write_json(self.output / "input_manifest.json", changed)
        summary = self.annotate()
        self.assertEqual(self.request.call_count, 1, "A conflicting old response should not be silently re-used")
        self.assertNotIn("accepted", summary["counts"], "A failed new input cannot retain an old accepted status in the final summary")
        annotation = read_json(self.output / "annotation.json")
        self.assertNotEqual(annotation["status"], "accepted")

    def test_completed_cache_repairs_a_missing_label_artifact(self):
        self.annotate()
        (self.output / "stage.npy").unlink()
        self.annotate()
        self.assertTrue((self.output / "stage.npy").exists(), "A cached status is valid only if the exported labels still exist")
        self.assertEqual(self.request.call_count, 1, "Labels can be regenerated from the persisted annotation")

    def test_completed_cache_does_not_crash_when_annotation_file_is_missing(self):
        self.annotate()
        (self.output / "annotation.json").unlink()
        self.annotate()
        self.assertTrue((self.output / "annotation.json").exists(), "Partial artifacts must be recovered or explicitly invalidated")

    def test_successful_repreparation_recovers_previously_unreadable_episode(self):
        write_json(self.output / "annotation.json", {
            **self.manifest, "status": "unreadable", "needs_review": False,
            "review_reasons": ["temporary video access failure"],
            "segments": [{"stage": -1, "start_frame": 0, "end_frame": 99}],
        })
        with patch("split_cloth.pipeline.prepare_episode", return_value=self.prepared):
            prepare_run(self.project, self.run, self.config)
        summary = self.annotate()
        self.assertEqual(summary["counts"], {"accepted": 1}, "Successful preparation must invalidate an obsolete unreadable status")


@unittest.skipUnless(HAS_RUNTIME, "Requires the installed server runtime (NumPy, PyAV, Pillow, httpx)")
class PreparationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"fake-video-cache-fixture")
        self.out = self.root / "prepared"
        self.out.mkdir()
        self.config = {
            "sample_fps": 1.0,
            "montage": {"columns": 3, "rows": 4, "overlap_samples": 2,
                "tile_image_size": [224, 224], "tile_label_height_px": 36,
                "page_header_height_px": 48, "gutter_px": 8},
        }
        self.manifest = {
            "source_video_path": str(self.source), "source_video_realpath": str(self.source),
            "source_fps": 20, "frame_count": 100,
            "dataset_id": "test", "episode_id": "episode_000001",
        }
        signature = {
            "size": self.source.stat().st_size, "mtime_ns": self.source.stat().st_mtime_ns,
            "sample_fps": 1.0, "layout": self.config["montage"], "frame_count": 100,
        }
        self.old = {**self.manifest, "preparation_signature": object_hash(signature), "input_hash": "old"}
        write_json(self.out / "input_manifest.json", self.old)
        write_json(self.out / "pages.json", [])
        # An invalidated cache must reach the decoder rather than return old input.
        self.decoder = patch("split_cloth.sampling.av.open", side_effect=RuntimeError("cache_invalidated"))
        self.decoder.start()
        self.addCleanup(self.decoder.stop)

    def test_missing_sample_registry_invalidates_preparation_cache(self):
        with self.assertRaisesRegex(RuntimeError, "cache_invalidated"):
            prepare_episode(self.manifest, self.out, self.config, preview=False)

    def test_changed_source_fps_invalidates_frame_mapping_cache(self):
        changed = {**self.manifest, "source_fps": 10}
        with self.assertRaisesRegex(RuntimeError, "cache_invalidated"):
            prepare_episode(changed, self.out, self.config, preview=False)


if __name__ == "__main__":
    unittest.main()
