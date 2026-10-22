"""Pipeline invariants for the user's fixed-node interval selection method."""
from pathlib import Path
import json
from unittest.mock import patch

from PIL import Image
import pytest

from split_cloth.interval_probe import _one, _request_record, build_interval_messages
from split_cloth.util import episode_dir, file_hash, read_json, write_json
from split_cloth.web import _inside
from split_cloth.stage_criteria import load_stage_criteria, validate_stage_criteria


def fixture(tmp_path):
    project = tmp_path
    baseline, run = project / "runs/base", project / "runs/new"
    video, parquet = project / "source.mp4", project / "source.parquet"
    video.write_bytes(b"video")
    parquet.write_bytes(b"parquet")
    (project / "standard.zip").write_bytes(b"standard")
    image = run / "standards/stage.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (224, 224)).save(image)
    write_json(run / "expert.json", {"directory": "reference_expert", "validated": True})
    write_json(run / "reference_expert/pages.json", [])
    write_json(run / "standard_catalog.json", {"standard_sha256": file_hash(project / "standard.zip"),
               "stages": [{"stage_id": i, "reference_image": "standards/stage.png", "description": "fold"} for i in range(1, 5)]})
    m = {"key": "ds__episode_000001", "dataset_id": "ds", "episode_id": "episode_000001", "episode_index": 1,
         "frame_count": 10, "source_fps": 20, "source_video_path": str(video), "parquet_path": str(parquet)}
    write_json(episode_dir(baseline, m) / "input_manifest.json", {**m, "source_video_sha256": file_hash(video)})
    (episode_dir(baseline, m) / "preview.mp4").write_bytes(b"browser-preview")
    nodes = {"alignment_verified": True, "nodes": [{"node_id": "a0001", "source_frame_id": 4, "timestamp_s": .2,
             "changes": [{"arm": "left", "kind": "open", "raw_value": "PRIVATE_RAW_MARKER"}],
             "open_arms": ["left"], "closed_arms": ["right"], "unknown_arms": [], "eligible_stage_end": True}]}
    target_image = episode_dir(run, m) / "interval_montage/i0000.png"
    target_image.parent.mkdir(parents=True)
    target_image.write_bytes(image.read_bytes())
    prepared = {"intervals": [{"interval_id": "i0000", "left_node_id": None, "right_node_id": "a0001",
                 "start_timestamp_s": 0, "end_timestamp_s": .2}],
                "samples": [{"sample_id": "i0000_s0000", "source_frame_id": 0, "source_timestamp_s": 0, "image_path": str(target_image)}],
                "pages": [{"page_id": "i0000_p0000", "interval_id": "i0000", "image_path": str(target_image), "sample_ids": ["i0000_s0000"]}]}
    config = {"nodes": {}, "provider": {"base_url": "https://provider.invalid", "model_id": "test", "max_images": 96}}
    return project, baseline, run, m, nodes, prepared, config


def test_messages_show_intervals_and_fixed_ids_without_raw_state_feedback(tmp_path):
    project, baseline, run, m, nodes, prepared, config = fixture(tmp_path)
    messages = build_interval_messages(run, episode_dir(run, m), nodes, prepared)
    texts = "\n".join(part["text"] for message in messages if isinstance(message["content"], list)
                      for part in message["content"] if part["type"] == "text")
    assert "a0001" in texts and "i0000" in texts
    assert "PRIVATE_RAW_MARKER" not in texts
    assert "422" in messages[0]["content"] and "423" in messages[0]["content"]
    assert "end_frame" in texts  # explicitly forbidden in schema instruction


def test_prepare_only_never_reads_credentials_or_calls_provider(tmp_path):
    project, baseline, run, m, nodes, prepared, config = fixture(tmp_path)
    with patch("split_cloth.interval_probe.decode_video_axis", return_value=[i / 20 for i in range(10)]), \
         patch("split_cloth.interval_probe.prepare_gripper_nodes", return_value=nodes), \
         patch("split_cloth.interval_probe.prepare_interval_montages", return_value=prepared), \
         patch("split_cloth.interval_probe.request_annotation", side_effect=AssertionError("No external model in prepare-only")) as provider:
        row = _one(project, run, baseline, m, config, "hash", prepare_only=True)
    assert row["status"] == "prepared_for_selection"
    assert provider.call_count == 0
    inventory = read_json(episode_dir(run, m) / "payload_inventory.json")
    assert inventory["raw_gripper_values_sent"] is False
    assert inventory["local_frame_refinement"] is False
    # GUI must be able to read the preview while preserving episode isolation.
    preview = episode_dir(run, m) / "preview.mp4"
    assert _inside(episode_dir(run, m), preview).read_bytes() == b"browser-preview"


def test_changed_request_hash_cannot_reuse_stale_model_response(tmp_path):
    messages = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
    provider = {"model_id": "test", "max_images": 96}
    _request_record(tmp_path, messages, provider)
    messages[0]["content"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="Changed"):
        _request_record(tmp_path, messages, provider)


def test_image_budget_never_discards_intervals(tmp_path):
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}] * 3}]
    with pytest.raises(ValueError, match="no intervals dropped"):
        _request_record(tmp_path, messages, {"model_id": "test", "max_images": 2})


def test_preflight_failure_is_recorded_without_breaking_entire_batch(tmp_path):
    project, baseline, run, m, nodes, prepared, config = fixture(tmp_path)
    Path(m["source_video_path"]).write_bytes(b"changed-video")
    with patch("split_cloth.interval_probe.request_annotation", side_effect=AssertionError("No API after changed source")) as provider:
        row = _one(project, run, baseline, m, config, "hash")
    assert row["status"] == "needs_review" and provider.call_count == 0
    annotation = read_json(episode_dir(run, m) / "annotation.json")
    assert annotation["execution_error"] == "ValueError"
    assert annotation["training_ready"] is False


def text_fixture(tmp_path):
    values = fixture(tmp_path)
    project, baseline, run, m, nodes, prepared, config = values
    criteria = load_stage_criteria(Path(__file__).parents[1] / "configs/fold_cloth_text_criteria_v1.json")
    write_json(run / "stage_criteria.json", criteria)
    (run / "expert.json").unlink()
    (run / "standards/stage.png").unlink()
    config.update(reference_mode="text_stage_criteria", temporal_order=True)
    config["nodes"]["node_policy"] = "any_open"
    return values


def test_text_mode_contains_only_target_images_and_no_expert_timings(tmp_path):
    project, baseline, run, m, nodes, prepared, config = text_fixture(tmp_path)
    messages = build_interval_messages(run, episode_dir(run, m), nodes, prepared)
    content = messages[1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    texts = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert len(images) == len(prepared["pages"]) == 1
    assert "REFERENCE_TEXT_CRITERIA" in texts
    assert "REFERENCE_STANDARD" not in texts
    assert "reference_expert/montage" not in texts
    assert "source_images" not in texts and "boundary_frames" not in texts
    assert "第一侧" in texts and "temporal_rules" in texts
    assert "每次稳定张开事件" in messages[0]["content"]


def test_text_prepare_has_no_reference_images_or_provider_calls(tmp_path):
    project, baseline, run, m, nodes, prepared, config = text_fixture(tmp_path)
    with patch("split_cloth.interval_probe.decode_video_axis", return_value=[i / 20 for i in range(10)]), \
         patch("split_cloth.interval_probe.prepare_gripper_nodes", return_value=nodes), \
         patch("split_cloth.interval_probe.prepare_interval_montages", return_value=prepared), \
         patch("split_cloth.interval_probe.request_annotation", side_effect=AssertionError("No provider")):
        row = _one(project, run, baseline, m, config, "hash", prepare_only=True)
    assert row["status"] == "prepared_for_selection" and row["images"] == 1
    inventory = read_json(episode_dir(run, m) / "payload_inventory.json")
    assert inventory["reference_image_count"] == 0
    assert inventory["node_policy"] == "any_open"
    assert inventory["temporal_order_validation"] is True
    assert len(inventory["image_files"]) == 1


def test_changed_text_criteria_requires_new_run(tmp_path):
    project, baseline, run, m, nodes, prepared, config = text_fixture(tmp_path)
    with patch("split_cloth.interval_probe.decode_video_axis", return_value=[i / 20 for i in range(10)]), \
         patch("split_cloth.interval_probe.prepare_gripper_nodes", return_value=nodes), \
         patch("split_cloth.interval_probe.prepare_interval_montages", return_value=prepared), \
         patch("split_cloth.interval_probe.request_annotation", side_effect=AssertionError("No provider")):
        assert _one(project, run, baseline, m, config, "hash", prepare_only=True)["status"] == "prepared_for_selection"
        criteria = read_json(run / "stage_criteria.json")
        criteria["stages"][0]["action"] += " updated"
        write_json(run / "stage_criteria.json", criteria)
        row = _one(project, run, baseline, m, config, "hash", prepare_only=True)
    assert row["status"] == "input_conflict"


@pytest.mark.parametrize("change", ["missing_stage", "stage_order", "empty_completion", "missing_provenance"])
def test_invalid_text_criteria_fail_before_inference(change):
    criteria = load_stage_criteria(Path(__file__).parents[1] / "configs/fold_cloth_text_criteria_v1.json")
    if change == "missing_stage":
        criteria["stages"].pop()
    elif change == "stage_order":
        criteria["stages"].reverse()
    elif change == "empty_completion":
        criteria["stages"][0]["completion_criteria"] = []
    else:
        criteria.pop("provenance")
    with pytest.raises(ValueError):
        validate_stage_criteria(criteria)
