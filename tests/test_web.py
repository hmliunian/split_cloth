import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from split_cloth.web import create_app


@pytest.fixture
def fixture_run(tmp_path):
    run = tmp_path / "runs" / "trial-100"
    episode = run / "episodes" / "fold" / "episode_000003"
    episode.mkdir(parents=True)
    manifest = {"key": "fold__episode_000003", "dataset_id": "fold", "episode_index": 3,
                "episode_id": "episode_000003", "frame_count": 100, "source_fps": 20,
                "source_video_path": str(tmp_path / "missing.av1.mp4")}
    (run / "run.json").write_text(json.dumps({"description": "trial"}))
    (run / "manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    (episode / "input_manifest.json").write_text(json.dumps(manifest))
    (episode / "annotation.json").write_text(json.dumps({"status": "accepted", "boundary_frames": [20, 40, 60], "stage4_completion_frame": 80}))
    (episode / "samples.jsonl").write_text(json.dumps({"sample_id": "s0", "source_frame_id": 0}) + "\n")
    return TestClient(create_app(tmp_path)), episode, run, manifest


def payload(**updates):
    return {"run_id": "trial-100", "key": "fold__episode_000003", "status": "approved", "reviewer": "tester", "version": 0, **updates}


def test_approval_uses_annotation_keeps_original_and_expands_closed_intervals(fixture_run):
    client, episode, _, _ = fixture_run
    original = (episode / "annotation.json").read_bytes()
    response = client.post("/api/review", json=payload())
    assert response.status_code == 200, response.text
    assert response.json()["version"] == 1
    labels = np.load(episode / "stage_reviewed.npy", allow_pickle=False)
    assert labels.tolist() == [1] * 21 + [2] * 20 + [3] * 20 + [4] * 39
    assert (episode / "annotation.json").read_bytes() == original
    assert client.get("/api/episodes", params={"run_id": "trial-100"}).json()["summary"]["reviewed"] == 1


@pytest.mark.parametrize("boundaries,completion", [
    ([True, 40, 60], 80), (["20", 40, 60], 80), ([20, 20, 60], 80),
    ([20, 40, 99], None), ([-1, 40, 60], 80), ([20, 40], 80),
    ([20, 40, 60], 60), ([20, 40, 60], 100), ([20, 40, 60], True),
])
def test_invalid_review_frames_are_rejected(fixture_run, boundaries, completion):
    client, episode, _, _ = fixture_run
    response = client.post("/api/review", json=payload(boundary_frames=boundaries, stage4_completion_frame=completion))
    assert response.status_code == 422, response.text
    assert not (episode / "review.json").exists()
    assert not (episode / "stage_reviewed.npy").exists()


def test_optimistic_lock_and_append_history(fixture_run):
    client, episode, _, _ = fixture_run
    assert client.post("/api/review", json=payload()).status_code == 200
    assert client.post("/api/review", json=payload(note="stale")).status_code == 409
    corrected = client.post("/api/review", json=payload(version=1, status="corrected", boundary_frames=[21, 42, 63], note="fixed"))
    assert corrected.status_code == 200
    assert corrected.json()["review"]["history"][0]["boundary_frames"] == [20, 40, 60]
    history = (episode / "review_history.jsonl").read_text().splitlines()
    assert [json.loads(row)["version"] for row in history] == [1, 2]


def test_assets_prevent_path_traversal_symlink_and_non_image(fixture_run, tmp_path):
    client, _, run, _ = fixture_run
    outside = tmp_path / "private.png"
    outside.write_bytes(b"secret")
    (run / "escape.png").symlink_to(outside)
    for path in ("../../private.png", str(outside), "escape.png"):
        response = client.get("/api/assets", params={"run_id": "trial-100", "path": path})
        assert response.status_code == 400
    assert client.get("/api/assets", params={"run_id": "trial-100", "path": "run.json"}).status_code == 404
    assert client.get("/api/episodes", params={"run_id": "../trial-100"}).status_code == 400


def test_missing_preview_is_not_replaced_by_source_av1(fixture_run):
    client, _, _, _ = fixture_run
    result = client.get("/api/episode", params={"run_id": "trial-100", "key": "fold__episode_000003"})
    assert result.status_code == 200
    assert result.json()["video_url"] is None
    assert result.json()["samples"][0]["source_frame_id"] == 0
    assert client.get("/api/video", params={"run_id": "trial-100", "key": "fold__episode_000003"}).status_code == 404


def test_preview_byte_ranges_and_frame_bounds(fixture_run):
    client, episode, _, _ = fixture_run
    (episode / "preview.mp4").write_bytes(b"0123456789")
    params = {"run_id": "trial-100", "key": "fold__episode_000003"}
    response = client.get("/api/video", params=params, headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert client.get("/api/video", params=params, headers={"Range": "bytes=20-30"}).status_code == 416
    assert client.get("/api/frame", params={**params, "frame": 100}).status_code == 422


def test_out_of_scope_can_be_reviewed_without_fabricating_boundaries(fixture_run):
    client, episode, _, _ = fixture_run
    (episode / "annotation.json").write_text(json.dumps({"status": "out_of_scope", "boundary_frames": [], "stage4_completion_frame": None}))
    result = client.post("/api/review", json=payload(status="out_of_scope"))
    assert result.status_code == 200
    assert not (episode / "stage_reviewed.npy").exists()


def test_approval_requires_final_stage_evidence(fixture_run):
    client, episode, _, _ = fixture_run
    result = client.post("/api/review", json=payload(stage4_completion_frame=None))
    assert result.status_code == 422
    assert not (episode / "review.json").exists()


def test_execution_failure_is_separate_from_model_abstention_in_gui(fixture_run):
    client, episode, _, _ = fixture_run
    (episode / "annotation.json").write_text(json.dumps({"status": "needs_review", "execution_error": "RuntimeError"}))
    response = client.get("/api/episodes", params={"run_id": "trial-100", "status": "failed"})
    assert response.status_code == 200
    assert len(response.json()["episodes"]) == 1
    assert response.json()["summary"]["failed"] == 1
    assert response.json()["summary"]["needs_review"] == 0
    assert response.json()["episodes"][0]["status"] == "failed"


def text_stage_criteria():
    return {
        "schema_version": "text_stage_criteria_v1",
        "stages": [{"stage_id": 1, "name": "第一侧纵折", "action": "将第一侧实际折入",
                    "completion_criteria": ["折边形成且参与夹爪完全松开"],
                    "exclude": ["提起、抖开、铺平不能算第一折"]}],
        "temporal_rules": ["准备之后才开始第一侧纵折"],
        "boundary_rule": "松开帧属于前阶段，下一帧属于后阶段",
        "provenance": {"method": "GPT image summary", "model": "gpt-test", "source_document": "standard.zip 5.2"},
        "limitations": ["图像归纳不是人工金标"],
    }


@pytest.mark.parametrize("legacy_expert", [False, True])
def test_text_criteria_reference_takes_priority_without_reading_image_expert(fixture_run, legacy_expert):
    client, _, run, manifest = fixture_run
    criteria = text_stage_criteria()
    (run / "stage_criteria.json").write_text(json.dumps(criteria, ensure_ascii=False))
    # Parsing this old image expert would fail: text mode must return first.
    if legacy_expert:
        (run / "expert.json").write_text("invalid legacy image expert")
    response = client.get("/api/expert", params={"run_id": run.name})
    assert response.status_code == 200, response.text
    reference = response.json()
    assert reference["reference_mode"] == "text_stage_criteria"
    assert reference["stage_criteria"] == criteria
    assert reference["provenance"] == criteria["provenance"]
    assert reference["validation_level"] == "gpt_visual_summary_not_human_gold"
    assert reference["pages"] == reference["page_urls"] == []
    assert "directory" not in reference and "annotation" not in reference
    detail = client.get("/api/episode", params={"run_id": run.name, "key": manifest["key"]})
    assert detail.status_code == 200, detail.text
    assert detail.json()["expert"] == reference


def test_legacy_image_expert_is_unchanged_without_text_criteria(fixture_run):
    client, _, run, _ = fixture_run
    directory = run / "expert"
    directory.mkdir()
    (directory / "legacy.png").write_bytes(b"image")
    (directory / "pages.json").write_text(json.dumps([{"page_id": "p0", "image_path": "legacy.png"}]))
    (run / "expert.json").write_text(json.dumps({"name": "Legacy expert", "directory": "expert"}))
    response = client.get("/api/expert", params={"run_id": run.name})
    assert response.status_code == 200, response.text
    reference = response.json()
    assert reference["name"] == "Legacy expert"
    assert "stage_criteria" not in reference
    assert len(reference["pages"]) == 1
    assert "legacy.png" in reference["pages"][0]["url"]
    assert reference["page_urls"] == [reference["pages"][0]["url"]]


@pytest.mark.parametrize("criteria", [[], {}, {"schema_version": "unknown", "stages": []},
                                       {"schema_version": "text_stage_criteria_v1", "stages": None}])
def test_invalid_text_criteria_does_not_silently_load_image_reference(fixture_run, criteria):
    client, _, run, _ = fixture_run
    (run / "stage_criteria.json").write_text(json.dumps(criteria))
    (run / "expert.json").write_text(json.dumps({"pages": ["old.png"]}))
    response = client.get("/api/expert", params={"run_id": run.name})
    assert response.status_code == 500
    assert "Invalid text stage criteria" in response.json()["detail"]


def test_text_criteria_reference_cannot_escape_run_via_symlink(fixture_run):
    client, _, run, _ = fixture_run
    outside = run.parent.parent / "outside_criteria.json"
    outside.write_text(json.dumps(text_stage_criteria()))
    (run / "stage_criteria.json").symlink_to(outside)
    response = client.get("/api/expert", params={"run_id": run.name})
    assert response.status_code == 400
