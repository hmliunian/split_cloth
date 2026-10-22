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
