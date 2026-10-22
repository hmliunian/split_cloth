"""Fixed event frames determine intervals; sampling/page geometry cannot move them."""
from pathlib import Path

from PIL import Image
import pytest

from split_cloth.interval_montage import build_intervals, prepare_interval_montages
from split_cloth.util import read_json, read_jsonl


def node(frame, node_id=None, node_type="release"):
    return {"node_id": node_id or f"a{frame}", "source_frame_id": frame, "node_type": node_type}


def test_interval_partition_has_no_overlap_or_gap_and_resets_one_fps_clock():
    axis = [frame / 20 for frame in range(60)]
    intervals = build_intervals([node(50), node(13)], axis)
    assert [(item["start_frame"], item["end_frame"]) for item in intervals] == [(0, 13), (14, 50), (51, 59)]
    assert [(item["left_node_id"], item["right_node_id"]) for item in intervals] == [(None, "a13"), ("a13", "a50"), ("a50", None)]
    assert [item["sample_frame_ids"] for item in intervals] == [[0, 13], [14, 34, 50], [51, 59]]
    assert [item["interval_id"] for item in intervals] == ["i0000", "i0001", "i0002"]
    flattened = [frame for item in intervals for frame in range(item["start_frame"], item["end_frame"] + 1)]
    assert flattened == list(range(60))


@pytest.mark.parametrize("frame_count, nodes, expected", [
    (1, [], [(0, 0)]),
    (1, [node(0)], [(0, 0)]),
    (5, [], [(0, 4)]),
    (5, [node(0)], [(0, 0), (1, 4)]),
    (5, [node(4)], [(0, 4)]),
    (5, [node(0), node(4)], [(0, 0), (1, 4)]),
])
def test_endpoint_nodes_never_create_empty_intervals(frame_count, nodes, expected):
    intervals = build_intervals(nodes, [frame / 20 for frame in range(frame_count)])
    assert [(item["start_frame"], item["end_frame"]) for item in intervals] == expected
    assert all(item["sample_frame_ids"] for item in intervals)


def test_pts_nearest_frame_and_earlier_tie_are_used_instead_of_grid_index():
    axis = [0, .2, .8, 1.2, 1.8, 2.2, 2.3]
    intervals = build_intervals([], axis)
    assert intervals[0]["sample_frame_ids"] == [0, 2, 4, 6]
    shifted = build_intervals([node(1)], axis)
    assert shifted[1]["sample_frame_ids"] == [2, 4, 6]


@pytest.mark.parametrize("nodes, axis, sample_fps", [
    ([node(1, "same"), node(2, "same")], [0, .1, .2], 1),
    ([node(1), node(1, "other")], [0, .1], 1),
    ([node(-1)], [0, .1], 1),
    ([node(2)], [0, .1], 1),
    ([node(True)], [0, .1], 1),
    ([node(1.5)], [0, .1, .2], 1),
    ([{"source_frame_id": 0}], [0], 1),
    ([], [], 1),
    ([], [0, .1, .1], 1),
    ([], [0, float("nan")], 1),
    ([], [-.1, 0], 1),
    ([], [0, .1], 0),
    ([], [0, .1], float("inf")),
])
def test_invalid_node_axis_or_sampling_is_rejected(nodes, axis, sample_fps):
    with pytest.raises(ValueError):
        build_intervals(nodes, axis, sample_fps)


def fake_extract(monkeypatch, calls):
    def extract(path, frame_ids, output, fps, source_timestamps=None):
        calls.append({"path": path, "frame_ids": list(frame_ids), "fps": fps,
                      "source_timestamps": source_timestamps})
        Path(output).mkdir(parents=True, exist_ok=True)
        rows = []
        for frame in frame_ids:
            image = Path(output) / f"f{frame}.png"
            Image.new("RGB", (224, 224), (frame % 255, 100, 100)).save(image)
            rows.append({"source_frame_id": frame, "source_timestamp_s": source_timestamps[frame],
                         "image_path": str(image.resolve())})
        return rows
    monkeypatch.setattr("split_cloth.interval_montage.extract_source_frames", extract)


def test_short_interval_pages_are_compact_and_never_mix_intervals(monkeypatch, tmp_path):
    calls = []
    fake_extract(monkeypatch, calls)
    axis = [frame / 20 for frame in range(12)]
    manifest = {"source_video_path": "source.mp4", "source_fps": 20, "frame_count": len(axis)}
    result = prepare_interval_montages(manifest, [node(3), node(8, node_type="close")], axis, tmp_path,
                                       {"columns": 3, "rows": 4, "overlap_samples": 2})
    assert len(calls) == 1
    assert calls[0]["frame_ids"] == [0, 3, 4, 8, 9, 11]
    assert calls[0]["source_timestamps"] == axis
    assert [page["frame_ids"] for page in result["pages"]] == [[0, 3], [4, 8], [9, 11]]
    assert [page["right_node_type"] for page in result["pages"]] == ["release", "close", None]
    for index, page in enumerate(result["pages"]):
        assert page["interval_id"] == f"i{index:04d}"
        assert len(page["sample_ids"]) == 2
        assert Path(page["image_path"]).is_absolute()
        with Image.open(page["image_path"]) as image:
            # Two 224-pixel cells, one actual row: no fixed 12-cell padding.
            assert image.width == 2 * 224 + 3 * 12
            assert image.height < 600
    assert read_json(tmp_path / "intervals.json") == result["intervals"]
    assert read_json(tmp_path / "interval_pages.json") == result["pages"]
    assert read_jsonl(tmp_path / "interval_samples.jsonl") == result["samples"]
    assert all(Path(sample["image_path"]).is_absolute() for sample in result["samples"])
    assert [sample["source_frame_id"] for sample in result["samples"] if sample["is_right_node"]] == [3, 8]


def test_long_interval_paginates_in_order_without_overlap_or_extra_samples(monkeypatch, tmp_path):
    calls = []
    fake_extract(monkeypatch, calls)
    axis = [frame / 10 for frame in range(141)]
    manifest = {"source_video_path": "source.mp4", "source_fps": 10, "frame_count": len(axis)}
    result = prepare_interval_montages(manifest, [], axis, tmp_path, {"columns": 3, "rows": 4})
    assert [len(page["frame_ids"]) for page in result["pages"]] == [12, 3]
    assert [frame for page in result["pages"] for frame in page["frame_ids"]] == list(range(0, 141, 10))
    assert all(page["interval_id"] == "i0000" for page in result["pages"])
    assert len(calls) == 1


def test_manifest_time_axis_mismatch_is_rejected_before_decode(monkeypatch, tmp_path):
    calls = []
    fake_extract(monkeypatch, calls)
    with pytest.raises(ValueError, match="count mismatch"):
        prepare_interval_montages({"frame_count": 2}, [], [0], tmp_path)
    assert not calls


def test_decoder_alignment_failure_is_propagated_without_publishing_indexes(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("Source PTS exceeds 0.5-frame CFR tolerance")
    monkeypatch.setattr("split_cloth.interval_montage.extract_source_frames", fail)
    manifest = {"source_video_path": "source.mp4", "source_fps": 20, "frame_count": 2}
    with pytest.raises(ValueError, match="Source PTS"):
        prepare_interval_montages(manifest, [], [0, .1], tmp_path)
    assert not (tmp_path / "intervals.json").exists()
