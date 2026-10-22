from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

av = pytest.importorskip("av")
Image = pytest.importorskip("PIL.Image")

from split_cloth import event_visuals
from split_cloth.event_visuals import decode_video_axis, extract_source_frames, render_frame_pages
from split_cloth.util import file_hash


class Frame:
    def __init__(self, pts, color=(255, 0, 0), time_base=Fraction(1, 1000)):
        self.pts = pts
        self.time_base = time_base
        self.color = color

    def to_image(self):
        return Image.new("RGB", (224, 224), self.color)


def fake_video(monkeypatch, frames):
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(thread_type=None)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, stream):
            return iter(frames)

    monkeypatch.setattr(event_visuals.av, "open", lambda path: Container())


def test_axis_uses_actual_pts_with_local_zero(monkeypatch):
    fake_video(monkeypatch, [Frame(10000), Frame(10100), Frame(10249)])
    assert decode_video_axis("video.mkv", 3, 10) == pytest.approx([0, .1, .249])


@pytest.mark.parametrize("frames, expected, message", [
    ([Frame(None)], 1, "Missing source PTS"),
    ([Frame(0, time_base=None)], 1, "Missing source PTS"),
    ([Frame(0), Frame(0)], 2, "Nonmonotonic"),
    ([Frame(100), Frame(99)], 2, "Nonmonotonic"),
    ([Frame(0), Frame(100), Frame(251)], 3, "0.5-frame"),
    ([Frame(0)], 2, "frame count mismatch"),
    ([], 1, "no decoded frames"),
])
def test_axis_rejects_unverified_time_axes(monkeypatch, frames, expected, message):
    fake_video(monkeypatch, frames)
    with pytest.raises(ValueError, match=message):
        decode_video_axis("video.mkv", expected, 10)


def test_axis_accepts_exactly_half_a_frame_error(monkeypatch):
    fake_video(monkeypatch, [Frame(0), Frame(100), Frame(250)])
    assert decode_video_axis("video.mkv", 3, 10)[-1] == .25


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan")])
def test_axis_rejects_invalid_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        decode_video_axis("unused.mkv", 1, fps)


def test_extract_originals_with_verified_timestamps(monkeypatch, tmp_path):
    fake_video(monkeypatch, [Frame(1000, (255, 0, 0)), Frame(1100), Frame(1200, (0, 255, 0))])
    rows = extract_source_frames("video.mkv", [2, 0, 2], tmp_path, 10, [0, .1, .2])
    assert [row["source_frame_id"] for row in rows] == [0, 2]
    assert [row["source_timestamp_s"] for row in rows] == pytest.approx([0, .2])
    for row, color in zip(rows, [(255, 0, 0), (0, 255, 0)]):
        path = Path(row["image_path"])
        assert path.is_absolute() and path.suffix == ".png"
        with Image.open(path) as image:
            assert image.size == (224, 224)
            assert image.getpixel((0, 0)) == color


@pytest.mark.parametrize("timestamps, message", [
    ([0, .11, .2], "disagrees"),
    ([0, .1], "disagrees"),
    ([0, .1, .2, .3], "count mismatch"),
])
def test_extract_rejects_inconsistent_supplied_axis_without_publishing(monkeypatch, tmp_path, timestamps, message):
    fake_video(monkeypatch, [Frame(1000), Frame(1100), Frame(1200)])
    with pytest.raises(ValueError, match=message):
        extract_source_frames("video.mkv", [0], tmp_path, 10, timestamps)
    assert list(tmp_path.iterdir()) == []


def test_extract_validates_after_last_requested_frame(monkeypatch, tmp_path):
    fake_video(monkeypatch, [Frame(0), Frame(100), Frame(None)])
    with pytest.raises(ValueError, match="Missing source PTS"):
        extract_source_frames("video.mkv", [0], tmp_path, 10)
    assert list(tmp_path.iterdir()) == []


def test_extract_rejects_out_of_range_request(monkeypatch, tmp_path):
    fake_video(monkeypatch, [Frame(0), Frame(100)])
    with pytest.raises(ValueError, match="outside the video"):
        extract_source_frames("video.mkv", [0, 2], tmp_path, 10)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("frame_id", [-1, .5, True])
def test_extract_rejects_invalid_frame_ids(tmp_path, frame_id):
    with pytest.raises(ValueError, match="frame ID"):
        extract_source_frames("unused.mkv", [frame_id], tmp_path, 10)


def test_actual_video_decode_and_png_export(tmp_path):
    path = tmp_path / "source.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width = stream.height = 224
        stream.pix_fmt = "bgr0"
        for index in range(4):
            frame = av.VideoFrame.from_image(Image.new("RGB", (224, 224), (index * 40, 20, 30)))
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    axis = decode_video_axis(path, 4, 10)
    assert axis == pytest.approx([0, .1, .2, .3])
    rows = extract_source_frames(path, [1, 3], tmp_path / "frames", 10, axis)
    for row, expected_red in zip(rows, [40, 120]):
        with Image.open(row["image_path"]) as image:
            assert image.size == (224, 224)
            assert image.getpixel((10, 10)) == (expected_red, 20, 30)


def test_pages_are_chronological_lossless_and_label_empty_cells(monkeypatch, tmp_path):
    red, green = tmp_path / "red.png", tmp_path / "green.png"
    Image.new("RGB", (224, 224), (255, 0, 0)).save(red)
    Image.new("RGB", (32, 16), (0, 255, 0)).save(green)
    rows = [
        {"source_frame_id": 422, "source_timestamp_s": 21.1, "image_path": str(green)},
        {"source_frame_id": 421, "source_timestamp_s": 21.05, "image_path": str(red)},
    ]
    labels = []
    original_text = event_visuals.ImageDraw.ImageDraw.text

    def record_text(self, position, text, *args, **kwargs):
        labels.append(text)
        return original_text(self, position, text, *args, **kwargs)

    monkeypatch.setattr(event_visuals.ImageDraw.ImageDraw, "text", record_text)
    pages = render_frame_pages(rows, tmp_path / "pages", "boundary", heading="Release", columns=3, rows_per_page=1)
    assert len(pages) == 1
    assert pages[0]["frame_ids"] == [421, 422]
    path = Path(pages[0]["image_path"])
    assert path.is_absolute() and pages[0]["sha256"] == file_hash(path)
    assert "EMPTY" in labels and "f=421" in labels and "f=422" in labels
    assert "t=21.100000s" in labels and "camera=base_0_rgb" in labels
    with Image.open(path) as page:
        pixels = list(page.getdata())
        assert pixels.count((255, 0, 0)) == 224 * 224
        assert pixels.count((0, 255, 0)) == 32 * 16


def test_pagination_has_no_missing_or_repeated_source_frames(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (224, 224), "black").save(path)
    rows = [{"source_frame_id": index, "source_timestamp_s": index / 10, "image_path": str(path)}
            for index in reversed(range(5))]
    pages = render_frame_pages(rows, tmp_path / "pages", "probe", columns=2, rows_per_page=1)
    assert [page["frame_ids"] for page in pages] == [[0, 1], [2, 3], [4]]
    assert len({page["image_path"] for page in pages}) == 3


def test_render_empty_rows(tmp_path):
    assert render_frame_pages([], tmp_path, "empty") == []


def test_render_rejects_duplicate_frame_ids(tmp_path):
    row = {"source_frame_id": 2, "source_timestamp_s": .2, "image_path": "unused.png"}
    with pytest.raises(ValueError, match="Duplicate"):
        render_frame_pages([row, row], tmp_path, "duplicate")
