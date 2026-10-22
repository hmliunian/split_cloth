"""Source-frame evidence for precise event boundary review.

Frame IDs are zero-based decoded video indices. Timestamps always come from
decoded presentation timestamps (PTS), relative to the first decoded frame.
"""

from __future__ import annotations

import math
import os
import tempfile
from numbers import Integral
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from .util import file_hash


def _positive_fps(fps):
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be finite and positive")
    return value


def _frame_id(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"Invalid zero-based source frame ID: {value!r}")
    return int(value)


def _decoded_frames(path, fps, expected_frames=None, source_timestamps=None):
    """Yield verified frames, then verify the complete video's frame count."""
    fps = _positive_fps(fps)
    if expected_frames is not None:
        expected_frames = _frame_id(expected_frames)
        if not expected_frames:
            raise ValueError("expected_frames must be positive")
    reference = None
    if source_timestamps is not None:
        reference = [float(value) for value in source_timestamps]
        if not reference or not all(math.isfinite(value) for value in reference):
            raise ValueError("source_timestamps must be a nonempty finite time axis")
        if expected_frames is not None and len(reference) != expected_frames:
            raise ValueError("source_timestamps/expected_frames count mismatch")
    first_pts = previous_pts = None
    count = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Missing source PTS at frame {index}")
            pts = float(frame.pts * frame.time_base)
            if not math.isfinite(pts):
                raise ValueError(f"Invalid source PTS at frame {index}")
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError(f"Nonmonotonic source PTS at frame {index}")
            if first_pts is None:
                first_pts = pts
            timestamp = pts - first_pts
            if abs(timestamp - index / fps) > 0.5 / fps + 1e-9:
                raise ValueError(f"Source PTS exceeds 0.5-frame CFR tolerance at frame {index}")
            if reference is not None:
                if index >= len(reference) or abs(timestamp - reference[index]) > 1e-6:
                    raise ValueError(f"source_timestamps disagrees with decoded PTS at frame {index}")
            previous_pts = pts
            count = index + 1
            yield index, timestamp, frame
    if not count:
        raise ValueError(f"Video contains no decoded frames: {path}")
    if expected_frames is not None and count != expected_frames:
        raise ValueError(f"Video/expected frame count mismatch: video={count}, expected={expected_frames}")
    if reference is not None and count != len(reference):
        raise ValueError(f"Video/source_timestamps count mismatch: video={count}, timestamps={len(reference)}")


def decode_video_axis(path, expected_frames, fps):
    """Return the full local, zero-origin PTS axis after strict video checks.

The metadata FPS is checked against every decoded PTS with a maximum error
of half a frame. Missing PTS are errors; row/FPS is never a fallback.
"""
    return [timestamp for _, timestamp, _ in _decoded_frames(path, fps, expected_frames)]


def extract_source_frames(path, frame_ids, output_dir, fps, source_timestamps=None):
    """Decode original frames to lossless PNGs and return chronological rows.

Optional ``source_timestamps`` is the complete axis returned by
``decode_video_axis``, not just the requested frames' timestamps. The entire
video is validated, including frames after the last requested frame. Output
paths are absolute. No PNG is published until validation succeeds.
"""
    wanted = sorted({_frame_id(value) for value in frame_ids})
    if not wanted:
        return []
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected, rows = set(wanted), []
    with tempfile.TemporaryDirectory(prefix=".source-frames-", dir=output) as staging:
        for index, timestamp, frame in _decoded_frames(path, fps, source_timestamps=source_timestamps):
            if index not in selected:
                continue
            filename = f"frame_{index:06d}.png"
            frame.to_image().convert("RGB").save(Path(staging) / filename, format="PNG")
            rows.append({"source_frame_id": index, "source_timestamp_s": timestamp,
                         "image_path": str(output / filename)})
        if len(rows) != len(wanted):
            missing = sorted(selected - {row["source_frame_id"] for row in rows})
            raise ValueError(f"Requested source frame IDs are outside the video: {missing}")
        for row in rows:
            destination = Path(row["image_path"])
            os.replace(Path(staging) / destination.name, destination)
    return rows


def _font(size):
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Monaco.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _wrapped(draw, text, font, width):
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for char in paragraph:
            if line and draw.textlength(line + char, font=font) > width:
                lines.append(line)
                line = ""
            line += char
        lines.append(line)
    return lines


def render_frame_pages(rows, output_dir, prefix, heading="", camera="base_0_rgb", columns=4, rows_per_page=4):
    """Render chronological frame evidence with separate, readable labels.

Images retain their aspect ratio inside 224 by 224 cells; original 224 by
224 images are pasted without resizing. Empty cells explicitly say EMPTY.
"""
    if (isinstance(columns, bool) or not isinstance(columns, Integral) or columns < 1
            or isinstance(rows_per_page, bool) or not isinstance(rows_per_page, Integral) or rows_per_page < 1):
        raise ValueError("columns and rows_per_page must be positive integers")
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."} or "\\" in prefix:
        raise ValueError("prefix must be a nonempty filename component")
    ordered = sorted(rows, key=lambda row: _frame_id(row["source_frame_id"]))
    frame_ids = [_frame_id(row["source_frame_id"]) for row in ordered]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Duplicate source frame IDs in page rows")
    timestamps = [float(row["source_timestamp_s"]) for row in ordered]
    if (not all(math.isfinite(value) and value >= 0 for value in timestamps)
            or any(right <= left for left, right in zip(timestamps, timestamps[1:]))):
        raise ValueError("Page timestamps must be finite, nonnegative, and increasing")
    if not ordered:
        return []
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tile, gap, line_height = 224, 12, 20
    width = columns * tile + (columns + 1) * gap
    label_font, header_font = _font(15), _font(16)
    measurement = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    camera_lines = _wrapped(measurement, f"camera={camera}", label_font, tile - 12)
    heading_lines = _wrapped(measurement, heading, header_font, width - 2 * gap) if heading else []
    capacity = columns * rows_per_page
    total_pages = math.ceil(len(ordered) / capacity)
    page_note = f"page {total_pages}/{total_pages} | left to right, top to bottom"
    note_lines = _wrapped(measurement, page_note, header_font, width - 2 * gap)
    label_height = (2 + len(camera_lines)) * line_height + 8
    header_height = (len(heading_lines) + len(note_lines)) * line_height + 12
    height = header_height + rows_per_page * (tile + label_height) + (rows_per_page + 1) * gap
    pages = []
    for page_index, start in enumerate(range(0, len(ordered), capacity)):
        members = ordered[start:start + capacity]
        image = Image.new("RGB", (width, height), "#101827")
        draw = ImageDraw.Draw(image)
        for line_index, line in enumerate(heading_lines):
            draw.text((gap, 6 + line_index * line_height), line, fill="white", font=header_font)
        current_note = f"page {page_index + 1}/{total_pages} | left to right, top to bottom"
        for line_index, line in enumerate(_wrapped(draw, current_note, header_font, width - 2 * gap)):
            draw.text((gap, 6 + (len(heading_lines) + line_index) * line_height),
                      line, fill="#c7d2e5", font=header_font)
        for cell in range(capacity):
            grid_row, column = divmod(cell, columns)
            x = gap + column * (tile + gap)
            y = header_height + gap + grid_row * (tile + label_height + gap)
            draw.rectangle((x, y, x + tile - 1, y + label_height + tile - 1), fill="#253249")
            if cell >= len(members):
                draw.text((x + 12, y + 12), "EMPTY", fill="#b2bfd2", font=header_font)
                continue
            row = members[cell]
            lines = [f"f={row['source_frame_id']}", f"t={float(row['source_timestamp_s']):.6f}s", *camera_lines]
            for line_index, line in enumerate(lines):
                draw.text((x + 6, y + 3 + line_index * line_height), line, fill="white", font=label_font)
            with Image.open(row["image_path"]) as source:
                original = source.convert("RGB")
                original.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                image.paste(original, (x + (tile - original.width) // 2,
                                       y + label_height + (tile - original.height) // 2))
        path = output / f"{prefix}_{page_index:03d}.png"
        image.save(path, format="PNG")
        pages.append({"image_path": str(path),
                      "frame_ids": [row["source_frame_id"] for row in members],
                      "sha256": file_hash(path)})
    return pages
