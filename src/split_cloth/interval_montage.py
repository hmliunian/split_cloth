"""Chronological, independently sampled evidence between fixed gripper nodes.

Node frames end the interval on their left. Each interval starts its own 1 FPS
sampling clock, retains its actual first/last source frames, and gets separate
pages. Neither padding cells nor page positions represent elapsed time.
"""
from __future__ import annotations

from bisect import bisect_left
import math
from numbers import Integral
from pathlib import Path

from .event_visuals import extract_source_frames, render_frame_pages
from .util import write_json, write_jsonl


def _time_axis(source_timestamps):
    timestamps = [float(value) for value in source_timestamps]
    if (not timestamps or not all(math.isfinite(value) and value >= 0 for value in timestamps)
            or any(right <= left for left, right in zip(timestamps, timestamps[1:]))):
        raise ValueError("source_timestamps must be a nonempty, nonnegative, strictly increasing time axis")
    return timestamps


def _validated_nodes(nodes, frame_count):
    ordered, ids, frames = [], set(), set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Nodes must be objects")
        node_id, frame_id = node.get("node_id"), node.get("source_frame_id")
        if not isinstance(node_id, str) or not node_id or node_id in ids:
            raise ValueError("Nodes require nonempty, unique node_id values")
        if (isinstance(frame_id, bool) or not isinstance(frame_id, Integral)
                or not 0 <= frame_id < frame_count):
            raise ValueError("Node source_frame_id must be an original frame inside the video")
        if frame_id in frames:
            raise ValueError("Nodes at the same source frame must be merged before building intervals")
        ordered.append({**node, "source_frame_id": int(frame_id)})
        ids.add(node_id)
        frames.add(frame_id)
    return sorted(ordered, key=lambda node: node["source_frame_id"])


def _sample_interval(timestamps, start, end, sample_fps):
    """Choose nearest original PTS inside this interval; ties go earlier."""
    chosen = {start, end}
    count = math.floor((timestamps[end] - timestamps[start]) * sample_fps + 1e-9)
    for offset in range(count + 1):
        target = timestamps[start] + offset / sample_fps
        right = bisect_left(timestamps, target, start, end + 1)
        if right > end:
            frame_id = end
        elif right == start:
            frame_id = start
        else:
            left = right - 1
            frame_id = left if target - timestamps[left] <= timestamps[right] - target + 1e-12 else right
        chosen.add(frame_id)
    return sorted(chosen)


def build_intervals(nodes, source_timestamps, sample_fps=1.0):
    """Partition all original frames with no overlaps or gaps.

    For sorted node frames a1..am, intervals are [0,a1], [a1+1,a2],
    ...,[am+1,N-1]. No empty tail is added when the final node is N-1.
    A node at zero produces the valid one-frame first interval [0,0].
    """
    timestamps = _time_axis(source_timestamps)
    sample_fps = float(sample_fps)
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    ordered = _validated_nodes(nodes, len(timestamps))
    by_id = {node["node_id"]: node for node in ordered}
    intervals, start, left_node_id = [], 0, None
    right_edges = [(node["source_frame_id"], node["node_id"]) for node in ordered]
    if not right_edges or right_edges[-1][0] != len(timestamps) - 1:
        right_edges.append((len(timestamps) - 1, None))
    for end, right_node_id in right_edges:
        interval_id = f"i{len(intervals):04d}"
        right_node = by_id.get(right_node_id, {})
        intervals.append({
            "interval_id": interval_id,
            "start_frame": start,
            "end_frame": end,
            "start_timestamp_s": timestamps[start],
            "end_timestamp_s": timestamps[end],
            "left_node_id": left_node_id,
            "right_node_id": right_node_id,
            "right_node_type": right_node.get("node_type", right_node.get("type")),
            "sample_fps": sample_fps,
            "sample_frame_ids": _sample_interval(timestamps, start, end, sample_fps),
        })
        start, left_node_id = end + 1, right_node_id
    return intervals


def prepare_interval_montages(manifest, nodes, source_timestamps, output, layout=None):
    """Extract all wanted source frames once, then render each interval alone.

    Source PTS/count/CFR alignment is checked by ``extract_source_frames``.
    The renderer preserves 224-pixel images. Existing montage size/overlap
    options do not change sampling or insert evidence between intervals.
    Returns and persists intervals, chronological pages, and actual samples.
    """
    timestamps = _time_axis(source_timestamps)
    if manifest["frame_count"] != len(timestamps):
        raise ValueError("Manifest frame_count and source_timestamps count mismatch")
    layout = layout or {}
    max_columns, max_rows = layout.get("columns", 3), layout.get("rows", 4)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
           for value in (max_columns, max_rows)):
        raise ValueError("Interval columns and rows must be positive integers")
    intervals = build_intervals(nodes, timestamps, sample_fps=1.0)
    output = Path(output).resolve()
    wanted = sorted({frame_id for interval in intervals for frame_id in interval["sample_frame_ids"]})
    extracted = extract_source_frames(
        manifest["source_video_path"], wanted, output / "interval_frames",
        manifest["source_fps"], source_timestamps=timestamps,
    )
    by_frame = {row["source_frame_id"]: row for row in extracted}
    samples, pages = [], []
    for interval in intervals:
        interval_id = interval["interval_id"]
        members = []
        for index, frame_id in enumerate(interval["sample_frame_ids"]):
            sample = {
                **by_frame[frame_id],
                "sample_id": f"{interval_id}_s{index:04d}",
                "interval_id": interval_id,
                "is_interval_start": frame_id == interval["start_frame"],
                "is_interval_end": frame_id == interval["end_frame"],
                "is_right_node": interval["right_node_id"] is not None and frame_id == interval["end_frame"],
            }
            members.append(sample)
        samples.extend(members)
        sample_ids = {sample["source_frame_id"]: sample["sample_id"] for sample in members}
        right_label = interval["right_node_id"] or "VIDEO_END (no gripper node)"
        node_type = interval["right_node_type"] or "none"
        heading = (
            f"{interval_id} | frames [{interval['start_frame']},{interval['end_frame']}] | "
            f"t=[{interval['start_timestamp_s']:.6f},{interval['end_timestamp_s']:.6f}]s\n"
            f"left node={interval['left_node_id'] or 'VIDEO_START'} | "
            f"right node={right_label} | type={node_type}\n"
            "1 FPS within this interval; first/last frames retained. "
            "Read timestamps; EMPTY cells and grid spacing do not denote time."
        )
        columns = min(max_columns, len(members))
        rows_per_page = min(max_rows, math.ceil(len(members) / columns))
        rendered = render_frame_pages(
            members, output / "interval_montage", interval_id, heading=heading,
            camera=manifest.get("camera_key", "base_0_rgb"),
            columns=columns, rows_per_page=rows_per_page,
        )
        interval["sample_ids"] = [sample["sample_id"] for sample in members]
        interval["page_ids"] = []
        for index, page in enumerate(rendered):
            page_id = f"{interval_id}_p{index:04d}"
            pages.append({
                **page, "page_id": page_id, "interval_id": interval_id,
                "start_frame": interval["start_frame"], "end_frame": interval["end_frame"],
                "left_node_id": interval["left_node_id"], "right_node_id": interval["right_node_id"],
                "right_node_type": interval["right_node_type"],
                "sample_ids": [sample_ids[frame_id] for frame_id in page["frame_ids"]],
            })
            interval["page_ids"].append(page_id)
    write_json(output / "intervals.json", intervals)
    write_json(output / "interval_pages.json", pages)
    write_jsonl(output / "interval_samples.jsonl", samples)
    return {"intervals": intervals, "pages": pages, "samples": samples}
