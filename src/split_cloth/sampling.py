from __future__ import annotations

import math
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .util import file_hash, object_hash, read_json, read_jsonl, write_json, write_jsonl


def sample_indices(frame_count, fps, sample_fps=1.0):
    if frame_count < 1 or fps <= 0 or sample_fps <= 0:
        raise ValueError("Invalid video time axis")
    times = [i / sample_fps for i in range(math.floor((frame_count - 1) / fps * sample_fps + 1e-8) + 1)]
    # Nearest source frame; ties select the earlier frame.
    values = [min(frame_count - 1, int(math.floor(t * fps + .5 - 1e-9))) for t in times]
    return list(dict.fromkeys(values + [frame_count - 1]))


def page_groups(count, capacity=12, overlap=2):
    if not 0 <= overlap < capacity:
        raise ValueError("Invalid overlap")
    result, start = [], 0
    while start < count:
        end = min(start + capacity, count)
        result.append(list(range(start, end)))
        if end == count:
            break
        start = end - overlap
    return result


def font(size=13):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Monaco.ttf"]:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def render_pages(output, manifest, samples, layout, stage_frames=None):
    output = Path(output)
    cols, rows = layout["columns"], layout["rows"]
    tw, th = layout["tile_image_size"]
    label, header, gap = layout["tile_label_height_px"], layout["page_header_height_px"], layout["gutter_px"]
    groups = page_groups(len(samples), cols * rows, layout["overlap_samples"])
    pages = []
    colors = ["#51c4b7", "#7ca5ff", "#ffbd66", "#d19aff"]
    for pi, indices in enumerate(groups):
        image = Image.new("RGB", (cols * tw + (cols + 1) * gap, header + rows * (th + label) + (rows + 1) * gap), "#121b2c")
        draw = ImageDraw.Draw(image)
        draw.text((gap, 7), f"{manifest['dataset_id']} / {manifest['episode_id']}", fill="white", font=font(12))
        draw.text((gap, 26), f"p{pi:03d} / {len(groups)}    READ: left -> right, top -> bottom" + ("    EXPERT" if stage_frames else ""), fill="#c7d2e5", font=font(12))
        cells = []
        for cell in range(cols * rows):
            row, col = divmod(cell, cols)
            x, y = gap + col * (tw + gap), header + gap + row * (th + label + gap)
            draw.rectangle((x, y, x + tw, y + th + label), fill="#223049")
            if cell >= len(indices):
                draw.text((x + 12, y + 12), "EMPTY", fill="#78869c", font=font(16))
                cells.append({"row": row, "column": col, "sample_id": None})
                continue
            seq = indices[cell]
            sample = samples[seq]
            with Image.open(output / sample["image_path"]) as raw:
                tile = ImageOps.pad(raw.convert("RGB"), (tw, th), color="black")
                image.paste(tile, (x, y + label))
            suffix = ""
            if stage_frames:
                stage = 1 + sum(sample["source_frame_id"] > b for b in stage_frames)
                suffix = f" | STAGE {stage}"
                draw.rectangle((x, y, x + tw, y + label), fill=colors[stage - 1])
            draw.text((x + 4, y + 2), f"{sample['sample_id']}{suffix}", fill="white" if not stage_frames else "#101827", font=font(12))
            draw.text((x + 4, y + 18), f"t={sample['source_timestamp_s']:.2f}s  f={sample['source_frame_id']}", fill="#e8edf7" if not stage_frames else "#101827", font=font(12))
            cells.append({"row": row, "column": col, "sample_id": sample["sample_id"], "is_overlap": pi > 0 and cell < layout["overlap_samples"]})
        path = output / "montage" / f"page_{pi:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        pages.append({"page_id": f"p{pi:03d}", "image_path": str(path.relative_to(output)), "sha256": file_hash(path), "width": image.width, "height": image.height, "cells": cells})
    write_json(output / "pages.json", pages)
    return pages


def prepare_episode(manifest, output, config, preview=True):
    output = Path(output)
    old = read_json(output / "input_manifest.json")
    source = Path(manifest["source_video_path"])
    signature = {"size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
        "sample_fps": config["sample_fps"], "layout": config["montage"], "frame_count": manifest["frame_count"],
        "source_fps": manifest["source_fps"], "source_video_realpath": str(source.resolve())}
    if old and old.get("preparation_signature") == object_hash(signature) and (output / "samples.jsonl").exists() and (output / "pages.json").exists() and all((output / x["image_path"]).is_file() for x in read_json(output / "pages.json", [])) and (not preview or (output / "preview.mp4").exists()):
        return old
    n, fps = manifest["frame_count"], manifest["source_fps"]
    wanted = set(sample_indices(n, fps, config["sample_fps"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    samples, count, first_pts, previous_pts = [], 0, None, None
    preview_tmp = output / "preview.tmp.mp4"
    preview_writer = av.open(str(preview_tmp), "w", options={"movflags": "+faststart"}) if preview else None
    encoder = None
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                pts = float(frame.time) if frame.time is not None else i / fps
                if first_pts is None:
                    first_pts = pts
                timestamp = pts - first_pts
                if previous_pts is not None and pts <= previous_pts:
                    raise ValueError("Nonmonotonic source PTS")
                if abs(timestamp - i / fps) > 2 / fps:
                    raise ValueError("Unverified variable or mismatched frame rate")
                previous_pts = pts
                if preview_writer:
                    if encoder is None:
                        encoder = preview_writer.add_stream("libx264", rate=int(fps))
                        encoder.width, encoder.height = frame.width, frame.height
                        encoder.pix_fmt = "yuv420p"
                        encoder.options = {"crf": "23", "preset": "veryfast"}
                    display_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                    display_frame.pts = i
                    for packet in encoder.encode(display_frame):
                        preview_writer.mux(packet)
                if i in wanted:
                    seq = len(samples)
                    image_path = output / "frames" / f"sample_{seq:06d}.jpg"
                    frame.to_image().save(image_path, quality=95)
                    tail = i == n - 1 and abs(timestamp * config["sample_fps"] - round(timestamp * config["sample_fps"])) > .01
                    samples.append({"sample_seq": seq, "sample_id": f"s{seq:06d}", "source_frame_id": i,
                        "source_timestamp_s": round(timestamp, 6), "source_pts_s": pts,
                        "sample_kind": "tail" if tail else "regular", "image_path": str(image_path.relative_to(output)), "sample_sha256": file_hash(image_path)})
                count = i + 1
        if count != n:
            raise ValueError(f"Video/episode frame count mismatch: video={count}, metadata={n}")
        import pyarrow.parquet as pq
        if pq.read_metadata(manifest["parquet_path"]).num_rows != n:
            raise ValueError("Parquet/episode frame count mismatch")
        if preview_writer and encoder:
            for packet in encoder.encode():
                preview_writer.mux(packet)
    finally:
        if preview_writer:
            preview_writer.close()
    if preview:
        preview_tmp.replace(output / "preview.mp4")
    if len(samples) != len(wanted):
        raise ValueError("Missing source samples")
    write_jsonl(output / "samples.jsonl", samples)
    pages = render_pages(output, manifest, samples, config["montage"])
    full = {**manifest, "source_video_sha256": file_hash(source), "preparation_signature": object_hash(signature),
        "sample_count": len(samples), "page_count": len(pages), "sample_fps": config["sample_fps"],
        "input_hash": object_hash({"source": file_hash(source), "samples": samples, "pages": pages, "config": config["montage"]})}
    write_json(output / "input_manifest.json", full)
    return full
