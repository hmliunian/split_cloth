"""Local-only review UI for remote annotation runs.

Run this service on the data host and use SSH local forwarding. The browser only
receives run assets and frames from a video already recorded in the manifest.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_REVIEW_STATUSES = {"approved", "corrected", "needs_review", "out_of_scope"}


def _json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"Invalid JSON in {path.name}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"Invalid JSONL in {path.name}") from exc


def _inside(base: Path, path: Path) -> Path:
    base, path = base.resolve(), path.resolve()
    if not path.is_relative_to(base):
        raise HTTPException(400, "Path escapes the run directory")
    return path


def _run(project_root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not _RUN_NAME.fullmatch(run_id):
        raise HTTPException(400, "Invalid run_id")
    base = (project_root / "runs").resolve()
    path = _inside(base, base / run_id)
    if not path.is_dir():
        raise HTTPException(404, "Run not found")
    return path


def _key(manifest: dict[str, Any]) -> str:
    if manifest.get("key"):
        return str(manifest["key"])
    episode_id = manifest.get("episode_id")
    if episode_id is None:
        episode_id = f"episode_{int(manifest['episode_index']):06d}"
    return f"{manifest['dataset_id']}__{episode_id}"


def _episode(run: Path, key: str) -> tuple[dict[str, Any], Path]:
    for manifest in _jsonl(run / "manifest.jsonl"):
        if _key(manifest) == key:
            dataset_id = str(manifest["dataset_id"])
            index = manifest.get("episode_index")
            if index is not None:
                episode_id = f"episode_{int(index):06d}"
            else:
                episode_id = str(manifest.get("episode_id", ""))
            directory = _inside(run, run / "episodes" / dataset_id / episode_id)
            # All filenames below must also remain inside the episode directory.
            # This blocks run-local symlink tricks from reaching external files.
            if not directory.is_dir():
                raise HTTPException(404, "Episode assets not found")
            inputs = _json(_inside(directory, directory / "input_manifest.json"), {})
            return {**manifest, **inputs, "key": key}, directory
    raise HTTPException(404, "Episode not found")


def _annotation(directory: Path) -> dict[str, Any] | None:
    annotation = _json(_inside(directory, directory / "annotation.json"))
    if not isinstance(annotation, dict):
        return None
    annotation = dict(annotation)
    if "boundary_frames" not in annotation:
        boundaries = annotation.get("boundaries", [])
        annotation["boundary_frames"] = [
            item.get("coarse_end_frame", item.get("first_complete_source_frame_id", item.get("source_frame_id")))
            for item in boundaries if isinstance(item, dict)
        ]
    if "stage4_completion_frame" not in annotation:
        completion = annotation.get("stage4_completion") or {}
        annotation["stage4_completion_frame"] = completion.get("source_frame_id") if isinstance(completion, dict) else None
    return annotation


def _asset_url(run: Path, path: Path) -> str:
    relative = str(_inside(run, path).relative_to(run))
    return "/api/assets?" + urlencode({"run_id": run.name, "path": relative})


def _page_path(run: Path, directory: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return _inside(run, path)
    if path.parts and path.parts[0] == "episodes":
        return _inside(run, run / path)
    return _inside(run, directory / path)


def _pages(run: Path, directory: Path) -> list[dict[str, Any]]:
    raw = _json(_inside(directory, directory / "pages.json"), [])
    if isinstance(raw, dict):
        raw = raw.get("pages", [])
    pages = []
    for index, page in enumerate(raw):
        item = {"path": page} if isinstance(page, str) else dict(page)
        raw_path = next((item[name] for name in ("path", "image_path", "file_path", "file", "filename") if item.get(name)), None)
        if raw_path:
            path = _page_path(run, directory, str(raw_path))
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                pages.append({**item, "page_id": item.get("page_id", index), "url": _asset_url(run, path)})
    if not pages:
        for index, path in enumerate(sorted((directory / "montage").glob("*.png"))):
            pages.append({"page_id": index, "url": _asset_url(run, path)})
    return pages


def _expert(run: Path) -> Any:
    criteria_path = _inside(run, run / "stage_criteria.json")
    if criteria_path.is_file():
        criteria = _json(criteria_path)
        if (not isinstance(criteria, dict)
                or criteria.get("schema_version") != "text_stage_criteria_v1"
                or not isinstance(criteria.get("stages"), list)):
            raise HTTPException(500, "Invalid text stage criteria in stage_criteria.json")
        # Text-reference runs must not resolve or expose the old image expert.
        return {
            "reference_mode": "text_stage_criteria",
            "name": "文字阶段判据",
            "validation_level": "gpt_visual_summary_not_human_gold",
            "stage_criteria": criteria,
            "provenance": criteria.get("provenance", {}),
            "limitations": criteria.get("limitations", []),
            "pages": [], "page_urls": [],
        }
    expert = _json(_inside(run, run / "expert.json"))
    if isinstance(expert, dict):
        expert = dict(expert)
        if expert.get("directory"):
            expert_dir = _inside(run, run / expert["directory"])
            expert["pages"] = _pages(run, expert_dir)
            expert["page_urls"] = [page["url"] for page in expert["pages"]]
        paths = expert.get("page_paths", [])
        if paths:
            expert["page_urls"] = [_asset_url(run, _page_path(run, run, str(path))) for path in paths]
        if expert.get("pages"):
            normalized = []
            for page in expert["pages"]:
                if isinstance(page, dict) and page.get("url"):
                    normalized.append(page)
                    continue
                item = {"path": page} if isinstance(page, str) else dict(page)
                raw_path = item.get("path") or item.get("image_path") or item.get("file_path")
                if raw_path:
                    item["url"] = _asset_url(run, _page_path(run, run, str(raw_path)))
                normalized.append(item)
            expert["pages"] = normalized
            expert.setdefault("page_urls", [page["url"] for page in normalized if page.get("url")])
    return expert


def _status(annotation: dict[str, Any] | None) -> str:
    if (annotation or {}).get("execution_error"):
        return "failed"
    return str((annotation or {}).get("status", "pending"))


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_frames(n: int, boundaries: Any, completion: Any) -> None:
    if not isinstance(boundaries, list) or len(boundaries) != 3 or not all(_strict_int(b) for b in boundaries):
        raise HTTPException(422, "boundary_frames must be exactly three integers")
    b1, b2, b3 = boundaries
    if not (0 <= b1 < b2 < b3 < n - 1):
        raise HTTPException(422, "Require 0 <= b1 < b2 < b3 < frame_count - 1")
    if completion is not None and (not _strict_int(completion) or not b3 < completion < n):
        raise HTTPException(422, "stage4_completion_frame must be an integer after b3 and before frame_count")


def _atomic_json(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_labels(path: Path, n: int, boundaries: list[int]) -> None:
    labels = np.empty(n, dtype=np.int64)
    start = 0
    for stage, end in enumerate([*boundaries, n - 1], 1):
        labels[start:end + 1] = stage
        start = end + 1
    fd, temporary = tempfile.mkstemp(prefix=".stage_reviewed.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, labels, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _review_lock(directory: Path):
    import fcntl

    lock = _inside(directory, directory / ".review.lock")
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@lru_cache(maxsize=96)
def _jpeg_frame(path_string: str, mtime_ns: int, frame_id: int) -> bytes:
    # Decode by frame index so variable frame-rate media never silently uses an
    # approximate timestamp as the requested source frame. Review cache is bounded.
    import av

    with av.open(path_string) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index == frame_id:
                buffer = io.BytesIO()
                frame.to_image().save(buffer, format="JPEG", quality=90)
                return buffer.getvalue()
    raise HTTPException(422, "Source video has fewer frames than its manifest")


def _range_video(request: Request, path: Path) -> Response:
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or not any(match.groups()) or size == 0:
        raise HTTPException(416, "Invalid byte range", headers={"Content-Range": f"bytes */{size}"})
    left, right = match.groups()
    if not left:
        suffix = int(right)
        if suffix < 1:
            raise HTTPException(416, "Invalid byte range", headers={"Content-Range": f"bytes */{size}"})
        start, end = max(0, size - suffix), size - 1
    else:
        start, end = int(left), min(int(right), size - 1) if right else size - 1
    if start > end or start >= size:
        raise HTTPException(416, "Unsatisfiable byte range", headers={"Content-Range": f"bytes */{size}"})

    def chunks():
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(chunks(), status_code=206, media_type="video/mp4", headers={
        "Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1),
    })


def create_app(project_root: Path) -> FastAPI:
    project_root = Path(project_root).resolve()
    app = FastAPI(title="Split Cloth Annotation Review", docs_url=None, redoc_url=None)
    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/")
    def index():
        path = static / "index.html"
        if not path.is_file():
            return JSONResponse({"detail": "GUI files are not installed"}, status_code=503)
        return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs")
    def runs():
        base = project_root / "runs"
        items = []
        if base.is_dir():
            for path in sorted(base.iterdir(), reverse=True):
                if path.is_dir() and _RUN_NAME.fullmatch(path.name) and path.resolve().is_relative_to(base.resolve()):
                    metadata = _json(path / "run.json", {})
                    items.append({**metadata, "run_id": path.name})
        return {"runs": items}

    @app.get("/api/episodes")
    def episodes(run_id: str, status: str = "", dataset: str = "", q: str = ""):
        run = _run(project_root, run_id)
        items = []
        summary = {name: 0 for name in ("total", "accepted", "needs_review", "failed", "out_of_scope", "unreadable", "pending", "reviewed", "approved", "corrected")}
        for entry in _jsonl(run / "manifest.jsonl"):
            key = _key(entry)
            try:
                directory = _inside(run, run / "episodes" / str(entry["dataset_id"]) / f"episode_{int(entry['episode_index']):06d}")
                if not directory.is_dir():
                    raise HTTPException(404, "Episode assets not found")
                manifest = {**entry, **_json(_inside(directory, directory / "input_manifest.json"), {}), "key": key}
                annotation = _annotation(directory)
                review = _json(_inside(directory, directory / "review.json"))
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                manifest, annotation, review = {**entry, "key": key}, None, None
            annotation_status = _status(annotation)
            review_status = (review or {}).get("status", "unreviewed")
            summary["total"] += 1
            summary[annotation_status if annotation_status in summary and annotation_status not in {"total", "reviewed"} else "pending"] += 1
            summary["reviewed"] += int(review is not None)
            if review_status in {"approved", "corrected"}:
                summary[review_status] += 1
            item = {
                **manifest, "key": key, "status": annotation_status, "review_status": review_status,
                "boundary_frames": (annotation or {}).get("boundary_frames", []),
                "stage4_completion_frame": (annotation or {}).get("stage4_completion_frame"),
                "annotation": annotation,
                "review_version": (review or {}).get("version", 0),
            }
            if dataset and str(manifest.get("dataset_id")) != dataset:
                continue
            if status and status not in {annotation_status, review_status}:
                continue
            if q and q.casefold() not in f"{key} {manifest.get('task', '')}".casefold():
                continue
            items.append(item)
        return {"episodes": items, "summary": summary}

    @app.get("/api/episode")
    def episode(run_id: str, key: str):
        run = _run(project_root, run_id)
        manifest, directory = _episode(run, key)
        preview = _inside(directory, directory / "preview.mp4")
        human_review = _json(_inside(directory, directory / "review.json"))
        return {
            "manifest": manifest, "annotation": _annotation(directory),
            "review": human_review, "human_review": human_review,
            "gpt_review": _json(_inside(directory, directory / "gpt_review.json")),
            "samples": _jsonl(_inside(directory, directory / "samples.jsonl")),
            "pages": _pages(run, directory),
            "video_url": "/api/video?" + urlencode({"run_id": run_id, "key": key}) if preview.is_file() else None,
            "expert": _expert(run),
        }

    @app.get("/api/expert")
    def expert(run_id: str):
        return _expert(_run(project_root, run_id))

    @app.get("/api/assets")
    def assets(run_id: str, path: str):
        run = _run(project_root, run_id)
        file = _inside(run, run / path)
        if file.suffix.lower() not in _IMAGE_SUFFIXES or not file.is_file():
            raise HTTPException(404, "Image asset not found")
        return FileResponse(file)

    @app.get("/api/video")
    def video(request: Request, run_id: str, key: str):
        _, directory = _episode(_run(project_root, run_id), key)
        preview = _inside(directory, directory / "preview.mp4")
        if not preview.is_file():
            raise HTTPException(404, "H.264 browser preview is not available")
        return _range_video(request, preview)

    @app.get("/api/frame")
    def frame(run_id: str, key: str, frame: int):
        manifest, _ = _episode(_run(project_root, run_id), key)
        n = int(manifest["frame_count"])
        if not 0 <= frame < n:
            raise HTTPException(422, "Frame is outside the episode")
        path = Path(manifest.get("source_video_path", ""))
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise HTTPException(404, "Source video is unavailable on this host")
        try:
            content = _jpeg_frame(str(path.resolve()), path.stat().st_mtime_ns, frame)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Frame decoding failed ({type(exc).__name__})") from exc
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/review")
    def review(body: dict[str, Any] = Body(...)):
        run_id, key = body.get("run_id"), body.get("key")
        if not isinstance(run_id, str) or not isinstance(key, str):
            raise HTTPException(422, "run_id and key are required strings")
        run = _run(project_root, run_id)
        manifest, directory = _episode(run, key)
        status = body.get("status")
        if not isinstance(status, str) or status not in _REVIEW_STATUSES:
            raise HTTPException(422, "Invalid review status")
        reviewer, note = body.get("reviewer", "anonymous"), body.get("note", "")
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
            raise HTTPException(422, "reviewer must be a nonempty string of at most 200 characters")
        if not isinstance(note, str) or len(note) > 20000:
            raise HTTPException(422, "note must be a string of at most 20000 characters")
        version = body.get("version", 0)
        if not _strict_int(version) or version < 0:
            raise HTTPException(422, "version must be a nonnegative integer")
        with _review_lock(directory):
            review_path = _inside(directory, directory / "review.json")
            current = _json(review_path, {})
            current_version = current.get("version", 0)
            if current_version != version:
                raise HTTPException(409, {"message": "Review changed; reload before saving", "current_version": current_version})
            annotation = _annotation(directory) or {}
            boundaries = body.get("boundary_frames", current.get("boundary_frames", annotation.get("boundary_frames")))
            completion = body.get("stage4_completion_frame", current.get("stage4_completion_frame", annotation.get("stage4_completion_frame")))
            if status in {"approved", "corrected"} and completion is None:
                raise HTTPException(422, "Approved reviews require stage4_completion_frame")
            if status in {"approved", "corrected"} or "boundary_frames" in body or ("stage4_completion_frame" in body and completion is not None):
                _validate_frames(int(manifest["frame_count"]), boundaries, completion)
            elif completion is not None and (not _strict_int(completion) or not 0 <= completion < int(manifest["frame_count"])):
                raise HTTPException(422, "Invalid stage4_completion_frame")
            previous_history = list(current.get("history", []))
            if current:
                previous_history.append({key: value for key, value in current.items() if key != "history"})
            value = {
                "run_id": run_id, "key": key, "reviewer": reviewer.strip(), "status": status,
                "boundary_frames": boundaries, "stage4_completion_frame": completion, "note": note,
                "version": version + 1, "updated_at": datetime.now(timezone.utc).isoformat(), "history": previous_history,
            }
            if status in {"approved", "corrected"}:
                _atomic_labels(_inside(directory, directory / "stage_reviewed.npy"), int(manifest["frame_count"]), boundaries)
            else:
                labels_path = _inside(directory, directory / "stage_reviewed.npy")
                if labels_path.exists():
                    labels_path.unlink()
            _atomic_json(review_path, value)
            history_path = _inside(directory, directory / "review_history.jsonl")
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({key: item for key, item in value.items() if key != "history"}, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return {"review": value, "version": value["version"]}

    return app


def serve(project_root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(project_root), host=host, port=port)
