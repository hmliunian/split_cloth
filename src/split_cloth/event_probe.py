"""Small, isolated event-guided experiment; source datasets remain read-only.

The uniform 1 FPS context identifies semantics. Feedback events only propose
locations. A separate request examines consecutive original video frames.
No baseline prediction or GPT comparison label is given to either request.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from .boundaries import TASK_TYPES, validate_review_frames
from .event_visuals import decode_video_axis, extract_source_frames, render_frame_pages
from .gripper_events import detect_gripper_event_candidates_from_parquet
from .pipeline import save_labels
from .prompts import image_item
from .provider import request_annotation
from .util import episode_dir, file_hash, now, object_hash, read_json, read_jsonl, write_json, write_jsonl

VERSION = "gripper_event_visual_v1"
SYSTEM = """你是机器人折衣阶段标注员。按规范判断动作完成且参与该动作的夹爪完全松开。
衣物形状完成、发出松爪指令、开始张爪都不足以结束阶段；松爪也可能只是调整或重抓。
四阶段只有三个分段边界，阶段4完成另作证据。完成帧属于前阶段，下一原始帧属于后阶段。
图片按页、从左到右再从上到下读取，EMPTY不是视频帧。帧号为0基原视频索引。
实测state事件来自自动阈值，只是候选而非物理释放真值。不能按时长等分或照搬示范时间。
不确定、遮挡、返工、缺阶段时如实拒判。只输出一个JSON对象，不输出代码块。"""


def _text(value):
    return {"type": "text", "text": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}


def _images(content, pages, title):
    for i, page in enumerate(pages):
        content.append(_text(f"{title} page={i}; frames={page.get('frame_ids', [])}"))
        content.append(image_item(page["image_path"]))


def _model_calibration(candidates):
    return {key: value for key, value in candidates["calibration"].items() if key != "parquet_path"}


def _record_request(directory, messages, config):
    """Prepare a reviewable request record without any provider/network call."""
    directory = Path(directory)
    signature_messages = deepcopy(messages)
    for message in signature_messages:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"]["url"] = hashlib.sha256(part["image_url"]["url"].encode()).hexdigest()
    fingerprint = object_hash({"version": VERSION, "messages": signature_messages, "provider": config})
    prior = read_json(directory / "request.json")
    if prior and prior.get("input_fingerprint") != fingerprint:
        raise ValueError("Changed request input: use a new experiment run")
    write_json(directory / "request.json", {"version": VERSION, "input_fingerprint": fingerprint,
               "created_at": prior.get("created_at") if prior else now(), "messages": signature_messages,
               "model_id": config["model_id"]})


def _call(project, directory, messages, config):
    """Bind provider's response cache to actual text, image bytes and settings."""
    directory = Path(directory)
    _record_request(directory, messages, config)
    response = request_annotation(project, directory, messages, config)
    raw = json.loads(response["content"])
    write_json(directory / "model_output.json", raw)
    return raw


def collect_usage(output):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    responses, attempts = [], 0
    output = Path(output)
    paths = list(output.rglob("response.json")) + list(output.glob("**/invalid_responses/response_*.json"))
    for path in paths:
        if path.name.endswith(".metadata.json"):
            continue
        data = read_json(path, {})
        if not isinstance(data, dict) or "usage" not in data:
            continue
        for key in totals:
            totals[key] += data.get("usage", {}).get(key, 0) or 0
        responses.append({"path": str(path.relative_to(output)), "request_id": data.get("request_id"),
                          "usage": data["usage"], "elapsed_s": data.get("elapsed_s")})
    for path in output.rglob("attempts.json"):
        attempts += len(read_json(path, []))
    return {"usage": totals, "response_count": len(responses), "attempt_count": attempts, "responses": responses}


def validate_selection(raw, events, samples):
    if not isinstance(raw, dict):
        raise ValueError("Selection must be a JSON object")
    required = {"task_type", "applicable", "all_stages_observed", "needs_review", "review_reasons", "boundaries", "stage4_completion"}
    if set(raw) != required or raw.get("task_type") not in TASK_TYPES:
        raise ValueError("Invalid selection schema/task type")
    for key in ("all_stages_observed", "needs_review"):
        if type(raw.get(key)) is not bool:
            raise ValueError(f"Invalid {key}")
    if raw["applicable"] is not None and type(raw["applicable"]) is not bool:
        raise ValueError("Invalid applicability")
    if not isinstance(raw["review_reasons"], list) or not all(isinstance(x, str) for x in raw["review_reasons"]):
        raise ValueError("Invalid review reasons")
    if not isinstance(raw["boundaries"], list) or len(raw["boundaries"]) != 3:
        raise ValueError("Exactly three boundary slots are required")
    by_id = {e["event_id"]: e for e in events}
    prior_frame = -1
    used_ids = set()
    for stage, boundary in enumerate(raw["boundaries"], 1):
        if not isinstance(boundary, dict) or set(boundary) != {"from_stage", "candidate_event_ids", "participant_arms", "evidence_text"}:
            raise ValueError("Invalid boundary selection schema")
        if type(boundary["from_stage"]) is not int or boundary["from_stage"] != stage:
            raise ValueError("Incorrect stage order")
        ids, arms = boundary["candidate_event_ids"], boundary["participant_arms"]
        if not isinstance(ids, list) or len(ids) > 2 or not all(type(x) is str and x in by_id for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("Unknown/duplicate event IDs")
        if not isinstance(arms, list) or not all(x in ("left", "right") for x in arms) or len(set(arms)) != len(arms):
            raise ValueError("Invalid participant arms")
        if not isinstance(boundary["evidence_text"], str):
            raise ValueError("Missing visible evidence")
        if ids:
            if used_ids.intersection(ids):
                raise ValueError("A release event cannot end two different stages")
            used_ids.update(ids)
            if not arms or any(by_id[x]["arm"] not in arms for x in ids):
                raise ValueError("Selected event must belong to a participant arm")
            anchor = max(by_id[x]["source_frame_id"] for x in ids)
            if anchor <= prior_frame:
                raise ValueError("Event boundaries must increase")
            prior_frame = anchor
    c = raw["stage4_completion"]
    if not isinstance(c, dict) or set(c) != {"observed", "completion_sample_id", "evidence_text"} or type(c["observed"]) is not bool or not isinstance(c["evidence_text"], str):
        raise ValueError("Invalid final completion evidence")
    by_sample = {s["sample_id"]: s for s in samples}
    if c["completion_sample_id"] is not None and (type(c["completion_sample_id"]) is not str or c["completion_sample_id"] not in by_sample):
        raise ValueError("Final completion must reference an actual context sample")
    if c["observed"] and c["completion_sample_id"] is None:
        raise ValueError("Observed final completion needs an input sample")
    return raw


def validate_refinement(raw, allowed_frames):
    expected = {"end_frame", "previous_not_complete_frame", "operation_complete", "participating_grippers_released", "first_complete_confirmed", "next_frame_consistent", "evidence_text", "review_reasons"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("Invalid dense refinement schema")
    for key in ("operation_complete", "participating_grippers_released", "first_complete_confirmed", "next_frame_consistent"):
        if type(raw[key]) is not bool:
            raise ValueError(f"Invalid refinement flag {key}")
    if not isinstance(raw["evidence_text"], str) or not isinstance(raw["review_reasons"], list) or not all(type(x) is str for x in raw["review_reasons"]):
        raise ValueError("Invalid refinement evidence")
    end, previous = raw["end_frame"], raw["previous_not_complete_frame"]
    if end is not None and (type(end) is not int or end not in allowed_frames):
        raise ValueError("Refined frame was not supplied to model")
    if previous is not None and (type(previous) is not int or previous not in allowed_frames):
        raise ValueError("Previous evidence frame was not supplied")
    if raw["first_complete_confirmed"] and (end is None or previous != end - 1 or end + 1 not in allowed_frames):
        raise ValueError("First-frame claim needs consecutive f-1/f/f+1 evidence")
    return raw


def make_annotation(manifest, selection, refinements, samples):
    """Only three visually supported source-frame proposals may form labels."""
    reasons = list(selection.get("review_reasons", []))
    out_of_scope = selection["task_type"] != "unknown" and (selection["applicable"] is False or selection["task_type"] in TASK_TYPES - {"fold_cloth", "unknown"})
    completion = selection["stage4_completion"]
    sample = next((s for s in samples if s["sample_id"] == completion["completion_sample_id"]), None)
    frames = [r.get("end_frame") for r in refinements]
    if not out_of_scope:
        if selection["task_type"] != "fold_cloth" or selection["applicable"] is not True:
            reasons.append("Folding applicability is unresolved")
        if not selection["all_stages_observed"] or selection["needs_review"]:
            reasons.append("Selection requested semantic review")
        if len(refinements) != 3 or any(not r.get("visual_confirmed") for r in refinements):
            reasons.append("Not all three first-release source frames are visually supported")
        if not completion["observed"] or sample is None:
            reasons.append("Final-stage completion is not observed")
        reasons.extend(validate_review_frames(manifest["frame_count"], frames, sample["source_frame_id"] if sample else None))
        for r in refinements:
            reasons.extend(r.get("review_reasons", []))
    status = "out_of_scope" if out_of_scope else ("needs_review" if reasons else "accepted")
    n = manifest["frame_count"]
    if status == "accepted":
        starts, ends = [0] + [x + 1 for x in frames], frames + [n - 1]
        segments = [{"stage": i, "start_frame": a, "end_frame": b} for i, (a, b) in enumerate(zip(starts, ends), 1)]
    else:
        segments = [{"stage": -1, "start_frame": 0, "end_frame": n - 1}]
    return {**manifest, "schema_version": "1.0", "spec_version": VERSION,
            "precision": "source_frame_visual_proposal", "status": status, "needs_review": status == "needs_review",
            "task_type": selection["task_type"], "applicable": selection["applicable"],
            "all_stages_observed": selection["all_stages_observed"], "review_reasons": list(dict.fromkeys(reasons)),
            "boundary_frames": frames if status == "accepted" else [], "boundaries": refinements, "segments": segments,
            "stage4_completion": {**completion, "source_frame_id": sample["source_frame_id"] if sample else None},
            "stage4_completion_frame": sample["source_frame_id"] if sample else None,
            "training_ready": False, "validation_level": "model_visual_and_state_not_human_gold",
            "reference_is_human_gold": False}


def _selection_messages(run, output, candidates, candidate_pages):
    catalog = read_json(run / "standard_catalog.json")
    expert = read_json(run / "expert.json")
    content = [_text({"stage_definitions": catalog["stages"]})]
    for stage in catalog["stages"]:
        content.extend([_text(f"REFERENCE stage={stage['stage_id']}"), image_item(run / stage["reference_image"])])
    content.append(_text("EXPERT为GPT视觉粗参考，不是人工真值。仅参考动作语义，不能复制时间。"))
    expert_dir = run / expert["directory"]
    _images(content, [{**p, "image_path": str(expert_dir / p["image_path"])} for p in read_json(expert_dir / "pages.json")], "EXPERT")
    samples = read_jsonl(output / "samples.jsonl")
    content.append(_text({"TARGET_1fps_samples": [{k: s[k] for k in ("sample_id", "source_frame_id", "source_timestamp_s")} for s in samples]}))
    _images(content, [{**p, "image_path": str(output / p["image_path"])} for p in read_json(output / "pages.json")], "TARGET full 1 FPS context")
    content.append(_text({"candidate_calibration": _model_calibration(candidates), "candidate_events": candidates["events"], "diagnostics": candidates["diagnostics"]}))
    _images(content, candidate_pages, "TARGET event f-1/f/f+1 evidence; shared frames appear once")
    content.append(_text("""选择三个阶段结束对应的松爪事件。全部松爪候选都已给出；取衣/抖衣/调整中的松爪不能作为折叠完成。
对于每个阶段，指出真正参与末次操作的手。若两手先后释放，应选择使本阶段所有参与手完成释放的事件，可引用1或2个event_id。
阈值事件不是真值，后续会在附近逐帧确认；本步不需要伪造精确frame。无法判断时候选列表为空并请求复核。
严格输出字段：task_type:fold_cloth|fold_pants|hang_cloth|mixed|other|unknown;
applicable:true|false|null; all_stages_observed:boolean; needs_review:boolean; review_reasons:string[];
boundaries:恰好3项，每项{from_stage:1或2或3,candidate_event_ids:输入event_id的数组(最多2个),participant_arms:[left和/或right],evidence_text:string};
stage4_completion:{observed:boolean,completion_sample_id:TARGET的1fps sample_id或null,evidence_text:string}。
三个事件时间必须递增；第4阶段完成必须另有较晚证据。完整性不足/不适用仍输出三个空槽位，不强凑事件。"""))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def _refinement_messages(run, boundary, events, candidates, pages_by_camera, trace, allowed):
    catalog = read_json(run / "standard_catalog.json")
    stage = catalog["stages"][boundary["from_stage"] - 1]
    content = [_text({"stage_definition": stage, "semantic_selection": boundary, "selected_candidates": events,
                      "calibration": _model_calibration(candidates), "source_state_per_frame": trace}), image_item(run / stage["reference_image"])]
    for camera, pages in pages_by_camera.items():
        _images(content, pages, f"TARGET dense original frames camera={camera}")
    content.append(_text("""以上各相机是同一段原始连续帧，并非1fps。根据连续变化选择本阶段操作完成、参与夹爪完全松开的首次原始帧。
state是实测反馈，但开阈值只是估计，不必机械选中阈值帧。图像与反馈矛盾/无法看清完全松开时应请求复核。
请检查f-1仍不满足完成、f首次满足、f+1与释放后的状态一致；稳定确认不能把边界后移。
可以在提供的窗口内修正事件时刻，不能选择没看过的帧。如果窗口不足，返回null并说明，不能选窗口边缘假装首次完成。
严格输出字段：end_frame:int|null,previous_not_complete_frame:int|null,
operation_complete:boolean,participating_grippers_released:boolean,first_complete_confirmed:boolean,next_frame_consistent:boolean,
evidence_text:string,review_reasons:string[]。无法区分相邻帧时first_complete_confirmed=false，保留不确定性。
可选原始帧号：""" + json.dumps(allowed)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def _camera_path(manifest, camera):
    root = Path(manifest["dataset_root"])
    info = read_json(root / "meta/info.json")
    if camera not in info.get("features", {}):
        raise ValueError(f"Missing participant camera: {camera}")
    index = manifest["episode_index"]
    return root / info["video_path"].format(episode_chunk=index // info.get("chunks_size", 1000), episode_index=index, video_key=camera)


def _one(project, run, baseline, manifest, config, algorithm_hash, prepare_only=False):
    output = episode_dir(run, manifest)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".experiment.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = episode_dir(baseline, manifest)
        prepared = read_json(source / "input_manifest.json")
        if not prepared:
            raise ValueError("Baseline has no prepared source context")
        source_hash = file_hash(manifest["source_video_path"])
        if source_hash != prepared["source_video_sha256"]:
            raise ValueError("Source video changed since baseline")
        camera_hashes = {arm: file_hash(_camera_path(manifest, f"{arm}_wrist_0_rgb")) for arm in ("left", "right")}
        catalog = read_json(run / "standard_catalog.json")
        expert_dir = run / read_json(run / "expert.json")["directory"]
        reference_hashes = {str(path): file_hash(path) for path in
                            [run / s["reference_image"] for s in catalog["stages"]] +
                            [expert_dir / p["image_path"] for p in read_json(expert_dir / "pages.json")]}
        fingerprint = object_hash({"config": config, "code": algorithm_hash, "video": source_hash,
                                  "wrist_videos": camera_hashes, "reference_images": reference_hashes,
                                  "parquet": file_hash(manifest["parquet_path"]), "context": prepared["input_hash"],
                                  "expert": read_json(run / "expert.json"), "catalog": read_json(run / "standard_catalog.json")})
        previous = read_json(output / "annotation.json")
        completion = read_json(output / "completion.json")
        if completion and completion.get("input_fingerprint") != fingerprint:
            raise ValueError("Changed experiment fingerprint: select a new run name")
        if completion and previous and not previous.get("execution_error"):
            save_labels(output, previous)
            return {"key": manifest["key"], "status": previous["status"], "cached": True}
        old_input = read_json(output / "experiment_input.json")
        if old_input and old_input["input_fingerprint"] != fingerprint:
            raise ValueError("Changed experiment input: use a new run")
        write_json(output / "experiment_input.json", {"input_fingerprint": fingerprint, "algorithm_hash": algorithm_hash,
                   "video_sha256": source_hash, "parquet_sha256": file_hash(manifest["parquet_path"]), "baseline": baseline.name})
        for name in ("input_manifest.json", "samples.jsonl", "pages.json"):
            shutil.copy2(source / name, output / name)
        for name in ("frames", "montage", "preview.mp4"):
            if (source / name).exists() and not (output / name).exists():
                (output / name).symlink_to(source / name, target_is_directory=(source / name).is_dir())
        samples = read_jsonl(output / "samples.jsonl")
        refinements = []
        try:
            axis = decode_video_axis(manifest["source_video_path"], manifest["frame_count"], manifest["source_fps"])
            write_json(output / "video_axis.json", {"timestamps_s": axis, "frame_index_base": 0})
            candidates = detect_gripper_event_candidates_from_parquet(manifest["parquet_path"], source_timestamps=axis, **config["events"]["detector"])
            write_json(output / "event_candidates.json", candidates)
            if not candidates.get("alignment_verified"):
                raise ValueError("Feedback/video alignment is not verified")
            if candidates["calibration"]["row_count"] != len(axis) or candidates["calibration"]["invalid_state_rows"]:
                raise ValueError("Dense evidence requires a complete finite state row for every source frame")
            keyframes = sorted({f for e in candidates["events"] for f in (e["source_frame_id"] - 1, e["source_frame_id"], e["source_frame_id"] + 1) if 0 <= f < len(axis)})
            rows = extract_source_frames(manifest["source_video_path"], keyframes, output / "event_frames", manifest["source_fps"], source_timestamps=axis)
            pages = render_frame_pages(rows, output / "event_montage", "events", heading=manifest["key"])
            if len(pages) > config["events"].get("max_candidate_pages", 16):
                raise ValueError("Candidate image budget exceeded; no events silently discarded")
            messages = _selection_messages(run, output, candidates, pages)
            if prepare_only:
                _record_request(output / "selection", messages, config["provider"])
                image_count = sum(part.get("type") == "image_url" for msg in messages if isinstance(msg["content"], list) for part in msg["content"])
                if image_count > config["provider"].get("max_images", 30):
                    raise ValueError("Initial request image budget exceeded")
                inventory = {
                    "destination": config["provider"]["base_url"], "model_id": config["provider"]["model_id"],
                    "source_key": manifest["key"], "initial_image_count": image_count,
                    "candidate_event_count": len(candidates["events"]), "source_fps": manifest["source_fps"],
                    "source_camera": "base_0_rgb", "source_frame_ids": keyframes,
                    "initial_payload": "stage standards, GPT-labelled expert, full 1fps main-view pages, event f-1/f/f+1 main-view pages, measured gripper candidate values and calibration",
                    "later_payload": "for each of at most three chosen boundaries: +/-0.75s of consecutive source frames from main and participant wrist cameras, corresponding measured left/right gripper feedback, stage reference",
                    "image_files": [{"path": str(path), "sha256": file_hash(path)} for path in
                        [run / s["reference_image"] for s in catalog["stages"]] +
                        [expert_dir / p["image_path"] for p in read_json(expert_dir / "pages.json")] +
                        [output / p["image_path"] for p in read_json(output / "pages.json")] +
                        [Path(p["image_path"]) for p in pages]],
                    "api_calls_made": 0,
                }
                write_json(output / "payload_inventory.json", inventory)
                return {"key": manifest["key"], "status": "prepared_for_selection", "candidate_events": len(candidates["events"]),
                        "selection_images": image_count, "usage": {"total_tokens": 0}}
            selection = _call(project, output / "selection", messages, config["provider"])
            validate_selection(selection, candidates["events"], samples)
            by_id = {e["event_id"]: e for e in candidates["events"]}
            if selection["applicable"] is not False and selection["task_type"] == "fold_cloth":
                import numpy as np
                import pyarrow.parquet as pq
                state_column = candidates["calibration"]["state_column"]
                table = pq.read_table(manifest["parquet_path"], columns=[state_column, "frame_index"])
                states = np.asarray(table[state_column].to_pylist())
                state_by_frame = {int(f): s for f, s in zip(table["frame_index"].to_pylist(), states)}
                for boundary in selection["boundaries"]:
                    selected = [by_id[x] for x in boundary["candidate_event_ids"]]
                    if not selected:
                        refinements.append({**boundary, "selected_event_ids": [], "end_frame": None, "visual_confirmed": False,
                                            "review_reasons": ["No event selected for this stage"]})
                        continue
                    stage_output = output / f"refine_stage_{boundary['from_stage']}"
                    pad = max(2, round(config["events"]["window_padding_s"] * manifest["source_fps"]))
                    anchor = max(e["source_frame_id"] for e in selected)
                    start, end = max(0, anchor - pad), min(len(axis) - 1, anchor + pad)
                    wanted = list(range(start, end + 1))
                    by_camera = {}
                    camera_paths = {"base_0_rgb": Path(manifest["source_video_path"])}
                    camera_paths.update({f"{arm}_wrist_0_rgb": _camera_path(manifest, f"{arm}_wrist_0_rgb") for arm in boundary["participant_arms"]})
                    for camera, path in camera_paths.items():
                        camera_axis = axis if camera == "base_0_rgb" else decode_video_axis(path, len(axis), manifest["source_fps"])
                        if any(abs(a - b) > .5 / manifest["source_fps"] + 1e-6 for a, b in zip(camera_axis, axis)):
                            raise ValueError(f"Camera timestamp misalignment: {camera}")
                        dense = extract_source_frames(path, wanted, stage_output / camera / "frames", manifest["source_fps"], source_timestamps=camera_axis)
                        by_camera[camera] = render_frame_pages(dense, stage_output / camera, "dense", heading=f"stage {boundary['from_stage']} {manifest['episode_id']}", camera=camera)
                    trace = [{"frame": f, "timestamp_s": axis[f], "left_raw": float(state_by_frame[f][6]), "right_raw": float(state_by_frame[f][13])} for f in wanted]
                    raw = _call(project, stage_output, _refinement_messages(run, boundary, selected, candidates, by_camera, trace, wanted), config["provider"])
                    validate_refinement(raw, wanted)
                    supported = all(raw[k] for k in ("operation_complete", "participating_grippers_released", "first_complete_confirmed", "next_frame_consistent")) and not raw["review_reasons"]
                    refinements.append({**boundary, **raw, "selected_event_ids": boundary["candidate_event_ids"],
                                        "visual_confirmed": supported, "window_start_frame": start, "window_end_frame": end,
                                        "selected_event_anchor_frame": anchor, "event_offset_frames": raw["end_frame"] - anchor if raw["end_frame"] is not None else None,
                                        "evidence_pages": by_camera, "camera_sha256": {k: file_hash(v) for k, v in camera_paths.items()}})
                    write_json(output / "refinement.json", {"boundaries": refinements, "source_fps": manifest["source_fps"]})
            annotation = make_annotation(manifest, selection, refinements, samples)
        except Exception as exc:
            annotation = {**manifest, "spec_version": VERSION, "precision": "source_frame_visual_proposal", "status": "needs_review",
                          "needs_review": True, "execution_error": type(exc).__name__, "review_reasons": [str(exc)[:240]],
                          "boundary_frames": [], "boundaries": refinements, "segments": [{"stage": -1, "start_frame": 0, "end_frame": manifest["frame_count"] - 1}],
                          "training_ready": False, "reference_is_human_gold": False}
        accounting = collect_usage(output)
        annotation.update({"usage": accounting["usage"], "api_accounting": accounting, "model_id": config["provider"]["model_id"],
                           "baseline_run": baseline.name, "input_fingerprint": fingerprint, "algorithm_hash": algorithm_hash})
        write_json(output / "refinement.json", {"boundaries": refinements, "source_fps": manifest["source_fps"]})
        write_json(output / "annotation.json", annotation)
        save_labels(output, annotation)
        if not annotation.get("execution_error"):
            write_json(output / "completion.json", {"input_fingerprint": fingerprint, "status": annotation["status"], "completed_at": now()})
        return {"key": manifest["key"], "status": annotation["status"], "boundaries": annotation["boundary_frames"],
                "usage": annotation["usage"], "reasons": annotation["review_reasons"]}


def run_event_probe(project, run, baseline, config, limit=None, prepare_only=False):
    project, run, baseline = Path(project).resolve(), Path(run).resolve(), Path(baseline).resolve()
    if run == baseline:
        raise ValueError("Experiment must use a new run directory")
    manifests = read_jsonl(baseline / "manifest.jsonl")
    if limit is not None:
        manifests = manifests[:limit]
    if not manifests:
        raise ValueError("Empty baseline manifest")
    baseline_config = read_json(baseline / "run.json")["config"]
    if config["provider"]["model_id"] != baseline_config["provider"]["model_id"]:
        raise ValueError("Use the same model snapshot for the algorithm comparison")
    algorithm_hash = object_hash({name: file_hash(Path(__file__).with_name(name)) for name in
                                  ("event_probe.py", "event_visuals.py", "gripper_events.py", "provider.py", "boundaries.py", "prompts.py")})
    run_spec = {"run_id": run.name, "config": config, "baseline": baseline.name, "algorithm_hash": algorithm_hash,
                "manifest_hash": object_hash(manifests), "spec_version": VERSION, "is_human_ground_truth": False}
    previous = read_json(run / "run.json")
    if previous and {k: previous.get(k) for k in run_spec} != run_spec:
        raise ValueError("Existing run has a different frozen experiment contract")
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "run.json", {**run_spec, "created_at": previous["created_at"] if previous else now()})
    write_jsonl(run / "manifest.jsonl", manifests)
    for name in ("expert.json", "standard_catalog.json"):
        shutil.copy2(baseline / name, run / name)
    for name in ("reference_expert", "standards"):
        if not (run / name).exists():
            (run / name).symlink_to(baseline / name, target_is_directory=True)
    results = []
    with ThreadPoolExecutor(max_workers=config["provider"].get("max_inflight", 2)) as pool:
        futures = [pool.submit(_one, project, run, baseline, m, config, algorithm_hash, prepare_only) for m in manifests]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"run_id": run.name, "baseline": baseline.name, "total": len(results), "counts": counts,
               "episodes": results, "prepare_only": prepare_only, "is_human_ground_truth": False, "updated_at": now()}
    write_json(run / "reports/summary.json", summary)
    return summary
