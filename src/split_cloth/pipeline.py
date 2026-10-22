from __future__ import annotations

import io
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .boundaries import expand_labels, normalize_annotation, validate_review_frames
from .prompts import PROMPT_VERSION, build_messages
from .provider import request_annotation
from .sampling import prepare_episode, render_pages
from .util import episode_dir, file_hash, now, object_hash, read_json, read_jsonl, write_json


def save_labels(output, annotation):
    b = io.BytesIO()
    np.save(b, expand_labels(annotation))
    from .util import atomic_bytes
    atomic_bytes(Path(output) / "stage.npy", b.getvalue())


def prepare_run(project, run, config, expert_only=False):
    run = Path(run)
    if expert_only:
        m = read_json(run / "expert_manifest.json")
        return prepare_episode(m, run / "reference_expert", config)
    manifests = read_jsonl(run / "manifest.jsonl")
    results = []
    def one(m):
        out = episode_dir(run, m)
        try:
            result = prepare_episode(m, out, config)
            previous = read_json(out / "annotation.json", {})
            if previous.get("status") == "unreadable":
                (out / "annotation.json").unlink(missing_ok=True)
                (out / "completion.json").unlink(missing_ok=True)
            return {"key": m["key"], "status": "prepared", "pages": result["page_count"]}
        except Exception as e:
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "input_manifest.json", m)
            annotation = {**m, "status": "unreadable", "needs_review": False, "review_reasons": [type(e).__name__ + ": " + str(e)[:180]],
                "segments": [{"stage": -1, "start_frame": 0, "end_frame": m["frame_count"] - 1}], "boundaries": [], "boundary_frames": []}
            write_json(out / "annotation.json", annotation)
            save_labels(out, annotation)
            return {"key": m["key"], "status": "unreadable", "reason": annotation["review_reasons"]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one, m) for m in manifests]):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    write_json(run / "reports/preparation.json", results)
    return results


def register_expert(project, run, config, review_path):
    run = Path(run)
    review = read_json(review_path)
    boundaries = review["boundary_frames"]
    directory = run / "reference_expert"
    manifest = read_json(directory / "input_manifest.json")
    samples = read_jsonl(directory / "samples.jsonl")
    sample_frames = {s["source_frame_id"] for s in samples}
    if validate_review_frames(manifest["frame_count"], boundaries, review.get("stage4_completion_frame")) or review.get("stage4_completion_frame") is None:
        raise ValueError("Invalid expert boundaries or final completion evidence")
    if not set(boundaries + [review["stage4_completion_frame"]]).issubset(sample_frames):
        raise ValueError("Expert must select observed source samples")
    if review.get("reviewer") != "gpt_visual" or review.get("status") != "approved" or not review.get("stage_definitions") or len(review["stage_definitions"]) != 4 or not all(isinstance(x, str) and x.strip() for x in review["stage_definitions"]):
        raise ValueError("Expert requires explicit visual-review provenance and four stage definitions")
    render_pages(directory, manifest, samples, config["montage"], boundaries)
    with ZipFile(Path(project) / "standard.zip") as z:
        refs = ["image 6.png", "image 10.png", "image 2.png", "image 9.png"]
        stages = []
        for i, name in enumerate(refs):
            relative = f"standards/stage_{i+1}.png"
            p = run / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read("图片和附件/" + name))
            stages.append({"stage_id": i + 1, "description": review["stage_definitions"][i], "reference_image": relative,
                "completion_condition": "本阶段操作完成且参与动作的夹爪完全松开"})
    write_json(run / "standard_catalog.json", {"version": "gpt_interpretation_v1", "interpretation_status": "gpt_visual_interpretation_requires_human_confirmation",
        "standard_sha256": file_hash(Path(project) / "standard.zip"), "stages": stages})
    write_json(directory / "gpt_review.json", review)
    write_json(run / "expert.json", {**review, "directory": "reference_expert", "validated": True,
        "validation_level": "gpt_visual_reference_not_human_ground_truth", "key": manifest["key"], "manifest": manifest})


def annotate_run(project, run, config, limit=None):
    run = Path(run)
    manifests = read_jsonl(run / "manifest.jsonl")
    if limit is not None:
        manifests = manifests[:limit]
    expert = read_json(run / "expert.json")
    catalog = read_json(run / "standard_catalog.json")
    if not expert or not catalog:
        raise ValueError("Register a visually validated expert first")
    fingerprint = object_hash({"provider": config["provider"], "prompt_version": PROMPT_VERSION, "spec": config["spec_version"], "expert": expert, "catalog": catalog})
    def one(m):
        out = episode_dir(run, m)
        previous = read_json(out / "annotation.json")
        if previous and previous.get("status") == "unreadable" and not read_json(out / "input_manifest.json", {}).get("input_hash"):
            return {"key": m["key"], "status": "unreadable"}
        completion = read_json(out / "completion.json")
        prepared = read_json(out / "input_manifest.json")
        if not prepared:
            return {"key": m["key"], "status": "not_prepared"}
        input_fingerprint = object_hash({"input": prepared.get("input_hash"), "run": fingerprint})
        if completion and completion.get("input_fingerprint") == input_fingerprint and previous and not previous.get("execution_error"):
            if not (out / "stage.npy").is_file():
                save_labels(out, previous)
            return {"key": m["key"], "status": previous["status"], "cached": True}
        try:
            if completion or (out / "response.json").exists():
                request_meta = read_json(out / "request.json", {})
                if request_meta.get("input_fingerprint") != input_fingerprint:
                    if previous:
                        write_json(out / "invalidated" / (object_hash(previous) + ".json"), previous)
                    raise ValueError("Changed inference fingerprint: use a new run instead of stale response")
            messages = build_messages(run, out)
            write_json(out / "request.json", {"model_id": config["provider"]["model_id"], "prompt_version": PROMPT_VERSION,
                "input_fingerprint": input_fingerprint, "target_pages": prepared["page_count"], "expert_key": expert["key"],
                "created_at": now(), "system_prompt": messages[0]["content"], "text_parts": [x["text"] for x in messages[1]["content"] if x["type"] == "text"]})
            response = request_annotation(project, out, messages, config["provider"])
            raw = json.loads(response["content"])
            annotation = normalize_annotation(raw, read_jsonl(out / "samples.jsonl"), prepared)
            annotation.update({"spec_version": config["spec_version"], "precision": "sampled_1fps", "prompt_version": PROMPT_VERSION,
                "model_id": response["model"], "input_fingerprint": input_fingerprint, "usage": response["usage"], "elapsed_s": response["elapsed_s"],
                "expert_key": expert["key"], "reference_label_source": "gpt_visual", "reference_is_human_gold": False})
        except Exception as e:
            annotation = {**prepared, "status": "needs_review", "needs_review": True,
                "review_reasons": [str(e)[:180] if isinstance(e, (ValueError, RuntimeError, json.JSONDecodeError)) else type(e).__name__],
                "execution_error": type(e).__name__, "boundaries": [], "boundary_frames": [],
                "segments": [{"stage": -1, "start_frame": 0, "end_frame": m["frame_count"] - 1}]}
        annotation.update({k: m[k] for k in ["key", "dataset_id", "episode_index", "episode_id", "frame_count", "source_fps", "split"]})
        write_json(out / "annotation.json", annotation)
        save_labels(out, annotation)
        if not annotation.get("execution_error"):
            write_json(out / "completion.json", {"input_fingerprint": input_fingerprint, "status": annotation["status"], "completed_at": now()})
        else:
            (out / "completion.json").unlink(missing_ok=True)
        return {"key": m["key"], "status": annotation["status"], "usage": annotation.get("usage"), "reason": annotation.get("review_reasons")}
    results = []
    with ThreadPoolExecutor(max_workers=config["provider"].get("max_inflight", 3)) as pool:
        for future in as_completed([pool.submit(one, m) for m in manifests]):
            try:
                result = future.result()
            except Exception as e:
                result = {"status": "execution_error", "reason": type(e).__name__ + ": " + str(e)[:160]}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return summarize(run)


def summarize(run):
    run = Path(run)
    counts, rows, tokens = {}, [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for m in read_jsonl(run / "manifest.jsonl"):
        out = episode_dir(run, m)
        annotation = read_json(out / "annotation.json", {})
        status = annotation.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
        review = read_json(out / "gpt_review.json")
        row = {"key": m["key"], "split": m["split"], "status": status, "execution_error": annotation.get("execution_error"), "boundaries": annotation.get("boundary_frames", []), "reasons": annotation.get("review_reasons", []), "gpt_review": review}
        rows.append(row)
        for k in tokens:
            tokens[k] += annotation.get("usage", {}).get(k, 0)
    summary = {"run_id": run.name, "total": len(rows), "counts": counts, "tokens": tokens,
        "execution_errors": sum(bool(x["execution_error"]) for x in rows),
        "model_needs_review": sum(x["status"] == "needs_review" and not x["execution_error"] for x in rows),
        "gpt_reviewed": sum(x["gpt_review"] is not None for x in rows), "gpt_review_is_independent_human_gold": False,
        "updated_at": now(), "episodes": rows}
    write_json(run / "reports/summary.json", summary)
    lines = [f"# {run.name} — {len(rows)}条试标", "", "方法：主视角、1 FPS、时序网格、expert图文提示。", "", "## 处理状态", "", *[f"- {k}: {v}" for k, v in counts.items()], f"- 其中接口/执行失败: {summary['execution_errors']}", f"- 模型语义或结构待复核: {summary['model_needs_review']}", "", f"- GPT视觉复核条数: {summary['gpt_reviewed']}", f"- 总token: {tokens['total_tokens']}", "", "GPT复核是模型视觉判断，不等同人工金标准；未建立100条独立人工真值，不能据此报告全量准确率。", "", "## 待复核", ""]
    lines.extend(f"- {x['key']}: {', '.join(x['reasons'])}" for x in rows if x["status"] != "accepted")
    (run / "reports/summary.md").write_text("\n".join(lines) + "\n")
    return summary
