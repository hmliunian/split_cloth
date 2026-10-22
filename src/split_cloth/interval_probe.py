"""Section 5.2 experiment: fixed feedback nodes -> interval montages -> AI IDs.

There is one semantic request per episode. The model cannot move a node,
interpret raw gripper values, or run a later frame refinement pass.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from .event_probe import collect_usage
from .event_visuals import decode_video_axis
from .gripper_nodes import prepare_gripper_nodes
from .interval_contract import SPEC_VERSION, normalize_interval_selection
from .interval_montage import prepare_interval_montages
from .pipeline import save_labels
from .prompts import image_item
from .provider import request_annotation
from .util import episode_dir, file_hash, now, object_hash, read_json, read_jsonl, write_json, write_jsonl

SYSTEM = """你是机器人折衣过程的阶段语义标注员。依据文档5.1四阶段图与5.2边界规范工作。
5.2原文示例：421帧衣物本体完成但夹爪尚未完全松开，仍属阶段1；422帧夹爪完全松开，阶段1结束；423帧在松爪状态运动，属于下一阶段。
本次算法：传统算法固定夹爪开闭节点；相邻节点切出连续区间，每区间内部1FPS抽帧并保留首尾，独立拼图；你只做阶段语义识别并选择给定节点ID。
节点估计来自实测夹爪反馈的稳定平台，时间戳已校验；它是释放候选，不是已校准的物理金标。开闭数值不交给你重新解释。
只允许eligible_stage_end=true的松开节点作阶段结束。参与本次操作的手必须在该节点open_arms中；unknown不能假装open。
一个阶段可跨多个区间。prepare包括取衣、抖开、重抓、铺平；这些动作完成并松爪不能单独结束阶段1。
first_fold必须看到第一侧实际纵折及折边，second_fold必须看到另一侧实际纵折成为长条，cross_fold必须看到长条横折成为小矩形，remove是随后拿走。
全过程按区间编号和时间读取；不能把前期反复铺衣当三次折叠，不能按时长等分或复制EXPERT时间。
节点所属帧归结束的阶段，下一原始帧归下一阶段。禁止输出或修改任何帧号、时间戳；只输出一个JSON对象。
不适用、缺阶段、返工、节点漏检、遮挡或不能确认时返回null节点并needs_review=true，不强凑三个边界。"""

SCHEMA = """输出严格包含：
task_type:fold_cloth|fold_pants|hang_cloth|mixed|other|unknown;
applicable:true|false|null; all_stages_observed:boolean; needs_review:boolean; review_reasons:string[];
interval_observations:每个TARGET区间恰好一项且顺序一致，
 {interval_id:string,activity:prepare|first_fold|second_fold|cross_fold|remove|other|unknown,operation_complete:boolean,evidence_text:string}；
stage_end_nodes:恰好三项按1/2/3顺序，
 {stage_id:int,node_id:给定eligible节点ID或null,participant_arms:[left和/或right],evidence_interval_ids:[TARGET区间ID],evidence_text:string}；
stage4_completion:{observed:boolean,interval_id:TARGET区间ID或null,evidence_text:string}。
每区间evidence_text简短描述可见动作，约10-25个汉字或一短句。operation_complete指该区间右端已完成该活动，可用false表示仅进行中。
每阶段的证据区间必须位于前一所选边界之后、当前边界以内，并包含以所选节点结束的末端区间；该末端区间operation_complete必须为true。
阶段1证据必须含first_fold，阶段2含second_fold，阶段3含cross_fold；prepare/other/unknown不能替代这些操作证据。
三个非空节点须严格按时间递增。第4阶段完成需另外引用第三边界后的remove区间及完成证据。
未解决的阶段保留对应null槽位。没有实际松爪节点时不能输出VIDEO_END代替。禁止额外字段，尤其禁止frame/end_frame/timestamp字段。"""


def _text(value):
    return {"type": "text", "text": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}


def build_interval_messages(run, output, node_result, prepared):
    run, output = Path(run), Path(output)
    catalog = read_json(run / "standard_catalog.json")
    expert = read_json(run / "expert.json")
    content = [_text({"stage_definitions": catalog["stages"], "boundary_spec": "standard.zip Untitled.md section 5.2"})]
    for stage in catalog["stages"]:
        content.extend([_text(f"REFERENCE_STANDARD stage={stage['stage_id']}"), image_item(run / stage["reference_image"])])
    expert_dir = run / expert["directory"]
    content.append(_text("EXPERT是另一条视频的GPT粗视觉参考，非人工金标。仅参考动作语义和阶段顺序，不可复制节点、帧号或时长比例。TARGET节点ID只在TARGET有效。"))
    for page in read_json(expert_dir / "pages.json"):
        content.extend([_text(f"EXPERT {page['page_id']}"), image_item(expert_dir / page["image_path"])])
    table = [{"node_id": n["node_id"], "time_s": round(n["timestamp_s"], 4),
              "changes": [{"arm": c["arm"], "kind": c["kind"]} for c in n["changes"]],
              "open_arms": n["open_arms"], "closed_arms": n["closed_arms"], "unknown_arms": n["unknown_arms"],
              "eligible_stage_end": n["eligible_stage_end"]} for n in node_result["nodes"]]
    content.append(_text({"TARGET_fixed_nodes": table, "alignment_verified": node_result["alignment_verified"],
                          "node_precision": "feedback_settled_estimate_not_human_ground_truth"}))
    by_interval = {i["interval_id"]: i for i in prepared["intervals"]}
    for page in prepared["pages"]:
        i = by_interval[page["interval_id"]]
        content.append(_text({"TARGET_interval": i["interval_id"], "page_id": page["page_id"],
                              "left_node": i["left_node_id"], "right_node": i["right_node_id"],
                              "start_s": round(i["start_timestamp_s"], 4), "end_s": round(i["end_timestamp_s"], 4),
                              "sample_ids": page["sample_ids"],
                              "instruction": "本页只含本区间；与上一页同区间则继续，否则进入下一个区间。"}))
        content.append(image_item(page["image_path"]))
    content.append(_text(SCHEMA))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def _request_record(output, messages, provider):
    safe = deepcopy(messages)
    images = 0
    for message in safe:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    images += 1
                    part["image_url"]["url"] = hashlib.sha256(part["image_url"]["url"].encode()).hexdigest()
    if images > provider.get("max_images", 96):
        raise ValueError(f"input_budget_exceeded: {images} images exceed explicit budget; no intervals dropped")
    fingerprint = object_hash({"spec": SPEC_VERSION, "messages": safe, "provider": provider})
    previous = read_json(output / "request.json")
    if previous and previous.get("input_fingerprint") != fingerprint:
        raise ValueError("Changed node-selection request; use a new run")
    write_json(output / "request.json", {"spec_version": SPEC_VERSION, "input_fingerprint": fingerprint,
               "model_id": provider["model_id"], "image_count": images, "messages": safe,
               "created_at": previous["created_at"] if previous else now()})
    return images


def _empty_annotation(manifest, exc):
    return {**manifest, "status": "needs_review", "needs_review": True, "spec_version": SPEC_VERSION,
            "precision": "fixed_gripper_node_selection", "execution_error": type(exc).__name__,
            "review_reasons": [str(exc)[:300]], "boundary_frames": [], "boundaries": [],
            "segments": [{"stage": -1, "start_frame": 0, "end_frame": manifest["frame_count"] - 1}],
            "training_ready": False, "reference_is_human_gold": False}


def _one(project, run, baseline, manifest, config, code_hash, prepare_only=False):
    output = episode_dir(run, manifest)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".interval.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        annotation = None
        try:
            source = episode_dir(baseline, manifest)
            old_input = read_json(source / "input_manifest.json")
            if not old_input:
                raise ValueError("Missing validated baseline source metadata")
            catalog = read_json(run / "standard_catalog.json")
            expert_dir = run / read_json(run / "expert.json")["directory"]
            reference_paths = [run / stage["reference_image"] for stage in catalog["stages"]]
            reference_paths += [expert_dir / p["image_path"] for p in read_json(expert_dir / "pages.json")]
            video_hash, parquet_hash = file_hash(manifest["source_video_path"]), file_hash(manifest["parquet_path"])
            if video_hash != old_input["source_video_sha256"]:
                raise ValueError("Source video has changed since baseline")
            standard_hash = file_hash(project / "standard.zip")
            if catalog.get("standard_sha256") and catalog["standard_sha256"] != standard_hash:
                raise ValueError("Source standard.zip changed since registered stage definitions")
            fingerprint = object_hash({"config": config, "code_hash": code_hash, "video": video_hash,
                                       "standard_zip_sha256": standard_hash,
                                       "parquet": parquet_hash, "expert": read_json(run / "expert.json"), "catalog": catalog,
                                       "references": [file_hash(path) for path in reference_paths]})
            prior_input = read_json(output / "experiment_input.json")
            if prior_input and prior_input["input_fingerprint"] != fingerprint:
                # Reject incompatible runs without replacing old annotations.
                return {"key": manifest["key"], "status": "input_conflict", "reasons": ["Changed frozen inputs: use a new run"]}
            previous = read_json(output / "annotation.json")
            completion = read_json(output / "completion.json")
            if completion and completion.get("input_fingerprint") == fingerprint and previous and not previous.get("execution_error"):
                save_labels(output, previous)
                return {"key": manifest["key"], "status": previous["status"], "cached": True,
                        "boundaries": previous["boundary_frames"], "usage": previous.get("usage", {})}
            write_json(output / "experiment_input.json", {"input_fingerprint": fingerprint, "algorithm_hash": code_hash,
                       "video_sha256": video_hash, "parquet_sha256": parquet_hash, "baseline_run": baseline.name,
                       "standard_zip_sha256": standard_hash,
                       "boundary_source": "standard.zip/Untitled.md section 5.2"})
            axis = decode_video_axis(manifest["source_video_path"], manifest["frame_count"], manifest["source_fps"])
            nodes = prepare_gripper_nodes(manifest["parquet_path"], axis, **config["nodes"])
            for node in nodes["nodes"]:
                node["node_type"] = "+".join(f"{change['arm']}_{change['kind']}" for change in node["changes"])
            write_json(output / "gripper_nodes.json", nodes)
            if not nodes["alignment_verified"]:
                raise ValueError("Unverified feedback/video alignment")
            prepared = prepare_interval_montages(manifest, nodes["nodes"], axis, output, config.get("interval_montage"))
            # Keep GUI-compatible indexes; algorithm reads only interval pages.
            write_json(output / "pages.json", [{**page, "image_path": str(Path(page["image_path"]).relative_to(output))} for page in prepared["pages"]])
            write_jsonl(output / "samples.jsonl", [{**sample, "sample_seq": seq,
                        "image_path": str(Path(sample["image_path"]).relative_to(output))} for seq, sample in enumerate(prepared["samples"])])
            write_json(output / "input_manifest.json", {**old_input, "sample_count": len(prepared["samples"]),
                       "page_count": len(prepared["pages"]), "precision": "interval_1fps_with_endpoints",
                       "input_hash": fingerprint, "sample_fps": 1.0})
            if (source / "preview.mp4").exists() and not (output / "preview.mp4").exists():
                shutil.copy2(source / "preview.mp4", output / "preview.mp4")
            messages = build_interval_messages(run, output, nodes, prepared)
            selection_dir = output / "selection"
            image_count = _request_record(selection_dir, messages, config["provider"])
            write_json(output / "payload_inventory.json", {
                "destination": config["provider"]["base_url"], "model_id": config["provider"]["model_id"],
                "node_count": len(nodes["nodes"]), "eligible_nodes": sum(n["eligible_stage_end"] for n in nodes["nodes"]),
                "interval_count": len(prepared["intervals"]), "interval_pages": len(prepared["pages"]), "image_count": image_count,
                "sampling": "1fps independently within each interval, plus original first and last frames",
                "raw_gripper_values_sent": False, "local_frame_refinement": False,
                "payload": "stage reference images, same GPT expert reference, fixed node/arm-state table, chronologically ordered independent interval montages",
                "image_files": [{"path": str(path), "sha256": file_hash(path)} for path in reference_paths + [Path(p["image_path"]) for p in prepared["pages"]]],
            })
            if prepare_only:
                return {"key": manifest["key"], "status": "prepared_for_selection", "nodes": len(nodes["nodes"]),
                        "intervals": len(prepared["intervals"]), "images": image_count, "usage": {"total_tokens": 0}}
            response = request_annotation(project, selection_dir, messages, config["provider"])
            raw = json.loads(response["content"])
            write_json(selection_dir / "model_output.json", raw)
            annotation = normalize_interval_selection(raw, nodes["nodes"], prepared["intervals"], manifest)
            annotation.update({**manifest, "input_fingerprint": fingerprint, "algorithm_hash": code_hash,
                               "model_id": response["model"], "baseline_run": baseline.name,
                               "boundary_spec": "standard.zip/Untitled.md section 5.2", "node_count": len(nodes["nodes"]),
                               "interval_count": len(prepared["intervals"]), "image_count": image_count})
        except Exception as exc:
            annotation = _empty_annotation(manifest, exc)
        accounting = collect_usage(output)
        annotation.update({"usage": accounting["usage"], "api_accounting": accounting,
                           "model_id": annotation.get("model_id", config["provider"]["model_id"])})
        write_json(output / "annotation.json", annotation)
        save_labels(output, annotation)
        if not annotation.get("execution_error"):
            write_json(output / "completion.json", {"input_fingerprint": fingerprint, "status": annotation["status"], "completed_at": now()})
        return {"key": manifest["key"], "status": annotation["status"], "boundaries": annotation["boundary_frames"],
                "usage": annotation["usage"], "reasons": annotation.get("review_reasons", [])}


def run_interval_probe(project, run, baseline, config, limit=None, prepare_only=False):
    project, run, baseline = Path(project).resolve(), Path(run).resolve(), Path(baseline).resolve()
    if run == baseline:
        raise ValueError("A separate interval run is required")
    manifests = read_jsonl(baseline / "manifest.jsonl")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        manifests = manifests[:limit]
    if not manifests:
        raise ValueError("Empty frozen sample manifest")
    original_config = read_json(baseline / "run.json")["config"]
    if original_config["provider"]["model_id"] != config["provider"]["model_id"]:
        raise ValueError("Model snapshot must match baseline")
    code_hash = object_hash({name: file_hash(Path(__file__).with_name(name)) for name in
                            ("interval_probe.py", "interval_montage.py", "interval_contract.py", "gripper_nodes.py",
                             "gripper_events.py", "event_visuals.py", "event_probe.py", "provider.py", "prompts.py", "util.py", "boundaries.py")})
    contract = {"run_id": run.name, "baseline_run": baseline.name, "config": config, "algorithm_hash": code_hash,
                "manifest_hash": object_hash(manifests), "spec_version": SPEC_VERSION, "is_human_ground_truth": False}
    prior = read_json(run / "run.json")
    if prior and any(prior.get(key) != value for key, value in contract.items()):
        raise ValueError("Changed frozen interval contract; use a new run name")
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "run.json", {**contract, "created_at": prior["created_at"] if prior else now()})
    write_jsonl(run / "manifest.jsonl", manifests)
    for name in ("expert.json", "standard_catalog.json"):
        shutil.copy2(baseline / name, run / name)
    for name in ("reference_expert", "standards"):
        if not (run / name).exists():
            # The review server confines assets to this run/episode. Copying
            # keeps those checks intact and makes the new run independently viewable.
            shutil.copytree(baseline / name, run / name)
    results = []
    with ThreadPoolExecutor(max_workers=config["provider"].get("max_inflight", 2)) as pool:
        futures = [pool.submit(_one, project, run, baseline, m, config, code_hash, prepare_only) for m in manifests]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"run_id": run.name, "baseline": baseline.name, "total": len(results), "counts": counts,
               "prepare_only": prepare_only, "episodes": results, "updated_at": now(), "is_human_ground_truth": False}
    write_json(run / "reports/summary.json", summary)
    return summary
