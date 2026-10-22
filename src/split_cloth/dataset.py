from __future__ import annotations

import json
import random
from pathlib import Path

from .util import now, read_json, read_jsonl, write_json, write_jsonl


def make_manifest(base, dataset_id):
    root = Path(base) / dataset_id
    info = read_json(root / "meta/info.json")
    if not info or info.get("codebase_version") != "v2.1":
        raise ValueError(f"Unsupported dataset metadata: {dataset_id}")
    rows = []
    for ep in read_jsonl(root / "meta/episodes.jsonl"):
        i = ep["episode_index"]
        chunk = i // info.get("chunks_size", 1000)
        fmt = dict(episode_index=i, episode_chunk=chunk, video_key="base_0_rgb")
        video = root / info["video_path"].format(**fmt)
        parquet = root / info["data_path"].format(**fmt)
        eid = f"episode_{i:06d}"
        rows.append({
            "key": f"{dataset_id}__{eid}", "dataset_id": dataset_id, "dataset_root": str(root),
            "episode_index": i, "episode_id": eid, "frame_count": ep["length"],
            "source_fps": info["fps"], "duration_s": ep["length"] / info["fps"],
            "camera_key": "base_0_rgb", "source_video_path": str(video),
            "source_video_realpath": str(video.resolve()), "parquet_path": str(parquet),
            "robot_type": info.get("robot_type"), "task_text": ep.get("tasks", []),
        })
    return rows


def select_pilot(project, run, config, count=100):
    run = Path(run)
    if (run / "manifest.jsonl").exists():
        raise FileExistsError("Run already has a frozen manifest; select a new run ID")
    rng = random.Random(config["seed"])
    selected = []
    # A separate reference never enters either calibration or heldout target set.
    expert = next(x for x in make_manifest(config["dataset_base"], "fold_cloth_v3") if x["episode_index"] == 0)
    weights = config["datasets"]
    if count != sum(weights.values()):
        raise ValueError("Pilot count must equal configured dataset allocation")
    for name, number in weights.items():
        pool = [x for x in make_manifest(config["dataset_base"], name) if x["key"] != expert["key"]]
        pool.sort(key=lambda x: (x["frame_count"], x["episode_index"]))
        bins = [pool[i * len(pool) // 4:(i + 1) * len(pool) // 4] for i in range(4)]
        picks = []
        for i, group in enumerate(bins):
            n = number // 4 + int(i < number % 4)
            picks.extend(rng.sample(group, n))
        rng.shuffle(picks)
        cutoff = round(number * .6)
        for i, item in enumerate(picks):
            item["split"] = "calibration" if i < cutoff else "heldout"
        selected.extend(picks)
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "manifest.jsonl", selected)
    write_json(run / "expert_manifest.json", expert)
    write_json(run / "run.json", {"run_id": run.name, "created_at": now(), "config": config,
        "spec_version": config["spec_version"], "total": len(selected),
        "expert_excluded_from_targets": True, "status": "selected"})
    write_json(run / "pilot_split.json", {"seed": config["seed"], "calibration": [x["key"] for x in selected if x["split"] == "calibration"],
        "heldout": [x["key"] for x in selected if x["split"] == "heldout"], "expert_key": expert["key"]})
    return selected
