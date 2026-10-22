"""Frozen textual criteria distilled once from the reference images by GPT.

The target VLM receives this text and target images only. Reference images,
expert timings and prior target predictions are not part of its payload.
"""
from pathlib import Path

from .util import read_json


SCHEMA_VERSION = "text_stage_criteria_v1"


def validate_stage_criteria(value):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Invalid text stage criteria schema")
    stages = value.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError("Text criteria require four ordered stages")
    for expected, stage in enumerate(stages, 1):
        if not isinstance(stage, dict) or type(stage.get("stage_id")) is not int or stage["stage_id"] != expected:
            raise ValueError("Text criteria stage IDs must be 1,2,3,4 in order")
        for field in ("name", "action"):
            if not isinstance(stage.get(field), str) or not stage[field].strip():
                raise ValueError(f"Missing text stage criterion: {field}")
        for field in ("completion_criteria", "exclude"):
            if not isinstance(stage.get(field), list) or not stage[field] or any(not isinstance(s, str) or not s.strip() for s in stage[field]):
                raise ValueError(f"Invalid text stage criterion list: {field}")
    for field in ("temporal_rules", "limitations"):
        if not isinstance(value.get(field), list) or not value[field] or any(not isinstance(s, str) or not s.strip() for s in value[field]):
            raise ValueError(f"Invalid text criteria {field}")
    if not isinstance(value.get("boundary_rule"), str) or not value["boundary_rule"].strip():
        raise ValueError("Text criteria require the 5.2 boundary rule")
    if not isinstance(value.get("provenance"), dict) or value["provenance"].get("method") != "gpt_visual_summary":
        raise ValueError("Text criteria require GPT visual summary provenance")
    return value


def load_stage_criteria(path):
    return validate_stage_criteria(read_json(Path(path)))
