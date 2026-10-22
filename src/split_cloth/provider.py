from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import httpx

from .util import file_hash, now, read_json, write_json


def credential(project, config):
    value = os.getenv("DASHSCOPE_API_KEY")
    if not value:
        p = Path(config["credential_file"])
        if not p.is_absolute():
            p = Path(project) / p
        value = p.read_text().strip().strip("\"'")
        match = re.search(r"sk-[A-Za-z0-9_-]{16,}", value)
        if match:
            value = match.group(0)
    if not value or any(c.isspace() for c in value):
        raise ValueError("Credential file must contain one credential; never log its contents")
    return value


def _reject_non_json_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def _content_is_json(content):
    # Schema and task correctness belong to the normalizer.  Only syntax is
    # checked here so an intentional model abstention is not billed again.
    if type(content) is not str:
        return False
    try:
        json.loads(content, parse_constant=_reject_non_json_constant)
    except (ValueError, TypeError, RecursionError):
        return False
    return True


def _archive_response(response_path, reason):
    """Retain the original response bytes while invalidating its cache entry."""
    directory = response_path.parent / "invalid_responses"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ("response_" + file_hash(response_path) + ".json")
    response_path.replace(destination)
    write_json(destination.with_suffix(".metadata.json"), {
        "reason": reason, "archived_at": now(), "response_file": destination.name,
    })
    return str(destination.relative_to(response_path.parent))


def request_annotation(project, output, messages, config):
    output = Path(output)
    response_path = output / "response.json"
    if response_path.exists():
        try:
            saved = read_json(response_path)
        except (ValueError, UnicodeError, RecursionError):
            _archive_response(response_path, "cached_response_unreadable")
        else:
            if type(saved) is dict and saved.get("finish_reason") == "stop" and _content_is_json(saved.get("content")):
                return saved
            reason = "cached_content_invalid_json" if type(saved) is dict and saved.get("finish_reason") == "stop" else "cached_response_incomplete"
            _archive_response(response_path, reason)
    images = sum(x.get("type") == "image_url" for m in messages if isinstance(m["content"], list) for x in m["content"])
    if images > config.get("max_images", 30):
        raise ValueError("input_budget_exceeded")
    attempts_file = output / "attempts.json"
    attempts = read_json(attempts_file, [])
    payload = {"model": config["model_id"], "messages": messages, "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_output_tokens", 3072), "stream": False,
        "response_format": {"type": "json_object"}, "enable_thinking": False}
    key = credential(project, config)
    with httpx.Client(timeout=config.get("timeout_s", 180)) as client:
        while len(attempts) < config.get("max_attempts", 3):
            event = {"attempt": len(attempts) + 1, "started_at": now(), "status": "inflight"}
            attempts.append(event)
            write_json(attempts_file, attempts)
            started = time.monotonic()
            retryable = False
            try:
                response = client.post(config["base_url"].rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=payload)
                event["http_status"] = response.status_code
                if response.status_code != 200:
                    # Do not retain error bodies: upstream may echo auth/request data.
                    retryable = response.status_code in (408, 429, 500, 502, 503, 504)
                    detail = ""
                    try:
                        error = response.json().get("error", {})
                        raw = str(error.get("code", "")) + " " + str(error.get("message", ""))
                        raw = raw.replace(key, "[REDACTED]")
                        detail = re.sub(r"[A-Za-z0-9_+/=-]{32,}", "[REDACTED]", raw)[:240]
                    except (ValueError, AttributeError):
                        pass
                    raise RuntimeError(f"provider_http_{response.status_code}: {detail}")
                data = response.json()
                choice = data["choices"][0]
                finish = choice.get("finish_reason")
                result = {"content": choice["message"]["content"], "finish_reason": finish,
                    "usage": data.get("usage", {}), "model": data.get("model", config["model_id"]),
                    "request_id": data.get("id"), "elapsed_s": round(time.monotonic() - started, 3), "image_count": images}
                event.update(status="responded", elapsed_s=result["elapsed_s"])
                write_json(attempts_file, attempts)
                write_json(response_path, result)
                if finish != "stop":
                    retryable = True
                    event["archived_response"] = _archive_response(response_path, "response_truncated")
                    raise ValueError("response_truncated")
                if not _content_is_json(result["content"]):
                    retryable = True
                    event["archived_response"] = _archive_response(response_path, "response_invalid_json")
                    raise ValueError("response_invalid_json")
                return result
            except (httpx.TimeoutException, httpx.TransportError):
                retryable = True
                event.update(status="transport_error", error="network_or_timeout")
            except Exception as exc:
                event.update(status="failed", error=str(exc)[:100] if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__)
                if not retryable:
                    write_json(attempts_file, attempts)
                    raise
            write_json(attempts_file, attempts)
            if retryable and len(attempts) < config.get("max_attempts", 3):
                time.sleep(2 ** len(attempts) + random.random())
    raise RuntimeError("retry_exhausted")
