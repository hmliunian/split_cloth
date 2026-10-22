"""Offline provider regressions: invalid caches, truncation, and cost caps."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from split_cloth.provider import request_annotation
from split_cloth.util import read_json, write_json


def reply(content='{"task_type":"unknown"}', finish="stop", request_id="test-1", status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = (
        {"error": {"code": "test_error", "message": "offline request failed"}}
        if status != 200 else {
            "id": request_id, "model": "offline-model",
            "usage": {"total_tokens": 8},
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
        }
    )
    return response


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "episode"
        self.output.mkdir()
        self.config = {
            "model_id": "offline-model", "base_url": "https://offline.invalid/v1",
            "credential_file": "unused.txt", "max_attempts": 3,
        }
        self.messages = [{"role": "user", "content": [{"type": "text", "text": "offline fixture"}]}]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.credential = self.stack.enter_context(patch("split_cloth.provider.credential", return_value="sk-offline-fixture-only"))
        self.stack.enter_context(patch("split_cloth.provider.time.sleep"))
        client_type = self.stack.enter_context(patch("split_cloth.provider.httpx.Client"))
        self.client = client_type.return_value.__enter__.return_value
        self.client.post.return_value = reply()

    def request(self):
        return request_annotation(self.output.parent, self.output, self.messages, self.config)

    def response_archives(self):
        directory = self.output / "invalid_responses"
        return sorted(path for path in directory.glob("response_*.json") if not path.name.endswith(".metadata.json"))

    def seed_attempts(self, count):
        write_json(self.output / "attempts.json", [
            {"attempt": index + 1, "status": "failed", "error": "previous_failure"}
            for index in range(count)
        ])

    def test_valid_cached_json_returns_without_network_or_credentials(self):
        saved = {"finish_reason": "stop", "content": '{"needs_review":true}', "model": "offline-model"}
        write_json(self.output / "response.json", saved)
        self.assertEqual(self.request(), saved)
        self.client.post.assert_not_called()
        self.credential.assert_not_called()

    def test_invalid_cached_content_is_preserved_then_retried(self):
        saved = {"finish_reason": "stop", "content": '{"broken":', "model": "offline-model"}
        write_json(self.output / "response.json", saved)
        original_bytes = (self.output / "response.json").read_bytes()
        self.seed_attempts(1)
        result = self.request()
        self.assertEqual(json.loads(result["content"])["task_type"], "unknown")
        self.assertEqual(self.client.post.call_count, 1)
        self.assertEqual(len(read_json(self.output / "attempts.json")), 2)
        self.assertEqual(len(self.response_archives()), 1)
        self.assertEqual(self.response_archives()[0].read_bytes(), original_bytes)
        self.assertEqual(read_json(self.output / "response.json")["finish_reason"], "stop")

    def test_new_invalid_json_retries_within_existing_total_budget(self):
        self.client.post.side_effect = [reply("broken", request_id="bad"), reply(request_id="good")]
        result = self.request()
        attempts = read_json(self.output / "attempts.json")
        self.assertEqual(result["request_id"], "good")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["error"], "response_invalid_json")
        self.assertEqual(attempts[0]["status"], "failed")
        self.assertEqual(attempts[1]["status"], "responded")
        self.assertEqual(read_json(self.response_archives()[0])["content"], "broken")

    def test_new_truncated_response_is_archived_and_retried(self):
        self.client.post.side_effect = [reply('{"truncated":', finish="length"), reply(request_id="complete")]
        result = self.request()
        self.assertEqual(result["request_id"], "complete")
        self.assertEqual(self.client.post.call_count, 2)
        self.assertEqual(read_json(self.response_archives()[0])["finish_reason"], "length")
        self.assertEqual(read_json(self.output / "attempts.json")[0]["error"], "response_truncated")

    def test_three_invalid_json_responses_exhaust_budget_without_reusable_cache(self):
        self.client.post.side_effect = [reply(f"broken-{index}", request_id=f"bad-{index}") for index in range(3)]
        with self.assertRaisesRegex(RuntimeError, "retry_exhausted"):
            self.request()
        self.assertEqual(self.client.post.call_count, 3)
        self.assertEqual(len(read_json(self.output / "attempts.json")), 3)
        self.assertEqual(len(self.response_archives()), 3)
        self.assertFalse((self.output / "response.json").exists())
        with self.assertRaisesRegex(RuntimeError, "retry_exhausted"):
            self.request()
        self.assertEqual(self.client.post.call_count, 3, "Resume must not silently reset the total billing budget")

    def test_exhausted_bad_cache_is_invalidated_without_extra_request(self):
        self.seed_attempts(3)
        write_json(self.output / "response.json", {"finish_reason": "stop", "content": "invalid"})
        with self.assertRaisesRegex(RuntimeError, "retry_exhausted"):
            self.request()
        self.client.post.assert_not_called()
        self.assertEqual(len(self.response_archives()), 1)
        self.assertFalse((self.output / "response.json").exists())

    def test_schema_abstention_is_not_retried(self):
        # Syntax-valid but incomplete schema must be handled by normalize_annotation,
        # rather than repeatedly buying another response from the provider.
        self.client.post.return_value = reply("{}")
        self.assertEqual(self.request()["content"], "{}")
        self.assertEqual(self.client.post.call_count, 1)
        self.assertEqual(len(self.response_archives()), 0)

    def test_http_400_does_not_consume_all_retry_slots(self):
        self.client.post.return_value = reply(status=400)
        with self.assertRaisesRegex(RuntimeError, "provider_http_400"):
            self.request()
        self.assertEqual(self.client.post.call_count, 1)
        self.assertEqual(len(read_json(self.output / "attempts.json")), 1)
        self.assertFalse((self.output / "response.json").exists())

    def test_corrupt_cache_wrapper_is_archived_byte_for_byte(self):
        original = b'not-a-json-response-file\n'
        (self.output / "response.json").write_bytes(original)
        self.request()
        self.assertEqual(self.response_archives()[0].read_bytes(), original)
        self.assertEqual(self.client.post.call_count, 1)

    def test_truncation_on_third_total_attempt_does_not_reset_budget(self):
        self.seed_attempts(2)
        self.client.post.return_value = reply("incomplete", finish="length")
        with self.assertRaisesRegex(RuntimeError, "retry_exhausted"):
            self.request()
        self.assertEqual(self.client.post.call_count, 1)
        self.assertEqual(len(read_json(self.output / "attempts.json")), 3)

    def test_non_json_numeric_constants_are_not_cached_as_valid_json(self):
        self.client.post.side_effect = [reply('{"value":NaN}'), reply(request_id="valid")]
        self.assertEqual(self.request()["request_id"], "valid")
        self.assertEqual(self.client.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
