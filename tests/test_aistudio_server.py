"""End-to-end server integration tests for the Google AI Studio upstream (`aistudio/*`)."""

import copy
import json
import threading
import unittest
import urllib.error
import urllib.request
from typing import Any

from bridge.aistudio_backend import AistudioBackend
from bridge.aistudio_cache import InteractionStepCache
from bridge.errors import AuthenticationError
from bridge.server import create_server


class _FakeNoGoogleClient:
    """Simulates missing/expired Antigravity OAuth."""

    def load_code_assist(self) -> dict[str, Any]:
        raise AuthenticationError("Antigravity OAuth missing")

    def fetch_available_models(self, project: str) -> list[Any]:
        raise AssertionError("fetch_available_models must not run when antigravity_available=False")


class _MockAistudioKeyProvider:
    def __init__(self, key: str | None = "AIzaSyMockServerKey1234567890") -> None:
        self.key = key
        self.service = "abridge-aistudio"
        self.account = "api-key"

    def has_api_key(self) -> bool:
        return bool(self.key)

    def get_api_key(self, force_refresh: bool = False) -> str:
        if not self.key:
            raise AuthenticationError("No AI Studio API key configured")
        return self.key


class _RecordingAistudioClient:
    """In-memory mock of AistudioClient recording exact payloads sent to Interactions API."""

    def __init__(self, key_provider: _MockAistudioKeyProvider) -> None:
        self.key_provider = key_provider
        self.recorded_payloads: list[dict[str, Any]] = []
        self.next_interactions: list[dict[str, Any]] = []
        self.next_streams: list[list[dict[str, Any]]] = []

    def probe_api_key(self, api_key: str | None = None, timeout: float = 10.0) -> tuple[str, str]:
        if not self.key_provider.has_api_key():
            return ("rejected", "Missing key")
        return ("verified", "Mock key accepted")

    def create_interaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = self.key_provider.get_api_key()
        self.recorded_payloads.append(copy.deepcopy(payload))
        if self.next_interactions:
            return copy.deepcopy(self.next_interactions.pop(0))
        return {
            "id": "int_default",
            "status": "completed",
            "steps": [
                {"type": "thought", "signature": "sig_default", "summary": [{"type": "text", "text": "Thought"}]},
                {"type": "model_output", "content": [{"type": "text", "text": "Default reply"}]},
            ],
            "usage": {"total_input_tokens": 12, "total_output_tokens": 5, "total_tokens": 17},
        }

    def stream_interaction(self, payload: dict[str, Any], metrics: Any = None):
        _ = self.key_provider.get_api_key()
        self.recorded_payloads.append(copy.deepcopy(payload))
        if self.next_streams:
            for ev in copy.deepcopy(self.next_streams.pop(0)):
                yield ev
            return
        yield {"event_type": "step.start", "index": 0, "step": {"type": "thought"}}
        yield {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "sig_stream"}}
        yield {"event_type": "step.stop", "index": 0}
        yield {"event_type": "step.start", "index": 1, "step": {"type": "model_output", "content": []}}
        yield {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "Streamed OK"}}
        yield {"event_type": "step.stop", "index": 1}
        yield {
            "event_type": "interaction.completed",
            "interaction": {"id": "int_stream", "status": "completed", "usage": {"total_input_tokens": 8, "total_output_tokens": 3, "total_tokens": 11}},
        }


class TestAistudioServerEndpoints(unittest.TestCase):
    """End-to-end HTTP verification across all 3 protocols without Antigravity OAuth (S4, S5, S6)."""

    def _start_server(self, *, key: str | None = "AIzaSyMockServerKey1234567890"):
        key_prov = _MockAistudioKeyProvider(key=key)
        client = _RecordingAistudioClient(key_prov)
        backend = AistudioBackend(key_provider=key_prov, client=client)  # type: ignore[arg-type]
        cache = InteractionStepCache()
        srv = create_server(
            host="127.0.0.1",
            port=0,
            client=_FakeNoGoogleClient(),
            project=None,
            no_auth=True,
            aistudio_backend=backend,
            aistudio_cache=cache,
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        return srv, thread, client, cache

    def _request(self, port: int, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any] | str]:
        url = f"http://127.0.0.1:{port}{path}"
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                raw = resp.read().decode("utf-8")
                ct = resp.headers.get("Content-Type", "")
                if "application/json" in ct:
                    return resp.status, json.loads(raw)
                return resp.status, raw
        except urllib.error.HTTPError as err:
            raw = err.read().decode("utf-8")
            try:
                return err.code, json.loads(raw)
            except Exception:
                return err.code, raw

    def test_models_and_status_work_without_antigravity_and_never_leak_key(self) -> None:
        srv, _, _, _ = self._start_server()
        try:
            port = srv.server_address[1]
            code, models_body = self._request(port, "GET", "/v1/models")
            self.assertEqual(code, 200)
            ids = [m["id"] for m in models_body["data"]]
            self.assertIn("aistudio/gemini-4-hydrogen", ids)
            self.assertIn("aistudio/gemini-3.8-flash", ids)
            self.assertIn("aistudio/gemini-4-argon", ids)

            code2, st_body = self._request(port, "GET", "/api/aistudio/status")
            self.assertEqual(code2, 200)
            self.assertTrue(st_body["available"])
            self.assertFalse(st_body["quota_supported"])
            self.assertNotIn("AIza", json.dumps(st_body))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_chat_completions_multi_turn_text_and_409_after_cache_reset(self) -> None:
        srv, _, client, cache = self._start_server()
        try:
            port = srv.server_address[1]
            client.next_interactions.append({
                "id": "int_t1",
                "status": "completed",
                "steps": [
                    {"type": "thought", "signature": "sig_exact_t1"},
                    {"type": "model_output", "content": [{"type": "text", "text": "First reply"}]},
                ],
            })
            c1, r1 = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"x-abridge-session-id": "chat-s1"},
            )
            self.assertEqual(c1, 200)
            self.assertEqual(r1["choices"][0]["message"]["content"], "First reply")
            self.assertIs(client.recorded_payloads[0]["store"], False)

            # Turn 2: replays Turn 1's exact thought signature + model_output
            client.next_interactions.append({
                "id": "int_t2",
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "Second reply"}]},
                ],
            })
            c2, r2 = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "First reply"},
                        {"role": "user", "content": "Continue"},
                    ],
                },
                headers={"x-abridge-session-id": "chat-s1"},
            )
            self.assertEqual(c2, 200)
            self.assertEqual(r2["choices"][0]["message"]["content"], "Second reply")
            sent_steps = client.recorded_payloads[1]["input"]
            self.assertEqual(sent_steps[1], {"type": "thought", "signature": "sig_exact_t1"})

            # Simulate restart by wiping cache -> must return explicit 409 interaction_state_missing!
            cache._sessions.clear()
            cache._fingerprint_index.clear()
            c3, r3 = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "First reply"},
                        {"role": "user", "content": "Continue"},
                    ],
                },
                headers={"x-abridge-session-id": "chat-s1"},
            )
            self.assertEqual(c3, 409)
            self.assertEqual(r3["error"]["type"], "interaction_state_missing")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_anthropic_messages_streaming_and_tool_roundtrip(self) -> None:
        srv, _, client, _ = self._start_server()
        try:
            port = srv.server_address[1]
            client.next_streams.append([
                {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
                {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "sig_ant_stream"}},
                {"event_type": "step.stop", "index": 0},
                {"event_type": "step.start", "index": 1, "step": {"type": "function_call", "id": "call_ant_1", "name": "echo", "arguments": {}}},
                {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '{"msg":"hi"}'}},
                {"event_type": "step.stop", "index": 1},
                {"event_type": "interaction.requires_action", "interaction": {"id": "int_a1", "status": "requires_action"}},
            ])
            c1, sse_text = self._request(
                port,
                "POST",
                "/v1/messages",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "max_tokens": 200,
                    "stream": True,
                    "tools": [{"name": "echo", "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}}}],
                    "messages": [{"role": "user", "content": "Call echo"}],
                },
                headers={"x-abridge-session-id": "ant-stream-1"},
            )
            self.assertEqual(c1, 200)
            self.assertIn("tool_use", sse_text)
            self.assertIn("call_ant_1", sse_text)

            # Follow-up non-streaming with tool_result -> must replay reconstructed SSE steps!
            client.next_interactions.append({
                "id": "int_a2",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Echoed hi"}]}],
            })
            c2, r2 = self._request(
                port,
                "POST",
                "/v1/messages",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "max_tokens": 200,
                    "tools": [{"name": "echo", "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}}}],
                    "messages": [
                        {"role": "user", "content": "Call echo"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "call_ant_1", "name": "echo", "input": {"msg": "hi"}},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "call_ant_1", "content": "hi"},
                            ],
                        },
                    ],
                },
                headers={"x-abridge-session-id": "ant-stream-1"},
            )
            self.assertEqual(c2, 200)
            self.assertEqual(r2["content"][0]["text"], "Echoed hi")
            replayed = client.recorded_payloads[1]["input"]
            self.assertEqual(replayed[1], {"type": "thought", "signature": "sig_ant_stream"})
            self.assertEqual(replayed[2], {"type": "function_call", "id": "call_ant_1", "name": "echo", "arguments": {"msg": "hi"}})
            self.assertEqual(replayed[3]["type"], "function_result")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_responses_previous_response_id_chaining_never_sent_to_google(self) -> None:
        srv, _, client, _ = self._start_server()
        try:
            port = srv.server_address[1]
            client.next_interactions.append({
                "id": "int_r1",
                "status": "requires_action",
                "steps": [
                    {"type": "thought", "signature": "sig_resp_keep"},
                    {"type": "function_call", "id": "call_r1", "name": "lookup", "arguments": {"k": "v"}},
                ],
            })
            c1, r1 = self._request(
                port,
                "POST",
                "/v1/responses",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "input": "Run lookup",
                    "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"k": {"type": "string"}}}}],
                },
            )
            self.assertEqual(c1, 200)
            resp_id = r1["id"]

            client.next_interactions.append({
                "id": "int_r2",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Done lookup"}]}],
            })
            c2, r2 = self._request(
                port,
                "POST",
                "/v1/responses",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "previous_response_id": resp_id,
                    "input": [{"type": "function_call_output", "call_id": "call_r1", "output": "found"}],
                },
            )
            self.assertEqual(c2, 200)
            self.assertEqual(r2["output"][0]["content"][0]["text"], "Done lookup")
            # Abridge response id never leaked to Google as previous_interaction_id!
            self.assertNotIn("previous_interaction_id", client.recorded_payloads[1])
            self.assertEqual(client.recorded_payloads[1]["input"][1], {"type": "thought", "signature": "sig_resp_keep"})
        finally:
            srv.shutdown()
            srv.server_close()

    def test_invalid_thinking_level_returns_400_and_missing_aistudio_key_does_not_crash_boot(self) -> None:
        srv, _, _, _ = self._start_server(key=None)
        try:
            port = srv.server_address[1]
            # Status gracefully reports not configured
            c_st, st = self._request(port, "GET", "/api/aistudio/status")
            self.assertEqual(c_st, 200)
            self.assertFalse(st["available"])
            self.assertEqual(st["auth_status"], "not_configured")

            # Invalid thinking level on Flash returns 400
            c_bad, err_bad = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {"model": "aistudio/gemini-3.8-flash-xhigh", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(c_bad, 400)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_chat_streaming_parallel_tools_reverse_results_and_disconnect_non_commit(self) -> None:
        srv, _, client, cache = self._start_server()
        try:
            port = srv.server_address[1]
            # 1. Abrupt mid-stream disconnect (missing interaction.completed) must NOT commit state
            client.next_streams.append([
                {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
                {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "sig_partial"}},
                {"event_type": "step.stop", "index": 0},
                {"event_type": "step.start", "index": 1, "step": {"type": "model_output", "content": []}},
                {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "Partial"}},
            ])
            c_part, part_body = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {"model": "aistudio/gemini-4-hydrogen", "stream": True, "messages": [{"role": "user", "content": "Partial?"}]},
                headers={"x-abridge-session-id": "partial-sess"},
            )
            self.assertEqual(c_part, 200)
            self.assertIn("Partial", part_body)
            self.assertEqual(len(cache._sessions), 0)

            # 2. Streaming parallel tool calls in Chat Completions + reverse-order results replay
            client.next_streams.append([
                {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
                {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "sig_par_stream"}},
                {"event_type": "step.stop", "index": 0},
                {"event_type": "step.start", "index": 1, "step": {"type": "function_call", "id": "toolu_p1", "name": "fn_a", "arguments": {}}},
                {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '{"x":1}'}},
                {"event_type": "step.stop", "index": 1},
                {"event_type": "step.start", "index": 2, "step": {"type": "function_call", "id": "toolu_p2", "name": "fn_b", "arguments": {}}},
                {"event_type": "step.delta", "index": 2, "delta": {"type": "arguments_delta", "arguments": '{"y":2}'}},
                {"event_type": "step.stop", "index": 2},
                {"event_type": "interaction.requires_action", "interaction": {"id": "int_par", "status": "requires_action"}},
            ])
            c_s, sse_raw = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "stream": True,
                    "tools": [
                        {"type": "function", "function": {"name": "fn_a", "parameters": {"type": "object"}}},
                        {"type": "function", "function": {"name": "fn_b", "parameters": {"type": "object"}}},
                    ],
                    "messages": [{"role": "user", "content": "Run fn_a and fn_b"}],
                },
                headers={"x-abridge-session-id": "chat-par-1"},
            )
            self.assertEqual(c_s, 200)
            # OpenAI adapter converts toolu_ -> call_ while InteractionStepCache maps call_ -> toolu_!
            self.assertIn("call_p1", sse_raw)
            self.assertIn("call_p2", sse_raw)

            client.next_interactions.append({
                "id": "int_par_done",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Both done"}]}],
            })
            c_f, f_body = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "tools": [
                        {"type": "function", "function": {"name": "fn_a", "parameters": {"type": "object"}}},
                        {"type": "function", "function": {"name": "fn_b", "parameters": {"type": "object"}}},
                    ],
                    "messages": [
                        {"role": "user", "content": "Run fn_a and fn_b"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "call_p1", "type": "function", "function": {"name": "fn_a", "arguments": '{"x":1}'}},
                                {"id": "call_p2", "type": "function", "function": {"name": "fn_b", "arguments": '{"y":2}'}},
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_p2", "content": "res_b"},
                        {"role": "tool", "tool_call_id": "call_p1", "content": "res_a"},
                    ],
                },
                headers={"x-abridge-session-id": "chat-par-1"},
            )
            self.assertEqual(c_f, 200)
            self.assertEqual(f_body["choices"][0]["message"]["content"], "Both done")
            sent = client.recorded_payloads[-1]["input"]
            self.assertEqual(sent[1], {"type": "thought", "signature": "sig_par_stream"})
            # Mapped back to original native IDs toolu_p1, toolu_p2 in original call order!
            self.assertEqual(sent[4]["call_id"], "toolu_p1")
            self.assertEqual(sent[5]["call_id"], "toolu_p2")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_concurrent_anonymous_sessions_with_identical_visible_output_return_409_ambiguous(self) -> None:
        srv, _, client, _ = self._start_server()
        try:
            port = srv.server_address[1]
            client.next_interactions.append({
                "id": "int_anon_1",
                "status": "completed",
                "steps": [
                    {"type": "thought", "signature": "sig_anon_A"},
                    {"type": "model_output", "content": [{"type": "text", "text": "OK"}]},
                ],
            })
            client.next_interactions.append({
                "id": "int_anon_2",
                "status": "completed",
                "steps": [
                    {"type": "thought", "signature": "sig_anon_B"},
                    {"type": "model_output", "content": [{"type": "text", "text": "OK"}]},
                ],
            })
            self._request(port, "POST", "/v1/chat/completions", {"model": "aistudio/gemini-4-hydrogen", "messages": [{"role": "user", "content": "Say OK"}]})
            self._request(port, "POST", "/v1/chat/completions", {"model": "aistudio/gemini-4-hydrogen", "messages": [{"role": "user", "content": "Say OK"}]})

            c_amb, r_amb = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [
                        {"role": "user", "content": "Say OK"},
                        {"role": "assistant", "content": "OK"},
                        {"role": "user", "content": "Follow up"},
                    ],
                },
            )
            self.assertEqual(c_amb, 409)
            self.assertEqual(r_amb["error"]["type"], "interaction_state_ambiguous")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
