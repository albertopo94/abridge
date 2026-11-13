"""Unit tests for Google AI Studio backend (models, thinking validation, SSE Step Assembler)."""

import copy
import unittest
from unittest.mock import MagicMock

from bridge.aistudio_auth import AistudioKeychainProvider
from bridge.aistudio_backend import (
    AistudioBackend,
    AistudioStepAssembler,
    extract_events_from_interaction,
    is_aistudio_model,
    resolve_aistudio_model_and_thinking,
)
from bridge.errors import AuthenticationError, BridgeError, InvalidRequestError, ProtocolError
from bridge.events import StreamEvent


class TestAistudioModelResolution(unittest.TestCase):
    """Model prefix detection and per-model thinking_level validation (S1, S2)."""

    def test_is_aistudio_model(self) -> None:
        self.assertTrue(is_aistudio_model("aistudio/gemini-4-hydrogen"))
        self.assertTrue(is_aistudio_model("AISTUDIO/gemini-3.8-flash-high"))
        self.assertFalse(is_aistudio_model("gemini-4-hydrogen-high"))
        self.assertFalse(is_aistudio_model("cursor/auto"))
        self.assertFalse(is_aistudio_model("opencode/beam-free"))
        self.assertFalse(is_aistudio_model("junie/junie-lite"))
        self.assertFalse(is_aistudio_model(""))

    def test_resolves_base_models_with_documented_defaults(self) -> None:
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-4-hydrogen"), ("gemini-4-hydrogen", "high"))
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-3.8-flash"), ("gemini-3.8-flash", "medium"))
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-4-argon"), ("gemini-4-argon", "high"))

    def test_resolves_tier_suffixed_models(self) -> None:
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-4-hydrogen-xhigh"), ("gemini-4-hydrogen", "xhigh"))
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-4-hydrogen-low"), ("gemini-4-hydrogen", "low"))
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-3.8-flash-high"), ("gemini-3.8-flash", "high"))
        self.assertEqual(resolve_aistudio_model_and_thinking("aistudio/gemini-4-argon-medium"), ("gemini-4-argon", "medium"))

    def test_rejects_invalid_thinking_levels_per_model(self) -> None:
        with self.assertRaises(InvalidRequestError):
            resolve_aistudio_model_and_thinking("aistudio/gemini-3.8-flash-xhigh")
        with self.assertRaises(InvalidRequestError):
            resolve_aistudio_model_and_thinking("aistudio/gemini-3.8-flash", explicit_level="xhigh")
        with self.assertRaises(InvalidRequestError):
            resolve_aistudio_model_and_thinking("aistudio/gemini-4-hydrogen", explicit_level="minimal")
        with self.assertRaises(InvalidRequestError):
            resolve_aistudio_model_and_thinking("aistudio/gemini-3.8-flash", explicit_level="minimal")
        with self.assertRaises(InvalidRequestError):
            resolve_aistudio_model_and_thinking("aistudio/gemini-4-argon", explicit_level="xhigh")

    def test_list_models_publishes_all_three_with_tags(self) -> None:
        backend = AistudioBackend()
        models = backend.list_models()
        ids = [m["id"] for m in models]
        self.assertEqual(
            ids,
            [
                "aistudio/gemini-4-hydrogen",
                "aistudio/gemini-3.8-flash",
                "aistudio/gemini-4-argon",
            ],
        )
        for m in models:
            self.assertEqual(m["owned_by"], "aistudio")
            self.assertIn("[AI Studio]", m["display_name"])


class TestAistudioStepAssembler(unittest.TestCase):
    """SSE Step Assembler must emit StreamEvent AND reconstruct unmodified raw steps (S4)."""

    def test_reconstructs_thought_signature_text_and_parallel_function_calls(self) -> None:
        sse_events = [
            {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary", "content": {"type": "text", "text": "Plan... "}}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "sig_abc_123"}},
            {"event_type": "step.stop", "index": 0},
            {"event_type": "step.start", "index": 1, "step": {"type": "function_call", "id": "call_g_1", "name": "get_temp", "arguments": {}}},
            {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '{"city":'}},
            {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '"Paris"}'}},
            {"event_type": "step.stop", "index": 1},
            {"event_type": "step.start", "index": 2, "step": {"type": "function_call", "id": "call_g_2", "name": "get_time", "arguments": {}}},
            {"event_type": "step.delta", "index": 2, "delta": {"type": "arguments_delta", "arguments": '{"city":"Tokyo"}'}},
            {"event_type": "step.stop", "index": 2},
            {
                "event_type": "interaction.requires_action",
                "interaction": {
                    "id": "int_99",
                    "status": "requires_action",
                    "usage": {"total_input_tokens": 42, "total_output_tokens": 18, "total_tokens": 60},
                },
            },
        ]

        assembler = AistudioStepAssembler()
        emitted: list[StreamEvent] = []
        for ev in sse_events:
            emitted.extend(assembler.feed(ev))
        emitted.extend(assembler.finalize())

        self.assertTrue(assembler.terminal_reached)
        self.assertEqual(assembler.final_status, "requires_action")
        self.assertEqual(len(assembler.raw_steps), 3)
        self.assertEqual(
            assembler.raw_steps[0],
            {"type": "thought", "summary": [{"type": "text", "text": "Plan... "}], "signature": "sig_abc_123"},
        )
        self.assertEqual(
            assembler.raw_steps[1],
            {"type": "function_call", "id": "call_g_1", "name": "get_temp", "arguments": {"city": "Paris"}},
        )
        self.assertEqual(
            assembler.raw_steps[2],
            {"type": "function_call", "id": "call_g_2", "name": "get_time", "arguments": {"city": "Tokyo"}},
        )

        # Normalized StreamEvents carry thought, signature, both tool calls, and tool_calls finish_reason
        self.assertTrue(any(e.delta_thought == "Plan... " for e in emitted))
        self.assertTrue(any(e.thought_signature == "sig_abc_123" for e in emitted))
        tool_events = [tc for e in emitted for tc in e.tool_calls]
        self.assertEqual([(tc.id, tc.name, tc.args) for tc in tool_events], [
            ("call_g_1", "get_temp", {"city": "Paris"}),
            ("call_g_2", "get_time", {"city": "Tokyo"}),
        ])
        self.assertEqual(emitted[-1].finish_reason, "tool_calls")
        self.assertEqual(emitted[-1].usage.to_dict(), {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60})

    def test_stream_text_and_incomplete_disconnect_fails_closed(self) -> None:
        assembler = AistudioStepAssembler()
        events = [
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output", "content": []}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Hello "}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "world"}},
        ]
        emitted: list[StreamEvent] = []
        for ev in events:
            emitted.extend(assembler.feed(ev))
        # Stream terminated abruptly without step.stop or interaction.completed!
        self.assertFalse(assembler.terminal_reached)
        with self.assertRaises(BridgeError):
            assembler.finalize()

    def test_malformed_function_arguments_json_raises_protocol_error(self) -> None:
        assembler = AistudioStepAssembler()
        assembler.feed({"event_type": "step.start", "index": 0, "step": {"type": "function_call", "id": "c1", "name": "fn", "arguments": {}}})
        assembler.feed({"event_type": "step.delta", "index": 0, "delta": {"type": "arguments_delta", "arguments": '{"broken": '}})
        with self.assertRaises(ProtocolError):
            list(assembler.feed({"event_type": "step.stop", "index": 0}))

    def test_unsupported_multimodal_delta_raises_invalid_request(self) -> None:
        assembler = AistudioStepAssembler()
        assembler.feed({"event_type": "step.start", "index": 0, "step": {"type": "model_output", "content": []}})
        with self.assertRaises(InvalidRequestError):
            list(assembler.feed({"event_type": "step.delta", "index": 0, "delta": {"type": "image", "data": "..."}}))

    def test_non_streaming_interaction_preserves_exact_raw_steps(self) -> None:
        raw_interaction = {
            "id": "int_non_stream",
            "status": "completed",
            "steps": [
                {"type": "thought", "signature": "raw_sig_exact", "summary": [{"type": "text", "text": "Reason"}]},
                {"type": "model_output", "content": [{"type": "text", "text": "Final answer"}]},
            ],
            "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_tokens": 15},
        }
        snapshot = copy.deepcopy(raw_interaction["steps"])
        events, raw_steps, status = extract_events_from_interaction(raw_interaction)
        self.assertEqual(status, "completed")
        self.assertEqual(raw_steps, snapshot)
        self.assertTrue(any(e.delta_thought == "Reason" for e in events))
        self.assertTrue(any(e.delta_text == "Final answer" for e in events))
        self.assertEqual(events[-1].finish_reason, "stop")


class TestAistudioBackendStatus(unittest.TestCase):
    """Independent availability & status checks without leaking secrets (S3, S6)."""

    def test_status_reports_not_configured_without_blocking_gateway(self) -> None:
        prov = MagicMock(spec=AistudioKeychainProvider)
        prov.service = "abridge-aistudio"
        prov.account = "api-key"
        prov.has_api_key.return_value = False
        backend = AistudioBackend(key_provider=prov)

        st = backend.status()
        self.assertFalse(st["available"])
        self.assertEqual(st["auth_status"], "not_configured")
        self.assertFalse(st["quota_supported"])
        self.assertEqual(st["model_count"], 3)

    def test_status_never_leaks_api_key(self) -> None:
        secret = "AIzaSyNeverLeakThisKey99999999"
        prov = MagicMock(spec=AistudioKeychainProvider)
        prov.service = "abridge-aistudio"
        prov.account = "api-key"
        prov.has_api_key.return_value = True
        prov.get_api_key.return_value = secret
        client = MagicMock()
        client.probe_api_key.return_value = ("verified", "API key accepted")

        backend = AistudioBackend(key_provider=prov, client=client)
        st = backend.status(verify_live=True)
        self.assertTrue(st["available"])
        self.assertEqual(st["auth_status"], "verified")
        self.assertNotIn("AIza", str(st))

    def test_missing_key_raises_authentication_error_on_inference(self) -> None:
        prov = MagicMock(spec=AistudioKeychainProvider)
        prov.has_api_key.return_value = False
        prov.get_api_key.side_effect = AuthenticationError("Missing key")
        client = MagicMock()
        client.create_interaction.side_effect = AuthenticationError("Missing key")
        backend = AistudioBackend(key_provider=prov, client=client)
        with self.assertRaises(AuthenticationError):
            backend.run_non_streaming({"model": "gemini-4-hydrogen", "input": []})


if __name__ == "__main__":
    unittest.main()
