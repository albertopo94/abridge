"""Unit tests for AI Studio protocol transformers (OpenAI Chat, Anthropic, OpenAI Responses)."""

import copy
import json
import unittest

from bridge.aistudio_cache import InteractionStepCache
from bridge.aistudio_transform import (
    anthropic_to_aistudio_request,
    chat_to_aistudio_request,
    responses_to_aistudio_request,
)
from bridge.errors import InteractionStateError, InvalidRequestError


class TestChatCompletionsToAistudio(unittest.TestCase):
    """OpenAI Chat Completions -> Interactions API conversion (S2, S5)."""

    def test_basic_system_and_user_turn(self) -> None:
        cache = InteractionStepCache()
        req = chat_to_aistudio_request(
            {
                "model": "aistudio/gemini-4-hydrogen-xhigh",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 256,
                "stop": ["END"],
            },
            cache=cache,
        )
        self.assertEqual(req.official_model, "gemini-4-hydrogen")
        self.assertEqual(req.payload["model"], "gemini-4-hydrogen")
        self.assertIs(req.payload["store"], False)
        self.assertEqual(req.payload["system_instruction"], "Be concise.")
        self.assertEqual(req.payload["input"], [{"type": "user_input", "content": [{"type": "text", "text": "Hi"}]}])
        self.assertEqual(req.payload["generation_config"]["thinking_level"], "xhigh")
        self.assertEqual(req.payload["generation_config"]["max_output_tokens"], 256)
        self.assertEqual(req.payload["generation_config"]["stop_sequences"], ["END"])
        # Temperature must not be forwarded into Interactions generation_config
        self.assertNotIn("temperature", req.payload["generation_config"])

    def test_rejects_multimodal_image_explicitly(self) -> None:
        cache = InteractionStepCache()
        with self.assertRaises(InvalidRequestError):
            chat_to_aistudio_request(
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Look"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                            ],
                        }
                    ],
                },
                cache=cache,
            )

    def test_parallel_tool_results_reordered_to_match_original_calls_and_raw_steps_replayed(self) -> None:
        cache = InteractionStepCache()
        raw_steps = [
            {"type": "thought", "signature": "sig_parallel"},
            {"type": "function_call", "id": "native_1", "name": "get_temp", "arguments": {"city": "Paris"}},
            {"type": "function_call", "id": "native_2", "name": "get_time", "arguments": {"city": "Tokyo"}},
        ]
        # Suppose public IDs used `call_` equivalents (or mapped IDs)
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Paris temp and Tokyo time"}],
            assistant_visible={
                "text": "",
                "tool_calls": [
                    {"id": "pub_1", "name": "get_temp", "arguments": {"city": "Paris"}},
                    {"id": "pub_2", "name": "get_time", "arguments": {"city": "Tokyo"}},
                ],
            },
            raw_steps=raw_steps,
            call_id_map={"pub_1": "native_1", "pub_2": "native_2"},
            session_id="sess-parallel",
        )

        # Client sends results in REVERSE order (pub_2 before pub_1)
        req = chat_to_aistudio_request(
            {
                "model": "aistudio/gemini-4-hydrogen",
                "messages": [
                    {"role": "user", "content": "Paris temp and Tokyo time"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "pub_1", "type": "function", "function": {"name": "get_temp", "arguments": '{"city": "Paris"}'}},
                            {"id": "pub_2", "type": "function", "function": {"name": "get_time", "arguments": '{"city": "Tokyo"}'}},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "pub_2", "content": "18:00"},
                    {"role": "tool", "tool_call_id": "pub_1", "content": "18C"},
                ],
            },
            cache=cache,
            session_id="sess-parallel",
        )
        steps = req.payload["input"]
        self.assertEqual(steps[0]["type"], "user_input")
        # Raw steps preserved untouched (thought + 2 function_calls)
        self.assertEqual(steps[1:4], raw_steps)
        # function_results reordered to match native_1 then native_2!
        self.assertEqual(steps[4]["type"], "function_result")
        self.assertEqual(steps[4]["call_id"], "native_1")
        self.assertEqual(steps[5]["type"], "function_result")
        self.assertEqual(steps[5]["call_id"], "native_2")

    def test_missing_parallel_tool_result_raises_invalid_request(self) -> None:
        cache = InteractionStepCache()
        raw_steps = [
            {"type": "function_call", "id": "native_1", "name": "a", "arguments": {}},
            {"type": "function_call", "id": "native_2", "name": "b", "arguments": {}},
        ]
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Run both"}],
            assistant_visible={
                "text": "",
                "tool_calls": [
                    {"id": "native_1", "name": "a", "arguments": {}},
                    {"id": "native_2", "name": "b", "arguments": {}},
                ],
            },
            raw_steps=raw_steps,
        )
        with self.assertRaises(InvalidRequestError):
            chat_to_aistudio_request(
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [
                        {"role": "user", "content": "Run both"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "native_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                                {"id": "native_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                            ],
                        },
                        {"role": "tool", "tool_call_id": "native_1", "content": "ok"},
                    ],
                },
                cache=cache,
            )

    def test_missing_state_for_assistant_turn_raises_409_never_invents_dummy_sig(self) -> None:
        cache = InteractionStepCache()
        with self.assertRaises(InteractionStateError):
            chat_to_aistudio_request(
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "messages": [
                        {"role": "user", "content": "Turn 1"},
                        {"role": "assistant", "content": "Prior reply not in cache"},
                        {"role": "user", "content": "Turn 2"},
                    ],
                },
                cache=cache,
            )


class TestAnthropicToAistudio(unittest.TestCase):
    """Anthropic Messages -> Interactions API conversion (S2, S5)."""

    def test_tool_use_and_is_error_translated_accurately(self) -> None:
        cache = InteractionStepCache()
        raw_steps = [
            {"type": "thought", "signature": "sig_ant_1"},
            {"type": "function_call", "id": "call_g_ant", "name": "read_file", "arguments": {"path": "missing.txt"}},
        ]
        cache.record_turn(
            protocol="anthropic",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Read missing.txt"}],
            assistant_visible={
                "text": "",
                "tool_calls": [{"id": "call_g_ant", "name": "read_file", "arguments": {"path": "missing.txt"}}],
            },
            raw_steps=raw_steps,
            session_id="ant-sess-1",
        )

        req = anthropic_to_aistudio_request(
            {
                "model": "aistudio/gemini-4-hydrogen",
                "system": "System rule",
                "max_tokens": 512,
                "output_config": {"effort": "medium"},
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read file",
                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                    }
                ],
                "tool_choice": {"type": "tool", "name": "read_file"},
                "messages": [
                    {"role": "user", "content": "Read missing.txt"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "...", "signature": "sig_ant_1"},
                            {"type": "tool_use", "id": "call_g_ant", "name": "read_file", "input": {"path": "missing.txt"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call_g_ant", "content": "ENOENT", "is_error": True},
                        ],
                    },
                ],
            },
            cache=cache,
            session_id="ant-sess-1",
        )
        self.assertEqual(req.payload["system_instruction"], "System rule")
        self.assertEqual(req.payload["generation_config"]["thinking_level"], "medium")
        self.assertEqual(req.payload["generation_config"]["tool_choice"], {"allowed_tools": {"mode": "any", "tools": ["read_file"]}})
        self.assertEqual(req.payload["tools"], [{"type": "function", "name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}])

        steps = req.payload["input"]
        self.assertEqual(steps[1:3], raw_steps)
        self.assertEqual(
            steps[3],
            {
                "type": "function_result",
                "name": "read_file",
                "call_id": "call_g_ant",
                "result": [{"type": "text", "text": "ENOENT"}],
                "is_error": True,
            },
        )

    def test_rejects_budget_tokens_on_hydrogen_explicitly(self) -> None:
        cache = InteractionStepCache()
        with self.assertRaises(InvalidRequestError):
            anthropic_to_aistudio_request(
                {
                    "model": "aistudio/gemini-4-hydrogen",
                    "max_tokens": 100,
                    "thinking": {"type": "enabled", "budget_tokens": 4096},
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                cache=cache,
            )


class TestResponsesToAistudio(unittest.TestCase):
    """OpenAI Responses -> Interactions API (full input replay & previous_response_id) (S5)."""

    def test_previous_response_id_delta_continuation(self) -> None:
        cache = InteractionStepCache()
        snapshot = [
            {"type": "user_input", "content": [{"type": "text", "text": "Check Tokyo"}]},
            {"type": "thought", "signature": "sig_resp_t1"},
            {"type": "function_call", "id": "call_tok_1", "name": "get_time", "arguments": {"city": "Tokyo"}},
        ]
        cache.record_turn(
            protocol="openai_responses",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Check Tokyo"}],
            assistant_visible={"text": "", "tool_calls": [{"id": "call_tok_1", "name": "get_time", "arguments": {"city": "Tokyo"}}]},
            raw_steps=snapshot[1:],
            response_id="resp_local_123",
            full_steps_snapshot=copy.deepcopy(snapshot),
        )

        req = responses_to_aistudio_request(
            {
                "model": "aistudio/gemini-4-hydrogen",
                "previous_response_id": "resp_local_123",
                "instructions": "Format concisely",
                "reasoning": {"effort": "high"},
                "input": [
                    {"type": "function_call_output", "call_id": "call_tok_1", "output": '{"local_time":"18:00"}'},
                ],
            },
            cache=cache,
        )
        # Never forward previous_response_id to Google!
        self.assertNotIn("previous_interaction_id", req.payload)
        self.assertIs(req.payload["store"], False)
        self.assertEqual(req.payload["system_instruction"], "Format concisely")
        self.assertEqual(len(req.payload["input"]), 4)
        self.assertEqual(req.payload["input"][:3], snapshot)
        self.assertEqual(
            req.payload["input"][3],
            {
                "type": "function_result",
                "name": "get_time",
                "call_id": "call_tok_1",
                "result": [{"type": "text", "text": '{"local_time":"18:00"}'}],
            },
        )

    def test_full_input_replay_without_previous_response_id(self) -> None:
        cache = InteractionStepCache()
        raw_steps = [
            {"type": "thought", "signature": "sig_r_full"},
            {"type": "function_call", "id": "call_r_1", "name": "ping", "arguments": {"host": "a"}},
        ]
        cache.record_turn(
            protocol="openai_responses",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Ping a"}],
            assistant_visible={"text": "", "tool_calls": [{"id": "call_r_1", "name": "ping", "arguments": {"host": "a"}}]},
            raw_steps=raw_steps,
        )

        req = responses_to_aistudio_request(
            {
                "model": "aistudio/gemini-4-hydrogen",
                "input": [
                    {"role": "user", "content": "Ping a"},
                    {"type": "function_call", "id": "fc_1", "call_id": "call_r_1", "name": "ping", "arguments": json.dumps({"host": "a"})},
                    {"type": "function_call_output", "call_id": "call_r_1", "output": "ok"},
                ],
            },
            cache=cache,
        )
        self.assertEqual(len(req.payload["input"]), 4)
        self.assertEqual(req.payload["input"][1:3], raw_steps)
        self.assertEqual(req.payload["input"][3]["type"], "function_result")

    def test_mixed_previous_response_id_with_full_replayed_prefix_avoids_duplication(self) -> None:
        cache = InteractionStepCache()
        snapshot = [
            {"type": "user_input", "content": [{"type": "text", "text": "Ping a"}]},
            {"type": "thought", "signature": "sig_mix_1"},
            {"type": "function_call", "id": "call_m_1", "name": "ping", "arguments": {"host": "a"}},
        ]
        cache.record_turn(
            protocol="openai_responses",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Ping a"}],
            assistant_visible={"text": "", "tool_calls": [{"id": "call_m_1", "name": "ping", "arguments": {"host": "a"}}]},
            raw_steps=snapshot[1:],
            response_id="resp_mix_1",
            full_steps_snapshot=copy.deepcopy(snapshot),
        )
        req = responses_to_aistudio_request(
            {
                "model": "aistudio/gemini-4-hydrogen",
                "previous_response_id": "resp_mix_1",
                "input": [
                    {"role": "user", "content": "Ping a"},
                    {"type": "function_call", "id": "fc_1", "call_id": "call_m_1", "name": "ping", "arguments": '{"host":"a"}'},
                    {"type": "function_call_output", "call_id": "call_m_1", "output": "pong"},
                ],
            },
            cache=cache,
        )
        # Must NOT duplicate user_input or function_call!
        self.assertEqual(len(req.payload["input"]), 4)
        self.assertEqual(req.payload["input"][:3], snapshot)
        self.assertEqual(req.payload["input"][3]["type"], "function_result")


if __name__ == "__main__":
    unittest.main()
