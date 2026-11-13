"""Unit tests for InteractionStepCache (stateless continuity, locks, TTL/LRU, ambiguity)."""

import copy
import threading
import time
import unittest

from bridge.aistudio_cache import (
    InteractionStepCache,
    canonical_hash,
)
from bridge.errors import InteractionStateError


class TestInteractionStepCache(unittest.TestCase):
    """RAM-only step cache invariants from Checkpoints 011, 012, 016."""

    def test_exact_raw_steps_preserved_and_resolved_by_prefix_hash(self) -> None:
        cache = InteractionStepCache()
        raw_steps = [
            {"type": "thought", "signature": "sig_raw_1", "summary": [{"type": "text", "text": "Think"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        snapshot = copy.deepcopy(raw_steps)
        prefix = [{"role": "user", "text": "Hi"}]
        assistant_vis = {"text": "Hello!", "tool_calls": []}

        rec = cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=raw_steps,
            request_hash="req_1",
        )
        # Mutating caller's input must not corrupt stored raw_steps
        raw_steps[0]["signature"] = "mutated"

        resolved = cache.resolve_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
        )
        self.assertEqual(resolved.raw_steps, snapshot)
        self.assertEqual(resolved.turn_id, rec.turn_id)

    def test_missing_or_truncated_prefix_raises_409_interaction_state_missing(self) -> None:
        cache = InteractionStepCache()
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Original turn 1"}],
            assistant_visible={"text": "Answer 1", "tool_calls": []},
            raw_steps=[{"type": "thought", "signature": "s1"}, {"type": "model_output", "content": [{"type": "text", "text": "Answer 1"}]}],
        )

        # Compacted/truncated history produces a different prefix -> fail closed (409)
        with self.assertRaises(InteractionStateError) as ctx:
            cache.resolve_turn(
                protocol="openai_chat",
                official_model="gemini-4-hydrogen",
                prefix_items=[{"role": "user", "text": "Compacted summary"}],
                assistant_visible={"text": "Answer 1", "tool_calls": []},
            )
        self.assertEqual(ctx.exception.code, "interaction_state_missing")

    def test_ambiguity_detected_when_two_sessions_have_identical_visible_history_and_different_signatures(self) -> None:
        cache = InteractionStepCache()
        prefix = [{"role": "user", "text": "Same prompt"}]
        assistant_vis = {"text": "Same reply", "tool_calls": []}

        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=[{"type": "thought", "signature": "sig_session_A"}, {"type": "model_output", "content": [{"type": "text", "text": "Same reply"}]}],
            request_hash="req_A",
        )
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=[{"type": "thought", "signature": "sig_session_B"}, {"type": "model_output", "content": [{"type": "text", "text": "Same reply"}]}],
            request_hash="req_B",
        )

        with self.assertRaises(InteractionStateError) as ctx:
            cache.resolve_turn(
                protocol="openai_chat",
                official_model="gemini-4-hydrogen",
                prefix_items=prefix,
                assistant_visible=assistant_vis,
            )
        self.assertEqual(ctx.exception.code, "interaction_state_ambiguous")

    def test_explicit_session_id_isolates_identical_histories(self) -> None:
        cache = InteractionStepCache()
        prefix = [{"role": "user", "text": "Same prompt"}]
        assistant_vis = {"text": "Same reply", "tool_calls": []}
        steps_a = [{"type": "thought", "signature": "sig_A"}, {"type": "model_output", "content": [{"type": "text", "text": "Same reply"}]}]
        steps_b = [{"type": "thought", "signature": "sig_B"}, {"type": "model_output", "content": [{"type": "text", "text": "Same reply"}]}]

        cache.record_turn(
            protocol="anthropic",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=steps_a,
            session_id="sess-A",
        )
        cache.record_turn(
            protocol="anthropic",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=steps_b,
            session_id="sess-B",
        )

        res_a = cache.resolve_turn(
            protocol="anthropic",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            session_id="sess-A",
        )
        res_b = cache.resolve_turn(
            protocol="anthropic",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            session_id="sess-B",
        )
        self.assertEqual(res_a.raw_steps[0]["signature"], "sig_A")
        self.assertEqual(res_b.raw_steps[0]["signature"], "sig_B")

    def test_idempotent_retry_does_not_create_duplicate_ambiguity(self) -> None:
        cache = InteractionStepCache()
        prefix = [{"role": "user", "text": "Hello"}]
        assistant_vis = {"text": "World", "tool_calls": []}
        steps = [{"type": "thought", "signature": "sig_1"}, {"type": "model_output", "content": [{"type": "text", "text": "World"}]}]

        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=steps,
            request_hash="req_same",
        )
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
            raw_steps=steps,
            request_hash="req_same",
        )
        resolved = cache.resolve_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=prefix,
            assistant_visible=assistant_vis,
        )
        self.assertEqual(resolved.raw_steps[0]["signature"], "sig_1")

    def test_responses_previous_response_id_lifecycle_and_isolation(self) -> None:
        cache = InteractionStepCache()
        full_steps = [
            {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
            {"type": "thought", "signature": "sig_resp_1"},
            {"type": "function_call", "id": "call_native_1", "name": "read_file", "arguments": {"path": "a.py"}},
        ]
        cache.record_turn(
            protocol="openai_responses",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"role": "user", "text": "Hi"}],
            assistant_visible={"text": "", "tool_calls": [{"id": "call_native_1", "name": "read_file"}]},
            raw_steps=full_steps[1:],
            response_id="resp_abridge_1",
            full_steps_snapshot=full_steps,
            call_id_map={"call_public_1": "call_native_1"},
        )

        rec = cache.resolve_response_id("resp_abridge_1", official_model="gemini-4-hydrogen")
        self.assertEqual(rec.full_steps_snapshot, full_steps)
        self.assertEqual(rec.call_id_map["call_public_1"], "call_native_1")

        # Model mismatch must reject
        with self.assertRaises(InteractionStateError):
            cache.resolve_response_id("resp_abridge_1", official_model="gemini-3.8-flash")

        # Unknown id must reject
        with self.assertRaises(InteractionStateError):
            cache.resolve_response_id("resp_unknown", official_model="gemini-4-hydrogen")

    def test_ttl_expiration_and_lru_eviction(self) -> None:
        fake_now = [1000.0]
        cache = InteractionStepCache(ttl_seconds=10.0, max_sessions=2, clock=lambda: fake_now[0])

        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"u": 1}],
            assistant_visible={"a": 1},
            raw_steps=[{"type": "model_output", "content": [{"type": "text", "text": "1"}]}],
            session_id="s1",
        )
        fake_now[0] = 1005.0
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"u": 2}],
            assistant_visible={"a": 2},
            raw_steps=[{"type": "model_output", "content": [{"type": "text", "text": "2"}]}],
            session_id="s2",
        )
        fake_now[0] = 1006.0
        cache.record_turn(
            protocol="openai_chat",
            official_model="gemini-4-hydrogen",
            prefix_items=[{"u": 3}],
            assistant_visible={"a": 3},
            raw_steps=[{"type": "model_output", "content": [{"type": "text", "text": "3"}]}],
            session_id="s3",
        )

        # s1 was LRU evicted when s3 entered (max_sessions=2)
        with self.assertRaises(InteractionStateError):
            cache.resolve_turn("openai_chat", "gemini-4-hydrogen", [{"u": 1}], {"a": 1}, session_id="s1")

        # Advance past TTL -> s2 and s3 expire
        fake_now[0] = 1020.0
        with self.assertRaises(InteractionStateError):
            cache.resolve_turn("openai_chat", "gemini-4-hydrogen", [{"u": 3}], {"a": 3}, session_id="s3")

    def test_session_lock_serializes_same_session_concurrent_requests(self) -> None:
        cache = InteractionStepCache()
        order: list[str] = []
        entered_first = threading.Event()

        def worker1() -> None:
            with cache.session_lock("sess-concurrent"):
                order.append("w1_start")
                entered_first.set()
                time.sleep(0.08)
                order.append("w1_end")

        def worker2() -> None:
            entered_first.wait(timeout=1.0)
            with cache.session_lock("sess-concurrent"):
                order.append("w2_start")
                order.append("w2_end")

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        self.assertEqual(order, ["w1_start", "w1_end", "w2_start", "w2_end"])

    def test_canonical_hash_is_order_independent_for_dicts(self) -> None:
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
