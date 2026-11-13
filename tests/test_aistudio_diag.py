"""Tests for opt-in AI Studio continuity diagnostics."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from unittest import mock

from bridge.aistudio_diag import diag_log, diagnostics_enabled, message_fingerprints


class AistudioDiagTest(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(diagnostics_enabled())

    def test_enabled_when_env_var_set(self) -> None:
        with mock.patch.dict(os.environ, {"ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS": "1"}):
            self.assertTrue(diagnostics_enabled())

    def test_emits_redacted_json_line_to_stderr(self) -> None:
        buf = io.StringIO()
        msgs = [{"i": 0, "role": "user", "h": "a" * 64}, {"i": 1, "role": "assistant", "h": "b" * 64}]
        with mock.patch.dict(os.environ, {"ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS": "1"}), mock.patch.object(sys, "stderr", buf):
            diag_log(
                event="aistudio.request",
                request_id="req_1",
                model="aistudio/gemini-3.8-flash-low",
                has_session_id=False,
                msgs=msgs,
            )
        line = buf.getvalue().strip()
        self.assertTrue(line.startswith("{"))
        self.assertIn('"evt":"aistudio.request"', line)
        self.assertIn('"request_id":"req_1"', line)
        # ``msgs`` must round-trip as a real JSON array of objects, not a stringified list.
        parsed = json.loads(line)
        self.assertIsInstance(parsed["msgs"], list)
        self.assertEqual(parsed["msgs"], msgs)
        self.assertEqual(len(parsed["msgs"][0]["h"]), 64)

    def test_no_output_when_disabled(self) -> None:
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sys, "stderr", buf):
            diag_log(event="aistudio.request", request_id="req_1")
        self.assertEqual(buf.getvalue(), "")

    def test_full_64_hex_hash_in_h(self) -> None:
        msgs = message_fingerprints([{"role": "user", "content": "hello"}])
        self.assertEqual(len(msgs), 1)
        h = msgs[0]["h"]
        # A full SHA-256 digest is 64 lowercase hex characters.
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))
        # The content hash is also a full 64-hex digest.
        self.assertEqual(len(msgs[0]["c"]), 64)

    def test_identifiers_are_redacted(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_plain_secret_1", "type": "function", "function": {"name": "fn", "arguments": '{"a":1}'}},
            ],
        }
        msgs = message_fingerprints([msg])
        entry = msgs[0]
        # The plain tool-call id must not appear verbatim.
        self.assertNotIn("call_plain_secret_1", str(entry))
        # Its redacted digest is a short 16-hex prefix.
        self.assertEqual(len(entry["tc"][0]), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in entry["tc"][0]))
        # A tool_call_id must also be redacted: original absent, digest present.
        tool_msg = {"role": "tool", "tool_call_id": "tool_result_secret_x", "content": "result"}
        entry2 = message_fingerprints([tool_msg])[0]
        self.assertNotIn("tool_result_secret_x", str(entry2))
        self.assertEqual(len(entry2["tcid"]), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in entry2["tcid"]))

    def test_any_field_change_changes_the_hash(self) -> None:
        base = {"role": "user", "content": "hello", "extra": 1}
        variants = [
            {"role": "user", "content": "hello", "extra": 2},  # value change
            {"role": "user", "content": "world", "extra": 1},  # content change
            {"role": "system", "content": "hello", "extra": 1},  # role change
            {"role": "user", "content": "hello"},  # field removed
        ]
        base_h = message_fingerprints([base])[0]["h"]
        for variant in variants:
            self.assertNotEqual(message_fingerprints([variant])[0]["h"], base_h)
        # Identical dicts produce identical hashes (order-independent canonicalization).
        reordered = {"extra": 1, "content": "hello", "role": "user"}
        self.assertEqual(message_fingerprints([base])[0]["h"], message_fingerprints([reordered])[0]["h"])

    def test_tolerates_malformed_structures(self) -> None:
        # Non-dict message, function as non-dict, tool_call without function, scalar tool_call.
        malformed = [
            "not a dict",
            {"role": "assistant", "tool_calls": [{"id": "tc1", "type": "function", "function": "bad"}]},
            {"role": "assistant", "tool_calls": [{"id": "tc2", "type": "function"}]},
            {"role": "assistant", "tool_calls": ["scalar"]},
            {"role": "assistant", "tool_calls": "not-a-list"},
            {"role": "user", "content": "ok"},
        ]
        # Must not raise; each malformed message still yields a fingerprint entry.
        msgs = message_fingerprints(malformed)
        self.assertEqual(len(msgs), len(malformed))
        for entry in msgs:
            self.assertIn("i", entry)
            if entry.get("role") == "?":
                # Non-dict message: only index + placeholder role, no structural hash.
                self.assertNotIn("h", entry)
            else:
                self.assertIn("h", entry)


if __name__ == "__main__":
    unittest.main()
