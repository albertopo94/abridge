"""Tests for opt-in AI Studio continuity diagnostics."""

from __future__ import annotations

import io
import os
import sys
import unittest
from unittest import mock

from bridge.aistudio_diag import diag_log, diagnostics_enabled


class AistudioDiagTest(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(diagnostics_enabled())

    def test_enabled_when_env_var_set(self) -> None:
        with mock.patch.dict(os.environ, {"ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS": "1"}):
            self.assertTrue(diagnostics_enabled())

    def test_emits_redacted_json_line_to_stderr(self) -> None:
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS": "1"}), mock.patch.object(sys, "stderr", buf):
            diag_log(event="aistudio.request", request_id="req_1", model="aistudio/gemini-3.8-flash-low", has_session_id=False)
        line = buf.getvalue().strip()
        self.assertTrue(line.startswith("{"))
        self.assertIn('"evt":"aistudio.request"', line)
        self.assertIn('"request_id":"req_1"', line)

    def test_no_output_when_disabled(self) -> None:
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sys, "stderr", buf):
            diag_log(event="aistudio.request", request_id="req_1")
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()