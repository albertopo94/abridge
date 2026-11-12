"""Unit tests for Junie ACP (Agent Client Protocol) client.

Junie's `junie --acp true` server, like OpenCode's, differs from Cursor's
`agent acp`:
- Handshake is `initialize` -> `session/new` (no authenticate step).
- Model selection is per-prompt via `session/set_model` before `session/prompt`.
- Region/availability and provider failures surface as JSON-RPC errors mapped
  to ForbiddenError / AuthenticationError (S2, S4, junie-upstream).
"""

import json
import queue
import unittest
from unittest.mock import MagicMock, patch

from bridge.errors import AuthenticationError, BridgeError, ForbiddenError, UpstreamTimeoutError
from bridge.junie_acp import (
    JunieAcpClient,
    extract_junie_error_message,
    is_junie_auth_error,
    is_junie_region_error,
)


class MockPipe:
    """Simulates stdio pipe with thread-safe line streaming."""

    def __init__(self) -> None:
        self.q: queue.Queue[bytes | None] = queue.Queue()

    def write_line(self, line_str: str) -> None:
        self.q.put(f"{line_str}\n".encode("utf-8"))

    def close(self) -> None:
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item


def _make_proc(session_id: str = "ses_test123"):
    """Builds a mocked Popen whose stdin writes answer ACP requests in order."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stderr = MagicMock()
    mock_stdout = MockPipe()
    mock_proc.stdout = mock_stdout
    state = {"model": None}

    def on_stdin_write(data: bytes) -> int:
        msg = json.loads(data.decode("utf-8").strip())
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "initialize":
            mock_stdout.write_line(json.dumps({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"protocolVersion": 1, "agentInfo": {"name": "Junie", "version": "26.11.1"}},
            }))
        elif method == "session/new":
            mock_stdout.write_line(json.dumps({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"sessionId": session_id},
            }))
        elif method == "session/set_model":
            state["model"] = params.get("modelId")
            mock_stdout.write_line(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}))
        elif method == "session/prompt":
            if state["model"] == "grok-4.6":
                mock_stdout.write_line(json.dumps({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": "Internal error: This model is not available in your country",
                        "data": {"service": "session", "errorName": "APIError"},
                    },
                }))
            elif state["model"] == "gemini-3-flash-preview":
                mock_stdout.write_line(json.dumps({
                    "jsonrpc": "2.0", "method": "session/update",
                    "params": {"sessionId": session_id, "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "PONG"},
                    }},
                }))
                mock_stdout.write_line(json.dumps({
                    "jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"},
                }))
            else:
                mock_stdout.write_line(json.dumps({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": "Authentication required: provider authentication required",
                              "data": {"providerId": "google"}},
                }))
        return len(data)

    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write.side_effect = on_stdin_write
    return mock_proc, mock_stdout


class TestJunieErrorHelpers(unittest.TestCase):
    """Tests for region/authentication error classification (S2, S4)."""

    def test_region_error_detected_from_acp_error(self) -> None:
        err = {
            "code": -32603,
            "message": "Internal error: This model is not available in your country",
            "data": {"service": "session", "errorName": "APIError"},
        }
        self.assertTrue(is_junie_region_error(err))

    def test_region_error_detected_from_stderr(self) -> None:
        text = '{"type":"error","data":{"message":"This model is not available in your country","statusCode":403}}'
        self.assertTrue(is_junie_region_error(text))

    def test_auth_error_is_not_region_error(self) -> None:
        err = {"code": -32000, "message": "Authentication required: provider authentication required"}
        self.assertFalse(is_junie_region_error(err))

    def test_generic_error_is_not_region_error(self) -> None:
        self.assertFalse(is_junie_region_error({"code": -32603, "message": "Internal error: boom"}))
        self.assertFalse(is_junie_region_error(None))
        self.assertFalse(is_junie_region_error(""))

    def test_auth_error_detected(self) -> None:
        err = {"code": -32000, "message": "Authentication required: provider authentication required"}
        self.assertTrue(is_junie_auth_error(err))

    def test_extract_message_from_dict_error(self) -> None:
        msg = extract_junie_error_message({"message": "Internal error: This model is not available in your country"})
        self.assertIn("not available in your country", msg)

    def test_extract_message_from_json_string(self) -> None:
        msg = extract_junie_error_message('{"error": {"data": {"message": "RegionError 403"}}}')
        self.assertIn("RegionError", msg)

    def test_extract_message_falls_back_to_raw_text(self) -> None:
        self.assertEqual(extract_junie_error_message("plain text failure"), "plain text failure")
        self.assertEqual(extract_junie_error_message(None), "")


class TestJunieAcpClient(unittest.TestCase):
    """Lifecycle, handshake, model selection, streaming and error mapping."""

    def test_init_state(self) -> None:
        client = JunieAcpClient(binary_path="/usr/local/bin/junie")
        self.assertFalse(client.is_alive())
        self.assertIsNone(client.session_id)

    @patch("subprocess.Popen")
    def test_start_handshake_no_authenticate_step(self, mock_popen: MagicMock) -> None:
        mock_proc, mock_stdout = _make_proc(session_id="ses_alpha")
        mock_popen.return_value = mock_proc

        client = JunieAcpClient(binary_path="/usr/local/bin/junie")
        client.start()

        self.assertTrue(client.is_alive())
        self.assertEqual(client.session_id, "ses_alpha")

        methods = []
        for call in mock_proc.stdin.write.call_args_list:
            payload = json.loads(call.args[0].decode("utf-8").strip())
            methods.append(payload.get("method"))
        self.assertIn("initialize", methods)
        self.assertIn("session/new", methods)
        self.assertNotIn("authenticate", methods)

        client.close()
        mock_stdout.close()

    @patch("subprocess.Popen")
    def test_start_uses_junie_acp_command(self, mock_popen: MagicMock) -> None:
        mock_proc, mock_stdout = _make_proc()
        mock_popen.return_value = mock_proc
        client = JunieAcpClient(binary_path="/usr/bin/junie")
        client.start()
        cmd = mock_popen.call_args.args[0]
        self.assertEqual(cmd, ["/usr/bin/junie", "--acp", "true"])
        client.close()
        mock_stdout.close()

    @patch("subprocess.Popen")
    def test_stream_prompt_selects_model_before_prompt(self, mock_popen: MagicMock) -> None:
        mock_proc, mock_stdout = _make_proc()
        mock_popen.return_value = mock_proc
        client = JunieAcpClient(binary_path="/usr/bin/junie")

        chunks = list(client.stream_prompt("Say PONG", model="gemini-3-flash-preview"))
        self.assertEqual("".join(chunks), "PONG")

        methods = []
        set_model_ids = []
        for call in mock_proc.stdin.write.call_args_list:
            payload = json.loads(call.args[0].decode("utf-8").strip())
            methods.append(payload.get("method"))
            if payload.get("method") == "session/set_model":
                set_model_ids.append(payload["params"]["modelId"])
        self.assertEqual(set_model_ids, ["gemini-3-flash-preview"])
        self.assertLess(methods.index("session/set_model"), methods.index("session/prompt"))
        client.close()
        mock_stdout.close()

    @patch("subprocess.Popen")
    def test_stream_prompt_region_error_raises_forbidden(self, mock_popen: MagicMock) -> None:
        # Region blocking is a 403 upstream (statusCode 403 / RegionError), so it maps
        # to ForbiddenError which the server already renders as 403 for all protocols.
        mock_proc, mock_stdout = _make_proc()
        mock_popen.return_value = mock_proc
        client = JunieAcpClient(binary_path="/usr/bin/junie")

        with self.assertRaises(ForbiddenError) as ctx:
            list(client.stream_prompt("Say PONG", model="grok-4.6", timeout=10.0))
        self.assertIn("not available in your country", str(ctx.exception))
        client.close()
        mock_stdout.close()

    @patch("subprocess.Popen")
    def test_stream_prompt_auth_error_raises_authentication_error(self, mock_popen: MagicMock) -> None:
        mock_proc, mock_stdout = _make_proc()
        mock_popen.return_value = mock_proc
        client = JunieAcpClient(binary_path="/usr/bin/junie")

        with self.assertRaises(AuthenticationError) as ctx:
            list(client.stream_prompt("Say PONG", model="grok-4.5", timeout=10.0))
        self.assertIn("Authentication required", str(ctx.exception))
        client.close()
        mock_stdout.close()

    @patch("subprocess.Popen")
    def test_permission_requests_are_auto_approved(self, mock_popen: MagicMock) -> None:
        mock_proc, mock_stdout = _make_proc()
        mock_popen.return_value = mock_proc
        client = JunieAcpClient(binary_path="/usr/bin/junie")
        client.start()

        # Simulate an incoming permission request from the agent.
        perm = {
            "jsonrpc": "2.0", "id": 555, "method": "session/request_permission",
            "params": {"sessionId": client.session_id, "options": [{"optionId": "allow-once"}]},
        }
        client._handle_incoming(perm)

        responses = []
        for call in mock_proc.stdin.write.call_args_list:
            payload = json.loads(call.args[0].decode("utf-8").strip())
            if payload.get("id") == 555 and "result" in payload:
                responses.append(payload)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["result"]["outcome"]["outcome"], "selected")
        client.close()
        mock_stdout.close()

    @patch("subprocess.Popen")
    def test_start_failure_binary_missing_raises_bridge_error(self, mock_popen: MagicMock) -> None:
        mock_popen.side_effect = FileNotFoundError("junie")
        client = JunieAcpClient(binary_path="/no/such/junie")
        with self.assertRaises(BridgeError) as ctx:
            client.start()
        self.assertIn("junie", str(ctx.exception).lower())

    @patch("subprocess.Popen")
    def test_start_handshake_timeout_raises(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = MagicMock()
        mock_proc.stdout = MockPipe()
        mock_proc.stdin = MagicMock()  # never answers
        mock_popen.return_value = mock_proc

        client = JunieAcpClient(binary_path="/usr/bin/junie")
        client._handshake_timeout = 0.4
        with self.assertRaises(UpstreamTimeoutError):
            client.start()
        mock_proc.stdin = MagicMock()  # keep close() quiet
        client.close()


if __name__ == "__main__":
    unittest.main()