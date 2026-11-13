"""Unit tests for Google AI Studio Interactions HTTP client."""

from io import BytesIO
import json
import socket
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from bridge.aistudio_auth import AistudioKeychainProvider
from bridge.aistudio_client import (
    AISTUDIO_API_REVISION,
    AistudioClient,
    AistudioStreamMetrics,
    verify_aistudio_api_key,
)
from bridge.errors import (
    AuthenticationError,
    CapacityExhaustedError,
    ForbiddenError,
    InvalidRequestError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamTimeoutError,
)


def _mock_http_error(code: int, body: dict | str, headers: dict | None = None) -> urllib.error.HTTPError:
    raw = json.dumps(body).encode("utf-8") if isinstance(body, dict) else body.encode("utf-8")
    err = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com/v1beta/interactions",
        code=code,
        msg="Error",
        hdrs=headers or {},
        fp=BytesIO(raw),
    )
    return err


class TestAistudioClientRequests(unittest.TestCase):
    """Non-streaming and streaming Interactions API HTTP behavior."""

    def _provider(self, key: str = "AIzaSyTestSecretKey1234567890") -> MagicMock:
        prov = MagicMock(spec=AistudioKeychainProvider)
        prov.get_token = None
        prov.get_api_key.return_value = key
        return prov

    @patch("bridge.aistudio_client.urllib.request.urlopen")
    def test_create_interaction_sends_x_goog_api_key_revision_and_store_false(self, mock_urlopen: MagicMock) -> None:
        resp_body = {"id": "int_1", "status": "completed", "steps": []}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_body).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = AistudioClient(key_provider=self._provider("AIzaSyMyKey999999999999"))
        out = client.create_interaction({"model": "gemini-4-hydrogen", "input": [], "store": True, "previous_interaction_id": "bad"})
        self.assertEqual(out, resp_body)

        req = mock_urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(req.headers.get("X-goog-api-key") or req.get_header("x-goog-api-key"), "AIzaSyMyKey999999999999")
        self.assertEqual(req.get_header("Api-revision"), AISTUDIO_API_REVISION)

        sent = json.loads(req.data.decode("utf-8"))
        self.assertIs(sent["store"], False)
        self.assertNotIn("previous_interaction_id", sent)

    @patch("bridge.aistudio_client.urllib.request.urlopen")
    def test_stream_interaction_parses_sse_and_records_metrics(self, mock_urlopen: MagicMock) -> None:
        sse_lines = [
            b"event: step.start\n",
            b'data: {"index": 0, "step": {"type": "model_output", "content": []}}\n',
            b"\n",
            b"event: step.delta\n",
            b'data: {"index": 0, "delta": {"type": "text", "text": "Hola"}}\n',
            b"\n",
            b"event: step.stop\n",
            b'data: {"index": 0}\n',
            b"\n",
            b"event: interaction.completed\n",
            b'data: {"interaction": {"id": "int_2", "status": "completed"}}\n',
            b"\n",
            b"data: [DONE]\n\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(sse_lines)
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        metrics = AistudioStreamMetrics()
        client = AistudioClient(key_provider=self._provider())
        events = list(client.stream_interaction({"model": "gemini-4-hydrogen", "input": []}, metrics=metrics))

        self.assertEqual([e["event_type"] for e in events], ["step.start", "step.delta", "step.stop", "interaction.completed"])
        self.assertEqual(metrics.event_count, 4)
        self.assertIsNotNone(metrics.ttfb_seconds)
        self.assertIsNotNone(metrics.first_delta_seconds)
        self.assertGreaterEqual(metrics.total_seconds, 0.0)

        req = mock_urlopen.call_args.args[0]
        self.assertTrue(req.full_url.endswith("/v1beta/interactions?alt=sse"))
        sent = json.loads(req.data.decode("utf-8"))
        self.assertIs(sent["stream"], True)
        self.assertIs(sent["store"], False)

    @patch("bridge.aistudio_client.urllib.request.urlopen")
    def test_http_errors_map_cleanly_and_redact_secret(self, mock_urlopen: MagicMock) -> None:
        secret = "AIzaSyDoNotLeakThisSecret123456"
        client = AistudioClient(key_provider=self._provider(secret))

        cases = [
            (_mock_http_error(400, {"error": {"message": f"API key not valid: {secret}", "details": [{"reason": "API_KEY_INVALID"}]}}), AuthenticationError),
            (_mock_http_error(400, {"error": {"message": "Unsupported field"}}), InvalidRequestError),
            (_mock_http_error(401, {"error": {"message": "Unauthorized"}}), AuthenticationError),
            (_mock_http_error(403, {"error": {"message": "Permission denied"}}), ForbiddenError),
            (_mock_http_error(404, {"error": {"message": "Model not found"}}), ModelNotFoundError),
            (_mock_http_error(429, {"error": {"message": "Quota exceeded"}}, headers={"Retry-After": "12"}), RateLimitError),
            (_mock_http_error(503, {"error": {"message": "Overloaded"}}), CapacityExhaustedError),
        ]
        for http_err, exc_cls in cases:
            mock_urlopen.side_effect = http_err
            with self.subTest(code=http_err.code, exc=exc_cls.__name__):
                with self.assertRaises(exc_cls) as ctx:
                    client.create_interaction({"model": "gemini-4-hydrogen", "input": []})
                self.assertNotIn(secret, str(ctx.exception))

    @patch("bridge.aistudio_client.urllib.request.urlopen")
    def test_timeout_raises_upstream_timeout_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = socket.timeout("read timed out")
        client = AistudioClient(key_provider=self._provider())
        with self.assertRaises(UpstreamTimeoutError):
            client.create_interaction({"model": "gemini-4-hydrogen", "input": []})

    @patch("bridge.aistudio_client.urllib.request.urlopen")
    def test_verify_api_key_classifies_verified_rejected_and_unverified(self, mock_urlopen: MagicMock) -> None:
        secret = "AIzaSyVerifyProbeSecret123456"

        ok_resp = MagicMock()
        ok_resp.read.return_value = b'{"models": []}'
        ok_resp.__enter__.return_value = ok_resp
        mock_urlopen.return_value = ok_resp
        status, _ = verify_aistudio_api_key(secret)
        self.assertEqual(status, "verified")

        mock_urlopen.side_effect = _mock_http_error(
            400,
            {"error": {"message": f"API key not valid ({secret})", "details": [{"reason": "API_KEY_INVALID"}]}},
        )
        status2, detail2 = verify_aistudio_api_key(secret)
        self.assertEqual(status2, "rejected")
        self.assertNotIn(secret, detail2)

        mock_urlopen.side_effect = urllib.error.URLError("DNS failure")
        status3, _ = verify_aistudio_api_key(secret)
        self.assertEqual(status3, "unverified")


if __name__ == "__main__":
    unittest.main()
