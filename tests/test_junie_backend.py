"""Unit tests for the Junie backend (model discovery, execution, error mapping)."""

import json
from unittest.mock import MagicMock, patch

from bridge.errors import AuthenticationError, BridgeError, ForbiddenError
from bridge.junie_backend import (
    JunieBackend,
    format_junie_model_display_name,
    is_junie_model,
    junie_acp_model_id,
    resolve_junie_model,
)
import unittest


MODELS_JSON = json.dumps({
    "models": [
        {"id": "gemini-3-flash-preview", "displayName": "Gemini 3 Flash Preview", "default": False},
        {"id": "grok-4.5", "displayName": "Grok 4.5", "default": False},
        {"id": "gemini-3.7-flash", "displayName": "Gemini 3.7 Flash", "default": True},
        {"id": "claude-sonnet-5-5", "displayName": "Claude Sonnet 5.5", "default": False},
        {"id": "junie-lite", "displayName": "Junie Lite", "default": False},
    ],
})


class TestModelIdPredicates(unittest.TestCase):
    """Junie model classification and id resolution (S2)."""

    def test_is_junie_model_accepts_prefixed_ids(self) -> None:
        self.assertTrue(is_junie_model("junie/gemini-3.7-flash"))
        self.assertTrue(is_junie_model("junie/grok-4.5"))
        self.assertFalse(is_junie_model("google/gemini-2.5-flash"))
        self.assertFalse(is_junie_model("cursor/auto"))
        self.assertFalse(is_junie_model("opencode/beam-free"))
        self.assertFalse(is_junie_model("gemini-2.5-pro"))
        self.assertFalse(is_junie_model(""))

    def test_resolve_adds_provider_prefix(self) -> None:
        self.assertEqual(resolve_junie_model("junie/gemini-3.7-flash"), "junie/gemini-3.7-flash")
        self.assertEqual(resolve_junie_model("gemini-3.7-flash"), "junie/gemini-3.7-flash")
        self.assertEqual(resolve_junie_model(""), "junie/gemini-3.7-flash")

    def test_junie_acp_model_id_strips_prefix(self) -> None:
        self.assertEqual(junie_acp_model_id("junie/grok-4.5"), "grok-4.5")
        self.assertEqual(junie_acp_model_id("junie/gemini-3.7-flash"), "gemini-3.7-flash")


class TestDisplayNames(unittest.TestCase):
    """Source tag parity with Cursor/OpenCode display names (S2)."""

    def test_display_name_has_source_tag(self) -> None:
        self.assertEqual(
            format_junie_model_display_name("junie/gemini-3.7-flash"),
            "Gemini 3.7 Flash [Junie]",
        )

    def test_display_name_other_models(self) -> None:
        self.assertEqual(
            format_junie_model_display_name("junie/grok-4.5"),
            "Grok 4.5 [Junie]",
        )
        self.assertEqual(
            format_junie_model_display_name("junie/claude-sonnet-5-5"),
            "Claude Sonnet 5 5 [Junie]",
        )

    def test_known_display_name_overrides_generation(self) -> None:
        # A curated name wins over the generated one but keeps the [Junie] tag.
        with patch("bridge.junie_backend.JUNIE_KNOWN_DISPLAY_NAMES", {"junie-lite": "Junie Lite"}):
            self.assertEqual(
                format_junie_model_display_name("junie/junie-lite"),
                "Junie Lite [Junie]",
            )


class TestModelDiscovery(unittest.TestCase):
    """`junie list-models` JSON parsing, caching, fallback (S1, S2)."""

    def _backend(self, stdout: str = MODELS_JSON, returncode: int = 0) -> JunieBackend:
        backend = JunieBackend(binary_path="/usr/bin/junie")
        backend._acp_client = MagicMock()  # never spawn real processes here
        run = MagicMock()
        run.return_value = MagicMock(returncode=returncode, stdout=stdout, stderr="")
        self._run = run
        return backend

    def test_list_models_parses_json_and_tags_source(self) -> None:
        backend = self._backend()
        with patch("bridge.junie_backend.subprocess.run", self._run):
            models = backend.list_models()
        ids = [m["id"] for m in models]
        self.assertEqual(
            ids,
            [
                "junie/gemini-3-flash-preview",
                "junie/grok-4.5",
                "junie/gemini-3.7-flash",
                "junie/claude-sonnet-5-5",
                "junie/junie-lite",
            ],
        )
        for m in models:
            self.assertEqual(m["owned_by"], "junie")
            self.assertEqual(m["slug"], m["id"])
            self.assertTrue(m["display_name"].endswith("[Junie]"))

    def test_list_models_falls_back_to_static_list(self) -> None:
        backend = self._backend(stdout="", returncode=1)
        with patch("bridge.junie_backend.subprocess.run", self._run):
            models = backend.list_models()
        ids = [m["id"] for m in models]
        self.assertIn("junie/gemini-3.7-flash", ids)
        self.assertIn("junie/grok-4.5", ids)
        self.assertIn("junie/junie-lite", ids)
        # Fallback never advertises other-provider prefixes.
        self.assertFalse(any(i.startswith(("google/", "openai/", "cursor/", "opencode/")) for i in ids))

    def test_list_models_uses_cache_on_second_call(self) -> None:
        backend = self._backend()
        with patch("bridge.junie_backend.subprocess.run", self._run):
            backend.list_models()
            backend.list_models()
        self.assertEqual(self._run.call_count, 1)

    def test_list_models_force_bypasses_cache(self) -> None:
        backend = self._backend()
        with patch("bridge.junie_backend.subprocess.run", self._run):
            backend.list_models()
            backend.list_models(force=True)
        self.assertEqual(self._run.call_count, 2)

    def test_discovery_timeout_never_breaks_catalog(self) -> None:
        import subprocess as sp
        backend = self._backend()
        with patch("bridge.junie_backend.subprocess.run", side_effect=sp.TimeoutExpired("junie", 5)):
            models = backend.list_models()
        self.assertTrue(any(m["id"] == "junie/gemini-3.7-flash" for m in models))


class TestBackendExecution(unittest.TestCase):
    """ACP-first execution with CLI fallback and readable error mapping (S2, S5)."""

    def test_stream_chat_prefers_acp_with_resolved_model(self) -> None:
        acp = MagicMock()
        acp.is_alive.return_value = True
        acp.stream_prompt.return_value = iter(["PO", "NG"])
        backend = JunieBackend(binary_path="/usr/bin/junie", acp_client=acp)

        out = "".join(backend.stream_chat("Say PONG", model="gemini-3.7-flash"))
        self.assertEqual(out, "PONG")
        kwargs = acp.stream_prompt.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.7-flash")

    def test_stream_chat_maps_acp_region_error_to_forbidden(self) -> None:
        acp = MagicMock()
        acp.is_alive.return_value = True

        def boom(*a, **k):
            raise ForbiddenError("Junie model 'junie/grok-4.6' is not available in your region (upstream: ...)")
            yield  # pragma: no cover

        acp.stream_prompt.side_effect = boom
        backend = JunieBackend(binary_path="/usr/bin/junie", acp_client=acp)
        with self.assertRaises(ForbiddenError) as ctx:
            list(backend.stream_chat("hi", model="junie/grok-4.6", timeout=5.0))
        self.assertIn("grok-4.6", str(ctx.exception))

    def test_stream_chat_falls_back_to_cli_when_acp_fails(self) -> None:
        acp = MagicMock()
        acp.is_alive.return_value = True

        def boom(*a, **k):
            raise BridgeError("ACP channel died")
            yield  # pragma: no cover

        acp.stream_prompt.side_effect = boom
        backend = JunieBackend(binary_path="/usr/bin/junie", acp_client=acp)

        proc = MagicMock()
        proc.stdout = iter([json.dumps({"result": "CLI-OK"}) + "\n"])
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = b""
        with patch("bridge.junie_backend.subprocess.Popen", return_value=proc) as popen:
            out = "".join(backend.stream_chat("hi", model="gemini-3.7-flash", timeout=5.0))
        self.assertEqual(out, "CLI-OK")
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:2], ["/usr/bin/junie", "--task"])
        self.assertIn("--model", cmd)
        self.assertIn("--output-format", cmd)
        idx = cmd.index("--model")
        self.assertEqual(cmd[idx + 1], "gemini-3.7-flash")

    def test_cli_region_error_maps_to_forbidden_with_model_name(self) -> None:
        backend = JunieBackend(binary_path="/usr/bin/junie", use_acp=False)
        proc = MagicMock()
        proc.stdout = iter([])
        proc.poll.return_value = 1
        proc.returncode = 1
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = (
            "Junie failed with the message: Model 'junie/grok-4.6' is not available in your country"
        )
        with patch("bridge.junie_backend.subprocess.Popen", return_value=proc):
            with self.assertRaises(ForbiddenError) as ctx:
                list(backend.stream_chat("hi", model="junie/grok-4.6", timeout=5.0))
        self.assertIn("grok-4.6", str(ctx.exception))
        self.assertIn("region", str(ctx.exception).lower())

    def test_cli_auth_error_maps_to_authentication_error(self) -> None:
        backend = JunieBackend(binary_path="/usr/bin/junie", use_acp=False)
        proc = MagicMock()
        proc.stdout = iter([])
        proc.poll.return_value = 1
        proc.returncode = 1
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = "Authentication required: provider authentication required"
        with patch("bridge.junie_backend.subprocess.Popen", return_value=proc):
            with self.assertRaises(AuthenticationError):
                list(backend.stream_chat("hi", model="gemini-3.7-flash", timeout=5.0))

    def test_cli_binary_missing_raises_bridge_error(self) -> None:
        backend = JunieBackend(binary_path="/no/such/junie", use_acp=False)
        with patch("bridge.junie_backend.subprocess.Popen", side_effect=FileNotFoundError()):
            with self.assertRaises(BridgeError) as ctx:
                list(backend.stream_chat("hi", model="gemini-3.7-flash", timeout=5.0))
        self.assertIn("junie", str(ctx.exception).lower())

    def test_generate_chat_collects_stream(self) -> None:
        acp = MagicMock()
        acp.is_alive.return_value = True
        acp.stream_prompt.return_value = iter(["a", "b", "c"])
        backend = JunieBackend(binary_path="/usr/bin/junie", acp_client=acp)
        self.assertEqual(backend.generate_chat("hi", model="gemini-3.7-flash", timeout=5.0), "abc")


class TestBackendWarmup(unittest.TestCase):
    """Warmup must be non-blocking and safe when ACP is disabled."""

    def test_warmup_disabled_is_noop(self) -> None:
        backend = JunieBackend(binary_path="/usr/bin/junie", use_acp=False)
        backend.warmup()  # must not raise nor spawn
        self.assertIsNone(backend._acp_client)


class TestBackendStatus(unittest.TestCase):
    """Availability status for the dashboard Junie tab (S3)."""

    def test_status_reports_availability_and_model_count(self) -> None:
        backend = JunieBackend(binary_path="/usr/bin/junie")
        backend._acp_client = MagicMock()
        run = MagicMock(return_value=MagicMock(returncode=0, stdout=MODELS_JSON, stderr=""))
        with patch("bridge.junie_backend.subprocess.run", run), \
             patch("bridge.junie_backend.shutil.which", return_value="/usr/bin/junie"), \
             patch("bridge.junie_backend.subprocess.check_output", return_value="Junie 26.11.1 (4020.7)\n"):
            st = backend.status()
        self.assertTrue(st["available"])
        self.assertEqual(st["version"], "26.11.1 (4020.7)")
        self.assertEqual(st["model_count"], 5)
        self.assertTrue(st["binary_path"].endswith("junie"))
        # Junie has no upstream quota API: the dashboard must not fake bars.
        self.assertFalse(st["quota_supported"])

    def test_status_when_binary_missing(self) -> None:
        backend = JunieBackend(binary_path="/no/such/junie")
        with patch("bridge.junie_backend.shutil.which", return_value=None), \
             patch("bridge.junie_backend.Path.is_file", return_value=False):
            st = backend.status()
        self.assertFalse(st["available"])
        self.assertIsNone(st["version"])
        self.assertEqual(st["model_count"], 0)
        self.assertFalse(st["quota_supported"])

    def test_status_version_failure_stays_available(self) -> None:
        import subprocess as sp
        backend = JunieBackend(binary_path="/usr/bin/junie")
        backend._acp_client = MagicMock()
        with patch("bridge.junie_backend.shutil.which", return_value="/usr/bin/junie"), \
             patch("bridge.junie_backend.subprocess.check_output", side_effect=sp.TimeoutExpired("junie", 2)), \
             patch("bridge.junie_backend.subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
            st = backend.status()
        self.assertTrue(st["available"])
        self.assertIsNone(st["version"])


if __name__ == "__main__":
    unittest.main()