"""Unit tests for Google AI Studio Keychain credential management & CLI."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from bridge.aistudio_auth import (
    AISTUDIO_KEYCHAIN_ACCOUNT,
    AISTUDIO_KEYCHAIN_SERVICE,
    AistudioKeychainProvider,
    check_aistudio_auth_health,
    redact_secret,
)
from bridge.cli.auth import handle_auth
from bridge.errors import AuthenticationError, InteractionStateError


class TestInteractionStateError(unittest.TestCase):
    """Verifies HTTP 409 error representation for missing/ambiguous continuity state."""

    def test_defaults_and_custom_code(self) -> None:
        err = InteractionStateError()
        self.assertEqual(err.code, "interaction_state_missing")
        self.assertIn("missing or expired", str(err))

        amb = InteractionStateError("Ambiguous conversation prefix", code="interaction_state_ambiguous")
        self.assertEqual(amb.code, "interaction_state_ambiguous")


class TestRedactSecret(unittest.TestCase):
    """Defensive redaction must mask both explicit secrets and Google API key shapes."""

    def test_redacts_explicit_secret(self) -> None:
        secret = "AIzaSySecretKey1234567890"
        text = f"HTTP error sending x-goog-api-key={secret} to endpoint"
        redacted = redact_secret(text, secret)
        self.assertNotIn(secret, redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_redacts_google_api_key_pattern_even_without_explicit_secret(self) -> None:
        text = "Upstream rejected key AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz123456"
        redacted = redact_secret(text)
        self.assertNotIn("AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz123456", redacted)
        self.assertIn("[REDACTED]", redacted)


class TestAistudioKeychainProvider(unittest.TestCase):
    """Isolated macOS Keychain lifecycle: never touches Antigravity OAuth or local api_key."""

    def test_default_service_and_account_are_isolated(self) -> None:
        prov = AistudioKeychainProvider()
        self.assertEqual(prov.service, "abridge-aistudio")
        self.assertEqual(prov.account, "api-key")
        self.assertNotEqual((prov.service, prov.account), ("gemini", "antigravity"))

    def test_rejects_unsafe_service_or_account_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            AistudioKeychainProvider(service="bad service\nname")
        with self.assertRaises(ValueError):
            AistudioKeychainProvider(account="bad;account")

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_has_api_key_checks_metadata_without_reading_secret(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        prov = AistudioKeychainProvider()
        self.assertTrue(prov.has_api_key())
        cmd = mock_run.call_args.args[0]
        self.assertIn("find-generic-password", cmd)
        self.assertIn(AISTUDIO_KEYCHAIN_SERVICE, cmd)
        self.assertIn(AISTUDIO_KEYCHAIN_ACCOUNT, cmd)
        self.assertNotIn("-w", cmd)

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_has_api_key_returns_false_when_missing(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=44, stdout="", stderr="SecKeychainSearchCopyNext: item not found")
        prov = AistudioKeychainProvider()
        self.assertFalse(prov.has_api_key())

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_get_api_key_caches_in_memory_and_invalidates(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="AIzaTestKey1234567890\n", stderr="")
        prov = AistudioKeychainProvider()
        self.assertEqual(prov.get_api_key(), "AIzaTestKey1234567890")
        self.assertEqual(prov.get_api_key(), "AIzaTestKey1234567890")
        self.assertEqual(mock_run.call_count, 1)

        prov.invalidate()
        self.assertEqual(prov.get_api_key(), "AIzaTestKey1234567890")
        self.assertEqual(mock_run.call_count, 2)

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_get_api_key_raises_authentication_error_when_missing(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=44, stdout="", stderr="The specified item could not be found")
        prov = AistudioKeychainProvider()
        with self.assertRaises(AuthenticationError) as ctx:
            prov.get_api_key()
        self.assertIn("abridge auth aistudio", str(ctx.exception))

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_get_api_key_timeout_raises_authentication_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired("security", 10)
        prov = AistudioKeychainProvider()
        with self.assertRaises(AuthenticationError) as ctx:
            prov.get_api_key()
        self.assertIn("Timed out", str(ctx.exception))

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_store_api_key_never_exposes_secret_in_argv(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        secret = "AIzaSySuperSecretValue987654321"
        prov = AistudioKeychainProvider()
        prov.store_api_key(secret)

        cmd = mock_run.call_args.args[0]
        kwargs = mock_run.call_args.kwargs
        # Secret must NEVER appear in command arguments (argv / ps)
        self.assertEqual(cmd[1:], ["-i"])
        for arg in cmd:
            self.assertNotIn(secret, arg)
        # Written over private stdin pipe with -U and hex encoding (-X)
        stdin_payload = kwargs.get("input", "")
        self.assertIn("add-generic-password", stdin_payload)
        self.assertIn("-U", stdin_payload)
        self.assertIn("-X ", stdin_payload)
        self.assertNotIn(secret, stdin_payload)  # hex encoded, not plaintext
        self.assertIn(secret.encode("utf-8").hex(), stdin_payload)
        self.assertFalse(kwargs.get("shell", False))

    def test_store_api_key_rejects_empty_secret(self) -> None:
        prov = AistudioKeychainProvider()
        with self.assertRaises(ValueError):
            prov.store_api_key("   \n\t ")

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_store_api_key_redacts_secret_on_error(self, mock_run: MagicMock) -> None:
        secret = "AIzaSyLeakedInStderr1234567890"
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr=f"Error with {secret}")
        prov = AistudioKeychainProvider()
        with self.assertRaises(AuthenticationError) as ctx:
            prov.store_api_key(secret)
        self.assertNotIn(secret, str(ctx.exception))

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_delete_api_key_and_missing_idempotent(self, mock_run: MagicMock) -> None:
        prov = AistudioKeychainProvider()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        self.assertTrue(prov.delete_api_key())
        cmd = mock_run.call_args.args[0]
        self.assertIn("delete-generic-password", cmd)
        self.assertIn("abridge-aistudio", cmd)

        mock_run.return_value = MagicMock(returncode=44, stdout="", stderr="not found")
        self.assertFalse(prov.delete_api_key())

    @patch("bridge.aistudio_auth.subprocess.run")
    def test_check_auth_health_reports_non_secret_states(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=44, stdout="", stderr="not found")
        prov = AistudioKeychainProvider()
        health = check_aistudio_auth_health(provider=prov)
        self.assertEqual(health["status"], "not_configured")
        self.assertFalse(health["configured"])

        mock_run.return_value = MagicMock(returncode=0, stdout="AIzaSySecret123456789012345", stderr="")
        health2 = check_aistudio_auth_health(provider=prov)
        self.assertEqual(health2["status"], "configured_unverified")
        self.assertTrue(health2["configured"])
        self.assertNotIn("AIza", str(health2))

        health3 = check_aistudio_auth_health(provider=prov, verifier=lambda k: ("verified", "ok"))
        self.assertEqual(health3["status"], "verified")
        self.assertNotIn("AIza", str(health3))


class TestCliAuthAistudio(unittest.TestCase):
    """CLI handler `abridge auth aistudio` (set, status, remove)."""

    def _mock_provider(self) -> MagicMock:
        prov = MagicMock(spec=AistudioKeychainProvider)
        prov.service = AISTUDIO_KEYCHAIN_SERVICE
        prov.account = AISTUDIO_KEYCHAIN_ACCOUNT
        return prov

    def _run_cli(self, argv: list[str], provider=None, verifier=None, stdin_text: str = "") -> tuple[int, str]:
        buf = StringIO()
        with patch("sys.stdin", StringIO(stdin_text)):
            with redirect_stdout(buf), redirect_stderr(buf):
                try:
                    code = handle_auth(
                        argv,
                        "abridge",
                        provider=provider,
                        verifier=verifier,
                    )
                except SystemExit as exc:
                    code = int(exc.code or 0)
        return code, buf.getvalue()

    def test_cli_auth_rejects_inline_api_key_flag(self) -> None:
        prov = self._mock_provider()
        code, out = self._run_cli(["aistudio", "--api-key", "AIza123"], provider=prov)
        self.assertNotEqual(code, 0)
        prov.store_api_key.assert_not_called()

    def test_cli_auth_set_interactive_hidden_prompt_and_verify(self) -> None:
        prov = self._mock_provider()
        secret = "AIzaSyInteractiveSecret123456789"
        with patch("bridge.cli.auth.getpass.getpass", return_value=secret):
            code, out = self._run_cli(
                ["aistudio"],
                provider=prov,
                verifier=lambda k: ("verified", "Gemini 4 Hydrogen reachable"),
            )
        self.assertEqual(code, 0)
        prov.store_api_key.assert_called_once_with(secret)
        self.assertNotIn(secret, out)
        self.assertIn("verified", out.lower())

    def test_cli_auth_set_via_stdin_without_echo(self) -> None:
        prov = self._mock_provider()
        secret = "AIzaSyPipedSecret1234567890"
        code, out = self._run_cli(
            ["aistudio", "set", "--stdin", "--skip-verify"],
            provider=prov,
            stdin_text=f"{secret}\n",
        )
        self.assertEqual(code, 0)
        prov.store_api_key.assert_called_once_with(secret)
        self.assertNotIn(secret, out)

    def test_cli_auth_status_reports_non_secret_state(self) -> None:
        prov = self._mock_provider()
        prov.has_api_key.return_value = True
        prov.get_api_key.return_value = "AIzaSySecretForStatusCheck12345"
        code, out = self._run_cli(["aistudio", "status"], provider=prov)
        self.assertEqual(code, 0)
        self.assertNotIn("AIza", out)
        self.assertIn("abridge-aistudio", out)

    def test_cli_auth_remove_deletes_only_aistudio_keychain_item(self) -> None:
        prov = self._mock_provider()
        prov.delete_api_key.return_value = True
        code, out = self._run_cli(["aistudio", "remove"], provider=prov)
        self.assertEqual(code, 0)
        prov.delete_api_key.assert_called_once()


if __name__ == "__main__":
    unittest.main()
