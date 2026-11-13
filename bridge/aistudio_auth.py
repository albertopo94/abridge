"""Google AI Studio Gemini API key management via macOS Keychain.

Zero external dependencies: pure standard library.
Stores the Gemini API key in a completely separate Keychain item
(`service="abridge-aistudio"`, `account="api-key"`), isolated from
Antigravity OAuth (`service="gemini"`, `account="antigravity"`) and the local
gateway API key (`~/.abridge/api_key`).

Secrets are never passed via command-line arguments (`argv`), shell commands,
configuration files, logs, or error messages. Writes use `/usr/bin/security -i`
over a private stdin pipe with hex encoding (`-X`).
"""

from __future__ import annotations

from collections.abc import Callable
import re
import subprocess
import threading
import time
from typing import Any

from bridge.auth import SECURITY_PATH
from bridge.errors import AuthenticationError

AISTUDIO_KEYCHAIN_SERVICE = "abridge-aistudio"
AISTUDIO_KEYCHAIN_ACCOUNT = "api-key"

_SAFE_KEYCHAIN_ATTR_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_GOOGLE_API_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{16,}")


def redact_secret(text: str, secret: str | None = None) -> str:
    """Masks any occurrence of `secret` or Google API key shapes from text."""
    if not text:
        return ""
    cleaned = str(text)
    if secret:
        stripped = secret.strip()
        if stripped:
            cleaned = cleaned.replace(stripped, "[REDACTED]")
    cleaned = _GOOGLE_API_KEY_RE.sub("[REDACTED]", cleaned)
    return cleaned


class AistudioKeychainProvider:
    """Manages the AI Studio Gemini API key in macOS Keychain with in-memory TTL cache."""

    def __init__(
        self,
        service: str = AISTUDIO_KEYCHAIN_SERVICE,
        account: str = AISTUDIO_KEYCHAIN_ACCOUNT,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        if not _SAFE_KEYCHAIN_ATTR_RE.match(service):
            raise ValueError(f"Invalid Keychain service identifier: {service!r}")
        if not _SAFE_KEYCHAIN_ATTR_RE.match(account):
            raise ValueError(f"Invalid Keychain account identifier: {account!r}")
        self.service = service
        self.account = account
        self.cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.Lock()
        self._cached_key: str | None = None
        self._cached_at: float = 0.0

    def invalidate(self) -> None:
        """Clears the in-memory cached API key."""
        with self._lock:
            self._cached_key = None
            self._cached_at = 0.0

    def has_api_key(self) -> bool:
        """Checks Keychain item existence via metadata only (never reads secret)."""
        cmd = [
            SECURITY_PATH,
            "find-generic-password",
            "-s",
            self.service,
            "-a",
            self.account,
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10.0,
            )
            return res.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def get_api_key(self, force_refresh: bool = False) -> str:
        """Retrieves the Gemini API key from in-memory cache or macOS Keychain."""
        with self._lock:
            now = time.time()
            if (
                not force_refresh
                and self._cached_key is not None
                and (now - self._cached_at) < self.cache_ttl_seconds
            ):
                return self._cached_key

            raw_key = self._read_keychain()
            clean_key = raw_key.strip()
            if not clean_key:
                raise AuthenticationError(
                    f"AI Studio Keychain item (service={self.service}, account={self.account}) "
                    "is empty. Run `abridge auth aistudio` to configure a valid Gemini API key."
                )
            self._cached_key = clean_key
            self._cached_at = now
            return clean_key

    def store_api_key(self, api_key: str) -> None:
        """Creates or updates the AI Studio Keychain item without exposing the secret in argv."""
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Gemini API key cannot be empty.")

        clean_key = api_key.strip()
        hex_secret = clean_key.encode("utf-8").hex()
        stdin_cmd = (
            f"add-generic-password -s {self.service} -a {self.account} -U -X {hex_secret}\n"
        )
        try:
            res = subprocess.run(
                [SECURITY_PATH, "-i"],
                input=stdin_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise AuthenticationError(
                f"Timed out saving AI Studio API key to Keychain "
                f"(service={self.service}, account={self.account})."
            ) from exc
        except OSError as exc:
            raise AuthenticationError(
                f"Failed to execute Keychain security CLI ({SECURITY_PATH}): {redact_secret(str(exc), clean_key)}"
            ) from exc

        if res.returncode != 0:
            detail = redact_secret(res.stderr.strip() or res.stdout.strip(), clean_key)
            raise AuthenticationError(
                f"Failed to store AI Studio API key in Keychain "
                f"(service={self.service}, account={self.account}, exit={res.returncode}): {detail}"
            )

        with self._lock:
            self._cached_key = clean_key
            self._cached_at = time.time()

    def delete_api_key(self) -> bool:
        """Deletes the AI Studio Keychain item. Returns False if it was already absent."""
        self.invalidate()
        cmd = [
            SECURITY_PATH,
            "delete-generic-password",
            "-s",
            self.service,
            "-a",
            self.account,
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise AuthenticationError(
                f"Timed out removing AI Studio API key from Keychain "
                f"(service={self.service}, account={self.account})."
            ) from exc
        except OSError as exc:
            raise AuthenticationError(
                f"Failed to execute Keychain security CLI ({SECURITY_PATH}): {exc}"
            ) from exc

        if res.returncode == 0:
            return True
        if res.returncode == 44:  # errSecItemNotFound
            return False

        detail = redact_secret(res.stderr.strip() or res.stdout.strip())
        raise AuthenticationError(
            f"Failed to delete AI Studio API key from Keychain "
            f"(service={self.service}, account={self.account}, exit={res.returncode}): {detail}"
        )

    def _read_keychain(self) -> str:
        cmd = [
            SECURITY_PATH,
            "find-generic-password",
            "-s",
            self.service,
            "-a",
            self.account,
            "-w",
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise AuthenticationError(
                f"Timed out reading AI Studio API key from Keychain "
                f"(service={self.service}, account={self.account})."
            ) from exc
        except OSError as exc:
            raise AuthenticationError(
                f"Failed to execute Keychain security CLI ({SECURITY_PATH}): {exc}"
            ) from exc

        if res.returncode != 0:
            err_msg = redact_secret(res.stderr.strip())
            raise AuthenticationError(
                f"AI Studio API key not found in Keychain (service={self.service}, "
                f"account={self.account}). Run `abridge auth aistudio` to configure it. "
                f"Details: {err_msg}"
            )
        return res.stdout


def check_aistudio_auth_health(
    provider: AistudioKeychainProvider | None = None,
    verifier: Callable[[str], tuple[str, str]] | None = None,
    verify_live: bool = False,
) -> dict[str, Any]:
    """Reports non-secret AI Studio credential health for CLI and dashboard.

    Possible `status` values:
      - `not_configured`: no Keychain item found
      - `configured_unverified`: Keychain item present, no live probe run
      - `verified`: live probe confirmed Google accepted the key
      - `rejected`: live probe rejected the key (401/403)
      - `unverified`: live probe failed due to connectivity/5xx/timeout
    """
    prov = provider if provider is not None else AistudioKeychainProvider()
    if not prov.has_api_key():
        return {
            "configured": False,
            "status": "not_configured",
            "service": prov.service,
            "account": prov.account,
            "detail": "AI Studio API key is not configured. Run `abridge auth aistudio`.",
        }

    if verifier is None and not verify_live:
        return {
            "configured": True,
            "status": "configured_unverified",
            "service": prov.service,
            "account": prov.account,
            "detail": "API key stored in macOS Keychain (not live-verified).",
        }

    try:
        secret = prov.get_api_key()
    except AuthenticationError as exc:
        return {
            "configured": False,
            "status": "not_configured",
            "service": prov.service,
            "account": prov.account,
            "detail": redact_secret(str(exc)),
        }

    probe_fn = verifier
    if probe_fn is None:
        probe_fn = _default_live_verifier

    try:
        status, detail = probe_fn(secret)
        return {
            "configured": True,
            "status": status,
            "service": prov.service,
            "account": prov.account,
            "detail": redact_secret(detail, secret),
        }
    except Exception as exc:
        return {
            "configured": True,
            "status": "unverified",
            "service": prov.service,
            "account": prov.account,
            "detail": redact_secret(str(exc), secret),
        }


def _default_live_verifier(api_key: str) -> tuple[str, str]:
    """Performs a minimal live probe against AI Studio when `bridge.aistudio_client` is available."""
    try:
        from bridge.aistudio_client import verify_aistudio_api_key
    except ImportError:
        return ("configured_unverified", "Live client verification module not loaded.")
    return verify_aistudio_api_key(api_key)
