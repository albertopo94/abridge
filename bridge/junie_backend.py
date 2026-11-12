"""Junie CLI backend: model discovery and execution adapter.

Zero external dependencies: pure standard library.

Exposes Junie CLI models (`junie/<id>`) as a fourth upstream next to
Antigravity, Cursor, and OpenCode. Display names carry a trailing `[Junie]`
source tag for parity with Cursor's `[Cursor - Free]` / `[Cursor - Pro]`
tags and OpenCode's `[OpenCode]` tag.

Model discovery reads the JSON catalog from `junie list-models`
(`{"models": [{"id", "displayName", ...}]}`), cached 300s, with a static
fallback so the catalog never advertises stale or removed model ids.

Execution prefers a warm `junie --acp true` process and falls back to
`junie --task <prompt> --model <id>` headless subprocesses, mirroring the
OpenCode ACP-then-CLI strategy.
"""

from __future__ import annotations

from bridge.errors import AuthenticationError, BridgeError, ForbiddenError, UpstreamTimeoutError
from bridge.junie_acp import (
    JunieAcpClient,
    extract_junie_error_message,
    find_junie_binary,
    is_junie_auth_error,
    is_junie_region_error,
)
from collections.abc import Iterator
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any

JUNIE_PROVIDER_PREFIX = "junie/"
DEFAULT_JUNIE_MODEL = "junie/junie-lite"

# Static fallback used when `junie list-models` is unavailable or slow, so the
# catalog still advertises the models Junie is known to serve.
JUNIE_KNOWN_MODELS: set[str] = {
    "junie/gemini-3-flash-preview",
    "junie/gemini-3.1-flash-lite",
    "junie/gemini-3.7-flash",
    "junie/gemini-3.8-flash",
    "junie/gemini-4-argon",
    "junie/gemini-4-carbon",
    "junie/gemini-3.5-flash-lite",
    "junie/gemini-3.6-flash",
    "junie/gemini-3.1-pro-preview",
    "junie/grok-4.3",
    "junie/grok-4.5",
    "junie/grok-4.6",
    "junie/grok-4.7",
    "junie/claude-sonnet-5",
    "junie/claude-sonnet-5-5",
    "junie/claude-haiku-5-5",
    "junie/claude-opus-4-6",
    "junie/claude-opus-4-7",
    "junie/claude-opus-4-8",
    "junie/claude-opus-5",
    "junie/claude-opus-5-5",
    "junie/junie-lite",
}

JUNIE_KNOWN_DISPLAY_NAMES: dict[str, str] = {
    "junie-lite": "Junie Lite",
    "gpt-5-2025-08-07": "GPT-5",
}

_ACRONYMS = {"vl", "glm", "ai", "api", "3p"}


def is_junie_model(model: str) -> bool:
    """Checks whether the requested model belongs to the Junie upstream."""
    if not model or not isinstance(model, str):
        return False
    return model.strip().lower().startswith(JUNIE_PROVIDER_PREFIX)


def resolve_junie_model(model: str) -> str:
    """Normalizes a model name to the `junie/<id>` form for gateway routing."""
    if not model or not isinstance(model, str):
        return DEFAULT_JUNIE_MODEL
    m = model.strip()
    if not m:
        return DEFAULT_JUNIE_MODEL
    if m.lower().startswith(JUNIE_PROVIDER_PREFIX):
        return m
    if "/" in m:
        return m
    return f"{JUNIE_PROVIDER_PREFIX}{m}"


def junie_acp_model_id(model: str) -> str:
    """Strips the `junie/` prefix to the raw model id Junie's ACP expects."""
    m = resolve_junie_model(model).strip()
    if m.lower().startswith(JUNIE_PROVIDER_PREFIX):
        return m[len(JUNIE_PROVIDER_PREFIX):]
    return m


def _capitalize_segment(segment: str) -> str:
    low = segment.lower()
    if low in _ACRONYMS:
        return segment.upper()
    if low.startswith("deepseek"):
        return "DeepSeek" + segment[len("deepseek"):]
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", segment):
        return segment
    if segment[0].isupper():
        return segment
    return segment.capitalize()


def format_junie_model_display_name(model_id: str) -> str:
    """Formats a user-facing display name with the trailing [Junie] source tag."""
    clean = resolve_junie_model(model_id)
    if clean.lower().startswith(JUNIE_PROVIDER_PREFIX):
        clean = clean[len(JUNIE_PROVIDER_PREFIX):]
    clean = clean.strip()
    if not clean:
        return "Junie [Junie]"

    base = JUNIE_KNOWN_DISPLAY_NAMES.get(clean)
    if not base:
        base = " ".join(_capitalize_segment(p) for p in re.split(r"[-_]+", clean) if p)
    return f"{base} [Junie]"


def _map_junie_error(model: str, message: str) -> BridgeError:
    """Maps a raw upstream error message to the bridge exception hierarchy."""
    if is_junie_region_error(message):
        return ForbiddenError(
            f"Junie model '{model}' is not available in your region (upstream: {message})"
        )
    if is_junie_auth_error(message):
        return AuthenticationError(
            f"Junie model '{model}' requires provider authentication (upstream: {message})"
        )
    return BridgeError(f"Junie upstream error for model '{model}': {message or 'unknown error'}")


class JunieBackend:
    """Backend adapter that delegates inference to the local Junie CLI."""

    def __init__(
        self,
        binary_path: str | None = None,
        acp_client: Any | None = None,
        use_acp: bool | None = None,
    ) -> None:
        self.binary_path = binary_path or find_junie_binary()
        if use_acp is None:
            self.use_acp = os.environ.get("JUNIE_DISABLE_ACP", "").strip() not in ("1", "true")
        else:
            self.use_acp = use_acp
        self._acp_client = acp_client
        self._acp_client_lock = threading.Lock()
        self._cached_models: list[dict[str, Any]] | None = None
        self._cached_models_time: float = 0.0
        self._models_lock = threading.Lock()

    def _get_acp_client(self) -> Any:
        if self._acp_client is not None:
            return self._acp_client
        with self._acp_client_lock:
            if self._acp_client is None:
                client = JunieAcpClient(binary_path=self.binary_path)
                client.start()
                self._acp_client = client
            return self._acp_client

    def warmup(self, async_mode: bool = True) -> None:
        """Pre-warms the background Junie ACP process without blocking callers."""
        if not self.use_acp:
            return

        def _do_warmup() -> None:
            try:
                acp = self._get_acp_client()
                if not acp.is_alive():
                    acp.start()
            except Exception as exc:
                import sys
                sys.stderr.write(f"Warning: Junie ACP warmup failed: {exc}\n")
                sys.stderr.flush()

        if async_mode:
            threading.Thread(target=_do_warmup, daemon=True, name="junie-acp-warmup").start()
        else:
            _do_warmup()

    def close(self) -> None:
        """Closes any active ACP background client."""
        with self._acp_client_lock:
            if self._acp_client is not None and hasattr(self._acp_client, "close"):
                self._acp_client.close()
                self._acp_client = None

    def is_available(self) -> bool:
        """Returns True when the Junie binary can be located."""
        return shutil.which(self.binary_path) is not None or Path(self.binary_path).is_file()

    def status(self) -> dict[str, Any]:
        """Reports availability for the dashboard Junie tab.

        Junie exposes no provider-side quota API through the CLI, so
        `quota_supported` is always False and the dashboard must not render
        fake limit bars (parity with the OpenCode tab).
        """
        available = self.is_available()
        version: str | None = None
        if available:
            with contextlib.suppress(Exception):
                raw = subprocess.check_output(  # noqa: S603
                    [self.binary_path, "--version"],
                    text=True,
                    timeout=2.0,
                    stderr=subprocess.DEVNULL,
                )
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                # `junie --version` prints a banner line then "Junie <ver> (build)".
                for line in reversed(lines):
                    if line.startswith("Junie "):
                        version = line[len("Junie "):].strip()
                        break
                if version is None and lines:
                    version = lines[-1] or None

        model_count = 0
        if available:
            with contextlib.suppress(Exception):
                model_count = len(self.list_models())

        return {
            "available": available,
            "binary_path": self.binary_path if available else None,
            "version": version,
            "model_count": model_count,
            "quota_supported": False,
        }

    def list_models(self, force: bool = False) -> list[dict[str, Any]]:
        """Discovers Junie models from `junie list-models` (JSON), cached, with a static fallback."""
        now = time.time()
        with self._models_lock:
            if not force and self._cached_models and (now - self._cached_models_time) < 300.0:
                return self._cached_models

        discovered: list[str] = []
        with contextlib.suppress(Exception):
            res = subprocess.run(
                [self.binary_path, "list-models"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            if res.returncode == 0:
                data = json.loads(res.stdout)
                if isinstance(data, dict):
                    for entry in data.get("models", []):
                        if not isinstance(entry, dict):
                            continue
                        model_id = entry.get("id")
                        if isinstance(model_id, str) and model_id.strip():
                            discovered.append(f"{JUNIE_PROVIDER_PREFIX}{model_id.strip()}")

        if not discovered:
            discovered = sorted(JUNIE_KNOWN_MODELS)

        result = [
            {
                "id": m,
                "slug": m,
                "display_name": format_junie_model_display_name(m),
                "owned_by": "junie",
            }
            for m in dict.fromkeys(discovered)
        ]
        with self._models_lock:
            self._cached_models = result
            self._cached_models_time = now
        return result

    def stream_chat(
        self,
        prompt: str,
        model: str = DEFAULT_JUNIE_MODEL,
        timeout: float = 300.0,
    ) -> Iterator[str]:
        """Streams text deltas, preferring the warm ACP process with a CLI fallback."""
        resolved = resolve_junie_model(model)
        raw_id = junie_acp_model_id(model)

        if self.use_acp:
            try:
                acp = self._get_acp_client()
                yield from acp.stream_prompt(prompt, model=raw_id, timeout=timeout)
                return
            except (ForbiddenError, AuthenticationError):
                # Availability verdicts are upstream facts; the CLI would only
                # repeat them after paying another cold start.
                raise
            except Exception as acp_err:
                import sys
                sys.stderr.write(
                    f"Warning: Junie ACP streaming failed, falling back to junie --task: {acp_err}\n"
                )
                sys.stderr.flush()

        yield from self._stream_chat_cli(prompt=prompt, model=raw_id, timeout=timeout)

    def generate_chat(
        self,
        prompt: str,
        model: str = DEFAULT_JUNIE_MODEL,
        timeout: float = 300.0,
    ) -> str:
        """Collects streaming chunks into a single complete response string."""
        return "".join(self.stream_chat(prompt=prompt, model=model, timeout=timeout))

    def _stream_chat_cli(
        self,
        prompt: str,
        model: str,
        timeout: float,
    ) -> Iterator[str]:
        """Runs `junie --task <prompt> --model <id> --output-format json` and yields the result.

        JSON mode keeps stdout free of the session banner, hook warnings, and
        MCP noise that pollute the text output, returning a single
        `{"result": "..."}` object with the task's final answer.
        """
        cmd = [
            self.binary_path,
            "--task",
            prompt,
            "--model",
            model,
            "--output-format",
            "json",
            "--skip-update-check",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise BridgeError(
                f"Junie CLI (junie) not found at '{self.binary_path}'. "
                "Install via: curl -fsSL https://junie.jetbrains.com/install.sh | bash"
            ) from exc
        except OSError as exc:
            raise BridgeError(f"Failed to execute Junie CLI ({self.binary_path}): {exc}") from exc

        if proc.stdout is None:
            if proc.poll() is None:
                proc.kill()
            raise BridgeError(f"Junie CLI produced no stdout for model '{model}'")

        stdout_lines: list[str] = []
        start = time.time()
        for raw_line in proc.stdout:
            if (time.time() - start) > timeout:
                if proc.poll() is None:
                    proc.kill()
                raise UpstreamTimeoutError(f"Junie CLI request timed out after {timeout}s")
            stdout_lines.append(raw_line)

        if proc.poll() is None:
            proc.kill()
        returncode = proc.returncode

        stdout_text = "".join(stdout_lines).strip()
        result_text: str | None = None
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            data = json.loads(stdout_text)
            if isinstance(data, dict) and isinstance(data.get("result"), str):
                result_text = data["result"]

        if returncode not in (0, None) or result_text is None:
            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read().strip()
            detail = stderr_text[:500] or f"exit {returncode}"
            raise _map_junie_error(model, detail)

        yield result_text
