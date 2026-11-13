"""HTTP client for Google AI Studio Interactions API (pure Python stdlib)."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass, field
import http.client
import json
import socket
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from bridge.aistudio_auth import AistudioKeychainProvider, redact_secret
from bridge.errors import (
    AuthenticationError,
    BridgeError,
    CapacityExhaustedError,
    ForbiddenError,
    InvalidRequestError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamError,
    UpstreamTimeoutError,
)

DEFAULT_AISTUDIO_BASE_URL = "https://generativelanguage.googleapis.com"
AISTUDIO_INTERACTIONS_PATH = "/v1beta/interactions"
AISTUDIO_MODELS_PATH = "/v1beta/models"
AISTUDIO_API_REVISION = "2026-05-20"


@dataclass
class AistudioStreamMetrics:
    """Timing instrumentation for a single AI Studio SSE stream."""

    start_time: float = field(default_factory=time.monotonic)
    ttfb_seconds: float | None = None
    first_delta_seconds: float | None = None
    total_seconds: float = 0.0
    max_gap_seconds: float = 0.0
    event_count: int = 0


def _parse_error_payload(raw_bytes: bytes) -> tuple[str, str]:
    """Extracts human error message and optional Google error reason from HTTP error JSON."""
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    reason = ""
    msg = raw_text
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return msg, reason

    if not isinstance(parsed, dict):
        return msg, reason

    err_obj = parsed.get("error")
    if isinstance(err_obj, str):
        msg = err_obj
    elif isinstance(err_obj, dict):
        msg = str(err_obj.get("message") or raw_text)
        details = err_obj.get("details")
        if isinstance(details, list):
            for detail in details:
                if isinstance(detail, dict) and detail.get("reason"):
                    reason = str(detail["reason"])
                    break
    return msg, reason


def map_aistudio_http_error(err: urllib.error.HTTPError, api_key: str) -> BridgeError:
    """Maps a Google AI Studio HTTP error to the centralized BridgeError hierarchy."""
    try:
        raw_bytes = err.read()
    except Exception:
        raw_bytes = b""
    finally:
        with contextlib.suppress(Exception):
            err.close()

    if raw_bytes:
        msg, reason = _parse_error_payload(raw_bytes)
    else:
        msg = str(getattr(err, "reason", None) or err)
        reason = ""

    clean_msg = redact_secret(msg, api_key)
    lower_msg = clean_msg.lower()

    if err.code == 401 or reason == "API_KEY_INVALID" or "api key not valid" in lower_msg:
        return AuthenticationError(f"AI Studio authentication error ({err.code}): {clean_msg}")
    if err.code == 400:
        return InvalidRequestError(f"AI Studio 400 Bad Request: {clean_msg}")
    if err.code == 403:
        return ForbiddenError(f"AI Studio forbidden (403): {clean_msg}")
    if err.code == 404:
        return ModelNotFoundError(f"AI Studio model or endpoint not found (404): {clean_msg}")
    if err.code == 429:
        retry_after_val: int | None = None
        if getattr(err, "headers", None):
            hdr_val = err.headers.get("Retry-After")
            if hdr_val:
                try:
                    retry_after_val = int(str(hdr_val).strip())
                except ValueError:
                    retry_after_val = None
        return RateLimitError(f"AI Studio rate limit / quota exhausted (429): {clean_msg}", retry_after=retry_after_val)
    if err.code == 503:
        return CapacityExhaustedError(f"AI Studio capacity exhausted (503): {clean_msg}")
    return UpstreamError(f"AI Studio HTTP error ({err.code}): {clean_msg}", status_code=err.code)


def verify_aistudio_api_key(
    api_key: str,
    base_url: str = DEFAULT_AISTUDIO_BASE_URL,
    timeout: float = 10.0,
) -> tuple[str, str]:
    """Probes Google AI Studio `/v1beta/models?pageSize=1` without consuming inference quota.

    Returns (status, detail) where status is `"verified"`, `"rejected"`, or `"unverified"`.
    """
    clean_key = api_key.strip()
    if not clean_key:
        return ("rejected", "Gemini API key is empty.")

    url = f"{base_url.rstrip('/')}{AISTUDIO_MODELS_PATH}?pageSize=1"
    req = urllib.request.Request(
        url,
        headers={
            "x-goog-api-key": clean_key,
            "Accept": "application/json",
            "User-Agent": "abridge-aistudio/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return ("verified", "Google AI Studio accepted the API key.")
    except urllib.error.HTTPError as err:
        mapped = map_aistudio_http_error(err, clean_key)
        if isinstance(mapped, (AuthenticationError, ForbiddenError)):
            return ("rejected", redact_secret(str(mapped), clean_key))
        return ("unverified", redact_secret(str(mapped), clean_key))
    except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
        return ("unverified", f"Could not reach Google AI Studio: {redact_secret(str(exc), clean_key)}")


class AistudioClient:
    """Pure-stdlib HTTP client for Google AI Studio Interactions API (`store=false`)."""

    def __init__(
        self,
        key_provider: AistudioKeychainProvider | None = None,
        base_url: str = DEFAULT_AISTUDIO_BASE_URL,
        api_revision: str = AISTUDIO_API_REVISION,
        timeout: float = 300.0,
    ) -> None:
        self.key_provider = key_provider if key_provider is not None else AistudioKeychainProvider()
        self.base_url = base_url.rstrip("/")
        self.api_revision = api_revision
        self.timeout = timeout

    def _build_headers(self, api_key: str, stream: bool = False) -> dict[str, str]:
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            "Api-Revision": self.api_revision,
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "abridge-aistudio/1.0",
        }

    def _prepare_stateless_payload(self, payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        # Hard contract (Checkpoint 016): always store=false, never previous_interaction_id.
        clean = {k: v for k, v in payload.items() if k != "previous_interaction_id"}
        clean["store"] = False
        if stream:
            clean["stream"] = True
        else:
            clean.pop("stream", None)
        return clean

    def probe_api_key(self, api_key: str | None = None, timeout: float = 10.0) -> tuple[str, str]:
        """Checks the configured AI Studio API key against Google."""
        secret = api_key if api_key is not None else self.key_provider.get_api_key()
        return verify_aistudio_api_key(secret, base_url=self.base_url, timeout=timeout)

    def create_interaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Executes a non-streaming stateless Interactions API call."""
        api_key = self.key_provider.get_api_key()
        req_payload = self._prepare_stateless_payload(payload, stream=False)
        body = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}{AISTUDIO_INTERACTIONS_PATH}"
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._build_headers(api_key, stream=False),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as err:
            raise map_aistudio_http_error(err, api_key) from None
        except (TimeoutError, socket.timeout) as exc:
            raise UpstreamTimeoutError(
                f"AI Studio request timed out after {self.timeout}s: {redact_secret(str(exc), api_key)}"
            ) from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            raise BridgeError(
                f"AI Studio network error: {redact_secret(str(exc), api_key)}"
            ) from exc

        try:
            decoded = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise BridgeError(f"AI Studio returned invalid JSON response: {exc}") from exc
        if not isinstance(decoded, dict):
            raise BridgeError("AI Studio returned a non-object JSON payload.")
        return decoded

    def stream_interaction(
        self,
        payload: dict[str, Any],
        metrics: AistudioStreamMetrics | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Executes a streaming stateless Interactions API call and yields SSE event dicts."""
        api_key = self.key_provider.get_api_key()
        req_payload = self._prepare_stateless_payload(payload, stream=True)
        body = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}{AISTUDIO_INTERACTIONS_PATH}?{urllib.parse.urlencode({'alt': 'sse'})}"
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._build_headers(api_key, stream=True),
            method="POST",
        )

        active_metrics = metrics if metrics is not None else AistudioStreamMetrics()
        active_metrics.start_time = time.monotonic()
        last_event_time: float | None = None

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            raise map_aistudio_http_error(err, api_key) from None
        except (TimeoutError, socket.timeout) as exc:
            raise UpstreamTimeoutError(
                f"AI Studio stream timed out after {self.timeout}s: {redact_secret(str(exc), api_key)}"
            ) from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            raise BridgeError(
                f"AI Studio network error: {redact_secret(str(exc), api_key)}"
            ) from exc

        current_event = ""
        data_lines: list[str] = []

        def _flush_frame() -> dict[str, Any] | None:
            nonlocal current_event, data_lines, last_event_time
            if not data_lines:
                current_event = ""
                return None
            data_text = "\n".join(data_lines)
            data_lines = []
            ev_name = current_event
            current_event = ""
            if data_text.strip() == "[DONE]":
                return None
            try:
                parsed = json.loads(data_text)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(parsed, dict):
                return None
            event_type = parsed.get("event_type") or ev_name
            if event_type and "event_type" not in parsed:
                parsed["event_type"] = event_type

            now = time.monotonic()
            if active_metrics.ttfb_seconds is None:
                active_metrics.ttfb_seconds = now - active_metrics.start_time
            if event_type == "step.delta" and active_metrics.first_delta_seconds is None:
                active_metrics.first_delta_seconds = now - active_metrics.start_time
            if last_event_time is not None:
                gap = now - last_event_time
                if gap > active_metrics.max_gap_seconds:
                    active_metrics.max_gap_seconds = gap
            last_event_time = now
            active_metrics.event_count += 1
            return parsed

        try:
            with resp:
                for raw_line in resp:
                    line = (
                        raw_line.decode("utf-8", errors="replace")
                        if isinstance(raw_line, bytes)
                        else str(raw_line)
                    ).rstrip("\r\n")
                    if not line:
                        frame = _flush_frame()
                        if frame is not None:
                            yield frame
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                trailing = _flush_frame()
                if trailing is not None:
                    yield trailing
        except (TimeoutError, socket.timeout) as exc:
            raise UpstreamTimeoutError(
                f"AI Studio stream timed out after {self.timeout}s: {redact_secret(str(exc), api_key)}"
            ) from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            raise BridgeError(
                f"AI Studio stream interrupted: {redact_secret(str(exc), api_key)}"
            ) from exc
        finally:
            active_metrics.total_seconds = time.monotonic() - active_metrics.start_time
