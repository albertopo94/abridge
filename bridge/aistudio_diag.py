"""Opt-in continuity diagnostics for Google AI Studio.

Enabled with env var ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS=1.
Emits structured, redacted one-line JSON records to stderr so a real
multi-turn Gentle Shell session can be traced without exposing secrets,
full prompts, tool results, arguments, thought signatures, or raw steps.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from typing import Any

_ENV_VAR = "ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS"


def diagnostics_enabled() -> bool:
    val = os.environ.get(_ENV_VAR, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _digest(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _redact_identifier(value: Any) -> str:
    """Returns a short stable digest of an identifier, avoiding plaintext in logs."""
    return _digest(str(value))[:16]


def message_fingerprints(messages: list[Any]) -> list[dict[str, Any]]:
    """Per-message structural fingerprints (redacted): index, role, hashes, tool IDs.

    Never records content, arguments, results, prompts, or secrets — only SHA-256
    digests and redacted identifier digests, so diverging history entries are
    locatable by index. ``h`` is the full SHA-256 of the canonical serialization of
    the whole message dict and changes on *any* field change; the individual
    structural fields (``c``, ``tc``, ``ta``, ``tcid``) only characterize specific
    differences and are not a complete description of what changed.
    """
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            out.append({"i": idx, "role": "?"})
            continue
        role = str(msg.get("role") or msg.get("type") or "?")
        entry: dict[str, Any] = {"i": idx, "role": role, "h": _digest(msg)}
        content = msg.get("content")
        if content is None:
            entry["c"] = "null"
        else:
            entry["c"] = _digest(content)
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            entry["tc"] = [_redact_identifier(tc.get("id") if isinstance(tc, dict) else tc) for tc in tcs]
            args: list[Any] = []
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        args.append(fn.get("arguments"))
                    else:
                        args.append(None)
                else:
                    args.append(None)
            entry["ta"] = [_digest(a) if a is not None else "null" for a in args]
        if "tool_call_id" in msg:
            entry["tcid"] = _redact_identifier(msg.get("tool_call_id"))
        out.append(entry)
    return out


def diag_log(*, event: str, **fields: Any) -> None:
    """Writes one redacted JSON line to stderr when diagnostics are enabled."""
    if not diagnostics_enabled():
        return
    payload: dict[str, Any] = {"t": round(time.time(), 3), "evt": event}
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    with contextlib.suppress(Exception):
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
