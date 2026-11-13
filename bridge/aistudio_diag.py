"""Opt-in continuity diagnostics for Google AI Studio.

Enabled with env var ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS=1.
Emits structured, redacted one-line JSON records to stderr so a real
multi-turn Gentle Shell session can be traced without exposing secrets,
full prompts, tool results, arguments, thought signatures, or raw steps.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import Any

_ENV_VAR = "ABRIDGE_AISTUDIO_CONTINUITY_DIAGNOSTICS"


def diagnostics_enabled() -> bool:
    val = os.environ.get(_ENV_VAR, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def diag_log(*, event: str, **fields: Any) -> None:
    """Writes one redacted JSON line to stderr when diagnostics are enabled."""
    if not diagnostics_enabled():
        return
    payload: dict[str, Any] = {"t": round(time.time(), 3), "evt": event}
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = str(value)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with contextlib.suppress(Exception):
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
