"""Ephemeral RAM-only step cache for Google AI Studio stateless (`store=false`) continuity.

Never persists conversations, prompts, or secrets to disk or logs.
Tracks exact raw Google `steps[]` JSON, public<->native `call_id` mappings,
per-session locking, TTL expiration, bounded LRU eviction, ambiguity detection,
and idempotency for exact retries.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
import contextlib
import copy
from dataclasses import dataclass, field
import hashlib
import json
import threading
import time
from typing import Any
import uuid

from bridge.errors import InteractionStateError

DEFAULT_CACHE_TTL_SECONDS = 1800.0  # 30 minutes
DEFAULT_MAX_SESSIONS = 256
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024  # 32 MiB


def canonical_hash(value: Any) -> str:
    """Computes a deterministic SHA-256 digest of a JSON-serializable object."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _estimate_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return 1024


def _extract_expected_calls(raw_steps: list[dict[str, Any]]) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for step in raw_steps:
        if not isinstance(step, dict) or step.get("type") != "function_call":
            continue
        call_id = str(step.get("id") or step.get("call_id") or "").strip()
        fn_name = str(step.get("name") or "").strip()
        if call_id and fn_name:
            calls.append((call_id, fn_name))
    return calls


@dataclass
class AistudioTurnRecord:
    """Cached raw Interactions state for one completed assistant turn."""

    turn_id: str
    protocol: str
    official_model: str
    session_id: str | None
    session_key: str
    prefix_hash: str
    assistant_visible_hash: str
    fingerprint: str
    raw_steps: list[dict[str, Any]]
    raw_steps_hash: str
    call_id_map: dict[str, str]
    expected_calls: list[tuple[str, str]]
    response_id: str | None
    full_steps_snapshot: list[dict[str, Any]] | None
    canonical_history: list[dict[str, Any]]
    request_hash: str | None
    created_at: float
    last_accessed_at: float
    approx_bytes: int


@dataclass
class _SessionBucket:
    session_key: str
    turns: list[AistudioTurnRecord] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    in_flight: int = 0
    last_accessed_at: float = 0.0
    total_bytes: int = 0


class InteractionStepCache:
    """Thread-safe, RAM-only, TTL + LRU bounded continuity store for AI Studio."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self.max_total_bytes = int(max_total_bytes)
        self._clock = clock if clock is not None else time.time

        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, _SessionBucket] = OrderedDict()
        self._fingerprint_index: dict[str, list[AistudioTurnRecord]] = {}
        self._response_index: dict[str, AistudioTurnRecord] = {}
        self._total_bytes: int = 0

    @contextlib.contextmanager
    def session_lock(self, session_key: str) -> Iterator[None]:
        """Serializes concurrent requests targeting the same session key."""
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            bucket = self._sessions.get(session_key)
            if bucket is None:
                bucket = _SessionBucket(session_key=session_key, last_accessed_at=now)
                self._sessions[session_key] = bucket
            else:
                self._sessions.move_to_end(session_key)
            bucket.in_flight += 1
            bucket.last_accessed_at = now

        try:
            with bucket.lock:
                yield
        finally:
            with self._lock:
                bucket.in_flight = max(0, bucket.in_flight - 1)
                bucket.last_accessed_at = self._clock()
                # Clean empty ephemeral lock buckets that hold no turns and are idle
                if bucket.in_flight == 0 and not bucket.turns and session_key in self._sessions:
                    del self._sessions[session_key]

    def record_turn(
        self,
        *,
        protocol: str,
        official_model: str,
        prefix_items: list[dict[str, Any]],
        assistant_visible: dict[str, Any],
        raw_steps: list[dict[str, Any]],
        call_id_map: dict[str, str] | None = None,
        session_id: str | None = None,
        response_id: str | None = None,
        full_steps_snapshot: list[dict[str, Any]] | None = None,
        canonical_history: list[dict[str, Any]] | None = None,
        request_hash: str | None = None,
    ) -> AistudioTurnRecord:
        """Commits raw Google steps for a terminal turn (`completed` or `requires_action`)."""
        now = self._clock()
        raw_copy = copy.deepcopy(raw_steps)
        snapshot_copy = copy.deepcopy(full_steps_snapshot) if full_steps_snapshot is not None else None
        canon_copy = copy.deepcopy(canonical_history) if canonical_history is not None else copy.deepcopy(prefix_items) + [copy.deepcopy(assistant_visible)]

        prefix_h = canonical_hash(prefix_items)
        asst_h = canonical_hash(assistant_visible)
        fingerprint = f"{protocol}|{official_model}|{prefix_h}|{asst_h}"
        raw_h = canonical_hash(raw_copy)
        c_map = dict(call_id_map or {})
        expected = _extract_expected_calls(raw_copy)
        # Ensure identity mappings exist for native ids if not explicitly overridden
        for native_id, _ in expected:
            c_map.setdefault(native_id, native_id)

        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        if session_id:
            session_key = f"sid:{protocol}:{session_id}"
        elif response_id:
            session_key = f"resp:{response_id}"
        else:
            session_key = f"anon:{fingerprint}:{raw_h[:12]}"

        approx = _estimate_bytes(raw_copy) + (_estimate_bytes(snapshot_copy) if snapshot_copy else 0) + _estimate_bytes(canon_copy)
        record = AistudioTurnRecord(
            turn_id=turn_id,
            protocol=protocol,
            official_model=official_model,
            session_id=session_id,
            session_key=session_key,
            prefix_hash=prefix_h,
            assistant_visible_hash=asst_h,
            fingerprint=fingerprint,
            raw_steps=raw_copy,
            raw_steps_hash=raw_h,
            call_id_map=c_map,
            expected_calls=expected,
            response_id=response_id,
            full_steps_snapshot=snapshot_copy,
            canonical_history=canon_copy,
            request_hash=request_hash,
            created_at=now,
            last_accessed_at=now,
            approx_bytes=approx,
        )

        with self._lock:
            self._prune_locked(now)
            bucket = self._sessions.get(session_key)
            if bucket is None:
                bucket = _SessionBucket(session_key=session_key, last_accessed_at=now)
                self._sessions[session_key] = bucket
            else:
                self._sessions.move_to_end(session_key)

            # Idempotency: if this session already recorded an identical turn, update timestamp and return it.
            for existing in bucket.turns:
                if existing.fingerprint == fingerprint and existing.raw_steps_hash == raw_h:
                    existing.last_accessed_at = now
                    bucket.last_accessed_at = now
                    if response_id and not existing.response_id:
                        existing.response_id = response_id
                        self._response_index[response_id] = existing
                    return self._clone_record(existing)

            # Also check exact request_hash idempotency inside the same session
            if request_hash:
                for existing in bucket.turns:
                    if existing.request_hash == request_hash:
                        existing.last_accessed_at = now
                        bucket.last_accessed_at = now
                        return self._clone_record(existing)

            bucket.turns.append(record)
            bucket.total_bytes += approx
            bucket.last_accessed_at = now
            self._total_bytes += approx

            if not session_id:
                entries = self._fingerprint_index.setdefault(fingerprint, [])
                # Deduplicate identical raw_steps_hash for the same fingerprint (e.g. idempotent retry)
                if not any(e.raw_steps_hash == raw_h for e in entries):
                    entries.append(record)

            if response_id:
                self._response_index[response_id] = record

            self._prune_locked(now, protected_key=session_key)
            return self._clone_record(record)

    def resolve_turn(
        self,
        protocol: str,
        official_model: str,
        prefix_items: list[dict[str, Any]],
        assistant_visible: dict[str, Any],
        session_id: str | None = None,
    ) -> AistudioTurnRecord:
        """Resolves a prior assistant turn back to its unmodified raw Google steps."""
        now = self._clock()
        prefix_h = canonical_hash(prefix_items)
        asst_h = canonical_hash(assistant_visible)
        fingerprint = f"{protocol}|{official_model}|{prefix_h}|{asst_h}"

        with self._lock:
            self._prune_locked(now)

            if session_id:
                session_key = f"sid:{protocol}:{session_id}"
                bucket = self._sessions.get(session_key)
                if bucket is None:
                    raise InteractionStateError(
                        "AI Studio conversation state is missing or expired for this session. Start a new conversation.",
                        code="interaction_state_missing",
                    )
                matches = [t for t in bucket.turns if t.fingerprint == fingerprint]
                if not matches:
                    raise InteractionStateError(
                        "AI Studio conversation state is missing or mismatched for this session. Start a new conversation.",
                        code="interaction_state_missing",
                    )
                if len({m.raw_steps_hash for m in matches}) > 1:
                    raise InteractionStateError(
                        "Ambiguous AI Studio state within session. Start a new conversation.",
                        code="interaction_state_ambiguous",
                    )
                chosen = matches[-1]
                chosen.last_accessed_at = now
                bucket.last_accessed_at = now
                self._sessions.move_to_end(session_key)
                return self._clone_record(chosen)

            # Global fallback (no explicit session_id header)
            candidates = self._fingerprint_index.get(fingerprint, [])
            active_candidates = [c for c in candidates if (now - c.last_accessed_at) <= self.ttl_seconds]
            if not active_candidates:
                raise InteractionStateError(
                    "AI Studio continuity state is missing or expired. Start a new conversation.",
                    code="interaction_state_missing",
                )

            distinct_hashes = {c.raw_steps_hash for c in active_candidates}
            if len(distinct_hashes) > 1:
                raise InteractionStateError(
                    "Ambiguous AI Studio conversation history detected across concurrent sessions; "
                    "pass header 'x-abridge-session-id' or start a new conversation.",
                    code="interaction_state_ambiguous",
                )

            chosen = active_candidates[-1]
            chosen.last_accessed_at = now
            bucket = self._sessions.get(chosen.session_key)
            if bucket is not None:
                bucket.last_accessed_at = now
                self._sessions.move_to_end(chosen.session_key)
            return self._clone_record(chosen)

    def resolve_response_id(
        self,
        previous_response_id: str,
        official_model: str,
        session_id: str | None = None,
    ) -> AistudioTurnRecord:
        """Resolves an Abridge-generated Responses `previous_response_id` locally."""
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            rec = self._response_index.get(previous_response_id)
            if rec is None:
                raise InteractionStateError(
                    f"AI Studio state for previous_response_id '{previous_response_id}' is missing or expired. "
                    "Start a new conversation.",
                    code="interaction_state_missing",
                )
            if rec.official_model != official_model:
                raise InteractionStateError(
                    f"previous_response_id '{previous_response_id}' belongs to model '{rec.official_model}', "
                    f"not '{official_model}'. Mid-session model switching is not supported.",
                    code="interaction_state_missing",
                )
            if session_id and rec.session_id and rec.session_id != session_id:
                raise InteractionStateError(
                    f"previous_response_id '{previous_response_id}' belongs to a different session.",
                    code="interaction_state_missing",
                )
            rec.last_accessed_at = now
            bucket = self._sessions.get(rec.session_key)
            if bucket is not None:
                bucket.last_accessed_at = now
                self._sessions.move_to_end(rec.session_key)
            return self._clone_record(rec)

    def _clone_record(self, rec: AistudioTurnRecord) -> AistudioTurnRecord:
        return AistudioTurnRecord(
            turn_id=rec.turn_id,
            protocol=rec.protocol,
            official_model=rec.official_model,
            session_id=rec.session_id,
            session_key=rec.session_key,
            prefix_hash=rec.prefix_hash,
            assistant_visible_hash=rec.assistant_visible_hash,
            fingerprint=rec.fingerprint,
            raw_steps=copy.deepcopy(rec.raw_steps),
            raw_steps_hash=rec.raw_steps_hash,
            call_id_map=dict(rec.call_id_map),
            expected_calls=list(rec.expected_calls),
            response_id=rec.response_id,
            full_steps_snapshot=copy.deepcopy(rec.full_steps_snapshot) if rec.full_steps_snapshot is not None else None,
            canonical_history=copy.deepcopy(rec.canonical_history),
            request_hash=rec.request_hash,
            created_at=rec.created_at,
            last_accessed_at=rec.last_accessed_at,
            approx_bytes=rec.approx_bytes,
        )

    def _remove_bucket_locked(self, session_key: str) -> None:
        bucket = self._sessions.pop(session_key, None)
        if bucket is None:
            return
        self._total_bytes = max(0, self._total_bytes - bucket.total_bytes)
        removed_ids = {t.turn_id for t in bucket.turns}
        for fp in list(self._fingerprint_index):
            remaining = [t for t in self._fingerprint_index[fp] if t.turn_id not in removed_ids]
            if remaining:
                self._fingerprint_index[fp] = remaining
            else:
                del self._fingerprint_index[fp]
        for t in bucket.turns:
            if t.response_id and self._response_index.get(t.response_id) is t:
                del self._response_index[t.response_id]

    def _prune_locked(self, now: float, protected_key: str | None = None) -> None:
        # 1. Remove TTL-expired idle sessions
        expired = [
            key
            for key, b in self._sessions.items()
            if b.in_flight == 0 and key != protected_key and (now - b.last_accessed_at) > self.ttl_seconds
        ]
        for key in expired:
            self._remove_bucket_locked(key)

        # 2. Enforce LRU session count and memory limits
        while len(self._sessions) > self.max_sessions or self._total_bytes > self.max_total_bytes:
            victim: str | None = None
            for key, b in self._sessions.items():
                if b.in_flight == 0 and key != protected_key:
                    victim = key
                    break
            if victim is None:
                break
            self._remove_bucket_locked(victim)
