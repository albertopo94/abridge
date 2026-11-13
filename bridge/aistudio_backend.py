"""Google AI Studio backend adapter: model resolution, SSE Step Assembler, catalog, and status."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from typing import Any

from bridge.aistudio_auth import AistudioKeychainProvider, check_aistudio_auth_health
from bridge.aistudio_client import AistudioClient, AistudioStreamMetrics
from bridge.errors import BridgeError, InvalidRequestError, ProtocolError
from bridge.events import StreamEvent, ToolCallDelta, Usage

AISTUDIO_PROVIDER_PREFIX = "aistudio/"
DEFAULT_AISTUDIO_MODEL = "aistudio/gemini-4-hydrogen"


@dataclass(frozen=True)
class AistudioModelSpec:
    """Declarative capability specification for one AI Studio Interactions model."""

    model_id: str
    display_name: str
    allowed_thinking_levels: tuple[str, ...]
    default_thinking_level: str


AISTUDIO_MODEL_SPECS: dict[str, AistudioModelSpec] = {
    "gemini-4-hydrogen": AistudioModelSpec(
        model_id="gemini-4-hydrogen",
        display_name="AI Studio — Gemini 4 Hydrogen [AI Studio]",
        allowed_thinking_levels=("low", "medium", "high", "xhigh"),
        default_thinking_level="high",
    ),
    "gemini-3.8-flash": AistudioModelSpec(
        model_id="gemini-3.8-flash",
        display_name="AI Studio — Gemini 3.8 Flash [AI Studio]",
        allowed_thinking_levels=("low", "medium", "high"),
        default_thinking_level="medium",
    ),
    "gemini-4-argon": AistudioModelSpec(
        model_id="gemini-4-argon",
        display_name="AI Studio — Gemini 4 Argon [AI Studio]",
        allowed_thinking_levels=("low", "medium", "high"),
        default_thinking_level="high",
    ),
}

_TIER_SUFFIXES: tuple[str, ...] = ("-xhigh", "-high", "-medium", "-low", "-minimal")


def is_aistudio_model(model: str | None) -> bool:
    """Returns True when the requested model id targets the AI Studio upstream."""
    if not model or not isinstance(model, str):
        return False
    return model.strip().lower().startswith(AISTUDIO_PROVIDER_PREFIX)


def strip_aistudio_prefix(model: str) -> str:
    """Removes the `aistudio/` prefix while preserving the remainder."""
    m = model.strip()
    if m.lower().startswith(AISTUDIO_PROVIDER_PREFIX):
        return m[len(AISTUDIO_PROVIDER_PREFIX):]
    return m


def resolve_aistudio_model_and_thinking(
    model: str | None,
    explicit_level: str | None = None,
) -> tuple[str, str | None]:
    """Resolves public `aistudio/*` id (base or tier-suffixed) to `(official_model_id, thinking_level)`.

    Strictly validates `thinking_level` against each model's supported set.
    """
    raw = (model or "").strip()
    if not raw:
        raw = DEFAULT_AISTUDIO_MODEL
    stripped = strip_aistudio_prefix(raw)
    lower = stripped.lower()

    implied_level: str | None = None
    base_candidate = lower
    for suffix in _TIER_SUFFIXES:
        if lower.endswith(suffix):
            candidate = lower[: -len(suffix)]
            if candidate in AISTUDIO_MODEL_SPECS:
                base_candidate = candidate
                implied_level = suffix[1:]
                break

    spec = AISTUDIO_MODEL_SPECS.get(base_candidate)
    if spec is None:
        if implied_level is None and explicit_level is None:
            return stripped, None
        level = (explicit_level or implied_level or "").strip().lower()
        if level == "minimal":
            raise InvalidRequestError(f"Thinking level 'minimal' is not supported by AI Studio model '{raw}'.")
        return stripped, level or None

    chosen = (explicit_level or implied_level or spec.default_thinking_level).strip().lower()
    if chosen not in spec.allowed_thinking_levels:
        allowed_str = ", ".join(spec.allowed_thinking_levels)
        raise InvalidRequestError(
            f"Thinking level '{chosen}' is not supported by '{spec.model_id}'. "
            f"Allowed levels: {allowed_str}."
        )
    return spec.model_id, chosen


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return str(content["text"])
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            t = _extract_text_from_content(item)
            if t:
                parts.append(t)
        return "".join(parts)
    return ""


def _extract_usage(source: dict[str, Any] | None) -> Usage | None:
    if not isinstance(source, dict):
        return None
    usage_obj = source.get("usage")
    if not isinstance(usage_obj, dict):
        return None
    in_tok = int(usage_obj.get("total_input_tokens") or usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens") or 0)
    out_tok = int(usage_obj.get("total_output_tokens") or usage_obj.get("output_tokens") or usage_obj.get("completion_tokens") or 0)
    thought_tok = int(usage_obj.get("total_reasoning_tokens") or usage_obj.get("thought_tokens") or 0)
    total_tok = int(usage_obj.get("total_tokens") or (in_tok + out_tok))
    return Usage(
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
        total_tokens=total_tok,
        thought_tokens=thought_tok,
    )


def _normalize_finish_reason(status: str | None, interaction: dict[str, Any] | None, saw_tool_call: bool) -> str:
    if saw_tool_call or status == "requires_action":
        return "tool_calls"
    if isinstance(interaction, dict):
        raw_reason = str(interaction.get("finish_reason") or interaction.get("stop_reason") or "").strip().lower()
        if raw_reason in ("max_tokens", "length"):
            return "length"
        if raw_reason in ("safety", "content_filter"):
            return "content_filter"
    return "stop"


@dataclass
class _StepAssembly:
    index: int
    step: dict[str, Any]
    argument_chunks: list[str] = field(default_factory=list)
    text_chunks: list[str] = field(default_factory=list)
    summary_parts: list[Any] = field(default_factory=list)
    signature: str | None = None
    stopped: bool = False


class AistudioStepAssembler:
    """Consumes Interactions SSE events, emitting StreamEvents while preserving exact raw steps."""

    def __init__(self) -> None:
        self._assemblies: dict[int, _StepAssembly] = {}
        self._saw_function_call: bool = False
        self.terminal_reached: bool = False
        self.final_status: str | None = None
        self.interaction_id: str | None = None

    @property
    def raw_steps(self) -> list[dict[str, Any]]:
        """Returns complete raw Google step dicts in chronological step index order."""
        return [copy.deepcopy(self._assemblies[idx].step) for idx in sorted(self._assemblies)]

    def feed(self, event: dict[str, Any]) -> list[StreamEvent]:
        """Processes one parsed SSE event and returns normalized StreamEvents."""
        if not isinstance(event, dict):
            return []

        event_type = str(event.get("event_type") or "")
        out: list[StreamEvent] = []

        if event_type == "error":
            err_obj = event.get("error") or event
            raise BridgeError(f"AI Studio stream error: {err_obj}")

        if event_type == "step.start":
            index = event.get("index")
            step = event.get("step")
            if not isinstance(index, int) or not isinstance(step, dict):
                raise ProtocolError("AI Studio step.start missing valid integer index and step object.")
            if index in self._assemblies:
                raise ProtocolError(f"Duplicate AI Studio step.start for index={index}.")
            step_type = str(step.get("type") or "")
            if step_type in ("image", "audio", "video", "document"):
                raise InvalidRequestError(f"Unsupported multimodal step type '{step_type}' in AI Studio v1.")
            self._assemblies[index] = _StepAssembly(index=index, step=copy.deepcopy(step))
            return out

        if event_type == "step.delta":
            index = event.get("index")
            delta = event.get("delta")
            if not isinstance(index, int) or index not in self._assemblies or not isinstance(delta, dict):
                raise ProtocolError("AI Studio step.delta does not match an active step.start.")
            assembly = self._assemblies[index]
            delta_type = str(delta.get("type") or "")

            if delta_type in ("image", "audio", "video", "document"):
                raise InvalidRequestError(f"Unsupported multimodal delta '{delta_type}' in AI Studio v1.")
            if delta_type == "text":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    assembly.text_chunks.append(text)
                    out.append(StreamEvent(delta_text=text, raw=event))
            elif delta_type == "thought_summary":
                content = delta.get("content")
                if content is not None:
                    assembly.summary_parts.append(copy.deepcopy(content))
                    summary_text = _extract_text_from_content(content)
                    if summary_text:
                        out.append(StreamEvent(delta_thought=summary_text, raw=event))
            elif delta_type == "thought_signature":
                sig = delta.get("signature")
                if isinstance(sig, str) and sig:
                    assembly.signature = sig
                    out.append(StreamEvent(thought_signature=sig, raw=event))
            elif delta_type == "arguments_delta":
                frag = delta.get("arguments")
                if not isinstance(frag, str):
                    raise ProtocolError("AI Studio arguments_delta missing string 'arguments'.")
                assembly.argument_chunks.append(frag)
            elif delta_type == "arguments":
                frag = delta.get("partial_arguments") or delta.get("arguments")
                if isinstance(frag, str):
                    assembly.argument_chunks.append(frag)
            return out

        if event_type == "step.stop":
            index = event.get("index")
            if not isinstance(index, int) or index not in self._assemblies:
                raise ProtocolError("AI Studio step.stop does not match an active step.start.")
            assembly = self._assemblies[index]
            if assembly.stopped:
                raise ProtocolError(f"Duplicate AI Studio step.stop for index={index}.")

            step_type = str(assembly.step.get("type") or "")
            if step_type == "function_call":
                self._saw_function_call = True
                arg_text = "".join(assembly.argument_chunks)
                if arg_text:
                    try:
                        parsed_args = json.loads(arg_text)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ProtocolError(f"AI Studio function_call arguments are not valid JSON: {exc}") from None
                    if not isinstance(parsed_args, dict):
                        raise ProtocolError("AI Studio function_call arguments must decode to a JSON object.")
                    assembly.step["arguments"] = parsed_args
                elif not isinstance(assembly.step.get("arguments"), dict):
                    raise ProtocolError("AI Studio function_call missing valid arguments dictionary.")

                call_id = assembly.step.get("id") or assembly.step.get("call_id") or ""
                fn_name = assembly.step.get("name") or ""
                if not call_id or not fn_name:
                    raise ProtocolError("AI Studio function_call missing id or name.")
                tc_sig = assembly.signature or assembly.step.get("signature") or assembly.step.get("thought_signature")
                out.append(
                    StreamEvent(
                        tool_calls=[
                            ToolCallDelta(
                                id=str(call_id),
                                name=str(fn_name),
                                args=copy.deepcopy(assembly.step["arguments"]),
                                thought_signature=str(tc_sig) if tc_sig else None,
                            )
                        ],
                        raw=event,
                    )
                )
            elif step_type == "thought":
                if assembly.signature:
                    assembly.step["signature"] = assembly.signature
                if assembly.summary_parts:
                    assembly.step["summary"] = assembly.summary_parts
            elif step_type == "model_output":
                content = assembly.step.get("content")
                if not isinstance(content, list):
                    content = []
                if assembly.text_chunks:
                    content.append({"type": "text", "text": "".join(assembly.text_chunks)})
                assembly.step["content"] = content

            assembly.stopped = True
            return out

        if event_type in ("interaction.completed", "interaction.complete", "interaction.requires_action"):
            interaction = event.get("interaction")
            interaction_dict = interaction if isinstance(interaction, dict) else None
            if interaction_dict:
                self.interaction_id = str(interaction_dict.get("id") or "") or self.interaction_id
                self.final_status = str(interaction_dict.get("status") or "") or None
            if event_type == "interaction.requires_action":
                self.final_status = "requires_action"
            elif self.final_status is None:
                self.final_status = "completed"

            self.terminal_reached = True
            usage = _extract_usage(interaction_dict) or _extract_usage(event)
            finish_reason = _normalize_finish_reason(self.final_status, interaction_dict, self._saw_function_call)
            out.append(StreamEvent(finish_reason=finish_reason, usage=usage, raw=event))
            return out

        return out

    def finalize(self) -> list[StreamEvent]:
        """Validates terminal stream completion; fails closed on incomplete streams or open steps."""
        if not self._assemblies and not self.terminal_reached:
            raise BridgeError("AI Studio stream ended without receiving any interaction steps.")
        incomplete = [idx for idx, asm in self._assemblies.items() if not asm.stopped]
        if incomplete:
            raise BridgeError(f"AI Studio stream disconnected before closing steps: {incomplete}")
        if not self.terminal_reached:
            raise BridgeError("AI Studio stream ended before terminal interaction event.")
        return []


def extract_events_from_interaction(
    interaction: dict[str, Any],
) -> tuple[list[StreamEvent], list[dict[str, Any]], str]:
    """Converts a non-streaming Interactions API JSON response into StreamEvents + unmodified raw steps."""
    steps = interaction.get("steps")
    if not isinstance(steps, list):
        raise ProtocolError("AI Studio interaction response missing 'steps' list.")

    raw_steps = [copy.deepcopy(s) for s in steps if isinstance(s, dict)]
    events: list[StreamEvent] = []
    saw_tool_call = False
    latest_sig: str | None = None

    for step in raw_steps:
        stype = str(step.get("type") or "")
        if stype == "thought":
            sig = step.get("signature") or step.get("thought_signature")
            if isinstance(sig, str) and sig:
                latest_sig = sig
            summary_text = _extract_text_from_content(step.get("summary"))
            if summary_text or latest_sig:
                events.append(
                    StreamEvent(
                        delta_thought=summary_text or None,
                        thought_signature=str(sig) if isinstance(sig, str) and sig else None,
                        raw=step,
                    )
                )
        elif stype == "model_output":
            content = step.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = str(part.get("type") or "")
                    if ptype == "text" and isinstance(part.get("text"), str) and part["text"]:
                        events.append(StreamEvent(delta_text=part["text"], raw=part))
                    elif ptype in ("image", "audio", "video", "document"):
                        raise InvalidRequestError(f"Unsupported multimodal output part '{ptype}' in AI Studio v1.")
        elif stype == "function_call":
            saw_tool_call = True
            call_id = step.get("id") or step.get("call_id") or ""
            fn_name = step.get("name") or ""
            args = step.get("arguments")
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            if not isinstance(args, dict):
                args = {}
            tc_sig = step.get("signature") or step.get("thought_signature") or latest_sig
            events.append(
                StreamEvent(
                    tool_calls=[
                        ToolCallDelta(
                            id=str(call_id),
                            name=str(fn_name),
                            args=copy.deepcopy(args),
                            thought_signature=str(tc_sig) if tc_sig else None,
                        )
                    ],
                    raw=step,
                )
            )
        elif stype in ("image", "audio", "video", "document"):
            raise InvalidRequestError(f"Unsupported multimodal step '{stype}' in AI Studio v1.")

    status = str(interaction.get("status") or ("requires_action" if saw_tool_call else "completed"))
    usage = _extract_usage(interaction)
    finish_reason = _normalize_finish_reason(status, interaction, saw_tool_call)
    events.append(StreamEvent(finish_reason=finish_reason, usage=usage, raw=interaction))
    return events, raw_steps, status


class AistudioBackend:
    """Backend orchestrator for Google AI Studio direct model inference (`aistudio/*`)."""

    def __init__(
        self,
        key_provider: AistudioKeychainProvider | None = None,
        client: AistudioClient | None = None,
    ) -> None:
        self.key_provider = key_provider if key_provider is not None else AistudioKeychainProvider()
        self.client = client if client is not None else AistudioClient(key_provider=self.key_provider)

    def is_available(self) -> bool:
        """Returns True when an AI Studio Gemini API key is present in macOS Keychain."""
        return self.key_provider.has_api_key()

    def list_models(self) -> list[dict[str, Any]]:
        """Returns the curated AI Studio model catalog without calling Antigravity OAuth."""
        return [
            {
                "id": f"{AISTUDIO_PROVIDER_PREFIX}{spec.model_id}",
                "slug": f"{AISTUDIO_PROVIDER_PREFIX}{spec.model_id}",
                "display_name": spec.display_name,
                "owned_by": "aistudio",
            }
            for spec in AISTUDIO_MODEL_SPECS.values()
        ]

    def status(self, verify_live: bool = False) -> dict[str, Any]:
        """Reports non-secret AI Studio credential & catalog status for CLI and dashboard."""
        verifier = self.client.probe_api_key if verify_live else None
        health = check_aistudio_auth_health(
            provider=self.key_provider,
            verifier=verifier,
            verify_live=verify_live,
        )
        return {
            "available": bool(health["configured"]) and health["status"] != "rejected",
            "configured": bool(health["configured"]),
            "auth_status": health["status"],
            "detail": health["detail"],
            "service": health["service"],
            "account": health["account"],
            "model_count": len(AISTUDIO_MODEL_SPECS),
            "quota_supported": False,
        }

    def run_non_streaming(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[StreamEvent], list[dict[str, Any]], str]:
        """Executes a non-streaming Interactions call and returns `(events, raw_steps, status)`."""
        interaction = self.client.create_interaction(payload)
        return extract_events_from_interaction(interaction)

    def iter_stream_events(
        self,
        payload: dict[str, Any],
        assembler: AistudioStepAssembler | None = None,
        metrics: AistudioStreamMetrics | None = None,
    ) -> Any:
        """Streams normalized StreamEvent items from AI Studio SSE and updates `assembler`."""
        active_assembler = assembler if assembler is not None else AistudioStepAssembler()
        for sse_event in self.client.stream_interaction(payload, metrics=metrics):
            for stream_ev in active_assembler.feed(sse_event):
                yield stream_ev
        for tail_ev in active_assembler.finalize():
            yield tail_ev
