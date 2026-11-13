"""Protocol transformers from OpenAI Chat, Anthropic Messages, and OpenAI Responses to AI Studio Interactions API."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from typing import Any

from bridge.aistudio_backend import resolve_aistudio_model_and_thinking
from bridge.aistudio_cache import InteractionStepCache, canonical_hash
from bridge.errors import InvalidRequestError


@dataclass
class AistudioTransformedRequest:
    """Normalized Interactions API request plus continuity metadata for cache commit."""

    protocol: str
    official_model: str
    thinking_level: str | None
    payload: dict[str, Any]
    prefix_items: list[dict[str, Any]]
    canonical_history: list[dict[str, Any]]
    request_hash: str
    session_id: str | None = None


def _extract_strict_text(content: Any, *, field_name: str = "content") -> str:
    """Extracts text strictly; raises InvalidRequestError on unsupported multimodal/non-text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                raise InvalidRequestError(f"Unsupported non-text {field_name} item in AI Studio v1.")
            ptype = str(item.get("type") or "").strip().lower()
            if ptype in ("text", "input_text", "output_text"):
                val = item.get("text")
                if isinstance(val, str):
                    parts.append(val)
            else:
                raise InvalidRequestError(
                    f"Content type '{ptype or 'unknown'}' in {field_name} is not supported by AI Studio v1."
                )
        return "".join(parts)
    raise InvalidRequestError(f"Unsupported {field_name} type in AI Studio v1.")


def _normalize_json_args(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidRequestError(f"Invalid JSON tool arguments in history: {exc}") from None
        if not isinstance(loaded, dict):
            raise InvalidRequestError("Tool arguments must decode to a JSON object.")
        return loaded
    raise InvalidRequestError("Unsupported tool arguments type in history.")


def _extract_explicit_effort(payload: dict[str, Any]) -> str | None:
    # Reject explicit thinking_budget / budget_tokens since Interactions Hydrogen rejects budgets
    if "thinking_budget" in payload and payload["thinking_budget"] is not None:
        raise InvalidRequestError(
            "'thinking_budget' is not supported by AI Studio Interactions API; use 'reasoning_effort' or 'thinking_level'."
        )

    output_cfg = payload.get("output_config") or payload.get("outputConfig")
    if isinstance(output_cfg, dict) and output_cfg.get("effort"):
        return str(output_cfg["effort"]).strip().lower()

    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        if "budget_tokens" in thinking and thinking["budget_tokens"] is not None:
            raise InvalidRequestError(
                "'thinking.budget_tokens' is not supported by AI Studio Interactions API; use 'output_config.effort' or 'thinking_level'."
            )
        if "thinkingBudget" in thinking and thinking["thinkingBudget"] is not None:
            raise InvalidRequestError(
                "'thinking.thinkingBudget' is not supported by AI Studio Interactions API."
            )
        if thinking.get("thinkingLevel"):
            return str(thinking["thinkingLevel"]).strip().lower()
        t_type = str(thinking.get("type") or "").strip().lower()
        if t_type in ("disabled", "off", "none"):
            raise InvalidRequestError(
                "Disabling thinking via thinking.type='disabled' is not supported by AI Studio Interactions models; "
                "use effort 'low' instead."
            )

    for key in ("thinking_level", "reasoning_effort", "effort"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()

    reasoning_obj = payload.get("reasoning")
    if isinstance(reasoning_obj, dict) and isinstance(reasoning_obj.get("effort"), str):
        eff = reasoning_obj["effort"].strip().lower()
        if eff:
            return eff

    return None


def _translate_tools_openai(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if not isinstance(tools, list):
        raise InvalidRequestError("'tools' must be a list.")
    out: list[dict[str, Any]] = []
    for entry in tools:
        if not isinstance(entry, dict):
            raise InvalidRequestError("Each tool entry must be an object.")
        ttype = str(entry.get("type") or "").strip().lower()
        if ttype == "function" and isinstance(entry.get("function"), dict):
            fn = entry["function"]
            name = str(fn.get("name") or "").strip()
            if not name:
                raise InvalidRequestError("Function tool declaration missing name.")
            decl: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": copy.deepcopy(fn.get("parameters") or {"type": "object", "properties": {}}),
            }
            if fn.get("description"):
                decl["description"] = str(fn["description"])
            out.append(decl)
        elif ttype == "function" and entry.get("name"):
            name = str(entry["name"]).strip()
            decl = {
                "type": "function",
                "name": name,
                "parameters": copy.deepcopy(entry.get("parameters") or {"type": "object", "properties": {}}),
            }
            if entry.get("description"):
                decl["description"] = str(entry["description"])
            out.append(decl)
        else:
            raise InvalidRequestError(f"Tool type '{ttype or 'unknown'}' is not supported by AI Studio v1.")
    return out or None


def _translate_tools_anthropic(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if not isinstance(tools, list):
        raise InvalidRequestError("'tools' must be a list.")
    out: list[dict[str, Any]] = []
    for entry in tools:
        if not isinstance(entry, dict):
            raise InvalidRequestError("Each Anthropic tool entry must be an object.")
        if entry.get("type") and entry.get("type") != "custom":
            raise InvalidRequestError(f"Server/built-in Anthropic tool '{entry.get('type')}' is not supported by AI Studio v1.")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise InvalidRequestError("Anthropic tool declaration missing name.")
        decl: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": copy.deepcopy(entry.get("input_schema") or {"type": "object", "properties": {}}),
        }
        if entry.get("description"):
            decl["description"] = str(entry["description"])
        out.append(decl)
    return out or None


def _translate_tool_choice(choice: Any, *, anthropic: bool = False) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        c = choice.strip().lower()
        if c == "auto":
            return "auto"
        if c in ("required", "any"):
            return "any"
        if c == "none":
            return "none"
        raise InvalidRequestError(f"Unsupported tool_choice '{choice}' in AI Studio v1.")
    if isinstance(choice, dict):
        ctype = str(choice.get("type") or "").strip().lower()
        if ctype == "auto":
            return "auto"
        if ctype in ("any", "required"):
            return "any"
        if ctype == "none":
            return "none"
        if ctype == "function":
            fn = choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else choice.get("name")) or ""
            if name:
                return {"allowed_tools": {"mode": "any", "tools": [str(name)]}}
        if anthropic and ctype == "tool" and choice.get("name"):
            return {"allowed_tools": {"mode": "any", "tools": [str(choice["name"])]}}
    raise InvalidRequestError("Unsupported tool_choice configuration in AI Studio v1.")


def _build_generation_config(
    payload: dict[str, Any],
    thinking_level: str | None,
    tool_choice: Any,
) -> dict[str, Any] | None:
    cfg: dict[str, Any] = {}
    if thinking_level:
        cfg["thinking_level"] = thinking_level
    if tool_choice is not None:
        cfg["tool_choice"] = tool_choice

    max_tok = payload.get("max_output_tokens") or payload.get("max_completion_tokens") or payload.get("max_tokens")
    if max_tok is not None:
        cfg["max_output_tokens"] = int(max_tok)

    stop = payload.get("stop_sequences") if "stop_sequences" in payload else payload.get("stop")
    if isinstance(stop, str) and stop:
        cfg["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        cfg["stop_sequences"] = [str(s) for s in stop if s]

    if "seed" in payload and payload["seed"] is not None:
        cfg["seed"] = int(payload["seed"])

    return cfg or None


def _reconcile_tool_results(
    expected_calls: list[tuple[str, str]],
    call_id_map: dict[str, str],
    submitted: list[tuple[str, str, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validates completeness/uniqueness and reorders function_results to match original calls."""
    if not expected_calls:
        raise InvalidRequestError("Received tool results for a turn that issued no function calls.")

    expected_by_native = {native_id: fn_name for native_id, fn_name in expected_calls}
    seen_native: dict[str, tuple[str, str, bool]] = {}

    for pub_id, result_text, is_err in submitted:
        native_id = call_id_map.get(pub_id, pub_id)
        if native_id not in expected_by_native:
            raise InvalidRequestError(f"Unknown tool result call_id '{pub_id}' for this AI Studio turn.")
        if native_id in seen_native:
            raise InvalidRequestError(f"Duplicate tool result for call_id '{pub_id}'.")
        seen_native[native_id] = (pub_id, result_text, is_err)

    missing = [native_id for native_id, _ in expected_calls if native_id not in seen_native]
    if missing:
        raise InvalidRequestError(
            f"Incomplete parallel tool results: missing result(s) for call_id(s): {', '.join(missing)}."
        )

    google_steps: list[dict[str, Any]] = []
    canonical_entries: list[dict[str, Any]] = []
    for native_id, fn_name in expected_calls:
        pub_id, result_text, is_err = seen_native[native_id]
        step: dict[str, Any] = {
            "type": "function_result",
            "name": fn_name,
            "call_id": native_id,
            "result": [{"type": "text", "text": result_text}],
        }
        if is_err:
            step["is_error"] = True
        google_steps.append(step)
        canonical_entries.append({
            "role": "tool",
            "call_id": pub_id,
            "text": result_text,
            "is_error": is_err,
        })

    return google_steps, canonical_entries


def chat_to_aistudio_request(
    payload: dict[str, Any],
    *,
    cache: InteractionStepCache,
    session_id: str | None = None,
) -> AistudioTransformedRequest:
    """Translates OpenAI Chat Completions payload into a stateless Interactions API request."""
    explicit_effort = _extract_explicit_effort(payload)
    official_model, thinking_level = resolve_aistudio_model_and_thinking(
        str(payload.get("model") or ""),
        explicit_level=explicit_effort,
    )

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("'messages' must be a non-empty list.")

    system_parts: list[str] = []
    input_steps: list[dict[str, Any]] = []
    canonical_prefix: list[dict[str, Any]] = []

    idx = 0
    while idx < len(messages):
        msg = messages[idx]
        if not isinstance(msg, dict):
            raise InvalidRequestError("Each message must be an object.")
        role = str(msg.get("role") or "").strip().lower()

        if role in ("system", "developer"):
            txt = _extract_strict_text(msg.get("content"), field_name=f"{role}.content")
            if txt:
                system_parts.append(txt)
            idx += 1
            continue

        if role == "user":
            txt = _extract_strict_text(msg.get("content"), field_name="user.content")
            input_steps.append({"type": "user_input", "content": [{"type": "text", "text": txt}]})
            canonical_prefix.append({"role": "user", "text": txt})
            idx += 1
            continue

        if role == "assistant":
            txt = _extract_strict_text(msg.get("content"), field_name="assistant.content")
            raw_tcs = msg.get("tool_calls") or []
            norm_tcs: list[dict[str, Any]] = []
            if isinstance(raw_tcs, list):
                for tc in raw_tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    norm_tcs.append({
                        "id": str(tc.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "arguments": _normalize_json_args(fn.get("arguments")),
                    })
            asst_visible = {"text": txt, "tool_calls": norm_tcs}
            turn_rec = cache.resolve_turn(
                protocol="openai_chat",
                official_model=official_model,
                prefix_items=canonical_prefix,
                assistant_visible=asst_visible,
                session_id=session_id,
            )
            input_steps.extend(copy.deepcopy(turn_rec.raw_steps))
            canonical_prefix.append(asst_visible)
            idx += 1

            if turn_rec.expected_calls:
                submitted: list[tuple[str, str, bool]] = []
                while idx < len(messages) and str(messages[idx].get("role") or "").strip().lower() == "tool":
                    tmsg = messages[idx]
                    call_id = str(tmsg.get("tool_call_id") or "").strip()
                    if not call_id:
                        raise InvalidRequestError("Tool message missing 'tool_call_id'.")
                    res_text = _extract_strict_text(tmsg.get("content"), field_name="tool.content")
                    submitted.append((call_id, res_text, False))
                    idx += 1
                g_steps, c_entries = _reconcile_tool_results(
                    turn_rec.expected_calls,
                    turn_rec.call_id_map,
                    submitted,
                )
                input_steps.extend(g_steps)
                canonical_prefix.extend(c_entries)
            continue

        if role == "tool":
            raise InvalidRequestError("Unexpected 'tool' message without a preceding assistant tool call turn.")

        raise InvalidRequestError(f"Unsupported message role '{role}' in AI Studio v1.")

    tools = _translate_tools_openai(payload.get("tools"))
    tool_choice = _translate_tool_choice(payload.get("tool_choice"))
    gen_cfg = _build_generation_config(payload, thinking_level, tool_choice)

    req_payload: dict[str, Any] = {
        "model": official_model,
        "store": False,
        "input": input_steps,
    }
    if system_parts:
        req_payload["system_instruction"] = "\n\n".join(system_parts)
    if tools is not None:
        req_payload["tools"] = tools
    if gen_cfg is not None:
        req_payload["generation_config"] = gen_cfg

    return AistudioTransformedRequest(
        protocol="openai_chat",
        official_model=official_model,
        thinking_level=thinking_level,
        payload=req_payload,
        prefix_items=canonical_prefix,
        canonical_history=copy.deepcopy(canonical_prefix),
        request_hash=canonical_hash({"payload": req_payload, "messages": messages}),
        session_id=session_id,
    )


def anthropic_to_aistudio_request(
    payload: dict[str, Any],
    *,
    cache: InteractionStepCache,
    session_id: str | None = None,
) -> AistudioTransformedRequest:
    """Translates Anthropic Messages payload into a stateless Interactions API request."""
    explicit_effort = _extract_explicit_effort(payload)
    official_model, thinking_level = resolve_aistudio_model_and_thinking(
        str(payload.get("model") or ""),
        explicit_level=explicit_effort,
    )

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("'messages' must be a non-empty list.")

    system_text = _extract_strict_text(payload.get("system"), field_name="system")
    input_steps: list[dict[str, Any]] = []
    canonical_prefix: list[dict[str, Any]] = []
    pending_turn: Any = None

    for msg in messages:
        if not isinstance(msg, dict):
            raise InvalidRequestError("Each Anthropic message must be an object.")
        role = str(msg.get("role") or "").strip().lower()
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                if pending_turn is not None and pending_turn.expected_calls:
                    raise InvalidRequestError("Expected tool_result blocks following assistant tool_use turn.")
                input_steps.append({"type": "user_input", "content": [{"type": "text", "text": content}]})
                canonical_prefix.append({"role": "user", "text": content})
                continue

            if not isinstance(content, list):
                raise InvalidRequestError("Anthropic user content must be a string or list of blocks.")

            tool_results: list[tuple[str, str, bool]] = []
            trailing_texts: list[str] = []
            saw_text_before_tool = False
            for blk in content:
                if not isinstance(blk, dict):
                    raise InvalidRequestError("Anthropic content block must be an object.")
                btype = str(blk.get("type") or "").strip().lower()
                if btype == "tool_result":
                    if saw_text_before_tool:
                        raise InvalidRequestError("Anthropic tool_result blocks must precede text blocks.")
                    tu_id = str(blk.get("tool_use_id") or "").strip()
                    if not tu_id:
                        raise InvalidRequestError("tool_result block missing 'tool_use_id'.")
                    res_text = _extract_strict_text(blk.get("content"), field_name="tool_result.content")
                    is_err = bool(blk.get("is_error", False))
                    tool_results.append((tu_id, res_text, is_err))
                elif btype == "text":
                    if not tool_results:
                        saw_text_before_tool = True
                    val = blk.get("text")
                    if isinstance(val, str):
                        trailing_texts.append(val)
                else:
                    raise InvalidRequestError(f"Unsupported Anthropic block type '{btype}' in AI Studio v1.")

            if tool_results:
                if pending_turn is None or not pending_turn.expected_calls:
                    raise InvalidRequestError("Received tool_result without a preceding tool_use turn.")
                g_steps, c_entries = _reconcile_tool_results(
                    pending_turn.expected_calls,
                    pending_turn.call_id_map,
                    tool_results,
                )
                input_steps.extend(g_steps)
                canonical_prefix.extend(c_entries)
                pending_turn = None
            elif pending_turn is not None and pending_turn.expected_calls:
                raise InvalidRequestError("Missing tool_result for pending assistant tool_use turn.")

            joined_text = "".join(trailing_texts)
            if joined_text or not tool_results:
                input_steps.append({"type": "user_input", "content": [{"type": "text", "text": joined_text}]})
                canonical_prefix.append({"role": "user", "text": joined_text})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            norm_tcs: list[dict[str, Any]] = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    btype = str(blk.get("type") or "").strip().lower()
                    if btype == "text":
                        if isinstance(blk.get("text"), str):
                            text_parts.append(blk["text"])
                    elif btype == "tool_use":
                        norm_tcs.append({
                            "id": str(blk.get("id") or ""),
                            "name": str(blk.get("name") or ""),
                            "arguments": _normalize_json_args(blk.get("input")),
                        })
                    elif btype in ("thinking", "redacted_thinking"):
                        # Thought blocks are reconstructed from InteractionStepCache, never forged from client text
                        continue
                    else:
                        raise InvalidRequestError(f"Unsupported assistant block type '{btype}' in AI Studio v1.")
            else:
                raise InvalidRequestError("Anthropic assistant content must be string or list.")

            asst_visible = {"text": "".join(text_parts), "tool_calls": norm_tcs}
            turn_rec = cache.resolve_turn(
                protocol="anthropic",
                official_model=official_model,
                prefix_items=canonical_prefix,
                assistant_visible=asst_visible,
                session_id=session_id,
            )
            input_steps.extend(copy.deepcopy(turn_rec.raw_steps))
            canonical_prefix.append(asst_visible)
            pending_turn = turn_rec
            continue

        raise InvalidRequestError(f"Unsupported Anthropic message role '{role}' in AI Studio v1.")

    if pending_turn is not None and pending_turn.expected_calls:
        raise InvalidRequestError("Missing tool_result blocks for pending assistant tool_use.")

    tools = _translate_tools_anthropic(payload.get("tools"))
    tool_choice = _translate_tool_choice(payload.get("tool_choice"), anthropic=True)
    gen_cfg = _build_generation_config(payload, thinking_level, tool_choice)

    req_payload: dict[str, Any] = {
        "model": official_model,
        "store": False,
        "input": input_steps,
    }
    if system_text:
        req_payload["system_instruction"] = system_text
    if tools is not None:
        req_payload["tools"] = tools
    if gen_cfg is not None:
        req_payload["generation_config"] = gen_cfg

    return AistudioTransformedRequest(
        protocol="anthropic",
        official_model=official_model,
        thinking_level=thinking_level,
        payload=req_payload,
        prefix_items=canonical_prefix,
        canonical_history=copy.deepcopy(canonical_prefix),
        request_hash=canonical_hash({"payload": req_payload, "messages": messages}),
        session_id=session_id,
    )


def responses_to_aistudio_request(
    payload: dict[str, Any],
    *,
    cache: InteractionStepCache,
    session_id: str | None = None,
) -> AistudioTransformedRequest:
    """Translates OpenAI Responses payload (both previous_response_id and full-input replay)."""
    explicit_effort = _extract_explicit_effort(payload)
    official_model, thinking_level = resolve_aistudio_model_and_thinking(
        str(payload.get("model") or ""),
        explicit_level=explicit_effort,
    )

    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        items: list[dict[str, Any]] = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        items = [copy.deepcopy(i) for i in raw_input]
    else:
        raise InvalidRequestError("Responses 'input' must be a string or list of items.")

    input_steps: list[dict[str, Any]] = []
    canonical_prefix: list[dict[str, Any]] = []
    pending_turn: Any = None

    prev_resp_id = payload.get("previous_response_id")
    if isinstance(prev_resp_id, str) and prev_resp_id.strip():
        prev_rec = cache.resolve_response_id(
            prev_resp_id.strip(),
            official_model=official_model,
            session_id=session_id,
        )
        if prev_rec.full_steps_snapshot is not None:
            input_steps = copy.deepcopy(prev_rec.full_steps_snapshot)
        else:
            input_steps = copy.deepcopy(prev_rec.raw_steps)
        canonical_prefix = copy.deepcopy(prev_rec.canonical_history)
        pending_turn = prev_rec

        # If caller mixed previous_response_id with full/partial replayed history items,
        # strip the matching cached prefix cleanly so steps are never duplicated.
        if items and canonical_prefix and _has_assistant_or_call_items(items):
            items = _strip_matching_responses_prefix(items, canonical_prefix)

    idx = 0
    while idx < len(items):
        item = items[idx]
        if not isinstance(item, dict):
            raise InvalidRequestError("Each Responses input item must be an object.")
        itype = str(item.get("type") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()

        if itype == "function_call_output":
            if pending_turn is None or not pending_turn.expected_calls:
                raise InvalidRequestError("Received function_call_output without an active function_call turn.")
            outputs: list[tuple[str, str, bool]] = []
            while idx < len(items) and str(items[idx].get("type") or "").strip().lower() == "function_call_output":
                out_item = items[idx]
                cid = str(out_item.get("call_id") or "").strip()
                if not cid:
                    raise InvalidRequestError("function_call_output missing 'call_id'.")
                out_text = _extract_strict_text(out_item.get("output"), field_name="function_call_output.output")
                outputs.append((cid, out_text, False))
                idx += 1
            g_steps, c_entries = _reconcile_tool_results(
                pending_turn.expected_calls,
                pending_turn.call_id_map,
                outputs,
            )
            input_steps.extend(g_steps)
            canonical_prefix.extend(c_entries)
            pending_turn = None
            continue

        # Assistant output items replayed in full input mode (message with role=assistant and/or function_call)
        if itype == "function_call" or role == "assistant":
            text_parts: list[str] = []
            norm_tcs: list[dict[str, Any]] = []
            while idx < len(items):
                cur = items[idx]
                cur_type = str(cur.get("type") or "").strip().lower()
                cur_role = str(cur.get("role") or "").strip().lower()
                if cur_type == "function_call":
                    norm_tcs.append({
                        "id": str(cur.get("call_id") or cur.get("id") or ""),
                        "name": str(cur.get("name") or ""),
                        "arguments": _normalize_json_args(cur.get("arguments")),
                    })
                    idx += 1
                elif cur_role == "assistant":
                    text_parts.append(_extract_strict_text(cur.get("content"), field_name="assistant.content"))
                    idx += 1
                else:
                    break
            asst_visible = {"text": "".join(text_parts), "tool_calls": norm_tcs}
            turn_rec = cache.resolve_turn(
                protocol="openai_responses",
                official_model=official_model,
                prefix_items=canonical_prefix,
                assistant_visible=asst_visible,
                session_id=session_id,
            )
            input_steps.extend(copy.deepcopy(turn_rec.raw_steps))
            canonical_prefix.append(asst_visible)
            pending_turn = turn_rec
            continue

        if role in ("system", "developer"):
            # Handled alongside instructions below
            idx += 1
            continue

        if role == "user" or itype == "message":
            txt = _extract_strict_text(item.get("content"), field_name="input.content")
            input_steps.append({"type": "user_input", "content": [{"type": "text", "text": txt}]})
            canonical_prefix.append({"role": "user", "text": txt})
            pending_turn = None
            idx += 1
            continue

        raise InvalidRequestError(f"Unsupported Responses input item type '{itype or role or 'unknown'}' in AI Studio v1.")

    if pending_turn is not None and pending_turn.expected_calls:
        raise InvalidRequestError("Missing function_call_output for pending tool calls.")

    sys_parts: list[str] = []
    if isinstance(payload.get("instructions"), str) and payload["instructions"].strip():
        sys_parts.append(payload["instructions"].strip())
    for item in (raw_input if isinstance(raw_input, list) else []):
        if isinstance(item, dict) and str(item.get("role") or "").strip().lower() in ("system", "developer"):
            t = _extract_strict_text(item.get("content"), field_name="system.content")
            if t:
                sys_parts.append(t)

    tools = _translate_tools_openai(payload.get("tools"))
    tool_choice = _translate_tool_choice(payload.get("tool_choice"))
    gen_cfg = _build_generation_config(payload, thinking_level, tool_choice)

    req_payload: dict[str, Any] = {
        "model": official_model,
        "store": False,
        "input": input_steps,
    }
    if sys_parts:
        req_payload["system_instruction"] = "\n\n".join(sys_parts)
    if tools is not None:
        req_payload["tools"] = tools
    if gen_cfg is not None:
        req_payload["generation_config"] = gen_cfg

    return AistudioTransformedRequest(
        protocol="openai_responses",
        official_model=official_model,
        thinking_level=thinking_level,
        payload=req_payload,
        prefix_items=canonical_prefix,
        canonical_history=copy.deepcopy(canonical_prefix),
        request_hash=canonical_hash({"payload": req_payload, "previous_response_id": prev_resp_id, "input": raw_input}),
        session_id=session_id,
    )


def _has_assistant_or_call_items(items: list[dict[str, Any]]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()
        if itype == "function_call" or role == "assistant":
            return True
    return False


def _strip_matching_responses_prefix(
    items: list[dict[str, Any]],
    canonical_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strips already-cached prefix items if caller mixed previous_response_id with replayed items."""
    last_asst_idx = -1
    for i, cur in enumerate(items):
        if not isinstance(cur, dict):
            continue
        cur_type = str(cur.get("type") or "").strip().lower()
        cur_role = str(cur.get("role") or "").strip().lower()
        if cur_type == "function_call" or cur_role == "assistant":
            last_asst_idx = i
    if last_asst_idx < 0:
        return items

    prefix_slice = items[: last_asst_idx + 1]
    # Validate leading user/assistant turns align cleanly when full history was replayed
    reconstructed: list[dict[str, Any]] = []
    j = 0
    while j < len(prefix_slice):
        cur = prefix_slice[j]
        cur_type = str(cur.get("type") or "").strip().lower()
        cur_role = str(cur.get("role") or "").strip().lower()
        if cur_role == "user" or cur_type == "message" and cur_role != "assistant":
            txt = _extract_strict_text(cur.get("content"), field_name="input.content")
            reconstructed.append({"role": "user", "text": txt})
            j += 1
        elif cur_type == "function_call" or cur_role == "assistant":
            text_parts: list[str] = []
            norm_tcs: list[dict[str, Any]] = []
            while j < len(prefix_slice):
                nxt = prefix_slice[j]
                nxt_type = str(nxt.get("type") or "").strip().lower()
                nxt_role = str(nxt.get("role") or "").strip().lower()
                if nxt_type == "function_call":
                    norm_tcs.append({
                        "id": str(nxt.get("call_id") or nxt.get("id") or ""),
                        "name": str(nxt.get("name") or ""),
                        "arguments": _normalize_json_args(nxt.get("arguments")),
                    })
                    j += 1
                elif nxt_role == "assistant":
                    text_parts.append(_extract_strict_text(nxt.get("content"), field_name="assistant.content"))
                    j += 1
                else:
                    break
            reconstructed.append({"text": "".join(text_parts), "tool_calls": norm_tcs})
        else:
            j += 1

    if reconstructed == canonical_history or (
        len(reconstructed) == 1 and canonical_history and reconstructed[0] == canonical_history[-1]
    ):
        return items[last_asst_idx + 1 :]
    raise InvalidRequestError(
        "Mixed 'previous_response_id' with conflicting replayed items in 'input'."
    )
