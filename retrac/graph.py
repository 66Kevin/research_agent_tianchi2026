from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from .config import ModelConfig
from .components import create_llm_node

_BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)
_QUERY_NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
MAX_SEARCH_QUERIES = 5
INVALID_STREAK_LIMIT = 2
OVERLAP_THRESHOLD = 3
PREFIX_MIN_SHARED_QUERIES = 4
PREFIX_MIN_TOKENS = 4
PREFIX_RATIO_THRESHOLD = 0.6

class WebCycleResearchConfig(BaseModel):
    """Web cycle research configuration"""
    model: ModelConfig
    fallback_model: ModelConfig | None = None
    stream_token_output: bool = False
    system_prompt: str
    continue_prompt: str
    summary_prompt: str
    max_cycles: int
    max_turns: int
    max_tool_turns_in_context: int = Field(default=5, ge=1)


class WebCycleResearchState(TypedDict, total=False):
    """Web cycle research graph state"""
    question: str
    output: Optional[str]
    messages: List[BaseMessage]
    tool_input: List[BaseMessage]
    cycle_histories: List[List[BaseMessage]]
    process_details: Dict[str, Any]
    context_trim_stats: Dict[str, Any]
    error: list[str]
    search_guard: Dict[str, Any]
    guard_action: Literal["run_tools", "reject_tools", "end_cycle"]
    guard_reason_code: str
    guard_invalid_streak: int
    guard_overlap_count: int


def _has_tool_calls(msg: BaseMessage) -> bool:
    return isinstance(msg, AIMessage) and bool(getattr(msg, "tool_calls", None))


def _has_data_inspection_error(errors: List[str] | None) -> bool:
    if not errors:
        return False
    for err in errors:
        lowered = str(err).lower()
        if "data_inspection_failed" in lowered or "datainspectionfailed" in lowered:
            return True
    return False


def _extract_tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        tool_call_id = tool_call.get("id")
    else:
        tool_call_id = getattr(tool_call, "id", None)
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id
    return None


def _default_search_guard() -> Dict[str, Any]:
    return {
        "last_executed_queries": [],
        "invalid_streak": 0,
        "force_end_cycle": False,
        "last_reason_code": "",
    }


def _normalize_query(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    lowered = lowered.replace('"', " ").replace("'", " ").replace("`", " ")
    lowered = _QUERY_NON_WORD_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _tokenize_query(text: str) -> List[str]:
    normalized = _normalize_query(text)
    if not normalized:
        return []
    return normalized.split(" ")


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
    else:
        name = getattr(tool_call, "name", None)
    return str(name or "")


def _tool_call_args(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        args = tool_call.get("args")
    else:
        args = getattr(tool_call, "args", None)
    if isinstance(args, dict):
        return dict(args)
    return {}


def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        copied = dict(tool_call)
        args = copied.get("args")
        copied["args"] = dict(args) if isinstance(args, dict) else {}
        return copied
    return {
        "name": _tool_call_name(tool_call),
        "args": _tool_call_args(tool_call),
        "id": _extract_tool_call_id(tool_call),
        "type": "tool_call",
    }


def _sanitize_ai_search_tool_calls(message: AIMessage) -> AIMessage:
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    if not raw_tool_calls:
        return message

    changed = False
    sanitized_tool_calls: List[Dict[str, Any]] = []
    for raw_tool_call in raw_tool_calls:
        tool_call = _tool_call_to_dict(raw_tool_call)
        if _tool_call_name(tool_call) == "search":
            args = _tool_call_args(tool_call)
            queries = args.get("query")
            if isinstance(queries, list) and len(queries) > MAX_SEARCH_QUERIES:
                args["query"] = queries[:MAX_SEARCH_QUERIES]
                tool_call["args"] = args
                changed = True
        sanitized_tool_calls.append(tool_call)

    if not changed:
        return message
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"tool_calls": sanitized_tool_calls})
    if hasattr(message, "copy"):
        return message.copy(update={"tool_calls": sanitized_tool_calls})
    return AIMessage(
        content=message.content,
        tool_calls=sanitized_tool_calls,
        invalid_tool_calls=getattr(message, "invalid_tool_calls", []) or [],
        additional_kwargs=getattr(message, "additional_kwargs", {}),
        response_metadata=getattr(message, "response_metadata", {}),
        name=getattr(message, "name", None),
        id=getattr(message, "id", None),
        usage_metadata=getattr(message, "usage_metadata", None) or {},
    )


def _detect_shape_violation(queries: List[str]) -> str | None:
    token_lists = [_tokenize_query(q) for q in queries if isinstance(q, str)]
    token_lists = [tokens for tokens in token_lists if tokens]
    if len(token_lists) < PREFIX_MIN_SHARED_QUERIES:
        return None

    prefix_counts: Dict[tuple[str, ...], int] = {}
    for tokens in token_lists:
        for k in range(PREFIX_MIN_TOKENS, len(tokens) + 1):
            prefix = tuple(tokens[:k])
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    for prefix, count in prefix_counts.items():
        if count < PREFIX_MIN_SHARED_QUERIES:
            continue
        prefix_len = len(prefix)
        candidates = [
            len(tokens)
            for tokens in token_lists
            if len(tokens) >= prefix_len and tuple(tokens[:prefix_len]) == prefix
        ]
        if not candidates:
            continue
        min_len = min(candidates)
        ratio = prefix_len / max(1, min_len)
        if prefix_len >= PREFIX_MIN_TOKENS and ratio >= PREFIX_RATIO_THRESHOLD:
            return "common_prefix_enumeration"
    return None


def _compute_overlap_with_last(queries: List[str], last_queries: List[str]) -> int:
    current_normalized = {_normalize_query(q) for q in queries if _normalize_query(q)}
    last_normalized = {_normalize_query(q) for q in last_queries if _normalize_query(q)}
    return len(current_normalized & last_normalized)


def _build_guard_reject_payload(
    *,
    tool_name: str,
    query: List[str],
    reason_code: str,
    invalid_streak: int,
    overlap_count: int,
) -> str:
    target = query[0] if query else ""
    payload = {
        "tool": tool_name,
        "ok": False,
        "summary": {
            "requested_queries": len(query),
            "total_queries": 0,
            "success_queries": 0,
            "failed_queries": 0,
        },
        "calls": [
            {
                "ok": False,
                "http_status": None,
                "status": "REJECTED",
                "elapsed_ms": 0,
                "error_type": "query_guard",
                "error_message": f"Tool call rejected by query guard: {reason_code}",
                "target": target,
            }
        ],
        "guard": {
            "reason_code": reason_code,
            "invalid_streak": invalid_streak,
            "overlap_count": overlap_count,
        },
        "data": [],
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_search_queries_from_tool_call(tool_call: Any) -> List[str]:
    args = _tool_call_args(tool_call)
    query = args.get("query")
    if not isinstance(query, list):
        return []
    return [str(item) for item in query]


def _strip_ai_tool_calls(message: AIMessage) -> AIMessage:
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"tool_calls": [], "invalid_tool_calls": []})
    if hasattr(message, "copy"):
        return message.copy(update={"tool_calls": [], "invalid_tool_calls": []})
    return AIMessage(
        content=message.content,
        additional_kwargs=getattr(message, "additional_kwargs", {}),
        response_metadata=getattr(message, "response_metadata", {}),
        name=getattr(message, "name", None),
        id=getattr(message, "id", None),
        usage_metadata=getattr(message, "usage_metadata", None) or {},
    )


def _extract_boxed_answer(text: str | None) -> str:
    if not text:
        return ""

    last_result: str | None = None
    i = 0
    n = len(text)

    while True:
        match = _BOXED_RE.search(text, i)
        if not match:
            break
        j = match.end()
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "{":
            i = j
            continue

        depth = 0
        k = j
        escaped = False
        found_closing = False
        while k < n:
            ch = text[k]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_result = text[j + 1 : k]
                    i = k + 1
                    found_closing = True
                    break
            k += 1

        if not found_closing and depth > 0:
            last_result = text[j + 1 : n]
            i = k
        elif not found_closing:
            i = j + 1

    if last_result is None:
        return ""
    result = last_result.strip()
    return result


def _is_ai_like_message(message: Any) -> bool:
    if isinstance(message, AIMessage):
        return True
    if isinstance(message, BaseMessage):
        return type(message).__name__.startswith("AIMessage")
    if isinstance(message, dict):
        msg_type = str(message.get("type", "") or "")
        role = str(message.get("role", "") or "")
        return msg_type.startswith("AIMessage") or role.lower() == "ai" or role.startswith("AIMessage")
    return False


def _message_to_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        message_text = getattr(message, "text", None)
        if isinstance(message_text, str) and message_text:
            return message_text
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)
    return str(message)


def _extract_boxed_from_ai_message(message: Any) -> str:
    if not _is_ai_like_message(message):
        return ""
    return _extract_boxed_answer(_message_to_text(message))


def _extract_latest_boxed_from_cycle_histories(cycle_histories: Any) -> str:
    if not isinstance(cycle_histories, list):
        return ""
    for cycle_history in reversed(cycle_histories):
        if not isinstance(cycle_history, list):
            continue
        for message in reversed(cycle_history):
            boxed = _extract_boxed_from_ai_message(message)
            if boxed:
                return boxed
    return ""


def _trim_tool_history(
    messages: List[BaseMessage],
    keep_last_turns: int,
) -> tuple[List[BaseMessage], Dict[str, Any]]:
    normalized: List[BaseMessage] = []
    turn_spans: list[tuple[int, int, int]] = []
    idx = 0
    orphan_removed = 0
    incomplete_stripped = 0

    while idx < len(messages):
        msg = messages[idx]

        if isinstance(msg, ToolMessage):
            orphan_removed += 1
            idx += 1
            continue

        if isinstance(msg, AIMessage):
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            expected_ids = [_extract_tool_call_id(tc) for tc in raw_tool_calls]
            expected_ids = [tcid for tcid in expected_ids if tcid]
            if raw_tool_calls:
                follow_idx = idx + 1
                follow_tool_messages: List[ToolMessage] = []
                while follow_idx < len(messages) and isinstance(messages[follow_idx], ToolMessage):
                    follow_tool_messages.append(messages[follow_idx])
                    follow_idx += 1

                responded_ids = {
                    tool_msg.tool_call_id
                    for tool_msg in follow_tool_messages
                    if isinstance(tool_msg.tool_call_id, str)
                }
                is_complete = bool(expected_ids) and all(tcid in responded_ids for tcid in expected_ids)

                if is_complete:
                    sanitized_ai_msg = _sanitize_ai_search_tool_calls(msg)
                    start = len(normalized)
                    normalized.append(sanitized_ai_msg)
                    normalized.extend(follow_tool_messages)
                    end = len(normalized)
                    turn_spans.append((start, end, len(follow_tool_messages)))
                    idx = follow_idx
                    continue

                incomplete_stripped += 1
                normalized.append(_strip_ai_tool_calls(msg))
                idx += 1
                continue

        normalized.append(msg)
        idx += 1

    tool_turns_before = len(turn_spans)
    dropped_turns = max(0, tool_turns_before - keep_last_turns)
    dropped_tool_messages = 0

    if dropped_turns == 0:
        stats = {
            "keep_last_turns": keep_last_turns,
            "tool_turns_before": tool_turns_before,
            "tool_turns_after": tool_turns_before,
            "dropped_turns": 0,
            "dropped_tool_messages": 0,
            "orphan_removed": orphan_removed,
            "incomplete_stripped": incomplete_stripped,
            "message_count_before": len(messages),
            "message_count_after": len(normalized),
        }
        return normalized, stats

    keep_flags = [True] * len(normalized)
    for start, end, tool_msg_count in turn_spans[:dropped_turns]:
        dropped_tool_messages += tool_msg_count
        for pos in range(start, end):
            keep_flags[pos] = False

    trimmed = [msg for pos, msg in enumerate(normalized) if keep_flags[pos]]
    stats = {
        "keep_last_turns": keep_last_turns,
        "tool_turns_before": tool_turns_before,
        "tool_turns_after": tool_turns_before - dropped_turns,
        "dropped_turns": dropped_turns,
        "dropped_tool_messages": dropped_tool_messages,
        "orphan_removed": orphan_removed,
        "incomplete_stripped": incomplete_stripped,
        "message_count_before": len(messages),
        "message_count_after": len(trimmed),
    }
    return trimmed, stats


def build_graph(cfg: dict) -> StateGraph:
    """Build web_cycle_research graph"""
    domain_cfg = WebCycleResearchConfig.model_validate(cfg)
    g = StateGraph(WebCycleResearchState)
    
    tool_node = ToolNode(domain_cfg.model.tools, messages_key="tool_input")
    llm_node = create_llm_node(
        model_cfg=domain_cfg.model,
        fallback_model_cfg=domain_cfg.fallback_model,
        timeout_retry_attempts=domain_cfg.max_cycles,
        timeout_fallback_mode="after_primary_timeout_retries",
        stream_token_output=domain_cfg.stream_token_output,
    )
    
    def init_graph(state: WebCycleResearchState) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}

        print(f"State: {state}")
        print(f"Domain config: {domain_cfg}")

        assert "question" in state and state["question"] is not None, "question is required"

        if "messages" not in state or not state.get("messages"):
            patch["messages"] = [
                SystemMessage(content=domain_cfg.system_prompt), 
                HumanMessage(content=state["question"])
            ]
        if "tool_input" not in state or state.get("tool_input") is None:
            patch["tool_input"] = []
        if "cycle_histories" not in state or state.get("cycle_histories") is None:
            patch["cycle_histories"] = []
        if "process_details" not in state or state.get("process_details") is None:
            patch["process_details"] = {'rollouts':[]}
        if "error" not in state or state.get("error") is None:
            patch["error"] = []
        if "search_guard" not in state or not isinstance(state.get("search_guard"), dict):
            patch["search_guard"] = _default_search_guard()
        return patch

    async def end_cycle(state: WebCycleResearchState) -> WebCycleResearchState:
        """End a cycle using summary strategy"""
        summary_prompt = domain_cfg.summary_prompt

        summary_base_messages = list(state.get("messages", []))
        pending_tool_messages = state.get("tool_input", [])
        if isinstance(pending_tool_messages, list) and pending_tool_messages:
            summary_base_messages.extend(pending_tool_messages)

        normalized_messages, summary_input_normalization = _trim_tool_history(
            summary_base_messages,
            keep_last_turns=domain_cfg.max_tool_turns_in_context,
        )
        summary = await llm_node(
            {"messages": normalized_messages + [HumanMessage(content=summary_prompt.format(input=state["question"]))]}
        )

        state_errors = list(state.get("error", []))
        summary_errors = summary.get("error", [])
        if summary_errors:
            state_errors.extend(summary_errors)

        summary_messages = summary.get("messages")
        if isinstance(summary_messages, list) and summary_messages:
            state["messages"] = summary_messages
            summary_last_message: BaseMessage | None = summary_messages[-1]
        else:
            state["messages"] = normalized_messages
            summary_last_message = None

        cycle_histories_raw = state.get("cycle_histories", [])
        cycle_histories = list(cycle_histories_raw) if isinstance(cycle_histories_raw, list) else []
        cycle_histories.append([summary_last_message])

        process_details = state.get("process_details", {})
        if not isinstance(process_details, dict):
            process_details = {"rollouts": []}
        rollouts = process_details.get("rollouts", [])
        if not isinstance(rollouts, list):
            rollouts = []
        rollouts.append(
            {
                "messages": state["messages"],
                "error": state_errors[-1] if state_errors else "",
                "summary_input_normalization": summary_input_normalization,
            }
        )
        process_details["rollouts"] = rollouts

        return {
            "cycle_histories": cycle_histories,
            "process_details": process_details,
            "messages": state["messages"],
            "tool_input": [],
            "error": state_errors,
        }
    
    def cycle_check(state: WebCycleResearchState) -> Literal["final", "start_cycle"]:
        if len(state["cycle_histories"]) >= domain_cfg.max_cycles:
            return "final"
        else:
            return "start_cycle"
    
    async def start_cycle(state: WebCycleResearchState) -> WebCycleResearchState:
        """Start a new cycle using summary strategy"""
        if len(state["cycle_histories"]) == 0:  # first cycle
            return {}
        
        messages = [
            SystemMessage(content=domain_cfg.system_prompt + "\nAlso there are some summary for the previous attempts you have made, you can use them to help you answer the question."), 
            HumanMessage(content=state["question"])
        ]
        contents_without_think = []
        for cycle_history in state["cycle_histories"]:
            contents = []
            for message in cycle_history:
                if message is None:
                    continue
                if isinstance(message, BaseMessage):
                    message_text = message.text if isinstance(message.text, str) and message.text else str(message.content)
                    contents.append(message_text)
                else:
                    message_content = getattr(message, "content", message)
                    contents.append(str(message_content))
            # Remove reasoning part if exists
            contents_without_think += [content.split("</think>")[-1] if "</think>" in content else content for content in contents]
        
        if contents_without_think:
            last_summary = contents_without_think[-1]
        else:
            last_summary = "Previous cycle ended without a usable summary."
        messages += [HumanMessage(content=domain_cfg.continue_prompt.format(last_summary=last_summary))]
        return {"messages": messages, "error": [], "search_guard": _default_search_guard()}

    def decide(state: WebCycleResearchState) -> Literal["tools", "end_cycle"]:
        """Decide next step: execute tools if tool calls exist, otherwise end cycle"""
        if _has_data_inspection_error(state.get("error")):
            return "end_cycle"

        last: AIMessage = state["messages"][-1]
        raw_usage_metadata = getattr(last, "usage_metadata", None)
        usage_metadata = raw_usage_metadata if isinstance(raw_usage_metadata, dict) else {}
        tool_calls_num = sum(1 for message in state["messages"] if isinstance(message, ToolMessage))
        max_context = domain_cfg.model.max_context_length
        
        # Check if limits exceeded
        if (max_context is not None and usage_metadata.get("total_tokens", 0) > max_context) or tool_calls_num > domain_cfg.max_turns:
            return "end_cycle"

        # Default strategy: execute tools if tool calls exist, otherwise end cycle
        return "tools" if _has_tool_calls(last) else "end_cycle"

    def prep_tools(state: WebCycleResearchState) -> Dict[str, Any]:
        guard = state.get("search_guard")
        if not isinstance(guard, dict):
            guard = _default_search_guard()

        last_executed_queries_raw = guard.get("last_executed_queries", [])
        last_executed_queries = (
            [str(q) for q in last_executed_queries_raw if isinstance(q, str)]
            if isinstance(last_executed_queries_raw, list)
            else []
        )
        invalid_streak_raw = guard.get("invalid_streak", 0)
        invalid_streak = invalid_streak_raw if isinstance(invalid_streak_raw, int) and invalid_streak_raw >= 0 else 0
        latest_message = state["messages"][-1]

        if not isinstance(latest_message, AIMessage):
            return {
                "tool_input": [latest_message],
                "search_guard": {
                    "last_executed_queries": last_executed_queries,
                    "invalid_streak": 0,
                    "force_end_cycle": False,
                    "last_reason_code": "",
                },
                "guard_action": "run_tools",
                "guard_reason_code": "",
                "guard_invalid_streak": 0,
                "guard_overlap_count": 0,
            }

        raw_tool_calls = getattr(latest_message, "tool_calls", None) or []
        tool_calls = [_tool_call_to_dict(tool_call) for tool_call in raw_tool_calls]
        search_calls = [tool_call for tool_call in tool_calls if _tool_call_name(tool_call) == "search"]

        reason_code = ""
        overlap_count = 0
        if search_calls:
            for search_call in search_calls:
                query = _extract_search_queries_from_tool_call(search_call)
                if len(query) > MAX_SEARCH_QUERIES:
                    reason_code = "too_many_queries"
                    break
                normalized_query = [_normalize_query(item) for item in query]
                normalized_query = [item for item in normalized_query if item]
                if len(normalized_query) != len(set(normalized_query)):
                    reason_code = "duplicate_queries"
                    break
                shape_reason = _detect_shape_violation(query)
                if shape_reason:
                    reason_code = shape_reason
                    break
                overlap_count = _compute_overlap_with_last(query, last_executed_queries)
                if overlap_count >= OVERLAP_THRESHOLD:
                    reason_code = "overlap_with_previous_round"
                    break

        if reason_code:
            next_invalid_streak = invalid_streak + 1
            guard_action: Literal["run_tools", "reject_tools", "end_cycle"]
            if next_invalid_streak >= INVALID_STREAK_LIMIT:
                guard_action = "end_cycle"
            else:
                guard_action = "reject_tools"

            reject_tool_messages: List[ToolMessage] = []
            for tool_call in tool_calls:
                name = _tool_call_name(tool_call)
                query = _extract_search_queries_from_tool_call(tool_call)
                call_reason = reason_code if name == "search" else "round_rejected_due_to_search_guard"
                reject_content = _build_guard_reject_payload(
                    tool_name=name or "unknown",
                    query=query,
                    reason_code=call_reason,
                    invalid_streak=next_invalid_streak,
                    overlap_count=overlap_count,
                )
                reject_tool_messages.append(
                    ToolMessage(
                        content=reject_content,
                        name=name or None,
                        tool_call_id=_extract_tool_call_id(tool_call) or "",
                    )
                )

            updated_guard = {
                "last_executed_queries": last_executed_queries,
                "invalid_streak": next_invalid_streak,
                "force_end_cycle": guard_action == "end_cycle",
                "last_reason_code": reason_code,
            }
            return {
                "tool_input": reject_tool_messages,
                "search_guard": updated_guard,
                "guard_action": guard_action,
                "guard_reason_code": reason_code,
                "guard_invalid_streak": next_invalid_streak,
                "guard_overlap_count": overlap_count,
            }

        sanitized_ai_message = _sanitize_ai_search_tool_calls(latest_message)
        sanitized_tool_calls = getattr(sanitized_ai_message, "tool_calls", None) or []
        executed_queries: List[str] = []
        for tool_call in sanitized_tool_calls:
            if _tool_call_name(tool_call) == "search":
                executed_queries.extend(_extract_search_queries_from_tool_call(tool_call))

        updated_guard = {
            "last_executed_queries": executed_queries if executed_queries else last_executed_queries,
            "invalid_streak": 0,
            "force_end_cycle": False,
            "last_reason_code": "",
        }
        return {
            "tool_input": [sanitized_ai_message],
            "search_guard": updated_guard,
            "guard_action": "run_tools",
            "guard_reason_code": "",
            "guard_invalid_streak": 0,
            "guard_overlap_count": 0,
        }

    def route_tools_prep(state: WebCycleResearchState) -> Literal["run_tools", "reject_tools", "end_cycle"]:
        action = state.get("guard_action")
        if action in {"run_tools", "reject_tools", "end_cycle"}:
            return action
        return "run_tools"

    def route_after_tools_merge(state: WebCycleResearchState) -> Literal["llm", "end_cycle"]:
        guard = state.get("search_guard")
        if isinstance(guard, dict) and bool(guard.get("force_end_cycle")):
            return "end_cycle"
        return "llm"

    def merge_tool_output(state: WebCycleResearchState) -> Dict[str, Any]:
        tool_msgs = state.get("tool_input", [])
        if not tool_msgs:
            return {"tool_input": []}
        new_messages = list(state["messages"]) + list(tool_msgs)
        trimmed_messages, trim_stats = _trim_tool_history(
            new_messages,
            keep_last_turns=domain_cfg.max_tool_turns_in_context,
        )
        return {
            "messages": trimmed_messages,
            "tool_input": [],
            "context_trim": trim_stats,
            "context_trim_stats": trim_stats,
        }

    def final(state: WebCycleResearchState) -> Dict[str, Any]:
        current_output = ""
        messages = state.get("messages")
        if isinstance(messages, list) and messages:
            current_output = _message_to_text(messages[-1])
            current_boxed = _extract_boxed_from_ai_message(messages[-1])
            if current_boxed:
                return {"output": current_output}

        recovered_boxed = _extract_latest_boxed_from_cycle_histories(state.get("cycle_histories"))
        if recovered_boxed:
            return {"output": f"\\boxed{{{recovered_boxed}}}"}
        return {"output": current_output}

    g.add_node("init_graph", init_graph)
    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node)
    g.add_node("start_cycle", start_cycle)
    g.add_node("tools_prep", prep_tools)
    g.add_node("end_cycle", end_cycle)
    g.add_node("tools_merge", merge_tool_output)
    g.add_node("final", final)

    g.set_entry_point("init_graph")
    g.add_edge("init_graph", "start_cycle")
    g.add_edge("start_cycle", "llm")
    g.add_conditional_edges("llm", decide, {"tools": "tools_prep", "end_cycle": "end_cycle"})
    g.add_conditional_edges(
        "tools_prep",
        route_tools_prep,
        {"run_tools": "tools", "reject_tools": "tools_merge", "end_cycle": "tools_merge"},
    )
    g.add_edge("tools", "tools_merge")
    g.add_conditional_edges("tools_merge", route_after_tools_merge, {"llm": "llm", "end_cycle": "end_cycle"})
    g.add_conditional_edges("end_cycle", cycle_check, {"final": "final", "start_cycle": "start_cycle"})
    g.add_edge("final", END)

    return g
