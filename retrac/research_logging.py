from __future__ import annotations

from collections import Counter
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage


MAX_CONTENT_CHARS = 8000
SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.IGNORECASE)
SK_VALUE_RE = re.compile(r"sk-[A-Za-z0-9\-_]{8,}")
BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9\.\-_]+", re.IGNORECASE)
UNMASKED_TOKEN_USAGE_KEYS = {"input_tokens", "output_tokens", "total_tokens"}


def create_run_directory(base_dir: str = "retrac/logs") -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    run_prefix = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_path / run_prefix
    suffix = 1
    while run_dir.exists():
        run_dir = base_path / f"{run_prefix}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _truncate_text(text: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}...[truncated {omitted} chars]"


def _mask_text(text: str) -> str:
    masked = SK_VALUE_RE.sub("sk-***", text)
    masked = BEARER_RE.sub("Bearer ***", masked)
    return masked


def _mask_value(value: str) -> str:
    value = _mask_text(str(value))
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-4:]}"


def _should_mask_key(key: str | None) -> bool:
    if not key:
        return False
    lowered = key.lower()
    if lowered in UNMASKED_TOKEN_USAGE_KEYS:
        return False
    return bool(SENSITIVE_KEY_RE.search(key))


def _sanitize(value: Any, parent_key: str | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        text = _mask_text(value)
        if _should_mask_key(parent_key):
            text = _mask_value(text)
        return _truncate_text(text)
    if isinstance(value, BaseMessage):
        return _serialize_message(value)
    if isinstance(value, list):
        return [_sanitize(v, parent_key=parent_key) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v, parent_key=parent_key) for v in value]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _should_mask_key(key):
                out[key] = _mask_value(str(v))
            else:
                out[key] = _sanitize(v, parent_key=key)
        return out
    return _truncate_text(_mask_text(str(value)))


def _message_text(message: BaseMessage) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for chunk in content:
            if isinstance(chunk, dict):
                chunk_text = chunk.get("text")
                if isinstance(chunk_text, str):
                    chunks.append(chunk_text)
                else:
                    chunks.append(str(chunk))
            else:
                chunks.append(str(chunk))
        return "\n".join(chunks)
    return str(content)


def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": type(message).__name__,
        "role": getattr(message, "type", None),
        "content": _sanitize(getattr(message, "content", None)),
    }

    for attr in (
        "name",
        "id",
        "tool_call_id",
        "tool_calls",
        "invalid_tool_calls",
        "usage_metadata",
        "response_metadata",
        "additional_kwargs",
    ):
        if hasattr(message, attr):
            value = getattr(message, attr)
            if value not in (None, "", [], {}):
                payload[attr] = _sanitize(value, parent_key=attr)

    return payload


def _extract_boxed_answer(text: str | None) -> str:
    if not isinstance(text, str) or not text:
        return ""
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else ""


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


def _extract_latest_boxed_from_cycle_histories(final_state: Dict[str, Any]) -> str:
    cycle_histories = final_state.get("cycle_histories")
    if not isinstance(cycle_histories, list):
        return ""
    for cycle_history in reversed(cycle_histories):
        if not isinstance(cycle_history, list):
            continue
        for message in reversed(cycle_history):
            if not _is_ai_like_message(message):
                continue
            if isinstance(message, BaseMessage):
                text = _message_text(message)
            elif isinstance(message, dict):
                content = message.get("content", "")
                text = content if isinstance(content, str) else str(content)
            else:
                text = str(getattr(message, "content", message))
            boxed = _extract_boxed_answer(text)
            if boxed:
                return boxed
    return ""


def _parse_structured_tool_payload(content_text: str) -> Dict[str, Any] | None:
    try:
        payload = json.loads(content_text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("tool"), str):
        return None
    calls = payload.get("calls")
    if calls is not None and not isinstance(calls, list):
        return None
    return payload


def _build_llm_content_preview(message: BaseMessage) -> str:
    text = _message_text(message).strip()
    if text:
        return _truncate_text(_sanitize(text), 300)

    tool_calls = getattr(message, "tool_calls", None) or []
    if isinstance(tool_calls, list) and tool_calls:
        compact_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                compact_calls.append(str(tool_call))
                continue
            compact_calls.append(
                {
                    "name": tool_call.get("name"),
                    "id": tool_call.get("id"),
                    "args": tool_call.get("args"),
                }
            )
        return _truncate_text(
            _sanitize(f"tool_calls={json.dumps(compact_calls, ensure_ascii=False)}"),
            300,
        )

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None) or []
    if isinstance(invalid_tool_calls, list) and invalid_tool_calls:
        return _truncate_text(_sanitize(f"invalid_tool_calls={invalid_tool_calls}"), 300)

    response_metadata = getattr(message, "response_metadata", None) or {}
    finish_reason = ""
    if isinstance(response_metadata, dict):
        finish_reason = str(response_metadata.get("finish_reason", "") or "")
    if finish_reason:
        return _truncate_text(_sanitize(f"finish_reason={finish_reason}"), 300)

    return "[empty_ai_message]"


class ResearchRunLogger:
    def __init__(
        self,
        run_dir: Path,
        task_id: int,
        question: str,
        cfg: dict,
        task_file_name: str | None = None,
        console_log: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.task_id = task_id
        self.cfg = cfg
        self.task_file_name = task_file_name
        self.console_log = console_log
        self.started_at = datetime.now()
        timestamp = self.started_at.strftime("%Y-%m-%d-%H-%M-%S")

        self.events_path = self.run_dir / f"task_{task_id}_{timestamp}.events.jsonl"
        self.final_path = self.run_dir / f"task_{task_id}_{timestamp}.json"

        self.cycle_index = 0

        self.payload: Dict[str, Any] = {
            "status": "running",
            "start_time": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": "",
            "task_id": task_id,
            "input": {
                "task_description": _sanitize(question),
                "task_file_name": task_file_name,
            },
            "ground_truth": None,
            "final_boxed_answer": "",
            "final_judge_result": "",
            "judge_type": "",
            "eval_details": None,
            "error": "",
            "current_main_turn_id": 0,
            "current_sub_agent_turn_id": 0,
            "sub_agent_counter": 0,
            "current_sub_agent_session_id": None,
            "env_info": self._build_env_info(),
            "log_dir": str(self.run_dir),
            "main_agent_message_history": {
                "system_prompt": _sanitize(self.cfg.get("system_prompt", "")),
                "message_history": [],
            },
            "sub_agent_message_history_sessions": {},
            "step_logs": [],
            "trace_data": {
                "llm_calls": [],
                "tool_calls": [],
                "cycle_transitions": [],
            },
        }

        self.log_step(
            step_name="Main | Task Start",
            message=f"Task {task_id} started.",
            info_level="info",
            metadata={"task_file_name": task_file_name},
        )

    def _build_env_info(self) -> Dict[str, Any]:
        model_cfg_raw = self.cfg.get("model", {}) if isinstance(self.cfg, dict) else {}
        model_cfg = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
        fallback_model_cfg_raw = self.cfg.get("fallback_model", {}) if isinstance(self.cfg, dict) else {}
        fallback_model_cfg = (
            fallback_model_cfg_raw if isinstance(fallback_model_cfg_raw, dict) else {}
        )
        return {
            "llm_base_url": _sanitize(model_cfg.get("base_url") or os.getenv("OPENAI_API_BASE")),
            "llm_model_name": _sanitize(model_cfg.get("model_name")),
            "llm_temperature": model_cfg.get("temperature"),
            "llm_top_p": model_cfg.get("top_p"),
            "llm_timeout_s": model_cfg.get("timeout_s"),
            "fallback_llm_base_url": _sanitize(
                fallback_model_cfg.get("base_url") or os.getenv("OPENAI_API_BASE")
            ),
            "fallback_llm_model_name": _sanitize(fallback_model_cfg.get("model_name")),
            "max_cycles": self.cfg.get("max_cycles"),
            "max_turns": self.cfg.get("max_turns"),
            "has_serper_api_key": bool(os.getenv("SERPER_API_KEY")),
            "has_jina_api_key": bool(os.getenv("JINA_API_KEY")),
            "has_openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "has_summary_llm_api_key": bool(os.getenv("API_KEY_FOR_VISIT_SUMMARIZE")),
            "summary_llm_base_url": _sanitize(os.getenv("BASE_URL_FOR_VISIT_SUMMARIZE")),
        }

    def _write_event(self, event: Dict[str, Any]) -> None:
        safe_event = _sanitize(event)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_event, ensure_ascii=False))
            f.write("\n")

    def log_step(
        self,
        step_name: str,
        message: str,
        info_level: str = "info",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        step = {
            "step_name": step_name,
            "message": _sanitize(message),
            "timestamp": _now_str(),
            "info_level": info_level,
            "metadata": _sanitize(metadata or {}),
        }
        self.payload["step_logs"].append(step)
        self._write_event({"event_type": "step", **step})
        if self.console_log:
            line = (
                f"[{step['timestamp']}] "
                f"[task {self.task_id}] "
                f"[{str(step['info_level']).upper()}] "
                f"{step['step_name']} | {step['message']}"
            )
            if step["metadata"]:
                line += f" | metadata={json.dumps(step['metadata'], ensure_ascii=False)}"
            print(line, flush=True)

    def log_graph_event(
        self,
        node_name: str,
        node_output: Any,
        merged_state: Dict[str, Any],
    ) -> None:
        self._write_event(
            {
                "event_type": "graph_node",
                "timestamp": _now_str(),
                "node_name": node_name,
                "node_output": _sanitize(node_output),
            }
        )

        if node_name == "init_graph":
            self.log_step("Main | Init Graph", "Graph initialized.")
            return

        if node_name == "start_cycle":
            self.cycle_index += 1
            self.payload["current_main_turn_id"] = self.cycle_index
            self.payload["trace_data"]["cycle_transitions"].append(
                {
                    "cycle_id": self.cycle_index,
                    "stage": "start",
                    "timestamp": _now_str(),
                }
            )
            self.log_step("Cycle | Start", f"Cycle {self.cycle_index} started.")
            return

        if node_name == "llm":
            self._record_llm_event(node_output, merged_state)
            return

        if node_name == "tools_prep":
            self._record_tool_requests(node_output, merged_state)
            return

        if node_name == "tools":
            self._record_tool_results(node_output)
            return

        if node_name == "tools_merge":
            metadata: Dict[str, Any] = {}
            if isinstance(node_output, dict):
                trim_stats = node_output.get("context_trim")
                if not isinstance(trim_stats, dict):
                    trim_stats = node_output.get("context_trim_stats")
                if isinstance(trim_stats, dict):
                    metadata["context_trim"] = trim_stats
            self.log_step(
                "Tools | Merge",
                "Tool outputs merged into messages.",
                metadata=metadata,
            )
            return

        if node_name == "end_cycle":
            self.payload["trace_data"]["cycle_transitions"].append(
                {
                    "cycle_id": self.cycle_index,
                    "stage": "end",
                    "timestamp": _now_str(),
                }
            )
            self.log_step("Cycle | End", f"Cycle {self.cycle_index} ended.")
            return

        if node_name == "final":
            output = ""
            if isinstance(node_output, dict):
                output = str(node_output.get("output", ""))
            self.log_step(
                "Main | Final Output",
                "Final output generated.",
                metadata={"output_preview": _truncate_text(_sanitize(output))},
            )

    def _record_llm_event(self, node_output: Any, merged_state: Dict[str, Any]) -> None:
        def _latest_ai_message(messages: Any) -> BaseMessage | None:
            if not isinstance(messages, list):
                return None
            for message in reversed(messages):
                if isinstance(message, BaseMessage) and _is_ai_like_message(message):
                    return message
            return None

        node_messages = None
        node_errors = None
        if isinstance(node_output, dict):
            node_messages = node_output.get("messages")
            node_errors = node_output.get("error")

        latest_ai = _latest_ai_message(node_messages)
        if latest_ai is None and isinstance(node_errors, list) and node_errors:
            last_error = str(node_errors[-1])
            self.payload["trace_data"]["llm_calls"].append(
                {
                    "timestamp": _now_str(),
                    "usage_metadata": {},
                    "tool_call_count": 0,
                    "content_preview": _truncate_text(_sanitize(last_error), 300),
                    "status": "error",
                    "error_count": len(node_errors),
                    "last_error": _truncate_text(_sanitize(last_error), 300),
                }
            )
            self.log_step(
                "LLM | Invoke Failed",
                "LLM node failed without AI response.",
                "warning",
                metadata={
                    "error_count": len(node_errors),
                    "last_error": _truncate_text(_sanitize(last_error), 300),
                },
            )
            return

        if latest_ai is None:
            latest_ai = _latest_ai_message(merged_state.get("messages", []))
        if latest_ai is None:
            self.log_step("LLM | Invoke", "LLM node finished without AIMessage.", "warning")
            return

        usage_metadata = _sanitize(getattr(latest_ai, "usage_metadata", {}) or {})
        tool_calls = _sanitize(getattr(latest_ai, "tool_calls", []) or [])
        self.payload["trace_data"]["llm_calls"].append(
            {
                "timestamp": _now_str(),
                "usage_metadata": usage_metadata,
                "tool_call_count": len(tool_calls),
                "content_preview": _build_llm_content_preview(latest_ai),
                "status": "success",
            }
        )
        self.log_step(
            "LLM | Invoke",
            f"LLM responded with {len(tool_calls)} tool call(s).",
            metadata={"usage_metadata": usage_metadata},
        )

    def _record_tool_requests(self, node_output: Any, merged_state: Dict[str, Any]) -> None:
        requested_calls = []
        messages = []
        if isinstance(node_output, dict):
            messages = node_output.get("tool_input", [])
        if not messages:
            state_messages = merged_state.get("messages") or []
            if state_messages:
                messages = [state_messages[-1]]

        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    requested_calls.append(
                        {
                            "timestamp": _now_str(),
                            "stage": "request",
                            "tool_name": _sanitize(tool_call.get("name")),
                            "tool_call_id": _sanitize(tool_call.get("id")),
                            "args": _sanitize(tool_call.get("args")),
                        }
                    )
        if requested_calls:
            self.payload["trace_data"]["tool_calls"].extend(requested_calls)
            self.log_step(
                "Tools | Prepare",
                f"Prepared {len(requested_calls)} tool call(s).",
                metadata={"tool_names": [c["tool_name"] for c in requested_calls]},
            )
        else:
            self.log_step("Tools | Prepare", "No tool calls found in tools_prep.", "warning")

    def _record_tool_results(self, node_output: Any) -> None:
        tool_messages = []
        if isinstance(node_output, dict):
            tool_messages = node_output.get("tool_input", []) or []

        results = []
        node_status_counts: Counter[str] = Counter()
        node_http_statuses: set[int] = set()
        node_tool_names: set[str] = set()
        for message in tool_messages:
            if isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", None)
                if not tool_name:
                    tool_name = (message.additional_kwargs or {}).get("name")
                tool_call_id = getattr(message, "tool_call_id", None)
                content_text = _message_text(message)
                structured_payload = _parse_structured_tool_payload(content_text)
                if structured_payload:
                    parsed_tool_name = structured_payload.get("tool") or tool_name or "unknown"
                    node_tool_names.add(str(parsed_tool_name))
                    calls = structured_payload.get("calls") or []
                    call_status_counts: Counter[str] = Counter()
                    call_http_statuses: set[int] = set()
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        status_name = str(call.get("status", "") or "")
                        if status_name:
                            node_status_counts[status_name] += 1
                            call_status_counts[status_name] += 1
                        http_status = call.get("http_status")
                        if isinstance(http_status, int):
                            node_http_statuses.add(http_status)
                            call_http_statuses.add(http_status)
                        elapsed_ms = call.get("elapsed_ms")
                        results.append(
                            {
                                "timestamp": _now_str(),
                                "stage": "result_detail",
                                "tool_name": _sanitize(parsed_tool_name),
                                "tool_call_id": _sanitize(tool_call_id),
                                "ok": bool(call.get("ok")),
                                "http_status": http_status if isinstance(http_status, int) else None,
                                "status": _sanitize(status_name),
                                "elapsed_ms": elapsed_ms if isinstance(elapsed_ms, int) else None,
                                "error_type": _sanitize(str(call.get("error_type", "") or "")),
                                "error_message": _truncate_text(_sanitize(str(call.get("error_message", "") or "")), 500),
                                "target": _sanitize(str(call.get("target", "") or "")),
                            }
                        )

                    summarize_status = structured_payload.get("summarize_status")
                    if isinstance(summarize_status, dict):
                        summarize_status_name = str(summarize_status.get("status", "") or "")
                        summarize_http_status = summarize_status.get("http_status")
                        if summarize_status_name:
                            node_status_counts[summarize_status_name] += 1
                            call_status_counts[summarize_status_name] += 1
                        if isinstance(summarize_http_status, int):
                            node_http_statuses.add(summarize_http_status)
                            call_http_statuses.add(summarize_http_status)
                        summarize_elapsed_ms = summarize_status.get("elapsed_ms")
                        results.append(
                            {
                                "timestamp": _now_str(),
                                "stage": "result_detail",
                                "tool_name": _sanitize(parsed_tool_name),
                                "tool_call_id": _sanitize(tool_call_id),
                                "ok": bool(summarize_status.get("ok")),
                                "http_status": summarize_http_status if isinstance(summarize_http_status, int) else None,
                                "status": _sanitize(summarize_status_name),
                                "elapsed_ms": summarize_elapsed_ms if isinstance(summarize_elapsed_ms, int) else None,
                                "error_type": _sanitize(str(summarize_status.get("error_type", "") or "")),
                                "error_message": _truncate_text(_sanitize(str(summarize_status.get("error_message", "") or "")), 500),
                                "target": _sanitize(str(summarize_status.get("target", "") or "")),
                                "phase": "summarize",
                            }
                        )

                    results.append(
                        {
                            "timestamp": _now_str(),
                            "stage": "result_summary",
                            "tool_name": _sanitize(parsed_tool_name),
                            "tool_call_id": _sanitize(tool_call_id),
                            "ok": bool(structured_payload.get("ok")),
                            "status_counts": _sanitize(dict(call_status_counts)),
                            "http_statuses": sorted(call_http_statuses),
                            "content_preview": _truncate_text(_sanitize(content_text), 500),
                        }
                    )
                    continue

                node_tool_names.add(str(tool_name or "unknown"))
                results.append(
                    {
                        "timestamp": _now_str(),
                        "stage": "result",
                        "tool_name": _sanitize(tool_name),
                        "tool_call_id": _sanitize(tool_call_id),
                        "content_preview": _truncate_text(_sanitize(content_text), 500),
                    }
                )
            else:
                results.append(
                    {
                        "timestamp": _now_str(),
                        "stage": "result",
                        "tool_name": "unknown",
                        "tool_call_id": "",
                        "content_preview": _truncate_text(_sanitize(str(message)), 500),
                    }
                )

        if results:
            self.payload["trace_data"]["tool_calls"].extend(results)
            self.log_step(
                "Tools | Execute",
                f"Received {len(results)} tool result message(s).",
                metadata={
                    "tool_names": sorted(node_tool_names) if node_tool_names else [r["tool_name"] for r in results],
                    "status_counts": _sanitize(dict(node_status_counts)),
                    "http_statuses": sorted(node_http_statuses),
                },
            )
        else:
            self.log_step("Tools | Execute", "Tools node returned no tool messages.", "warning")

    def _update_message_history(self, final_state: Dict[str, Any]) -> None:
        messages = final_state.get("messages", [])
        serialized_messages = []
        system_prompt = self.payload["main_agent_message_history"].get("system_prompt", "")

        for message in messages:
            if isinstance(message, BaseMessage):
                if isinstance(message, SystemMessage) and not system_prompt:
                    system_prompt = _sanitize(_message_text(message))
                serialized_messages.append(_serialize_message(message))
            else:
                serialized_messages.append(_sanitize(message))

        self.payload["main_agent_message_history"]["system_prompt"] = system_prompt
        self.payload["main_agent_message_history"]["message_history"] = serialized_messages

    def _write_final_json(self) -> None:
        with self.final_path.open("w", encoding="utf-8") as f:
            json.dump(_sanitize(self.payload), f, ensure_ascii=False, indent=2)

    def finalize_success(self, final_state: Dict[str, Any]) -> None:
        output = final_state.get("output", "")
        output_text = output if isinstance(output, str) else str(output)
        boxed_answer = _extract_boxed_answer(output_text)
        if not boxed_answer:
            boxed_answer = _extract_latest_boxed_from_cycle_histories(final_state)
        self.payload["status"] = "success"
        self.payload["error"] = ""
        self.payload["end_time"] = _now_str()
        self.payload["final_boxed_answer"] = boxed_answer
        self._update_message_history(final_state)
        self.log_step(
            "Main | Task Completed",
            f"Task {self.task_id} completed successfully.",
            metadata={"final_boxed_answer": boxed_answer},
        )
        self._write_final_json()

    def finalize_error(self, exc: Exception | str, final_state: Dict[str, Any] | None = None) -> None:
        error_text = str(exc)
        self.payload["status"] = "failed"
        self.payload["error"] = _sanitize(error_text)
        self.payload["end_time"] = _now_str()
        if final_state:
            self._update_message_history(final_state)
        self.log_step(
            "Main | Task Failed",
            f"Task {self.task_id} failed: {error_text}",
            info_level="error",
        )
        self._write_final_json()
