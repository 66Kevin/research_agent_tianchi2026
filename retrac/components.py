from __future__ import annotations
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal
import logging
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from openai import BadRequestError
from langchain_core.messages.tool import ToolCall
from .config import ModelConfig

try:
    from langchain_core.messages import message_chunk_to_message
except ImportError:  # pragma: no cover
    message_chunk_to_message = None

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", 5))
RETRY_INTERVAL = int(os.getenv("LLM_RETRY_INTERVAL", 10))
LLM_QPS_LIMIT = int(os.getenv("LLM_QPS_LIMIT", 40))
TimeoutFallbackMode = Literal[
    "after_primary_timeout_retries",
    "immediate_on_primary_timeout",
]


# QPS limiter
_async_qps_limit_states = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _is_data_inspection_failed(exc: Exception, msg: str) -> bool:
    lowered = msg.lower()
    markers = (
        "data_inspection_failed",
        "datainspectionfailed",
        "input text data may contain inappropriate content",
        "output data may contain inappropriate content",
        "inappropriate content",
    )
    if any(marker in lowered for marker in markers):
        return True
    if isinstance(exc, BadRequestError):
        return "inspection" in lowered
    return False


def _is_stream_unsupported_error(exc: Exception) -> bool:
    if isinstance(exc, (NotImplementedError, AttributeError)):
        return True
    msg = str(exc).lower()
    return (
        "stream" in msg
        and ("not support" in msg or "unsupported" in msg or "unavailable" in msg)
    )


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk

    chunk_text = getattr(chunk, "text", None)
    if isinstance(chunk_text, str) and chunk_text:
        return chunk_text

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text_part = item.get("text")
            if isinstance(text_part, str) and text_part:
                parts.append(text_part)
        return "".join(parts)
    return ""


def _stream_result_to_ai_message(aggregated_chunk: Any) -> AIMessage:
    if isinstance(aggregated_chunk, AIMessage):
        return aggregated_chunk

    converted = None
    if message_chunk_to_message is not None:
        converted = message_chunk_to_message(aggregated_chunk)
    if isinstance(converted, AIMessage):
        return converted

    return AIMessage(
        content=getattr(aggregated_chunk, "content", ""),
        tool_calls=getattr(aggregated_chunk, "tool_calls", []) or [],
        invalid_tool_calls=getattr(aggregated_chunk, "invalid_tool_calls", []) or [],
        id=getattr(aggregated_chunk, "id", None),
        additional_kwargs=getattr(aggregated_chunk, "additional_kwargs", {}) or {},
        response_metadata=getattr(aggregated_chunk, "response_metadata", {}) or {},
        usage_metadata=getattr(aggregated_chunk, "usage_metadata", None) or {},
    )


async def _invoke_with_stream(
    llm_client: Any,
    msgs: list[BaseMessage],
    idle_timeout_s: int | None,
    stream_token_output: bool,
) -> AIMessage:
    stream_iter = llm_client.astream(msgs).__aiter__()
    aggregated_chunk: Any = None
    emitted_any_text = False

    try:
        while True:
            try:
                if idle_timeout_s and idle_timeout_s > 0:
                    chunk = await asyncio.wait_for(anext(stream_iter), timeout=idle_timeout_s)
                else:
                    chunk = await anext(stream_iter)
            except StopAsyncIteration:
                break

            aggregated_chunk = chunk if aggregated_chunk is None else (aggregated_chunk + chunk)
            if not stream_token_output:
                continue
            delta = _chunk_text(chunk)
            if delta:
                print(delta, end="", flush=True)
                emitted_any_text = True
    finally:
        aclose = getattr(stream_iter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass
        if stream_token_output and emitted_any_text:
            print("", flush=True)

    if aggregated_chunk is None:
        raise RuntimeError("LLM stream returned no chunks")
    return _stream_result_to_ai_message(aggregated_chunk)

def async_qps_limiter(qps: int = LLM_QPS_LIMIT):
    """Async QPS limiter decorator"""
    def decorator(func):
        state_key = f"async_qps_limiter_{id(func)}"
        
        async def wrapper(*args, **kwargs):
            if state_key not in _async_qps_limit_states:
                _async_qps_limit_states[state_key] = {
                    "lock": asyncio.Lock(),
                    "call_times": [],
                    "qps": qps
                }
            
            state = _async_qps_limit_states[state_key]
            async with state["lock"]:
                current_time = time.monotonic()
                state["call_times"] = [
                    call_time for call_time in state["call_times"]
                    if current_time - call_time < 1.0
                ]
                
                if len(state["call_times"]) >= state["qps"]:
                    oldest_call = min(state["call_times"])
                    wait_time = 1.0 - (current_time - oldest_call)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        current_time = time.monotonic()
                        state["call_times"] = [
                            call_time for call_time in state["call_times"]
                            if current_time - call_time < 1.0
                        ]
                
                state["call_times"].append(current_time)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def _resolve_invoke_kwargs(model_cfg: ModelConfig, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    invoke_kwargs = dict(base_kwargs)
    merged_extra_body: Dict[str, Any] = {}
    if model_cfg.extra_body:
        merged_extra_body.update(model_cfg.extra_body)
    extra_body_override = invoke_kwargs.get("extra_body")
    if isinstance(extra_body_override, dict):
        merged_extra_body.update(extra_body_override)
    if merged_extra_body:
        invoke_kwargs["extra_body"] = merged_extra_body
    else:
        invoke_kwargs.pop("extra_body", None)
    return invoke_kwargs


def _build_llm_client(
    model_cfg: ModelConfig,
    tools: list[Any],
    invoke_kwargs: Dict[str, Any],
):
    use_responses_api = (model_cfg.api_type == "responses")
    llm = ChatOpenAI(
        model=model_cfg.model_name,
        base_url=model_cfg.base_url,
        api_key=model_cfg.api_key,
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        timeout=model_cfg.timeout_s,
        use_responses_api=use_responses_api,
        use_previous_response_id=use_responses_api,
        **invoke_kwargs,
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


def create_llm_node(
    model_cfg: ModelConfig,
    key: str = "messages",
    fallback_model_cfg: ModelConfig | None = None,
    timeout_retry_attempts: int | None = None,
    timeout_fallback_mode: TimeoutFallbackMode = "after_primary_timeout_retries",
    stream_token_output: bool = False,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:

    tools = model_cfg.tools

    async def _llm_ainvoke(
        state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if key not in state:
            raise KeyError(f"State must contain '{key}' field")
        if not state.get(key):
            raise ValueError(f"State '{key}' field cannot be empty")

        msgs = state[key]
        if any(not isinstance(m, BaseMessage) for m in msgs):
            bad = next(type(m).__name__ for m in msgs if not isinstance(m, BaseMessage))
            raise TypeError(f"state['{key}'] must be List[BaseMessage], found {bad}")

        if timeout_fallback_mode not in (
            "after_primary_timeout_retries",
            "immediate_on_primary_timeout",
        ):
            raise ValueError(
                "timeout_fallback_mode must be one of "
                "['after_primary_timeout_retries', 'immediate_on_primary_timeout']"
            )

        timeout_retry_limit = (
            timeout_retry_attempts
            if isinstance(timeout_retry_attempts, int) and timeout_retry_attempts > 0
            else RETRY_ATTEMPTS
        )
        primary_timeout_retry_limit = (
            1 if timeout_fallback_mode == "immediate_on_primary_timeout" else timeout_retry_limit
        )

        primary_invoke_kwargs = _resolve_invoke_kwargs(model_cfg, kwargs)
        primary_llm = _build_llm_client(model_cfg=model_cfg, tools=tools, invoke_kwargs=primary_invoke_kwargs)
        fallback_llm = None
        if fallback_model_cfg is not None:
            fallback_invoke_kwargs = _resolve_invoke_kwargs(fallback_model_cfg, kwargs)
            fallback_llm = _build_llm_client(
                model_cfg=fallback_model_cfg,
                tools=tools,
                invoke_kwargs=fallback_invoke_kwargs,
            )

        state_error = state.get("error", [])
        error = list(state_error) if isinstance(state_error, list) else []
        llm_attempts: list[Dict[str, Any]] = []

        async def _invoke_stage(
            llm_client: Any,
            stage_model_cfg: ModelConfig,
            model_role: Literal["primary", "fallback"],
            timeout_limit: int,
        ) -> Dict[str, Any]:
            timeout_failures = 0
            non_timeout_failures = 0
            use_stream = True

            while True:
                attempt_no = len(llm_attempts) + 1
                attempt_started_at = _utc_now_iso()
                attempt_start = time.perf_counter()
                try:
                    if use_stream:
                        resp = await _invoke_with_stream(
                            llm_client=llm_client,
                            msgs=msgs,
                            idle_timeout_s=stage_model_cfg.timeout_s,
                            stream_token_output=stream_token_output,
                        )
                    else:
                        if stage_model_cfg.timeout_s and stage_model_cfg.timeout_s > 0:
                            resp = await asyncio.wait_for(
                                llm_client.ainvoke(msgs),
                                timeout=stage_model_cfg.timeout_s,
                            )
                        else:
                            resp = await llm_client.ainvoke(msgs)
                    resp_text = resp.text or ""
                    logger.debug("LLM invocation successful, response length: %s", len(resp_text))

                    if "<tool_call>finish</tool_call>" in resp_text:
                        logger.error("Finish tool call found in response: %s", resp_text)
                        finish_tool_call: ToolCall = ToolCall(name="finish", args={}, id=None)
                        resp.tool_calls.append(finish_tool_call)
                    elif "</tool_call>" in resp_text:
                        logger.error("Tool call not parsed correctly, response: %s", resp_text, exc_info=True)
                        non_timeout_failures += 1
                        error.append("Tool call not parsed correctly")
                        llm_attempts.append(
                            {
                                "attempt": attempt_no,
                                "status": "invalid_tool_call_format",
                                "started_at": attempt_started_at,
                                "ended_at": _utc_now_iso(),
                                "duration_ms": int((time.perf_counter() - attempt_start) * 1000),
                                "exception_type": "ToolCallParseError",
                                "exception": "Tool call not parsed correctly",
                                "model_role": model_role,
                                "model_name": stage_model_cfg.model_name,
                                "error_kind": "non_timeout",
                            }
                        )
                        if non_timeout_failures >= RETRY_ATTEMPTS:
                            return {"exhausted_kind": "non_timeout"}
                        await asyncio.sleep(RETRY_INTERVAL)
                        continue

                    llm_attempts.append(
                        {
                            "attempt": attempt_no,
                            "status": "success",
                            "started_at": attempt_started_at,
                            "ended_at": _utc_now_iso(),
                            "duration_ms": int((time.perf_counter() - attempt_start) * 1000),
                            "tool_call_count": len(getattr(resp, "tool_calls", []) or []),
                            "model_role": model_role,
                            "model_name": stage_model_cfg.model_name,
                            "error_kind": "",
                        }
                    )
                    return {"messages": msgs + [resp]}
                except Exception as e:
                    if use_stream and _is_stream_unsupported_error(e):
                        use_stream = False
                        logger.warning(
                            "Model backend does not support token streaming cleanly; "
                            "falling back to non-stream invoke for %s.",
                            stage_model_cfg.model_name,
                        )
                        continue

                    is_timeout = isinstance(e, asyncio.TimeoutError)
                    if is_timeout:
                        timeout_failures += 1
                        failure_idx = timeout_failures
                        failure_limit = timeout_limit
                        msg = f"LLM request timed out after {stage_model_cfg.timeout_s}s"
                        error_kind = "timeout"
                    else:
                        non_timeout_failures += 1
                        failure_idx = non_timeout_failures
                        failure_limit = RETRY_ATTEMPTS
                        msg = str(e)
                        error_kind = "non_timeout"

                    logger.error("LLM invocation failed: %s", msg, exc_info=True)
                    llm_attempts.append(
                        {
                            "attempt": attempt_no,
                            "status": "error",
                            "started_at": attempt_started_at,
                            "ended_at": _utc_now_iso(),
                            "duration_ms": int((time.perf_counter() - attempt_start) * 1000),
                            "exception_type": type(e).__name__,
                            "exception": msg,
                            "model_role": model_role,
                            "model_name": stage_model_cfg.model_name,
                            "error_kind": error_kind,
                        }
                    )

                    if _is_data_inspection_failed(e, msg):
                        non_retry_msg = (
                            f"LLM non-retryable error [data_inspection_failed] on attempt "
                            f"[{failure_idx}/{failure_limit}] : {msg}"
                        )
                        logger.warning(non_retry_msg)
                        error.append(non_retry_msg)
                        return {
                            "terminal": True,
                            "llm_error_type": "data_inspection_failed",
                        }

                    error.append(f"LLM invocation failed on attempt [{failure_idx}/{failure_limit}] : {msg}")
                    if isinstance(e, BadRequestError) and "context length" in msg.lower():
                        return {"terminal": True}
                    if failure_idx >= failure_limit:
                        return {"exhausted_kind": error_kind}
                    await asyncio.sleep(RETRY_INTERVAL)

        primary_result = await _invoke_stage(
            llm_client=primary_llm,
            stage_model_cfg=model_cfg,
            model_role="primary",
            timeout_limit=primary_timeout_retry_limit,
        )
        if primary_result.get("messages"):
            return {key: primary_result["messages"], "llm_attempts": llm_attempts}
        if primary_result.get("llm_error_type"):
            return {
                "error": error,
                "llm_error_type": primary_result["llm_error_type"],
                "llm_attempts": llm_attempts,
            }
        if primary_result.get("terminal"):
            return {"error": error, "llm_attempts": llm_attempts}

        should_switch_to_fallback = (
            fallback_llm is not None
            and fallback_model_cfg is not None
            and primary_result.get("exhausted_kind") == "timeout"
        )
        if not should_switch_to_fallback:
            return {"error": error, "llm_attempts": llm_attempts}

        switch_timestamp = _utc_now_iso()
        llm_attempts.append(
            {
                "attempt": len(llm_attempts) + 1,
                "status": "switch",
                "started_at": switch_timestamp,
                "ended_at": switch_timestamp,
                "duration_ms": 0,
                "model_role": "fallback",
                "model_name": fallback_model_cfg.model_name,
                "error_kind": "timeout",
                "switch_event": "switch_to_fallback",
                "from_model_name": model_cfg.model_name,
                "to_model_name": fallback_model_cfg.model_name,
            }
        )

        fallback_result = await _invoke_stage(
            llm_client=fallback_llm,
            stage_model_cfg=fallback_model_cfg,
            model_role="fallback",
            timeout_limit=timeout_retry_limit,
        )
        if fallback_result.get("messages"):
            return {key: fallback_result["messages"], "llm_attempts": llm_attempts}
        if fallback_result.get("llm_error_type"):
            return {
                "error": error,
                "llm_error_type": fallback_result["llm_error_type"],
                "llm_attempts": llm_attempts,
            }
        return {"error": error, "llm_attempts": llm_attempts}

    async def llm_ainvoke(
        state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await _llm_ainvoke(state, **kwargs)

    @async_qps_limiter(qps=LLM_QPS_LIMIT)
    async def qps_limited_llm_ainvoke(
        state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await _llm_ainvoke(state, **kwargs)

    return llm_ainvoke if not model_cfg.enable_qps_limit else qps_limited_llm_ainvoke
