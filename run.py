import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, convert_to_openai_messages

from retrac.components import create_llm_node
from retrac.config import ModelConfig
from retrac.graph import build_graph
from retrac.research_logging import ResearchRunLogger, create_run_directory

DEFAULT_TASK_FILE = "/data/dataset2/Workshop/wangyueyi/test/question.jsonl"
TASK_FORCE_END_CYCLE_S = 550.0
TASK_FINAL_FALLBACK_S = 597.0
TASK_HARD_TIMEOUT_S = 600.0
TOOL_NODE_WATCHDOG_S = 75.0


def extract_boxed_answer(text: str | None) -> str:
    if not text:
        return ""

    boxed_re = re.compile(r"\\boxed\b", re.DOTALL)

    last_result: str | None = None
    i = 0
    n = len(text)

    while True:
        m = boxed_re.search(text, i)
        if not m:
            break
        j = m.end()

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

    black_list = {"", "?", "??", "???", "？", "……", "…", "...", "unknown"}
    if last_result is None:
        return ""
    result = last_result.strip()
    return "" if result.lower() in black_list else result


def message_to_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        text = getattr(message, "text", None)
        if isinstance(text, str) and text:
            return text
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)
    return str(message)


def _extract_tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        tool_call_id = tool_call.get("id")
    else:
        tool_call_id = getattr(tool_call, "id", None)
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id
    return None


def normalize_messages_for_llm(messages: list[BaseMessage]) -> tuple[list[BaseMessage], dict[str, Any]]:
    normalized: list[BaseMessage] = []
    idx = 0
    orphan_tool_messages = 0
    truncated_unmatched_tool_calls = False

    while idx < len(messages):
        msg = messages[idx]

        if isinstance(msg, ToolMessage):
            orphan_tool_messages += 1
            idx += 1
            continue

        if isinstance(msg, AIMessage):
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            expected_ids = [_extract_tool_call_id(tc) for tc in raw_tool_calls]
            expected_ids = [tc_id for tc_id in expected_ids if tc_id]
            if raw_tool_calls:
                follow_idx = idx + 1
                follow_tool_messages: list[ToolMessage] = []
                while follow_idx < len(messages) and isinstance(messages[follow_idx], ToolMessage):
                    follow_tool_messages.append(messages[follow_idx])
                    follow_idx += 1

                responded_ids = {
                    getattr(tool_msg, "tool_call_id", None)
                    for tool_msg in follow_tool_messages
                    if isinstance(getattr(tool_msg, "tool_call_id", None), str)
                }
                if not expected_ids or not all(tc_id in responded_ids for tc_id in expected_ids):
                    truncated_unmatched_tool_calls = True
                    break

                normalized.append(msg)
                normalized.extend(follow_tool_messages)
                idx = follow_idx
                continue

        normalized.append(msg)
        idx += 1

    stats: dict[str, Any] = {
        "input_messages": len(messages),
        "normalized_messages": len(normalized),
        "orphan_tool_messages": orphan_tool_messages,
        "truncated_unmatched_tool_calls": truncated_unmatched_tool_calls,
    }
    return normalized, stats


def extract_last_completed_cycle_answer(state: Dict[str, Any]) -> str:
    cycle_histories = state.get("cycle_histories", [])
    if not isinstance(cycle_histories, list):
        return ""

    for cycle_history in reversed(cycle_histories):
        if not isinstance(cycle_history, list):
            continue
        for message in reversed(cycle_history):
            if message is None:
                continue
            answer = extract_boxed_answer(message_to_text(message))
            if answer:
                return answer
    return ""


def apply_timeout_fallback(state: Dict[str, Any]) -> str:
    fallback_answer = extract_last_completed_cycle_answer(state)
    state["output"] = f"\\boxed{{{fallback_answer}}}" if fallback_answer else ""
    state["timeout_fallback"] = {
        "applied": True,
        "answer": fallback_answer,
    }
    return fallback_answer


async def run_forced_end_cycle_summary(
    cfg: dict,
    question: str,
    state: Dict[str, Any],
    timeout_s: float,
) -> tuple[bool, str, dict[str, Any]]:
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, "", {
            "input_messages": 0,
            "normalized_messages": 0,
            "orphan_tool_messages": 0,
            "truncated_unmatched_tool_calls": False,
        }

    final_summary_prompt = cfg.get("final_summary_prompt")
    summary_prompt = cfg.get("summary_prompt")
    if isinstance(final_summary_prompt, str) and final_summary_prompt.strip():
        summary_prompt_tmpl = final_summary_prompt
    elif isinstance(summary_prompt, str) and summary_prompt.strip():
        summary_prompt_tmpl = summary_prompt
    else:
        return False, "", {
            "input_messages": len(messages),
            "normalized_messages": 0,
            "orphan_tool_messages": 0,
            "truncated_unmatched_tool_calls": False,
            "prompt_missing": True,
        }

    normalized_messages, normalize_stats = normalize_messages_for_llm(messages)
    if not normalized_messages:
        return False, "", normalize_stats

    model_cfg = ModelConfig.model_validate(cfg.get("model", {}))
    fallback_model_cfg = None
    fallback_model_raw = cfg.get("fallback_model")
    if isinstance(fallback_model_raw, dict) and fallback_model_raw:
        fallback_model_cfg = ModelConfig.model_validate(fallback_model_raw)
    max_cycles = cfg.get("max_cycles")
    timeout_retry_attempts = max_cycles if isinstance(max_cycles, int) and max_cycles > 0 else None
    llm_node = create_llm_node(
        model_cfg=model_cfg,
        fallback_model_cfg=fallback_model_cfg,
        timeout_retry_attempts=timeout_retry_attempts,
        timeout_fallback_mode="immediate_on_primary_timeout",
        stream_token_output=bool(cfg.get("stream_token_output")),
    )

    summary_input = {
        "messages": list(normalized_messages) + [HumanMessage(content=summary_prompt_tmpl.format(input=question))],
        "error": list(state.get("error", [])),
    }
    summary = await asyncio.wait_for(llm_node(summary_input), timeout=timeout_s)

    state_errors = list(state.get("error", []))
    summary_errors = summary.get("error", [])
    if isinstance(summary_errors, list) and summary_errors:
        state_errors.extend(str(err) for err in summary_errors)
    state["error"] = state_errors

    summary_messages = summary.get("messages")
    if not isinstance(summary_messages, list) or not summary_messages:
        return False, "", normalize_stats

    state["messages"] = summary_messages
    last_message = summary_messages[-1]

    cycle_histories = state.get("cycle_histories")
    if not isinstance(cycle_histories, list):
        cycle_histories = []
    cycle_histories = cycle_histories + [[last_message]]
    state["cycle_histories"] = cycle_histories

    process_details = state.get("process_details")
    if not isinstance(process_details, dict):
        process_details = {"rollouts": []}
    rollouts = process_details.get("rollouts")
    if not isinstance(rollouts, list):
        rollouts = []
    rollouts.append(
        {
            "messages": summary_messages,
            "error": state_errors[-1] if state_errors else "",
        }
    )
    process_details["rollouts"] = rollouts
    state["process_details"] = process_details

    output_text = message_to_text(last_message)
    state["output"] = output_text
    return True, output_text, normalize_stats


def format_runtime(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def record_task_outputs(run_dir: Path, task_id: int, answer: str, runtime_seconds: float) -> None:
    append_jsonl(
        run_dir / "final_answer.jsonl",
        {"id": task_id, "answer": answer},
    )
    append_jsonl(
        run_dir / "task_runtime.jsonl",
        {"id": task_id, "runtime": format_runtime(runtime_seconds)},
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_jsonl_task(line: str, line_no: int) -> tuple[int | None, str | None, str | None]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, None, f"Line {line_no}: invalid JSON ({exc})"

    if not isinstance(payload, dict):
        return None, None, f"Line {line_no}: item must be a JSON object"

    task_id = payload.get("task_id")
    question = payload.get("task_question")

    if not isinstance(task_id, int) or isinstance(task_id, bool):
        return None, None, f"Line {line_no}: task_id must be an integer"
    if not isinstance(question, str) or not question.strip():
        return task_id, None, f"Line {line_no}: task_question must be a non-empty string"
    return task_id, question, None


def find_task_question_by_id(task_file: Path, target_task_id: int) -> tuple[str | None, str | None]:
    with task_file.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            task_id, question, error = parse_jsonl_task(line, line_no)
            if error:
                continue
            if task_id == target_task_id:
                return question, None
    return None, f"task_id={target_task_id} not found in {task_file}"


async def execute_task(
    app: Any,
    task_id: int,
    question: str,
    cfg: dict,
    run_dir: Path,
    task_file_name: str | None = None,
    stream_messages: bool = False,
    console_log: bool = False,
) -> tuple[bool, Dict[str, Any], float]:
    logger = ResearchRunLogger(
        run_dir=run_dir,
        task_id=task_id,
        question=question,
        cfg=cfg,
        task_file_name=task_file_name,
        console_log=console_log,
    )
    merged_state: Dict[str, Any] = {}
    seen_messages: set[int] = set()
    cycle_open = False
    recursion_limit = int(os.getenv("RECURSION_LIMIT", "10000"))
    run_config = {"recursion_limit": recursion_limit}
    start_time = time.perf_counter()
    event_stream = app.astream({"question": question}, config=run_config)
    event_iter = event_stream.__aiter__()
    event_iter_closed = False
    last_node_name = ""
    last_node_ts = start_time

    try:
        timeout_trigger = ""
        force_end_cycle = False
        force_end_trigger = ""
        tools_watchdog_elapsed_s = 0.0
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= TASK_HARD_TIMEOUT_S:
                timeout_trigger = "hard_timeout_before_next_event"
                break
            if last_node_name == "tools_prep":
                waited_after_tools_prep = time.perf_counter() - last_node_ts
                if waited_after_tools_prep >= TOOL_NODE_WATCHDOG_S:
                    force_end_cycle = True
                    force_end_trigger = "tools_watchdog"
                    tools_watchdog_elapsed_s = waited_after_tools_prep
                    break
            if cycle_open and elapsed >= TASK_FORCE_END_CYCLE_S:
                force_end_cycle = True
                force_end_trigger = "time_window"
                break

            try:
                wait_timeout = max(0.0, TASK_HARD_TIMEOUT_S - elapsed)
                if cycle_open:
                    wait_timeout = min(wait_timeout, max(0.0, TASK_FORCE_END_CYCLE_S - elapsed))
                if last_node_name == "tools_prep":
                    waited_after_tools_prep = time.perf_counter() - last_node_ts
                    wait_timeout = min(wait_timeout, max(0.0, TOOL_NODE_WATCHDOG_S - waited_after_tools_prep))
                event = await asyncio.wait_for(
                    anext(event_iter),
                    timeout=wait_timeout,
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                elapsed_after_timeout = time.perf_counter() - start_time
                if last_node_name == "tools_prep":
                    waited_after_tools_prep = time.perf_counter() - last_node_ts
                    if waited_after_tools_prep >= TOOL_NODE_WATCHDOG_S:
                        force_end_cycle = True
                        force_end_trigger = "tools_watchdog"
                        tools_watchdog_elapsed_s = waited_after_tools_prep
                    elif cycle_open and elapsed_after_timeout >= TASK_FORCE_END_CYCLE_S:
                        force_end_cycle = True
                        force_end_trigger = "time_window"
                    else:
                        timeout_trigger = "hard_timeout_waiting_next_event"
                    break
                if cycle_open and elapsed_after_timeout >= TASK_FORCE_END_CYCLE_S:
                    force_end_cycle = True
                    force_end_trigger = "time_window"
                else:
                    timeout_trigger = "hard_timeout_waiting_next_event"
                break

            for node_name, node_output in event.items():
                if node_output is not None:
                    merged_state.update(node_output)
                logger.log_graph_event(node_name, node_output, merged_state)
                if node_name == "start_cycle":
                    cycle_open = True
                elif node_name == "end_cycle":
                    cycle_open = False
                last_node_name = node_name
                last_node_ts = time.perf_counter()

            if stream_messages:
                messages: list[BaseMessage] = merged_state.get("messages", [])
                for msg in messages:
                    msg_hash = hash(str(msg))
                    if msg_hash in seen_messages:
                        continue
                    print(convert_to_openai_messages(msg))
                    seen_messages.add(msg_hash)

        if force_end_cycle:
            if force_end_trigger == "tools_watchdog":
                logger.log_step(
                    "Main | Tools Watchdog",
                    f"Task {task_id} interrupted due to tools watchdog timeout.",
                    info_level="warning",
                    metadata={
                        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                        "elapsed_since_tools_prep_s": round(tools_watchdog_elapsed_s, 3),
                        "watchdog_s": TOOL_NODE_WATCHDOG_S,
                        "last_node": last_node_name,
                    },
                )

            logger.log_step(
                "Main | Force End Cycle",
                f"Task {task_id} forcing end-cycle summary; interrupting active cycle.",
                info_level="warning",
                metadata={
                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                    "force_end_cycle_s": TASK_FORCE_END_CYCLE_S,
                    "final_fallback_s": TASK_FINAL_FALLBACK_S,
                    "trigger_reason": force_end_trigger or "unknown",
                },
            )

            aclose = getattr(event_iter, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass
                event_iter_closed = True

            elapsed = time.perf_counter() - start_time
            summary_budget_s = TASK_FINAL_FALLBACK_S - elapsed
            force_summary_output = ""
            force_summary_ok = False
            force_summary_normalize_stats: dict[str, Any] = {}
            force_summary_reason = "summary_not_attempted"

            if summary_budget_s > 0:
                try:
                    force_summary_ok, force_summary_output, force_summary_normalize_stats = await run_forced_end_cycle_summary(
                        cfg=cfg,
                        question=question,
                        state=merged_state,
                        timeout_s=summary_budget_s,
                    )
                    if force_summary_ok:
                        if extract_boxed_answer(force_summary_output):
                            logger.log_step(
                                "Main | Force End Cycle",
                                "Forced end_cycle summary completed with boxed answer; finishing task.",
                                metadata={
                                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                                    "summary_budget_s": round(summary_budget_s, 3),
                                    "message_normalization": force_summary_normalize_stats,
                                },
                            )
                            logger.finalize_success(merged_state)
                            return True, merged_state, time.perf_counter() - start_time
                        force_summary_reason = "summary_without_boxed_answer"
                    else:
                        force_summary_reason = "summary_missing_messages"
                except asyncio.TimeoutError:
                    force_summary_reason = "summary_timeout_before_final_fallback"
                except Exception as exc:  # noqa: BLE001
                    force_summary_reason = f"summary_exception: {exc}"
            else:
                force_summary_reason = "no_budget_before_final_fallback"

            fallback_answer = apply_timeout_fallback(merged_state)
            logger.log_step(
                "Main | Timeout Fallback",
                f"Task {task_id} fallback applied after forced-summary phase.",
                info_level="warning",
                metadata={
                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                    "force_end_cycle_s": TASK_FORCE_END_CYCLE_S,
                    "final_fallback_s": TASK_FINAL_FALLBACK_S,
                    "hard_timeout_s": TASK_HARD_TIMEOUT_S,
                    "force_summary_reason": force_summary_reason,
                    "force_summary_output_preview": force_summary_output[:300],
                    "force_summary_message_normalization": force_summary_normalize_stats,
                    "fallback_answer": fallback_answer,
                    "completed_cycles": len(merged_state.get("cycle_histories", []) or []),
                },
            )
            logger.finalize_success(merged_state)
            return True, merged_state, time.perf_counter() - start_time

        if timeout_trigger:
            fallback_answer = apply_timeout_fallback(merged_state)
            logger.log_step(
                "Main | Timeout Fallback",
                f"Task {task_id} ended by timeout control ({timeout_trigger}).",
                info_level="warning",
                metadata={
                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                    "force_end_cycle_s": TASK_FORCE_END_CYCLE_S,
                    "final_fallback_s": TASK_FINAL_FALLBACK_S,
                    "hard_timeout_s": TASK_HARD_TIMEOUT_S,
                    "cycle_open": cycle_open,
                    "completed_cycles": len(merged_state.get("cycle_histories", []) or []),
                    "fallback_answer": fallback_answer,
                },
            )
            logger.finalize_success(merged_state)
            return True, merged_state, time.perf_counter() - start_time

        logger.finalize_success(merged_state)
        return True, merged_state, time.perf_counter() - start_time
    except Exception as exc:  # noqa: BLE001
        logger.finalize_error(exc, merged_state)
        return False, merged_state, time.perf_counter() - start_time
    finally:
        if not event_iter_closed:
            aclose = getattr(event_iter, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass


async def run_single_task(
    app: Any,
    cfg: dict,
    run_dir: Path,
    task_id: int,
    question: str,
    task_file_name: str | None,
    non_streaming: bool,
    console_log: bool,
) -> int:
    ok, state, runtime_seconds = await execute_task(
        app=app,
        task_id=task_id,
        question=question,
        cfg=cfg,
        run_dir=run_dir,
        task_file_name=task_file_name,
        stream_messages=(not non_streaming) and (not bool(cfg.get("stream_token_output"))),
        console_log=console_log,
    )
    answer = ""
    if ok:
        answer = extract_boxed_answer(state.get("output"))
        if not answer:
            answer = extract_last_completed_cycle_answer(state)
    record_task_outputs(run_dir=run_dir, task_id=task_id, answer=answer, runtime_seconds=runtime_seconds)

    if state.get("output"):
        output_text = str(state["output"])
        output_boxed = extract_boxed_answer(output_text)
        if output_boxed:
            if bool(cfg.get("stream_token_output")):
                print(f"\\boxed{{{output_boxed}}}")
            else:
                print(output_text)
        elif answer:
            print(f"\\boxed{{{answer}}}")
        elif not bool(cfg.get("stream_token_output")):
            print(output_text)
    elif ok and answer:
        print(f"\\boxed{{{answer}}}")
    if not ok:
        print("Task failed. Please check the generated log file for details.")
        return 1
    return 0


async def run_batch_tasks(
    app: Any,
    cfg: dict,
    run_dir: Path,
    task_file: Path,
    console_log: bool,
) -> int:
    success_count = 0
    failed_count = 0
    total_count = 0

    with task_file.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_count += 1

            task_id, question, error = parse_jsonl_task(line, line_no)
            if error:
                fallback_task_id = task_id if isinstance(task_id, int) else -line_no
                logger = ResearchRunLogger(
                    run_dir=run_dir,
                    task_id=fallback_task_id,
                    question="",
                    cfg=cfg,
                    task_file_name=task_file.name,
                    console_log=console_log,
                )
                logger.log_step(
                    "Main | Task Validation",
                    error,
                    info_level="error",
                    metadata={"line_no": line_no},
                )
                logger.finalize_error(error, final_state={"messages": [], "output": ""})
                record_task_outputs(
                    run_dir=run_dir,
                    task_id=fallback_task_id,
                    answer="",
                    runtime_seconds=0.0,
                )
                print(f"[task {fallback_task_id}] failed validation: {error}")
                failed_count += 1
                continue

            ok, state, runtime_seconds = await execute_task(
                app=app,
                task_id=task_id,
                question=question,
                cfg=cfg,
                run_dir=run_dir,
                task_file_name=task_file.name,
                stream_messages=False,
                console_log=console_log,
            )
            answer = ""
            if ok:
                answer = extract_boxed_answer(state.get("output"))
                if not answer:
                    answer = extract_last_completed_cycle_answer(state)
            record_task_outputs(run_dir=run_dir, task_id=task_id, answer=answer, runtime_seconds=runtime_seconds)
            if ok:
                success_count += 1
                print(f"[task {task_id}] success")
            else:
                failed_count += 1
                error_text = state.get("error", "unknown error")
                print(f"[task {task_id}] failed: {error_text}")

    print(
        f"Batch completed. total={total_count}, success={success_count}, failed={failed_count}, log_dir={run_dir}"
    )
    return 0 if failed_count == 0 else 1


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Re-TRAC graph by task_id from JSONL, or run full JSONL batch.")
    parser.add_argument(
        "--config",
        type=str,
        default="retrac/deep_research.yaml",
        help="Path to YAML config file.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--question",
        type=int,
        help="Single-task mode: input task_id and load task_question from JSONL.",
    )
    input_group.add_argument(
        "--task-file",
        type=str,
        help="Batch mode: run all tasks in this JSONL file.",
    )
    parser.add_argument(
        "--question-source",
        type=str,
        default=DEFAULT_TASK_FILE,
        help=f"JSONL source for --question task_id mode (default: {DEFAULT_TASK_FILE}).",
    )
    parser.add_argument(
        "--non-streaming",
        action="store_true",
        help="Only for --question mode. Disable real-time LLM token streaming output.",
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Print step logs to console in real time (works in both single and batch modes).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("stream_token_output", bool(args.question is not None and not args.non_streaming))
    graph = build_graph(cfg)
    app = graph.compile()
    run_dir = create_run_directory("retrac/logs")
    print(f"Logging run directory: {run_dir}")

    if args.task_file:
        exit_code = await run_batch_tasks(
            app=app,
            cfg=cfg,
            run_dir=run_dir,
            task_file=Path(args.task_file),
            console_log=args.console_log,
        )
    else:
        question_source = Path(args.question_source)
        question, error = find_task_question_by_id(question_source, args.question)
        if error:
            logger = ResearchRunLogger(
                run_dir=run_dir,
                task_id=args.question,
                question="",
                cfg=cfg,
                task_file_name=question_source.name,
                console_log=args.console_log,
            )
            logger.log_step("Main | Task Lookup", error, info_level="error")
            logger.finalize_error(error, final_state={"messages": [], "output": ""})
            record_task_outputs(run_dir=run_dir, task_id=args.question, answer="", runtime_seconds=0.0)
            print(error)
            raise SystemExit(1)

        exit_code = await run_single_task(
            app=app,
            cfg=cfg,
            run_dir=run_dir,
            task_id=args.question,
            question=question or "",
            task_file_name=question_source.name,
            non_streaming=args.non_streaming,
            console_log=args.console_log,
        )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
