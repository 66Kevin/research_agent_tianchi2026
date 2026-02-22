import argparse
import concurrent.futures
import contextlib
import io
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_RUNTIME_LINES = 500
MAX_MESSAGE_HISTORY = 120
MAX_SUB_SESSIONS = 8
MAX_TEXT_LEN = 2000


def load_questions(question_file: Path, count: int | None) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with question_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                row["id"] = idx
            questions.append(row)
            if count is not None and count > 0 and len(questions) >= count:
                break
    return questions


def extract_final_answer(result: Any) -> str:
    text = "" if result is None else str(result).strip()
    if not text:
        return ""
    match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = text
    return " ".join(answer.split())


def _clear_proxy_env() -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "SOCKS_PROXY",
        "socks_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        os.environ[key] = ""


def _now_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_file_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def _message_to_dict(msg: Any) -> dict[str, Any]:
    raw: Any = msg
    if hasattr(raw, "model_dump"):
        try:
            raw = raw.model_dump()
        except Exception:
            raw = {"content": str(msg)}
    elif hasattr(raw, "dict"):
        try:
            raw = raw.dict()
        except Exception:
            raw = {"content": str(msg)}

    if not isinstance(raw, dict):
        return {"role": "unknown", "content": _truncate_text(raw)}

    normalized: dict[str, Any] = {
        "role": str(raw.get("role", "unknown")),
        "content": "",
    }
    if "name" in raw:
        normalized["name"] = str(raw.get("name", ""))
    if "tool_call_id" in raw:
        normalized["tool_call_id"] = str(raw.get("tool_call_id", ""))

    content = raw.get("content", "")
    if isinstance(content, (dict, list, tuple, set)):
        content = json.dumps(_to_jsonable(content), ensure_ascii=False)
    normalized["content"] = _truncate_text(content)

    if "tool_calls" in raw:
        normalized["tool_calls"] = _truncate_text(raw.get("tool_calls", ""))

    return normalized


def _pack_message_history(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system_prompt = ""
    message_history: list[dict[str, Any]] = []
    for msg in messages:
        m = _message_to_dict(msg)
        role = m.get("role")
        if role == "system" and not system_prompt:
            system_prompt = str(m.get("content", ""))
            continue
        message_history.append(m)
    return {"system_prompt": system_prompt, "message_history": message_history}


def _truncate_text(value: Any, limit: int = MAX_TEXT_LEN) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + " ...<truncated>"


def _trim_history_payload(history: dict[str, Any], max_messages: int = MAX_MESSAGE_HISTORY) -> dict[str, Any]:
    messages = history.get("message_history", [])
    trimmed: list[dict[str, Any]] = []
    for m in messages[:max_messages]:
        msg = _message_to_dict(m)
        trimmed.append(msg)
    return {"system_prompt": _truncate_text(history.get("system_prompt", "")), "message_history": trimmed}


def _extract_tool_name(line: str) -> str:
    # Example: Function '_execute_tool_call' called with args: search_google: executed ...
    match = re.search(r"args:\s*([a-zA-Z0-9_]+)\s*:", line)
    if match:
        return match.group(1)
    return ""


def _runtime_line_to_step_event(line: str, current_agent: str) -> tuple[str, str, dict[str, Any], str]:
    source = current_agent
    line_low = line.lower()

    if "iter " in line and "planner " in line:
        source = "main_agent"
        return "👑 Main Agent | Iteration", line.strip(), {"source_agent": source}, source

    if "iter " in line and "actor actor_for_step_" in line:
        source = "sub_agent"
        return "🤖 Sub Agent | Iteration", line.strip(), {"source_agent": source}, source

    if "starting execution of step" in line_low or "found [" in line_low:
        source = "main_agent"
        return "👑 Main Agent | Behavior", line.strip(), {"source_agent": source}, source

    if "completed execution of step" in line_low:
        source = "sub_agent"
        return "🤖 Sub Agent | Behavior", line.strip(), {"source_agent": source}, source

    if "_execute_tool_call" in line:
        tool_name = _extract_tool_name(line)
        step_name = "👑 Main Agent | Tool Call" if source == "main_agent" else "🤖 Sub Agent | Tool Call"
        metadata = {"source_agent": source}
        if tool_name:
            metadata["tool_name"] = tool_name
        return step_name, line.strip(), metadata, source

    if "chat_to_llm messages:" in line or "create_with_tools messages:" in line:
        step_name = "👑 Main Agent | LLM Prompt" if source == "main_agent" else "🤖 Sub Agent | LLM Prompt"
        return step_name, line.strip(), {"source_agent": source}, source

    if "llm with tools chat completions response" in line_low:
        step_name = "👑 Main Agent | LLM Response" if source == "main_agent" else "🤖 Sub Agent | LLM Response"
        return step_name, line.strip(), {"source_agent": source}, source

    if "max_iteration response" in line_low:
        step_name = "👑 Main Agent | Max Iteration" if source == "main_agent" else "🤖 Sub Agent | Max Iteration"
        return step_name, line.strip(), {"source_agent": source}, source

    step_name = "👑 Main Agent | Runtime" if source == "main_agent" else "🤖 Sub Agent | Runtime"
    return step_name, line.strip(), {"source_agent": source}, source


def _build_step_logs(
    runtime_lines: list[str],
    main_history: dict[str, Any],
    sub_histories: dict[str, dict[str, Any]],
    fallback_used: bool,
) -> list[dict[str, Any]]:
    step_logs: list[dict[str, Any]] = []

    def add_event(step_name: str, message: str, info_level: str = "INFO", metadata: dict[str, Any] | None = None):
        step_logs.append(
            {
                "step_name": step_name,
                "message": message[:3000],
                "timestamp": _now_human(),
                "info_level": info_level,
                "metadata": metadata or {},
            }
        )

    add_event("Main | Task Start", "Task execution started", metadata={"source_agent": "main_agent"})

    current_agent = "main_agent"
    for line in runtime_lines[:1500]:
        text = line.strip()
        if not text:
            continue
        step_name, message, metadata, current_agent = _runtime_line_to_step_event(text, current_agent)
        add_event(step_name, message, metadata=metadata)

    for i, msg in enumerate(main_history.get("message_history", [])[:200]):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        add_event(
            "👑 Main Agent | Speech",
            f"[{role}] {str(content)[:1200]}",
            metadata={"source_agent": "main_agent", "role": role, "message_index": i},
        )

    for session_id, session in list(sub_histories.items())[:20]:
        msgs = session.get("message_history", [])
        for i, msg in enumerate(msgs[:120]):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            add_event(
                "🤖 Sub Agent | Speech",
                f"[{role}] {str(content)[:1200]}",
                metadata={
                    "source_agent": "sub_agent",
                    "role": role,
                    "message_index": i,
                    "session_id": session_id,
                },
            )

    if fallback_used:
        add_event(
            "👑 Main Agent | Fallback",
            "Primary framework answer was empty or timed out; fallback direct answer was used.",
            metadata={"source_agent": "main_agent"},
        )

    add_event("task_execution_finished", "Task execution completed", metadata={})
    return step_logs


def _collect_env_info() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "workspace_path": os.environ.get("WORKSPACE_PATH", ""),
        "model": "qwen3.5-plus",
    }


def _read_runtime_lines(runtime_log_path: Path) -> list[str]:
    if not runtime_log_path.exists():
        return []
    try:
        text = runtime_log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return [_truncate_text(line) for line in text.splitlines()[:MAX_RUNTIME_LINES]]


def _run_single_question_worker(question: str, runtime_log_path_str: str, result_file_path_str: str) -> None:
    _clear_proxy_env()
    from app.manus.manus import Manus
    from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision

    runtime_log_path = Path(runtime_log_path_str)
    runtime_log_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = _now_human()
    result: dict[str, Any] = {
        "status": "failed:unknown",
        "answer": "",
        "error": "",
        "raw_result": "",
        "planner_history": {"system_prompt": "", "message_history": []},
        "sub_agent_histories": {},
        "start_time": start_time,
        "end_time": start_time,
    }

    output_format = (
        "仅返回题目要求的最终答案文本。"
        "禁止输出思考过程、解释、前后缀、标签、额外句子。"
        "若题目有格式要求，严格按题目要求输出。"
    )

    try:
        manus = Manus(llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision)
        with runtime_log_path.open("w", encoding="utf-8", buffering=1) as runtime_fp:
            with contextlib.redirect_stdout(runtime_fp), contextlib.redirect_stderr(runtime_fp):
                raw_result = manus.execute(question, output_format=output_format)

        planner_messages = manus.task_planner_agent.history if isinstance(manus.task_planner_agent.history, list) else []
        planner_history = _trim_history_payload(
            _pack_message_history(planner_messages)
        )

        sub_histories_raw = manus.actor_message_history_sessions if isinstance(manus.actor_message_history_sessions, dict) else {}
        sub_histories: dict[str, dict[str, Any]] = {}
        for idx, (session_id, messages) in enumerate(sub_histories_raw.items()):
            if idx >= MAX_SUB_SESSIONS:
                break
            if isinstance(messages, list):
                sub_histories[str(session_id)] = _trim_history_payload(_pack_message_history(messages))
            else:
                sub_histories[str(session_id)] = {"system_prompt": "", "message_history": []}

        answer = extract_final_answer(raw_result)
        result.update(
            {
                "status": "ok",
                "answer": answer,
                "raw_result": _truncate_text(raw_result, 4000),
                "planner_history": planner_history,
                "sub_agent_histories": sub_histories,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        result.update(
            {
                "status": f"failed:{type(exc).__name__}: {exc}",
                "error": _truncate_text(traceback.format_exc(), 6000),
            }
        )
    finally:
        result["end_time"] = _now_human()
        result_file = Path(result_file_path_str)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with result_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)


def run_with_timeout(question: str, timeout_seconds: int, runtime_log_path: Path) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_file_path = runtime_log_path.with_suffix(".result.json")
    if result_file_path.exists():
        try:
            result_file_path.unlink()
        except Exception:
            pass
    proc = ctx.Process(target=_run_single_question_worker, args=(question, str(runtime_log_path), str(result_file_path)))
    started = time.time()
    start_time = _now_human()
    proc.start()

    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(2)
        return {
            "status": "timeout",
            "answer": "",
            "error": "",
            "raw_result": "",
            "runtime_lines": _read_runtime_lines(runtime_log_path),
            "planner_history": {"system_prompt": "", "message_history": []},
            "sub_agent_histories": {},
            "start_time": start_time,
            "end_time": _now_human(),
            "elapsed_seconds": round(time.time() - started, 2),
        }

    if not result_file_path.exists():
        return {
            "status": "failed:no-result",
            "answer": "",
            "error": "",
            "raw_result": "",
            "runtime_lines": _read_runtime_lines(runtime_log_path),
            "planner_history": {"system_prompt": "", "message_history": []},
            "sub_agent_histories": {},
            "start_time": start_time,
            "end_time": _now_human(),
            "elapsed_seconds": round(time.time() - started, 2),
        }

    try:
        result = json.loads(result_file_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status": "failed:bad-result-file",
            "answer": "",
            "error": "",
            "raw_result": "",
            "runtime_lines": _read_runtime_lines(runtime_log_path),
            "planner_history": {"system_prompt": "", "message_history": []},
            "sub_agent_histories": {},
            "start_time": start_time,
            "end_time": _now_human(),
            "elapsed_seconds": round(time.time() - started, 2),
        }

    result["runtime_lines"] = _read_runtime_lines(runtime_log_path)
    if "elapsed_seconds" not in result:
        result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def _fallback_direct_answer(question: str) -> str:
    from llm import llm_for_act

    messages = [
        {
            "role": "system",
            "content": (
                "You are a QA assistant. Return only the final answer text with no explanation, labels, or extra words."
            ),
        },
        {"role": "user", "content": question},
    ]
    result = llm_for_act.chat_to_llm(messages)
    return extract_final_answer(result)


def _write_task_log(
    task_log_dir: Path,
    question_id: Any,
    question_text: str,
    timeout_seconds: int,
    result: dict[str, Any],
    final_answer: str,
    fallback_used: bool,
) -> Path:
    task_log_dir.mkdir(parents=True, exist_ok=True)
    task_file = task_log_dir / f"task_{question_id}_attempt-1_format-retry-0_{_now_file_suffix()}.json"

    main_history = result.get("planner_history", {"system_prompt": "", "message_history": []})
    sub_histories = result.get("sub_agent_histories", {})
    step_logs = _build_step_logs(
        runtime_lines=result.get("runtime_lines", []),
        main_history=main_history,
        sub_histories=sub_histories,
        fallback_used=fallback_used,
    )

    task_payload = {
        "status": "success" if final_answer else result.get("status", "failed"),
        "start_time": result.get("start_time", _now_human()),
        "end_time": result.get("end_time", _now_human()),
        "task_id": str(question_id),
        "input": {"question": question_text, "id": question_id},
        "ground_truth": None,
        "final_boxed_answer": final_answer,
        "final_judge_result": "",
        "judge_type": "",
        "eval_details": None,
        "error": result.get("error", ""),
        "current_main_turn_id": len(main_history.get("message_history", [])),
        "current_sub_agent_turn_id": sum(len(v.get("message_history", [])) for v in sub_histories.values()),
        "sub_agent_counter": len(sub_histories),
        "current_sub_agent_session_id": next(iter(sub_histories), None) if sub_histories else None,
        "env_info": _collect_env_info(),
        "log_dir": str(task_log_dir),
        "main_agent_message_history": main_history,
        "sub_agent_message_history_sessions": sub_histories,
        "step_logs": step_logs,
        "trace_data": {},
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "raw_status": result.get("status", ""),
    }

    with task_file.open("w", encoding="utf-8") as f:
        json.dump(task_payload, f, ensure_ascii=False, indent=2)
    return task_file


def _run_one_question(
    row: dict[str, Any],
    timeout_seconds: int,
    task_log_dir: Path,
    disable_fallback: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = row["id"]
    question_text = row["question"]

    runtime_log_path = task_log_dir / f"task_{question_id}_runtime_{_now_file_suffix()}.log"
    result = run_with_timeout(question_text, timeout_seconds, runtime_log_path)
    answer = result.get("answer", "")
    status = result.get("status", "failed")
    fallback_used = False

    if not answer and not disable_fallback:
        try:
            answer = _fallback_direct_answer(question_text)
            if answer:
                status = f"{status}|fallback_ok"
                fallback_used = True
            else:
                status = f"{status}|fallback_empty"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            status = f"{status}|fallback_failed:{type(exc).__name__}: {exc}"

    task_log_path = _write_task_log(
        task_log_dir=task_log_dir,
        question_id=question_id,
        question_text=question_text,
        timeout_seconds=timeout_seconds,
        result=result,
        final_answer=answer,
        fallback_used=fallback_used,
    )

    output_row = {"id": question_id, "answer": answer}
    run_log = {
        "id": question_id,
        "question": question_text,
        "answer": answer,
        "status": status,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "timeout_seconds": timeout_seconds,
        "task_log_path": str(task_log_path),
    }
    return output_row, run_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Co-Sight on the first N Tianchi questions with per-question timeout and detailed agent logs."
    )
    parser.add_argument("--question-file", type=Path, required=True, help="Path to question.jsonl")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to output JSONL (id + answer)")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to run summary JSON")
    parser.add_argument("--task-log-dir", type=Path, required=True, help="Directory to store per-task detailed logs")
    parser.add_argument("--count", type=int, default=2, help="How many questions to run from the beginning; <=0 means all")
    parser.add_argument("--parallel", type=int, default=1, help="How many questions to run concurrently")
    parser.add_argument("--timeout", type=int, default=600, help="Per-question timeout in seconds")
    parser.add_argument("--disable-fallback", action="store_true", help="Disable direct-answer fallback when framework answer is empty")
    args = parser.parse_args()

    requested_count = None if args.count <= 0 else args.count
    questions = load_questions(args.question_file, requested_count)
    if not questions:
        raise RuntimeError(f"No questions loaded from {args.question_file}")

    parallel = max(1, int(args.parallel))
    total = len(questions)
    print(f"loaded questions: {total}, parallel={parallel}, timeout={args.timeout}s")

    output_rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []

    if parallel == 1:
        for row in questions:
            output_row, run_log = _run_one_question(
                row=row,
                timeout_seconds=args.timeout,
                task_log_dir=args.task_log_dir,
                disable_fallback=args.disable_fallback,
            )
            output_rows.append(output_row)
            run_logs.append(run_log)
            print(
                f"[id={run_log['id']}] status={run_log['status']} elapsed={run_log.get('elapsed_seconds')}s "
                f"answer={run_log['answer']} task_log={run_log['task_log_path']}"
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_row = {
                executor.submit(
                    _run_one_question,
                    row,
                    args.timeout,
                    args.task_log_dir,
                    args.disable_fallback,
                ): row
                for row in questions
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_row):
                completed += 1
                row = future_to_row[future]
                question_id = row.get("id")
                try:
                    output_row, run_log = future.result()
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    output_row = {"id": question_id, "answer": ""}
                    run_log = {
                        "id": question_id,
                        "question": row.get("question", ""),
                        "answer": "",
                        "status": f"failed:executor:{type(exc).__name__}: {exc}",
                        "elapsed_seconds": None,
                        "timeout_seconds": args.timeout,
                        "task_log_path": "",
                    }
                output_rows.append(output_row)
                run_logs.append(run_log)
                print(
                    f"[{completed}/{total}] [id={run_log['id']}] status={run_log['status']} "
                    f"elapsed={run_log.get('elapsed_seconds')}s answer={run_log['answer']} "
                    f"task_log={run_log['task_log_path']}"
                )

    output_rows.sort(key=lambda x: x["id"])
    run_logs.sort(key=lambda x: x["id"])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as f:
        json.dump(run_logs, f, ensure_ascii=False, indent=2)

    print(f"saved answers: {args.output_file}")
    print(f"saved run summary: {args.log_file}")
    print(f"saved per-task logs dir: {args.task_log_dir}")


if __name__ == "__main__":
    main()
