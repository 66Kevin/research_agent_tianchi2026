import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import time
from pathlib import Path
from typing import Any


def load_questions(question_file: Path, count: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with question_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                row["id"] = idx
            questions.append(row)
            if len(questions) >= count:
                break
    return questions


def extract_final_answer(result: Any) -> str:
    text = "" if result is None else str(result).strip()
    if not text:
        return ""

    # Co-Sight finalize output is expected in <final_answer>...</final_answer>.
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


def _run_single_question(question: str) -> str:
    _clear_proxy_env()
    from app.manus.manus import Manus
    from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision

    manus = Manus(llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision)
    output_format = (
        "仅返回题目要求的最终答案文本。"
        "禁止输出思考过程、解释、前后缀、标签、额外句子。"
        "若题目有格式要求，严格按题目要求输出。"
    )
    actor_question = (
        f"{question}\n\n"
        f"输出要求：{output_format}"
    )
    raw_result = manus.execute_actor(actor_question)
    return extract_final_answer(raw_result)


def _fallback_direct_answer(question: str) -> str:
    from llm import llm_for_act

    messages = [
        {
            "role": "system",
            "content": (
                "你是答题助手。只输出最终答案文本，不要解释，不要标签。"
                "若题目要求公司名格式，使用完整英文公司名。"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    result = llm_for_act.chat_to_llm(messages)
    return extract_final_answer(result)


def _run_single_question_worker(question: str, result_queue: Any) -> None:
    try:
        answer = _run_single_question(question)
        result_queue.put({"answer": answer, "status": "ok"})
    except Exception as exc:  # pragma: no cover - runtime safeguard
        result_queue.put({"answer": "", "status": f"failed:{type(exc).__name__}: {exc}"})


def run_with_timeout(question: str, timeout_seconds: int) -> tuple[str, str]:
    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_single_question_worker, args=(question, result_queue))
    proc.start()

    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(2)
        return "", "timeout"

    try:
        result = result_queue.get_nowait()
        return str(result.get("answer", "")), str(result.get("status", "failed:no-status"))
    except queue.Empty:
        if proc.exitcode and proc.exitcode != 0:
            return "", f"failed:exitcode:{proc.exitcode}"
        return "", "failed:no-result"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Co-Sight on the first N Tianchi questions with per-question timeout.")
    parser.add_argument("--question-file", type=Path, required=True, help="Path to question.jsonl")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to output JSONL (id + answer)")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to run log JSON")
    parser.add_argument("--count", type=int, default=2, help="How many questions to run from the beginning")
    parser.add_argument("--timeout", type=int, default=600, help="Per-question timeout in seconds")
    args = parser.parse_args()

    questions = load_questions(args.question_file, args.count)
    if not questions:
        raise RuntimeError(f"No questions loaded from {args.question_file}")

    output_rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []

    for row in questions:
        question_id = row["id"]
        question_text = row["question"]
        started_at = time.time()
        answer, status = run_with_timeout(question_text, args.timeout)
        if not answer:
            try:
                answer = _fallback_direct_answer(question_text)
                if answer:
                    status = f"{status}|fallback_ok"
                else:
                    status = f"{status}|fallback_empty"
            except Exception as exc:  # pragma: no cover - runtime safeguard
                status = f"{status}|fallback_failed:{type(exc).__name__}: {exc}"
        elapsed_seconds = round(time.time() - started_at, 2)

        output_rows.append({"id": question_id, "answer": answer})
        run_logs.append(
            {
                "id": question_id,
                "question": question_text,
                "answer": answer,
                "status": status,
                "elapsed_seconds": elapsed_seconds,
                "timeout_seconds": args.timeout,
            }
        )

        print(f"[id={question_id}] status={status} elapsed={elapsed_seconds}s answer={answer}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as f:
        json.dump(run_logs, f, ensure_ascii=False, indent=2)

    print(f"saved answers: {args.output_file}")
    print(f"saved logs: {args.log_file}")


if __name__ == "__main__":
    main()
