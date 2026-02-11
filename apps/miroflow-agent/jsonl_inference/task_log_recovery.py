# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmarks.common_benchmark import BenchmarkResult, BenchmarkTask
from src.utils.prompt_utils import (
    BLOCKED_BY_POLICY_MESSAGE,
    BLOCKED_JUDGE_TYPE,
    FORMAT_ERROR_MESSAGE,
)

TASK_LOG_FILENAME_PATTERN = re.compile(
    r"^task_(?P<task_id>.+?)_attempt-(?P<attempt>\d+)_format-retry-(?P<retry>\d+)_(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.json$"
)


def task_key(task_id: Any) -> str:
    return str(task_id)


def load_tasks_from_jsonl(
    input_jsonl: Path,
    task_id_field: str = "task_id",
    question_field: str = "task_question",
    ground_truth_field: str = "ground_truth",
    file_name_field: Optional[str] = "file_name",
) -> List[BenchmarkTask]:
    """Load BenchmarkTask records from JSONL with strict validation."""
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    tasks: List[BenchmarkTask] = []
    excluded_keys = {
        task_id_field,
        question_field,
        ground_truth_field,
        file_name_field,
    }

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                raise ValueError(
                    f"Invalid empty line in input JSONL at line {line_number}"
                )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON in input JSONL at line {line_number}: {e}"
                ) from e

            if task_id_field not in data:
                raise KeyError(
                    f"Missing required field '{task_id_field}' at line {line_number}"
                )
            if question_field not in data:
                raise KeyError(
                    f"Missing required field '{question_field}' at line {line_number}"
                )

            file_path = None
            if file_name_field:
                file_path = data.get(file_name_field)

            metadata = {k: v for k, v in data.items() if k not in excluded_keys}
            task = BenchmarkTask(
                task_id=data[task_id_field],
                task_question=data[question_field],
                ground_truth=data.get(ground_truth_field),
                file_path=file_path,
                metadata=metadata,
            )
            tasks.append(task)

    return tasks


def load_existing_benchmark_results(
    benchmark_results_path: Path,
) -> Dict[str, BenchmarkResult]:
    """Load existing benchmark results JSONL if present."""
    results: Dict[str, BenchmarkResult] = {}
    if not benchmark_results_path.exists():
        return results

    with open(benchmark_results_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: invalid JSON in benchmark results at line {line_number}: {e}"
                )
                continue

            try:
                result = BenchmarkResult(**data)
            except Exception as e:
                print(
                    f"Warning: invalid benchmark result schema at line {line_number}: {e}"
                )
                continue

            results[task_key(result.task_id)] = result

    return results


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)


def write_benchmark_results_jsonl(
    output_path: Path, ordered_results: List[BenchmarkResult]
) -> None:
    write_jsonl_atomic(output_path, [asdict(result) for result in ordered_results])


def write_final_answers_jsonl(
    output_path: Path, ordered_results: List[BenchmarkResult]
) -> None:
    rows = [
        {
            "id": result.task_id,
            "answer": result.model_boxed_answer or "",
        }
        for result in ordered_results
    ]
    write_jsonl_atomic(output_path, rows)


def write_blocked_tasks_jsonl(
    output_path: Path, ordered_results: List[BenchmarkResult]
) -> None:
    rows: List[Dict[str, Any]] = []
    for result in ordered_results:
        is_blocked = (
            result.status == "blocked"
            or result.final_judge_result == BLOCKED_BY_POLICY_MESSAGE
            or result.model_boxed_answer == BLOCKED_BY_POLICY_MESSAGE
        )
        if not is_blocked:
            continue

        rows.append(
            {
                "id": result.task_id,
                "status": "blocked",
                "reason": result.error_message or "Blocked by provider safety policy",
                "log_file_path": result.log_file_path or "",
            }
        )

    write_jsonl_atomic(output_path, rows)


def build_ordered_results(
    task_order: List[str], results_by_key: Dict[str, BenchmarkResult]
) -> List[BenchmarkResult]:
    return [results_by_key[key] for key in task_order if key in results_by_key]


def _parse_task_log_filename(
    filename: str,
) -> Optional[Tuple[str, int, int, str]]:
    match = TASK_LOG_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return (
        match.group("task_id"),
        int(match.group("attempt")),
        int(match.group("retry")),
        match.group("timestamp"),
    )


def _is_recoverable_task_log(log_data: Dict[str, Any]) -> bool:
    status = log_data.get("status")
    answer = log_data.get("final_boxed_answer")
    final_judge = log_data.get("final_judge_result")

    if status == "blocked":
        return bool(
            answer == BLOCKED_BY_POLICY_MESSAGE
            or final_judge == BLOCKED_BY_POLICY_MESSAGE
        )

    if status != "success":
        return False

    if not answer:
        return False
    if answer == FORMAT_ERROR_MESSAGE:
        return False
    if isinstance(answer, str) and answer.startswith(FORMAT_ERROR_MESSAGE):
        return False
    return True


def _select_best_task_logs(
    run_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Select the best recoverable log for each task."""
    selected: Dict[str, Dict[str, Any]] = {}

    for log_path in sorted(run_dir.glob("task_*.json")):
        parsed = _parse_task_log_filename(log_path.name)
        if parsed is None:
            continue

        task_id_str, attempt_number, format_retry, timestamp = parsed
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read task log {log_path}: {e}")
            continue

        if not _is_recoverable_task_log(log_data):
            continue

        rank = (attempt_number, format_retry, timestamp)
        current = selected.get(task_id_str)
        if current is None or rank > current["rank"]:
            selected[task_id_str] = {
                "rank": rank,
                "log_path": log_path,
                "log_data": log_data,
                "attempt_number": attempt_number,
            }

    return selected


def merge_backfilled_results_from_task_logs(
    tasks: List[BenchmarkTask],
    run_dir: Path,
    existing_results_by_key: Optional[Dict[str, BenchmarkResult]] = None,
    k_value: int = 1,
) -> Tuple[Dict[str, BenchmarkResult], int]:
    """
    Merge recoverable task-log results into existing benchmark results.

    Returns:
        (merged_results_by_key, recovered_count)
    """
    merged: Dict[str, BenchmarkResult] = dict(existing_results_by_key or {})
    selected_logs = _select_best_task_logs(run_dir)
    recovered_count = 0

    for task in tasks:
        key = task_key(task.task_id)
        if key in merged:
            continue

        selected = selected_logs.get(key)
        if selected is None:
            continue

        log_data = selected["log_data"]
        answer = log_data.get("final_boxed_answer") or ""
        log_file_path = str(selected["log_path"])
        attempt_number = selected["attempt_number"]
        eval_details = log_data.get("eval_details")
        status = log_data.get("status", "success")
        blocked_reason = (
            log_data.get("error")
            or log_data.get("trace_data", {}).get("blocked_reason")
            or ""
        )

        final_judge_result = log_data.get("final_judge_result")
        if not final_judge_result:
            if status == "blocked" or answer == BLOCKED_BY_POLICY_MESSAGE:
                final_judge_result = BLOCKED_BY_POLICY_MESSAGE
            else:
                final_judge_result = (
                    "TEST_SET_MODE" if task.ground_truth is None else "PASS_AT_K_FAILED"
                )

        judge_type = log_data.get("judge_type")
        if not judge_type:
            judge_type = (
                BLOCKED_JUDGE_TYPE
                if final_judge_result == BLOCKED_BY_POLICY_MESSAGE
                else "pass_at_k"
            )

        attempt_entry: Dict[str, Any] = {
            "attempt_number": attempt_number,
            "model_boxed_answer": answer,
            "status": status,
            "log_file_path": log_file_path,
            "final_judge_result": final_judge_result,
            "judge_type": judge_type,
            "is_correct": final_judge_result == "CORRECT",
        }
        if eval_details is not None:
            attempt_entry["eval_details"] = eval_details

        merged[key] = BenchmarkResult(
            task_id=task.task_id,
            task_question=task.task_question,
            ground_truth=task.ground_truth,
            file_path=task.file_path,
            status=status,
            model_boxed_answer=answer,
            metadata=task.metadata.copy(),
            error_message=blocked_reason,
            final_judge_result=final_judge_result,
            judge_type=judge_type,
            log_file_path=log_file_path,
            attempts=[attempt_entry],
            pass_at_k_success=final_judge_result == "CORRECT",
            k_value=k_value,
        )
        recovered_count += 1

    return merged, recovered_count
