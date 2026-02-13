# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import argparse
import sys
from pathlib import Path

# Ensure compatibility with existing benchmark import style.
CURRENT_DIR = Path(__file__).resolve().parent
APP_ROOT = CURRENT_DIR.parent
BENCHMARKS_DIR = APP_ROOT / "benchmarks"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from jsonl_inference.task_log_recovery import (
    build_ordered_results,
    load_existing_benchmark_results,
    load_tasks_from_jsonl,
    merge_backfilled_results_from_task_logs,
    task_key,
    write_benchmark_results_jsonl,
    write_final_answers_jsonl,
    write_task_runtimes_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill benchmark_results.jsonl and final_answers.jsonl from existing task_*.json logs."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing task_*.json logs.")
    parser.add_argument("--input-jsonl", required=True, help="Input task JSONL file.")
    parser.add_argument("--task-id-field", default="task_id")
    parser.add_argument("--question-field", default="task_question")
    parser.add_argument("--ground-truth-field", default="ground_truth")
    parser.add_argument("--file-name-field", default="file_name")
    parser.add_argument("--k-value", type=int, default=1, help="k_value written into recovered BenchmarkResult.")
    parser.add_argument(
        "--benchmark-results-out",
        default=None,
        help="Path to benchmark_results.jsonl (default: <run-dir>/benchmark_results.jsonl)",
    )
    parser.add_argument(
        "--final-answers-out",
        default=None,
        help="Path to final_answers.jsonl (default: <run-dir>/final_answers.jsonl)",
    )
    parser.add_argument(
        "--task-runtimes-out",
        default=None,
        help="Path to task_runtimes.jsonl (default: <run-dir>/task_runtimes.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    input_jsonl = Path(args.input_jsonl).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    benchmark_results_out = (
        Path(args.benchmark_results_out).resolve()
        if args.benchmark_results_out
        else run_dir / "benchmark_results.jsonl"
    )
    final_answers_out = (
        Path(args.final_answers_out).resolve()
        if args.final_answers_out
        else run_dir / "final_answers.jsonl"
    )
    task_runtimes_out = (
        Path(args.task_runtimes_out).resolve()
        if args.task_runtimes_out
        else run_dir / "task_runtimes.jsonl"
    )

    tasks = load_tasks_from_jsonl(
        input_jsonl=input_jsonl,
        task_id_field=args.task_id_field,
        question_field=args.question_field,
        ground_truth_field=args.ground_truth_field,
        file_name_field=args.file_name_field,
    )
    task_order = [task_key(task.task_id) for task in tasks]

    existing_results = load_existing_benchmark_results(benchmark_results_out)
    existing_count = len(existing_results)

    merged_results, recovered_count = merge_backfilled_results_from_task_logs(
        tasks=tasks,
        run_dir=run_dir,
        existing_results_by_key=existing_results,
        k_value=args.k_value,
    )

    ordered_results = build_ordered_results(task_order, merged_results)
    write_benchmark_results_jsonl(benchmark_results_out, ordered_results)
    write_final_answers_jsonl(final_answers_out, ordered_results)
    write_task_runtimes_jsonl(task_runtimes_out, ordered_results, run_dir)

    print("Backfill completed.")
    print(f"Run directory: {run_dir}")
    print(f"Input JSONL: {input_jsonl}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Existing benchmark results: {existing_count}")
    print(f"Recovered from task logs: {recovered_count}")
    print(f"Total saved results: {len(ordered_results)}")
    print(f"Pending tasks after backfill: {len(tasks) - len(ordered_results)}")
    print(f"benchmark_results.jsonl: {benchmark_results_out}")
    print(f"final_answers.jsonl: {final_answers_out}")
    print(f"task_runtimes.jsonl: {task_runtimes_out}")


if __name__ == "__main__":
    main()
