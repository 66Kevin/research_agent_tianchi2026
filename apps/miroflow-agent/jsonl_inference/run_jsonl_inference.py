# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure compatibility with existing benchmark import style:
# common_benchmark.py imports `evaluators.*` as top-level modules.
CURRENT_DIR = Path(__file__).resolve().parent
APP_ROOT = CURRENT_DIR.parent
BENCHMARKS_DIR = APP_ROOT / "benchmarks"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from benchmarks.common_benchmark import (
    BenchmarkResult,
    BenchmarkTask,
    GenericEvaluator,
    _task_worker,
)
from jsonl_inference.task_log_recovery import (
    build_ordered_results,
    load_existing_benchmark_results,
    merge_backfilled_results_from_task_logs,
    task_key,
    write_blocked_tasks_jsonl,
    write_benchmark_results_jsonl,
    write_final_answers_jsonl,
)
from src.logging.summary_time_cost import generate_summary


class InferenceOnlyEvaluator(GenericEvaluator):
    """Evaluator for inference-only JSONL inputs."""

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """
        Load tasks from JSONL with strict validation.

        Required fields:
        - task_id
        - task_question

        Optional fields:
        - ground_truth
        - file_name (or mapped file field)
        """
        print(f"Loading inference tasks from {self.metadata_file}")

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        tasks: List[BenchmarkTask] = []
        excluded_keys = {
            self.task_id_field,
            self.question_field,
            self.ground_truth_field,
            self.file_name_field,
        }

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if limit is not None and len(tasks) >= limit:
                    break

                text = line.strip()
                if not text:
                    raise ValueError(
                        f"Invalid empty line in metadata file at line {line_number}"
                    )

                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON at line {line_number}: {e}"
                    ) from e

                if self.task_id_field not in data:
                    raise KeyError(
                        f"Missing required field '{self.task_id_field}' at line {line_number}"
                    )
                if self.question_field not in data:
                    raise KeyError(
                        f"Missing required field '{self.question_field}' at line {line_number}"
                    )

                file_path = None
                if self.file_name_field:
                    file_path = data.get(self.file_name_field)

                metadata = {k: v for k, v in data.items() if k not in excluded_keys}

                task = BenchmarkTask(
                    task_id=data[self.task_id_field],
                    task_question=data[self.question_field],
                    ground_truth=data.get(self.ground_truth_field),
                    file_path=file_path,
                    metadata=metadata,
                )
                tasks.append(task)

        self.tasks = tasks
        print(f"Loaded {len(tasks)} inference tasks")
        return tasks

    def run_parallel_inference_with_callback(
        self,
        tasks: List[BenchmarkTask],
        max_concurrent: int = 3,
        on_result: Optional[Callable[[BenchmarkResult], None]] = None,
    ) -> List[BenchmarkResult]:
        """Run inference in parallel and invoke callback whenever a task finishes."""
        print(
            f"Running inference on {len(tasks)} tasks with max_concurrent={max_concurrent} (multiprocessing)"
        )

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        shuffled_tasks = tasks.copy()
        random.shuffle(shuffled_tasks)

        evaluator_kwargs = {
            "data_dir": str(self.data_dir),
            "benchmark_name": self.benchmark_name,
            "log_dir": str(self.get_log_dir()),
        }
        if hasattr(self, "metadata_file"):
            evaluator_kwargs["metadata_file"] = str(self.metadata_file.name)
        if hasattr(self, "task_id_field"):
            evaluator_kwargs["task_id_field"] = self.task_id_field
        if hasattr(self, "question_field"):
            evaluator_kwargs["question_field"] = self.question_field
        if hasattr(self, "ground_truth_field"):
            evaluator_kwargs["ground_truth_field"] = self.ground_truth_field
        if hasattr(self, "file_name_field"):
            evaluator_kwargs["file_name_field"] = self.file_name_field

        worker_args = []
        for task in shuffled_tasks:
            task_dict = {
                "task_id": task.task_id,
                "task_question": task.task_question,
                "ground_truth": task.ground_truth,
                "file_path": task.file_path,
                "metadata": task.metadata,
            }
            worker_args.append((task_dict, cfg_dict, evaluator_kwargs))

        task_map = {str(task.task_id): task for task in shuffled_tasks}
        results_dict: Dict[str, BenchmarkResult] = {}
        future_to_task_key = {}
        executor = None

        try:
            executor = ProcessPoolExecutor(max_workers=max_concurrent)

            for args in worker_args:
                task_dict = args[0]
                task_key = str(task_dict["task_id"])
                future = executor.submit(_task_worker, *args)
                future_to_task_key[future] = task_key

            for future in as_completed(future_to_task_key):
                task_key = future_to_task_key[future]
                task = task_map[task_key]

                try:
                    result_dict = future.result()
                    result = BenchmarkResult(**result_dict)
                except Exception as e:
                    print(f"Exception in task {task.task_id}: {e}")
                    result = BenchmarkResult(
                        task_id=task.task_id,
                        task_question=task.task_question,
                        ground_truth=task.ground_truth,
                        file_path=task.file_path,
                        model_boxed_answer="",
                        status="failed",
                        metadata=task.metadata.copy(),
                        error_message=str(e),
                    )

                results_dict[task_key] = result

                if on_result is not None:
                    try:
                        on_result(result)
                    except Exception as callback_error:
                        print(
                            f"Warning: failed to run completion callback for task {task.task_id}: {callback_error}"
                        )

                print(f"Progress: {len(results_dict)}/{len(shuffled_tasks)} tasks completed")

        except KeyboardInterrupt:
            print("\n⚠️  Received interrupt signal, shutting down gracefully...")
            if executor is not None:
                for future in future_to_task_key:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if executor is not None:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass

        processed_results = [
            results_dict[str(task.task_id)]
            for task in shuffled_tasks
            if str(task.task_id) in results_dict
        ]

        task_id_to_index = {str(task.task_id): i for i, task in enumerate(tasks)}
        processed_results.sort(
            key=lambda r: task_id_to_index.get(str(r.task_id), len(tasks))
        )

        self.results = processed_results
        return processed_results


class JSONLInferenceRunner:
    """Runs inference-only workflow and writes both detailed and lightweight outputs."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.benchmark_name = cfg.benchmark.name

        evaluator_kwargs = cfg.benchmark.get("evaluator_kwargs", OmegaConf.create({}))

        # Support legacy benchmark config structure
        if "metadata_file" in cfg.benchmark.data:
            evaluator_kwargs["metadata_file"] = cfg.benchmark.data.metadata_file
        if "field_mapping" in cfg.benchmark.data:
            mapping = cfg.benchmark.data.field_mapping
            if "task_id_field" in mapping:
                evaluator_kwargs["task_id_field"] = mapping.task_id_field
            if "task_question_field" in mapping:
                evaluator_kwargs["question_field"] = mapping.task_question_field
            if "ground_truth_field" in mapping:
                evaluator_kwargs["ground_truth_field"] = mapping.ground_truth_field
            if "file_name_field" in mapping:
                evaluator_kwargs["file_name_field"] = mapping.file_name_field

        self.evaluator = InferenceOnlyEvaluator(
            data_dir=cfg.benchmark.data.data_dir,
            benchmark_name=self.benchmark_name,
            cfg=cfg,
            **evaluator_kwargs,
        )

    @staticmethod
    def _task_key(task_id: object) -> str:
        return task_key(task_id)

    def _save_benchmark_results_jsonl(
        self, output_file: str, ordered_results: List[BenchmarkResult]
    ) -> str:
        output_path = Path(output_file)
        write_benchmark_results_jsonl(output_path, ordered_results)
        print(f"Benchmark results saved to {output_path}")
        return str(output_path)

    def _save_final_answers_jsonl(
        self, output_file: str, ordered_results: List[BenchmarkResult]
    ) -> str:
        output_path = Path(output_file)
        write_final_answers_jsonl(output_path, ordered_results)
        print(f"Final answers saved to {output_path}")
        return str(output_path)

    def _load_existing_benchmark_results(
        self, input_file: Path
    ) -> Dict[str, BenchmarkResult]:
        existing_results = load_existing_benchmark_results(input_file)
        if existing_results:
            print(
                f"Loaded {len(existing_results)} existing completed task(s) from {input_file}"
            )
        return existing_results

    def _build_ordered_results(
        self, task_order: List[str], results_by_key: Dict[str, BenchmarkResult]
    ) -> List[BenchmarkResult]:
        return build_ordered_results(task_order, results_by_key)

    def _persist_checkpoint(
        self,
        task_order: List[str],
        results_by_key: Dict[str, BenchmarkResult],
        benchmark_results_path: Path,
        final_answers_path: Path,
        blocked_tasks_path: Path,
    ) -> None:
        ordered_results = self._build_ordered_results(task_order, results_by_key)
        self.evaluator.results = ordered_results
        self._save_benchmark_results_jsonl(str(benchmark_results_path), ordered_results)
        self._save_final_answers_jsonl(str(final_answers_path), ordered_results)
        write_blocked_tasks_jsonl(blocked_tasks_path, ordered_results)

    def run_inference(self) -> Dict[str, str]:
        print(f"Starting inference for benchmark: {self.benchmark_name}")
        print(f"LLM Provider: {self.evaluator.llm_provider}")
        print(f"LLM Model: {self.evaluator.llm_model}")

        self.evaluator.load_tasks(limit=self.cfg.benchmark.execution.max_tasks)
        all_tasks = self.evaluator.tasks
        if not all_tasks:
            print("No tasks loaded. Exiting.")
            return {}

        log_dir = self.evaluator.get_log_dir()
        benchmark_results_path = log_dir / "benchmark_results.jsonl"    
        final_answers_path = log_dir / "final_answers.jsonl"
        blocked_tasks_path = log_dir / "blocked_tasks.jsonl"
        summary_path = log_dir / "summary_time_cost.json"

        task_order = [self._task_key(task.task_id) for task in all_tasks]
        results_by_key = self._load_existing_benchmark_results(benchmark_results_path)
        backfill_enabled = (
            os.getenv("BACKFILL_FROM_TASK_LOGS", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        print(f"Task-log backfill enabled: {backfill_enabled}")

        recovered_from_logs = 0
        if backfill_enabled:
            results_by_key, recovered_from_logs = merge_backfilled_results_from_task_logs(
                tasks=all_tasks,
                run_dir=log_dir,
                existing_results_by_key=results_by_key,
                k_value=self.evaluator.pass_at_k,
            )
            if recovered_from_logs > 0:
                print(
                    f"Recovered {recovered_from_logs} additional task(s) from task logs in {log_dir}"
                )

        if results_by_key:
            self._persist_checkpoint(
                task_order,
                results_by_key,
                benchmark_results_path,
                final_answers_path,
                blocked_tasks_path,
            )

        pending_tasks = [
            task
            for task in all_tasks
            if self._task_key(task.task_id) not in results_by_key
        ]

        print(
            f"Total tasks: {len(all_tasks)}, completed after merge: {len(results_by_key)}, pending: {len(pending_tasks)}"
        )

        if not pending_tasks:
            print("All tasks are already completed. Skipping new inference.")
            generate_summary(log_dir)
            return {
                "benchmark_results": str(benchmark_results_path),
                "final_answers": str(final_answers_path),
                "blocked_tasks": str(blocked_tasks_path),
                "summary_time_cost": str(summary_path),
            }

        print(
            f"\nStarting parallel inference with {self.cfg.benchmark.execution.max_concurrent} concurrent tasks..."
        )
        print(f"Using pass@{self.evaluator.pass_at_k} execution...")

        def _on_task_completed(result: BenchmarkResult) -> None:
            result_key = self._task_key(result.task_id)
            results_by_key[result_key] = result
            self._persist_checkpoint(
                task_order,
                results_by_key,
                benchmark_results_path,
                final_answers_path,
                blocked_tasks_path,
            )
            print(
                f"Checkpoint updated: {len(results_by_key)}/{len(all_tasks)} task(s) saved"
            )

        new_results = self.evaluator.run_parallel_inference_with_callback(
            pending_tasks,
            max_concurrent=self.cfg.benchmark.execution.max_concurrent,
            on_result=_on_task_completed,
        )

        for result in new_results:
            results_by_key[self._task_key(result.task_id)] = result

        self._persist_checkpoint(
            task_order,
            results_by_key,
            benchmark_results_path,
            final_answers_path,
            blocked_tasks_path,
        )

        generate_summary(log_dir)

        print("\nInference completed successfully!")
        print(f"  benchmark_results: {benchmark_results_path}")
        print(f"  final_answers: {final_answers_path}")
        print(f"  blocked_tasks: {blocked_tasks_path}")
        print(f"  summary: {summary_path}")

        return {
            "benchmark_results": str(benchmark_results_path),
            "final_answers": str(final_answers_path),
            "blocked_tasks": str(blocked_tasks_path),
            "summary_time_cost": str(summary_path),
        }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_jsonl_inference(cfg: DictConfig) -> None:
    print("Inference configuration:\n", OmegaConf.to_yaml(cfg.benchmark))

    runner = JSONLInferenceRunner(cfg)
    artifacts = runner.run_inference()

    if artifacts:
        print("\nGenerated artifacts:")
        for name, path in artifacts.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    run_jsonl_inference()
