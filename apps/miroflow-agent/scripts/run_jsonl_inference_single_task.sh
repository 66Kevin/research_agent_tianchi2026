#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$APP_DIR"

TASK_ID_INPUT="${1:-${TASK_ID:-}}"
if [[ -z "${TASK_ID_INPUT}" ]]; then
    echo "Usage: $0 <task_id>"
    echo "or set env: TASK_ID=<task_id> $0"
    exit 1
fi

# Parse environment variables, use defaults if not set
INPUT_JSONL=${INPUT_JSONL:-"../../data/gaia-2023-validation/standardized_data.jsonl"}
LLM_CONFIG=${LLM_CONFIG:-"deepseek-v3-2"}
LLM_PROVIDER=${LLM_PROVIDER:-"deepseek"}
LLM_MODEL=${LLM_MODEL:-"deepseek-chat"}
BASE_URL=${BASE_URL:-"https://api.deepseek.com"}
API_KEY=${API_KEY:-"xxx"}
AGENT_SET=${AGENT_SET:-"mirothinker_v1.5_keep5_max200"}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-262144}
MAX_CONCURRENT=${MAX_CONCURRENT:-1}
PASS_AT_K=${PASS_AT_K:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
BACKFILL_FROM_TASK_LOGS=${BACKFILL_FROM_TASK_LOGS:-"false"}

BENCHMARK_NAME=${BENCHMARK_NAME:-"gaia-validation-single-task"}
TASK_ID_SAFE="$(echo "${TASK_ID_INPUT}" | tr -c '[:alnum:]_.-' '_')"
RESULTS_BASE_DIR="../../logs/${BENCHMARK_NAME}/${LLM_PROVIDER}_${LLM_MODEL}_${AGENT_SET}"
DEFAULT_RUN_ID="single_task_${TASK_ID_SAFE}_$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}}"
RESULTS_DIR="${RESULTS_BASE_DIR}/${RUN_ID}"
RUN_OUTPUT_LOG="${RESULTS_DIR}/${RUN_ID}_output.log"
FILTERED_INPUT_JSONL="${RESULTS_DIR}/single_task_input_${TASK_ID_SAFE}.jsonl"

echo "=========================================="
echo "Starting single-task JSONL inference..."
echo "Task ID: ${TASK_ID_INPUT}"
echo "Source JSONL: ${INPUT_JSONL}"
echo "Benchmark: ${BENCHMARK_NAME}"
echo "Run ID: ${RUN_ID}"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="

mkdir -p "$RESULTS_DIR"

TASK_ID="${TASK_ID_INPUT}" INPUT_JSONL="$INPUT_JSONL" FILTERED_INPUT_JSONL="$FILTERED_INPUT_JSONL" \
uv run python - <<'PY'
import json
import os
from pathlib import Path

task_id = os.environ["TASK_ID"]
input_path = Path(os.environ["INPUT_JSONL"]).resolve()
output_path = Path(os.environ["FILTERED_INPUT_JSONL"]).resolve()

if not input_path.exists():
    raise FileNotFoundError(f"Input JSONL not found: {input_path}")

matched_rows = []
with input_path.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {line_number}: {e}") from e
        if "task_id" not in row:
            continue
        if str(row["task_id"]) == str(task_id):
            matched_rows.append(row)

if not matched_rows:
    raise ValueError(f"Task id {task_id} not found in {input_path}")

if len(matched_rows) > 1:
    raise ValueError(
        f"Task id {task_id} appears {len(matched_rows)} times in {input_path}; expected unique"
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    f.write(json.dumps(matched_rows[0], ensure_ascii=False) + "\n")

print(f"Prepared single-task input: {output_path}")
PY

INPUT_DATA_DIR="$(dirname "$FILTERED_INPUT_JSONL")"
INPUT_METADATA_FILE="$(basename "$FILTERED_INPUT_JSONL")"

export BACKFILL_FROM_TASK_LOGS

uv run python jsonl_inference/run_jsonl_inference.py \
    benchmark=gaia-validation \
    benchmark.data.data_dir="$INPUT_DATA_DIR" \
    benchmark.data.metadata_file="$INPUT_METADATA_FILE" \
    llm="$LLM_CONFIG" \
    llm.provider="$LLM_PROVIDER" \
    llm.model_name="$LLM_MODEL" \
    llm.base_url="$BASE_URL" \
    llm.async_client=true \
    llm.temperature="$TEMPERATURE" \
    llm.max_context_length="$MAX_CONTEXT_LENGTH" \
    llm.api_key="$API_KEY" \
    benchmark.execution.max_tasks=null \
    benchmark.execution.max_concurrent="$MAX_CONCURRENT" \
    benchmark.execution.pass_at_k="$PASS_AT_K" \
    agent="$AGENT_SET" \
    hydra.run.dir="$RESULTS_DIR" \
    2>&1 | tee "$RUN_OUTPUT_LOG"

FINAL_ANSWERS_FILE="${RESULTS_DIR}/final_answers.jsonl"
BENCHMARK_RESULTS_FILE="${RESULTS_DIR}/benchmark_results.jsonl"
SUMMARY_FILE="${RESULTS_DIR}/summary_time_cost.json"
BLOCKED_TASKS_FILE="${RESULTS_DIR}/blocked_tasks.jsonl"

echo "=========================================="
echo "Single-task inference completed!"
echo "task_id: ${TASK_ID_INPUT}"
echo "single-task input: ${FILTERED_INPUT_JSONL}"
echo "final_answers.jsonl: ${FINAL_ANSWERS_FILE}"
echo "benchmark_results.jsonl: ${BENCHMARK_RESULTS_FILE}"
echo "blocked_tasks.jsonl: ${BLOCKED_TASKS_FILE}"
echo "summary_time_cost.json: ${SUMMARY_FILE}"
echo "run output log: ${RUN_OUTPUT_LOG}"
echo "=========================================="

