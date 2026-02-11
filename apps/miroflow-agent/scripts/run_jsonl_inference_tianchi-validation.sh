#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$APP_DIR"

# Parse environment variables, use defaults if not set
INPUT_JSONL=${INPUT_JSONL:-"../../data/tianchi/standardized_data.jsonl"}
LLM_CONFIG=${LLM_CONFIG:-"deepseek-v3-2"}
LLM_PROVIDER=${LLM_PROVIDER:-"deepseek"}
LLM_MODEL=${LLM_MODEL:-"deepseek-chat"}
BASE_URL=${BASE_URL:-"https://api.deepseek.com"}
API_KEY=${API_KEY:-"xxx"}
AGENT_SET=${AGENT_SET:-"mirothinker_v1.5_keep5_max200"}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-262144}
MAX_CONCURRENT=${MAX_CONCURRENT:-10}
PASS_AT_K=${PASS_AT_K:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
RESUME=${RESUME:-"true"}
BACKFILL_FROM_TASK_LOGS=${BACKFILL_FROM_TASK_LOGS:-"true"}

BENCHMARK_NAME="tianchi-validation"

INPUT_DATA_DIR="$(dirname "$INPUT_JSONL")"
INPUT_METADATA_FILE="$(basename "$INPUT_JSONL")"
RESULTS_BASE_DIR="../../logs/${BENCHMARK_NAME}/${LLM_PROVIDER}_${LLM_MODEL}_${AGENT_SET}"

DEFAULT_RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_ID:-}" ]]; then
    SELECTED_RUN_ID="${RUN_ID}"
elif [[ "${RESUME}" == "true" && -d "${RESULTS_BASE_DIR}" ]]; then
    LATEST_RUN_DIR="$(find "${RESULTS_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
    if [[ -n "${LATEST_RUN_DIR}" ]]; then
        SELECTED_RUN_ID="$(basename "${LATEST_RUN_DIR}")"
    else
        SELECTED_RUN_ID="${DEFAULT_RUN_ID}"
    fi
else
    SELECTED_RUN_ID="${DEFAULT_RUN_ID}"
fi

RUN_ID="${SELECTED_RUN_ID}"
RESULTS_DIR="${RESULTS_BASE_DIR}/${RUN_ID}"
RUN_OUTPUT_LOG="${RESULTS_DIR}/${RUN_ID}_output.log"

echo "=========================================="
echo "Starting single-run JSONL inference..."
echo "Input JSONL: ${INPUT_JSONL}"
echo "Benchmark: ${BENCHMARK_NAME}"
echo "Run ID: ${RUN_ID}"
echo "Resume mode: ${RESUME}"
echo "Backfill from task logs: ${BACKFILL_FROM_TASK_LOGS}"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="

mkdir -p "$RESULTS_DIR"

if [[ "${BACKFILL_FROM_TASK_LOGS}" == "true" ]]; then
    echo "Running pre-inference backfill from existing task logs..."
    uv run python jsonl_inference/backfill_from_task_logs.py \
        --run-dir "$RESULTS_DIR" \
        --input-jsonl "$INPUT_JSONL"
fi

export BACKFILL_FROM_TASK_LOGS

uv run python jsonl_inference/run_jsonl_inference.py \
    benchmark=$BENCHMARK_NAME \
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
echo "JSONL inference completed!"
echo "final_answers.jsonl: ${FINAL_ANSWERS_FILE}"
echo "benchmark_results.jsonl: ${BENCHMARK_RESULTS_FILE}"
echo "blocked_tasks.jsonl: ${BLOCKED_TASKS_FILE}"
echo "summary_time_cost.json: ${SUMMARY_FILE}"
echo "run output log: ${RUN_OUTPUT_LOG}"
echo "=========================================="
