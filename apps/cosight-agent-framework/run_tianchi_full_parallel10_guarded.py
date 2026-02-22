import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


GOOGLE_FAIL_LINE_RE = (
    r"serper search failed|google search failed|google search unavailable"
)
JINA_FAIL_LINE_RE = (
    r"(?:\bjina\b.*(?:failed|error|timeout|429|500|503|unavailable))"
    r"|(?:(?:failed|error|timeout|429|500|503|unavailable).*\bjina\b)"
)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def scan_new_runtime_lines(
    task_log_dir: Path,
    offsets: Dict[str, int],
) -> Tuple[int, int]:
    import re

    google_inc = 0
    jina_inc = 0
    google_re = re.compile(GOOGLE_FAIL_LINE_RE, re.IGNORECASE)
    jina_re = re.compile(JINA_FAIL_LINE_RE, re.IGNORECASE)

    for log_path in sorted(task_log_dir.glob("task_*_runtime_*.log")):
        key = str(log_path)
        offset = offsets.get(key, 0)
        try:
            size = log_path.stat().st_size
            if size < offset:
                offset = 0
            with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                for line in f:
                    if google_re.search(line):
                        google_inc += 1
                    if jina_re.search(line):
                        jina_inc += 1
                offsets[key] = f.tell()
        except FileNotFoundError:
            continue
    return google_inc, jina_inc


def terminate_process_group(proc: subprocess.Popen, grace_seconds: int = 10) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    except Exception:
        pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full Tianchi set and stop if Google/Jina failures exceed threshold."
    )
    parser.add_argument("--serper-api-key", required=True, help="SERPER_API_KEY to use for this run.")
    parser.add_argument("--jina-api-key", required=True, help="JINA_API_KEY to use for this run.")
    parser.add_argument("--failure-threshold", type=int, default=10, help="Stop once either Google or Jina failure count exceeds this value.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-question timeout in seconds.")
    parser.add_argument("--parallel", type=int, default=10, help="Question-level concurrency.")
    parser.add_argument("--jina-enrich-top-k", type=int, default=1, help="Number of top Serper results to enrich with Jina reader.")
    parser.add_argument("--jina-enrich-timeout", type=int, default=8, help="Per-request timeout (seconds) for Jina reader.")
    parser.add_argument("--jina-enrich-max-chars", type=int, default=1200, help="Max chars retained from each Jina reader response.")
    parser.add_argument("--poll-seconds", type=int, default=3, help="Failure scan interval.")
    parser.add_argument("--report-seconds", type=int, default=20, help="Progress print interval.")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent
    out_base = cwd.parent
    ts = now_ts()

    run_tag = f"qwen_qwen3.5-plus_cosight_agentlog_v7_p{args.parallel}_serper_jina_guard_{ts}"
    task_log_dir = out_base / "logs" / "tianchi-validation" / run_tag
    answers_file = out_base / f"full100_answers_{run_tag}.jsonl"
    runlog_file = out_base / f"full100_runlog_{run_tag}.json"
    console_file = out_base / f"full100_console_{run_tag}.log"
    guard_report_file = out_base / f"full100_guard_report_{run_tag}.json"

    env = os.environ.copy()
    env["WORKSPACE_PATH"] = str(cwd)
    env["SERPER_API_KEY"] = args.serper_api_key.strip()
    env["JINA_API_KEY"] = args.jina_api_key.strip()
    env["JINA_ENRICH_ENABLED"] = "1"
    env["JINA_ENRICH_TOP_K"] = str(max(0, args.jina_enrich_top_k))
    env["JINA_ENRICH_TIMEOUT"] = str(max(2, args.jina_enrich_timeout))
    env["JINA_ENRICH_MAX_CHARS"] = str(max(200, args.jina_enrich_max_chars))

    cmd = [
        str(cwd / ".venv" / "bin" / "python"),
        str(cwd / "run_tianchi_first2_with_agent_logs.py"),
        "--question-file",
        str(out_base / "question.jsonl"),
        "--output-file",
        str(answers_file),
        "--log-file",
        str(runlog_file),
        "--task-log-dir",
        str(task_log_dir),
        "--count",
        "0",
        "--parallel",
        str(args.parallel),
        "--timeout",
        str(args.timeout),
    ]

    print(f"TASK_LOG_DIR={task_log_dir}")
    print(f"ANS_FILE={answers_file}")
    print(f"RUN_FILE={runlog_file}")
    print(f"CONSOLE_FILE={console_file}")
    print(f"GUARD_REPORT_FILE={guard_report_file}")
    print(f"FAILURE_THRESHOLD={args.failure_threshold}")
    print(f"JINA_ENRICH_TOP_K={env['JINA_ENRICH_TOP_K']}")
    print(f"JINA_ENRICH_TIMEOUT={env['JINA_ENRICH_TIMEOUT']}")
    print(f"JINA_ENRICH_MAX_CHARS={env['JINA_ENRICH_MAX_CHARS']}")
    sys.stdout.flush()

    task_log_dir.mkdir(parents=True, exist_ok=True)
    console_file.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    google_fail_total = 0
    jina_fail_total = 0
    offsets: Dict[str, int] = {}
    stop_reason = ""

    with console_file.open("w", encoding="utf-8") as console_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=console_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

        last_report = 0.0
        while True:
            g_inc, j_inc = scan_new_runtime_lines(task_log_dir, offsets)
            google_fail_total += g_inc
            jina_fail_total += j_inc

            now = time.time()
            if now - last_report >= args.report_seconds:
                attempt_count = len(list(task_log_dir.glob("task_*_attempt-1_format-retry-0_*.json")))
                runtime_count = len(list(task_log_dir.glob("task_*_runtime_*.log")))
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"pid={proc.pid} attempts={attempt_count} runtime_logs={runtime_count} "
                    f"google_fail={google_fail_total} jina_fail={jina_fail_total}"
                )
                sys.stdout.flush()
                last_report = now

            if google_fail_total > args.failure_threshold:
                stop_reason = (
                    f"google_fail_exceeded: {google_fail_total} > {args.failure_threshold}"
                )
                terminate_process_group(proc)
                break

            if jina_fail_total > args.failure_threshold:
                stop_reason = f"jina_fail_exceeded: {jina_fail_total} > {args.failure_threshold}"
                terminate_process_group(proc)
                break

            rc = proc.poll()
            if rc is not None:
                break

            time.sleep(max(1, args.poll_seconds))

        # Final scan after process termination.
        g_inc, j_inc = scan_new_runtime_lines(task_log_dir, offsets)
        google_fail_total += g_inc
        jina_fail_total += j_inc

        return_code = proc.poll()

    elapsed = round(time.time() - start_time, 2)
    attempt_count = len(list(task_log_dir.glob("task_*_attempt-1_format-retry-0_*.json")))
    runtime_count = len(list(task_log_dir.glob("task_*_runtime_*.log")))
    result_count = len(list(task_log_dir.glob("task_*_runtime_*.result.json")))

    report = {
        "task_log_dir": str(task_log_dir),
        "answers_file": str(answers_file),
        "runlog_file": str(runlog_file),
        "console_file": str(console_file),
        "elapsed_seconds": elapsed,
        "process_return_code": return_code,
        "stop_reason": stop_reason,
        "failure_threshold": args.failure_threshold,
        "google_fail_total": google_fail_total,
        "jina_fail_total": jina_fail_total,
        "attempt_count": attempt_count,
        "runtime_count": runtime_count,
        "result_count": result_count,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    guard_report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
