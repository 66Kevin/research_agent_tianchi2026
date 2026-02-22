import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TASK_LOG_PATTERN = "task_*_attempt-1_format-retry-0_*.json"
TASK_ID_RE = re.compile(r"task_(\d+)_attempt-1_format-retry-0_")
TOOL_CALL_RE = re.compile(
    r"Function '_execute_tool_call' called with args:\s*([a-zA-Z0-9_]+)\s*:\s*executed in\s*([0-9.]+)\s*seconds"
)


def _sanitize_mermaid_text(value: str) -> str:
    # Keep labels as plain text to avoid parser collisions with Mermaid/HTML syntax.
    replacements = {
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
        '"': "'",
        "<": "&lt;",
        ">": "&gt;",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value.replace("%%", "% %")


def _clean_text(text: Any, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    value = re.sub(r"\s+", " ", value)
    value = _sanitize_mermaid_text(value)
    if len(value) <= limit:
        return value
    return value[:limit] + " ..."


def _extract_task_id(path: Path) -> int | None:
    match = TASK_ID_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _group_latest_by_task(paths: Iterable[Path]) -> Dict[int, Path]:
    latest: Dict[int, Path] = {}
    for p in paths:
        task_id = _extract_task_id(p)
        if task_id is None:
            continue
        old = latest.get(task_id)
        if old is None or p.name > old.name:
            latest[task_id] = p
    return latest


def _tool_call_detail(message: str) -> Tuple[str, str] | None:
    match = TOOL_CALL_RE.search(message)
    if not match:
        return None
    return match.group(1), match.group(2)


def _append_event_lines(
    step_name: str,
    message: str,
    metadata: Dict[str, Any],
    lines: List[str],
    mode: str,
) -> None:
    src = metadata.get("source_agent", "")
    source = "M" if src == "main_agent" else "S"
    target = "M" if source == "S" else "S"
    clean_msg = _clean_text(message, limit=140)

    if step_name == "Main | Task Start":
        lines.append("Note over M: Task execution started")
        return

    if step_name == "task_execution_finished":
        lines.append("Note over M: Task execution finished")
        return

    if "LLM Prompt" in step_name:
        lines.append(f"{source}->>L: prompt | {clean_msg}")
        return

    if "LLM Response" in step_name:
        lines.append(f"L-->>{source}: response | {clean_msg}")
        return

    if "Tool Call" in step_name:
        detail = _tool_call_detail(message)
        if detail:
            tool_name, seconds = detail
            lines.append(f"{source}->>T: {tool_name}()")
            lines.append(f"T-->>{source}: done in {seconds}s")
        else:
            tool_name = metadata.get("tool_name", "tool")
            lines.append(f"{source}->>T: {tool_name}()")
            lines.append(f"T-->>{source}: {_clean_text(message, 90)}")
        return

    if "Behavior" in step_name:
        lower = message.lower()
        if "starting execution of step" in lower:
            lines.append(f"M->>S: {_clean_text(message, 90)}")
        elif "completed execution of step" in lower:
            lines.append(f"S-->>M: {_clean_text(message, 90)}")
        else:
            lines.append(f"Note over {source}: {_clean_text(message, 90)}")
        return

    if "Max Iteration" in step_name:
        lines.append(f"Note over {source}: max iteration reached")
        return

    if mode == "full" and ("Speech" in step_name or "Runtime" in step_name):
        lines.append(f"Note over {source}: {clean_msg}")


def build_sequence_mermaid(task_payload: Dict[str, Any], source_path: Path, mode: str, max_events: int) -> str:
    task_id = task_payload.get("task_id", source_path.stem)
    status = task_payload.get("status", "")
    elapsed = task_payload.get("elapsed_seconds", "")
    question = _clean_text(task_payload.get("input", {}).get("question", ""), limit=100)

    lines: List[str] = [
        "sequenceDiagram",
        "autonumber",
        "participant U as User",
        "participant M as MainAgent",
        "participant S as SubAgent",
        'participant L as "LLM(qwen3.5-plus)"',
        "participant T as Tools",
        f"U->>M: Task {task_id} | {question}",
        f"Note right of M: status={status}, elapsed={elapsed}s",
    ]

    step_logs = task_payload.get("step_logs", [])
    for event in step_logs[:max_events]:
        step_name = str(event.get("step_name", ""))
        message = str(event.get("message", ""))
        metadata = event.get("metadata", {}) or {}
        _append_event_lines(step_name, message, metadata, lines, mode=mode)

    if len(step_logs) > max_events:
        lines.append(f"Note over M: {len(step_logs) - max_events} events omitted")

    return "\n".join(lines) + "\n"


def export_sequences(
    task_log_dir: Path,
    output_dir: Path,
    task_ids: List[int] | None,
    mode: str,
    max_events: int,
    output_format: str,
) -> List[Tuple[int, Path, Path]]:
    if not task_log_dir.exists():
        raise FileNotFoundError(f"task log dir not found: {task_log_dir}")

    all_paths = list(task_log_dir.glob(TASK_LOG_PATTERN))
    latest_by_task = _group_latest_by_task(all_paths)
    if task_ids:
        chosen_items = [(task_id, latest_by_task[task_id]) for task_id in task_ids if task_id in latest_by_task]
    else:
        chosen_items = sorted(latest_by_task.items(), key=lambda x: x[0])

    output_dir.mkdir(parents=True, exist_ok=True)

    exported: List[Tuple[int, Path, Path]] = []
    for task_id, src_path in chosen_items:
        with src_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        mermaid = build_sequence_mermaid(payload, src_path, mode=mode, max_events=max_events)
        suffix = "md" if output_format == "md" else "mmd"
        out_path = output_dir / f"task_{task_id}_sequence.{suffix}"
        with out_path.open("w", encoding="utf-8") as f:
            if output_format == "md":
                f.write("# Sequence Diagram\n\n```mermaid\n")
                f.write(mermaid)
                f.write("```\n")
            else:
                f.write(mermaid)
        exported.append((task_id, src_path, out_path))
    return exported


def parse_task_ids(raw: str | None) -> List[int] | None:
    if not raw:
        return None
    ids: List[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tianchi task logs to Mermaid sequence diagrams.")
    parser.add_argument("--task-log-dir", type=Path, required=True, help="Directory containing task_*_attempt-1_format-retry-0_*.json logs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for .mmd files (default: <task-log-dir>/mermaid)")
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated task ids, e.g. 0,1")
    parser.add_argument("--mode", choices=["compact", "full"], default="compact", help="compact keeps interaction events; full also includes runtime/speech notes")
    parser.add_argument("--max-events", type=int, default=360, help="Max step events to render per task")
    parser.add_argument("--output-format", choices=["mmd", "md"], default="mmd", help="Output file format")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.task_log_dir / "mermaid")
    task_ids = parse_task_ids(args.task_ids)

    exported = export_sequences(
        task_log_dir=args.task_log_dir,
        output_dir=output_dir,
        task_ids=task_ids,
        mode=args.mode,
        max_events=args.max_events,
        output_format=args.output_format,
    )

    if not exported:
        print("No task logs matched.")
        return

    for task_id, src, out in exported:
        print(f"[task={task_id}] source={src} mermaid={out}")


if __name__ == "__main__":
    main()
