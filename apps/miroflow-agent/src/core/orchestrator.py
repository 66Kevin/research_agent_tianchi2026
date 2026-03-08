# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Orchestrator module for coordinating agent task execution.

This module contains the main Orchestrator class that manages the execution of tasks
by coordinating between the main agent, sub-agents, and various tools.
"""

import asyncio
import gc
import logging
import time
import uuid
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

from miroflow_tools.manager import ToolManager
from omegaconf import DictConfig

from ..config.settings import expose_sub_agents_as_tools
from ..io.input_handler import process_input
from ..io.output_formatter import OutputFormatter
from ..llm.base_client import BaseClient
from ..llm.errors import PolicyBlockedError
from ..logging.task_logger import TaskLog, get_utc_plus_8_time
from ..utils.parsing_utils import extract_llm_response_text
from ..utils.localization_gate_utils import (
    LocalizationGateBudgetDecision,
    LocalizationGateDecision,
    decide_localization_gate_mode_from_remaining,
    parse_localization_gate_decision,
    should_run_localization_gate,
)
from ..utils.prompt_utils import (
    BLOCKED_BY_POLICY_MESSAGE,
    generate_agent_specific_system_prompt,
    generate_agent_summarize_prompt,
    generate_localization_gate_decision_prompt,
    generate_localization_gate_prompt,
    generate_localization_gate_result_prompt,
    mcp_tags,
    refusal_keywords,
)
from ..utils.temperature_utils import resolve_temperature
from .answer_generator import AnswerGenerator
from .stream_handler import StreamHandler
from .tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default timeout for LLM calls in seconds
DEFAULT_LLM_TIMEOUT = 600

# Safety limits for retry loops
DEFAULT_MAX_CONSECUTIVE_ROLLBACKS = 5

# Additional attempts beyond max_turns for total loop protection
EXTRA_ATTEMPTS_BUFFER = 200

# Hard limit for budgeted web tool calls. Set to 0 to disable.
DEFAULT_WEB_TOOL_CALL_HARD_LIMIT = 0
LOCALIZATION_GATE_AGENT_NAME = "Pre-Summary Localization Gate"
LOCALIZATION_GATE_ALLOWED_TOOLS = {
    ("search_and_scrape_webpage", "google_search"),
    ("jina_scrape_llm_summary", "scrape_and_extract_info"),
}


def _list_tools(sub_agent_tool_managers: Dict[str, ToolManager]):
    """
    Create a cached async function for fetching sub-agent tool definitions.

    This factory function returns an async closure that lazily fetches and caches
    tool definitions from all sub-agent tool managers. The cache ensures that
    tool definitions are only fetched once per orchestrator instance.

    Args:
        sub_agent_tool_managers: Dictionary mapping sub-agent names to their ToolManager instances.

    Returns:
        An async function that returns a dictionary of tool definitions for each sub-agent.
    """
    cache = None

    async def wrapped():
        nonlocal cache
        if cache is None:
            # Only fetch tool definitions if not already cached
            result = {
                name: await tool_manager.get_all_tool_definitions()
                for name, tool_manager in sub_agent_tool_managers.items()
            }
            cache = result
        return cache

    return wrapped


class Orchestrator:
    """
    Main orchestrator for coordinating agent task execution.

    Manages the execution loop for main and sub-agents, coordinating
    LLM calls, tool execution, streaming events, and context management.
    """

    def __init__(
        self,
        main_agent_tool_manager: ToolManager,
        sub_agent_tool_managers: Dict[str, ToolManager],
        llm_client: BaseClient,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: Optional["TaskLog"] = None,
        stream_queue: Optional[Any] = None,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        sub_agent_tool_definitions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            main_agent_tool_manager: Tool manager for main agent
            sub_agent_tool_managers: Dictionary of tool managers for sub-agents
            llm_client: The LLM client for API calls
            output_formatter: Formatter for output processing
            cfg: Configuration object
            task_log: Logger for task execution
            stream_queue: Optional async queue for streaming events
            tool_definitions: Pre-fetched tool definitions (optional)
            sub_agent_tool_definitions: Pre-fetched sub-agent tool definitions (optional)
        """
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        self.stream_queue = stream_queue
        self.tool_definitions = tool_definitions
        self.sub_agent_tool_definitions = sub_agent_tool_definitions

        # Initialize sub-agent tool list function
        self._list_sub_agent_tools = None
        if sub_agent_tool_managers:
            self._list_sub_agent_tools = _list_tools(sub_agent_tool_managers)

        # Pass task_log to llm_client
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log

        # Track boxed answers extracted during main loop turns
        self.intermediate_boxed_answers: List[str] = []

        # Record used subtask / q / Query to detect duplicates
        self.used_queries: Dict[str, Dict[str, int]] = {}

        # Retry loop protection limits
        self.MAX_CONSECUTIVE_ROLLBACKS = DEFAULT_MAX_CONSECUTIVE_ROLLBACKS

        # Context management settings
        self.context_compress_limit = cfg.agent.get("context_compress_limit", 0)
        try:
            self.web_tool_call_hard_limit = int(
                self.cfg.agent.main_agent.get(
                    "web_tool_call_hard_limit", DEFAULT_WEB_TOOL_CALL_HARD_LIMIT
                )
                or 0
            )
        except (TypeError, ValueError):
            self.web_tool_call_hard_limit = DEFAULT_WEB_TOOL_CALL_HARD_LIMIT
        self.task_status = "running"
        self.blocked_reason: Optional[str] = None
        self.current_agent_id: Optional[str] = None

        # Initialize helper components
        self.stream = StreamHandler(stream_queue)
        self.tool_executor = ToolExecutor(
            main_agent_tool_manager=main_agent_tool_manager,
            sub_agent_tool_managers=sub_agent_tool_managers,
            output_formatter=output_formatter,
            task_log=task_log,
            stream_handler=self.stream,
            max_consecutive_rollbacks=DEFAULT_MAX_CONSECUTIVE_ROLLBACKS,
        )
        self.answer_generator = AnswerGenerator(
            llm_client=llm_client,
            output_formatter=output_formatter,
            task_log=task_log,
            stream_handler=self.stream,
            cfg=cfg,
            intermediate_boxed_answers=self.intermediate_boxed_answers,
        )

    def _save_message_history(
        self, system_prompt: str, message_history: List[Dict[str, Any]]
    ):
        """Save message history to task log."""
        self.task_log.main_agent_message_history = {
            "system_prompt": system_prompt,
            "message_history": message_history,
        }
        self.task_log.save()

    async def _handle_policy_blocked(
        self, task_id: str, workflow_id: str, blocked_reason: str
    ) -> tuple[str, str, Optional[str]]:
        """Finalize the current task as blocked and clean up streams/workflow."""
        self.task_status = "blocked"
        self.blocked_reason = blocked_reason
        self.task_log.trace_data["blocked_reason"] = blocked_reason
        self.task_log.log_step(
            "error",
            "Main Agent | Policy Blocked",
            f"Task {task_id} blocked by provider safety policy: {blocked_reason}",
        )

        # Best-effort cleanup: some streams may not be active depending on where block happened.
        try:
            await self.stream.end_llm("Final Summary")
        except Exception:
            pass
        try:
            await self.stream.end_agent("Final Summary", self.current_agent_id)
        except Exception:
            pass
        try:
            await self.stream.end_llm(LOCALIZATION_GATE_AGENT_NAME)
        except Exception:
            pass
        try:
            await self.stream.end_agent(
                LOCALIZATION_GATE_AGENT_NAME, self.current_agent_id
            )
        except Exception:
            pass
        try:
            await self.stream.end_llm("main")
        except Exception:
            pass
        try:
            await self.stream.end_agent("main", self.current_agent_id)
        except Exception:
            pass
        try:
            await self.stream.end_workflow(workflow_id)
        except Exception:
            pass

        gc.collect()
        return (
            f"Task blocked by provider safety policy: {blocked_reason}",
            BLOCKED_BY_POLICY_MESSAGE,
            None,
        )

    async def _handle_response_format_issues(
        self,
        assistant_response_text: str,
        message_history: List[Dict[str, Any]],
        turn_count: int,
        consecutive_rollbacks: int,
        total_attempts: int,
        max_attempts: int,
        agent_name: str,
    ) -> tuple:
        """
        Handle MCP tag format errors and refusal keywords.

        Args:
            assistant_response_text: The LLM response text
            message_history: Current message history
            turn_count: Current turn count
            consecutive_rollbacks: Current consecutive rollback count
            total_attempts: Total attempts made
            max_attempts: Maximum allowed attempts
            agent_name: Name of the agent for logging

        Returns:
            Tuple of (should_continue, should_break, turn_count, consecutive_rollbacks, message_history)
        """
        # Check for MCP tags in response (format error)
        if any(mcp_tag in assistant_response_text for mcp_tag in mcp_tags):
            if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                turn_count -= 1
                consecutive_rollbacks += 1
                if message_history[-1]["role"] == "assistant":
                    message_history.pop()
                self.task_log.log_step(
                    "warning",
                    f"{agent_name} | Turn: {turn_count} | Rollback",
                    f"Tool call format incorrect - found MCP tags in response. "
                    f"Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, "
                    f"Total attempts: {total_attempts}/{max_attempts}",
                )
                return True, False, turn_count, consecutive_rollbacks, message_history
            else:
                self.task_log.log_step(
                    "warning",
                    f"{agent_name} | Turn: {turn_count} | End After Max Rollbacks",
                    f"Ending agent loop after {consecutive_rollbacks} consecutive MCP format errors",
                )
                return False, True, turn_count, consecutive_rollbacks, message_history

        # Check for refusal keywords
        if any(keyword in assistant_response_text for keyword in refusal_keywords):
            matched_keywords = [
                kw for kw in refusal_keywords if kw in assistant_response_text
            ]
            if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                turn_count -= 1
                consecutive_rollbacks += 1
                if message_history[-1]["role"] == "assistant":
                    message_history.pop()
                self.task_log.log_step(
                    "warning",
                    f"{agent_name} | Turn: {turn_count} | Rollback",
                    f"LLM refused to answer - found refusal keywords: {matched_keywords}. "
                    f"Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, "
                    f"Total attempts: {total_attempts}/{max_attempts}",
                )
                return True, False, turn_count, consecutive_rollbacks, message_history
            else:
                self.task_log.log_step(
                    "warning",
                    f"{agent_name} | Turn: {turn_count} | End After Max Rollbacks",
                    f"Ending agent loop after {consecutive_rollbacks} consecutive refusals with keywords: {matched_keywords}",
                )
                return False, True, turn_count, consecutive_rollbacks, message_history

        # No format issues - normal end without tool calls
        return False, True, turn_count, consecutive_rollbacks, message_history

    async def _check_duplicate_query(
        self,
        tool_name: str,
        arguments: dict,
        cache_name: str,
        consecutive_rollbacks: int,
        turn_count: int,
        total_attempts: int,
        max_attempts: int,
        message_history: List[Dict[str, Any]],
        agent_name: str,
    ) -> tuple:
        """
        Check for duplicate queries and handle rollback if needed.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            cache_name: Name of the query cache to use
            consecutive_rollbacks: Current consecutive rollback count
            turn_count: Current turn count
            total_attempts: Total attempts made
            max_attempts: Maximum allowed attempts
            message_history: Current message history
            agent_name: Name of the agent for logging

        Returns:
            Tuple of (is_duplicate, should_rollback, turn_count, consecutive_rollbacks, message_history)
        """
        query_str = self.tool_executor.get_query_str_from_tool_call(
            tool_name, arguments
        )
        if not query_str:
            return False, False, turn_count, consecutive_rollbacks, message_history

        self.used_queries.setdefault(cache_name, defaultdict(int))
        count = self.used_queries[cache_name][query_str]

        if count > 0:
            if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                message_history.pop()
                consecutive_rollbacks += 1
                self.task_log.log_step(
                    "warning",
                    f"{agent_name} | Turn: {turn_count} | Rollback",
                    f"Duplicate query detected - tool: {tool_name}, query: '{query_str}', "
                    f"previous count: {count}. Consecutive rollbacks: {consecutive_rollbacks}/"
                    f"{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}. "
                    "Current turn budget is consumed to avoid infinite duplicate loops.",
                )
                return True, True, turn_count, consecutive_rollbacks, message_history
            else:
                self.task_log.log_step(
                    "warning",
                    f"{agent_name} | Turn: {turn_count} | Allow Duplicate",
                    f"Allowing duplicate query after {consecutive_rollbacks} rollbacks - "
                    f"tool: {tool_name}, query: '{query_str}', previous count: {count}",
                )

        return False, False, turn_count, consecutive_rollbacks, message_history

    async def _record_query(self, cache_name: str, tool_name: str, arguments: dict):
        """Record a successful query execution."""
        query_str = self.tool_executor.get_query_str_from_tool_call(
            tool_name, arguments
        )
        if query_str:
            self.used_queries.setdefault(cache_name, defaultdict(int))
            self.used_queries[cache_name][query_str] += 1

    @staticmethod
    def _build_tool_index(
        tool_definitions: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Set[str]]:
        """Build {server_name: {tool_name}} for fast tool-routing validation."""
        index: Dict[str, Set[str]] = {}
        if not tool_definitions:
            return index
        for server in tool_definitions:
            server_name = server.get("name")
            if not server_name:
                continue
            for tool in server.get("tools", []) or []:
                tool_name = tool.get("name")
                if tool_name:
                    index.setdefault(server_name, set()).add(tool_name)
        return index

    @staticmethod
    def _build_tool_to_server_index(
        tool_index: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Build reverse index {tool_name: {server_name}} from tool index."""
        tool_to_server: Dict[str, Set[str]] = {}
        for server_name, tools in tool_index.items():
            for tool_name in tools:
                tool_to_server.setdefault(tool_name, set()).add(server_name)
        return tool_to_server

    def _repair_tool_call_routing(
        self,
        server_name: str,
        tool_name: str,
        tool_index: Dict[str, Set[str]],
        tool_to_server_index: Dict[str, Set[str]],
    ) -> Tuple[str, str, Optional[str]]:
        """
        Attempt lightweight auto-repair for common server/tool routing mistakes.
        """
        if not tool_index:
            return server_name, tool_name, None

        if tool_name in tool_index.get(server_name, set()):
            return server_name, tool_name, None

        # Case 1: server/tool accidentally swapped.
        if tool_name in tool_index and server_name in tool_index.get(tool_name, set()):
            return (
                tool_name,
                server_name,
                "swapped server_name/tool_name",
            )

        # Case 2: tool name is valid and maps to exactly one server.
        candidate_servers = tool_to_server_index.get(tool_name, set())
        if len(candidate_servers) == 1:
            repaired_server = next(iter(candidate_servers))
            if repaired_server != server_name:
                return (
                    repaired_server,
                    tool_name,
                    f"remapped server_name '{server_name}' -> '{repaired_server}' by unique tool ownership",
                )

        return server_name, tool_name, None

    def _validate_tool_call(
        self,
        server_name: str,
        tool_name: str,
        tool_index: Dict[str, Set[str]],
    ) -> Tuple[bool, str]:
        """
        Validate tool routing using currently available tool definitions.
        """
        if not tool_index:
            return True, ""

        available_tools = tool_index.get(server_name)
        if not available_tools:
            return False, f"server '{server_name}' is not available"

        if tool_name not in available_tools:
            return (
                False,
                f"tool '{tool_name}' is not available on server '{server_name}'",
            )

        return True, ""

    @staticmethod
    def _is_budgeted_web_tool(tool_name: str) -> bool:
        """
        Identify tool calls that should count toward web search budget.
        """
        return tool_name in {
            "google_search",
            "sogou_search",
            "search_and_browse",
            "scrape",
            "scrape_website",
            "scrape_and_extract_info",
        }

    @staticmethod
    def _compute_task_deadlines(
        task_start_time: float,
        task_timeout_seconds: float,
        localization_gate_reserve_seconds: float,
        final_summary_reserve_seconds: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute task, main-loop, and gate deadlines from soft reserve settings."""
        if task_timeout_seconds <= 0:
            return None, None, None

        task_deadline = task_start_time + task_timeout_seconds
        main_loop_deadline = (
            task_deadline
            - max(0.0, localization_gate_reserve_seconds)
            - max(0.0, final_summary_reserve_seconds)
        )
        gate_deadline = task_deadline - max(0.0, final_summary_reserve_seconds)
        return task_deadline, main_loop_deadline, gate_deadline

    @staticmethod
    def _remaining_seconds(deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return None
        return deadline - time.time()

    @staticmethod
    def _filter_localization_gate_tool_definitions(
        tool_definitions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep only the tools allowed inside the pre-summary localization gate."""
        filtered: List[Dict[str, Any]] = []
        for server in tool_definitions:
            server_name = server.get("name")
            tools = []
            for tool in server.get("tools", []) or []:
                tool_name = tool.get("name")
                if (server_name, tool_name) in LOCALIZATION_GATE_ALLOWED_TOOLS:
                    tools.append(tool)
            if tools:
                filtered.append({"name": server_name, "tools": tools})
        return filtered

    @staticmethod
    def _has_recent_url_candidate(message_history: List[Dict[str, Any]], url: str) -> bool:
        """Allow direct scrape only when the exact URL appears in recent context."""
        if not url:
            return False
        recent_messages = message_history[-6:]
        for message in recent_messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )
            if url in str(content):
                return True
        return False

    @staticmethod
    def _should_run_localization_gate(
        decision: Optional[LocalizationGateDecision],
    ) -> bool:
        return should_run_localization_gate(decision)

    def _decide_localization_gate_mode(
        self,
        task_deadline: Optional[float],
        final_summary_reserve_seconds: float,
    ) -> LocalizationGateBudgetDecision:
        """Choose full, degraded, or skip mode based on remaining task time."""
        if task_deadline is None:
            return LocalizationGateBudgetDecision(
                mode="full",
                remaining_seconds=float("inf"),
                reason="Task deadline is not configured; defaulting to full gate.",
            )

        remaining_seconds = task_deadline - time.time()
        full_min_remaining_seconds = float(
            self.cfg.benchmark.execution.get(
                "localization_gate_full_min_remaining_seconds", 20
            )
        )
        degraded_min_remaining_seconds = float(
            self.cfg.benchmark.execution.get(
                "localization_gate_degraded_min_remaining_seconds", 8
            )
        )
        return decide_localization_gate_mode_from_remaining(
            remaining_seconds=remaining_seconds,
            final_summary_reserve_seconds=final_summary_reserve_seconds,
            full_min_remaining_seconds=full_min_remaining_seconds,
            degraded_min_remaining_seconds=degraded_min_remaining_seconds,
        )

    async def _run_localization_gate_decision(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        task_description: str,
        turn_count: int,
    ) -> Optional[LocalizationGateDecision]:
        """Run the JSON-only decision call that determines whether the gate is needed."""
        decision_history = [message.copy() for message in message_history]
        decision_history.append(
            {
                "role": "user",
                "content": generate_localization_gate_decision_prompt(task_description),
            }
        )

        (
            decision_text,
            _,
            tool_calls,
            _,
        ) = await self.answer_generator.handle_llm_call(
            system_prompt,
            decision_history,
            [],
            turn_count + 1000,
            "Main Agent | Localization Gate Decision",
            agent_type="main",
            temperature_override=resolve_temperature(self.cfg, "final_summary"),
        )

        if tool_calls:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate Decision",
                "Decision step attempted tool use. Skipping localization gate.",
            )
            return None

        decision = parse_localization_gate_decision(decision_text or "")
        if decision is None:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate Decision",
                "Failed to parse localization gate decision JSON. Skipping gate.",
            )
            return None

        self.task_log.log_step(
            "info",
            "Main Agent | Localization Gate Decision",
            (
                f"candidate='{decision.candidate_answer}', entity_type={decision.entity_type}, "
                f"question_language={decision.question_language}, "
                f"candidate_language={decision.candidate_answer_language}, "
                f"localized_name_status={decision.localized_name_status}, "
                f"should_run_gate={decision.should_run_gate}"
            ),
        )
        return decision

    async def _execute_localization_gate_tool_call(
        self,
        tool_call: Dict[str, Any],
        message_history: List[Dict[str, Any]],
        tool_index: Dict[str, Set[str]],
        tool_to_server_index: Dict[str, Set[str]],
    ) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], bool]:
        """Execute one gate tool call with stricter routing and no rollback logic."""
        server_name = tool_call["server_name"]
        tool_name = tool_call["tool_name"]
        arguments = self.tool_executor.fix_tool_call_arguments(
            tool_name, tool_call["arguments"]
        )
        call_id = tool_call["id"]

        repaired_server_name, repaired_tool_name, repair_reason = (
            self._repair_tool_call_routing(
                server_name=server_name,
                tool_name=tool_name,
                tool_index=tool_index,
                tool_to_server_index=tool_to_server_index,
            )
        )
        if repair_reason:
            self.task_log.log_step(
                "info",
                "Main Agent | Localization Gate Routing Repaired",
                (
                    f"Auto-repaired routing ({repair_reason}): "
                    f"{server_name}/{tool_name} -> {repaired_server_name}/{repaired_tool_name}"
                ),
            )
            server_name = repaired_server_name
            tool_name = repaired_tool_name

        is_valid, validation_error = self._validate_tool_call(
            server_name, tool_name, tool_index
        )
        if not is_valid:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate",
                f"Invalid gate tool routing: {validation_error}. Ending gate.",
            )
            return None, True

        if (server_name, tool_name) not in LOCALIZATION_GATE_ALLOWED_TOOLS:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate",
                f"Disallowed gate tool requested: {server_name}/{tool_name}. Ending gate.",
            )
            return None, True

        if (
            tool_name == "scrape_and_extract_info"
            and not self._has_recent_url_candidate(
                message_history, str(arguments.get("url", ""))
            )
        ):
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate",
                "First scrape call lacked a recent URL candidate in context. Ending gate.",
            )
            return None, True

        call_start_time = time.time()
        tool_call_id = await self.stream.tool_call(tool_name, arguments)
        try:
            tool_result = await self.main_agent_tool_manager.execute_tool_call(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
            )
            tool_result = self.tool_executor.post_process_tool_call_result(
                tool_name, tool_result
            )
            result = tool_result.get("result") or tool_result.get("error")
            await self.stream.tool_call(
                tool_name, {"result": result}, tool_call_id=tool_call_id
            )
        except PolicyBlockedError:
            raise
        except Exception as e:
            tool_result = {
                "server_name": server_name,
                "tool_name": tool_name,
                "error": str(e),
            }
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate",
                f"Gate tool {tool_name} failed: {str(e)}",
            )

        call_duration_ms = int((time.time() - call_start_time) * 1000)
        self.task_log.log_step(
            "info",
            "Main Agent | Localization Gate Tool Call",
            f"Tool {tool_name} completed in {call_duration_ms}ms",
        )

        tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)
        return (call_id, tool_result_for_llm), False

    async def _run_localization_gate_result_step(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        task_description: str,
        decision: LocalizationGateDecision,
        turn_count: int,
        gate_mode: str = "full",
    ) -> Optional[str]:
        """Generate the authoritative localization gate result block."""
        result_history = [message.copy() for message in message_history]
        result_history.append(
            {
                "role": "user",
                "content": generate_localization_gate_result_prompt(
                    task_description,
                    decision.candidate_answer,
                    decision.entity_type,
                    gate_mode=gate_mode,
                ),
            }
        )

        (
            result_text,
            _,
            tool_calls,
            _,
        ) = await self.answer_generator.handle_llm_call(
            system_prompt,
            result_history,
            [],
            turn_count + 3000,
            "Main Agent | Localization Gate Result",
            agent_type="main",
            temperature_override=resolve_temperature(self.cfg, "final_summary"),
        )

        if tool_calls:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate Result",
                "Result step attempted tool use. Skipping localization result block.",
            )
            return None

        cleaned_text = (result_text or "").strip()
        if not cleaned_text:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate Result",
                "Localization gate result step returned empty output.",
            )
            return None

        return cleaned_text

    @staticmethod
    def _build_fallback_localization_gate_result_text(
        decision: LocalizationGateDecision,
        gate_mode: str,
        notes: str,
    ) -> str:
        """Construct a deterministic fallback localization gate result block."""
        normalized_question_language = decision.question_language or "unknown"
        original_name_requested = "yes" if decision.original_name_requested else "no"
        verified_original_full_name = decision.candidate_answer or ""
        source_quality = "weak" if gate_mode == "skip" else "mixed"
        source_basis = (
            "Localization gate skipped before tool use."
            if gate_mode == "skip"
            else "Localization gate completed without a usable structured result."
        )

        return (
            "Localization Gate Result\n"
            f"- candidate_answer: {decision.candidate_answer}\n"
            f"- entity_type: {decision.entity_type}\n"
            f"- question_language: {normalized_question_language}\n"
            f"- original_name_requested: {original_name_requested}\n"
            "- localized_name_status: NOT_FOUND\n"
            "- localized_form_in_question_language: \n"
            f"- verified_original_full_name: {verified_original_full_name}\n"
            f"- source_basis: {source_basis}\n"
            f"- source_quality: {source_quality}\n"
            f"- notes: {notes}\n"
        )

    def _append_skip_localization_gate_result(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        decision: LocalizationGateDecision,
        notes: str,
    ) -> List[Dict[str, Any]]:
        """Append a structured NOT_FOUND result block when the gate is skipped."""
        fallback_text = self._build_fallback_localization_gate_result_text(
            decision=decision,
            gate_mode="skip",
            notes=notes,
        )
        message_history.append({"role": "assistant", "content": fallback_text})
        self.task_log.log_step(
            "warning",
            "Main Agent | Localization Gate Result",
            fallback_text[:800],
        )
        self._save_message_history(system_prompt, message_history)
        return message_history

    async def _run_pre_summary_localization_gate(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        tool_definitions: List[Dict[str, Any]],
        task_description: str,
        turn_count: int,
        tool_index: Dict[str, Set[str]],
        tool_to_server_index: Dict[str, Set[str]],
        active_main_temperature: float,
        task_deadline: Optional[float],
        gate_deadline: Optional[float],
        final_summary_reserve_seconds: float,
    ) -> List[Dict[str, Any]]:
        """Run the bounded pre-summary localization gate when needed."""
        if not bool(
            self.cfg.agent.main_agent.get("localization_gate_enabled", False)
        ):
            return message_history

        decision = await self._run_localization_gate_decision(
            system_prompt=system_prompt,
            message_history=message_history,
            task_description=task_description,
            turn_count=turn_count,
        )

        if not self._should_run_localization_gate(decision):
            self.task_log.log_step(
                "info",
                "Main Agent | Localization Gate",
                "Skipping localization gate: trigger conditions not met.",
            )
            return message_history

        gate_budget_decision = self._decide_localization_gate_mode(
            task_deadline=task_deadline,
            final_summary_reserve_seconds=final_summary_reserve_seconds,
        )
        gate_mode = gate_budget_decision.mode
        self.task_log.log_step(
            "info",
            "Main Agent | Localization Gate Budget",
            gate_budget_decision.reason,
        )
        if gate_mode == "skip":
            return self._append_skip_localization_gate_result(
                system_prompt=system_prompt,
                message_history=message_history,
                decision=decision,
                notes=(
                    "Gate was skipped before tool use because remaining time was below "
                    "the degraded-gate threshold."
                ),
            )

        gate_tool_definitions = self._filter_localization_gate_tool_definitions(
            tool_definitions
        )
        if not gate_tool_definitions:
            self.task_log.log_step(
                "warning",
                "Main Agent | Localization Gate",
                "No allowed localization gate tools were available. Skipping gate.",
            )
            return self._append_skip_localization_gate_result(
                system_prompt=system_prompt,
                message_history=message_history,
                decision=decision,
                notes=(
                    "Gate was skipped because no allowed localization tools were "
                    "available."
                ),
            )

        gate_history = [message.copy() for message in message_history]
        gate_history.append(
            {
                "role": "user",
                "content": generate_localization_gate_prompt(
                    task_description=task_description,
                    candidate_answer=decision.candidate_answer,
                    entity_type=decision.entity_type,
                    question_language=decision.question_language,
                    mode=gate_mode,
                ),
            }
        )

        if gate_mode == "degraded":
            max_tool_calls = int(
                self.cfg.agent.main_agent.get(
                    "localization_gate_degraded_max_tool_calls", 1
                )
                or 1
            )
        else:
            max_tool_calls = int(
                self.cfg.agent.main_agent.get(
                    "localization_gate_max_tool_calls", 2
                )
                or 2
            )
        gate_tool_calls_used = 0

        self.current_agent_id = await self.stream.start_agent(LOCALIZATION_GATE_AGENT_NAME)
        await self.stream.start_llm(LOCALIZATION_GATE_AGENT_NAME)
        self.task_log.log_step(
            "info",
            "Main Agent | Localization Gate",
            (
                f"Starting {gate_mode} localization gate for candidate "
                f"'{decision.candidate_answer}'."
            ),
        )

        try:
            while gate_tool_calls_used < max_tool_calls:
                if gate_deadline is not None and time.time() >= gate_deadline:
                    self.task_log.log_step(
                        "warning",
                        "Main Agent | Localization Gate",
                        "Gate deadline reached. Ending localization gate.",
                    )
                    break

                (
                    assistant_response_text,
                    _,
                    tool_calls,
                    gate_history,
                ) = await self.answer_generator.handle_llm_call(
                    system_prompt,
                    gate_history,
                    gate_tool_definitions,
                    turn_count + 2000 + gate_tool_calls_used,
                    f"Main Agent | Localization Gate (step {gate_tool_calls_used + 1})",
                    agent_type="main",
                    temperature_override=active_main_temperature,
                )

                if assistant_response_text:
                    text_response = extract_llm_response_text(assistant_response_text)
                    if text_response:
                        await self.stream.tool_call("show_text", {"text": text_response})

                if not tool_calls:
                    self.task_log.log_step(
                        "info",
                        "Main Agent | Localization Gate",
                        "Localization gate ended without additional tool calls.",
                    )
                    break

                if len(tool_calls) > 1:
                    if gate_mode == "degraded":
                        self.task_log.log_step(
                            "warning",
                            "Main Agent | Localization Gate",
                            "Degraded gate truncated multiple tool calls to 1.",
                        )
                    else:
                        self.task_log.log_step(
                            "warning",
                            "Main Agent | Localization Gate",
                            "Gate produced multiple tool calls in one step. Only allowed calls within budget will be executed.",
                        )

                all_tool_results_content_with_id = []
                stop_gate = False
                for tool_call in tool_calls:
                    if gate_tool_calls_used >= max_tool_calls:
                        break

                    if (
                        gate_mode == "degraded"
                        and tool_call["tool_name"] != "google_search"
                    ):
                        if tool_call["tool_name"] != "scrape_and_extract_info" or not self._has_recent_url_candidate(
                            gate_history, str(tool_call["arguments"].get("url", ""))
                        ):
                            self.task_log.log_step(
                                "warning",
                                "Main Agent | Localization Gate",
                                "Degraded gate only allows one direct google_search unless an exact recent URL candidate is present. Ending gate.",
                            )
                            stop_gate = True
                            break

                    if gate_tool_calls_used == 0 and tool_call["tool_name"] != "google_search":
                        if tool_call["tool_name"] != "scrape_and_extract_info" or not self._has_recent_url_candidate(
                            gate_history, str(tool_call["arguments"].get("url", ""))
                        ):
                            self.task_log.log_step(
                                "warning",
                                "Main Agent | Localization Gate",
                                "First gate call must be google_search unless an exact recent URL candidate is present. Ending gate.",
                            )
                            stop_gate = True
                            break

                    formatted_result, should_stop = (
                        await self._execute_localization_gate_tool_call(
                            tool_call=tool_call,
                            message_history=gate_history,
                            tool_index=tool_index,
                            tool_to_server_index=tool_to_server_index,
                        )
                    )
                    if formatted_result is not None:
                        all_tool_results_content_with_id.append(formatted_result)
                        gate_tool_calls_used += 1
                    if should_stop:
                        stop_gate = True
                        break

                if all_tool_results_content_with_id:
                    gate_history = self.llm_client.update_message_history(
                        gate_history, all_tool_results_content_with_id
                    )
                    self._save_message_history(system_prompt, gate_history)

                if stop_gate:
                    break

            gate_result_text = await self._run_localization_gate_result_step(
                system_prompt=system_prompt,
                message_history=gate_history,
                task_description=task_description,
                decision=decision,
                turn_count=turn_count,
                gate_mode=gate_mode,
            )
            if gate_result_text:
                gate_history.append({"role": "assistant", "content": gate_result_text})
                self.task_log.log_step(
                    "info",
                    "Main Agent | Localization Gate Result",
                    gate_result_text[:800],
                )
                self._save_message_history(system_prompt, gate_history)
            else:
                fallback_text = self._build_fallback_localization_gate_result_text(
                    decision=decision,
                    gate_mode=gate_mode,
                    notes=(
                        "Gate result step failed to produce a structured result; "
                        "using deterministic fallback."
                    ),
                )
                gate_history.append({"role": "assistant", "content": fallback_text})
                self.task_log.log_step(
                    "warning",
                    "Main Agent | Localization Gate Result",
                    fallback_text[:800],
                )
                self._save_message_history(system_prompt, gate_history)

            return gate_history
        finally:
            try:
                await self.stream.end_llm(LOCALIZATION_GATE_AGENT_NAME)
            except Exception:
                pass
            try:
                await self.stream.end_agent(
                    LOCALIZATION_GATE_AGENT_NAME, self.current_agent_id
                )
            except Exception:
                pass
            self.current_agent_id = None

    async def run_sub_agent(
        self,
        sub_agent_name: str,
        task_description: str,
    ):
        """
        Run a sub-agent to handle a subtask.

        Args:
            sub_agent_name: Name of the sub-agent to run
            task_description: Description of the subtask

        Returns:
            The final answer text from the sub-agent
        """
        task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Task Description",
            f"Subtask: {task_description}",
        )

        # Stream sub-agent start
        display_name = sub_agent_name.replace("agent-", "")
        sub_agent_id = await self.stream.start_agent(display_name)
        await self.stream.start_llm(display_name)

        # Start new sub-agent session
        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        # Initialize message history
        message_history = [{"role": "user", "content": task_description}]

        # Get sub-agent tool definitions
        if not self.sub_agent_tool_definitions:
            tool_definitions = await self._list_sub_agent_tools()
            tool_definitions = tool_definitions.get(sub_agent_name, {})
        else:
            tool_definitions = self.sub_agent_tool_definitions[sub_agent_name]

        if not tool_definitions:
            self.task_log.log_step(
                "warning",
                f"{sub_agent_name} | No Tools",
                "No tool definitions available.",
            )
        tool_index = self._build_tool_index(tool_definitions)
        tool_to_server_index = self._build_tool_to_server_index(tool_index)

        # Generate sub-agent system prompt
        sub_agent_extra_instruction = ""
        if self.cfg.agent.sub_agents and sub_agent_name in self.cfg.agent.sub_agents:
            sub_agent_extra_instruction = self.cfg.agent.sub_agents[sub_agent_name].get(
                "system_prompt_skill", ""
            )

        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(
            agent_type=sub_agent_name,
            extra_instruction=sub_agent_extra_instruction,
        )

        # Limit sub-agent turns
        if self.cfg.agent.sub_agents:
            max_turns = self.cfg.agent.sub_agents[sub_agent_name].max_turns
        else:
            max_turns = 0
        turn_count = 0
        total_attempts = 0
        max_attempts = max_turns + EXTRA_ATTEMPTS_BUFFER
        consecutive_rollbacks = 0
        web_tool_calls = 0
        active_main_temperature = resolve_temperature(self.cfg, "main_agent")

        while turn_count < max_turns and total_attempts < max_attempts:
            turn_count += 1
            total_attempts += 1

            if consecutive_rollbacks >= self.MAX_CONSECUTIVE_ROLLBACKS:
                self.task_log.log_step(
                    "error",
                    f"{sub_agent_name} | Too Many Rollbacks",
                    f"Reached {consecutive_rollbacks} consecutive rollbacks, breaking loop.",
                )
                break

            self.task_log.save()

            # Reset 'last_call_tokens'
            self.llm_client.last_call_tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

            # LLM call using answer generator
            try:
                (
                    assistant_response_text,
                    should_break,
                    tool_calls,
                    message_history,
                ) = await self.answer_generator.handle_llm_call(
                    system_prompt,
                    message_history,
                    tool_definitions,
                    turn_count,
                    f"{sub_agent_name} | Turn: {turn_count}",
                    agent_type=sub_agent_name,
                    temperature_override=active_main_temperature,
                )
            except PolicyBlockedError:
                await self.stream.end_llm(display_name)
                await self.stream.end_agent(display_name, sub_agent_id)
                raise

            if should_break:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "should break is True, breaking the loop",
                )
                break

            if assistant_response_text:
                text_response = extract_llm_response_text(assistant_response_text)
                if text_response:
                    await self.stream.tool_call("show_text", {"text": text_response})
            else:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "LLM call failed",
                )
                await asyncio.sleep(5)
                continue

            # Handle no tool calls case
            if not tool_calls:
                (
                    should_continue,
                    should_break_loop,
                    turn_count,
                    consecutive_rollbacks,
                    message_history,
                ) = await self._handle_response_format_issues(
                    assistant_response_text,
                    message_history,
                    turn_count,
                    consecutive_rollbacks,
                    total_attempts,
                    max_attempts,
                    sub_agent_name,
                )
                if should_continue:
                    continue
                if should_break_loop:
                    if not any(
                        mcp_tag in assistant_response_text for mcp_tag in mcp_tags
                    ) and not any(
                        keyword in assistant_response_text
                        for keyword in refusal_keywords
                    ):
                        self.task_log.log_step(
                            "info",
                            f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                            f"No tool calls found in {sub_agent_name}, ending on turn {turn_count}",
                        )
                    break

            # Execute tool calls
            tool_calls_data = []
            all_tool_results_content_with_id = []
            should_rollback_turn = False
            force_end_loop = False

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                repaired_server_name, repaired_tool_name, repair_reason = (
                    self._repair_tool_call_routing(
                        server_name=server_name,
                        tool_name=tool_name,
                        tool_index=tool_index,
                        tool_to_server_index=tool_to_server_index,
                    )
                )
                if repair_reason:
                    self.task_log.log_step(
                        "info",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Routing Repaired",
                        (
                            f"Auto-repaired routing ({repair_reason}): "
                            f"{server_name}/{tool_name} -> {repaired_server_name}/{repaired_tool_name}"
                        ),
                    )
                    server_name = repaired_server_name
                    tool_name = repaired_tool_name

                # Fix common parameter name mistakes
                arguments = self.tool_executor.fix_tool_call_arguments(
                    tool_name, arguments
                )

                # Validate server/tool routing before execution.
                is_valid, validation_error = self._validate_tool_call(
                    server_name, tool_name, tool_index
                )
                if not is_valid:
                    if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                        if message_history and message_history[-1]["role"] == "assistant":
                            message_history.pop()
                        consecutive_rollbacks += 1
                        should_rollback_turn = True
                        self.task_log.log_step(
                            "warning",
                            f"{sub_agent_name} | Turn: {turn_count} | Rollback",
                            f"Invalid tool routing: {validation_error}. "
                            f"Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}",
                        )
                        break
                    self.task_log.log_step(
                        "warning",
                        f"{sub_agent_name} | Turn: {turn_count} | End After Max Rollbacks",
                        f"Ending loop after repeated invalid tool routing: {validation_error}",
                    )
                    force_end_loop = True
                    break

                # Enforce hard budget for web tool calls.
                if self.web_tool_call_hard_limit > 0 and self._is_budgeted_web_tool(
                    tool_name
                ):
                    if web_tool_calls >= self.web_tool_call_hard_limit:
                        self.task_log.log_step(
                            "warning",
                            f"{sub_agent_name} | Web Tool Budget Reached",
                            (
                                f"Reached web tool hard limit "
                                f"({self.web_tool_call_hard_limit}), moving to final summary."
                            ),
                        )
                        force_end_loop = True
                        break
                    web_tool_calls += 1

                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                    f"Executing {tool_name} on {server_name}",
                )

                call_start_time = time.time()
                try:
                    # Check for duplicate query
                    cache_name = sub_agent_id + "_" + tool_name
                    (
                        is_duplicate,
                        should_rollback,
                        turn_count,
                        consecutive_rollbacks,
                        message_history,
                    ) = await self._check_duplicate_query(
                        tool_name,
                        arguments,
                        cache_name,
                        consecutive_rollbacks,
                        turn_count,
                        total_attempts,
                        max_attempts,
                        message_history,
                        sub_agent_name,
                    )
                    if should_rollback:
                        should_rollback_turn = True
                        break

                    # Send stream event
                    tool_call_id = await self.stream.tool_call(tool_name, arguments)

                    # Execute tool call
                    tool_result = await self.sub_agent_tool_managers[
                        sub_agent_name
                    ].execute_tool_call(server_name, tool_name, arguments)

                    # Update query count if successful
                    if "error" not in tool_result:
                        await self._record_query(cache_name, tool_name, arguments)

                    # Post-process result
                    tool_result = self.tool_executor.post_process_tool_call_result(
                        tool_name, tool_result
                    )
                    result = (
                        tool_result.get("result")
                        if tool_result.get("result")
                        else tool_result.get("error")
                    )

                    # Check for errors that should trigger rollback
                    if self.tool_executor.should_rollback_result(
                        tool_name, result, tool_result
                    ):
                        if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                            message_history.pop()
                            consecutive_rollbacks += 1
                            should_rollback_turn = True
                            self.task_log.log_step(
                                "warning",
                                f"{sub_agent_name} | Turn: {turn_count} | Rollback",
                                f"Tool result error - tool: {tool_name}, result: '{str(result)[:200]}'",
                            )
                            break

                    await self.stream.tool_call(
                        tool_name, {"result": result}, tool_call_id=tool_call_id
                    )
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    self.task_log.log_step(
                        "info",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} completed in {call_duration_ms}ms",
                    )

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )

                except PolicyBlockedError:
                    raise
                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": str(e),
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    tool_result = {
                        "error": f"Tool call failed: {str(e)}",
                        "server_name": server_name,
                        "tool_name": tool_name,
                    }
                    self.task_log.log_step(
                        "error",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
                    )

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            if should_rollback_turn:
                continue
            if force_end_loop:
                break

            # Reset consecutive rollbacks on successful execution
            if consecutive_rollbacks > 0:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Recovery",
                    f"Successfully recovered after {consecutive_rollbacks} consecutive rollbacks",
                )
            consecutive_rollbacks = 0

            # Update message history
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id
            )

            # Check context length
            temp_summary_prompt = generate_agent_summarize_prompt(
                task_description,
                agent_type=sub_agent_name,
            )

            pass_length_check, message_history = self.llm_client.ensure_summary_context(
                message_history, temp_summary_prompt
            )

            if not pass_length_check:
                turn_count = max_turns
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                )
                break

        # Log loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )
        else:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        # Generate final summary
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Final Summary",
            f"Generating {sub_agent_name} final summary",
        )

        summary_prompt = generate_agent_summarize_prompt(
            task_description,
            agent_type=sub_agent_name,
        )

        if message_history[-1]["role"] == "user":
            message_history.pop()
        message_history.append({"role": "user", "content": summary_prompt})

        await self.stream.tool_call(
            "Partial Summary", {}, tool_call_id=str(uuid.uuid4())
        )

        # Generate final answer
        try:
            (
                final_answer_text,
                should_break,
                tool_calls_info,
                message_history,
            ) = await self.answer_generator.handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count + 1,
                f"{sub_agent_name} | Final summary",
                agent_type=sub_agent_name,
                temperature_override=active_main_temperature,
            )
        except PolicyBlockedError:
            await self.stream.end_llm(display_name)
            await self.stream.end_agent(display_name, sub_agent_id)
            raise

        if final_answer_text:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Final Answer",
                "Final answer generated successfully",
            )
        else:
            final_answer_text = (
                f"No final answer generated by sub agent {sub_agent_name}."
            )
            self.task_log.log_step(
                "error",
                f"{sub_agent_name} | Final Answer",
                "Unable to generate final answer",
            )

        # Save session history
        self.task_log.sub_agent_message_history_sessions[
            self.task_log.current_sub_agent_session_id
        ] = {"system_prompt": system_prompt, "message_history": message_history}

        self.task_log.save()
        self.task_log.end_sub_agent_session(sub_agent_name)

        # Remove thinking content
        final_answer_text = final_answer_text.split("<think>")[-1].strip()
        final_answer_text = final_answer_text.split("</think>")[-1].strip()

        # Stream sub-agent end
        await self.stream.end_llm(display_name)
        await self.stream.end_agent(display_name, sub_agent_id)

        return final_answer_text

    async def run_main_agent(
        self, task_description, task_file_name=None, task_id="default_task"
    ):
        """
        Execute the main end-to-end task.

        Args:
            task_description: Description of the task to execute
            task_file_name: Optional file associated with the task
            task_id: Unique identifier for the task

        Returns:
            Tuple of (final_summary, final_boxed_answer, failure_experience_summary)
        """
        self.task_status = "running"
        self.blocked_reason = None
        workflow_id = await self.stream.start_workflow(task_description)

        self.task_log.log_step("info", "Main Agent", f"Start task with id: {task_id}")
        self.task_log.log_step(
            "info", "Main Agent", f"Task description: {task_description}"
        )
        if task_file_name:
            self.task_log.log_step(
                "info", "Main Agent", f"Associated file: {task_file_name}"
            )

        # Process input
        initial_user_content, processed_task_desc = process_input(
            task_description, task_file_name
        )
        message_history = [{"role": "user", "content": initial_user_content}]

        # Record initial user input
        user_input = processed_task_desc
        if task_file_name:
            user_input += f"\n[Attached file: {task_file_name}]"

        # Get tool definitions
        if not self.tool_definitions:
            tool_definitions = (
                await self.main_agent_tool_manager.get_all_tool_definitions()
            )
            if self.cfg.agent.sub_agents is not None:
                tool_definitions += expose_sub_agents_as_tools(
                    self.cfg.agent.sub_agents
                )
        else:
            tool_definitions = self.tool_definitions

        if not tool_definitions:
            self.task_log.log_step(
                "warning",
                "Main Agent | Tool Definitions",
                "Warning: No tool definitions found. LLM cannot use any tools.",
            )
        tool_index = self._build_tool_index(tool_definitions)
        tool_to_server_index = self._build_tool_to_server_index(tool_index)

        # Generate system prompt
        main_agent_extra_instruction = self.cfg.agent.main_agent.get(
            "system_prompt_skill", ""
        )
        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(
            agent_type="main",
            extra_instruction=main_agent_extra_instruction,
        )
        system_prompt = system_prompt.strip()

        # Main loop configuration
        max_turns = self.cfg.agent.main_agent.max_turns
        turn_count = 0
        total_attempts = 0
        max_attempts = max_turns + EXTRA_ATTEMPTS_BUFFER
        consecutive_rollbacks = 0
        web_tool_calls = 0
        active_main_temperature = resolve_temperature(self.cfg, "main_agent")
        task_start_time = time.time()
        task_timeout_seconds = float(
            self.cfg.benchmark.execution.get("task_timeout_seconds", 0)
        )
        localization_gate_reserve_seconds = float(
            self.cfg.benchmark.execution.get("localization_gate_reserve_seconds", 30)
        )
        final_summary_reserve_seconds = float(
            self.cfg.benchmark.execution.get("final_summary_reserve_seconds", 30)
        )
        (
            task_deadline,
            main_loop_deadline,
            gate_deadline,
        ) = self._compute_task_deadlines(
            task_start_time=task_start_time,
            task_timeout_seconds=task_timeout_seconds,
            localization_gate_reserve_seconds=localization_gate_reserve_seconds,
            final_summary_reserve_seconds=final_summary_reserve_seconds,
        )

        self.current_agent_id = await self.stream.start_agent("main")
        await self.stream.start_llm("main")

        while turn_count < max_turns and total_attempts < max_attempts:
            if main_loop_deadline is not None and time.time() >= main_loop_deadline:
                self.task_log.log_step(
                    "warning",
                    "Main Agent | Time Budget Reached",
                    (
                        "Main loop reached time budget; stopping tool-use loop and "
                        "moving to final summary."
                    ),
                )
                break

            turn_count += 1
            total_attempts += 1

            if consecutive_rollbacks >= self.MAX_CONSECUTIVE_ROLLBACKS:
                self.task_log.log_step(
                    "error",
                    "Main Agent | Too Many Rollbacks",
                    f"Reached {consecutive_rollbacks} consecutive rollbacks, breaking loop.",
                )
                break

            self.task_log.save()

            # LLM call
            try:
                (
                    assistant_response_text,
                    should_break,
                    tool_calls,
                    message_history,
                ) = await self.answer_generator.handle_llm_call(
                    system_prompt,
                    message_history,
                    tool_definitions,
                    turn_count,
                    f"Main agent | Turn: {turn_count}",
                    agent_type="main",
                    temperature_override=active_main_temperature,
                )
            except PolicyBlockedError as e:
                return await self._handle_policy_blocked(
                    task_id=task_id,
                    workflow_id=workflow_id,
                    blocked_reason=str(e),
                )

            # Process LLM response
            if assistant_response_text:
                text_response = extract_llm_response_text(assistant_response_text)
                if text_response:
                    await self.stream.tool_call("show_text", {"text": text_response})

                # Extract boxed content
                boxed_content = self.output_formatter._extract_boxed_content(
                    assistant_response_text
                )
                if boxed_content:
                    self.intermediate_boxed_answers.append(boxed_content)

                if should_break:
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | LLM Call",
                        "should break is True, breaking the loop",
                    )
                    break
            else:
                self.task_log.log_step(
                    "warning",
                    f"Main Agent | Turn: {turn_count} | LLM Call",
                    "No valid response from LLM, retrying with turn budget consumed",
                )
                await asyncio.sleep(5)
                continue

            # Handle no tool calls case
            if not tool_calls:
                (
                    should_continue,
                    should_break_loop,
                    turn_count,
                    consecutive_rollbacks,
                    message_history,
                ) = await self._handle_response_format_issues(
                    assistant_response_text,
                    message_history,
                    turn_count,
                    consecutive_rollbacks,
                    total_attempts,
                    max_attempts,
                    "Main Agent",
                )
                if should_continue:
                    continue
                if should_break_loop:
                    if not any(
                        mcp_tag in assistant_response_text for mcp_tag in mcp_tags
                    ) and not any(
                        keyword in assistant_response_text
                        for keyword in refusal_keywords
                    ):
                        self.task_log.log_step(
                            "info",
                            f"Main Agent | Turn: {turn_count} | LLM Call",
                            "LLM did not request tool usage, ending process.",
                        )
                    break

            # Execute tool calls
            tool_calls_data = []
            all_tool_results_content_with_id = []
            should_rollback_turn = False
            force_end_loop = False
            main_agent_last_call_tokens = self.llm_client.last_call_tokens

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                repaired_server_name, repaired_tool_name, repair_reason = (
                    self._repair_tool_call_routing(
                        server_name=server_name,
                        tool_name=tool_name,
                        tool_index=tool_index,
                        tool_to_server_index=tool_to_server_index,
                    )
                )
                if repair_reason:
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | Tool Routing Repaired",
                        (
                            f"Auto-repaired routing ({repair_reason}): "
                            f"{server_name}/{tool_name} -> {repaired_server_name}/{repaired_tool_name}"
                        ),
                    )
                    server_name = repaired_server_name
                    tool_name = repaired_tool_name

                # Fix common parameter name mistakes
                arguments = self.tool_executor.fix_tool_call_arguments(
                    tool_name, arguments
                )

                # Validate server/tool routing before execution.
                is_valid, validation_error = self._validate_tool_call(
                    server_name, tool_name, tool_index
                )
                if not is_valid:
                    if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                        if message_history and message_history[-1]["role"] == "assistant":
                            message_history.pop()
                        consecutive_rollbacks += 1
                        should_rollback_turn = True
                        self.task_log.log_step(
                            "warning",
                            f"Main Agent | Turn: {turn_count} | Rollback",
                            f"Invalid tool routing: {validation_error}. "
                            f"Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}",
                        )
                        break
                    self.task_log.log_step(
                        "warning",
                        "Main Agent | End After Max Rollbacks",
                        f"Ending loop after repeated invalid tool routing: {validation_error}",
                    )
                    force_end_loop = True
                    break

                # Enforce hard budget for web tool calls.
                if self.web_tool_call_hard_limit > 0 and self._is_budgeted_web_tool(
                    tool_name
                ):
                    if web_tool_calls >= self.web_tool_call_hard_limit:
                        self.task_log.log_step(
                            "warning",
                            "Main Agent | Web Tool Budget Reached",
                            (
                                f"Reached web tool hard limit "
                                f"({self.web_tool_call_hard_limit}), moving to final summary."
                            ),
                        )
                        force_end_loop = True
                        break
                    web_tool_calls += 1

                call_start_time = time.time()
                try:
                    if server_name.startswith("agent-") and self.cfg.agent.sub_agents:
                        # Sub-agent execution
                        cache_name = "main_" + tool_name
                        (
                            is_duplicate,
                            should_rollback,
                            turn_count,
                            consecutive_rollbacks,
                            message_history,
                        ) = await self._check_duplicate_query(
                            tool_name,
                            arguments,
                            cache_name,
                            consecutive_rollbacks,
                            turn_count,
                            total_attempts,
                            max_attempts,
                            message_history,
                            "Main Agent",
                        )
                        if should_rollback:
                            should_rollback_turn = True
                            break

                        # Stream events
                        await self.stream.end_llm("main")
                        await self.stream.end_agent("main", self.current_agent_id)

                        # Execute sub-agent
                        try:
                            sub_agent_result = await self.run_sub_agent(
                                server_name,
                                arguments["subtask"],
                            )
                        except PolicyBlockedError as e:
                            return await self._handle_policy_blocked(
                                task_id=task_id,
                                workflow_id=workflow_id,
                                blocked_reason=str(e),
                            )

                        # Update query count
                        await self._record_query(cache_name, tool_name, arguments)

                        tool_result = {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "result": sub_agent_result,
                        }
                        self.current_agent_id = await self.stream.start_agent(
                            "main", display_name="Summarizing"
                        )
                        await self.stream.start_llm("main", display_name="Summarizing")
                    else:
                        # Regular tool execution
                        cache_name = "main_" + tool_name
                        (
                            is_duplicate,
                            should_rollback,
                            turn_count,
                            consecutive_rollbacks,
                            message_history,
                        ) = await self._check_duplicate_query(
                            tool_name,
                            arguments,
                            cache_name,
                            consecutive_rollbacks,
                            turn_count,
                            total_attempts,
                            max_attempts,
                            message_history,
                            "Main Agent",
                        )
                        if should_rollback:
                            should_rollback_turn = True
                            break

                        # Send stream event
                        tool_call_id = await self.stream.tool_call(tool_name, arguments)

                        # Execute tool call
                        tool_result = (
                            await self.main_agent_tool_manager.execute_tool_call(
                                server_name=server_name,
                                tool_name=tool_name,
                                arguments=arguments,
                            )
                        )

                        # Update query count if successful
                        if "error" not in tool_result:
                            await self._record_query(cache_name, tool_name, arguments)

                        # Post-process result
                        tool_result = self.tool_executor.post_process_tool_call_result(
                            tool_name, tool_result
                        )
                        result = (
                            tool_result.get("result")
                            if tool_result.get("result")
                            else tool_result.get("error")
                        )

                        # Check for errors that should trigger rollback
                        if self.tool_executor.should_rollback_result(
                            tool_name, result, tool_result
                        ):
                            if (
                                consecutive_rollbacks
                                < self.MAX_CONSECUTIVE_ROLLBACKS - 1
                            ):
                                message_history.pop()
                                consecutive_rollbacks += 1
                                should_rollback_turn = True
                                self.task_log.log_step(
                                    "warning",
                                    f"Main Agent | Turn: {turn_count} | Rollback",
                                    f"Tool result error - tool: {tool_name}, result: '{str(result)[:200]}'",
                                )
                                break

                        await self.stream.tool_call(
                            tool_name, {"result": result}, tool_call_id=tool_call_id
                        )

                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} completed in {call_duration_ms}ms",
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": str(e),
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    tool_result = {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": str(e),
                    }
                    self.task_log.log_step(
                        "error",
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
                    )

                # Format results for LLM
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            if should_rollback_turn:
                continue
            if force_end_loop:
                break

            # Reset consecutive rollbacks on successful execution
            if consecutive_rollbacks > 0:
                self.task_log.log_step(
                    "info",
                    f"Main Agent | Turn: {turn_count} | Recovery",
                    f"Successfully recovered after {consecutive_rollbacks} consecutive rollbacks",
                )
            consecutive_rollbacks = 0

            # Update 'last_call_tokens'
            self.llm_client.last_call_tokens = main_agent_last_call_tokens

            # Update message history
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id
            )

            self.task_log.main_agent_message_history = {
                "system_prompt": system_prompt,
                "message_history": message_history,
            }
            self.task_log.save()

            # Check context length
            temp_summary_prompt = generate_agent_summarize_prompt(
                task_description,
                agent_type="main",
            )

            pass_length_check, message_history = self.llm_client.ensure_summary_context(
                message_history, temp_summary_prompt
            )

            if not pass_length_check:
                turn_count = max_turns
                self.task_log.log_step(
                    "warning",
                    f"Main Agent | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                )
                break

        await self.stream.end_llm("main")
        await self.stream.end_agent("main", self.current_agent_id)

        # Determine if max turns was reached
        reached_max_turns = turn_count >= max_turns
        if reached_max_turns:
            self.task_log.log_step(
                "warning",
                "Main Agent | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )
        else:
            self.task_log.log_step(
                "info",
                "Main Agent | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        try:
            message_history = await self._run_pre_summary_localization_gate(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                task_description=task_description,
                turn_count=turn_count,
                tool_index=tool_index,
                tool_to_server_index=tool_to_server_index,
                active_main_temperature=active_main_temperature,
                task_deadline=task_deadline,
                gate_deadline=gate_deadline,
                final_summary_reserve_seconds=final_summary_reserve_seconds,
            )
        except PolicyBlockedError as e:
            return await self._handle_policy_blocked(
                task_id=task_id,
                workflow_id=workflow_id,
                blocked_reason=str(e),
            )

        # Final summary
        self.task_log.log_step(
            "info", "Main Agent | Final Summary", "Generating final summary"
        )

        self.current_agent_id = await self.stream.start_agent("Final Summary")
        await self.stream.start_llm("Final Summary")

        # Generate final answer using answer generator
        try:
            (
                final_summary,
                final_boxed_answer,
                failure_experience_summary,
                usage_log,
                message_history,
            ) = await self.answer_generator.generate_and_finalize_answer(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                turn_count=turn_count,
                task_description=task_description,
                reached_max_turns=reached_max_turns,
                save_callback=self._save_message_history,
            )
        except PolicyBlockedError as e:
            return await self._handle_policy_blocked(
                task_id=task_id,
                workflow_id=workflow_id,
                blocked_reason=str(e),
            )

        await self.stream.tool_call("show_text", {"text": final_boxed_answer})
        await self.stream.end_llm("Final Summary")
        await self.stream.end_agent("Final Summary", self.current_agent_id)
        await self.stream.end_workflow(workflow_id)

        self.task_log.log_step(
            "info", "Main Agent | Usage Calculation", f"Usage log: {usage_log}"
        )

        self.task_log.log_step(
            "info",
            "Main Agent | Final boxed answer",
            f"Final boxed answer:\n\n{final_boxed_answer}",
        )

        self.task_log.log_step(
            "info",
            "Main Agent | Task Completed",
            f"Main agent task {task_id} completed successfully",
        )
        self.task_status = "success"
        gc.collect()
        return final_summary, final_boxed_answer, failure_experience_summary
