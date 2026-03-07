# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
OpenAI-compatible LLM client implementation.

This module provides the OpenAIClient class for interacting with OpenAI's API
and OpenAI-compatible endpoints (such as vLLM, Qwen, DeepSeek, etc.).

Features:
- Async and sync API support
- Automatic retry with exponential backoff
- Token usage tracking and context length management
- MCP tool call parsing and response processing
"""

import asyncio
import dataclasses
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import tiktoken
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI

from ...utils.prompt_utils import generate_mcp_system_prompt
from ..base_client import BaseClient
from ..errors import PolicyBlockedError

logger = logging.getLogger("miroflow_agent")


@dataclasses.dataclass
class OpenAIClient(BaseClient):
    @staticmethod
    def _is_data_inspection_failed_error(error: Exception) -> bool:
        """Detect provider data-inspection failures that can benefit from context rollback."""
        error_text = str(error).lower()
        return (
            "data_inspection_failed" in error_text
            or "internalerror.algo.datainspectionfailed" in error_text
        )

    @staticmethod
    def _truncate_message_content(content: Any, max_chars: int) -> Any:
        """Truncate message content while preserving schema shape."""
        if isinstance(content, str):
            if len(content) <= max_chars:
                return content
            return content[:max_chars] + "\n...[truncated for provider safety retry]"

        if isinstance(content, list):
            truncated_blocks = []
            for block in content:
                if not isinstance(block, dict):
                    truncated_blocks.append(block)
                    continue
                block_copy = dict(block)
                text_value = block_copy.get("text")
                if isinstance(text_value, str) and len(text_value) > max_chars:
                    block_copy["text"] = (
                        text_value[:max_chars]
                        + "\n...[truncated for provider safety retry]"
                    )
                truncated_blocks.append(block_copy)
            return truncated_blocks

        return content

    def _rollback_messages_for_data_inspection(
        self, messages_for_llm: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Roll back the most recent turn to reduce risky context after data inspection errors.
        """
        rolled_back = [m.copy() for m in messages_for_llm]

        # Remove trailing tool-result/user/assistant messages from the latest turn.
        removed = 0
        while (
            rolled_back
            and removed < 2
            and rolled_back[-1].get("role") in {"assistant", "user", "tool"}
        ):
            rolled_back.pop()
            removed += 1

        # Keep message list non-empty and clamp long first user content.
        if not rolled_back:
            rolled_back = [m.copy() for m in messages_for_llm[:1]]

        for msg in rolled_back:
            if msg.get("role") == "user":
                msg["content"] = self._truncate_message_content(
                    msg.get("content"), max_chars=2000
                )
                break

        return rolled_back

    def _minimal_messages_for_data_inspection(
        self, messages_for_llm: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build a minimal safe context for last-chance retry on data inspection failures.
        """
        minimal: List[Dict[str, Any]] = []

        # Keep leading system/developer prompt if present.
        if messages_for_llm and messages_for_llm[0].get("role") in {
            "system",
            "developer",
        }:
            minimal.append(messages_for_llm[0].copy())

        # Keep first user question (truncated).
        first_user = next(
            (m for m in messages_for_llm if m.get("role") == "user"),
            None,
        )
        if first_user is not None:
            user_copy = first_user.copy()
            user_copy["content"] = self._truncate_message_content(
                user_copy.get("content"), max_chars=1200
            )
            minimal.append(user_copy)

        return minimal or [m.copy() for m in messages_for_llm[:1]]

    @staticmethod
    def _build_streamed_response(chunks: List[Any]) -> Any:
        """Build a response-like object from streaming chunks."""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason = "stop"
        usage_data = None

        for chunk in chunks:
            if getattr(chunk, "usage", None) is not None:
                usage_data = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            delta_content = getattr(delta, "content", None)
            if delta_content:
                content_parts.append(delta_content)
            delta_reasoning = getattr(delta, "reasoning_content", None)
            if delta_reasoning:
                reasoning_parts.append(delta_reasoning)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(
                        role="assistant",
                        content="".join(content_parts),
                        reasoning_content="".join(reasoning_parts)
                        if reasoning_parts
                        else None,
                    ),
                )
            ],
            usage=usage_data,
        )

    def _create_client(self) -> Union[AsyncOpenAI, OpenAI]:
        """Create LLM client"""
        http_client_args = {"headers": {"x-upstream-session-id": self.task_id}}
        if self.async_client:
            return AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=DefaultAsyncHttpxClient(**http_client_args),
            )
        else:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=DefaultHttpxClient(**http_client_args),
            )

    def _update_token_usage(self, usage_data: Any) -> None:
        """Update cumulative token usage"""
        if usage_data:
            input_tokens = getattr(usage_data, "prompt_tokens", 0)
            output_tokens = getattr(usage_data, "completion_tokens", 0)
            prompt_tokens_details = getattr(usage_data, "prompt_tokens_details", None)
            if prompt_tokens_details:
                cached_tokens = (
                    getattr(prompt_tokens_details, "cached_tokens", None) or 0
                )
            else:
                cached_tokens = 0

            # Record token usage for the most recent call
            self.last_call_tokens = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }

            # OpenAI does not provide cache_creation_input_tokens
            self.token_usage["total_input_tokens"] += input_tokens
            self.token_usage["total_output_tokens"] += output_tokens
            self.token_usage["total_cache_read_input_tokens"] += cached_tokens

            self.task_log.log_step(
                "info",
                "LLM | Token Usage",
                f"Input: {self.token_usage['total_input_tokens']}, "
                f"Output: {self.token_usage['total_output_tokens']}",
            )

    @staticmethod
    def _is_policy_blocked_error(error: Exception) -> bool:
        """Detect provider policy/safety blocked errors that should not be retried."""
        error_text = str(error).lower()
        if "content exists risk" in error_text:
            return True
        if "invalid_request_error" in error_text and "content" in error_text and "risk" in error_text:
            return True
        return False

    @staticmethod
    def _extract_policy_blocked_reason(error: Exception) -> str:
        """Build a concise blocked reason string from provider error."""
        return f"Provider blocked request: {str(error)}"

    async def _create_message(
        self,
        system_prompt: str,
        messages_history: List[Dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
        temperature_override: float | None = None,
    ):
        """
        Send message to OpenAI API.
        :param system_prompt: System prompt string.
        :param messages_history: Message history list.
        :return: OpenAI API response object or None (if error occurs).
        """
        active_temperature = (
            temperature_override
            if temperature_override is not None
            else self.temperature
        )

        self.task_log.log_step(
            "info",
            "LLM | Call Start",
            (
                f"Calling LLM ({'async' if self.async_client else 'sync'}) "
                f"with temperature={active_temperature}"
            ),
        )

        # Create a copy for sending to LLM (to avoid modifying the original)
        messages_for_llm = [m.copy() for m in messages_history]

        # put the system prompt in the first message since OpenAI API does not support system prompt in
        if system_prompt:
            # Check if there's already a system or developer message
            if messages_for_llm and messages_for_llm[0]["role"] in [
                "system",
                "developer",
            ]:
                messages_for_llm[0] = {
                    "role": "system",
                    "content": system_prompt,
                }

            else:
                messages_for_llm.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

        # Filter tool results to save tokens (only affects messages sent to LLM)
        messages_for_llm = self._remove_tool_result_from_messages(
            messages_for_llm, keep_tool_result
        )

        # Retry loop with dynamic max_tokens adjustment
        max_retries = int(self.cfg.llm.get("max_retries", 10))
        if max_retries < 1:
            max_retries = 1
        base_wait_time = float(self.cfg.llm.get("retry_base_wait_seconds", 30))
        if base_wait_time < 0:
            base_wait_time = 0.0
        current_max_tokens = self.max_tokens
        use_stream = bool(self.cfg.llm.get("stream", False))
        data_inspection_retry_count = 0

        for attempt in range(max_retries):
            params = {
                "model": self.model_name,
                "temperature": active_temperature,
                "messages": messages_for_llm,
                "stream": use_stream,
                "top_p": self.top_p,
                "extra_body": {},
            }
            # Check if the model is GPT-5, and adjust the parameter accordingly
            if "gpt-5" in self.model_name:
                # Use 'max_completion_tokens' for GPT-5
                params["max_completion_tokens"] = current_max_tokens
            else:
                # Use 'max_tokens' for GPT-4 and other models
                params["max_tokens"] = current_max_tokens

            # Add repetition_penalty if it's not the default value
            if self.repetition_penalty != 1.0:
                params["extra_body"]["repetition_penalty"] = self.repetition_penalty

            if "deepseek-v3-1" in self.model_name:
                params["extra_body"]["thinking"] = {"type": "enabled"}

            if "qwen3" in self.model_name or "qwen-max" in self.model_name:
                params["extra_body"]["enable_thinking"] = True

            # auto-detect if we need to continue from the last assistant message
            if messages_for_llm and messages_for_llm[-1].get("role") == "assistant":
                params["extra_body"]["continue_final_message"] = True
                params["extra_body"]["add_generation_prompt"] = False

            try:
                if self.async_client and use_stream:
                    stream = await self.client.chat.completions.create(**params)
                    chunks = []
                    async for chunk in stream:
                        chunks.append(chunk)
                    response = self._build_streamed_response(chunks)
                elif self.async_client:
                    response = await self.client.chat.completions.create(**params)
                elif use_stream:
                    stream = self.client.chat.completions.create(**params)
                    chunks = [chunk for chunk in stream]
                    response = self._build_streamed_response(chunks)
                else:
                    response = self.client.chat.completions.create(**params)
                # Update token count
                self._update_token_usage(getattr(response, "usage", None))
                self.task_log.log_step(
                    "info",
                    "LLM | Response Status",
                    f"{getattr(response.choices[0], 'finish_reason', 'N/A')}",
                )

                # Check if response was truncated due to length limit
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    # If this is not the last retry, increase max_tokens and retry
                    if attempt < max_retries - 1:
                        # Increase max_tokens by 10%
                        current_max_tokens = int(current_max_tokens * 1.1)
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached",
                            f"Response was truncated due to length limit (attempt {attempt + 1}/{max_retries}). Increasing max_tokens to {current_max_tokens} and retrying...",
                        )
                        await asyncio.sleep(base_wait_time)
                        continue
                    else:
                        # Last retry, return the truncated response instead of raising exception
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached - Returning Truncated Response",
                            f"Response was truncated after {max_retries} attempts. Returning truncated response to allow ReAct loop to continue.",
                        )
                        # Return the truncated response and let the orchestrator handle it
                        return response, messages_history

                # Check if the last 50 characters of the response appear more than 5 times in the response content.
                # If so, treat it as a severe repeat and trigger a retry.
                if hasattr(response.choices[0], "message") and hasattr(
                    response.choices[0].message, "content"
                ):
                    resp_content = response.choices[0].message.content or ""
                else:
                    resp_content = getattr(response.choices[0], "text", "")

                if resp_content and len(resp_content) >= 50:
                    tail_50 = resp_content[-50:]
                    repeat_count = resp_content.count(tail_50)
                    if repeat_count > 5:
                        # If this is not the last retry, retry
                        if attempt < max_retries - 1:
                            self.task_log.log_step(
                                "warning",
                                "LLM | Repeat Detected",
                                f"Severe repeat: the last 50 chars appeared over 5 times (attempt {attempt + 1}/{max_retries}), retrying...",
                            )
                            await asyncio.sleep(base_wait_time)
                            continue
                        else:
                            # Last retry, return anyway
                            self.task_log.log_step(
                                "warning",
                                "LLM | Repeat Detected - Returning Anyway",
                                f"Severe repeat detected after {max_retries} attempts. Returning response anyway.",
                            )

                # Success - return the original messages_history (not the filtered copy)
                # This ensures that the complete conversation history is preserved in logs
                return response, messages_history

            except asyncio.TimeoutError as e:
                if attempt < max_retries - 1:
                    self.task_log.log_step(
                        "warning",
                        "LLM | Timeout Error",
                        f"Timeout error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...",
                    )
                    await asyncio.sleep(base_wait_time)
                    continue
                else:
                    self.task_log.log_step(
                        "error",
                        "LLM | Timeout Error",
                        f"Timeout error after {max_retries} attempts: {str(e)}",
                    )
                    raise e
            except asyncio.CancelledError as e:
                self.task_log.log_step(
                    "error",
                    "LLM | Request Cancelled",
                    f"Request was cancelled: {str(e)}",
                )
                raise e
            except Exception as e:
                if self._is_policy_blocked_error(e):
                    blocked_reason = self._extract_policy_blocked_reason(e)
                    self.task_log.log_step(
                        "error",
                        "LLM | Policy Blocked",
                        blocked_reason,
                    )
                    raise PolicyBlockedError(blocked_reason) from e

                if self._is_data_inspection_failed_error(e):
                    data_inspection_retry_count += 1

                    if attempt < max_retries - 1:
                        if data_inspection_retry_count == 1:
                            previous_len = len(messages_for_llm)
                            messages_for_llm = self._rollback_messages_for_data_inspection(
                                messages_for_llm
                            )
                            self.task_log.log_step(
                                "warning",
                                "LLM | Data Inspection Failed",
                                (
                                    "Data inspection failed; rolled back recent turn and retrying "
                                    f"(attempt {attempt + 1}/{max_retries}, messages {previous_len}->{len(messages_for_llm)})."
                                ),
                            )
                            continue

                        if data_inspection_retry_count == 2:
                            previous_len = len(messages_for_llm)
                            messages_for_llm = self._minimal_messages_for_data_inspection(
                                messages_for_llm
                            )
                            self.task_log.log_step(
                                "warning",
                                "LLM | Data Inspection Failed",
                                (
                                    "Data inspection failed again; switched to minimal context and retrying "
                                    f"(attempt {attempt + 1}/{max_retries}, messages {previous_len}->{len(messages_for_llm)})."
                                ),
                            )
                            continue

                    self.task_log.log_step(
                        "error",
                        "LLM | Data Inspection Failed",
                        f"Persistent data inspection failure after {attempt + 1} attempts: {str(e)}",
                    )
                    blocked_reason = (
                        "Provider blocked request due to data inspection policy: "
                        f"{str(e)}"
                    )
                    self.task_log.log_step(
                        "error",
                        "LLM | Policy Blocked",
                        blocked_reason,
                    )
                    raise PolicyBlockedError(blocked_reason) from e

                if "Error code: 400" in str(e) and "longer than the model" in str(e):
                    self.task_log.log_step(
                        "error",
                        "LLM | Context Length Error",
                        f"Error: {str(e)}",
                    )
                    raise e
                else:
                    if attempt < max_retries - 1:
                        self.task_log.log_step(
                            "warning",
                            "LLM | API Error",
                            f"Error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...",
                        )
                        await asyncio.sleep(base_wait_time)
                        continue
                    else:
                        self.task_log.log_step(
                            "error",
                            "LLM | API Error",
                            f"Error after {max_retries} attempts: {str(e)}",
                        )
                        raise e

        # Should never reach here, but just in case
        raise Exception("Unexpected error: retry loop completed without returning")

    def process_llm_response(
        self, llm_response: Any, message_history: List[Dict], agent_type: str = "main"
    ) -> tuple[str, bool, List[Dict]]:
        """Process LLM response"""
        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            self.task_log.log_step(
                "error", "LLM | Response Error", f"Error: {error_msg}"
            )
            return "", True, message_history  # Exit loop, return message_history

        # Extract LLM response text
        if llm_response.choices[0].finish_reason == "stop":
            assistant_response_text = llm_response.choices[0].message.content or ""
            reasoning_content = getattr(llm_response.choices[0].message, "reasoning_content", None)

            if reasoning_content:
                self.task_log.log_step(
                    "info",
                    "LLM | Thinking Process",
                    reasoning_content,
                )

            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )

        elif llm_response.choices[0].finish_reason == "length":
            assistant_response_text = llm_response.choices[0].message.content or ""
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty."
            elif "Context length exceeded" in assistant_response_text:
                # This is the case where context length is exceeded, needs special handling
                self.task_log.log_step(
                    "warning",
                    "LLM | Context Length",
                    "Detected context length exceeded, returning error status",
                )
                message_history.append(
                    {"role": "assistant", "content": assistant_response_text}
                )
                return (
                    assistant_response_text,
                    True,
                    message_history,
                )  # Return True to indicate need to exit loop

            # Add assistant response to history
            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )

        else:
            raise ValueError(
                f"Unsupported finish reason: {llm_response.choices[0].finish_reason}"
            )

        return assistant_response_text, False, message_history

    def extract_tool_calls_info(
        self, llm_response: Any, assistant_response_text: str
    ) -> List[Dict]:
        """Extract tool call information from LLM response"""
        from ...utils.parsing_utils import parse_llm_response_for_tool_calls

        return parse_llm_response_for_tool_calls(assistant_response_text)

    def update_message_history(
        self, message_history: List[Dict], all_tool_results_content_with_id: List[Tuple]
    ) -> List[Dict]:
        """Update message history with tool calls data (llm client specific)"""

        merged_text = "\n".join(
            [
                item[1]["text"]
                for item in all_tool_results_content_with_id
                if item[1]["type"] == "text"
            ]
        )

        message_history.append(
            {
                "role": "user",
                "content": merged_text,
            }
        )

        return message_history

    def generate_agent_system_prompt(self, date: Any, mcp_servers: List[Dict]) -> str:
        return generate_mcp_system_prompt(date, mcp_servers)

    def _estimate_tokens(self, text: str) -> int:
        """Use tiktoken to estimate the number of tokens in text"""
        if not hasattr(self, "encoding"):
            # Initialize tiktoken encoder
            try:
                self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                # If o200k_base is not available, use cl100k_base as fallback
                self.encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # If encoding fails, use simple estimation: approximately 1 token per 4 characters
            self.task_log.log_step(
                "error",
                "LLM | Token Estimation Error",
                f"Error: {str(e)}",
            )
            return len(text) // 4

    def ensure_summary_context(
        self, message_history: list, summary_prompt: str
    ) -> tuple[bool, list]:
        """
        Check if current message_history + summary_prompt will exceed context
        If it will exceed, remove the last assistant-user pair and return False
        Return True to continue, False if messages have been rolled back
        """
        # Get token usage from the last LLM call
        last_prompt_tokens = self.last_call_tokens.get("prompt_tokens", 0)
        last_completion_tokens = self.last_call_tokens.get("completion_tokens", 0)
        buffer_factor = 1.5

        # Calculate token count for summary prompt
        summary_tokens = int(self._estimate_tokens(summary_prompt) * buffer_factor)

        # Calculate token count for the last user message in message_history
        last_user_tokens = 0
        if message_history[-1]["role"] == "user":
            content = message_history[-1]["content"]
            last_user_tokens = int(self._estimate_tokens(str(content)) * buffer_factor)

        # Calculate total token count: last prompt + completion + last user message + summary + reserved response space
        estimated_total = (
            last_prompt_tokens
            + last_completion_tokens
            + last_user_tokens
            + summary_tokens
            + self.max_tokens
            + 1000  # Add 1000 tokens as buffer
        )

        if estimated_total >= self.max_context_length:
            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                "Context limit reached, proceeding to step back and summarize the conversation",
            )

            # Remove the last user message (tool call results)
            if message_history[-1]["role"] == "user":
                message_history.pop()

            # Remove the second-to-last assistant message (tool call request)
            if message_history[-1]["role"] == "assistant":
                message_history.pop()

            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                f"Removed the last assistant-user pair, current message_history length: {len(message_history)}",
            )

            return False, message_history

        self.task_log.log_step(
            "info",
            "LLM | Context Limit Not Reached",
            f"{estimated_total}/{self.max_context_length}",
        )
        return True, message_history

    def format_token_usage_summary(self) -> tuple[List[str], str]:
        """Format token usage statistics, return summary_lines for format_final_summary and log string"""
        token_usage = self.get_token_usage()

        total_input = token_usage.get("total_input_tokens", 0)
        total_output = token_usage.get("total_output_tokens", 0)
        cache_input = token_usage.get("total_cache_input_tokens", 0)

        summary_lines = []
        summary_lines.append("\n" + "-" * 20 + " Token Usage " + "-" * 20)
        summary_lines.append(f"Total Input Tokens: {total_input}")
        summary_lines.append(f"Total Cache Input Tokens: {cache_input}")
        summary_lines.append(f"Total Output Tokens: {total_output}")
        summary_lines.append("-" * (40 + len(" Token Usage ")))
        summary_lines.append("Pricing is disabled - no cost information available")
        summary_lines.append("-" * (40 + len(" Token Usage ")))

        # Generate log string
        log_string = (
            f"[{self.model_name}] Total Input: {total_input}, "
            f"Cache Input: {cache_input}, "
            f"Output: {total_output}"
        )

        return summary_lines, log_string

    def get_token_usage(self):
        return self.token_usage.copy()
