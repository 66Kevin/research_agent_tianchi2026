# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import os
from typing import List, Dict, Any
from openai import OpenAI

from app.manus.task.time_record_util import time_record


class ChatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, client: OpenAI, max_tokens: int = 4096,
                 temperature: float = 0.0, stream: bool = False, tools: List[Any] = None):
        self.tools = tools or []
        self.client = client
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def clean_none_values(data):
        """
        递归遍历数据结构，将所有 None 替换为 ""
        静态方法，无需实例化类即可调用
        """
        if isinstance(data, dict):
            return {k: ChatLLM.clean_none_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ChatLLM.clean_none_values(item) for item in data]
        elif data is None:
            return ""
        else:
            return data

    @staticmethod
    def _summarize_messages(messages: List[Dict[str, Any]]) -> str:
        role_counts: Dict[str, int] = {}
        total_chars = 0
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            content = msg.get("content", "")
            total_chars += len(str(content))
            if "tool_calls" in msg:
                total_chars += len(str(msg.get("tool_calls")))
        return f"count={len(messages)}, total_chars={total_chars}, roles={role_counts}"

    @staticmethod
    def _summarize_tool_calls(response_message: Any) -> str:
        tool_calls = getattr(response_message, "tool_calls", None) or []
        if not tool_calls:
            return "[]"
        names = []
        for call in tool_calls:
            try:
                names.append(call.function.name)
            except Exception:
                names.append("unknown_tool")
        return str(names)

    @staticmethod
    def _shorten(text: Any, max_chars: int = 300) -> str:
        value = str(text or "")
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + " ...<truncated>"

    @time_record
    def create_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict]):
        """
        Create a chat completion with support for function/tool calls
        """
        import time
        # 清洗提示词，去除None
        messages = ChatLLM.clean_none_values(messages)
        debug_payload = os.getenv("DEBUG_LLM_PAYLOAD", "").lower() in {"1", "true", "yes"}
        if debug_payload:
            print(f"create_with_tools messages:{messages}")
        else:
            print(f"create_with_tools messages:{self._summarize_messages(messages)}")
        max_retries = 2
        for attempt in range(max_retries):
            model_name = self.model
            try:
                if attempt == 1:
                    model_name = 'anthropic/claude-sonnet-4'
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature
                )
                if debug_payload:
                    print(f"LLM with tools chat completions response{attempt + 1} is {response}")
                else:
                    message = response.choices[0].message
                    print(
                        f"LLM with tools chat completions response{attempt + 1}: "
                        f"tool_calls={self._summarize_tool_calls(message)}, "
                        f"content_preview={self._shorten(message.content)}"
                    )
                break
            except Exception as e:
                print(f"JSON decode error: {e} on attempt {attempt + 1}, retrying...")
                if attempt == max_retries - 1:
                    print(f"Failed to create after {max_retries} attempts.")
                    raise
                time.sleep(3)  # 增加等待时间，避免频繁重试

        # 去除think标签
        content = response.choices[0].message.content
        if content is not None and '</think>' in content:
            response.choices[0].message.content = content.split('</think>')[-1].strip('\n')

        return response.choices[0].message

    @time_record
    def chat_to_llm(self, messages: List[Dict[str, Any]]):
        # 清洗提示词，去除None
        import time
        # 清洗提示词，去除None
        messages = ChatLLM.clean_none_values(messages)
        debug_payload = os.getenv("DEBUG_LLM_PAYLOAD", "").lower() in {"1", "true", "yes"}
        if debug_payload:
            print(f"chat_to_llm messages:{messages}")
        else:
            print(f"chat_to_llm messages:{self._summarize_messages(messages)}")
        max_retries = 2
        for attempt in range(max_retries):
            model_name = self.model
            try:
                if attempt == 1:
                    model_name = 'anthropic/claude-sonnet-4'
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                if debug_payload:
                    print(f"LLM with tools chat completions response{attempt + 1} is {response}")
                else:
                    message = response.choices[0].message
                    print(
                        f"LLM with tools chat completions response{attempt + 1}: "
                        f"content_preview={self._shorten(message.content)}"
                    )
                break
            except Exception as e:
                print(f"JSON decode error: {e} on attempt {attempt + 1}, retrying...")
                if attempt == max_retries - 1:
                    print(f"Failed to create after {max_retries} attempts.")
                    raise
                time.sleep(3)  # 增加等待时间，避免频繁重试
        # print(f"response is {response}")
        # 去除think标签
        content = response.choices[0].message.content
        if content is not None and '</think>' in content:
            response.choices[0].message.content = content.split('</think>')[-1].strip('\n')

        return response.choices[0].message.content
