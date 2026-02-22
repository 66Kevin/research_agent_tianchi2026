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

import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import PureWindowsPath, PurePosixPath
from typing import List, Optional, Dict, Tuple


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...<truncated>"


MAX_FACTS_CHARS = _safe_int_env("PLAN_FACTS_MAX_CHARS", 8000)
MAX_STEP_NOTES_CHARS = _safe_int_env("PLAN_STEP_NOTES_MAX_CHARS", 2000)
MAX_STEP_NOTES_IN_PROMPT_CHARS = _safe_int_env("PLAN_STEP_NOTES_IN_PROMPT_MAX_CHARS", 320)


class Plan:
    """Represents a single plan with steps, statuses, and execution details as a DAG."""

    def __init__(self, title: str = "", steps: List[str] = None, dependencies: Dict[int, List[int]] = None):
        self.title = title
        self.steps = steps if steps else []
        # 使用步骤内容（中文）作为key存储状态、备注和详细信息
        self.step_statuses = {step: "not_started" for step in self.steps}
        self.step_notes = {step: "" for step in self.steps}
        self.step_details = {step: "" for step in self.steps}
        self.step_files = {step: "" for step in self.steps}
        # 新增：步骤执行工具记录
        self.step_tools = {step: [] for step in self.steps}  # 记录每个步骤使用的工具
        self.facts = ""
        # 使用邻接表表示依赖关系
        if dependencies:
            self.dependencies = dependencies
        else:
            self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))} if len(self.steps) > 1 else {}
        self.result = ""

    def set_plan_result(self, plan_result):
        self.result = plan_result

    def get_plan_result(self):
        return self.result

    def update_facts(self, new_fact):
        fact_text = str(new_fact or "")
        self.facts = _truncate_text(fact_text, MAX_FACTS_CHARS)
        preview = _truncate_text(self.facts.replace("\n", " "), 300)
        print(f"facts of plan:{preview}")

    def get_ready_steps(self) -> List[int]:
        """获取所有前置依赖都已完成的步骤索引

        返回:
            List[int]: 可立即执行的步骤索引列表（返回所有符合条件的步骤）
        """
        print(f"get_ready_steps dependencies: {self.dependencies}")
        ready_steps = []
        for step_index in range(len(self.steps)):
            # 获取该步骤的所有依赖
            dependencies = self.dependencies.get(step_index, [])

            # 检查所有依赖是否都已完成
            if all(self.step_statuses.get(self.steps[dep]) != "not_started" for dep in dependencies):
                # 检查步骤本身是否未开始
                if self.step_statuses.get(self.steps[step_index]) == "not_started":
                    ready_steps.append(step_index)

        return ready_steps

    def update(self, title: Optional[str] = None, steps: Optional[List[str]] = None,
               dependencies: Optional[Dict[int, List[int]]] = None) -> None:
        """Update the plan with new title, steps, or dependencies while preserving completed steps."""
        if title:
            self.title = title

        if steps:
            # Preserve all existing steps and their statuses
            new_steps = []
            new_statuses = {}
            new_notes = {}
            new_details = {}
            new_tools = {}  # 新增：用于保存工具记录

            # First, process all steps in the input order
            for step in steps:
                # If step exists in current steps and is started, preserve it
                if step in self.steps and self.step_statuses.get(step) != "not_started":
                    new_steps.append(step)
                    new_statuses[step] = self.step_statuses.get(step)
                    new_notes[step] = self.step_notes.get(step)
                    new_details[step] = self.step_details.get(step)
                    new_tools[step] = self.step_tools.get(step, [])  # 保留工具记录
                # If step exists in current steps and is not started, preserve as not_started
                elif step in self.steps:
                    new_steps.append(step)
                    new_statuses[step] = "not_started"
                    new_notes[step] = self.step_notes.get(step)
                    new_details[step] = self.step_details.get(step)
                    new_tools[step] = []  # 重置工具记录
                # If step is new, add as not_started
                else:
                    new_steps.append(step)
                    new_statuses[step] = "not_started"
                    new_notes[step] = ""
                    new_details[step] = ""
                    new_tools[step] = []  # 初始化空工具记录

            self.steps = new_steps
            self.step_statuses = new_statuses
            self.step_notes = new_notes
            self.step_details = new_details
            self.step_tools = new_tools  # 更新工具记录
        print(f"before update dependencies: {self.dependencies}")
        if dependencies:
            self.dependencies.clear()
            normalized_dependencies: Dict[int, List[int]] = {}
            for k, v in dependencies.items():
                dep_list = v if isinstance(v, list) else []
                normalized_dependencies[int(k)] = [int(dep) for dep in dep_list]
            self.dependencies.update(normalized_dependencies)
        else:
            self.dependencies = {i: [i - 1] for i in range(1, len(steps))} if len(steps) > 1 else {}
        print(f"after update dependencies: {self.dependencies}")

    def mark_step(self, step_index: int, step_status: Optional[str] = None, step_notes: Optional[str] = None) -> None:
        """Mark a single step with specific statuses, notes, and details.

        Args:
            step_index (int): Index of the step to update
            step_status (Optional[str]): New status for the step
            step_notes (Optional[str]): Notes for the step
        """
        # Validate step index
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(self.steps) - 1}.")
        print(f"step_index: {step_index}, step_status is {step_status},step_notes is {step_notes}")
        step = self.steps[step_index]

        # Update step status
        if step_status is not None:
            self.step_statuses[step] = step_status

        # Update step notes
        if step_notes is not None:
            step_notes, file_path_info = process_text_with_workspace(step_notes)
            step_notes = _truncate_text(step_notes, MAX_STEP_NOTES_CHARS)
            self.step_notes[step] = step_notes
            self.step_files[step] = file_path_info

    def get_progress(self) -> Dict[str, int]:
        """Get progress statistics of the plan."""
        return {
            "total": len(self.steps),
            "completed": sum(1 for status in self.step_statuses.values() if status == "completed"),
            "in_progress": sum(1 for status in self.step_statuses.values() if status == "in_progress"),
            "blocked": sum(1 for status in self.step_statuses.values() if status == "blocked"),
            "not_started": sum(1 for status in self.step_statuses.values() if status == "not_started")
        }

    def record_tool_execution(self, step_index: int, tool_name: str, tool_args: Dict, result: str):
        """记录工具执行结果并更新步骤进度
        
        Args:
            step_index (int): 步骤索引
            tool_name (str): 工具名称
            tool_args (Dict): 工具入参
            result (str): 工具执行结果
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        step = self.steps[step_index]

        # 创建唯一标识：工具名称 + 入参的JSON字符串
        tool_id = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

        # 直接追加新记录
        self.step_tools[step].append({
            "tool_id": tool_id,  # 唯一标识
            "tool": tool_name,
            "args": tool_args,  # 记录工具入参
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    def get_step_execution_history(self, step_index: int) -> List[Dict]:
        """获取步骤的执行历史记录
        
        Args:
            step_index (int): 步骤索引
            
        Returns:
            List[Dict]: 工具执行历史记录
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}")

        return self.step_tools[self.steps[step_index]]

    def format(self, with_detail: bool = False) -> str:
        """Format the plan for display."""
        output = f"Plan: {self.title}\n"
        output += "=" * len(output) + "\n\n"

        progress = self.get_progress()
        output += f"Progress: {progress['completed']}/{progress['total']} steps completed "
        if progress['total'] > 0:
            percentage = (progress['completed'] / progress['total']) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {progress['completed']} completed, {progress['in_progress']} in progress, "
        output += f"{progress['blocked']} blocked, {progress['not_started']} not started\n\n"
        output += "Steps:\n"

        for i, step in enumerate(self.steps):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(self.step_statuses.get(step), "[ ]")

            deps = self.dependencies.get(i, [])
            dep_str = f" (depends on: {', '.join(map(str, deps))})" if deps else ""
            output += f"Step{i} :{status_symbol} {step}{dep_str}\n"
            if with_detail and self.step_tools.get(step):
                output += "   Tool Execution History:\n"
                for tool_exec in self.step_tools[step]:
                    output += f"     -Tool: {tool_exec['tool']} (args: {tool_exec['args']}) ({tool_exec['timestamp']}): {tool_exec['result'][:100]}\n"

            if self.step_notes.get(step):
                notes = self.step_notes.get(step)
                if not with_detail:
                    notes = _truncate_text(notes, MAX_STEP_NOTES_IN_PROMPT_CHARS)
                    output += f"   Notes: {notes}\n"
                else:
                    output += f"   Notes: {notes}\nDetails: {self.step_details.get(step)}\n"

        return output

    def has_blocked_steps(self) -> bool:
        """Check if there are any blocked steps in the plan.
        
        Returns:
            bool: True if any step is blocked, False otherwise
        """
        return any(status == "blocked" for status in self.step_statuses.values())


def get_last_folder_name() -> str:
    workspace_path = os.environ.get('WORKSPACE_PATH')
    if not workspace_path:
        raise ValueError("环境变量 'WORKSPACE_PATH' 未设置。")

    current_os = platform.system()
    if current_os == 'Windows':
        path_obj = PureWindowsPath(workspace_path)
    else:
        path_obj = PurePosixPath(workspace_path)

    return path_obj.name


def extract_and_replace_paths(text: str, folder_name: str) -> Tuple[str, List[Dict[str, str]]]:
    # 支持的文件扩展名
    valid_extensions = r"(txt|md|pdf|docx|xlsx|csv|json|xml|html|png|jpg|jpeg|svg|py)"

    # ✅ Linux/macOS 风格: /xxx/yyy/zzz/file.ext
    # ✅ Windows 风格: C:\xxx\yyy\file.ext （或 UNC 网络路径 \\Server\Share\file.ext）
    path_file_pattern = rf'([a-zA-Z]:\\[^\s《》]+?\.{valid_extensions}|/[^\s《》]+?\.{valid_extensions})'

    # ✅ 中文书名号引用的文件名（不区分平台）
    quoted_file_pattern = rf'《([^《》\s]+?\.{valid_extensions})》'

    result_list: List[Dict[str, str]] = []

    def replace_path_file(match):
        full_path = match.group(1)
        filename = os.path.basename(full_path.replace("\\", "/"))  # 把反斜杠变成斜杠后再提取
        new_path = f"{folder_name}/{filename}"
        result_list.append({
            "name": filename,
            "path": new_path
        })
        return new_path

    def replace_quoted_file(match):
        filename = match.group(1)
        new_path = f"{folder_name}/{filename}"
        result_list.append({
            "name": filename,
            "path": new_path
        })
        return new_path

    new_text = re.sub(path_file_pattern, replace_path_file, text)
    new_text = re.sub(quoted_file_pattern, replace_quoted_file, new_text)

    return new_text, result_list


def process_text_with_workspace(text: str) -> Tuple[str, List[Dict[str, str]]]:
    folder_name = get_last_folder_name()
    return extract_and_replace_paths(text, folder_name)


if __name__ == "__main__":
    # 创建Plan
    plan = Plan("测试计划", [
        "步骤1",
        "步骤2",
        "步骤3",
        "步骤4",
        "步骤5"
    ])
    # 设置依赖关系
    plan.add_dependency(1, 0)  # 步骤2依赖于步骤1
    plan.add_dependency(2, 0)  # 步骤3依赖于步骤1
    plan.add_dependency(3, 1)  # 步骤4依赖于步骤2
    plan.add_dependency(4, 2)  # 步骤5依赖于步骤3
    plan.mark_step(0, "completed")
    plan.mark_step(1, "completed")
    plan.mark_step(2, "completed")
    result = plan.get_ready_steps()
    print(result)
