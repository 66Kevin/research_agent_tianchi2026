# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Prompt templates and utilities for agent system prompts.

This module provides:
- System prompt generation for MCP tool usage
- Agent-specific prompt generation (main agent, browsing agent)
- Summary prompt templates for final answer generation
- Failure experience templates for retry mechanisms
"""

# ============================================================================
# Format Error Messages
# ============================================================================

FORMAT_ERROR_MESSAGE = "No \\boxed{} content found in the final answer."
BLOCKED_BY_POLICY_MESSAGE = "BLOCKED_BY_POLICY"
BLOCKED_JUDGE_TYPE = "provider_safety"

# ============================================================================
# Failure Experience Templates (for format error retry)
# ============================================================================

# Header that appears once before all failure experiences
FAILURE_EXPERIENCE_HEADER = """

=== Previous Attempts Analysis ===
The following summarizes what was tried before and why it didn't work. Use this to guide a NEW approach.

"""

# Template for each individual failure experience (used multiple times)
FAILURE_EXPERIENCE_ITEM = """[Attempt {attempt_number}]
{failure_summary}

"""

# Footer that appears once after all failure experiences
FAILURE_EXPERIENCE_FOOTER = """=== End of Analysis ===

Based on the above, you should try a different strategy this time.
"""

FAILURE_SUMMARY_PROMPT = """The task was not completed successfully. Do NOT call any tools. Provide a summary:

Failure type: [incomplete / blocked / misdirected / format_missed]
  - incomplete: ran out of turns before finishing
  - blocked: got stuck due to tool failure or missing information
  - misdirected: went down the wrong path
  - format_missed: found the answer but forgot to use \\boxed{}
What happened: [describe the approach taken and why a final answer was not reached]
Useful findings: [list any facts, intermediate results, or conclusions discovered that should be reused]"""

# Assistant prefix for failure summary generation (guides model to follow structured format)
FAILURE_SUMMARY_THINK_CONTENT = """We need to write a structured post-mortem style summary **without calling any tools**, explaining why the task was not completed, using these required sections:

* **Failure type**: pick one from **incomplete / blocked / misdirected / format_missed**
* **What happened**: describe the approach taken and why it didn't reach a final answer
* **Useful findings**: list any facts, intermediate results, or conclusions that can be reused"""

FAILURE_SUMMARY_ASSISTANT_PREFIX = (
    f"<think>\n{FAILURE_SUMMARY_THINK_CONTENT}\n</think>\n\n"
)

# ============================================================================
# MCP Tags for Parsing
# ============================================================================

mcp_tags = [
    "<use_mcp_tool>",
    "</use_mcp_tool>",
    "<server_name>",
    "</server_name>",
    "<arguments>",
    "</arguments>",
]

refusal_keywords = [
    "time constraint",
    "I’m sorry, but I can’t",
    "I'm sorry, I cannot solve",
]


def generate_mcp_system_prompt(date, mcp_servers):
    """
    Generate the MCP (Model Context Protocol) system prompt for LLM.

    Creates a structured prompt that instructs the LLM on how to use available
    MCP tools. Includes tool definitions, XML formatting instructions, and
    general task-solving guidelines.

    Args:
        date: Current date object for timestamp inclusion
        mcp_servers: List of server definitions, each containing 'name' and 'tools'

    Returns:
        Complete system prompt string with tool definitions and usage instructions
    """
    formatted_date = date.strftime("%Y-%m-%d")

    # Start building the template, now follows https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#tool-use-system-prompt
    template = f"""In this environment you have access to a set of tools you can use to answer the user's question. 

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {formatted_date}

# Tool-Use Formatting Instructions 

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description: 
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{
"param1": "value1",
"param2": "value2 \\"escaped string\\""
}}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:

"""

    # Add MCP servers section
    if mcp_servers and len(mcp_servers) > 0:
        for server in mcp_servers:
            template += f"\n## Server name: {server['name']}\n"

            if "tools" in server and len(server["tools"]) > 0:
                for tool in server["tools"]:
                    # Skip tools that failed to load (they only have 'error' key)
                    if "error" in tool and "name" not in tool:
                        continue
                    template += f"### Tool name: {tool['name']}\n"
                    template += f"Description: {tool['description']}\n"
                    template += f"Input JSON schema: {tool['schema']}\n"

    # Add the full objective system prompt
    template += """
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

"""

    return template


def generate_no_mcp_system_prompt(date):
    """
    Generate a minimal system prompt without MCP tool definitions.

    Used when no tools are available or when running in tool-less mode.

    Args:
        date: Current date object for timestamp inclusion

    Returns:
        Basic system prompt string without tool definitions
    """
    formatted_date = date.strftime("%Y-%m-%d")

    # Start building the template, now follows https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#tool-use-system-prompt
    template = """In this environment you have access to a set of tools you can use to answer the user's question. """

    template += f" Today is: {formatted_date}\n"

    template += """
Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
"""

    # Add the full objective system prompt
    template += """
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

"""
    return template


def generate_agent_specific_system_prompt(
    agent_type="", extra_instruction: str = ""
):
    """
    Generate agent-specific objective prompts based on agent type.

    Different agent types have different objectives:
    - main: Task-solving agent that uses tools to answer questions
    - agent-browsing: Web search and browsing agent for information retrieval

    Args:
        agent_type: Type of agent ("main", "agent-browsing", or "browsing-agent")
        extra_instruction: Optional task-specific instruction block appended to
            the agent objective prompt.

    Returns:
        Agent-specific objective prompt string
    """
    if agent_type == "main":
        system_prompt = """\n
# Agent Specific Objective

You are a task-solving agent that uses tools step-by-step to answer the user's question. Your goal is to provide complete, accurate and well-reasoned answers using additional tools.

"""
    elif agent_type == "agent-browsing" or agent_type == "browsing-agent":
        system_prompt = """# Agent Specific Objective

You are an agent that performs the task of searching and browsing the web for specific information and generating the desired answer. Your task is to retrieve reliable, factual, and verifiable information that fills in knowledge gaps.
Do not infer, speculate, summarize broadly, or attempt to fill in missing parts yourself. Only return factual content.
"""
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if extra_instruction:
        system_prompt = (
            f"{system_prompt.strip()}\n\n"
            "# Task-Specific Execution Skill\n\n"
            f"{extra_instruction.strip()}\n"
        )

    return system_prompt.strip()


def generate_agent_summarize_prompt(task_description, agent_type=""):
    """
    Generate the final summarization prompt for an agent.

    Creates prompts that instruct agents to summarize their work and provide
    final answers. Different agent types have different summarization formats:
    - main: Must wrap answer in \\boxed{} with strict formatting rules
    - agent-browsing: Provides structured report of findings

    Args:
        task_description: The original task/question to reference in the summary
        agent_type: Type of agent ("main" or "agent-browsing")

    Returns:
        Summarization prompt string with formatting instructions
    """
    if agent_type == "main":
        summarize_prompt = (
            "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
            "Do NOT call any tools. Do NOT perform any new search, scrape, browsing, code execution, or external lookup.\n"
            "Use only the information already present in the conversation.\n\n"
            "You must follow these steps in order:\n\n"
            "Step 1. Determine the semantic answer from the full conversation.\n"
            "- Read the full conversation carefully and identify the best-supported final answer to the original question.\n"
            "- If a clear answer has already been established earlier in the conversation, reuse it instead of re-deriving it.\n"
            "- Do not broaden the reasoning again if the answer is already supported by the collected evidence.\n"
            "- If the evidence is incomplete, choose the best-supported answer from the conversation rather than inventing a new one.\n\n"
            "Step 2. Normalize the answer into the correct final form.\n"
            "- First check whether the original question explicitly asks for an original name, native name, original-language title, real name, or similar.\n"
            "- If yes, output the official original/native full form, even if it is in a different language from the question.\n"
            "- Otherwise, output the answer in the same language as the question.\n"
            "- If the conversation already contains a verified localized full name in the question language, use that form.\n"
            "- Use a localized full name in the question language only if that localized form is already supported by the conversation.\n"
            "- Do not invent or infer a translation that was not verified during the conversation.\n"
            "- If the question is in Chinese and the answer entity is found in English, prefer the verified official Chinese full name or verified standard Chinese translation.\n"
            "- If the question is in English, prefer the verified official English full name.\n"
            "- If the conversation shows that localization verification was completed and no reliable localized form was found, fall back to the verified official original full name.\n"
            "- If the question explicitly asks for the original/native form, ignore localization and output the verified original/native full form.\n"
            "- Always prefer the official full name/title over a short alias, abbreviation, shortened brand name, or partial name.\n"
            "- For people, output the full personal name.\n"
            "- For organizations, publishers, companies, schools, institutions, places, works, and titles, output the official full name/title.\n"
            "- Do not assume that a shorter or more familiar form is better if a fuller verified official name exists.\n"
            "- If both an abbreviated and a full official form appear in the conversation, prefer the full official form.\n"
            "- If both an original-language full name and a verified localized full name appear in the conversation, choose according to the question language unless the question explicitly requests the original/native form.\n"
            "- Do not output only a first name, only a surname, or a shortened form when a fuller official answer is supported by the conversation.\n"
            "- Do not remove words that are part of the official name.\n"
            "- The final summary stage cannot perform any new search. It must only normalize the answer based on evidence already present in the conversation.\n"
            "- Do not ask for or imply further tool use in the final summary stage.\n"
            "- Do not add sentence-final punctuation unless it is part of the official title or name itself.\n\n"
            "Step 3. Format the final answer.\n"
            "- Wrap the final answer in \\boxed{}.\n"
            "- Output only the final answer, not an explanation.\n"
            "- Strictly follow any formatting instructions stated in the original question, including language, ordering, separators, units, rounding, decimal places, capitalization, and required format.\n"
            "- If the answer is a number, output digits rather than words unless the question explicitly requires otherwise.\n"
            "- If the answer is a comma-separated list, apply the same normalization rules to each item.\n\n"
            "The original question is repeated here for reference:\n\n"
            f'"{task_description}"\n\n'
            "Your output must contain the final answer in \\boxed{}.\n"
            "Do NOT include any invisible or non-printable characters in the answer output.\n"
            "If you attempt to call any tool, it will be considered a mistake."
        )
    elif agent_type == "agent-browsing":
        summarize_prompt = (
            "This is a direct instruction to you (the assistant), not the result of a tool call.\n\n"
            "We are now ending this session, and your conversation history will be deleted. "
            "You must NOT initiate any further tool use. This is your final opportunity to report "
            "*all* of the information gathered during the session.\n\n"
            "The original task is repeated here for reference:\n\n"
            f'"{task_description}"\n\n'
            "Summarize the above search and browsing history. Output the FINAL RESPONSE and detailed supporting information of the task given to you.\n\n"
            "If you found any useful facts, data, quotes, or answers directly relevant to the original task, include them clearly and completely.\n"
            "If you reached a conclusion or answer, include it as part of the response.\n"
            "If the task could not be fully answered, do NOT make up any content. Instead, return all partially relevant findings, "
            "Search results, quotes, and observations that might help a downstream agent solve the problem.\n"
            "If partial, conflicting, or inconclusive information was found, clearly indicate this in your response.\n\n"
            "Your final response should be a clear, complete, and structured report.\n"
            "Organize the content into logical sections with appropriate headings.\n"
            "Do NOT include any tool call instructions, speculative filler, or vague summaries.\n"
            "Focus on factual, specific, and well-organized information."
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return summarize_prompt.strip()
