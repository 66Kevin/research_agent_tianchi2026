from src.utils.prompt_utils import (
    generate_agent_summarize_prompt,
    generate_localization_gate_decision_prompt,
    generate_localization_gate_prompt,
    generate_localization_gate_result_prompt,
)


def test_main_agent_summarize_prompt_uses_three_step_normalization_rules():
    prompt = generate_agent_summarize_prompt(
        "请回答这家出版社的官方全名。",
        agent_type="main",
    )

    assert "Step 1. Determine the semantic answer" in prompt
    assert "Step 2. Normalize the answer into the correct final form." in prompt
    assert "Step 3. Format the final answer." in prompt
    assert "same language as the question" in prompt
    assert "official full name" in prompt
    assert "official original/native full form" in prompt
    assert "Localization Gate Result" in prompt
    assert "authoritative localization status" in prompt
    assert "usually first name + last name" in prompt
    assert "Do not automatically include a middle name" in prompt
    assert "cannot perform any new search" in prompt
    assert "Wrap the final answer in \\boxed{}." in prompt
    assert "as few words as possible" not in prompt
    assert "don't use articles" not in prompt
    assert "Do NOT include any punctuation such as '.', '!', or '?'" not in prompt
    assert "Do not invent or infer a translation" not in prompt
    assert (
        "If no reliable localized official name is available in the conversation, "
        "fall back to the official original full name."
        not in prompt
    )
    assert (
        "If no verified localized form was found during the conversation, you may still produce a best-effort localized rendering"
        not in prompt
    )


def test_localization_gate_decision_prompt_is_json_only_and_no_tools():
    prompt = generate_localization_gate_decision_prompt("请回答出版社中文名。")

    assert "Do NOT call any tools." in prompt
    assert "Return JSON only" in prompt
    assert '"should_run_gate"' in prompt
    assert '"localized_name_status"' in prompt


def test_localization_gate_prompt_limits_scope_and_tools():
    prompt = generate_localization_gate_prompt(
        task_description="请给出该人物的中文标准姓名。",
        candidate_answer="Alexandre Exquemelin",
        entity_type="person",
        question_language="zh",
    )

    assert "Pre-Summary Localization Gate" in prompt
    assert "Do NOT reopen the broader task-solving process." in prompt
    assert "at most 2 tool calls" in prompt
    assert "first name + last name" in prompt
    assert "Do not automatically include middle names" in prompt


def test_localization_gate_result_prompt_requests_structured_result_block():
    prompt = generate_localization_gate_result_prompt(
        task_description="请给出该出版社的中文全称。",
        candidate_answer="Arnoldo Mondadori Editore",
        entity_type="publisher",
    )

    assert "Localization Gate Result" in prompt
    assert "localized_name_status: verified / best_effort / NOT_FOUND" in prompt
    assert "Do NOT call any tools." in prompt
    assert "Do not include any extra sections, JSON, or markdown." in prompt
