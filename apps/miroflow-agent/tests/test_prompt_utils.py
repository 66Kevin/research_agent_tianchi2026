from src.utils.prompt_utils import generate_agent_summarize_prompt


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
    assert "Wrap the final answer in \\boxed{}." in prompt
    assert "as few words as possible" not in prompt
    assert "don't use articles" not in prompt
    assert "Do NOT include any punctuation such as '.', '!', or '?'" not in prompt
