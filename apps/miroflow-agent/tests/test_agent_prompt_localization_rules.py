from pathlib import Path


def test_tianchi_agent_prompt_contains_localization_rules():
    agent_cfg_path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "agent"
        / "mirothinker_v1.5_keep5_max200_tianchi.yaml"
    )

    prompt = agent_cfg_path.read_text()

    assert "LOCALIZED NAME / TITLE RESOLUTION RULES" in prompt
    assert "dedicated localization-verification round" in prompt
    assert "localized_name_status" in prompt
    assert "answer-form requirements are resolved" in prompt
    assert "Wikipedia or other major reference encyclopedias" in prompt
    assert "at most 3 high-information-gain queries" in prompt
    assert "at most 2 scrape calls" in prompt
