from src.utils.localization_gate_utils import (
    LocalizationGateDecision,
    parse_localization_gate_decision,
    should_run_localization_gate,
)


def test_parse_localization_gate_decision_accepts_valid_json():
    raw_text = """
    {
      "should_run_gate": true,
      "candidate_answer": "Arnoldo Mondadori Editore",
      "entity_type": "publisher",
      "question_language": "zh",
      "candidate_answer_language": "en",
      "original_name_requested": false,
      "localized_name_status": "unresolved",
      "reason": "Cross-language publisher answer is still unresolved."
    }
    """

    decision = parse_localization_gate_decision(raw_text)

    assert decision is not None
    assert decision.candidate_answer == "Arnoldo Mondadori Editore"
    assert decision.entity_type == "publisher"
    assert decision.question_language == "zh"
    assert decision.candidate_answer_language == "en"
    assert decision.localized_name_status == "unresolved"


def test_parse_localization_gate_decision_returns_none_for_invalid_json():
    assert parse_localization_gate_decision("not json") is None


def test_should_run_localization_gate_for_cross_language_named_entity():
    decision = LocalizationGateDecision(
        should_run_gate=True,
        candidate_answer="Arnoldo Mondadori Editore",
        entity_type="publisher",
        question_language="zh",
        candidate_answer_language="en",
        original_name_requested=False,
        localized_name_status="unresolved",
        reason="test",
    )

    assert should_run_localization_gate(decision) is True


def test_should_not_run_localization_gate_for_original_name_request():
    decision = LocalizationGateDecision(
        should_run_gate=True,
        candidate_answer="Alexandre Exquemelin",
        entity_type="person",
        question_language="zh",
        candidate_answer_language="en",
        original_name_requested=True,
        localized_name_status="unresolved",
        reason="test",
    )

    assert should_run_localization_gate(decision) is False


def test_should_not_run_localization_gate_for_non_named_entity():
    decision = LocalizationGateDecision(
        should_run_gate=True,
        candidate_answer="42",
        entity_type="non_named_entity",
        question_language="zh",
        candidate_answer_language="en",
        original_name_requested=False,
        localized_name_status="unresolved",
        reason="test",
    )

    assert should_run_localization_gate(decision) is False
