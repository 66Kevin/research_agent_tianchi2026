from src.utils.localization_gate_utils import (
    decide_localization_gate_mode_from_remaining,
    LocalizationGateBudgetDecision,
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
      "target_answer_language": "zh",
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
    assert decision.target_answer_language == "zh"
    assert decision.candidate_answer_language == "en"
    assert decision.localized_name_status == "unresolved"


def test_parse_localization_gate_decision_falls_back_target_language_to_question_language():
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
    assert decision.target_answer_language == "zh"


def test_parse_localization_gate_decision_returns_none_for_invalid_json():
    assert parse_localization_gate_decision("not json") is None


def test_should_run_localization_gate_for_cross_language_named_entity():
    decision = LocalizationGateDecision(
        should_run_gate=True,
        candidate_answer="Arnoldo Mondadori Editore",
        entity_type="publisher",
        question_language="zh",
        target_answer_language="zh",
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
        target_answer_language="zh",
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
        target_answer_language="zh",
        candidate_answer_language="en",
        original_name_requested=False,
        localized_name_status="unresolved",
        reason="test",
    )

    assert should_run_localization_gate(decision) is False


def test_should_not_run_localization_gate_when_target_language_matches_candidate_language():
    decision = LocalizationGateDecision(
        should_run_gate=True,
        candidate_answer="Arnoldo Mondadori Editore",
        entity_type="publisher",
        question_language="zh",
        target_answer_language="en",
        candidate_answer_language="en",
        original_name_requested=False,
        localized_name_status="unresolved",
        reason="Explicit English answer requested.",
    )

    assert should_run_localization_gate(decision) is False


def test_should_run_localization_gate_when_explicit_target_language_differs_from_question_language():
    decision = LocalizationGateDecision(
        should_run_gate=True,
        candidate_answer="Arnoldo Mondadori Editore",
        entity_type="publisher",
        question_language="en",
        target_answer_language="zh",
        candidate_answer_language="en",
        original_name_requested=False,
        localized_name_status="unresolved",
        reason="Question is English but explicitly requests a Chinese answer.",
    )

    assert should_run_localization_gate(decision) is True


def test_decide_localization_gate_mode_full():
    budget_decision = decide_localization_gate_mode_from_remaining(
        remaining_seconds=65.0,
        final_summary_reserve_seconds=40.0,
        full_min_remaining_seconds=20.0,
        degraded_min_remaining_seconds=8.0,
    )

    assert isinstance(budget_decision, LocalizationGateBudgetDecision)
    assert budget_decision.mode == "full"


def test_decide_localization_gate_mode_degraded():
    budget_decision = decide_localization_gate_mode_from_remaining(
        remaining_seconds=50.0,
        final_summary_reserve_seconds=40.0,
        full_min_remaining_seconds=20.0,
        degraded_min_remaining_seconds=8.0,
    )

    assert budget_decision.mode == "degraded"


def test_decide_localization_gate_mode_skip():
    budget_decision = decide_localization_gate_mode_from_remaining(
        remaining_seconds=45.0,
        final_summary_reserve_seconds=40.0,
        full_min_remaining_seconds=20.0,
        degraded_min_remaining_seconds=8.0,
    )

    assert budget_decision.mode == "skip"
