"""Helpers for pre-summary localization gate decision parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Optional


NAMED_ENTITY_TYPES = {
    "person",
    "organization",
    "publisher",
    "company",
    "school",
    "institution",
    "place",
    "work",
    "title",
    "other_named_entity",
}

ENTITY_TYPE_VALUES = NAMED_ENTITY_TYPES | {"non_named_entity", "unknown"}
LANGUAGE_VALUES = {"zh", "en", "other", "mixed", "unknown"}
LOCALIZATION_STATUS_VALUES = {
    "resolved_verified",
    "resolved_best_effort",
    "resolved_not_found",
    "unresolved",
    "skip_original_requested",
    "not_applicable",
}


@dataclass(frozen=True)
class LocalizationGateDecision:
    should_run_gate: bool
    candidate_answer: str
    entity_type: str
    question_language: str
    candidate_answer_language: str
    original_name_requested: bool
    localized_name_status: str
    reason: str

    @property
    def is_named_entity(self) -> bool:
        return self.entity_type in NAMED_ENTITY_TYPES


def should_run_localization_gate(
    decision: Optional["LocalizationGateDecision"],
) -> bool:
    """Return whether the pre-summary localization gate should run."""

    if decision is None:
        return False

    return (
        decision.should_run_gate
        and decision.is_named_entity
        and decision.question_language != decision.candidate_answer_language
        and not decision.original_name_requested
        and decision.localized_name_status == "unresolved"
    )


def _extract_json_candidate(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced_match:
        return fenced_match.group(1).strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]

    return text


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _coerce_enum(value: object, allowed: set[str], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in allowed:
            return normalized
    return default


def parse_localization_gate_decision(
    raw_text: str,
) -> Optional[LocalizationGateDecision]:
    """Parse a JSON-only gate decision response into a validated object."""

    json_candidate = _extract_json_candidate(raw_text)
    if not json_candidate:
        return None

    try:
        payload = json.loads(json_candidate)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    decision = LocalizationGateDecision(
        should_run_gate=_coerce_bool(payload.get("should_run_gate"), False),
        candidate_answer=str(payload.get("candidate_answer", "")).strip(),
        entity_type=_coerce_enum(
            payload.get("entity_type"), ENTITY_TYPE_VALUES, "unknown"
        ),
        question_language=_coerce_enum(
            payload.get("question_language"), LANGUAGE_VALUES, "unknown"
        ),
        candidate_answer_language=_coerce_enum(
            payload.get("candidate_answer_language"), LANGUAGE_VALUES, "unknown"
        ),
        original_name_requested=_coerce_bool(
            payload.get("original_name_requested"), False
        ),
        localized_name_status=_coerce_enum(
            payload.get("localized_name_status"),
            LOCALIZATION_STATUS_VALUES,
            "unresolved",
        ),
        reason=str(payload.get("reason", "")).strip(),
    )

    return decision
