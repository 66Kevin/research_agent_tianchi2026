# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""Helpers for resolving stage-specific LLM temperature overrides."""

from omegaconf import DictConfig


def resolve_temperature(cfg: DictConfig, stage_name: str) -> float:
    """
    Resolve the effective temperature for a given stage.

    Falls back to the global `cfg.llm.temperature` when the stage override is
    missing, null, or invalid.
    """

    default_temperature = float(cfg.llm.temperature)
    overrides = cfg.llm.get("temperature_overrides")

    if not overrides:
        return default_temperature

    stage_temperature = overrides.get(stage_name)
    if stage_temperature is None:
        return default_temperature

    try:
        return float(stage_temperature)
    except (TypeError, ValueError):
        return default_temperature
