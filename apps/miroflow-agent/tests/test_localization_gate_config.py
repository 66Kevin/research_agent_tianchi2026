from pathlib import Path

from omegaconf import OmegaConf


def test_tianchi_benchmark_uses_30s_gate_and_30s_final_summary_reserves():
    benchmark_cfg_path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "benchmark"
        / "tianchi-validation.yaml"
    )

    cfg = OmegaConf.load(benchmark_cfg_path)

    assert cfg.execution.task_timeout_seconds == 600
    assert cfg.execution.localization_gate_reserve_seconds == 30
    assert cfg.execution.final_summary_reserve_seconds == 40
    assert cfg.execution.localization_gate_full_min_remaining_seconds == 20
    assert cfg.execution.localization_gate_degraded_min_remaining_seconds == 8
    assert (
        cfg.execution.task_timeout_seconds
        - cfg.execution.localization_gate_reserve_seconds
        - cfg.execution.final_summary_reserve_seconds
        == 530
    )
