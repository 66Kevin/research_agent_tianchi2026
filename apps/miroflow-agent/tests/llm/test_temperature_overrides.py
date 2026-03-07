from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from src.config.settings import create_mcp_server_parameters
from src.core.answer_generator import AnswerGenerator
from src.io.output_formatter import OutputFormatter
from src.utils.temperature_utils import resolve_temperature


class DummyTaskLog:
    def log_step(self, *args, **kwargs):
        return None


class DummyStreamHandler:
    async def show_error(self, *args, **kwargs):
        return None


def make_cfg(overrides=None, default_temperature=1.0):
    return OmegaConf.create(
        {
            "llm": {
                "provider": "qwen",
                "model_name": "qwen3.5-plus",
                "temperature": default_temperature,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                "max_context_length": 262144,
                "max_tokens": 16384,
                "async_client": True,
                "repetition_penalty": 1.05,
                "temperature_overrides": (
                    overrides
                    if overrides is not None
                    else {
                        "main_agent": 0.4,
                        "final_summary": 0.2,
                        "jina_summary": 0.2,
                    }
                ),
            },
            "agent": {
                "keep_tool_result": 5,
                "context_compress_limit": 5,
                "main_agent": {"max_turns": 200},
                "sub_agents": None,
            },
            "benchmark": {"execution": {}},
        }
    )


def make_answer_generator(cfg):
    llm_client = Mock()
    llm_client.create_message = AsyncMock()
    llm_client.process_llm_response = Mock()
    llm_client.extract_tool_calls_info = Mock()

    return AnswerGenerator(
        llm_client=llm_client,
        output_formatter=OutputFormatter(),
        task_log=DummyTaskLog(),
        stream_handler=DummyStreamHandler(),
        cfg=cfg,
        intermediate_boxed_answers=[],
    )


def test_resolve_temperature_uses_stage_override_and_fallback():
    cfg = make_cfg()
    assert resolve_temperature(cfg, "main_agent") == 0.4
    assert resolve_temperature(cfg, "final_summary") == 0.2
    assert resolve_temperature(cfg, "jina_summary") == 0.2

    fallback_cfg = make_cfg(
        overrides={
            "main_agent": None,
            "final_summary": None,
            "jina_summary": None,
        },
        default_temperature=0.7,
    )
    assert resolve_temperature(fallback_cfg, "main_agent") == 0.7
    assert resolve_temperature(fallback_cfg, "final_summary") == 0.7
    assert resolve_temperature(fallback_cfg, "jina_summary") == 0.7


@pytest.mark.asyncio
async def test_handle_llm_call_uses_main_agent_temperature_by_default():
    cfg = make_cfg()
    answer_generator = make_answer_generator(cfg)

    answer_generator.llm_client.create_message.return_value = (
        "provider-response",
        [{"role": "user", "content": "task"}],
    )
    answer_generator.llm_client.process_llm_response.return_value = (
        "assistant-response",
        False,
        [{"role": "assistant", "content": "assistant-response"}],
    )
    answer_generator.llm_client.extract_tool_calls_info.return_value = []

    await answer_generator.handle_llm_call(
        system_prompt="system",
        message_history=[{"role": "user", "content": "task"}],
        tool_definitions=[],
        step_id=1,
        purpose="Main agent | Turn: 1",
        agent_type="main",
    )

    assert (
        answer_generator.llm_client.create_message.await_args.kwargs[
            "temperature_override"
        ]
        == 0.4
    )


@pytest.mark.asyncio
async def test_generate_final_answer_uses_final_summary_temperature():
    cfg = make_cfg()
    answer_generator = make_answer_generator(cfg)
    answer_generator.handle_llm_call = AsyncMock(
        return_value=(
            r"\boxed{RepRapPro Limited}",
            False,
            None,
            [{"role": "assistant", "content": r"\boxed{RepRapPro Limited}"}],
        )
    )

    await answer_generator.generate_final_answer_with_retries(
        system_prompt="system",
        message_history=[{"role": "user", "content": "task"}],
        tool_definitions=[],
        turn_count=3,
        task_description="task",
    )

    assert (
        answer_generator.handle_llm_call.await_args.kwargs["temperature_override"]
        == 0.2
    )


@pytest.mark.asyncio
async def test_generate_failure_summary_uses_final_summary_temperature():
    cfg = make_cfg()
    answer_generator = make_answer_generator(cfg)
    answer_generator.handle_llm_call = AsyncMock(
        return_value=(
            "Failure type: incomplete\nWhat happened: budget reached\nUseful findings: candidate A was weak",
            False,
            None,
            [],
        )
    )

    await answer_generator.generate_failure_summary(
        system_prompt="system",
        message_history=[{"role": "assistant", "content": "partial"}],
        tool_definitions=[],
        turn_count=5,
    )

    assert (
        answer_generator.handle_llm_call.await_args.kwargs["temperature_override"]
        == 0.2
    )


def test_create_mcp_server_parameters_sets_summary_temperature_env():
    cfg = make_cfg()
    agent_cfg = OmegaConf.create({"tools": ["jina_scrape_llm_summary"]})

    configs, _ = create_mcp_server_parameters(cfg, agent_cfg)
    jina_config = next(item for item in configs if item["name"] == "jina_scrape_llm_summary")

    assert jina_config["params"].env["SUMMARY_LLM_TEMPERATURE"] == "0.2"
