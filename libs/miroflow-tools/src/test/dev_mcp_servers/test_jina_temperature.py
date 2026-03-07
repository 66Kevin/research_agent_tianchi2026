import json

import pytest

from miroflow_tools.dev_mcp_servers import jina_scrape_llm_summary as mod


pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_temperature", "expected_temperature"),
    [
        ("0.2", 0.2),
        (None, 1.0),
        ("not-a-number", 1.0),
    ],
)
async def test_extract_info_with_llm_uses_configured_temperature(
    monkeypatch, configured_temperature, expected_temperature
):
    captured_payload = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload, ensure_ascii=False)

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            captured_payload.update(kwargs["json"])
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "rational": "supported",
                                        "evidence": ["Revenue for 2024 was 42."],
                                        "summary": "42",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"total_tokens": 9},
                }
            )

    monkeypatch.setattr(mod, "SUMMARY_LLM_BASE_URL", "https://llm.example.com")
    monkeypatch.setattr(mod, "SUMMARY_LLM_TEMPERATURE", configured_temperature)
    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)

    result = await mod.extract_info_with_llm(
        url="https://example.com/report",
        content="Revenue for 2024 was 42.",
        info_to_extract="Target question: What was revenue in 2024?",
        model="qwen3.5-flash",
    )

    assert result["success"] is True
    assert captured_payload["temperature"] == expected_temperature
