import json

import pytest

from miroflow_tools.dev_mcp_servers import jina_scrape_llm_summary as mod


pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_scrape_and_extract_info_returns_structured_json(monkeypatch):
    async def fake_scrape(url, custom_headers=None):
        return {
            "success": True,
            "content": "Example content about revenue being 42 in 2024.",
            "error": "",
            "line_count": 1,
            "char_count": 42,
            "last_char_line": 1,
            "all_content_displayed": True,
        }

    async def fake_extract(**kwargs):
        return {
            "success": True,
            "extracted_info": mod.normalize_extracted_info(
                json.dumps(
                    {
                        "rational": "The page directly states the revenue figure.",
                        "evidence": ["Revenue for 2024 was 42."],
                        "summary": "42",
                    }
                )
            ),
            "error": "",
            "model_used": "mock-model",
            "tokens_used": 12,
        }

    monkeypatch.setattr(mod, "scrape_url_with_jina", fake_scrape)
    monkeypatch.setattr(mod, "extract_info_with_llm", fake_extract)

    result = json.loads(
        await mod.scrape_and_extract_info(
            "https://example.com/report",
            "Target question: What was revenue in 2024?",
        )
    )
    extracted = json.loads(result["extracted_info"])

    assert result["success"] is True
    assert set(extracted.keys()) == {"rational", "evidence", "summary"}
    assert extracted["summary"] == "42"
    assert extracted["evidence"] == ["Revenue for 2024 was 42."]


@pytest.mark.asyncio
async def test_scrape_and_extract_info_uses_strict_not_found_contract(monkeypatch):
    async def fake_scrape(url, custom_headers=None):
        return {
            "success": True,
            "content": "This page discusses company history, not revenue.",
            "error": "",
            "line_count": 1,
            "char_count": 48,
            "last_char_line": 1,
            "all_content_displayed": True,
        }

    async def fake_extract(**kwargs):
        return {
            "success": True,
            "extracted_info": mod.normalize_extracted_info(
                json.dumps(
                    {
                        "rational": "Some unrelated explanation",
                        "evidence": ["irrelevant snippet"],
                        "summary": "NOT_FOUND",
                    }
                )
            ),
            "error": "",
            "model_used": "mock-model",
            "tokens_used": 8,
        }

    monkeypatch.setattr(mod, "scrape_url_with_jina", fake_scrape)
    monkeypatch.setattr(mod, "extract_info_with_llm", fake_extract)

    result = json.loads(
        await mod.scrape_and_extract_info(
            "https://example.com/history",
            "Target question: What was revenue in 2024?",
        )
    )
    extracted = json.loads(result["extracted_info"])

    assert result["success"] is True
    assert extracted == {
        "rational": mod.NOT_FOUND_RATIONAL,
        "evidence": [],
        "summary": mod.NOT_FOUND_SUMMARY,
    }


@pytest.mark.asyncio
async def test_extract_info_with_llm_normalizes_plain_text_response(monkeypatch):
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
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "The page states revenue was 42 in 2024."
                            }
                        }
                    ],
                    "usage": {"total_tokens": 21},
                }
            )

    monkeypatch.setattr(mod, "SUMMARY_LLM_BASE_URL", "https://llm.example.com")
    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)

    result = await mod.extract_info_with_llm(
        url="https://example.com/report",
        content="Revenue for 2024 was 42.",
        info_to_extract="Target question: What was revenue in 2024?",
        model="mock-model",
    )
    extracted = json.loads(result["extracted_info"])

    assert result["success"] is True
    assert extracted["rational"] == mod.FALLBACK_RATIONAL
    assert extracted["evidence"] == []
    assert extracted["summary"] == "The page states revenue was 42 in 2024."


def test_normalize_extracted_info_coerces_string_evidence():
    normalized = json.loads(
        mod.normalize_extracted_info(
            json.dumps(
                {
                    "rational": "The answer is explicitly stated.",
                    "evidence": "Revenue for 2024 was 42.",
                    "summary": "42",
                }
            )
        )
    )

    assert normalized["evidence"] == ["Revenue for 2024 was 42."]
    assert normalized["summary"] == "42"


def test_normalize_extracted_info_defaults_missing_keys():
    normalized = json.loads(mod.normalize_extracted_info(json.dumps({"summary": "42"})))

    assert normalized == {
        "rational": "",
        "evidence": [],
        "summary": "42",
    }
