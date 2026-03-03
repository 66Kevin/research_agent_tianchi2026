import asyncio
import json
import os
import time
from typing import Any, Dict, Tuple

import aiohttp
import openai
from langchain.tools import tool
from pydantic import BaseModel, Field

import dotenv
dotenv.load_dotenv()


TOOL_SERVER_URL = None  # for custom tool server
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

MODEL_FOR_VISIT_SUMMARIZE = os.getenv("MODEL_FOR_VISIT_SUMMARIZE")
BASE_URL_FOR_VISIT_SUMMARIZE = os.getenv("BASE_URL_FOR_VISIT_SUMMARIZE")
API_KEY_FOR_VISIT_SUMMARIZE = os.getenv("API_KEY_FOR_VISIT_SUMMARIZE")

SEARCH_TIMEOUT_TOTAL_S = 20
SEARCH_TIMEOUT_CONNECT_S = 5
SEARCH_TIMEOUT_READ_S = 15

VISIT_FETCH_TIMEOUT_TOTAL_S = 25
VISIT_FETCH_TIMEOUT_CONNECT_S = 5
VISIT_FETCH_TIMEOUT_READ_S = 20

VISIT_SUMMARIZE_TIMEOUT_S = 45
TOOL_HTTP_RETRY_ATTEMPTS = int(os.getenv("TOOL_HTTP_RETRY_ATTEMPTS", "2"))
TOOL_HTTP_RETRY_STATUSES = {429, 500, 502, 503, 504}
TOOL_RETRY_BACKOFF_S = 0.6

print("=" * 100)
print(f"SERPER_API_KEY: {SERPER_API_KEY}")
print(f"JINA_API_KEY: {JINA_API_KEY}")
print(f"MODEL_FOR_VISIT_SUMMARIZE: {MODEL_FOR_VISIT_SUMMARIZE}")
print(f"BASE_URL_FOR_VISIT_SUMMARIZE: {BASE_URL_FOR_VISIT_SUMMARIZE}")
print(f"API_KEY_FOR_VISIT_SUMMARIZE: {API_KEY_FOR_VISIT_SUMMARIZE}")
print("=" * 100)

SUMMARIZE_PROMPT = """Please process the following webpage content and user goal to extract relevant
information:
## **Webpage Content**
{webpage_content}
## **User Goal**
{goal}
## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the specific sections/data directly
related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the most relevant
information from the content, output the full original context as far as possible
3. **Summary Output for Summary**: Organize into a concise paragraph with logical
flow, prioritizing clarity
**Final Output Format using JSON format has "rational", "evidence", "summary"
fields**
"""

class SearchInput(BaseModel):
    query: list[str] = Field(description="The list of search query strings")

class VisitInput(BaseModel):
    url: list[str] = Field(description="List of URLs to visit")
    goal: str = Field(description="The goal or question to answer from the URLs")

def _truncate_text(text: str, max_len: int = 700) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...[truncated {len(text) - max_len} chars]"


def _status(
    *,
    ok: bool,
    status: str,
    target: str,
    elapsed_ms: int,
    http_status: int | None = None,
    error_type: str = "",
    error_message: str = "",
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "http_status": http_status,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "error_type": error_type,
        "error_message": _truncate_text(error_message),
        "target": target,
    }


def _should_retry(http_status: int | None, status: str) -> bool:
    if status == "TIMEOUT":
        return True
    if http_status is None:
        return False
    return http_status in TOOL_HTTP_RETRY_STATUSES


async def _request_json_with_retry(
    *,
    method: str,
    url: str,
    target: str,
    timeout: aiohttp.ClientTimeout,
    headers: Dict[str, str] | None = None,
    payload: Dict[str, Any] | None = None,
    retry_attempts: int = TOOL_HTTP_RETRY_ATTEMPTS,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    last_status = _status(
        ok=False,
        status="EXCEPTION",
        target=target,
        elapsed_ms=0,
        error_type="UnknownError",
        error_message="Uninitialized tool call status.",
    )

    for attempt in range(1, retry_attempts + 1):
        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                response = await session.request(method=method, url=url, headers=headers, json=payload)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                http_status = response.status
                body_text = await response.text()
                if http_status >= 400:
                    status = _status(
                        ok=False,
                        status="HTTP_ERROR",
                        target=target,
                        elapsed_ms=elapsed_ms,
                        http_status=http_status,
                        error_type="HTTPError",
                        error_message=body_text or f"HTTP {http_status}",
                    )
                    if attempt < retry_attempts and _should_retry(http_status, status["status"]):
                        last_status = status
                        await asyncio.sleep(TOOL_RETRY_BACKOFF_S * attempt)
                        continue
                    return None, status

                try:
                    parsed = json.loads(body_text)
                except json.JSONDecodeError:
                    return None, _status(
                        ok=False,
                        status="EXCEPTION",
                        target=target,
                        elapsed_ms=elapsed_ms,
                        http_status=http_status,
                        error_type="InvalidJSON",
                        error_message=body_text,
                    )
                return parsed, _status(
                    ok=True,
                    status="OK",
                    target=target,
                    elapsed_ms=elapsed_ms,
                    http_status=http_status,
                )
        except asyncio.TimeoutError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            status = _status(
                ok=False,
                status="TIMEOUT",
                target=target,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc) or "Request timed out.",
            )
            if attempt < retry_attempts and _should_retry(None, status["status"]):
                last_status = status
                await asyncio.sleep(TOOL_RETRY_BACKOFF_S * attempt)
                continue
            return None, status
        except aiohttp.ClientError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            status = _status(
                ok=False,
                status="EXCEPTION",
                target=target,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if attempt < retry_attempts and _should_retry(None, status["status"]):
                last_status = status
                await asyncio.sleep(TOOL_RETRY_BACKOFF_S * attempt)
                continue
            return None, status
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return None, _status(
                ok=False,
                status="EXCEPTION",
                target=target,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    return None, last_status


async def _request_text_with_retry(
    *,
    method: str,
    url: str,
    target: str,
    timeout: aiohttp.ClientTimeout,
    headers: Dict[str, str] | None = None,
    payload: Dict[str, Any] | None = None,
    retry_attempts: int = TOOL_HTTP_RETRY_ATTEMPTS,
) -> Tuple[str | None, Dict[str, Any]]:
    last_status = _status(
        ok=False,
        status="EXCEPTION",
        target=target,
        elapsed_ms=0,
        error_type="UnknownError",
        error_message="Uninitialized tool call status.",
    )

    for attempt in range(1, retry_attempts + 1):
        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                response = await session.request(method=method, url=url, headers=headers, json=payload)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                http_status = response.status
                text = await response.text()
                if http_status >= 400:
                    status = _status(
                        ok=False,
                        status="HTTP_ERROR",
                        target=target,
                        elapsed_ms=elapsed_ms,
                        http_status=http_status,
                        error_type="HTTPError",
                        error_message=text or f"HTTP {http_status}",
                    )
                    if attempt < retry_attempts and _should_retry(http_status, status["status"]):
                        last_status = status
                        await asyncio.sleep(TOOL_RETRY_BACKOFF_S * attempt)
                        continue
                    return None, status
                return text, _status(
                    ok=True,
                    status="OK",
                    target=target,
                    elapsed_ms=elapsed_ms,
                    http_status=http_status,
                )
        except asyncio.TimeoutError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            status = _status(
                ok=False,
                status="TIMEOUT",
                target=target,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc) or "Request timed out.",
            )
            if attempt < retry_attempts and _should_retry(None, status["status"]):
                last_status = status
                await asyncio.sleep(TOOL_RETRY_BACKOFF_S * attempt)
                continue
            return None, status
        except aiohttp.ClientError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            status = _status(
                ok=False,
                status="EXCEPTION",
                target=target,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if attempt < retry_attempts and _should_retry(None, status["status"]):
                last_status = status
                await asyncio.sleep(TOOL_RETRY_BACKOFF_S * attempt)
                continue
            return None, status
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return None, _status(
                ok=False,
                status="EXCEPTION",
                target=target,
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    return None, last_status


async def serper_search(query: str) -> Tuple[list[Dict[str, Any]] | None, Dict[str, Any]]:
    if not SERPER_API_KEY:
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=query,
            elapsed_ms=0,
            error_type="MissingAPIKey",
            error_message="SERPER_API_KEY is not configured.",
        )

    result, status = await _request_json_with_retry(
        method="POST",
        url="https://google.serper.dev/search",
        target=query,
        timeout=aiohttp.ClientTimeout(
            total=SEARCH_TIMEOUT_TOTAL_S,
            connect=SEARCH_TIMEOUT_CONNECT_S,
            sock_read=SEARCH_TIMEOUT_READ_S,
        ),
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        payload={"q": query},
    )
    if not status["ok"] or result is None:
        return None, status

    organic = result.get("organic")
    if not isinstance(organic, list):
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=query,
            elapsed_ms=status["elapsed_ms"],
            http_status=status["http_status"],
            error_type="InvalidPayload",
            error_message=f"Missing 'organic' list in response: {result}",
        )
    return organic, status


async def _search(query: str) -> Tuple[list[Dict[str, Any]] | None, Dict[str, Any]]:
    if not TOOL_SERVER_URL:
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=query,
            elapsed_ms=0,
            error_type="MissingToolServerURL",
            error_message="TOOL_SERVER_URL is not configured.",
        )

    result, status = await _request_json_with_retry(
        method="POST",
        url=f"{TOOL_SERVER_URL}/search",
        target=query,
        timeout=aiohttp.ClientTimeout(
            total=SEARCH_TIMEOUT_TOTAL_S,
            connect=SEARCH_TIMEOUT_CONNECT_S,
            sock_read=SEARCH_TIMEOUT_READ_S,
        ),
        payload={"query": query, "provider": "google"},
    )
    if not status["ok"] or result is None:
        return None, status

    items = result.get("items")
    if not isinstance(items, list):
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=query,
            elapsed_ms=status["elapsed_ms"],
            http_status=status["http_status"],
            error_type="InvalidPayload",
            error_message=f"Missing 'items' list in response: {result}",
        )
    return items, status


async def jina_browse(url: str) -> Tuple[str | None, Dict[str, Any]]:
    if not JINA_API_KEY:
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=url,
            elapsed_ms=0,
            error_type="MissingAPIKey",
            error_message="JINA_API_KEY is not configured.",
        )

    return await _request_text_with_retry(
        method="GET",
        url=f"https://r.jina.ai/{url}",
        target=url,
        timeout=aiohttp.ClientTimeout(
            total=VISIT_FETCH_TIMEOUT_TOTAL_S,
            connect=VISIT_FETCH_TIMEOUT_CONNECT_S,
            sock_read=VISIT_FETCH_TIMEOUT_READ_S,
        ),
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    )


async def _summarize_visit_content(webpage_content: str, goal: str) -> Tuple[str | None, Dict[str, Any]]:
    target = f"visit_summarize:{_truncate_text(goal, 120)}"
    if not MODEL_FOR_VISIT_SUMMARIZE or not BASE_URL_FOR_VISIT_SUMMARIZE or not API_KEY_FOR_VISIT_SUMMARIZE:
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=target,
            elapsed_ms=0,
            error_type="MissingSummarizeConfig",
            error_message="MODEL_FOR_VISIT_SUMMARIZE/BASE_URL_FOR_VISIT_SUMMARIZE/API_KEY_FOR_VISIT_SUMMARIZE is missing.",
        )

    prompt = SUMMARIZE_PROMPT.format(webpage_content=webpage_content, goal=goal)
    client = openai.AsyncOpenAI(
        base_url=BASE_URL_FOR_VISIT_SUMMARIZE,
        api_key=API_KEY_FOR_VISIT_SUMMARIZE,
    )
    start = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_FOR_VISIT_SUMMARIZE,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=VISIT_SUMMARIZE_TIMEOUT_S,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        content = response.choices[0].message.content if response.choices else ""
        return content or "", _status(
            ok=True,
            status="OK",
            target=target,
            elapsed_ms=elapsed_ms,
            http_status=200,
        )
    except asyncio.TimeoutError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return None, _status(
            ok=False,
            status="TIMEOUT",
            target=target,
            elapsed_ms=elapsed_ms,
            error_type=type(exc).__name__,
            error_message=f"Visit summarize LLM timed out after {VISIT_SUMMARIZE_TIMEOUT_S}s.",
        )
    except openai.APIStatusError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return None, _status(
            ok=False,
            status="HTTP_ERROR",
            target=target,
            elapsed_ms=elapsed_ms,
            http_status=exc.status_code,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return None, _status(
            ok=False,
            status="EXCEPTION",
            target=target,
            elapsed_ms=elapsed_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _fallback_visit_summary(urls: list[str], goal: str, contents: list[str], calls: list[Dict[str, Any]]) -> str:
    if contents:
        joined = "\n\n".join(_truncate_text(content, 1200) for content in contents[:2])
        return (
            f"Best-effort fallback summary for goal: {goal}\n"
            f"Visited URLs: {urls}\n"
            f"Extracted raw content snippets:\n{joined}"
        )

    error_lines = []
    for call in calls:
        error_lines.append(
            f"- target={call.get('target')} status={call.get('status')} "
            f"http_status={call.get('http_status')} error={call.get('error_message')}"
        )
    return (
        f"Visit failed to fetch usable content for goal: {goal}\n"
        f"Visited URLs: {urls}\n"
        f"Errors:\n" + "\n".join(error_lines)
    )


async def jina_visit(urls: list[str], goal: str) -> Dict[str, Any]:
    browse_results = await asyncio.gather(*[jina_browse(url) for url in urls], return_exceptions=False)
    calls: list[Dict[str, Any]] = []
    webpage_contents: list[str] = []
    for content, call_status in browse_results:
        calls.append(call_status)
        if call_status.get("ok") and isinstance(content, str) and content:
            webpage_contents.append(content)

    success_urls = sum(1 for call in calls if call.get("ok"))
    failed_urls = len(calls) - success_urls
    summarize_status = _status(
        ok=False,
        status="EXCEPTION",
        target="visit_summarize:skipped",
        elapsed_ms=0,
        error_type="Skipped",
        error_message="Summarization skipped because no content was fetched.",
    )
    semantic_document = ""

    if webpage_contents:
        summary_content, summarize_status = await _summarize_visit_content("\n".join(webpage_contents), goal)
        if summarize_status.get("ok") and isinstance(summary_content, str) and summary_content.strip():
            semantic_document = (
                f"The useful information in {urls} for user goal {goal} as follows: {summary_content}"
            )

    if not semantic_document:
        semantic_document = _fallback_visit_summary(urls, goal, webpage_contents, calls)

    return {
        "ok": bool(semantic_document.strip()) and (success_urls > 0 or summarize_status.get("ok")),
        "summary": {
            "total_urls": len(urls),
            "success_urls": success_urls,
            "failed_urls": failed_urls,
        },
        "calls": calls,
        "summarize_status": summarize_status,
        "semanticDocument": semantic_document,
    }


async def _visit(urls: list[str], goal: str) -> Dict[str, Any]:
    if not TOOL_SERVER_URL:
        call = _status(
            ok=False,
            status="EXCEPTION",
            target="tool_server_visit",
            elapsed_ms=0,
            error_type="MissingToolServerURL",
            error_message="TOOL_SERVER_URL is not configured.",
        )
        return {
            "ok": False,
            "summary": {"total_urls": len(urls), "success_urls": 0, "failed_urls": len(urls)},
            "calls": [call],
            "summarize_status": call,
            "semanticDocument": _fallback_visit_summary(urls, goal, [], [call]),
        }

    result, call_status = await _request_json_with_retry(
        method="POST",
        url=f"{TOOL_SERVER_URL}/visit",
        target=f"tool_server_visit:{_truncate_text(goal, 120)}",
        timeout=aiohttp.ClientTimeout(
            total=VISIT_FETCH_TIMEOUT_TOTAL_S,
            connect=VISIT_FETCH_TIMEOUT_CONNECT_S,
            sock_read=VISIT_FETCH_TIMEOUT_READ_S,
        ),
        payload={"urls": urls, "goal": goal, "style": "tongyi"},
    )

    semantic_document = ""
    if call_status.get("ok") and isinstance(result, dict):
        semantic_document = str(result.get("semanticDocument", "")).strip()
    if not semantic_document:
        semantic_document = _fallback_visit_summary(urls, goal, [], [call_status])
    return {
        "ok": bool(semantic_document.strip()) and bool(call_status.get("ok")),
        "summary": {
            "total_urls": len(urls),
            "success_urls": len(urls) if call_status.get("ok") else 0,
            "failed_urls": 0 if call_status.get("ok") else len(urls),
        },
        "calls": [call_status],
        "summarize_status": call_status,
        "semanticDocument": semantic_document,
    }


@tool("search", args_schema=SearchInput, description="Search the web for information about a query using google search.")
async def search(query: list[str]) -> str:
    """Search the web for information about a query using google search."""

    search_func = _search if TOOL_SERVER_URL else serper_search
    batch_results = await asyncio.gather(*[search_func(q) for q in query], return_exceptions=True)

    calls: list[Dict[str, Any]] = []
    data: list[Dict[str, Any]] = []
    for idx, result in enumerate(batch_results):
        query_text = query[idx] if idx < len(query) else ""
        if isinstance(result, BaseException):
            calls.append(
                _status(
                    ok=False,
                    status="EXCEPTION",
                    target=query_text,
                    elapsed_ms=0,
                    error_type=type(result).__name__,
                    error_message=str(result),
                )
            )
            continue

        items, call_status = result
        calls.append(call_status)
        if call_status.get("ok") and isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item.pop("error_message", None)
                    data.append(item)
                else:
                    data.append({"value": str(item)})

    success_queries = sum(1 for call in calls if call.get("ok"))
    payload = {
        "tool": "search",
        "ok": success_queries > 0,
        "summary": {
            "total_queries": len(query),
            "success_queries": success_queries,
            "failed_queries": len(query) - success_queries,
        },
        "calls": calls,
        "data": data,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool("visit", args_schema=VisitInput, description="Visit multiple URLs and extract information based on a goal.")
async def visit(url: list[str], goal: str) -> str:
    """Visit multiple URLs and extract information based on a goal."""
    visit_func = _visit if TOOL_SERVER_URL else jina_visit
    result = await visit_func(url, goal)
    payload = {
        "tool": "visit",
        "ok": bool(result.get("ok")),
        "summary": result.get("summary", {}),
        "calls": result.get("calls", []),
        "summarize_status": result.get("summarize_status", {}),
        "data": {"semanticDocument": result.get("semanticDocument", "")},
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    async def test_search_tools(query: list[str]) -> str:
        result = await search.ainvoke({"query": query})
        print(result)

    asyncio.run(test_search_tools(["Elden Ring","Golden Tree"]))

    async def test_visit_tools(urls: list[str], goal: str) -> str:
        result = await visit.ainvoke({"url": urls, "goal": goal})
        print(result)

    asyncio.run(test_visit_tools(["https://en.wikipedia.org/wiki/Elden_Ring"], "Who is the main character of Elden Ring?"))

   
