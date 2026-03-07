# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import asyncio
import json
import logging
import os
from typing import Any, Dict

import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging
logger = logging.getLogger("miroflow")

SUMMARY_LLM_BASE_URL = os.environ.get("SUMMARY_LLM_BASE_URL")
SUMMARY_LLM_MODEL_NAME = os.environ.get("SUMMARY_LLM_MODEL_NAME")
SUMMARY_LLM_API_KEY = os.environ.get("SUMMARY_LLM_API_KEY")
SUMMARY_LLM_TEMPERATURE = os.environ.get("SUMMARY_LLM_TEMPERATURE")

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://r.jina.ai")

# Initialize FastMCP server
mcp = FastMCP("jina_scrape_llm_summary")

NOT_FOUND_SUMMARY = "NOT_FOUND"
NOT_FOUND_RATIONAL = "Target not found in the provided webpage content."
FALLBACK_RATIONAL = (
    "Fallback normalization applied because the model did not return valid JSON."
)


def _get_summary_llm_temperature(default: float = 1.0) -> float:
    """Parse the configured summary LLM temperature with safe fallback."""
    if not SUMMARY_LLM_TEMPERATURE or not SUMMARY_LLM_TEMPERATURE.strip():
        return default

    try:
        return float(SUMMARY_LLM_TEMPERATURE)
    except ValueError:
        return default


@mcp.tool()
async def scrape_and_extract_info(
    url: str, info_to_extract: str, custom_headers: Dict[str, str] = None
):
    """
    Scrape content from a URL, including web pages, PDFs, code files, and other
    supported resources, then extract target-specific information using an LLM.
    If you need to extract information from a PDF, please use this tool.

    Args:
        url (str): The URL to scrape content from. Supports various types of URLs such as web pages, PDFs, raw text/code files (e.g., GitHub, Gist), and similar sources.
        info_to_extract (str): The visit target for this page. Use a precise sub-question
            rather than a generic summary request.
        custom_headers (Dict[str, str]): Additional headers to include in the scraping request

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - url (str): The original URL
            - extracted_info (str): A JSON string with `rational`, `evidence`, and
              `summary` fields
            - error (str): Error message if the operation failed
            - scrape_stats (Dict): Statistics about the scraped content
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """
    if _is_huggingface_dataset_or_space_url(url):
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose.",
                "scrape_stats": {},
                "tokens_used": 0,
            },
            ensure_ascii=False,
        )

    # First, scrape the content with Jina
    scrape_result = await scrape_url_with_jina(url, custom_headers)

    # If Jina fails, try direct Python scraping as fallback
    if not scrape_result["success"]:
        logger.warning(
            f"Jina Scrape and Extract Info: Jina scraping failed: {scrape_result['error']}, trying direct Python scraping as fallback"
        )
        scrape_result = await scrape_url_with_python(url, custom_headers)

        if not scrape_result["success"]:
            logger.error(
                f"Jina Scrape and Extract Info: Both Jina and Python scraping failed: {scrape_result['error']}"
            )
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": f"Scraping failed (both Jina and Python): {scrape_result['error']}",
                    "scrape_stats": {},
                    "tokens_used": 0,
                },
                ensure_ascii=False,
            )
        else:
            logger.info(
                f"Jina Scrape and Extract Info: Python fallback scraping succeeded for URL: {url}"
            )

    # Then, extract target-specific information from the content
    extracted_result = await extract_info_with_llm(
        url=url,
        content=scrape_result["content"],
        info_to_extract=info_to_extract,
        model=SUMMARY_LLM_MODEL_NAME,
        max_tokens=8192,
    )

    # Combine results
    return json.dumps(
        {
            "success": extracted_result["success"],
            "url": url,
            "extracted_info": extracted_result["extracted_info"],
            "error": extracted_result["error"],
            "scrape_stats": {
                "line_count": scrape_result["line_count"],
                "char_count": scrape_result["char_count"],
                "last_char_line": scrape_result["last_char_line"],
                "all_content_displayed": scrape_result["all_content_displayed"],
            },
            "model_used": extracted_result["model_used"],
            "tokens_used": extracted_result["tokens_used"],
        },
        ensure_ascii=False,
    )


def _is_huggingface_dataset_or_space_url(url):
    """
    Check if the URL is a HuggingFace dataset or space URL.
    :param url: The URL to check
    :return: True if it's a HuggingFace dataset or space URL, False otherwise
    """
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


async def scrape_url_with_jina(
    url: str, custom_headers: Dict[str, str] = None, max_chars: int = 102400 * 4
) -> Dict[str, Any]:
    """
    Scrape content from a URL and save to a temporary file. Need to read the content from the temporary file.


    Args:
        url (str): The URL to scrape content from
        custom_headers (Dict[str, str]): Additional headers to include in the request
        max_chars (int): Maximum number of characters to reserve for the scraped content

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - filename (str): Absolute path to the temporary file containing the scraped content
            - content (str): The scraped content of the first 40k characters
            - error (str): Error message if the operation failed
            - line_count (int): Number of lines in the scraped content
            - char_count (int): Number of characters in the scraped content
            - last_char_line (int): Line number where the last displayed character is located
            - all_content_displayed (bool): Signal indicating if all content was displayed (True if content <= 40k chars)
    """

    # Validate input
    if not url or not url.strip():
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "URL cannot be empty",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get API key from environment
    if not JINA_API_KEY:
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "JINA_API_KEY environment variable is not set",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Avoid duplicate Jina URL prefix
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    # Construct the Jina.ai API URL
    jina_url = f"{JINA_BASE_URL}/{url}"

    try:
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }

        # Add custom headers if provided
        if custom_headers:
            headers.update(custom_headers)

        # Retry configuration
        retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(retry_delays, 1):
            try:
                # Make the request using httpx library
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        jina_url,
                        headers=headers,
                        timeout=httpx.Timeout(None, connect=20, read=60),
                        follow_redirects=True,  # Follow redirects (equivalent to curl -L)
                    )

                # Check if request was successful
                response.raise_for_status()
                break  # Success, exit retry loop

            except httpx.ConnectTimeout as e:
                # connection timeout, retry
                if attempt < len(retry_delays):
                    logger.info(
                        f"Jina Scrape: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.ConnectError as e:
                # connection error, retry
                if attempt < len(retry_delays):
                    logger.info(
                        f"Jina Scrape: Connection error: {e}, {delay}s before next attempt"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.ReadTimeout as e:
                # read timeout, retry
                if attempt < len(retry_delays):
                    logger.info(
                        f"Jina Scrape: Read timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Read timeout retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Retryable: 5xx (server errors) + specific 4xx (408, 409, 425, 429)
                should_retry = status_code >= 500 or status_code in [408, 409, 425, 429]

                if should_retry and attempt < len(retry_delays):
                    logger.info(
                        f"Jina Scrape: HTTP {status_code} (retryable), retry in {delay}s, url: {url}"
                    )
                    await asyncio.sleep(delay)
                    continue
                elif should_retry:
                    logger.error(
                        f"Jina Scrape: HTTP {status_code} retry exhausted, url: {url}"
                    )
                    raise e
                else:
                    logger.error(
                        f"Jina Scrape: HTTP {status_code} (non-retryable), url: {url}"
                    )
                    raise e

            except httpx.RequestError as e:
                if attempt < len(retry_delays):
                    logger.info(
                        f"Jina Scrape: Unknown request exception: {e}, url: {url}, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Unknown request exception retry attempts exhausted, url: {url}"
                    )
                    raise e

    except Exception as e:
        error_msg = f"Jina Scrape: Unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": error_msg,
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get the scraped content
    content = response.text

    if not content:
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "No content returned from Jina.ai API",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # handle insufficient balance error
    try:
        content_dict = json.loads(content)
    except json.JSONDecodeError:
        content_dict = None
    if (
        isinstance(content_dict, dict)
        and content_dict.get("name") == "InsufficientBalanceError"
    ):
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "Insufficient balance",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get content statistics
    total_char_count = len(content)
    total_line_count = content.count("\n") + 1 if content else 0

    # Extract first max_chars characters
    displayed_content = content[:max_chars]
    all_content_displayed = total_char_count <= max_chars

    # Calculate the line number of the last character displayed
    if displayed_content:
        # Count newlines up to the last displayed character
        last_char_line = displayed_content.count("\n") + 1
    else:
        last_char_line = 0

    return {
        "success": True,
        "content": displayed_content,
        "error": "",
        "line_count": total_line_count,
        "char_count": total_char_count,
        "last_char_line": last_char_line,
        "all_content_displayed": all_content_displayed,
    }


async def scrape_url_with_python(
    url: str, custom_headers: Dict[str, str] = None, max_chars: int = 102400 * 4
) -> Dict[str, Any]:
    """
    Fallback scraping method using Python's httpx library directly.

    Args:
        url (str): The URL to scrape content from
        custom_headers (Dict[str, str]): Additional headers to include in the request
        max_chars (int): Maximum number of characters to reserve for the scraped content

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - content (str): The scraped content
            - error (str): Error message if the operation failed
            - line_count (int): Number of lines in the scraped content
            - char_count (int): Number of characters in the scraped content
            - last_char_line (int): Line number where the last displayed character is located
            - all_content_displayed (bool): Signal indicating if all content was displayed
    """
    # Validate input
    if not url or not url.strip():
        return {
            "success": False,
            "content": "",
            "error": "URL cannot be empty",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    try:
        # Prepare headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Add custom headers if provided
        if custom_headers:
            headers.update(custom_headers)

        # Retry configuration
        retry_delays = [1, 2, 4]

        for attempt, delay in enumerate(retry_delays, 1):
            try:
                # Make the request using httpx library
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url,
                        headers=headers,
                        timeout=httpx.Timeout(None, connect=20, read=60),
                        follow_redirects=True,
                    )

                # Check if request was successful
                response.raise_for_status()
                break  # Success, exit retry loop

            except httpx.ConnectTimeout as e:
                if attempt < len(retry_delays):
                    logger.info(
                        f"Python Scrape: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Python Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.ConnectError as e:
                if attempt < len(retry_delays):
                    logger.info(
                        f"Python Scrape: Connection error: {e}, {delay}s before next attempt"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Python Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.ReadTimeout as e:
                if attempt < len(retry_delays):
                    logger.info(
                        f"Python Scrape: Read timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Python Scrape: Read timeout retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Retryable: 5xx (server errors) + specific 4xx (408, 409, 425, 429)
                should_retry = status_code >= 500 or status_code in [408, 409, 425, 429]

                if should_retry and attempt < len(retry_delays):
                    logger.info(
                        f"Python Scrape: HTTP {status_code} (retryable), retry in {delay}s, url: {url}"
                    )
                    await asyncio.sleep(delay)
                    continue
                elif should_retry:
                    logger.error(
                        f"Python Scrape: HTTP {status_code} retry exhausted, url: {url}"
                    )
                    raise e
                else:
                    logger.error(
                        f"Python Scrape: HTTP {status_code} (non-retryable), url: {url}"
                    )
                    raise e

            except httpx.RequestError as e:
                if attempt < len(retry_delays):
                    logger.info(
                        f"Python Scrape: Unknown request exception: {e}, url: {url}, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Python Scrape: Unknown request exception retry attempts exhausted, url: {url}"
                    )
                    raise e

    except Exception as e:
        error_msg = f"Python Scrape: Unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "content": "",
            "error": error_msg,
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get the scraped content
    content = response.text

    if not content:
        return {
            "success": False,
            "content": "",
            "error": "No content returned from URL",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get content statistics
    total_char_count = len(content)
    total_line_count = content.count("\n") + 1 if content else 0

    # Extract first max_chars characters
    displayed_content = content[:max_chars]
    all_content_displayed = total_char_count <= max_chars

    # Calculate the line number of the last character displayed
    if displayed_content:
        last_char_line = displayed_content.count("\n") + 1
    else:
        last_char_line = 0

    return {
        "success": True,
        "content": displayed_content,
        "error": "",
        "line_count": total_line_count,
        "char_count": total_char_count,
        "last_char_line": last_char_line,
        "all_content_displayed": all_content_displayed,
    }


EXTRACT_INFO_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## Webpage Content
{webpage_content}

## User Goal
{goal}

## Task Guidelines
1. Content Scanning for `rational`: locate the sections or data directly related to the user goal.
2. Key Extraction for Evidence: extract the most relevant original context and preserve the original wording as much as possible.
3. Summary Output: write a concise answer-focused summary with logical flow.
4. If the page does not contain enough information to answer the goal, do not infer. Return `summary` as `NOT_FOUND`.
5. Output valid JSON only with keys `rational`, `evidence`, and `summary`.
6. Do not include markdown fences, extra commentary, or unrelated page summary.

Output schema:
{{
  "rational": "...",
  "evidence": ["...", "..."],
  "summary": "..."
}}"""


def _build_not_found_payload() -> Dict[str, Any]:
    return {
        "rational": NOT_FOUND_RATIONAL,
        "evidence": [],
        "summary": NOT_FOUND_SUMMARY,
    }


def normalize_extracted_info(raw_output: str) -> str:
    """Normalize LLM output into the strict extracted-info JSON contract."""
    text = (raw_output or "").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if NOT_FOUND_SUMMARY in text.upper():
            normalized = _build_not_found_payload()
        else:
            normalized = {
                "rational": FALLBACK_RATIONAL,
                "evidence": [],
                "summary": text,
            }
        return json.dumps(normalized, ensure_ascii=False)

    if not isinstance(parsed, dict):
        normalized = {
            "rational": FALLBACK_RATIONAL,
            "evidence": [],
            "summary": text,
        }
        return json.dumps(normalized, ensure_ascii=False)

    rational = str(parsed.get("rational", "") or "").strip()
    summary = str(parsed.get("summary", "") or "").strip()
    evidence = parsed.get("evidence", [])

    if isinstance(evidence, str):
        evidence = [evidence]
    elif isinstance(evidence, list):
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
    elif evidence:
        evidence = [str(evidence).strip()]
    else:
        evidence = []

    if summary.upper() == NOT_FOUND_SUMMARY:
        normalized = _build_not_found_payload()
    else:
        normalized = {
            "rational": rational,
            "evidence": evidence,
            "summary": summary,
        }

    return json.dumps(normalized, ensure_ascii=False)


def get_prompt_with_truncation(
    info_to_extract: str, content: str, truncate_last_num_chars: int = -1
) -> str:
    if truncate_last_num_chars > 0:
        content = content[:-truncate_last_num_chars] + "[...truncated]"

    # Prepare the prompt
    prompt = EXTRACT_INFO_PROMPT.format(
        webpage_content=content,
        goal=info_to_extract,
    )
    return prompt


async def extract_info_with_llm(
    url: str,
    content: str,
    info_to_extract: str,
    model: str = "LLM",
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Extract target-specific information from scraped content using an LLM.

    Args:
        content (str): The scraped page content to analyze
        info_to_extract (str): The visit target for this page
        model (str): The model to use for targeted extraction
        max_tokens (int): Maximum tokens for the response

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - extracted_info (str): A JSON string with `rational`, `evidence`, and
              `summary`
            - error (str): Error message if the operation failed
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """

    # Validate input
    if not content or not content.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "Content cannot be empty",
            "model_used": model,
            "tokens_used": 0,
        }

    prompt = get_prompt_with_truncation(info_to_extract, content)

    # Prepare the payload
    active_temperature = _get_summary_llm_temperature()
    logger.info(
        "Jina Scrape and Extract Info: Using summary LLM temperature=%s",
        active_temperature,
    )

    if "gpt" in model:
        payload = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": active_temperature,
        }
        # Add cost-saving parameters for GPT-5 models
        if "gpt-5" in model.lower() or "gpt5" in model.lower():
            payload["service_tier"] = "flex"
            payload["reasoning_effort"] = "minimal"
    else:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": active_temperature,
            # "top_p": 0.8,
            # "top_k": 20,
        }

    # Validate LLM endpoint configuration early for clearer errors
    if not SUMMARY_LLM_BASE_URL or not SUMMARY_LLM_BASE_URL.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_BASE_URL environment variable is not set",
            "model_used": model,
            "tokens_used": 0,
        }

    # Prepare headers (add Authorization if API key is available)
    headers = {"Content-Type": "application/json"}
    if SUMMARY_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {SUMMARY_LLM_API_KEY}"

    try:
        # Retry configuration
        connect_retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(connect_retry_delays, 1):
            try:
                # Make the API request using httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        SUMMARY_LLM_BASE_URL,
                        headers=headers,
                        json=payload,
                        timeout=httpx.Timeout(None, connect=30, read=300),
                    )
                    if response.text and len(response.text) >= 50:
                        tail_50 = response.text[-50:]
                        repeat_count = response.text.count(tail_50)
                        if repeat_count > 5:
                            logger.info("Repeat detected in extract_info_with_llm")
                            continue

                # Check if the request was successful
                if (
                    "Requested token count exceeds the model's maximum context length"
                    in response.text
                    or "longer than the model's context length" in response.text
                ):
                    prompt = get_prompt_with_truncation(
                        info_to_extract,
                        content,
                        truncate_last_num_chars=40960 * attempt,
                    )  # remove 40k * num_attempts chars from the end of the content
                    payload["messages"][0]["content"] = prompt
                    continue  # no need to raise error here, just try again

                response.raise_for_status()
                break  # Success, exit retry loop

            except httpx.ConnectTimeout as e:
                # connection timeout, retry
                if attempt < len(connect_retry_delays):
                    logger.info(
                        f"Jina Scrape and Extract Info: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except httpx.ConnectError as e:
                # connection error, retry
                if attempt < len(connect_retry_delays):
                    logger.info(
                        f"Jina Scrape and Extract Info: Connection error: {e}, {delay}s before next attempt"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except httpx.ReadTimeout as e:
                # read timeout, LLM API is too slow, no need to retry
                if attempt < len(connect_retry_delays):
                    logger.info(
                        f"Jina Scrape and Extract Info: LLM API attempt {attempt} read timeout"
                    )
                    continue
                else:
                    logger.error(
                        f"Jina Scrape and Extract Info: LLM API read timeout retry attempts exhausted, please check the request complexity, information to extract: {info_to_extract}, length of content: {len(content)}, url: {url}"
                    )
                    raise e

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Special case: GPT-5 service_tier parameter compatibility issue
                if (
                    "gpt-5" in model.lower() or "gpt5" in model.lower()
                ) and "service_tier" in payload:
                    logger.info(
                        "Extract Info: GPT-5 service_tier error, removing and retrying"
                    )
                    payload.pop("service_tier", None)
                    if attempt < len(connect_retry_delays):
                        await asyncio.sleep(delay)
                        continue

                # Retryable: 5xx (server errors) + specific 4xx (408, 409, 425, 429)
                should_retry = status_code >= 500 or status_code in [408, 409, 425, 429]

                if should_retry and attempt < len(connect_retry_delays):
                    logger.info(
                        f"Extract Info: HTTP {status_code} (retryable), retry in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                elif should_retry:
                    logger.error(f"Extract Info: HTTP {status_code} retry exhausted")
                    raise e
                else:
                    logger.error(f"Extract Info: HTTP {status_code} (non-retryable)")
                    raise httpx.HTTPStatusError(
                        f"response.text: {response.text}",
                        request=e.request,
                        response=e.response,
                    ) from e

            except httpx.RequestError as e:
                logger.error(
                    f"Jina Scrape and Extract Info: Unknown request exception: {e}"
                )
                raise e

    except Exception as e:
        error_msg = f"Jina Scrape and Extract Info: Unexpected error during LLM API call: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }

    # Parse the response
    try:
        response_data = response.json()

    except json.JSONDecodeError as e:
        error_msg = (
            f"Jina Scrape and Extract Info: Failed to parse LLM API response: {str(e)}"
        )
        logger.error(error_msg)
        logger.error(f"Raw response: {response.text}")
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }

    # Extract summary from response
    if "choices" in response_data and len(response_data["choices"]) > 0:
        try:
            summary = response_data["choices"][0]["message"]["content"]
        except Exception as e:
            error_msg = f"Jina Scrape and Extract Info: Failed to get summary from LLM API response: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "extracted_info": "",
                "error": error_msg,
                "model_used": model,
                "tokens_used": 0,
            }

        # Extract token usage if available
        tokens_used = 0
        if "usage" in response_data:
            tokens_used = response_data["usage"].get("total_tokens", 0)

        return {
            "success": True,
            "extracted_info": normalize_extracted_info(summary),
            "error": "",
            "model_used": model,
            "tokens_used": tokens_used,
        }
    elif "error" in response_data:
        error_msg = (
            f"Jina Scrape and Extract Info: LLM API error: {response_data['error']}"
        )
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }
    else:
        error_msg = f"Jina Scrape and Extract Info: No valid response from LLM API, response data: {response_data}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }


if __name__ == "__main__":
    # Example usage and testing

    # Run the MCP server
    mcp.run(transport="stdio")
