import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union
from functools import wraps
import time
from openai import OpenAI
import json
from datetime import datetime
import calendar
from bs4 import BeautifulSoup
import urllib.parse
import os
from urllib.parse import unquote
from pathlib import Path
from uuid import uuid4

from app.manus.gate.format_gate import format_check
from app.manus.tool.google_api_key import apikeys
import requests
from urllib.parse import urlparse


def retry(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    # Check if result is empty or contains error
                    if (result and not (isinstance(result, (list, dict)) and len(result) == 0) and
                            not (isinstance(result, dict) and "error" in result)):
                        return result
                except Exception as e:
                    if retries == max_retries - 1:
                        return {"error": f"Max retries reached. Last error: {str(e)}"}
                retries += 1
                time.sleep(delay)
            return {"error": "Max retries reached. No valid result found."}

        return wrapper

    return decorator


class SearchToolkit:
    r"""A class representing a toolkit for web search.

    This class provides methods for searching information on the web using
    search engines like Google, DuckDuckGo, Wikipedia and Wolfram Alpha, Brave.
    """
    def __init__(self, llm_config, cache_dir: Optional[str] = None):
        self.llm_config = llm_config
        self.cache_dir = "tmp/"
        if cache_dir:
            self.cache_dir = cache_dir

    _client: OpenAI = None
    _wiki_history_cache: Dict[str, str] = {}
    _jina_reader_cache: Dict[str, str] = {}
    _jina_reader_fail_cache: Dict[str, str] = {}
    @property
    def client(self) -> OpenAI:
        llm_config = {"api_key": self.llm_config['api_key'],
                      "base_url": self.llm_config['base_url']
                      }
        """Cached ChatOpenAI client instance."""
        if self._client is None:
            self._client = OpenAI(**llm_config)
        return self._client

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_int_env(name: str, default: int, min_value: int = 0, max_value: int = 1000) -> int:
        raw = os.getenv(name)
        try:
            value = int(raw) if raw is not None else default
        except Exception:
            value = default
        value = max(min_value, value)
        value = min(max_value, value)
        return value

    def _fetch_jina_reader_content(self, url: str) -> str:
        parsed = urlparse(url or "")
        if parsed.scheme not in {"http", "https"}:
            return ""
        cache_key = url.strip()
        if cache_key in self._jina_reader_cache:
            return self._jina_reader_cache[cache_key]
        if cache_key in self._jina_reader_fail_cache:
            return ""

        jina_api_key = (os.getenv("JINA_API_KEY") or "").strip()
        timeout_sec = self._safe_int_env("JINA_ENRICH_TIMEOUT", 8, min_value=2, max_value=30)
        max_chars = self._safe_int_env("JINA_ENRICH_MAX_CHARS", 1200, min_value=200, max_value=8000)
        reader_url = f"https://r.jina.ai/{cache_key}"
        headers = {"Accept": "text/plain"}
        if jina_api_key:
            headers["X-API-KEY"] = jina_api_key

        try:
            resp = requests.get(reader_url, headers=headers, timeout=timeout_sec)
            resp.raise_for_status()
            content = (resp.text or "").strip()
            if not content:
                self._jina_reader_fail_cache[cache_key] = "empty"
                print(f"jina_reader execute result = [{{'error': 'empty content', 'url': '{cache_key}'}}]")
                return ""
            if len(content) > max_chars:
                content = content[:max_chars] + " ...<truncated>"
            self._jina_reader_cache[cache_key] = content
            print(
                "jina_reader execute result = "
                f"[{{'url': '{cache_key}', 'chars': {len(content)}}}]"
            )
            return content
        except requests.RequestException as e:
            self._jina_reader_fail_cache[cache_key] = str(e)
            print(
                "jina_reader execute result = "
                f"[{{'error': 'jina reader failed. {e}', 'url': '{cache_key}'}}]"
            )
            return ""

    @retry(max_retries=1, delay=0)
    @format_check()
    def search_wiki_history_url(self, query: str) -> str:
        r"""Search for the relevant URL of an old/historical Wikipedia page.

        Args:
            query (str): The task/question to ask about the old/historical Wikipedia page search

        Returns:
            str: The URL of the expected Wikipedia page.
        """
        normalized_query = (query or "").strip()
        if normalized_query in self._wiki_history_cache:
            return self._wiki_history_cache[normalized_query]

        completion = self.client.chat.completions.create(
            extra_headers={'Content-Type': 'application/json',
                           'Authorization': 'Bearer %s' % self.llm_config['api_key']},
            model=self.llm_config['model'],
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text",
                         "text": """
                         **System Prompt:**
                        You are a highly specialized and honest Information Extraction Agent. Your sole purpose is to analyze a user's `task_prompt` and extract key information based on a predefined schema. You must strictly adhere to the output format.
                        ### **Schema & Rules**
                        You must extract information and format it into a single, valid JSON object.                        
                        **JSON Schema:**
                        * `subject`: (String, **Required**) The primary topic, theme, or entity of the prompt.
                        * `year`: (Integer, Optional) The 4-digit year.
                        * `month`: (Integer, Optional) The numeric month of the year (1-12). Default is 12.
                        * `day`: (Integer, Optional) The numeric day of the month (1-31). Default is 31.                 
                        **Processing Logic:**
                        1.  **Subject Identification**: Always identify and extract the `subject`. This field must be present in the output.
                        2.  **Date Inference**:
                            * **Contextual Date**: All date calculations must be based on the current date: **2025-08-12**.
                            * **Absolute Dates**: Extract explicit dates like "November 30, 2024".
                            * **Relative Dates**: Accurately resolve relative expressions. For example:
                                * "latest 2022" -> `2022-12-31`
                                * "as of August 2023" -> `2023-08-31`
                                * "before 2020" -> `2019-12-31`
                                * "earliest 2023" -> `2023-1-1`
                        3.  **Output Generation**:
                            * Construct a single JSON object containing the extracted fields.
                            * If a date component (`year`, `month`, or `day`) cannot be reasonably determined from the prompt, its corresponding key **MUST be omitted** from the final JSON object.
                            * Your final response must contain **ONLY the JSON object** and no other text, explanation, or commentary.
                        ### **Examples**                        
                        **Input (`query`)**: "Find research papers about LLMs published around November 30, 2024."
                        **Output format (MUST be strictly followed)**:
                        {
                          "subject": "Research papers about LLMs",
                          "year": 2024,
                          "month": 11,
                          "day": 30
                        }
                         """},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": normalized_query}]},
            ],
            temperature=self.llm_config['temperature'],
            max_tokens=300,
            stream=False,
        )

        full_response = completion.choices[0].message.content or ""
        print(f'full_response:{full_response}')
        dict4wiki = self.process_llm_output(full_response)
        if not isinstance(dict4wiki, dict) or "subject" not in dict4wiki:
            return "The corresponding URL is: None, the title of the wikipedia page is: None"

        if "year" not in dict4wiki:
            dict4wiki["year"] = datetime.now().year
        if "month" not in dict4wiki:
            dict4wiki["month"] = 12
        if "day" not in dict4wiki:
            dict4wiki["day"] = 31

        url = self.get_wikipedia_url_for_date(dict4wiki["subject"], dict4wiki["year"], dict4wiki["month"], dict4wiki["day"])

        if url is None:
            title = self.get_wikipedia_article_title(dict4wiki["subject"])
            url = self.get_wikipedia_url_for_date(title, dict4wiki["year"], dict4wiki["month"], dict4wiki["day"])

        else:
            title = dict4wiki["subject"]

        result = f"The corresponding URL is: {url}, the title of the wikipedia page is: {title}"
        print(result)
        self._wiki_history_cache[normalized_query] = result
        return result


    @format_check()
    def process_llm_output(self, input_str):
        """
        This function takes a string in one of two formats and converts it into a Python dictionary.

        It handles two cases:
        1. A string containing a JSON object enclosed in ```json ... ```.
        2. A string that is a plain JSON object.

        Args:
          input_str: The string to be converted.

        Returns:
          A dictionary representation of the input string, or None if parsing fails.
        """
        # Clean the string by removing leading/trailing whitespace.
        cleaned_str = input_str.strip()

        # Case 1: If the string is enclosed in markdown-style code fences, remove them.
        if cleaned_str.startswith("```json"):
            # Remove the starting '```json\n' and the ending '```'
            cleaned_str = cleaned_str.replace("```json\n", "").replace("\n```", "").strip()

        # Now, the string should be a valid JSON object.
        # We use a try-except block to handle potential errors during parsing.
        try:
            # Use json.loads() to parse the cleaned string into a dictionary.
            # 'loads' stands for 'load string'.
            return json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            # If the string is not valid JSON, print an error and return None.
            print(f"Error decoding JSON: {e}")
            return None


    @format_check()
    def get_wikipedia_url_at_timestamp(self, page_title: str, target_datetime: datetime, proxies: dict = None):
        """
        Core function: Gets the URL for the last revision of a Wikipedia page at or before a precise timestamp.

        Args:
            page_title (str): The title of the Wikipedia page.
            target_datetime (datetime): The target date and time.
            proxies (dict, optional): A dictionary for proxy configuration. Defaults to None.

        Returns:
            str: A permalink URL to the specific revision, or None if not found.
        """
        API_URL = "https://en.wikipedia.org/w/api.php"
        headers = {'User-Agent': 'MyURLFetcher/1.0 (https://example.com/mybot)'}
        timestamp_str = target_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

        params = {
            "action": "query", "prop": "revisions", "titles": page_title,
            "rvlimit": "1", "rvdir": "older", "rvstart": timestamp_str, "format": "json"
        }

        try:
            print(f"Searching for a version of '{page_title}' on or before {target_datetime.strftime('%Y-%m-%d')}...")
            response = requests.get(API_URL, params=params, headers=headers, proxies=proxies, timeout=15)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page_id = next(iter(pages))

            if page_id == "-1" or "revisions" not in pages[page_id]:
                print(f"Info: No revisions found for '{page_title}' on or before the specified date.")
                return None

            revision_id = pages[page_id]["revisions"][0]["revid"]
            encoded_title = requests.utils.quote(page_title)
            permalink = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&oldid={revision_id}"

            print(f"Success! Found URL for revision ID: {revision_id}")
            return permalink

        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while processing the response: {e}")
            return None

    @format_check()
    def get_wikipedia_url_for_date(self, page_title: str, year: int, month: int = 12, day: int = 31):
        """
        Convenience function: Gets the URL by year, with optional month and day.
        It safely handles invalid days (e.g., day=30 for February).

        Args:
            page_title (str): The title of the Wikipedia page.
            year (int): The year (e.g., 2023).
            month (int, optional): The month (1-12).
            day (int, optional): The day (1-31).
            proxies (dict, optional): Proxy configuration.

        Returns:
            str: A permalink URL to the specific revision, or None on failure.
        """

        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY")
            or os.getenv("http_proxy")
            or ""
        )
        proxies = {'http': proxy_url, 'https': proxy_url} if proxy_url else None

        if not 1 <= month <= 12:
            print(f"Error: Invalid month '{month}'. Please use a number between 1 and 12.")
            return None

        # Safely determine the actual day to use
        last_day_of_month = calendar.monthrange(year, month)[1]
        actual_day = day
        if day > last_day_of_month:
            actual_day = last_day_of_month
            print(
                f"Info: Day '{day}' is invalid for the given month. Using the last day of the month instead: '{actual_day}'.")

        # Create a datetime object for the last moment of that day
        target_dt = datetime(year, month, actual_day, 23, 59, 59)

        print(f"Query configuration: Year={year}, Month={month}, Day={actual_day} (End of day)")

        return self.get_wikipedia_url_at_timestamp(page_title, target_dt, proxies)

    @format_check()
    def get_wikipedia_article_title(self, keyword: str) -> str:
        """
        Searches for a keyword on Wikipedia and returns the exact title of the
        resulting article page without using the Wikipedia API.

        Args:
            keyword: The search term.

        Returns:
            The exact article title as listed on the Wikipedia page,
            or an informational message if not found.
        """
        if not keyword:
            return "Please provide a keyword to search."

        search_url = "https://en.wikipedia.org/w/index.php"
        params = {
            "search": keyword
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            # The requests library automatically handles redirects
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            soup = BeautifulSoup(response.text, 'html.parser')

            # The main title of a Wikipedia article is in an <h1> tag with the id 'firstHeading'
            heading_tag = soup.find('h1', id='firstHeading')

            if heading_tag:
                title = heading_tag.get_text()
                # If the page is a search results page, the heading will be "Search results"
                if title.lower() == "search results":
                    # Try to find the title of the first search result instead
                    first_result = soup.find('div', class_='mw-search-result-heading')
                    if first_result and first_result.a:
                        return first_result.a.get_text()
                    else:
                        return f"No direct article found for '{keyword}'."
                else:
                    return title
            else:
                return f"Could not determine the article title for '{keyword}'. The page structure may have changed."

        except requests.exceptions.RequestException as e:
            return f"An error occurred during the request: {e}"

    @format_check()
    def download_wiki_main_image(self, wiki_url):
        """
        Downloads the main/primary image from the given URL of a Wikipedia page

        Args:
        - wiki_url (str): The full URL of the Wikipedia page.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            }
            response = requests.get(wiki_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            full_image_link_element = soup.select_one('.infobox a.image > img')
            image_url = None

            if full_image_link_element:
                parent_a_tag = full_image_link_element.find_parent('a', class_='image')
                if parent_a_tag and parent_a_tag.has_attr('href'):
                    full_image_href = parent_a_tag['href']
                    full_image_page_url = 'https://' + urlparse(wiki_url).netloc + full_image_href
                    image_page_response = requests.get(full_image_page_url, headers=headers)
                    image_page_response.raise_for_status()
                    image_page_soup = BeautifulSoup(image_page_response.text, 'html.parser')
                    full_image_element = image_page_soup.select_one('#file a.internal')
                    if full_image_element and full_image_element.has_attr('href'):
                        image_url = 'https:' + full_image_element['href']

            if not image_url:
                image_tag = soup.select_one('.infobox a:has(img) img')
                if image_tag and image_tag.has_attr('src'):
                    image_url = image_tag['src']
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                else:
                    print(f"Error: Could not find the main image on the page {wiki_url}.")
                    return f"Error: Could not find the main image on the page {wiki_url}"

            print(f"Found image URL: {image_url}")

            image_response = requests.get(image_url, headers=headers, stream=True)
            image_response.raise_for_status()

            # Extract filename from URL
            filename = os.path.basename(urlparse(image_url).path)

            with open(filename, 'wb') as f:
                for chunk in image_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Image successfully downloaded and saved as: {filename} in the current directory.")
            return f"Image successfully downloaded and saved as: {filename} in the current directory."

        except requests.exceptions.RequestException as e:
            print(f"A network request failed: {e}")
            return f"A network request failed: {e}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"


    @format_check()
    def download_wiki_commons_image(self, url):
        """
        Downloads the image for a  Wikimedia Commons file page with given URL.

        Args:
            url (str): The URL of the Wikimedia Commons page (e.g., 'https://commons.wikimedia.org/wiki/File:...'')
        """
        try:
            # Send an HTTP GET request to the provided URL to get the page content
            page_response = requests.get(url)
            # Raise an exception if the request was unsuccessful (e.g., 404 Not Found, 500 Server Error)
            page_response.raise_for_status()

            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(page_response.text, 'html.parser')

            # Find the specific 'div' element that contains the link to the full-resolution image.
            # On Wikimedia Commons pages, this link is typically inside a div with the class "fullImageLink".
            full_image_link_container = soup.find('div', class_='fullImageLink')
            if not full_image_link_container:
                print("Could not find the link container for the full image.")
                return "Could not find the link container for the full image."

            # Within that 'div', find the anchor 'a' tag which holds the URL in its 'href' attribute.
            image_anchor_tag = full_image_link_container.find('a')
            if not image_anchor_tag or 'href' not in image_anchor_tag.attrs:
                print("Could not find the anchor tag with the image URL.")
                return "Could not find the anchor tag with the image URL."

            # Extract the URL of the full-resolution image from the 'href' attribute.
            image_url = image_anchor_tag['href']

            # Now, send another GET request, this time to the direct image URL.
            # 'stream=True' is used to download large files efficiently without loading the entire content into memory at once.
            image_response = requests.get(image_url, stream=True)
            image_response.raise_for_status()

            # Extract the filename from the image URL (e.g., "Brazilian_Army_-_CMA.png")
            file_name = os.path.basename(image_url)

            # Open a new file in binary write mode ('wb'). The file will have the name we just extracted.
            with open(file_name, 'wb') as file:
                # Iterate over the image data in chunks (8192 bytes at a time) and write each chunk to the file.
                for chunk in image_response.iter_content(chunk_size=8192):
                    file.write(chunk)

            # Print a success message to the console.
            print(f"Image successfully downloaded as: {file_name}")
            return f"Image successfully downloaded as: {file_name}"

        except requests.exceptions.RequestException as e:
            # Handle any network-related errors that might occur during the requests.
            print(f"An error occurred during download: {e}")
            return f"An error occurred during download: {e}"

    @format_check()
    def get_wikipedia_revision_record(self, topic: str) -> str:
        """
        Get the URL to view the revision records of a Wikipedia page for a given topic.

        Args:
          topic: The topic of the Wikipedia page.

        Returns:
          The URL of the Wikipedia page's revision history, limited to 500 entries.
        """
        base_url = "https://en.wikipedia.org/w/index.php"
        topic = self.get_wikipedia_article_title(topic)  # Match to the most likely entry on Wikipedia
        encoded_topic = urllib.parse.quote(topic, safe='')
        history_url = f"{base_url}?title={encoded_topic}&action=history&offset=&limit=500"
        print("history url:", history_url)
        return history_url

    @retry()
    @format_check()
    def search_wiki(self, entity: str) -> str:
        r"""Search the entity in WikiPedia and return the summary of the
            required page, containing factual information about
            the given entity.

        Args:
            entity (str): The entity to be searched.

        Returns:
            str: The search result. If the page corresponding to the entity
                exists, return the summary of this entity in a string.
        """
        import wikipedia
        print(f"start search_wiki")
        result: str

        # Match to the most likely entry on Wikipedia
        entity = self.get_wikipedia_article_title(entity)
        print('matched entity:', entity)

        try:
            result = wikipedia.summary(entity, sentences=5, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            result = wikipedia.summary(
                e.options[0], sentences=5, auto_suggest=False
            )
        except wikipedia.exceptions.PageError:
            result = (
                "There is no page in Wikipedia corresponding to entity "
                f"{entity}, please specify another word to describe the"
                " entity to be searched."
            )
        except wikipedia.exceptions.WikipediaException as e:
            result = f"An exception occurred during the search: {e}"
        print(f"result of search_wiki: {result}")
        return result

    @retry()
    def search_linkup(
            self,
            query: str,
            depth: Literal["standard", "deep"] = "standard",
            output_type: Literal[
                "searchResults", "sourcedAnswer", "structured"
            ] = "searchResults",
            structured_output_schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        r"""Search for a query in the Linkup API and return results in various
        formats.

        Args:
            query (str): The search query.
            depth (Literal["standard", "deep"]): The depth of the search.
                "standard" for a straightforward search, "deep" for a more
                comprehensive search.
            output_type (Literal["searchResults", "sourcedAnswer",
                "structured"]): The type of output:
                - "searchResults" for raw search results,
                - "sourcedAnswer" for an answer with supporting sources,
                - "structured" for output based on a provided schema.
            structured_output_schema (Optional[str]): If `output_type` is
                "structured", specify the schema of the output. Must be a
                string representing a valid object JSON schema.

        Returns:
            Dict[str, Any]: A dictionary representing the search result. The
                structure depends on the `output_type`. If an error occurs,
                returns an error message.
        """
        try:
            from linkup import LinkupClient

            # Initialize the Linkup client with the API key
            LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
            client = LinkupClient(api_key=LINKUP_API_KEY)

            # Perform the search using the specified output_type
            response = client.search(
                query=query,
                depth=depth,
                output_type=output_type,
                structured_output_schema=structured_output_schema,
            )

            if output_type == "searchResults":
                results = [
                    item.__dict__
                    for item in response.__dict__.get('results', [])
                ]
                return {"results": results}

            elif output_type == "sourcedAnswer":
                answer = response.__dict__.get('answer', '')
                sources = [
                    item.__dict__
                    for item in response.__dict__.get('sources', [])
                ]
                return {"answer": answer, "sources": sources}

            elif output_type == "structured" and structured_output_schema:
                return response.__dict__

            else:
                return {"error": f"Invalid output_type: {output_type}"}

        except Exception as e:
            return {"error": f"An unexpected error occurred: {e!s}"}

    @retry()
    def search_duckduckgo(
            self, query: str, source: str = "text", max_results: int = 5
    ) -> List[Dict[str, Any]]:
        r"""Use DuckDuckGo search engine to search information for
        the given query.

        This function queries the DuckDuckGo API for related topics to
        the given search term. The results are formatted into a list of
        dictionaries, each representing a search result.

        Args:
            query (str): The query to be searched.
            source (str): The type of information to query (e.g., "text",
                "images", "videos"). Defaults to "text".
            max_results (int): Max number of results, defaults to `5`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary
                represents a search result.
        """
        from duckduckgo_search import DDGS
        from requests.exceptions import RequestException

        ddgs = DDGS()
        responses: List[Dict[str, Any]] = []

        if source == "text":
            try:
                results = ddgs.text(keywords=query, max_results=max_results)
            except RequestException as e:
                # Handle specific exceptions or general request exceptions
                responses.append({"error": f"duckduckgo search failed.{e}"})

            # Iterate over results found
            for i, result in enumerate(results, start=1):
                # Creating a response object with a similar structure
                response = {
                    "result_id": i,
                    "title": result["title"],
                    "description": result["body"],
                    "url": result["href"],
                }
                responses.append(response)

        elif source == "images":
            try:
                results = ddgs.images(keywords=query, max_results=max_results)
            except RequestException as e:
                # Handle specific exceptions or general request exceptions
                responses.append({"error": f"duckduckgo search failed.{e}"})

            # Iterate over results found
            for i, result in enumerate(results, start=1):
                # Creating a response object with a similar structure
                response = {
                    "result_id": i,
                    "title": result["title"],
                    "image": result["image"],
                    "url": result["url"],
                    "source": result["source"],
                }
                responses.append(response)

        elif source == "videos":
            try:
                results = ddgs.videos(keywords=query, max_results=max_results)
            except RequestException as e:
                # Handle specific exceptions or general request exceptions
                responses.append({"error": f"duckduckgo search failed.{e}"})

            # Iterate over results found
            for i, result in enumerate(results, start=1):
                # Creating a response object with a similar structure
                response = {
                    "result_id": i,
                    "title": result["title"],
                    "description": result["description"],
                    "embed_url": result["embed_url"],
                    "publisher": result["publisher"],
                    "duration": result["duration"],
                    "published": result["published"],
                }
                responses.append(response)

        # If no answer found, return an empty list
        return responses

    @retry()
    def search_brave(
            self,
            q: str,
            country: str = "US",
            search_lang: str = "en",
            ui_lang: str = "en-US",
            count: int = 20,
            offset: int = 0,
            safesearch: str = "moderate",
            freshness: Optional[str] = None,
            text_decorations: bool = True,
            spellcheck: bool = True,
            result_filter: Optional[str] = None,
            goggles_id: Optional[str] = None,
            units: Optional[str] = None,
            extra_snippets: Optional[bool] = None,
            summary: Optional[bool] = None,
    ) -> Dict[str, Any]:
        r"""This function queries the Brave search engine API and returns a
        dictionary, representing a search result.
        See https://api.search.brave.com/app/documentation/web-search/query
        for more details.

        Args:
            q (str): The user's search query term. Query cannot be empty.
                Maximum of 400 characters and 50 words in the query.
            country (str): The search query country where results come from.
                The country string is limited to 2 character country codes of
                supported countries. For a list of supported values, see
                Country Codes. (default: :obj:`US `)
            search_lang (str): The search language preference. The 2 or more
                character language code for which search results are provided.
                For a list of possible values, see Language Codes.
            ui_lang (str): User interface language preferred in response.
                Usually of the format '<language_code>-<country_code>'. For
                more, see RFC 9110. For a list of supported values, see UI
                Language Codes.
            count (int): The number of search results returned in response.
                The maximum is 20. The actual number delivered may be less than
                requested. Combine this parameter with offset to paginate
                search results.
            offset (int): The zero based offset that indicates number of search
                results per page (count) to skip before returning the result.
                The maximum is 9. The actual number delivered may be less than
                requested based on the query. In order to paginate results use
                this parameter together with count. For example, if your user
                interface displays 20 search results per page, set count to 20
                and offset to 0 to show the first page of results. To get
                subsequent pages, increment offset by 1 (e.g. 0, 1, 2). The
                results may overlap across multiple pages.
            safesearch (str): Filters search results for adult content.
                The following values are supported:
                - 'off': No filtering is done.
                - 'moderate': Filters explicit content, like images and videos,
                    but allows adult domains in the search results.
                - 'strict': Drops all adult content from search results.
            freshness (Optional[str]): Filters search results by when they were
                discovered:
                - 'pd': Discovered within the last 24 hours.
                - 'pw': Discovered within the last 7 Days.
                - 'pm': Discovered within the last 31 Days.
                - 'py': Discovered within the last 365 Days.
                - 'YYYY-MM-DDtoYYYY-MM-DD': Timeframe is also supported by
                    specifying the date range e.g. '2022-04-01to2022-07-30'.
            text_decorations (bool): Whether display strings (e.g. result
                snippets) should include decoration markers (e.g. highlighting
                characters).
            spellcheck (bool): Whether to spellcheck provided query. If the
                spellchecker is enabled, the modified query is always used for
                search. The modified query can be found in altered key from the
                query response model.
            result_filter (Optional[str]): A comma delimited string of result
                types to include in the search response. Not specifying this
                parameter will return back all result types in search response
                where data is available and a plan with the corresponding
                option is subscribed. The response always includes query and
                type to identify any query modifications and response type
                respectively. Available result filter values are:
                - 'discussions'
                - 'faq'
                - 'infobox'
                - 'news'
                - 'query'
                - 'summarizer'
                - 'videos'
                - 'web'
                - 'locations'
            goggles_id (Optional[str]): Goggles act as a custom re-ranking on
                top of Brave's search index. For more details, refer to the
                Goggles repository.
            units (Optional[str]): The measurement units. If not provided,
                units are derived from search country. Possible values are:
                - 'metric': The standardized measurement system
                - 'imperial': The British Imperial system of units.
            extra_snippets (Optional[bool]): A snippet is an excerpt from a
                page you get as a result of the query, and extra_snippets
                allow you to get up to 5 additional, alternative excerpts. Only
                available under Free AI, Base AI, Pro AI, Base Data, Pro Data
                and Custom plans.
            summary (Optional[bool]): This parameter enables summary key
                generation in web search results. This is required for
                summarizer to be enabled.

        Returns:
            Dict[str, Any]: A dictionary representing a search result.
        """

        import requests

        BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Content-Type": "application/json",
            "X-BCP-APIV": "1.0",
            "X-Subscription-Token": BRAVE_API_KEY,
        }

        ParamsType: TypeAlias = Dict[
            str,
            Union[str, int, float, List[Union[str, int, float]], None],
        ]

        params: ParamsType = {
            "q": q,
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "count": count,
            "offset": offset,
            "safesearch": safesearch,
            "freshness": freshness,
            "text_decorations": text_decorations,
            "spellcheck": spellcheck,
            "result_filter": result_filter,
            "goggles_id": goggles_id,
            "units": units,
            "extra_snippets": extra_snippets,
            "summary": summary,
        }

        response = requests.get(url, headers=headers, params=params)
        data = response.json()["web"]
        return data

    @retry()
    def search_google(
            self, query: str
    ) -> List[Dict[str, Any]]:
        r"""Use Google search engine to search information for the given query.

        Args:
            query (str): The query to be searched.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary
            represents a website.
                Each dictionary contains the following keys:
                - 'result_id': A number in order.
                - 'title': The title of the website.
                - 'description': A brief description of the website.
                - 'long_description': More detail of the website.
                - 'url': The URL of the website.

                Example:
                {
                    'result_id': 1,
                    'title': 'OpenAI',
                    'description': 'An organization focused on ensuring that
                    artificial general intelligence benefits all of humanity.',
                    'long_description': 'OpenAI is a non-profit artificial
                    intelligence research company. Our goal is to advance
                    digital intelligence in the way that is most likely to
                    benefit humanity as a whole',
                    'url': 'https://www.openai.com'
                }
            title, description, url of a website.
        """
        import requests

        responses = []

        def _is_valid_value(value: Optional[str]) -> bool:
            if not value:
                return False
            stripped = value.strip()
            if not stripped:
                return False
            if "{" in stripped or "}" in stripped:
                return False
            if "如：" in stripped:
                return False
            return True

        # https://developers.google.com/custom-search/v1/overview
        google_api_key = os.getenv("GOOGLE_API_KEY")
        # https://cse.google.com/cse/all
        search_engine_id = os.getenv("SEARCH_ENGINE_ID")
        use_google_cse = _is_valid_value(google_api_key) and _is_valid_value(search_engine_id)

        if use_google_cse:
            google_api_key, search_engine_id = apikeys.get(google_api_key, search_engine_id)

            # Using the first page
            start_page_idx = 1
            # Different language may get different result
            search_language = "en"
            # How many pages to return
            num_result_pages = 3
            # Constructing the URL
            # Doc: https://developers.google.com/custom-search/v1/using_rest
            url = (
                f"https://www.googleapis.com/customsearch/v1?"
                f"key={google_api_key}&cx={search_engine_id}&q={query}&start="
                f"{start_page_idx}&lr={search_language}&num={num_result_pages}"
                f"&gl=us"
            )

            try:
                result = requests.get(url, timeout=30)
                data = result.json()

                if "items" in data:
                    search_items = data.get("items")

                    for i, search_item in enumerate(search_items, start=1):
                        if "pagemap" not in search_item:
                            continue
                        if "metatags" not in search_item["pagemap"]:
                            continue
                        if "og:description" in search_item["pagemap"]["metatags"][0]:
                            long_description = search_item["pagemap"]["metatags"][0]["og:description"]
                        else:
                            long_description = "N/A"
                        title = search_item.get("title")
                        snippet = search_item.get("snippet")
                        link = search_item.get("link")
                        response = {
                            "result_id": i,
                            "title": title,
                            "description": snippet,
                            "long_description": long_description,
                            "url": link,
                        }
                        responses.append(response)
                else:
                    responses.append({"error": "google search failed. 'items' not in data"})
                    apikeys.next()

            except requests.RequestException as e:
                responses.append({"error": f"google search failed. {e}"})
                apikeys.next()
        else:
            serper_api_key = os.getenv("SERPER_API_KEY")
            if not _is_valid_value(serper_api_key):
                responses.append({"error": "google search unavailable. Set GOOGLE_API_KEY+SEARCH_ENGINE_ID or SERPER_API_KEY"})
            else:
                try:
                    headers = {
                        "X-API-KEY": serper_api_key.strip(),
                        "Content-Type": "application/json",
                    }
                    payload = {"q": query, "gl": "us", "hl": "en", "num": 10}
                    result = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=30)
                    result.raise_for_status()
                    data = result.json()
                    organic_results = data.get("organic", [])
                    for i, item in enumerate(organic_results, start=1):
                        responses.append({
                            "result_id": i,
                            "title": item.get("title", ""),
                            "description": item.get("snippet", ""),
                            "long_description": item.get("snippet", ""),
                            "url": item.get("link", ""),
                        })

                    # Use Jina Reader to fetch page-level details for top Serper results.
                    jina_api_key = os.getenv("JINA_API_KEY")
                    jina_enabled = _is_valid_value(jina_api_key) and self._env_flag("JINA_ENRICH_ENABLED", True)
                    top_k = self._safe_int_env("JINA_ENRICH_TOP_K", 1, min_value=0, max_value=10)
                    enrich_limit = min(top_k, len(responses))
                    jina_success = 0
                    if jina_enabled and enrich_limit > 0:
                        for idx in range(enrich_limit):
                            target_url = responses[idx].get("url", "")
                            detail = self._fetch_jina_reader_content(target_url)
                            if detail:
                                responses[idx]["long_description"] = detail
                                responses[idx]["content"] = detail
                                responses[idx]["detail_source"] = "jina_reader"
                                jina_success += 1
                        print(
                            "jina_reader enrich summary = "
                            f"{{'query': '{str(query)[:120]}', 'requested': {enrich_limit}, 'success': {jina_success}}}"
                        )

                    if not responses:
                        responses.append({"error": "serper search returned no organic results"})
                except requests.RequestException as e:
                    responses.append({"error": f"serper search failed. {e}"})

        # If no answer found, return an empty list
        print(f'google_search execute result = {responses}')
        return responses

    @retry()
    def query_wolfram_alpha(
            self, query: str, is_detailed: bool = False
    ) -> Union[str, Dict[str, Any]]:
        r"""Queries Wolfram|Alpha and returns the result. Wolfram|Alpha is an
        answer engine developed by Wolfram Research. It is offered as an online
        service that answers factual queries by computing answers from
        externally sourced data.

        Args:
            query (str): The query to send to Wolfram Alpha.
            is_detailed (bool): Whether to include additional details
                including step by step information in the result.
                (default: :obj:`False`)

        Returns:
            Union[str, Dict[str, Any]]: The result from Wolfram Alpha.
                Returns a string if `is_detailed` is False, otherwise returns
                a dictionary with detailed information.
        """
        import wolframalpha

        WOLFRAMALPHA_APP_ID = os.environ.get("WOLFRAMALPHA_APP_ID")
        if not WOLFRAMALPHA_APP_ID:
            raise ValueError(
                "`WOLFRAMALPHA_APP_ID` not found in environment "
                "variables. Get `WOLFRAMALPHA_APP_ID` here: `https://products.wolframalpha.com/api/`."
            )

        try:
            client = wolframalpha.Client(WOLFRAMALPHA_APP_ID)
            res = client.query(query)

        except Exception as e:
            return f"Wolfram Alpha wasn't able to answer it. Error: {e}"

        pased_result = self._parse_wolfram_result(res)

        if is_detailed:
            step_info = self._get_wolframalpha_step_by_step_solution(
                WOLFRAMALPHA_APP_ID, query
            )
            pased_result["steps"] = step_info
            return pased_result

        return pased_result["final_answer"]

    def _parse_wolfram_result(self, result) -> Dict[str, Any]:
        r"""Parses a Wolfram Alpha API result into a structured dictionary
        format.

        Args:
            result: The API result returned from a Wolfram Alpha
                query, structured with multiple pods, each containing specific
                information related to the query.

        Returns:
            dict: A structured dictionary with the original query and the
                final answer.
        """

        # Extract the original query
        query = result.get("@inputstring", "")

        # Initialize a dictionary to hold structured output
        output = {"query": query, "pod_info": [], "final_answer": None}

        # Loop through each pod to extract the details
        for pod in result.get("pod", []):
            # Handle the case where subpod might be a list
            subpod_data = pod.get("subpod", {})
            if isinstance(subpod_data, list):
                # If it's a list, get the first item for 'plaintext' and 'img'
                description, image_url = next(
                    (
                        (data["plaintext"], data["img"])
                        for data in subpod_data
                        if "plaintext" in data and "img" in data
                    ),
                    ("", ""),
                )
            else:
                # Otherwise, handle it as a dictionary
                description = subpod_data.get("plaintext", "")
                image_url = subpod_data.get("img", {}).get("@src", "")

            pod_info = {
                "title": pod.get("@title", ""),
                "description": description,
                "image_url": image_url,
            }

            # For Results pod, collect all plaintext values from subpods
            if pod.get("@title") == "Results":
                results_text = []
                if isinstance(subpod_data, list):
                    for subpod in subpod_data:
                        if subpod.get("plaintext"):
                            results_text.append(subpod["plaintext"])
                else:
                    if description:
                        results_text.append(description)
                pod_info["description"] = "\n".join(results_text)

            # Add to steps list
            output["pod_info"].append(pod_info)

            # Get final answer
            if pod.get("@primary", False):
                output["final_answer"] = description

        return output

    def _get_wolframalpha_step_by_step_solution(
            self, app_id: str, query: str
    ) -> dict:
        r"""Retrieve a step-by-step solution from the Wolfram Alpha API for a
        given query.

        Args:
            app_id (str): Your Wolfram Alpha API application ID.
            query (str): The mathematical or computational query to solve.

        Returns:
            dict: The step-by-step solution response text from the Wolfram
                Alpha API.
        """
        # Define the base URL
        url = "https://api.wolframalpha.com/v2/query"

        # Set up the query parameters
        params = {
            "appid": app_id,
            "input": query,
            "podstate": ["Result__Step-by-step solution", "Show all steps"],
            "format": "plaintext",
        }

        # Send the request
        response = requests.get(url, params=params)
        root = ET.fromstring(response.text)

        # Extracting step-by-step steps, including 'SBSStep' and 'SBSHintStep'
        steps = []
        # Find all subpods within the 'Results' pod
        for subpod in root.findall(".//pod[@title='Results']//subpod"):
            # Check if the subpod has the desired stepbystepcontenttype
            content_type = subpod.find("stepbystepcontenttype")
            if content_type is not None and content_type.text in [
                "SBSStep",
                "SBSHintStep",
            ]:
                plaintext = subpod.find("plaintext")
                if plaintext is not None and plaintext.text:
                    step_text = plaintext.text.strip()
                    cleaned_step = step_text.replace(
                        "Hint: |", ""
                    ).strip()  # Remove 'Hint: |' if present
                    steps.append(cleaned_step)

        # Structuring the steps into a dictionary
        structured_steps = {}
        for i, step in enumerate(steps, start=1):
            structured_steps[f"step{i}"] = step

        return structured_steps

    @retry()
    def tavily_search(
            self, query: str, num_results: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        r"""Use Tavily Search API to search information for the given query.

        Args:
            query (str): The query to be searched.
            num_results (int): The number of search results to retrieve
                (default is `5`).
            **kwargs: Additional optional parameters supported by Tavily's API:
                - search_depth (str): "basic" or "advanced" search depth.
                - topic (str): The search category, e.g., "general" or "news."
                - days (int): Time frame in days for news-related searches.
                - max_results (int): Max number of results to return
                  (overrides `num_results`).
                See https://docs.tavily.com/docs/python-sdk/tavily-search/
                api-reference for details.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing search
                results. Each dictionary contains:
                - 'result_id' (int): The result's index.
                - 'title' (str): The title of the result.
                - 'description' (str): A brief description of the result.
                - 'long_description' (str): Detailed information, if available.
                - 'url' (str): The URL of the result.
                - 'content' (str): Relevant content from the search result.
                - 'images' (list): A list of related images (if
                  `include_images` is True).
                - 'published_date' (str): Publication date for news topics
                  (if available).
        """
        from tavily import TavilyClient  # type: ignore[import-untyped]

        Tavily_API_KEY = os.getenv("TAVILY_API_KEY")
        if not Tavily_API_KEY:
            raise ValueError(
                "`TAVILY_API_KEY` not found in environment variables. "
                "Get `TAVILY_API_KEY` here: `https://www.tavily.com/api/`."
            )

        client = TavilyClient(Tavily_API_KEY)

        try:
            results = client.search(query, max_results=num_results, **kwargs)
            return results
        except Exception as e:
            return [{"error": f"An unexpected error occurred: {e!s}"}]


if __name__ == '__main__':
    toolKit = SearchToolkit()
    result = toolKit.tavily_search("哪吒", 5)
    # result = toolKit.search_google("哪吒", 5)
    # result = toolKit.search_duckduckgo("哪吒")
    print(result)
