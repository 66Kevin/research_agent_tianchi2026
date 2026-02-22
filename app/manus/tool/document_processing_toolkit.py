# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import json
import mimetypes
import os
import subprocess
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import nest_asyncio
import requests
import xmltodict
# -*- coding: utf-8 -*-
from docx2markdown._docx_to_markdown import docx_to_markdown
from openai import OpenAI
from retry import retry

from app.manus.gate.format_gate import format_check
from app.manus.tool.excel_toolkit import extract_excel_content
from app.manus.tool.pptx_toolkit import extract_pptx_content
from config.config import get_plan_model_config, get_model_config

nest_asyncio.apply()


class DocumentProcessingToolkit:
    r"""A class representing a toolkit for processing document and return the content of the document.

    This class provides method for processing docx, pdf, pptx, etc. It cannot process excel files.
    """

    def __init__(self, llm_config, cache_dir: Optional[str] = None):
        self.llm_config = llm_config
        self.cache_dir = "tmp/"
        if cache_dir:
            self.cache_dir = cache_dir

    _client: OpenAI = None
    _document_content_cache: Dict[str, str] = {}
    _qa_cache: Dict[str, str] = {}

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...<truncated>"

    @staticmethod
    def _normalize_document_path(document_path: str) -> str:
        parsed_url = urlparse(document_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        if is_url:
            return document_path
        return os.path.abspath(document_path)

    def _prepare_document_context(self, content: str) -> str:
        max_chars_raw = os.getenv("DOC_QA_MAX_CHARS", "40000")
        try:
            max_chars = max(2000, int(max_chars_raw))
        except ValueError:
            max_chars = 40000

        if len(content) <= max_chars:
            return content

        # Keep both head and tail so metadata/introduction and conclusions are preserved.
        head_chars = int(max_chars * 0.75)
        tail_chars = max_chars - head_chars
        return (
            content[:head_chars]
            + "\n\n[...document content truncated for efficiency...]\n\n"
            + content[-tail_chars:]
        )

    def _set_cache(self, cache: Dict[str, str], key: str, value: str, max_items: int) -> None:
        if len(cache) >= max_items and key not in cache:
            oldest_key = next(iter(cache.keys()))
            cache.pop(oldest_key, None)
        cache[key] = value

    @property
    def client(self) -> OpenAI:
        llm_config = {"api_key": self.llm_config['api_key'],
                      "base_url": self.llm_config['base_url']
                      }
        """Cached ChatOpenAI client instance."""
        if self._client is None:
            self._client = OpenAI(**llm_config)
        return self._client

    @format_check()
    def ask_question_about_document(self, document_path: str, task_prompt: str) -> str:
        r"""Extract content from a document and use it as context to answer a task prompt using LLM.

        Args:
            document_path (str): Path to the document to extract content from
            task_prompt (str): The task/question to ask about the document content

        Returns:
            str: The LLM's response to the task prompt based on the document content
        """
        normalized_path = self._normalize_document_path(document_path)
        normalized_prompt = (task_prompt or "").strip()
        qa_cache_key = f"{normalized_path}\n{normalized_prompt}"
        cached_qa = self._qa_cache.get(qa_cache_key)
        if cached_qa:
            return cached_qa

        cached_content = self._document_content_cache.get(normalized_path)
        if cached_content is None:
            raw_content = self.extract_document_content(document_path)
            if isinstance(raw_content, str):
                cached_content = raw_content
            else:
                cached_content = json.dumps(raw_content, ensure_ascii=False)
            self._set_cache(self._document_content_cache, normalized_path, cached_content, max_items=24)

        if not cached_content.strip():
            return "Result: Document content is empty."

        context = self._prepare_document_context(cached_content)
        completion = self.client.chat.completions.create(
            extra_headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config['api_key']}",
            },
            model=self.llm_config["model"],
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a document QA assistant. Answer strictly from the provided "
                                "document content. Be concise and include only information relevant "
                                "to the question."
                            ),
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Document Content (possibly truncated):\n{context}"},
                        {"type": "text", "text": normalized_prompt},
                    ],
                },
            ],
            temperature=self.llm_config["temperature"],
            max_tokens=self.llm_config.get("max_tokens") or 1200,
            stream=False,
        )

        answer_text = completion.choices[0].message.content or ""
        full_response = "Result: " + answer_text.strip()
        full_response = self._truncate_text(full_response, 12000)
        self._set_cache(self._qa_cache, qa_cache_key, full_response, max_items=96)
        return full_response

    @retry((requests.RequestException))
    @format_check()
    def extract_document_content(self, document_path: str) -> Tuple[bool, str]:
        r"""Extract the content of a given document (or url) and return the processed text.
        It may filter out some information, resulting in inaccurate content.

        Args:
            document_path (str): The path of the document to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the document was processed successfully, and the content of the document (if success).
        """
        print(f"Calling extract_document_content function with document_path=`{document_path}`")
        if any(document_path.endswith(ext) for ext in ['txt', 'html', 'md']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()
            return content

        if any(document_path.endswith(ext) for ext in ['zip']):
            extracted_files = self._unzip_file(document_path)
            return f"The extracted files are: {extracted_files}"

        if any(document_path.endswith(ext) for ext in ['json', 'jsonl', 'jsonld']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            f.close()
            return content

        if any(document_path.endswith(ext) for ext in ['py']):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()
            return content

        if any(document_path.endswith(ext) for ext in ['xlsx', 'xls', 'csv']):
            content = extract_excel_content(document_path)
            return content

        if any(document_path.endswith(ext) for ext in ['pptx']):
            content = extract_pptx_content(document_path)
            return content

        if any(document_path.endswith(ext) for ext in ['xml']):
            data = None
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            f.close()

            try:
                data = xmltodict.parse(content)
                print(f"The extracted xml data is: {data}")
                return data

            except Exception as e:
                print(f"The raw xml data is: {content}")
                return content

        if self._is_webpage(document_path):
            extracted_text = self._extract_webpage_content(document_path)
            return extracted_text


        else:
            # judge if url
            parsed_url = urlparse(document_path)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                if not os.path.exists(document_path):
                    return f"Document not found at path: {document_path}."

            # if is docx file, use docx2markdown to convert it
            if document_path.endswith(".docx"):
                if is_url:
                    tmp_path = self._download_file(document_path)
                else:
                    tmp_path = document_path

                file_name = os.path.basename(tmp_path)
                md_file_path = f"{file_name}.md"
                docx_to_markdown(tmp_path, md_file_path)

                # load content of md file
                with open(md_file_path, "r") as f:
                    extracted_text = f.read()
                f.close()
                return extracted_text
            if document_path.endswith(".pdf"):
                # try using pypdf to extract text from pdf
                try:
                    from PyPDF2 import PdfReader
                    if is_url:
                        tmp_path = self._download_file(document_path)
                        document_path = tmp_path

                    with open(document_path, 'rb') as f:
                        reader = PdfReader(f)
                        extracted_text = ""
                        for page in reader.pages:
                            extracted_text += page.extract_text()

                    return extracted_text
                except Exception as ex:
                    print(f'parse document error : {str(ex)}')
            return ""

    def _is_webpage(self, url: str) -> bool:
        r"""Judge whether the given URL is a webpage."""
        try:
            parsed_url = urlparse(url)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                return False

            path = parsed_url.path
            file_type, _ = mimetypes.guess_type(path)
            if 'text/html' in file_type:
                return True

            response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = response.headers.get("Content-Type", "").lower()

            if "text/html" in content_type:
                return True
            else:
                return False

        except requests.exceptions.RequestException as e:
            # raise RuntimeError(f"Error while checking the URL: {e}")
            print(f"Error while checking the URL: {e}")
            return False

        except TypeError:
            return True

    def _download_file(self, url: str):
        r"""Download a file from a URL and save it to the cache directory."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            file_name = url.split("/")[-1]

            file_path = os.path.join(self.cache_dir, file_name)
            os.makedirs(self.cache_dir, exist_ok=True)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")

    def _get_formatted_time(self) -> str:
        import time
        return time.strftime("%m%d%H%M")

    def _unzip_file(self, zip_path: str) -> List[str]:
        if not zip_path.endswith('.zip'):
            raise ValueError("Only .zip files are supported")

        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_path = os.path.join(self.cache_dir, zip_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            subprocess.run(["unzip", "-o", zip_path, "-d", extract_path], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to unzip file: {e}")

        extracted_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                extracted_files.append(os.path.join(root, file))

        return extracted_files


# llm_config = get_model_config()
# print("here")
# tool=DocumentProcessingToolkit(llm_config=llm_config, cache_dir=None)
# print("here")
#
# result=tool.ask_question_about_document(r"F:\project\agent\Co-Sight\workspace\20250527_142235\114d5fd0-e2ae-4b6d-a65a-870da2d19c08\book.pdf","In the endnote found in the second-to-last paragraph of page 11 of the book with the doi 10.2307/j.ctv9b2xdv, what date in November was the Wikipedia article accessed? Just give the day of the month.")
# print(result)
