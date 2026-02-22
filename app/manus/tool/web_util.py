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
import os
import sys
import traceback
import aiohttp  # For asynchronous PDF downloading

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import asyncio

# from aworld.logs.util import logger
# from browser_use.agent.memory import MemoryConfig

# Create PDF download directory if it doesn't exist
PDF_DOWNLOAD_DIR = "downloaded_pdfs"
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)

browser_system_prompt = """
===== NAVIGATION STRATEGY =====

1.  START: Navigate to the most authoritative source for this information
- For general queries: Use Google with specific search terms
- For known sources: Go directly to the relevant website

2.  EVALUATE: Assess each page methodically
- Scan headings and highlighted text first
- Look for data tables, charts, or official statistics
- Check publication dates for timeliness
- You MUST browse the entire web page by scrolling up and down with the mouse or other means to prevent information from being missed
- When you need to recognize the content of an image, you SHOULD click on the high-definition image with the left or right mouse button on the corresponding web page and then proceed with the recognition

3.  EXTRACT: Capture exactly what's needed
- Take screenshots of visual evidence (charts, tables, etc.)
- Copy precise text that answers the query
- Note source URLs for citation

4.  DOWNLOAD: Save the most relevant file to local path for further processing
- Save the text if possible for further text reading and analysis
- Save the image if possible for further image reasoning analysis
- Save the pdf if possible for further pdf reading and analysis
- If you find the PDF link, please view it online. If you cannot view it, please introduce the relevant information of the link's URL in the returned result

5.  PDF HANDLING: Special procedures for PDF files
- Identify PDF links by looking for .pdf extensions in URLs or "PDF" text in link descriptions
- When a relevant PDF is found, first attempt to download it
- Record the PDF's source URL, download status, and local file path
- If download fails, note the reason and keep the original URL for reference

6.  ROBOT DETECTION:
- If the page is a robot detection page, abort immediately
- Navigate to the most authoritative source for similar information instead

===== CRITICAL INSTRUCTIONS (READ BEFORE ANY ACTION) =====
⚠️ **NEVER RELY ON FIXED ELEMENT INDICES (e.g., index: 441)**
- Element indices are DYNAMIC and change every time the page loads or updates.
- Using a fixed index like 441 is the #1 cause of failure.

✅ **ALWAYS PREFER SEMANTIC & VISUAL CUES**
- Look for text content: "Close", "Got it", "Dismiss", "Skip", "×", "OK".
- Look for icons: a small "x" icon in the corner of a dialog is a universal close symbol.
- Prioritize actions based on text or role, NOT numbers.

🔄 **FAILURE RECOVERY & ALTERNATIVE ACTIONS**
- If a click action fails (especially one using an index), DO NOT retry the same action more than 2 times.
- After 2 consecutive failures on the SAME action:
1.  IMMEDIATELY STOP retrying that specific action.
2.  Actively search for ALTERNATIVE ways to achieve the goal.
3.  For a welcome dialog, your alternatives are:
- Click any button with text: "Close", "Got it", "Dismiss", "Skip", "×".
- Look for a small "x" icon in the top-right corner of the overlay.
- Try scrolling or waiting 3 seconds to see if the dialog changes.
- Your goal is PROGRESS, not stubbornly repeating a failed command.
- Move the options table up and down or left and right to check if there is any undisplayed content.
- Try full-page recognition to prevent information omission.

===== EFFICIENCY GUIDELINES =====
- Use specific search queries with key terms from the task
- Avoid getting distracted by tangential information
- Use archive.org to search the historical domain names of web pages
- Document each significant finding clearly and concisely

Your goal is to extract precisely the information needed with minimal browsing steps.
"""


async def download_pdf(pdf_url: str) -> tuple[bool, str]:
    """
    Asynchronously download a PDF file to the local directory

    Args:
        pdf_url: URL of the PDF file

    Returns:
        A tuple containing download success status and file path/error message
    """
    try:
        # Extract filename from URL
        filename = pdf_url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"

        # Ensure filename uniqueness
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(PDF_DOWNLOAD_DIR, filename)):
            filename = f"{base}_{counter}{ext}"
            counter += 1

        file_path = os.path.join(PDF_DOWNLOAD_DIR, filename)

        # Download PDF
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    print(f"Successfully downloaded PDF to: {file_path}")
                    return (True, file_path)
                else:
                    error_msg = f"Download failed, HTTP status code: {response.status}"
                    print(error_msg)
                    return (False, error_msg)

    except Exception as e:
        error_msg = f"Error occurred while downloading PDF: {str(e)}"
        print(error_msg)
        return (False, error_msg)


async def extract_and_download_pdfs(browser_context: BrowserContext) -> list[dict]:
    """
    Extract all PDF links from current page and download them

    Args:
        browser_context: Browser context object

    Returns:
        List containing information about each PDF download
    """
    pdf_downloads = []

    try:
        # Execute JavaScript to extract all links containing .pdf
        pdf_links = await browser_context.page.evaluate("""
            () => {
                const links = Array.from(document.getElementsByTagName('a'));
                return links
                    .filter(link => link.href && link.href.toLowerCase().endsWith('.pdf'))
                    .map(link => ({
                        url: link.href,
                        text: link.textContent?.trim() || 'No description'
                    }));
            }
        """)

        print(f"Found {len(pdf_links)} PDF links")

        # Download each PDF
        for link in pdf_links:
            success, result = await download_pdf(link['url'])
            pdf_downloads.append({
                'url': link['url'],
                'description': link['text'],
                'downloaded': success,
                'path_or_error': result
            })

    except Exception as e:
        print(f"Error occurred while extracting PDF links: {str(e)}")

    return pdf_downloads


async def browser_use(
        task_prompt: str = Field(description="The task to perform using the browser.")  # New parameter to control PDF downloading
) -> str:
    """
    Perform browser actions using the browser-use package.
    Args:
        task_prompt (str): The task to perform using the browser.

    Returns:
        str: The result of the browser actions.
    """
    download_pdfs = True
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            new_context_config=BrowserContextConfig(
                disable_security=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                minimum_wait_page_load_time=10,
                maximum_wait_page_load_time=30,
            ),
        )
    )
    browser_context = BrowserContext(
        config=BrowserContextConfig(
            trace_path=os.getenv("browser_trace.log")
        ),
        browser=browser,
    )
    agent = Agent(
        task=task_prompt,
        llm=ChatOpenAI(
            model=os.getenv("TOOL_MODEL_NAME"),
            api_key=os.getenv("TOOL_API_KEY"),
            base_url=os.getenv("TOOL_API_BASE_URL"),
            model_name=os.getenv("TOOL_MODEL_NAME"),
            openai_api_base=os.getenv("TOOL_API_BASE_URL"),
            openai_api_key=os.getenv("TOOL_API_KEY"),
            temperature=1.0,
        ),
        browser_context=browser_context,
        extend_system_message=browser_system_prompt,
    )

    pdf_results = []  # Store PDF download results

    try:
        browser_execution: AgentHistoryList = await agent.run(max_steps=50)

        # If PDF download is enabled, try to extract and download PDFs
        if download_pdfs:
            pdf_results = await extract_and_download_pdfs(browser_context)

        if (
                browser_execution is not None
                and browser_execution.is_done()
                and browser_execution.is_successful()
        ):
            exec_trace = browser_execution.extracted_content()
            print(
                ">>> 🌏 Browse Execution Succeed!\n"
                f">>> 💡 Result: {json.dumps(exec_trace, ensure_ascii=False, indent=4)}\n"
                ">>> 🌏 Browse Execution Succeed!\n"
            )

            final_result = browser_execution.final_result()

            # Add PDF download results to final result
            if pdf_results:
                final_result += "\n\nPDF Download Results:\n" + json.dumps(
                    pdf_results,
                    ensure_ascii=False,
                    indent=2
                )

            print(f"Browser execution success for task: {task_prompt}, result {final_result}")
            return final_result

        else:
            result = f"Browser execution failed for task: {task_prompt}"

            # Add PDF download results even if main task failed
            if pdf_results:
                result += "\n\nPDF Download Results:\n" + json.dumps(
                    pdf_results,
                    ensure_ascii=False,
                    indent=2
                )
            return result

    except Exception as e:
        error_msg = f"Browser execution failed for task: {task_prompt} due to {str(e)}"
        print(f"Browser execution failed: {traceback.format_exc()}")

        # Add PDF download results even if exception occurred
        if pdf_results:
            error_msg += "\n\nPDF Download Results:\n" + json.dumps(
                pdf_results,
                ensure_ascii=False,
                indent=2
            )
        return error_msg

    finally:
        await browser.close()
        print("Browser Closed!")


def sync_browser_use(task_prompt):
    return asyncio.run(browser_use(task_prompt))


if __name__ == "__main__":
    load_dotenv(override=True)
    os.environ['OPENAI_API_KEY'] = os.environ.get('TOOL_API_KEY', '')
    try:
        # Example: Find Wikipedia page for China and download any found PDFs
        sync_browser_use("Find https://pmc.ncbi.nlm.nih.gov/articles/PMC4116197/ in PDF, and view the PDF of the subject")

    except Exception as e:
        print(f"Error: {e}")
